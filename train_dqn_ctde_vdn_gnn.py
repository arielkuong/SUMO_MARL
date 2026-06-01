#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Dict
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime

from marl_utils.models import GNNPolicyQ          # reuse your SimpleAttnMP, GNNPolicyQ
from marl_utils.replay_buffers import JointReplayBuffer
from marl_utils.network_update import vdn_update_gnn, soft_update   # <-- new updater you’ll add below
from marl_utils.common import (
    parse_args,
    set_global_seed,
    build_grid_edge_index,       # graph over TLS grid (you already have this)
    EvalHistory,
    clear_eval_history,
)
from env_builder import build_train_env, get_or_create_eval_pool


# --------- evaluation (decentralized greedy over cached fixed-route envs) ----------
@torch.no_grad()
def evaluate_vdn_gnn_shared(
    args,
    shared_q_network: GNNPolicyQ,
    agent_id_list: List[str],
    edge_index: torch.Tensor,
    run_name: str
) -> float:

    original_mode = shared_q_network.training
    shared_q_network.eval()
    device = next(shared_q_network.parameters()).device

    eval_env_pool = get_or_create_eval_pool(args)
    recorder = EvalHistory(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)

    try:
        returns_all, throughput_all, mean_travel_all, mean_wait_all = [], [], [], []

        for eval_env in eval_env_pool.envs:
            episode_return = 0.0
            last_info_dict: Dict = {}
            obs_dict = eval_env.reset()
            done = False

            while not done:
                X_np = np.stack([obs_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)  # [N,O]
                X = torch.as_tensor(X_np, dtype=torch.float32, device=device)                         # [N,O]

                q_values = shared_q_network(X, edge_index.to(device))  # [N,A]
                greedy = torch.argmax(q_values, dim=-1).tolist()
                action_dict = {aid: int(greedy[i]) for i, aid in enumerate(agent_id_list)}

                next_obs, reward_dict, done, info = eval_env.step(action_dict)
                episode_return += float(np.sum(list(reward_dict.values())))
                obs_dict = next_obs
                last_info_dict = info

            kpis = last_info_dict.get("network_kpis", {})
            returns_all.append(episode_return)
            throughput_all.append(float(kpis.get("throughput_veh_per_hour", 0.0)))
            mean_travel_all.append(float(kpis.get("mean_travel_time_s", 0.0)))
            mean_wait_all.append(float(kpis.get("mean_waiting_time_s", 0.0)))

        avg_return = float(np.mean(returns_all))
        avg_throughput = float(np.mean(throughput_all))
        avg_travel_s = float(np.mean(mean_travel_all))
        avg_wait_s = float(np.mean(mean_wait_all))

        print(
            f"[{datetime.now()}, EVAL {run_name}] MEAN over {len(eval_env_pool.envs)} cases | "
            f"return={avg_return:.2f} | throughput={avg_throughput:.2f} veh/h | "
            f"mean travel time={avg_travel_s:.2f}s | avg waiting time={avg_wait_s:.2f}s"
        )
        recorder.save(avg_return, avg_throughput, avg_travel_s, avg_wait_s)
        return avg_return
    finally:
        shared_q_network.train(original_mode)


# ------------------ training ------------------
def run_training(args):
    # device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    device = torch.device(args.device)
    set_global_seed(args.seed)
    best_eval_return = -np.inf

    # Env & shapes
    train_env = build_train_env(args)
    obs_dict = train_env.reset()
    agent_id_list = list(train_env.agent_ids)
    N = len(agent_id_list)
    O = len(next(iter(obs_dict.values())))
    A_dim = train_env.action_spaces[agent_id_list[0]].n

    # Fixed grid topology (directed 4-neighbour edges)
    edge_index = build_grid_edge_index(agent_id_list)  # [2,E] torch.long (no edges across samples)

    # Shared per-agent GNN Q_i (CTDE training, decentralized execution)
    online_q = GNNPolicyQ(node_dim=O, actions=A_dim, hidden=args.hidden, layers=args.gnn_layers).to(device)
    target_q = GNNPolicyQ(node_dim=O, actions=A_dim, hidden=args.hidden, layers=args.gnn_layers).to(device)
    target_q.load_state_dict(online_q.state_dict())
    target_q.eval()

    optim_q = optim.Adam(online_q.parameters(), lr=args.lr)

    # CTDE: joint replay buffer (stores one full-graph transition per step)
    joint_replay = JointReplayBuffer(capacity=args.replay_size, num_agents=N, obs_dim=O, seed=args.rb_seed)

    eps = args.eps_start
    total_steps = 0
    run_name = "vdn_ctde_gnn_shared"

    clear_eval_history(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)

    for ep_idx in range(1, args.episodes + 1):
        obs_dict = train_env.reset()
        done = False
        steps_this_ep = 0

        while not done and steps_this_ep < args.episode_steps:
            X_np = np.stack([obs_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)    # [N,O]
            X = torch.as_tensor(X_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                q_all = online_q(X, edge_index.to(device))                                        # [N,A]
                greedy = torch.argmax(q_all, dim=-1).cpu().numpy()                                # [N]

            # epsilon-greedy per agent
            actions_np = greedy.copy()
            for i in range(N):
                if np.random.rand() < eps:
                    actions_np[i] = np.random.randint(A_dim)

            action_dict = {aid: int(actions_np[i]) for i, aid in enumerate(agent_id_list)}
            next_obs_dict, reward_dict, done, _ = train_env.step(action_dict)

            X_next_np = np.stack([next_obs_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)  # [N,O]
            R_np = np.array([float(reward_dict[aid]) for aid in agent_id_list], dtype=np.float32)           # [N]

            # push ONE joint transition per env step
            joint_replay.push(
                state_all=X_np,
                actions_all=actions_np.astype(np.int64),
                rewards_all=R_np,
                next_state_all=X_next_np,
                done_flag=float(done),
            )

            obs_dict = next_obs_dict
            total_steps += 1
            steps_this_ep += 1

        # -------- CTDE update (VDN + Double-DQN over GNN Q_i) --------
        mean_loss = 0.0
        if total_steps >= args.warmup_steps and len(joint_replay) >= args.batch_size:
            losses = []
            for _ in range(args.updates_per_ep):
                batch = joint_replay.sample(args.batch_size)  # (S, A, R, NS, D)
                loss_val = vdn_update_gnn(
                    device=device,
                    gnn_online_q=online_q,
                    gnn_target_q=target_q,
                    optimizer=optim_q,
                    batch_tuple=batch,
                    graph_edge_index=edge_index,   # same edges for all batch items
                    gamma=args.gamma,
                    double_dqn=True,
                    grad_clip=1.0,
                )
                losses.append(loss_val)
                soft_update(target_q, online_q, args.tau)

            mean_loss = float(np.mean(losses))
            print(f"[ep {ep_idx}] train_loss={mean_loss:.3f} eps={eps:.3f}")

            if ep_idx % args.eval_every == 0:
                eval_return = evaluate_vdn_gnn_shared(args, online_q, agent_id_list, edge_index, run_name=run_name)
                if eval_return > best_eval_return:
                    best_eval_return = eval_return
                    save_path = f"{args.logdir}_grid_{args.grid_n}/seed{args.seed}/model_best_{run_name}_seed{args.seed}.pt"
                    torch.save(online_q.state_dict(), save_path)

            eps = max(args.eps_end, eps * args.eps_decay)
        else:
            print(f"[ep {ep_idx}] total steps={total_steps} (warming up) eps={eps:.3f}")

    train_env.close()


if __name__ == "__main__":
    run_training(parse_args())
