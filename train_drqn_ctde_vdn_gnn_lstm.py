#!/usr/bin/env python3
from __future__ import annotations
import argparse
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.optim as optim
from datetime import datetime

from marl_utils.models import GNNLSTMPolicyQ
from marl_utils.network_update import vdn_update_gnn_lstm, soft_update
from marl_utils.common import (
    set_global_seed,
    build_grid_edge_index,
    EvalHistory,
    clear_eval_history,
)
from marl_utils.replay_buffers import GlobalSequenceReplay   # <— use your existing buffer
from env_builder import build_train_env, get_or_create_eval_pool


# ---------------- evaluation (greedy, recurrent) ----------------
@torch.no_grad()
def evaluate_vdn_gnnlstm_shared(
    args,
    online_q: GNNLSTMPolicyQ,
    agent_id_list: List[str],
    edge_index: torch.Tensor,
    run_name: str
) -> float:

    original_mode = online_q.training
    online_q.eval()
    device = next(online_q.parameters()).device

    eval_env_pool = get_or_create_eval_pool(args)
    recorder = EvalHistory(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)

    try:
        returns_all, throughput_all, mean_travel_all, mean_wait_all = [], [], [], []

        for env in eval_env_pool.envs:
            episode_return = 0.0
            last_info: Dict = {}
            obs_dict = env.reset()
            done = False
            rnn_state = None

            while not done:
                X_np = np.stack([obs_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)   # [N,O]
                X = torch.as_tensor(X_np, dtype=torch.float32, device=device).unsqueeze(0)             # [1,N,O]
                q_t, rnn_state = online_q.step(X, edge_index.to(device), rnn_state)                    # [1,N,A]
                greedy = torch.argmax(q_t[0], dim=-1).tolist()                                         # [N]
                action_dict = {aid: int(greedy[i]) for i, aid in enumerate(agent_id_list)}

                next_obs, reward_dict, done, info = env.step(action_dict)
                episode_return += float(np.sum(list(reward_dict.values())))
                obs_dict = next_obs
                last_info = info

            kpis = last_info.get("network_kpis", {})
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
        online_q.train(original_mode)


# ------------------ training (with replay) ------------------
def run_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    set_global_seed(args.seed)
    best_eval_return = -np.inf

    assert args.seq_len > 0 and 0 <= args.burn_in < args.seq_len, \
        f"Require seq_len>0 and 0<=burn_in<seq_len, got seq_len={args.seq_len}, burn_in={args.burn_in}"

    # Env & shapes
    train_env = build_train_env(args)
    obs_dict = train_env.reset()
    agent_id_list = list(train_env.agent_ids)
    N = len(agent_id_list)
    O = len(next(iter(obs_dict.values())))
    A_dim = train_env.action_spaces[agent_id_list[0]].n

    # Fixed grid topology (CPU; move to device on use)
    edge_index = build_grid_edge_index(agent_id_list)  # [2,E] torch.long

    # Spatio-temporal Q (online & target)
    online_q = GNNLSTMPolicyQ(node_dim=O, actions=A_dim, hidden=args.hidden, gnn_layers=2).to(device)
    target_q = GNNLSTMPolicyQ(node_dim=O, actions=A_dim, hidden=args.hidden, gnn_layers=2).to(device)
    target_q.load_state_dict(online_q.state_dict())
    target_q.eval()

    optim_q = optim.Adam(online_q.parameters(), lr=args.lr)

    # ---- Replay buffer (episodes -> sequences) ----
    replay = GlobalSequenceReplay(capacity_steps=args.replay_size, num_agents=N, obs_dim=O, seed=args.rb_seed)

    eps = args.eps_start
    total_steps = 0
    run_name = f"vdn_ctde_gnn_lstm_shared_seqlen{args.seq_len}"
    clear_eval_history(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)

    for ep_idx in range(1, args.episodes + 1):
        obs_dict = train_env.reset()
        done = False
        steps = 0

        # episode buffers (joint over all agents)
        X_list, A_list, R_list, D_list, Xn_list = [], [], [], [], []

        while not done and steps < args.episode_steps:
            X_np = np.stack([obs_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)   # [N,O]

            # epsilon-greedy from online net (stateless step for action selection)
            with torch.no_grad():
                q_all, _ = online_q.step(torch.as_tensor(X_np, device=device).unsqueeze(0), edge_index.to(device))
                greedy = torch.argmax(q_all[0], dim=-1).cpu().numpy()  # [N]

            actions_np = greedy.copy()
            for i in range(N):
                if np.random.rand() < eps:
                    actions_np[i] = np.random.randint(A_dim)

            action_dict = {aid: int(actions_np[i]) for i, aid in enumerate(agent_id_list)}
            next_obs_dict, reward_dict, done, _ = train_env.step(action_dict)

            Xn_np = np.stack([next_obs_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)  # [N,O]
            R_np  = np.array([float(reward_dict[aid]) for aid in agent_id_list], dtype=np.float32)      # [N]

            # record this step
            X_list.append(X_np)
            A_list.append(actions_np.astype(np.int64))
            R_list.append(R_np)
            D_list.append(float(done))
            Xn_list.append(Xn_np)

            obs_dict = next_obs_dict
            total_steps += 1
            steps += 1

        # push full joint episode
        if steps > 0:
            replay.push_episode(
                states=np.stack(X_list,  axis=0),     # [T,N,O]
                actions=np.stack(A_list,  axis=0),    # [T,N]
                rewards=np.stack(R_list,  axis=0),    # [T,N]
                next_states=np.stack(Xn_list, axis=0),# [T,N,O]
                dones=np.stack(D_list,  axis=0),      # [T]
            )

        # -------- Centralised training (VDN) from replayed sequences --------
        mean_loss = 0.0
        if total_steps >= args.warmup_steps and len(replay) > 0:
            losses = []
            for _ in range(args.updates_per_ep):
                # Sample sequences of length L (pads repeat final frame; done=1 on pads)
                S, A, R, NS, D = replay.sample(args.batch_size, args.seq_len)  # [B,L,N,O], [B,L,N], ..., [B,L]
                loss_val = vdn_update_gnn_lstm(
                    device=device,
                    gnnlstm_online_q=online_q,
                    gnnlstm_target_q=target_q,
                    optimizer=optim_q,
                    seq_batch=(S, A, R, NS, D),
                    graph_edge_index=edge_index,
                    gamma=args.gamma,
                    double_dqn=True,
                    burn_in=args.burn_in,
                    grad_clip=1.0,
                )
                losses.append(loss_val)
                soft_update(target_q, online_q, args.tau)

            mean_loss = float(np.mean(losses)) if losses else 0.0
            print(f"[ep {ep_idx}] train_loss={mean_loss:.3f} eps={eps:.3f}")

            # periodic evaluation
            if ep_idx % args.eval_every == 0:
                eval_return = evaluate_vdn_gnnlstm_shared(args, online_q, agent_id_list, edge_index, run_name=run_name)
                if eval_return > best_eval_return:
                    best_eval_return = eval_return
                    save_path = f"{args.logdir}_grid_{args.grid_n}/seed{args.seed}/model_best_{run_name}_seed{args.seed}.pt"
                    torch.save(online_q.state_dict(), save_path)

            # epsilon decay per episode
            eps = max(args.eps_end, eps * args.eps_decay)
        else:
            print(f"[ep {ep_idx}] total steps={total_steps} (warming up) eps={eps:.3f}")

    train_env.close()


# ------------------ CLI ------------------
def parse_args():
    parser = argparse.ArgumentParser("CTDE VDN (GNN+LSTM, SHARED, replayed sequences)")
    parser.add_argument('--grid-n', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--episode-steps', type=int, default=100)
    parser.add_argument('--sumo-steps-per-env-step', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--gui-delay-ms', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--logdir', type=str, default='logs')

    # GNN+LSTM & opt
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)

    # Exploration
    parser.add_argument('--eps-start', type=float, default=0.5)
    parser.add_argument('--eps-end', type=float, default=0.05)
    parser.add_argument('--eps-decay', type=float, default=0.998)

    # Replay / recurrent training
    parser.add_argument('--replay-size', type=int, default=100000)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seq-len', type=int, default=8)
    parser.add_argument('--burn-in', type=int, default=4)
    parser.add_argument('--updates-per-ep', type=int, default=32)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--rb-seed', type=int, default=1234)

    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
