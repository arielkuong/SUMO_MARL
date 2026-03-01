from __future__ import annotations
import argparse
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from datetime import datetime

# Your modules
from marl_utils.models import RecurrentQNet  # expects forward([B,T,O], hidden) -> ([B,T,A], new_hidden)
from marl_utils.replay_buffers import JointSequenceReplay
from marl_utils.network_update import vdn_update_lstm
from marl_utils.common import set_global_seed, EvalHistory, clear_eval_history
from env_builder import build_train_env, get_or_create_eval_pool

# --------- evaluation (decentralized greedy with 1-step LSTM unroll) ----------
@torch.no_grad()
def evaluate_idqn_lstm_shared(args, shared_q: RecurrentQNet, agent_ids: List[str], run_name: str) -> float:
    original_mode = shared_q.training
    shared_q.eval()
    device = next(shared_q.parameters()).device

    eval_pool = get_or_create_eval_pool(args)
    recorder = EvalHistory(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)

    try:
        returns_all, throughput_all, mean_travel_all, mean_wait_all = [], [], [], []

        for env in eval_pool.envs:
            obs_dict = env.reset()
            done = False
            ep_ret = 0.0
            last_info: Dict = {}

            # Hidden over agents: B = N, T=1 per tick
            N = len(agent_ids)
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

            while not done:
                obs_mat = np.stack([obs_dict[aid] for aid in agent_ids], axis=0).astype(np.float32)  # [N,O]
                obs_t = torch.as_tensor(obs_mat, dtype=torch.float32, device=device).unsqueeze(1)     # [N,1,O]
                q_seq, hidden = shared_q(obs_t, hidden)                                              # [N,1,A]
                q = q_seq[:, -1, :]                                                                  # [N,A]
                greedy = torch.argmax(q, dim=-1).tolist()
                action_dict = {aid: int(greedy[i]) for i, aid in enumerate(agent_ids)}
                next_obs, rew, done, info = env.step(action_dict)
                ep_ret += float(np.sum(list(rew.values())))
                obs_dict = next_obs
                last_info = info

            kpis = last_info.get("network_kpis", {})
            returns_all.append(ep_ret)
            throughput_all.append(float(kpis.get("throughput_veh_per_hour", 0.0)))
            mean_travel_all.append(float(kpis.get("mean_travel_time_s", 0.0)))
            mean_wait_all.append(float(kpis.get("mean_waiting_time_s", 0.0)))

        avg_return = float(np.mean(returns_all))
        avg_throughput = float(np.mean(throughput_all))
        avg_travel_s = float(np.mean(mean_travel_all))
        avg_wait_s = float(np.mean(mean_wait_all))

        print(
            f"[{datetime.now()}, EVAL {run_name}] MEAN over {len(eval_pool.envs)} cases | "
            f"return={avg_return:.2f} | throughput={avg_throughput:.2f} veh/h | "
            f"mean travel time={avg_travel_s:.2f}s | avg waiting time={avg_wait_s:.2f}s"
        )
        recorder.save(avg_return, avg_throughput, avg_travel_s, avg_wait_s)
        return avg_return
    finally:
        shared_q.train(original_mode)

# ------------------------------------------------------------
def run_training(args):
    # device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    device = torch.device(args.device)
    set_global_seed(args.seed)
    best_eval_return = -np.inf

    # Env
    env = build_train_env(args)
    obs_dict = env.reset()
    agent_ids = list(env.agent_ids)
    N = len(agent_ids)
    O = len(next(iter(obs_dict.values())))
    A_dim = env.action_spaces[agent_ids[0]].n

    # Shared LSTM Q_i
    online_q = RecurrentQNet(O, A_dim, hidden_units=args.hidden).to(device)
    target_q = RecurrentQNet(O, A_dim, hidden_units=args.hidden).to(device)
    target_q.load_state_dict(online_q.state_dict())
    target_q.eval()

    optim_q = optim.Adam(online_q.parameters(), lr=args.lr)

    # Joint sequence replay for CTDE
    replay = JointSequenceReplay(capacity_steps=args.replay_size, num_agents=N, obs_dim=O, seed=args.rb_seed)

    eps = args.eps_start
    total_steps = 0
    run_name = "vdn_ctde_lstm_shared_seqlen" + str(args.seq_len)

    clear_eval_history(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)

    for ep_idx in range(1, args.episodes + 1):
        obs_dict = env.reset()
        done = False
        steps = 0

        # Episode buffers
        S_list, A_list, R_list, NS_list, D_list = [], [], [], [], []

        # Hidden state across time for action selection (B=N)
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        while not done and steps < args.episode_steps:
            obs_mat = np.stack([obs_dict[aid] for aid in agent_ids], axis=0).astype(np.float32)  # [N,O]
            obs_t  = torch.as_tensor(obs_mat, dtype=torch.float32, device=device).unsqueeze(1)   # [N,1,O]

            with torch.no_grad():
                q_seq, hidden = online_q(obs_t, hidden)     # [N,1,A]
                q = q_seq[:, -1, :]                         # [N,A]
                greedy = torch.argmax(q, dim=-1).cpu().numpy()  # [N]

            actions = greedy.copy()
            for i in range(N):
                if np.random.rand() < eps:
                    actions[i] = np.random.randint(A_dim)

            action_dict = {aid: int(actions[i]) for i, aid in enumerate(agent_ids)}
            next_obs, reward_dict, done, _ = env.step(action_dict)

            next_mat = np.stack([next_obs[aid] for aid in agent_ids], axis=0).astype(np.float32)  # [N,O]
            rewards  = np.array([float(reward_dict[aid]) for aid in agent_ids], dtype=np.float32) # [N]

            # record step
            S_list.append(obs_mat)
            A_list.append(actions.astype(np.int64))
            R_list.append(rewards)
            NS_list.append(next_mat)
            D_list.append(float(done))

            obs_dict = next_obs
            total_steps += 1
            steps += 1

        # push full joint episode
        replay.push_episode(
            states=np.asarray(S_list,  dtype=np.float32),     # [T,N,O]
            actions=np.asarray(A_list, dtype=np.int64),       # [T,N]
            rewards=np.asarray(R_list, dtype=np.float32),     # [T,N]
            next_states=np.asarray(NS_list, dtype=np.float32),# [T,N,O]
            dones=np.asarray(D_list, dtype=np.float32),       # [T]
        )

        # -------- Centralized training (VDN) from sequences --------
        mean_loss = 0.0
        if total_steps >= args.warmup_steps and len(replay) > 0:
            losses = []
            for _ in range(args.updates_per_ep):
                S, A, R, NS, D, M = replay.sample(args.batch_size, args.seq_len)
                loss = vdn_update_lstm(
                    device=device,
                    online_q=online_q,
                    target_q=target_q,
                    optimizer=optim_q,
                    batch_tuple=(S, A, R, NS, D, M),
                    gamma=args.gamma,
                    burn_in=args.burn_in,
                    double_dqn=True,
                    grad_clip=1.0,
                )
                losses.append(loss)
                # Polyak
                with torch.no_grad():
                    for tp, op in zip(target_q.parameters(), online_q.parameters()):
                        tp.data.mul_(1.0 - args.tau).add_(args.tau * op.data)

            mean_loss = float(np.mean(losses))
            print(f"[ep {ep_idx}] train_loss={mean_loss:.3f} eps={eps:.3f}")

            if ep_idx % args.eval_every == 0:
                eval_return = evaluate_idqn_lstm_shared(args, online_q, agent_ids, run_name=run_name)
                if eval_return > best_eval_return:
                    best_eval_return = eval_return
                    save_path = f"{args.logdir}_grid_{args.grid_n}/seed{args.seed}/model_best_{run_name}_seed{args.seed}.pt"
                    torch.save(online_q.state_dict(), save_path)

            # epsilon decay
            eps = max(args.eps_end, eps * args.eps_decay)
        else:
            print(f"[ep {ep_idx}] total steps={total_steps} (warming up) eps={eps:.3f}")

    env.close()

# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("CTDE VDN (LSTM, SHARED) — Train on Random Flows, Eval on Fixed Trips")
    p.add_argument('--grid-n', type=int, default=3)
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--episode-steps', type=int, default=100)
    p.add_argument('--sumo-steps-per-env-step', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gui', action='store_true')
    p.add_argument('--gui-delay-ms', type=int, default=0)
    p.add_argument('--logdir', type=str, default='logs')
    p.add_argument('--device', type=str, default='cuda')

    # LSTM model
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)

    # Exploration
    p.add_argument('--eps-start', type=float, default=0.5)
    p.add_argument('--eps-end', type=float, default=0.05)
    p.add_argument('--eps-decay', type=float, default=0.998)

    # Replay/sequence training
    p.add_argument('--replay-size', type=int, default=100000)
    p.add_argument('--warmup-steps', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seq-len', type=int, default=8)
    p.add_argument('--burn-in', type=int, default=4)
    p.add_argument('--updates-per-ep', type=int, default=32)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--rb-seed', type=int, default=1234)
    return p.parse_args()

if __name__ == "__main__":
    run_training(parse_args())
