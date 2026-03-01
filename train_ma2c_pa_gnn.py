#!/usr/bin/env python3
from __future__ import annotations
import argparse
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from datetime import datetime

from marl_utils.models import ActorGNNAttn, CriticGNNPerAgentAttn
from marl_utils.network_update import ma2c_pa_update_gnn_attn
from marl_utils.common import (
    set_global_seed,
    EvalHistory,
    clear_eval_history,
    build_grid_edge_index,           # helper: builds [2,E] from agent ids
)
from env_builder import build_train_env, get_or_create_eval_pool


# ------------------ evaluation (decentralised greedy; actor only) ------------------
@torch.no_grad()
def evaluate_ma2c_gnn_shared(args, actor: nn.Module, agent_id_list: List[str],
                             edge_index_cpu: torch.Tensor, run_name: str) -> float:
    original_mode = actor.training
    actor.eval()
    device = next(actor.parameters()).device

    eval_env_pool = get_or_create_eval_pool(args)
    recorder = EvalHistory(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)

    try:
        returns_all, throughput_all, mean_travel_all, mean_wait_all = [], [], [], []
        for eval_env in eval_env_pool.envs:
            episode_return = 0.0
            last_info_dict: Dict = {}
            observation_dict = eval_env.reset()
            done_flag = False

            while not done_flag:
                obs_matrix = np.stack([observation_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)
                obs_tensor = torch.as_tensor(obs_matrix, dtype=torch.float32, device=device)  # [N,O]
                logits = actor(obs_tensor, edge_index_cpu.to(device))                         # [N,A]
                greedy_actions = torch.argmax(logits, dim=-1).tolist()
                action_dict = {aid: int(greedy_actions[i]) for i, aid in enumerate(agent_id_list)}

                next_obs, reward_dict, done_flag, info = eval_env.step(action_dict)
                episode_return += float(np.sum(list(reward_dict.values())))
                observation_dict = next_obs
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
        actor.train(original_mode)


# ---------------- training loop (single-episode updates; MA2C-PA + GNN) ----------------
def run_training(args):
    # device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    device = torch.device(args.device)
    set_global_seed(args.seed)
    best_eval_return = -np.inf

    # Env & shapes
    train_env = build_train_env(args)
    first_obs = train_env.reset()
    agent_id_list = list(train_env.agent_ids)
    N = len(agent_id_list)
    O = len(next(iter(first_obs.values())))
    A_dim = train_env.action_spaces[agent_id_list[0]].n

    # Build fixed grid topology once (keep on CPU; move to device on use)
    edge_index_cpu = build_grid_edge_index(agent_id_list)  # [2,E], long

    # Shared decentralised actor (GNN) + central per-agent critic (GNN)
    actor  = ActorGNNAttn(obs_dim=O, act_dim=A_dim, hidden=args.hidden, layers=args.gnn_layers, edge_dim=0).to(device)
    critic = CriticGNNPerAgentAttn(obs_dim=O, hidden=args.hidden, layers=args.gnn_layers, edge_dim=0).to(device)

    optim_actor  = optim.Adam(actor.parameters(),  lr=args.lr_actor)
    optim_critic = optim.Adam(critic.parameters(), lr=args.lr_critic)

    run_name = "ma2c_pa_gnn"
    clear_eval_history(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)

    # Entropy coefficient (multiplicative decay per update)
    entropy_coef = args.entropy_coef_start

    for ep_idx in range(1, args.episodes + 1):
        # -------- single on-policy rollout --------
        actor.eval(); critic.eval()

        obs_dict = train_env.reset()
        done = False
        steps = 0

        obs_seq_list, act_seq_list, rew_seq_list, done_seq_list = [], [], [], []

        while not done and steps < args.episode_steps:
            obs_matrix = np.stack([obs_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)
            obs_t = torch.as_tensor(obs_matrix, dtype=torch.float32, device=device)  # [N,O]

            with torch.no_grad():
                logits = actor(obs_t, edge_index_cpu.to(device))  # [N,A]
                dist = Categorical(logits=logits)
                a_sample = dist.sample()                           # [N]

            action_np = a_sample.cpu().numpy()
            action_dict = {aid: int(action_np[i]) for i, aid in enumerate(agent_id_list)}
            next_obs_dict, reward_dict, done, _ = train_env.step(action_dict)

            rewards_vec = np.array([float(reward_dict[aid]) for aid in agent_id_list], dtype=np.float32)

            obs_seq_list.append(obs_t)                                                   # [N,O]
            act_seq_list.append(a_sample.long())                                         # [N]
            rew_seq_list.append(torch.as_tensor(rewards_vec, device=device))             # [N]
            done_seq_list.append(torch.as_tensor(float(done), device=device))            # []

            obs_dict = next_obs_dict
            steps += 1

        if steps == 0:
            print(f"[ep {ep_idx}] rollout empty (env ended immediately).")
            continue

        # Stack to tensors
        obs_seq  = torch.stack(obs_seq_list,  dim=0)        # [T,N,O]
        act_seq  = torch.stack(act_seq_list,  dim=0).long() # [T,N]
        rew_seq  = torch.stack(rew_seq_list,  dim=0)        # [T,N]
        done_seq = torch.stack(done_seq_list, dim=0)        # [T]
        # Time-limit termination -> prevent bootstrap past horizon
        done_seq[-1] = 1.0

        # Last joint obs for bootstrap
        last_obs_mat = np.stack([obs_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)
        last_obs = torch.as_tensor(last_obs_mat, dtype=torch.float32, device=device)  # [N,O]

        # -------- single update --------
        actor.train(); critic.train()
        logs = ma2c_pa_update_gnn_attn(
            actor=actor,
            critic_pa=critic,
            optim_actor=optim_actor,
            optim_critic=optim_critic,
            obs_seq=obs_seq,
            act_seq=act_seq,
            rew_seq=rew_seq,
            done_seq=done_seq,
            last_obs=last_obs,
            edge_index=edge_index_cpu,               # pass the graph (kept on CPU; moved inside)
            edge_attr=None,                          # add edge features if available
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            entropy_coef=entropy_coef,               # decayed
            value_coef=args.value_coef,
            grad_clip=args.grad_clip,
            # MA2C-PA knobs
            advantage_mode=args.advantage_mode,
            team_reward_reduce=args.team_reward_reduce,
            normalize_rewards=args.normalize_rewards,
            reward_scale=args.reward_scale,
            normalize_adv=args.normalize_adv,
            huber_delta=args.huber_delta,
            value_clip_eps=args.value_clip_eps,
        )

        print(f"[ep {ep_idx}] T={steps} | "
              f"pi={logs['loss_policy']:.3f} v={logs['loss_value']:.3f} H={logs['entropy']:.3f} "
              f"| Vmean={logs['v_mean']:.3f} AdvMean={logs['adv_mean']:.3f} | ent={entropy_coef:.4f}")

        # Entropy multiplicative decay
        entropy_coef = max(args.entropy_coef_end, entropy_coef * args.entropy_coef_decay)

        # -------- periodic evaluation --------
        if ep_idx % args.eval_every == 0:
            eval_return = evaluate_ma2c_gnn_shared(args, actor, agent_id_list, edge_index_cpu, run_name=run_name)
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                save_path = f"{args.logdir}_grid_{args.grid_n}/seed{args.seed}/model_best_{run_name}_seed{args.seed}.pt"
                torch.save(actor.state_dict(), save_path)

    train_env.close()


# ------------------ CLI ------------------
def parse_args():
    parser = argparse.ArgumentParser("MA2C-PA (Per-Agent Central Critic, GNN-Attn, single-episode updates)")
    parser.add_argument('--grid-n', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--episode-steps', type=int, default=100)
    parser.add_argument('--sumo-steps-per-env-step', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--gui-delay-ms', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--device', type=str, default='cuda')

    # A2C/GAE + losses
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--gae-lambda', type=float, default=0.9)
    parser.add_argument('--value-coef', type=float, default=0.7)
    parser.add_argument('--grad-clip', type=float, default=1.0)

    # Entropy (multiplicative decay)
    parser.add_argument('--entropy-coef-start', type=float, default=0.05)
    parser.add_argument('--entropy-coef-end',   type=float, default=0.005)
    parser.add_argument('--entropy-coef-decay', type=float, default=0.995)

    # Stabilisers / reward handling
    parser.add_argument('--normalize-adv', action='store_true', default=True)
    parser.add_argument('--huber-delta', type=float, default=1.0)
    parser.add_argument('--value-clip-eps', type=float, default=0.2)

    # MA2C-PA specific knobs
    parser.add_argument('--advantage-mode', type=str, default='per_agent', choices=['per_agent','team'])
    parser.add_argument('--team-reward-reduce', type=str, default='mean', choices=['mean','sum'])
    parser.add_argument('--normalize-rewards', type=str, default='per_agent', choices=['off','per_agent','global'])
    parser.add_argument('--reward-scale', type=float, default=1.0)

    # GNN + opt
    parser.add_argument('--gnn-layers', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--lr-actor', type=float, default=3e-4)
    parser.add_argument('--lr-critic', type=float, default=1e-3)

    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
