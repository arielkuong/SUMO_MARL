#!/usr/bin/env python3
from __future__ import annotations
import argparse
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from datetime import datetime

from marl_utils.models import ActorLSTM, CriticMLP
from marl_utils.network_update import ia2c_update_lstm_actor_mlp
from marl_utils.common import set_global_seed, EvalHistory, clear_eval_history
from env_builder import build_train_env, get_or_create_eval_pool


@torch.no_grad()
def evaluate_ia2c_lstm_actor(args, actor: nn.Module, agent_id_list: List[str], run_name: str) -> float:
    original_mode = actor.training
    actor.eval()
    device = next(actor.parameters()).device

    eval_env_pool = get_or_create_eval_pool(args)
    recorder = EvalHistory(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)

    try:
        returns_all, throughput_all, mean_travel_all, mean_wait_all = [], [], [], []
        for eval_env in eval_env_pool.envs:
            ep_ret = 0.0
            last_info: Dict = {}
            obs_dict = eval_env.reset()
            done = False
            actor_h = None  # carry LSTM hidden across evaluation steps

            while not done:
                obs_mat = np.stack([obs_dict[aid] for aid in agent_id_list], 0).astype(np.float32)  # [N,O]
                obs_t = torch.as_tensor(obs_mat, dtype=torch.float32, device=device)
                logits_t, actor_h = actor.step(obs_t, actor_h)  # [N,A]
                greedy = torch.argmax(logits_t, dim=-1).tolist()
                action_dict = {aid: int(greedy[i]) for i, aid in enumerate(agent_id_list)}

                next_obs, reward_dict, done, info = eval_env.step(action_dict)
                ep_ret += float(np.sum(list(reward_dict.values())))
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
            f"[{datetime.now()}, EVAL {run_name}] MEAN over {len(eval_env_pool.envs)} cases | "
            f"return={avg_return:.2f} | throughput={avg_throughput:.2f} veh/h | "
            f"mean travel time={avg_travel_s:.2f}s | avg waiting time={avg_wait_s:.2f}s"
        )
        recorder.save(avg_return, avg_throughput, avg_travel_s, avg_wait_s)
        return avg_return
    finally:
        actor.train(original_mode)


def run_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    set_global_seed(args.seed)
    best_eval_return = -np.inf

    # Env & shapes
    train_env = build_train_env(args)
    first_obs = train_env.reset()
    agent_id_list = list(train_env.agent_ids)
    N = len(agent_id_list)
    O = len(next(iter(first_obs.values())))
    A_dim = train_env.action_spaces[agent_id_list[0]].n

    # LSTM actor (shared, decentralized); per-agent critic is MLP on local obs
    actor  = ActorLSTM(O, A_dim, hidden=args.hidden).to(device)
    critic = CriticMLP(O, hidden=args.hidden).to(device)

    optim_actor  = optim.Adam(actor.parameters(),  lr=args.lr_actor)
    optim_critic = optim.Adam(critic.parameters(), lr=args.lr_critic)

    run_name = "ia2c_lstm_actor_mlp_critic"
    clear_eval_history(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)

    # Entropy schedule (multiplicative decay per update)
    entropy_coef = args.entropy_coef_start

    for ep_idx in range(1, args.episodes + 1):
        # -------- on-policy rollout (single episode) --------
        actor.eval(); critic.eval()

        obs_dict = train_env.reset()
        done = False
        steps = 0
        actor_h = None

        obs_list, act_list, rew_list, done_list = [], [], [], []

        while not done and steps < args.episode_steps:
            obs_mat = np.stack([obs_dict[aid] for aid in agent_id_list], 0).astype(np.float32)  # [N,O]
            obs_t = torch.as_tensor(obs_mat, dtype=torch.float32, device=device)

            with torch.no_grad():
                logits_t, actor_h = actor.step(obs_t, actor_h)      # [N,A]
                dist = Categorical(logits=logits_t)
                a_sample = dist.sample()                            # [N]

            action_dict = {aid: int(a_sample[i].item()) for i, aid in enumerate(agent_id_list)}
            next_obs_dict, reward_dict, done, _ = train_env.step(action_dict)

            rewards_vec = np.array([float(reward_dict[aid]) for aid in agent_id_list], dtype=np.float32)

            # record
            obs_list.append(obs_t)                                        # [N,O] on device
            act_list.append(a_sample.long())                               # [N]
            rew_list.append(torch.as_tensor(rewards_vec, device=device))   # [N]
            done_list.append(torch.as_tensor(float(done), device=device))  # []

            obs_dict = next_obs_dict
            steps += 1

        if steps == 0:
            print(f"[ep {ep_idx}] rollout empty (env ended immediately).")
            continue

        # Stack rollout
        obs_seq  = torch.stack(obs_list,  dim=0)        # [T,N,O]
        act_seq  = torch.stack(act_list,  dim=0).long() # [T,N]
        rew_seq  = torch.stack(rew_list,  dim=0)        # [T,N]
        done_seq = torch.stack(done_list, dim=0)        # [T]

        # Final local obs for per-agent bootstrap
        last_obs_mat = np.stack([obs_dict[aid] for aid in agent_id_list], 0).astype(np.float32)  # [N,O]
        last_obs = torch.as_tensor(last_obs_mat, dtype=torch.float32, device=device)              # [N,O]

        # -------- single update --------
        actor.train(); critic.train()
        logs = ia2c_update_lstm_actor_mlp(
            actor=actor,
            critic=critic,
            optim_actor=optim_actor,
            optim_critic=optim_critic,
            obs_seq=obs_seq,
            act_seq=act_seq,
            rew_seq=rew_seq,
            done_seq=done_seq,
            last_obs=last_obs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            entropy_coef=entropy_coef,
            value_coef=args.value_coef,
            grad_clip=args.grad_clip,
            # stabilizers (match your IA2C style)
            normalize_rewards=args.normalize_rewards,
            reward_scale=args.reward_scale,
            normalize_adv=args.normalize_adv,
            huber_delta=args.huber_delta,
            value_clip_eps=args.value_clip_eps,
        )

        print(f"[ep {ep_idx}] | "
              f"loss_total={logs['loss_total']:.3f} "
              f"(pi={logs['loss_policy']:.3f}, v={logs['loss_value']:.3f}, H={logs['entropy']:.3f}) "
              f"| ent_coef={entropy_coef:.4f}")

        # entropy decay
        entropy_coef = max(args.entropy_coef_end, entropy_coef * args.entropy_coef_decay)

        # periodic eval
        if ep_idx % args.eval_every == 0:
            eval_return = evaluate_ia2c_lstm_actor(args, actor, agent_id_list, run_name=run_name)
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                save_path = f"{args.logdir}_grid_{args.grid_n}/seed{args.seed}/model_best_{run_name}_seed{args.seed}.pt"
                torch.save(actor.state_dict(), save_path)

    train_env.close()


def parse_args():
    parser = argparse.ArgumentParser("IA2C (LSTM actor, MLP critic, SHARED)")
    parser.add_argument('--grid-n', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--episode-steps', type=int, default=100)
    parser.add_argument('--sumo-steps-per-env-step', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--gui-delay-ms', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--logdir', type=str, default='logs')

    # Learning & losses
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--gae-lambda', type=float, default=0.90)
    parser.add_argument('--value-coef', type=float, default=0.7)
    parser.add_argument('--grad-clip', type=float, default=1.0)

    # Entropy decay
    parser.add_argument('--entropy-coef-start', type=float, default=0.05)
    parser.add_argument('--entropy-coef-end',   type=float, default=0.005)
    parser.add_argument('--entropy-coef-decay', type=float, default=0.995)

    # Stabilization knobs
    parser.add_argument('--normalize-rewards', action='store_true', default=True)
    parser.add_argument('--reward-scale', type=float, default=1.0)
    parser.add_argument('--normalize-adv', action='store_true', default=True)
    parser.add_argument('--huber-delta', type=float, default=1.0)
    parser.add_argument('--value-clip-eps', type=float, default=0.2)

    # Nets/opt
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--lr-actor', type=float, default=3e-4)
    parser.add_argument('--lr-critic', type=float, default=1e-3)

    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
