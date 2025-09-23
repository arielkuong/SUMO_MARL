#!/usr/bin/env python3
from __future__ import annotations
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.optim as optim

from marl_utils.models import RecurrentQNet
from marl_utils.replay_buffers import AgentEpisode, SequenceReplay
from marl_utils.network_update import drqn_update, soft_update
from marl_utils.common import (
    set_global_seed,
    EvalHistory,
    clear_eval_history,
    build_neighbor_map,
    augment_obs_with_neighbors,
)
from env_builder import build_train_env, get_or_create_eval_pool

# --------- evaluation (greedy over cached fixed-route envs, LSTM) ----------
@torch.no_grad()
def evaluate_idqn_lstm_shared_neighbor_obs(
    args,
    shared_q_network: RecurrentQNet,
    agent_id_list: List[str],
    run_name: str,
) -> float:

    original_mode = shared_q_network.training
    shared_q_network.eval()
    device = next(shared_q_network.parameters()).device

    eval_env_pool = get_or_create_eval_pool(args)
    recorder = EvalHistory(args.logdir + '_grid_' + str(args.grid_n), run_name)

    try:
        returns_all, throughput_all, mean_travel_all, mean_wait_all = [], [], [], []

        for eval_env in eval_env_pool.envs:
            episode_return = 0.0
            last_info_dict = {}
            observation_dict = eval_env.reset()
            done_flag = False

            # Rebuild neighbour map for THIS env's TLS ordering
            eval_agent_ids = list(eval_env.agent_ids)
            eval_neighbor_map = build_neighbor_map(eval_agent_ids)

            # Per-agent hidden state for rollout
            hidden_states: Dict[str, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {aid: None for aid in eval_agent_ids}

            while not done_flag:
                # Augment with neighbour summaries
                aug_obs_dict = augment_obs_with_neighbors(observation_dict, eval_agent_ids, eval_neighbor_map)

                # Act per agent using 1-step LSTM unroll, keeping hidden states
                action_dict: Dict[str, int] = {}
                for aid in eval_agent_ids:
                    obs_ = torch.as_tensor(aug_obs_dict[aid], dtype=torch.float32, device=device)
                    obs_ = obs_.view(1, 1, -1)  # [B=1, T=1, O]
                    q_values_seq, hidden_states[aid] = shared_q_network(obs_, hidden_states[aid])  # [1,1,A], (h,c)
                    q_values = q_values_seq[:, -1, :]  # [1, A]
                    greedy_action = int(torch.argmax(q_values, dim=-1).item())
                    action_dict[aid] = greedy_action

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
            f"[EVAL {run_name}] MEAN over {len(eval_env_pool.envs)} cases | "
            f"return={avg_return:.2f} | throughput={avg_throughput:.2f} veh/h | "
            f"mean travel time={avg_travel_s:.2f}s | "
            f"avg waiting time={avg_wait_s:.2f}s"
        )

        recorder.save(avg_return, avg_throughput, avg_travel_s, avg_wait_s)
        return avg_return
    finally:
        shared_q_network.train(original_mode)


# ------------------------------------------------------------

def run_training(args):
    compute_device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    set_global_seed(args.seed)

    # Build training environment (random flows; respects args.grid_n)
    train_env = build_train_env(args)
    observation_dict = train_env.reset()
    agent_id_list = list(train_env.agent_ids)

    # Build neighbour map ONCE for training TLS ids
    neighbor_map = build_neighbor_map(agent_id_list)

    # Work out augmented obs dim (apply augmentation once on the initial obs)
    aug_once = augment_obs_with_neighbors(observation_dict, agent_id_list, neighbor_map)
    augmented_observation_dim = len(next(iter(aug_once.values())))
    action_dim = train_env.action_spaces[agent_id_list[0]].n

    # Shared LSTM Q-network + target
    shared_q_network = RecurrentQNet(augmented_observation_dim, action_dim, hidden_units=args.hidden).to(compute_device)
    target_q_network = RecurrentQNet(augmented_observation_dim, action_dim, hidden_units=args.hidden).to(compute_device)
    target_q_network.load_state_dict(shared_q_network.state_dict())
    target_q_network.eval()

    optimizer_q = optim.Adam(shared_q_network.parameters(), lr=args.lr)

    # One shared SEQUENCE replay that stores full per-agent episodes
    seq_replay = SequenceReplay(capacity_steps=args.replay_size, observation_dim=augmented_observation_dim, seed=args.rb_seed)

    exploration_epsilon = args.eps_start
    total_env_steps_collected = 0
    run_name = "drqn_lstm_shared_nbobs_seqlen" + str(args.seq_len)

    clear_eval_history(args.logdir + '_grid_' + str(args.grid_n), run_name)

    for episode_idx in range(1, args.episodes + 1):
        observation_dict = train_env.reset()
        episode_done_flag = False

        # Buffers to record sequences per agent for this episode
        # We accumulate full sequences then push to SequenceReplay at episode end.
        per_agent_obs: Dict[str, List[np.ndarray]] = {aid: [] for aid in agent_id_list}
        per_agent_next_obs: Dict[str, List[np.ndarray]] = {aid: [] for aid in agent_id_list}
        per_agent_actions: Dict[str, List[int]] = {aid: [] for aid in agent_id_list}
        per_agent_rewards: Dict[str, List[float]] = {aid: [] for aid in agent_id_list}
        per_agent_dones: Dict[str, List[float]] = {aid: [] for aid in agent_id_list}

        # Per-agent hidden states during rollout
        hidden_states: Dict[str, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {aid: None for aid in agent_id_list}

        steps_this_episode = 0

        while not episode_done_flag and steps_this_episode < args.episode_steps:
            # Augment with neighbour summaries
            aug_obs_dict = augment_obs_with_neighbors(observation_dict, agent_id_list, neighbor_map)

            # Select actions with LSTM (1-step unroll), keep hidden state per agent
            action_dict: Dict[str, int] = {}
            with torch.no_grad():
                for aid in agent_id_list:
                    obs_ = torch.as_tensor(aug_obs_dict[aid], dtype=torch.float32, device=compute_device)
                    obs_ = obs_.view(1, 1, -1)  # [B=1, T=1, O]
                    q_values_seq, hidden_states[aid] = shared_q_network(obs_, hidden_states[aid])  # [1,1,A]
                    q_values = q_values_seq[:, -1, :]  # [1, A]
                    greedy = int(torch.argmax(q_values, dim=-1).item())
                    act = greedy if (np.random.rand() >= exploration_epsilon) else int(np.random.randint(action_dim))
                    action_dict[aid] = act

            next_observation_dict, reward_dict, episode_done_flag, _ = train_env.step(action_dict)

            # Build next augmented observations
            aug_next_obs_dict = augment_obs_with_neighbors(next_observation_dict, agent_id_list, neighbor_map)

            # Record per-agent step into episode buffers
            for aid in agent_id_list:
                per_agent_obs[aid].append(aug_obs_dict[aid])
                per_agent_next_obs[aid].append(aug_next_obs_dict[aid])
                per_agent_actions[aid].append(int(action_dict[aid]))
                per_agent_rewards[aid].append(float(reward_dict[aid]))
                per_agent_dones[aid].append(float(episode_done_flag))

            observation_dict = next_observation_dict
            total_env_steps_collected += 1
            steps_this_episode += 1

        # Push each agent's whole episode into sequence replay
        for aid in agent_id_list:
            ep = AgentEpisode(
                observations=np.asarray(per_agent_obs[aid], dtype=np.float32),
                actions=np.asarray(per_agent_actions[aid], dtype=np.int64),
                rewards=np.asarray(per_agent_rewards[aid], dtype=np.float32),
                dones=np.asarray(per_agent_dones[aid], dtype=np.float32),
                next_observations=np.asarray(per_agent_next_obs[aid], dtype=np.float32),
            )
            seq_replay.push_episode(ep)

        # ---------------- Parameter updates (shared DRQN) ----------------
        mean_training_loss = 0.0
        # if total_env_steps_collected >= args.warmup_steps and len(seq_replay) > 0:
        if total_env_steps_collected >= args.warmup_steps:
            loss_values_for_logging: List[float] = []
            for _ in range(args.updates_per_ep):
                batch = seq_replay.sample(batch_size=args.batch_size, seq_len=args.seq_len)
                loss_val = drqn_update(
                    device=compute_device,
                    batch=batch,
                    online_qnet=shared_q_network,
                    target_qnet=target_q_network,
                    optimizer_q=optimizer_q,
                    gamma=args.gamma,
                    burn_in=args.burn_in,
                    double_dqn=True,
                )
                loss_values_for_logging.append(loss_val)
                # soft target update every grad step
                soft_update(target_q_network, shared_q_network, args.tau)

            mean_training_loss = float(np.mean(loss_values_for_logging)) if loss_values_for_logging else 0.0
            print(f"[ep {episode_idx}] train_loss={mean_training_loss:.3f} eps={exploration_epsilon:.3f}")

            if episode_idx % args.eval_every == 0:
                evaluate_idqn_lstm_shared_neighbor_obs(args, shared_q_network, agent_id_list, run_name=run_name)

            exploration_epsilon = max(args.eps_end, exploration_epsilon * args.eps_decay)
        else:
            print(f"[ep {episode_idx}] total steps={total_env_steps_collected} (warming up) eps={exploration_epsilon:.3f}")

    train_env.close()


def parse_args():
    parser = argparse.ArgumentParser("IDQN (LSTM/DRQN, SHARED) + 1-hop neighbour-augmented obs — Train on Random, Eval on Fixed")
    parser.add_argument('--grid-n', type=int, default=3, help='Core grid size N (NxN traffic lights)')
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

    # model / opt
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tau', type=float, default=0.005)

    # exploration
    parser.add_argument('--eps-start', type=float, default=0.5)
    parser.add_argument('--eps-end', type=float, default=0.05)
    parser.add_argument('--eps-decay', type=float, default=0.998)

    # Replay / optimisation
    parser.add_argument('--replay-size', type=int, default=100000, help='capacity in steps for sequence replay')
    parser.add_argument('--rb-seed', type=int, default=1234)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16, help='number of sequences per update batch')
    parser.add_argument('--seq-len', type=int, default=8, help='training unroll length (after burn-in)')
    parser.add_argument('--burn-in', type=int, default=4, help='burn-in steps for hidden warmup inside DRQN update')
    parser.add_argument('--updates-per-ep', type=int, default=32)


    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
