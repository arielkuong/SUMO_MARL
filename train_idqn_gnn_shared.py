#!/usr/bin/env python3
from __future__ import annotations
import argparse
from typing import List
import numpy as np
import torch
import torch.optim as optim

from marl_utils.models import GNNPolicyQ
from marl_utils.replay_buffers import GlobalReplayBuffer
from marl_utils.network_update import dqn_update_shared_gnn, soft_update
from marl_utils.common import (
    set_global_seed,
    build_grid_edge_index,
    EvalHistory,
    clear_eval_history,
)
from env_builder import build_train_env, get_or_create_eval_pool

# --------- evaluation (greedy over cached fixed-route envs) ----------
@torch.no_grad()
def evaluate_idqn_gnn_shared(
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
    recorder = EvalHistory(args.logdir + '_grid_' + str(args.grid_n), run_name)

    try:
        returns_all, throughput_all, mean_travel_all, mean_wait_all = [], [], [], []

        for eval_env in eval_env_pool.envs:
            episode_return = 0.0
            last_info_dict = {}
            observation_dict = eval_env.reset()
            done_flag = False

            while not done_flag:
                # Node features for full graph in fixed agent order
                X_np = np.stack([observation_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)  # [N, O]
                X = torch.as_tensor(X_np, dtype=torch.float32, device=device)

                q_values = shared_q_network(X, edge_index.to(device))  # [N, A]
                greedy_actions = torch.argmax(q_values, dim=-1).tolist()
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

    # Training env (random flows)
    train_env = build_train_env(args)
    observation_dict = train_env.reset()
    agent_id_list = list(train_env.agent_ids)

    # Fixed grid topology (directed 4-neighbour edges)
    edge_index = build_grid_edge_index(agent_id_list)  # [2, E] (torch.long)

    observation_dim = len(next(iter(observation_dict.values())))
    action_dim = train_env.action_spaces[agent_id_list[0]].n

    # Shared GNN Q-networks (online + target)
    shared_q_network = GNNPolicyQ(node_dim=observation_dim, actions=action_dim, hidden=args.hidden, layers=2).to(compute_device)
    target_q_network = GNNPolicyQ(node_dim=observation_dim, actions=action_dim, hidden=args.hidden, layers=2).to(compute_device)
    target_q_network.load_state_dict(shared_q_network.state_dict())
    target_q_network.eval()

    optimizer_q = optim.Adam(shared_q_network.parameters(), lr=args.lr)

    # Joint (global) replay buffer: stores full-graph snapshots per step
    shared_replay_buffer = GlobalReplayBuffer(
        capacity=args.replay_size,
        num_agents=len(agent_id_list),
        obs_dim=observation_dim,
        seed=args.rb_seed
    )

    exploration_epsilon = args.eps_start
    total_env_steps_collected = 0
    run_name = "dqn_gnn_shared"

    clear_eval_history(args.logdir + '_grid_' + str(args.grid_n), run_name)

    for episode_idx in range(1, args.episodes + 1):
        observation_dict = train_env.reset()
        episode_done_flag = False
        steps_this_episode = 0

        while not episode_done_flag and steps_this_episode < args.episode_steps:
            X_np = np.stack([observation_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)  # [N, O]
            X = torch.as_tensor(X_np, dtype=torch.float32, device=compute_device)

            with torch.no_grad():
                q_values_all = shared_q_network(X, edge_index.to(compute_device))  # [N, A]
                greedy_actions = torch.argmax(q_values_all, dim=-1).cpu().numpy()  # [N]

            chosen_actions_np = greedy_actions.copy()
            for i in range(len(agent_id_list)):
                if np.random.rand() < exploration_epsilon:
                    chosen_actions_np[i] = np.random.randint(action_dim)

            action_dict = {aid: int(chosen_actions_np[i]) for i, aid in enumerate(agent_id_list)}

            next_observation_dict, reward_dict, episode_done_flag, _ = train_env.step(action_dict)

            X_next_np = np.stack([next_observation_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)  # [N, O]
            R_np = np.array([float(reward_dict[aid]) for aid in agent_id_list], dtype=np.float32)                   # [N]

            # Push full-graph transition
            shared_replay_buffer.push(
                state_all=X_np,
                actions_all=chosen_actions_np.astype(np.int64),
                rewards_all=R_np,
                next_state_all=X_next_np,
                done_flag=float(episode_done_flag),
            )

            observation_dict = next_observation_dict
            total_env_steps_collected += 1
            steps_this_episode += 1

        # Parameter updates (shared)
        mean_training_loss = 0.0
        if total_env_steps_collected >= args.warmup_steps:
            loss_values_for_logging: List[float] = []
            for _ in range(args.updates_per_ep):
                if len(shared_replay_buffer) < args.batch_size:
                    break
                batch_tuple = shared_replay_buffer.sample(args.batch_size)
                loss_val = dqn_update_shared_gnn(
                    device=compute_device,
                    gnn_online_q_network=shared_q_network,
                    gnn_target_q_network=target_q_network,
                    optimizer_q=optimizer_q,
                    batch_tuple=batch_tuple,
                    graph_edge_index=edge_index,    # fixed grid edges
                    graph_edge_features=None,       # no edge feats by default
                    discount_gamma=args.gamma,
                    double_dqn=True,
                )
                loss_values_for_logging.append(loss_val)

                if episode_idx % args.eval_every == 0:
                    # soft target update every grad step
                    soft_update(target_q_network, shared_q_network, args.tau)

            mean_training_loss = float(np.mean(loss_values_for_logging)) if loss_values_for_logging else 0.0
            print(f"[ep {episode_idx}] train_loss={mean_training_loss:.3f} eps={exploration_epsilon:.3f}")

            if episode_idx % args.eval_every == 0:
                # Evaluate across ALL fixed cases (via cached env pool)
                evaluate_idqn_gnn_shared(args, shared_q_network, agent_id_list, edge_index, run_name=run_name)

            # epsilon decay
            exploration_epsilon = max(args.eps_end, exploration_epsilon * args.eps_decay)
        else:
            print(f"[ep {episode_idx}] total steps={total_env_steps_collected} (warming up) eps={exploration_epsilon:.3f}")

    train_env.close()


def parse_args():
    parser = argparse.ArgumentParser("IDQN (GNN, SHARED) — Train on Random Flows, Eval on Fixed Trips")
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

    parser.add_argument('--hidden', type=int, default=128)   # GNN hidden width
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eps-start', type=float, default=0.5)
    parser.add_argument('--eps-end', type=float, default=0.05)
    parser.add_argument('--eps-decay', type=float, default=0.998)
    parser.add_argument('--replay-size', type=int, default=100000)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=128)   # smaller default—GNN forward is heavier
    parser.add_argument('--updates-per-ep', type=int, default=32)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--rb-seed', type=int, default=1234)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
