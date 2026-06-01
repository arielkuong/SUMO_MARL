#!/usr/bin/env python3
from __future__ import annotations
import os
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from marl_utils.models import QNetMLP, QMIXMixer
from marl_utils.replay_buffers import JointReplayBuffer
from marl_utils.network_update import qmix_update_mlp, soft_update
from marl_utils.common import parse_args, set_global_seed, EvalHistory, clear_eval_history
from env_builder import build_train_env, get_or_create_eval_pool


# --------- evaluation (decentralized greedy over cached fixed-route envs) ----------
@torch.no_grad()
def evaluate_qmix_mlp_shared(
    args,
    shared_q_network: QNetMLP,
    agent_id_list: List[str],
    run_name: str,
) -> float:
    """
    Evaluation remains decentralised.

    The QMIX mixer is not used here.

    Each agent selects:

        a_i = argmax_a Q_i(o_i, a)
    """

    original_mode = shared_q_network.training
    shared_q_network.eval()

    device = next(shared_q_network.parameters()).device

    eval_env_pool = get_or_create_eval_pool(args)
    recorder = EvalHistory(
        args.logdir + "_grid_" + str(args.grid_n),
        run_name,
        args.seed,
    )

    try:
        returns_all = []
        throughput_all = []
        mean_travel_all = []
        mean_wait_all = []

        for eval_env in eval_env_pool.envs:
            episode_return = 0.0
            last_info_dict = {}

            observation_dict = eval_env.reset()
            done_flag = False

            while not done_flag:
                obs_matrix = np.stack(
                    [observation_dict[aid] for aid in agent_id_list],
                    axis=0,
                ).astype(np.float32)                                  # [N, O]

                obs_tensor = torch.as_tensor(
                    obs_matrix,
                    dtype=torch.float32,
                    device=device,
                )                                                     # [N, O]

                q_values = shared_q_network(obs_tensor)                # [N, A]
                greedy_actions = torch.argmax(q_values, dim=-1).tolist()

                action_dict = {
                    aid: int(greedy_actions[i])
                    for i, aid in enumerate(agent_id_list)
                }

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
            f"mean travel time={avg_travel_s:.2f}s | "
            f"avg waiting time={avg_wait_s:.2f}s"
        )

        recorder.save(avg_return, avg_throughput, avg_travel_s, avg_wait_s)

        return avg_return

    finally:
        shared_q_network.train(original_mode)

# ------------------------------------------------------------

def run_training(args):
    device = torch.device(args.device)

    set_global_seed(args.seed)

    best_eval_return = -np.inf

    # Build training environment.
    train_env = build_train_env(args)
    observation_dict = train_env.reset()

    agent_id_list = list(train_env.agent_ids)
    N = len(agent_id_list)
    O = len(next(iter(observation_dict.values())))
    A_dim = train_env.action_spaces[agent_id_list[0]].n

    state_dim = N * O

    print(
        f"QMIX training setup | "
        f"N={N} agents | O={O} obs dim | A={A_dim} actions | "
        f"state_dim={state_dim} | device={device}"
    )

    # Shared per-agent Q_i(o_i, a_i).
    # Execution is decentralised.
    online_q = QNetMLP(O, A_dim, hidden_units=args.hidden).to(device)
    target_q = QNetMLP(O, A_dim, hidden_units=args.hidden).to(device)

    target_q.load_state_dict(online_q.state_dict())
    target_q.eval()

    # Centralised QMIX mixers.
    # These are used only during training.
    online_mixer = QMIXMixer(
        num_agents=N,
        state_dim=state_dim,
        mixing_embed_dim=args.mixing_embed_dim,
        hypernet_embed_dim=args.hypernet_embed_dim,
    ).to(device)

    target_mixer = QMIXMixer(
        num_agents=N,
        state_dim=state_dim,
        mixing_embed_dim=args.mixing_embed_dim,
        hypernet_embed_dim=args.hypernet_embed_dim,
    ).to(device)

    target_mixer.load_state_dict(online_mixer.state_dict())
    target_mixer.eval()

    optimizer = optim.Adam(
        list(online_q.parameters()) + list(online_mixer.parameters()),
        lr=args.lr,
    )

    # CTDE joint replay.
    joint_replay = JointReplayBuffer(
        capacity=args.replay_size,
        num_agents=N,
        obs_dim=O,
        seed=args.rb_seed,
    )

    eps = args.eps_start
    total_steps = 0
    run_name = "qmix_ctde_mlp_shared"

    log_root = args.logdir + "_grid_" + str(args.grid_n)
    save_dir = f"{log_root}/seed{args.seed}"
    os.makedirs(save_dir, exist_ok=True)

    # Wipe old eval history of this run.
    clear_eval_history(log_root, run_name, args.seed)

    for ep_idx in range(1, args.episodes + 1):
        observation_dict = train_env.reset()

        done = False
        steps_this_ep = 0
        episode_return = 0.0

        while not done and steps_this_ep < args.episode_steps:
            # Local observations stacked.
            obs_matrix = np.stack(
                [observation_dict[aid] for aid in agent_id_list],
                axis=0,
            ).astype(np.float32)                                      # [N, O]

            obs_tensor = torch.as_tensor(
                obs_matrix,
                dtype=torch.float32,
                device=device,
            )                                                         # [N, O]

            with torch.no_grad():
                q_all = online_q(obs_tensor)                          # [N, A]
                greedy = torch.argmax(q_all, dim=-1).cpu().numpy()    # [N]

            # Epsilon-greedy per agent.
            # This is still decentralised action selection.
            actions_np = greedy.copy()

            for i in range(N):
                if np.random.rand() < eps:
                    actions_np[i] = np.random.randint(A_dim)

            action_dict = {
                aid: int(actions_np[i])
                for i, aid in enumerate(agent_id_list)
            }

            next_obs_dict, reward_dict, done, _ = train_env.step(action_dict)

            next_obs_matrix = np.stack(
                [next_obs_dict[aid] for aid in agent_id_list],
                axis=0,
            ).astype(np.float32)                                      # [N, O]

            rewards_vec = np.array(
                [float(reward_dict[aid]) for aid in agent_id_list],
                dtype=np.float32,
            )                                                        # [N]

            episode_return += float(np.sum(rewards_vec))

            # CTDE: push one joint transition per environment step.
            joint_replay.push(
                state_all=obs_matrix,
                actions_all=actions_np.astype(np.int64),
                rewards_all=rewards_vec,
                next_state_all=next_obs_matrix,
                done_flag=float(done),
            )

            observation_dict = next_obs_dict

            total_steps += 1
            steps_this_ep += 1

        # ----------------------------------------------------------------------
        # Centralised training with QMIX
        # ----------------------------------------------------------------------

        mean_loss = 0.0

        if total_steps >= args.warmup_steps and len(joint_replay) >= args.batch_size:
            losses = []

            for _ in range(args.updates_per_ep):
                batch = joint_replay.sample(args.batch_size)

                loss_val = qmix_update_mlp(
                    device=device,
                    online_q=online_q,
                    target_q=target_q,
                    online_mixer=online_mixer,
                    target_mixer=target_mixer,
                    optimizer=optimizer,
                    batch_tuple=batch,
                    gamma=args.gamma,
                    double_dqn=True,
                    grad_clip=args.grad_clip,
                )

                losses.append(loss_val)

                # Soft target updates.
                soft_update(target_q, online_q, args.tau)
                soft_update(target_mixer, online_mixer, args.tau)

            mean_loss = float(np.mean(losses))

            print(
                f"[ep {ep_idx}] "
                f"return={episode_return:.2f} | "
                f"train_loss={mean_loss:.3f} | "
                f"eps={eps:.3f} | "
                f"steps={steps_this_ep} | "
                f"buffer={len(joint_replay)}"
            )

            if ep_idx % args.eval_every == 0:
                # Evaluation is still decentralised greedy.
                eval_return = evaluate_qmix_mlp_shared(
                    args=args,
                    shared_q_network=online_q,
                    agent_id_list=agent_id_list,
                    run_name=run_name,
                )

                if eval_return > best_eval_return:
                    best_eval_return = eval_return

                    save_path = (
                        f"{save_dir}/model_best_{run_name}_seed{args.seed}.pt"
                    )

                    torch.save(
                        {
                            "online_q": online_q.state_dict(),
                            "target_q": target_q.state_dict(),
                            "online_mixer": online_mixer.state_dict(),
                            "target_mixer": target_mixer.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "agent_id_list": agent_id_list,
                            "num_agents": N,
                            "obs_dim": O,
                            "action_dim": A_dim,
                            "state_dim": state_dim,
                            "mixing_embed_dim": args.mixing_embed_dim,
                            "hypernet_embed_dim": args.hypernet_embed_dim,
                            "best_eval_return": best_eval_return,
                        },
                        save_path,
                    )

                    print(
                        f"[ep {ep_idx}] saved new best QMIX model to {save_path} | "
                        f"best_eval_return={best_eval_return:.2f}"
                    )

            # Epsilon decay after training starts.
            eps = max(args.eps_end, eps * args.eps_decay)

        else:
            print(
                f"[ep {ep_idx}] "
                f"return={episode_return:.2f} | "
                f"total_steps={total_steps} warming up | "
                f"eps={eps:.3f} | "
                f"buffer={len(joint_replay)}"
            )

    train_env.close()


if __name__ == "__main__":
    run_training(parse_args())
