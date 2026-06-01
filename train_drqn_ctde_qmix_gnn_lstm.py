#!/usr/bin/env python3
from __future__ import annotations
import os
from typing import List, Dict
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime

from marl_utils.models import GNNLSTMPolicyQ, QMIXMixer
from marl_utils.network_update import qmix_update_gnn_lstm, soft_update
from marl_utils.common import (
    parse_args,
    set_global_seed,
    build_grid_edge_index,
    EvalHistory,
    clear_eval_history,
)
from marl_utils.replay_buffers import GlobalSequenceReplay   # <— use your existing buffer
from env_builder import build_train_env, get_or_create_eval_pool


# ---------------- evaluation (greedy, recurrent) ----------------
@torch.no_grad()
def evaluate_qmix_gnnlstm_shared(
    args,
    online_q: GNNLSTMPolicyQ,
    agent_id_list: List[str],
    edge_index: torch.Tensor,
    run_name: str,
) -> float:
    """
    Evaluation remains decentralised.

    The QMIX mixer is not used during execution.

    Each agent chooses:

        a_i = argmax_a Q_i(o_i, h_i, a)

    using the shared GNN-LSTM Q-network.
    """

    original_mode = online_q.training
    online_q.eval()

    device = next(online_q.parameters()).device
    edge_index_device = edge_index.to(device)

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

        for env in eval_env_pool.envs:
            episode_return = 0.0
            last_info: Dict = {}

            obs_dict = env.reset()
            done = False

            # Recurrent state for the whole multi-agent graph.
            rnn_state = None

            while not done:
                X_np = np.stack(
                    [obs_dict[aid] for aid in agent_id_list],
                    axis=0,
                ).astype(np.float32)                                  # [N, O]

                X = torch.as_tensor(
                    X_np,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)                                         # [1, N, O]

                q_t, rnn_state = online_q.step(
                    X,
                    edge_index_device,
                    rnn_state,
                )                                                       # [1, N, A]

                greedy = torch.argmax(q_t[0], dim=-1).tolist()          # [N]

                action_dict = {
                    aid: int(greedy[i])
                    for i, aid in enumerate(agent_id_list)
                }

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
            f"[{datetime.now()}, EVAL {run_name}] "
            f"MEAN over {len(eval_env_pool.envs)} cases | "
            f"return={avg_return:.2f} | "
            f"throughput={avg_throughput:.2f} veh/h | "
            f"mean travel time={avg_travel_s:.2f}s | "
            f"avg waiting time={avg_wait_s:.2f}s"
        )

        recorder.save(
            avg_return,
            avg_throughput,
            avg_travel_s,
            avg_wait_s,
        )

        return avg_return

    finally:
        online_q.train(original_mode)


# ------------------ training (with replay) ------------------
def run_training(args):
    device = torch.device(args.device)

    set_global_seed(args.seed)

    best_eval_return = -np.inf

    assert args.seq_len > 0 and 0 <= args.burn_in < args.seq_len, (
        f"Require seq_len > 0 and 0 <= burn_in < seq_len, "
        f"got seq_len={args.seq_len}, burn_in={args.burn_in}"
    )

    # ------------------------------------------------------------------
    # Environment and shapes
    # ------------------------------------------------------------------

    train_env = build_train_env(args)
    obs_dict = train_env.reset()

    agent_id_list = list(train_env.agent_ids)

    N = len(agent_id_list)
    O = len(next(iter(obs_dict.values())))
    A_dim = train_env.action_spaces[agent_id_list[0]].n

    state_dim = N * O

    print(
        f"QMIX-GNN-LSTM training setup | "
        f"N={N} agents | "
        f"O={O} obs dim | "
        f"A={A_dim} actions | "
        f"state_dim={state_dim} | "
        f"seq_len={args.seq_len} | "
        f"burn_in={args.burn_in} | "
        f"device={device}"
    )

    # Fixed grid topology on CPU. Move to device when used.
    edge_index = build_grid_edge_index(agent_id_list)  # [2, E], torch.long

    # ------------------------------------------------------------------
    # Spatio-temporal per-agent Q networks
    # ------------------------------------------------------------------

    online_q = GNNLSTMPolicyQ(
        node_dim=O,
        actions=A_dim,
        hidden=args.hidden,
        gnn_layers=2,
    ).to(device)

    target_q = GNNLSTMPolicyQ(
        node_dim=O,
        actions=A_dim,
        hidden=args.hidden,
        gnn_layers=2,
    ).to(device)

    target_q.load_state_dict(online_q.state_dict())
    target_q.eval()

    # ------------------------------------------------------------------
    # QMIX mixers
    # ------------------------------------------------------------------

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

    # Optimise both the GNN-LSTM Q-network and the QMIX mixer.
    optimizer = optim.Adam(
        list(online_q.parameters()) + list(online_mixer.parameters()),
        lr=args.lr,
    )

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------

    replay = GlobalSequenceReplay(
        capacity_steps=args.replay_size,
        num_agents=N,
        obs_dim=O,
        seed=args.rb_seed,
    )

    eps = args.eps_start
    total_steps = 0

    run_name = f"qmix_ctde_gnn_lstm_shared_seqlen{args.seq_len}"

    log_root = args.logdir + "_grid_" + str(args.grid_n)
    save_dir = f"{log_root}/seed{args.seed}"
    os.makedirs(save_dir, exist_ok=True)

    clear_eval_history(log_root, run_name, args.seed)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    for ep_idx in range(1, args.episodes + 1):
        obs_dict = train_env.reset()

        done = False
        steps = 0
        episode_return = 0.0

        # Episode buffers.
        X_list = []
        A_list = []
        R_list = []
        D_list = []
        Xn_list = []

        # Same behaviour as your VDN script: stateless step for action selection.
        # You can maintain a recurrent state here too, but this keeps the change minimal.
        while not done and steps < args.episode_steps:
            X_np = np.stack(
                [obs_dict[aid] for aid in agent_id_list],
                axis=0,
            ).astype(np.float32)                                      # [N, O]

            with torch.no_grad():
                X = torch.as_tensor(
                    X_np,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)                                         # [1, N, O]

                q_all, _ = online_q.step(
                    X,
                    edge_index.to(device),
                )                                                       # [1, N, A]

                greedy = torch.argmax(q_all[0], dim=-1).cpu().numpy()   # [N]

            # Epsilon-greedy per agent.
            actions_np = greedy.copy()

            for i in range(N):
                if np.random.rand() < eps:
                    actions_np[i] = np.random.randint(A_dim)

            action_dict = {
                aid: int(actions_np[i])
                for i, aid in enumerate(agent_id_list)
            }

            next_obs_dict, reward_dict, done, _ = train_env.step(action_dict)

            Xn_np = np.stack(
                [next_obs_dict[aid] for aid in agent_id_list],
                axis=0,
            ).astype(np.float32)                                      # [N, O]

            R_np = np.array(
                [float(reward_dict[aid]) for aid in agent_id_list],
                dtype=np.float32,
            )                                                        # [N]

            episode_return += float(np.sum(R_np))

            # Record this joint step.
            X_list.append(X_np)
            A_list.append(actions_np.astype(np.int64))
            R_list.append(R_np)
            D_list.append(float(done))
            Xn_list.append(Xn_np)

            obs_dict = next_obs_dict

            total_steps += 1
            steps += 1

        # Push full joint episode.
        if steps > 0:
            replay.push_episode(
                states=np.stack(X_list, axis=0),       # [T, N, O]
                actions=np.stack(A_list, axis=0),      # [T, N]
                rewards=np.stack(R_list, axis=0),      # [T, N]
                next_states=np.stack(Xn_list, axis=0), # [T, N, O]
                dones=np.stack(D_list, axis=0),        # [T]
            )

        # ------------------------------------------------------------------
        # Centralised training: QMIX + Double-DQN over GNN-LSTM Q_i
        # ------------------------------------------------------------------

        mean_loss = 0.0

        if total_steps >= args.warmup_steps and len(replay) > 0:
            losses = []

            for _ in range(args.updates_per_ep):
                # S  : [B, L, N, O]
                # A  : [B, L, N]
                # R  : [B, L, N]
                # NS : [B, L, N, O]
                # D  : [B, L]
                S, A, R, NS, D = replay.sample(
                    args.batch_size_seq,
                    args.seq_len,
                )

                loss_val = qmix_update_gnn_lstm(
                    device=device,
                    gnnlstm_online_q=online_q,
                    gnnlstm_target_q=target_q,
                    online_mixer=online_mixer,
                    target_mixer=target_mixer,
                    optimizer=optimizer,
                    seq_batch=(S, A, R, NS, D),
                    graph_edge_index=edge_index,
                    gamma=args.gamma,
                    double_dqn=True,
                    burn_in=args.burn_in,
                    grad_clip=args.grad_clip,
                )

                losses.append(loss_val)

                # Polyak update both target networks.
                soft_update(target_q, online_q, args.tau)
                soft_update(target_mixer, online_mixer, args.tau)

            mean_loss = float(np.mean(losses)) if losses else 0.0

            print(
                f"[ep {ep_idx}] "
                f"return={episode_return:.2f} | "
                f"train_loss={mean_loss:.3f} | "
                f"eps={eps:.3f} | "
                f"steps={steps} | "
                f"buffer={len(replay)}"
            )

            if ep_idx % args.eval_every == 0:
                eval_return = evaluate_qmix_gnnlstm_shared(
                    args=args,
                    online_q=online_q,
                    agent_id_list=agent_id_list,
                    edge_index=edge_index,
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
                            "edge_index": edge_index.cpu(),
                            "seq_len": args.seq_len,
                            "burn_in": args.burn_in,
                            "mixing_embed_dim": args.mixing_embed_dim,
                            "hypernet_embed_dim": args.hypernet_embed_dim,
                            "best_eval_return": best_eval_return,
                        },
                        save_path,
                    )

                    print(
                        f"[ep {ep_idx}] saved new best QMIX-GNN-LSTM model to {save_path} | "
                        f"best_eval_return={best_eval_return:.2f}"
                    )

            eps = max(args.eps_end, eps * args.eps_decay)

        else:
            print(
                f"[ep {ep_idx}] "
                f"return={episode_return:.2f} | "
                f"total_steps={total_steps} warming up | "
                f"eps={eps:.3f} | "
                f"buffer={len(replay)}"
            )

    train_env.close()


if __name__ == "__main__":
    run_training(parse_args())
