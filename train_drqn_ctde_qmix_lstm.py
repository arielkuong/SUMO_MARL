from __future__ import annotations
import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime

# Your modules
from marl_utils.models import RecurrentQNet, QMIXMixer
from marl_utils.replay_buffers import JointSequenceReplay
from marl_utils.network_update import qmix_update_lstm, soft_update
from marl_utils.common import parse_args, set_global_seed, EvalHistory, clear_eval_history
from env_builder import build_train_env, get_or_create_eval_pool

# --------- evaluation (decentralized greedy with 1-step LSTM unroll) ----------
@torch.no_grad()
def evaluate_qmix_lstm_shared(
    args,
    shared_q: RecurrentQNet,
    agent_ids: List[str],
    run_name: str,
) -> float:
    """
    Evaluation remains decentralised.

    The QMIX mixer is not used at execution time.

    Each agent selects:

        a_i = argmax_a Q_i(o_i, h_i, a)
    """

    original_mode = shared_q.training
    shared_q.eval()

    device = next(shared_q.parameters()).device

    eval_pool = get_or_create_eval_pool(args)
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

        for env in eval_pool.envs:
            obs_dict = env.reset()
            done = False
            ep_ret = 0.0
            last_info: Dict = {}

            N = len(agent_ids)

            # Hidden state is maintained independently for each agent.
            # Batch dimension is N, sequence length is 1 at each environment tick.
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

            while not done:
                obs_mat = np.stack(
                    [obs_dict[aid] for aid in agent_ids],
                    axis=0,
                ).astype(np.float32)                                  # [N, O]

                obs_t = torch.as_tensor(
                    obs_mat,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(1)                                         # [N, 1, O]

                q_seq, hidden = shared_q(obs_t, hidden)                 # [N, 1, A]
                q = q_seq[:, -1, :]                                     # [N, A]

                greedy = torch.argmax(q, dim=-1).tolist()               # [N]

                action_dict = {
                    aid: int(greedy[i])
                    for i, aid in enumerate(agent_ids)
                }

                next_obs, reward_dict, done, info = env.step(action_dict)

                # Team return = sum of per-agent rewards.
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
            f"[{datetime.now()}, EVAL {run_name}] "
            f"MEAN over {len(eval_pool.envs)} cases | "
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
        shared_q.train(original_mode)

# ------------------------------------------------------------
def run_training(args):
    device = torch.device(args.device)

    set_global_seed(args.seed)

    best_eval_return = -np.inf

    # ------------------------------------------------------------------
    # Environment and shapes
    # ------------------------------------------------------------------

    env = build_train_env(args)
    obs_dict = env.reset()

    agent_ids = list(env.agent_ids)

    N = len(agent_ids)
    O = len(next(iter(obs_dict.values())))
    A_dim = env.action_spaces[agent_ids[0]].n

    state_dim = N * O

    print(
        f"QMIX-DRQN training setup | "
        f"N={N} agents | "
        f"O={O} obs dim | "
        f"A={A_dim} actions | "
        f"state_dim={state_dim} | "
        f"seq_len={args.seq_len} | "
        f"burn_in={args.burn_in} | "
        f"device={device}"
    )

    # ------------------------------------------------------------------
    # Shared recurrent per-agent Q_i network
    # ------------------------------------------------------------------

    online_q = RecurrentQNet(
        O,
        A_dim,
        hidden_units=args.hidden,
    ).to(device)

    target_q = RecurrentQNet(
        O,
        A_dim,
        hidden_units=args.hidden,
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

    # Optimise both recurrent Q-network and QMIX mixer.
    optimizer = optim.Adam(
        list(online_q.parameters()) + list(online_mixer.parameters()),
        lr=args.lr,
    )

    # ------------------------------------------------------------------
    # Joint sequence replay
    # ------------------------------------------------------------------

    replay = JointSequenceReplay(
        capacity_steps=args.replay_size,
        num_agents=N,
        obs_dim=O,
        seed=args.rb_seed,
    )

    eps = args.eps_start
    total_steps = 0

    run_name = "qmix_ctde_lstm_shared_seqlen" + str(args.seq_len)

    log_root = args.logdir + "_grid_" + str(args.grid_n)
    save_dir = f"{log_root}/seed{args.seed}"
    os.makedirs(save_dir, exist_ok=True)

    clear_eval_history(log_root, run_name, args.seed)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    for ep_idx in range(1, args.episodes + 1):
        obs_dict = env.reset()

        done = False
        steps = 0
        episode_return = 0.0

        # Episode buffers.
        S_list = []
        A_list = []
        R_list = []
        NS_list = []
        D_list = []

        # Hidden state across time for action selection.
        # Batch dimension is N agents.
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        while not done and steps < args.episode_steps:
            obs_mat = np.stack(
                [obs_dict[aid] for aid in agent_ids],
                axis=0,
            ).astype(np.float32)                                      # [N, O]

            obs_t = torch.as_tensor(
                obs_mat,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1)                                             # [N, 1, O]

            with torch.no_grad():
                q_seq, hidden = online_q(obs_t, hidden)                # [N, 1, A]
                q = q_seq[:, -1, :]                                    # [N, A]
                greedy = torch.argmax(q, dim=-1).cpu().numpy()         # [N]

            # Epsilon-greedy per agent.
            actions = greedy.copy()

            for i in range(N):
                if np.random.rand() < eps:
                    actions[i] = np.random.randint(A_dim)

            action_dict = {
                aid: int(actions[i])
                for i, aid in enumerate(agent_ids)
            }

            next_obs, reward_dict, done, _ = env.step(action_dict)

            next_mat = np.stack(
                [next_obs[aid] for aid in agent_ids],
                axis=0,
            ).astype(np.float32)                                      # [N, O]

            rewards = np.array(
                [float(reward_dict[aid]) for aid in agent_ids],
                dtype=np.float32,
            )                                                        # [N]

            # Team return for logging.
            episode_return += float(np.sum(rewards))

            # Record one joint step.
            S_list.append(obs_mat)
            A_list.append(actions.astype(np.int64))
            R_list.append(rewards)
            NS_list.append(next_mat)
            D_list.append(float(done))

            obs_dict = next_obs
            total_steps += 1
            steps += 1

        # Push one full joint episode.
        replay.push_episode(
            states=np.asarray(S_list, dtype=np.float32),          # [T, N, O]
            actions=np.asarray(A_list, dtype=np.int64),           # [T, N]
            rewards=np.asarray(R_list, dtype=np.float32),         # [T, N]
            next_states=np.asarray(NS_list, dtype=np.float32),    # [T, N, O]
            dones=np.asarray(D_list, dtype=np.float32),           # [T]
        )

        # ------------------------------------------------------------------
        # Centralised training: QMIX + Double-DQN over recurrent Q_i
        # ------------------------------------------------------------------

        mean_loss = 0.0

        if total_steps >= args.warmup_steps and len(replay) > 0:
            losses = []

            for _ in range(args.updates_per_ep):
                S, A, R, NS, D, M = replay.sample(
                    args.batch_size_seq,
                    args.seq_len,
                )

                loss = qmix_update_lstm(
                    device=device,
                    online_q=online_q,
                    target_q=target_q,
                    online_mixer=online_mixer,
                    target_mixer=target_mixer,
                    optimizer=optimizer,
                    batch_tuple=(S, A, R, NS, D, M),
                    gamma=args.gamma,
                    burn_in=args.burn_in,
                    double_dqn=True,
                    grad_clip=args.grad_clip,
                )

                losses.append(loss)

                # Polyak update both recurrent Q-network and QMIX mixer.
                soft_update(target_q, online_q, args.tau)
                soft_update(target_mixer, online_mixer, args.tau)

            mean_loss = float(np.mean(losses))

            print(
                f"[ep {ep_idx}] "
                f"return={episode_return:.2f} | "
                f"train_loss={mean_loss:.3f} | "
                f"eps={eps:.3f} | "
                f"steps={steps} | "
                f"buffer={len(replay)}"
            )

            if ep_idx % args.eval_every == 0:
                eval_return = evaluate_qmix_lstm_shared(
                    args=args,
                    shared_q=online_q,
                    agent_ids=agent_ids,
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
                            "agent_ids": agent_ids,
                            "num_agents": N,
                            "obs_dim": O,
                            "action_dim": A_dim,
                            "state_dim": state_dim,
                            "seq_len": args.seq_len,
                            "burn_in": args.burn_in,
                            "mixing_embed_dim": args.mixing_embed_dim,
                            "hypernet_embed_dim": args.hypernet_embed_dim,
                            "best_eval_return": best_eval_return,
                        },
                        save_path,
                    )

                    print(
                        f"[ep {ep_idx}] saved new best QMIX-DRQN model to {save_path} | "
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

    env.close()


if __name__ == "__main__":
    run_training(parse_args())
