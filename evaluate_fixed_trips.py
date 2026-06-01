#!/usr/bin/env python3
from __future__ import annotations
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------- Your modules ----------
# Fixed-routes evaluation environment
from sumo_marl_fixed_routes_env import SumoGridMARLFixedEnv
# Models
from marl_utils.models import (
    QNetMLP,           # IDQN / CTDE-VDN (MLP)
    RecurrentQNet,    # IDRQN / CTDE-VDN (LSTM)
    GNNPolicyQ,       # DQN / CTDE-VDN (GNN)
    GNNLSTMPolicyQ,   # DRQN / CTDE-VDN (GNN+LSTM)
)
# Grid topology helper for GNN variants
from marl_utils.common import build_grid_edge_index, set_global_seed


# ----------------------------- Model builder -----------------------------
def build_model(method: str,
                obs_dim: int,
                act_dim: int,
                hidden: int = 128,
                gnn_layers: int = 2) -> nn.Module:
    """
    Map method -> model class.
    For evaluation, CTDE-trained models use the same per-agent inference nets.
    """
    method = method.lower()
    if method in ("idqn_mlp", "dqn_mlp", "ctde_vdn_mlp", "ctde_mlp", "dqn"):
        return QNetMLP(observation_dim=obs_dim, action_dim=act_dim, hidden_units=hidden)
    elif method in ("idrqn_lstm", "drqn_lstm", "ctde_vdn_lstm", "ctde_lstm", "idrqn"):
        return RecurrentQNet(observation_dim=obs_dim, action_dim=act_dim, hidden_units=hidden)
    elif method in ("dqn_gnn", "ctde_vdn_gnn", "ctde_gnn"):
        return GNNPolicyQ(node_dim=obs_dim, actions=act_dim, hidden=hidden, layers=gnn_layers)
    elif method in ("drqn_gnn_lstm", "ctde_vdn_gnn_lstm", "ctde_gnn_lstm", "drqn"):
        return GNNLSTMPolicyQ(node_dim=obs_dim, actions=act_dim, hidden=hidden, gnn_layers=gnn_layers)
    else:
        raise ValueError(f"Unknown method: {method}")


# ------------------------- Action selection wrappers -------------------------
@torch.no_grad()
def select_actions_stateless_mlp(model: QNetMLP,
                                 obs_mat: np.ndarray,
                                 device: torch.device) -> List[int]:
    """
    obs_mat: [N,O]
    returns list[int] actions for N agents
    """
    X = torch.as_tensor(obs_mat, dtype=torch.float32, device=device)   # [N,O]
    q = model(X)                                                       # [N,A]
    return torch.argmax(q, dim=-1).tolist()

@torch.no_grad()
def select_actions_recurrent_lstm(model: RecurrentQNet,
                                  obs_mat: np.ndarray,
                                  device: torch.device,
                                  rnn_state: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[List[int], Tuple[torch.Tensor, torch.Tensor]]:
    """
    obs_mat: [N,O], run as B=N, T=1
    """
    obs_t = torch.as_tensor(obs_mat, dtype=torch.float32, device=device).unsqueeze(1)  # [N,1,O]
    q_seq, rnn_state = model(obs_t, rnn_state)                                         # [N,1,A]
    q = q_seq[:, -1, :]                                                                # [N,A]
    greedy = torch.argmax(q, dim=-1).tolist()
    return greedy, rnn_state

@torch.no_grad()
def select_actions_gnn(model: GNNPolicyQ,
                       obs_mat: np.ndarray,
                       edge_index: torch.Tensor,
                       device: torch.device) -> List[int]:
    X = torch.as_tensor(obs_mat, dtype=torch.float32, device=device)  # [N,O]
    q = model(X, edge_index.to(device))                               # [N,A]
    return torch.argmax(q, dim=-1).tolist()

@torch.no_grad()
def select_actions_gnn_lstm(model: GNNLSTMPolicyQ,
                            obs_mat: np.ndarray,
                            edge_index: torch.Tensor,
                            device: torch.device,
                            rnn_state: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[List[int], Tuple[torch.Tensor, torch.Tensor]]:
    X = torch.as_tensor(obs_mat, dtype=torch.float32, device=device).unsqueeze(0)  # [1,N,O]
    q_t, rnn_state = model.step(X, edge_index.to(device), rnn_state)               # q_t: [1,N,A]
    greedy = torch.argmax(q_t[0], dim=-1).tolist()                                 # [N]
    return greedy, rnn_state


# ------------------------------- Evaluation -------------------------------
def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_global_seed(args.seed)

    # ---- Build env & attach fixed input ----
    env = SumoGridMARLFixedEnv(
        grid_n=args.grid_n,
        episode_steps=args.steps,
        sumo_steps_per_env_step=args.sumo_steps_per_env_step,
        gui=args.gui,
        gui_delay_ms=args.gui_delay_ms,
        seed=args.seed,
        verbose=args.verbose,
        suppress_sumo_output=not args.verbose,
        fixed_routes_file=args.routes if args.routes else None,
        fixed_trips_file=args.trips if args.trips else None,
        duarouter_seed=args.duarouter_seed,
    )

    # ---- Reset env, get shapes ----
    obs_dict = env.reset()
    agent_ids = list(env.agent_ids)
    N = len(agent_ids)
    if N == 0:
        raise RuntimeError("No traffic lights discovered in SUMO. Check your network/trips.")
    O = len(next(iter(obs_dict.values())))
    A_dim = env.action_spaces[agent_ids[0]].n

    # Optional: build grid edge index for GNNs (same ordering as agent_ids)
    edge_index = build_grid_edge_index(agent_ids)  # torch.long [2,E]

    # ---- Build & load model ----
    model = build_model(args.method, obs_dim=O, act_dim=A_dim, hidden=args.hidden, gnn_layers=args.gnn_layers).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    # ---- Rollout ----
    episode_return = 0.0
    final_kpis: Optional[Dict] = None

    rnn_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    is_mlp      = isinstance(model, QNetMLP)
    is_rnn      = isinstance(model, RecurrentQNet)
    is_gnn      = isinstance(model, GNNPolicyQ)
    is_gnn_lstm = isinstance(model, GNNLSTMPolicyQ)

    for t in range(env.episode_steps):
        obs_mat = np.stack([obs_dict[aid] for aid in agent_ids], axis=0).astype(np.float32)  # [N,O]

        if is_mlp:
            greedy = select_actions_stateless_mlp(model, obs_mat, device)
        elif is_rnn:
            greedy, rnn_state = select_actions_recurrent_lstm(model, obs_mat, device, rnn_state)
        elif is_gnn:
            greedy = select_actions_gnn(model, obs_mat, edge_index, device)
        elif is_gnn_lstm:
            greedy, rnn_state = select_actions_gnn_lstm(model, obs_mat, edge_index, device, rnn_state)
        else:
            raise RuntimeError("Unsupported model type at inference.")

        action_dict = {aid: int(greedy[i]) for i, aid in enumerate(agent_ids)}
        next_obs, reward_dict, done, info = env.step(action_dict)

        episode_return += float(np.sum(list(reward_dict.values())))
        obs_dict = next_obs
        final_kpis = info.get("network_kpis", None)

        if done:
            break

    env.close()

    # ---- Summarize KPIs ----
    if final_kpis is None:
        final_kpis = {
            "completed_vehicles": 0.0,
            "throughput_veh_per_hour": 0.0,
            "mean_travel_time_s": 0.0,
            "mean_waiting_time_s": 0.0,
        }

    completed = float(final_kpis.get("completed_vehicles", 0.0))
    throughput_vph = float(final_kpis.get("throughput_veh_per_hour", 0.0))
    mean_travel_s = float(final_kpis.get("mean_travel_time_s", 0.0))
    mean_wait_s = float(final_kpis.get("mean_waiting_time_s", 0.0))

    print("\n=== Evaluation (single case) ===")
    print(f"Method           : {args.method}")
    print(f"Grid             : {args.grid_n}×{args.grid_n}")
    print(f"Episode steps    : {env.episode_steps} (SUMO steps/step = {args.sumo_steps_per_env_step})")
    if args.routes:
        print(f"Routes file      : {args.routes}")
    if args.trips:
        print(f"Trips file       : {args.trips}  (duarouter_seed={args.duarouter_seed})")
    print(f"Completed veh    : {completed:.2f}")
    print(f"Throughput (vph) : {throughput_vph:.2f}")
    print(f"Mean travel (s)  : {mean_travel_s:.2f}")
    print(f"Avg waiting (s)  : {mean_wait_s:.2f}")
    print(f"Episode return   : {episode_return:.2f}")


# ---------------------------------- CLI ----------------------------------
def parse_args():
    p = argparse.ArgumentParser("Evaluate a trained MARL TSC model on a single fixed trip/routes file.")
    # Fixed input
    p.add_argument("--routes", type=str, default=None, help="Path to fixed routes file (.rou.xml)")
    p.add_argument("--trips",  type=str, default=None, help="Path to fixed trips/flows file (.xml) to duaroute")
    p.add_argument("--duarouter-seed", type=int, default=2025, help="Seed for duarouter when using --trips")
    # Env
    p.add_argument("--grid-n", type=int, default=3, help="Grid size N (NxN core)")
    p.add_argument("--steps", type=int, default=200, help="Episode steps (env steps)")
    p.add_argument("--sumo-steps-per-env-step", type=int, default=5, help="SUMO internal steps per env step")
    p.add_argument("--gui", action="store_true", help="Use SUMO-GUI")
    p.add_argument("--gui-delay-ms", type=int, default=0, help="Delay per SUMO step in GUI")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--verbose", action="store_true")
    # Model
    p.add_argument("--method", type=str, required=True,
                   choices=[
                       # Independent
                       "idqn_mlp", "idrqn_lstm", "dqn_gnn", "drqn_gnn_lstm",
                       # CTDE (VDN) variants — same inference nets
                       "ctde_vdn_mlp", "ctde_vdn_lstm", "ctde_vdn_gnn", "ctde_vdn_gnn_lstm",
                       # Common short aliases
                       "dqn", "idrqn", "drqn", "ctde_mlp", "ctde_lstm", "ctde_gnn", "ctde_gnn_lstm"
                   ],
                   help="Learning method / architecture to load.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model .pt")
    p.add_argument("--hidden", type=int, default=128, help="Hidden width for the model")
    p.add_argument("--gnn-layers", type=int, default=2, help="Number of GNN layers (for GNN variants)")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Safety: need exactly one of routes or trips
    if not args.routes and not args.trips:
        raise SystemExit("Please pass exactly one of --routes or --trips.")
    if args.routes and args.trips:
        raise SystemExit("Please pass either --routes or --trips, not both.")
    run_evaluation(args)
