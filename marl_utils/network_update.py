#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List, Optional

from marl_utils.common import to_tensor
from marl_utils.models import QNetMLP, RecurrentQNet, GNNPolicyQ, GNNLSTMPolicyQ

# =================================== DQN Update ====================================

def dqn_update(device: torch.device, online_q: QNetMLP, target_q: QNetMLP,
               optimizer_q: optim.Optimizer, batch_tuple, gamma: float = 0.99,
               double_dqn: bool = True) -> float:
    states, actions, rewards, next_states, dones = batch_tuple
    state_t = to_tensor(device, states); next_state_t = to_tensor(device, next_states)
    action_t = torch.as_tensor(actions, device=device, dtype=torch.long)
    reward_t = to_tensor(device, rewards); done_t = to_tensor(device, dones)
    q_vals = online_q(state_t); q_taken = q_vals.gather(1, action_t.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = (online_q if double_dqn else target_q)(next_state_t).argmax(dim=1)
        next_q = target_q(next_state_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = reward_t + (1.0 - done_t) * gamma * next_q
    loss = nn.functional.mse_loss(q_taken, target)
    optimizer_q.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(online_q.parameters(), 1.0); optimizer_q.step()
    return float(loss.item())

# =============================== DRQN Update for LSTM ===============================
def drqn_update(
    device: torch.device,
    batch: Dict[str, np.ndarray],
    online_qnet: RecurrentQNet,
    target_qnet: RecurrentQNet,
    optimizer_q: optim.Optimizer,
    gamma: float = 0.99,
    burn_in: int = 8,
    double_dqn: bool = True,
) -> float:
    """
    DRQN/LSTM DQN update on a batch of sequences with:
      - burn-in warmup of hidden states, and
      - hidden-state reset (masking) INSIDE sequences when terminals occur.

    Expects `batch` with keys:
      - 'obs':       [B, T, obs_dim]
      - 'next_obs':  [B, T, obs_dim]
      - 'actions':   [B, T] (int)
      - 'rewards':   [B, T] (float)
      - 'dones':     [B, T] (float; 1.0 means terminal / padding)
    """
    # --- Move batch to device ---
    observations_seq      = to_tensor(device, batch['obs'])            # [B, T, O]
    next_observations_seq = to_tensor(device, batch['next_obs'])       # [B, T, O]
    action_indices        = torch.as_tensor(batch['actions'], device=device, dtype=torch.long)   # [B, T]
    rewards_seq           = to_tensor(device, batch['rewards'])        # [B, T]
    dones_seq             = to_tensor(device, batch['dones'])          # [B, T]

    B, T, _ = observations_seq.shape

    # --- Clamp burn-in so at least one learning step remains ---
    burn = int(burn_in)
    if burn >= T:
        burn = max(0, T - 1)

    # --- Hidden-state warmup (burn-in) ---
    with torch.no_grad():
        h_online = None
        h_target = None
        if burn > 0:
            _q_bi, h_online = online_qnet(observations_seq[:, :burn, :], None)
            _tq_bi, h_target = target_qnet(next_observations_seq[:, :burn, :], None)

    # --- Slice learning window ---
    S_w   = observations_seq[:, burn:, :]        # [B, Tw, O]
    NS_w  = next_observations_seq[:, burn:, :]   # [B, Tw, O]
    A_w   = action_indices[:, burn:]             # [B, Tw]
    R_w   = rewards_seq[:, burn:]                # [B, Tw]
    D_w   = dones_seq[:, burn:]                  # [B, Tw]
    Tw = S_w.size(1)

    q_taken_list: List[torch.Tensor] = []
    target_list:  List[torch.Tensor] = []

    # --- Unroll step-by-step to apply hidden masking on terminals ---
    for t in range(Tw):
        # Current step slices
        s_t   = S_w[:, t:t+1, :]          # [B,1,O]
        ns_t  = NS_w[:, t:t+1, :]         # [B,1,O]
        a_t   = A_w[:, t]                 # [B]
        r_t   = R_w[:, t]                 # [B]
        d_t   = D_w[:, t]                 # [B]  (1.0 = terminal/pad)

        # Online Q(s_t, ·) and pick Q_taken
        q_t, h_online = online_qnet(s_t, h_online)              # q_t: [B,1,A], h_online updated
        q_t_s = q_t.squeeze(1)                                  # [B,A]
        q_taken_t = q_t_s.gather(1, a_t.unsqueeze(1)).squeeze(1)  # [B]
        q_taken_list.append(q_taken_t)

        # ---- Targets (no grad) ----
        with torch.no_grad():
            # Argmax from online on next_obs (do not advance online hidden we keep for learning path)
            qn_online_t, _ = online_qnet(ns_t, h_online)        # [B,1,A], temp hidden discarded
            qn_target_t, h_target = target_qnet(ns_t, h_target) # [B,1,A], advance target hidden

            if double_dqn:
                next_a_t = torch.argmax(qn_online_t.squeeze(1), dim=-1)   # [B]
            else:
                next_a_t = torch.argmax(qn_target_t.squeeze(1), dim=-1)   # [B]

            next_q_t = qn_target_t.squeeze(1).gather(1, next_a_t.unsqueeze(1)).squeeze(1)  # [B]
            target_t = r_t + (1.0 - d_t) * gamma * next_q_t
            target_list.append(target_t)

        # ---- Hidden reset on terminals at this step (mask where done==1) ----
        keep_mask = (1.0 - d_t).view(1, B, 1)   # [1,B,1]
        if h_online is not None:
            h_online = (h_online[0] * keep_mask, h_online[1] * keep_mask)  # each [1,B,H]
        if h_target is not None:
            h_target = (h_target[0] * keep_mask, h_target[1] * keep_mask)

    # Stack over Tw
    Q_taken = torch.stack(q_taken_list, dim=1)   # [B, Tw]
    Target  = torch.stack(target_list,  dim=1)   # [B, Tw]

    # --- Loss & optimization ---
    loss_value = nn.functional.smooth_l1_loss(Q_taken, Target)  # Huber

    optimizer_q.zero_grad()
    loss_value.backward()
    nn.utils.clip_grad_norm_(online_qnet.parameters(), 1.0)
    optimizer_q.step()

    return float(loss_value.item())


# =============================== DQN Update (shared-GNN IDQN) ===============================

def dqn_update_shared_gnn(
    device: torch.device,
    gnn_online_q_network: GNNPolicyQ,
    gnn_target_q_network: GNNPolicyQ,
    optimizer_q: optim.Optimizer,
    batch_tuple,
    graph_edge_index: torch.Tensor,
    graph_edge_features: Optional[torch.Tensor] = None,
    discount_gamma: float = 0.99,
    double_dqn: bool = True,
) -> float:
    """
    Perform a Double-DQN update with a shared GNN Q-network.

    Args:
        batch_tuple:
            Tuple (states, actions, rewards, next_states, dones) with shapes
              states       : [batch_size, num_nodes, obs_dim]
              actions      : [batch_size, num_nodes]        (int64)
              rewards      : [batch_size, num_nodes]        (float32)
              next_states  : [batch_size, num_nodes, obs_dim]
              dones        : [batch_size]                   (float32 scalar per transition)
        graph_edge_index: [2, num_edges] directed edges (same graph for all batch samples)
        graph_edge_features : [num_edges, edge_feat_dim] or None
        discount_gamma: discount factor for TD target
        double_dqn: if True, use online net for argmax and target net for evaluation

    Returns:
        float: scalar MSE loss value.
    """
    states_batch, actions_batch, rewards_batch, next_states_batch, terminal_flags_batch = batch_tuple
    batch_size, num_nodes, obs_dim = states_batch.shape

    edge_index_device = graph_edge_index.to(device)
    edge_features_device = graph_edge_features.to(device) if graph_edge_features is not None else None

    total_mse_loss = torch.zeros((), device=device)

    for batch_idx in range(batch_size):
        # Move a single transition (all nodes) to device
        state_nodes = torch.as_tensor(states_batch[batch_idx], dtype=torch.float32, device=device)       # [N, O]
        next_state_nodes = torch.as_tensor(next_states_batch[batch_idx], dtype=torch.float32, device=device)  # [N, O]
        action_nodes = torch.as_tensor(actions_batch[batch_idx], dtype=torch.long, device=device)        # [N]
        reward_nodes = torch.as_tensor(rewards_batch[batch_idx], dtype=torch.float32, device=device)     # [N]
        done_flag = torch.as_tensor(terminal_flags_batch[batch_idx], dtype=torch.float32, device=device) # scalar

        # Q(s, ·) and Q(s, a)
        q_values_current = gnn_online_q_network(state_nodes, edge_index_device, edge_features_device)     # [N, A]
        q_values_taken = q_values_current.gather(1, action_nodes.view(-1, 1)).squeeze(1)                # [N]

        with torch.no_grad():
            q_values_next_online = gnn_online_q_network(next_state_nodes, edge_index_device, edge_features_device)  # [N, A]
            q_values_next_target = gnn_target_q_network(next_state_nodes, edge_index_device, edge_features_device)  # [N, A]

            if double_dqn:
                next_actions = torch.argmax(q_values_next_online, dim=1)                                 # [N]
            else:
                next_actions = torch.argmax(q_values_next_target, dim=1)                                 # [N]

            next_q_values = q_values_next_target.gather(1, next_actions.view(-1, 1)).squeeze(1)         # [N]
            target_q_values = reward_nodes + (1.0 - done_flag) * discount_gamma * next_q_values         # [N]; done_flag broadcasts

        total_mse_loss = total_mse_loss + nn.functional.mse_loss(q_values_taken, target_q_values)

    total_mse_loss = total_mse_loss / batch_size

    optimizer_q.zero_grad()
    total_mse_loss.backward()
    nn.utils.clip_grad_norm_(gnn_online_q_network.parameters(), 1.0)
    optimizer_q.step()

    return float(total_mse_loss.item())

# =============================== DQN Update (shared-GNN+LSTM IDQN) ===============================

def dqn_update_shared_gnn_lstm(
    device: torch.device,
    online_q: "GNNLSTMPolicyQ",
    target_q: "GNNLSTMPolicyQ",
    optimizer_q: torch.optim.Optimizer,
    batch_tuple,
    edge_index: torch.Tensor,
    gamma: float = 0.99,
    burn_in: int = 8,
    double_dqn: bool = True,
    grad_clip: float = 1.0,
) -> float:
    S, A, R, NS, D = batch_tuple
    S  = torch.as_tensor(S,  device=device, dtype=torch.float32)  # [B,T,N,O]
    NS = torch.as_tensor(NS, device=device, dtype=torch.float32)  # [B,T,N,O]
    A  = torch.as_tensor(A,  device=device, dtype=torch.long)     # [B,T,N]
    R  = torch.as_tensor(R,  device=device, dtype=torch.float32)  # [B,T,N]
    D  = torch.as_tensor(D,  device=device, dtype=torch.float32)  # [B,T] or [B,T,N]

    if D.dim() == 2:  # -> [B,T,N]
        D = D.unsqueeze(-1).expand(-1, -1, S.size(2))

    B, T, N, O = S.shape
    burn = min(max(int(burn_in), 0), max(T - 1, 0))
    Ei = edge_index.to(device)

    # ---- burn-in to warm hidden ----
    with torch.no_grad():
        h_online = h_target = None
        if burn > 0:
            _, h_online = online_q(S[:, :burn], Ei, None)
            _, h_target = target_q(NS[:, :burn], Ei, None)

    # ---- learning window ----
    S_w, NS_w = S[:, burn:], NS[:, burn:]           # [B,Tw,N,O]
    A_w, R_w, D_w = A[:, burn:], R[:, burn:], D[:, burn:]  # [B,Tw,N]
    Tw = S_w.size(1)

    h_o, h_t = h_online, h_target
    q_list, qn_online_list, qn_target_list = [], [], []

    for t in range(Tw):
        # ----- ONLINE: forward with grad to produce Q_taken -----
        # Use full forward with T=1, not .step (which is @torch.no_grad in your class)
        q_t, h_o = online_q(S_w[:, t].unsqueeze(1), Ei, h_o)  # q_t: [B,1,N,A]
        q_list.append(q_t)  # keep [B,1,N,A]

        # ----- Argmax and Target: no grad -----
        with torch.no_grad():
            # Online for argmax (do not backprop through this path)
            qn_o_t, _ = online_q(NS_w[:, t].unsqueeze(1), Ei, h_o)    # [B,1,N,A]
            # Target for evaluation
            qn_t_t, h_t = target_q(NS_w[:, t].unsqueeze(1), Ei, h_t)  # [B,1,N,A]

        qn_online_list.append(qn_o_t)
        qn_target_list.append(qn_t_t)

        # ----- hidden reset on dones at this step -----
        if h_o is not None:
            done_mask = (1.0 - D_w[:, t]).view(1, B, N, 1)        # [1,B,N,1]
            h0 = h_o[0].view(1, B, N, -1); c0 = h_o[1].view(1, B, N, -1)
            h0.mul_(done_mask); c0.mul_(done_mask)
            h_o = (h0.view(1, B * N, -1), c0.view(1, B * N, -1))
        if h_t is not None:
            done_mask = (1.0 - D_w[:, t]).view(1, B, N, 1)
            h1 = h_t[0].view(1, B, N, -1); c1 = h_t[1].view(1, B, N, -1)
            h1.mul_(done_mask); c1.mul_(done_mask)
            h_t = (h1.view(1, B * N, -1), c1.view(1, B * N, -1))

    Q_seq = torch.cat(q_list, dim=1)                 # [B,Tw,1,N,A] → actually [B,Tw,N,A]
    Q_seq = Q_seq.squeeze(2)                         # remove the size-1 dim if present -> [B,Tw,N,A]
    Qn_o  = torch.cat(qn_online_list, dim=1).squeeze(2)  # [B,Tw,N,A]
    Qn_t  = torch.cat(qn_target_list, dim=1).squeeze(2)  # [B,Tw,N,A]

    # Gather Q(s,a)
    Q_taken = Q_seq.gather(-1, A_w.unsqueeze(-1)).squeeze(-1)  # [B,Tw,N]

    # Targets (no grad)
    with torch.no_grad():
        next_idx = torch.argmax(Qn_o, dim=-1) if double_dqn else torch.argmax(Qn_t, dim=-1)  # [B,Tw,N]
        Qn_star = Qn_t.gather(-1, next_idx.unsqueeze(-1)).squeeze(-1)                         # [B,Tw,N]
        target = R_w + (1.0 - D_w) * gamma * Qn_star

    loss = nn.functional.smooth_l1_loss(Q_taken, target)

    optimizer_q.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_q.parameters(), grad_clip)
    optimizer_q.step()

    return float(loss.item())



# ============================= CTDE-specific: VDN update (Double-DQN) =============================
def vdn_update_mlp(
    device: torch.device,
    online_q: QNetMLP,          # shared per-agent Q_i(o_i, a_i)
    target_q: QNetMLP,          # target copy
    optimizer: optim.Optimizer,
    batch_tuple,
    gamma: float = 0.99,
    double_dqn: bool = True,
    grad_clip: float = 1.0,
) -> float:
    """
    VDN: Q_tot = sum_i Q_i. Double-DQN for targets, with **team reward**.
    Inputs (from JointReplayBuffer.sample):
      states      : [B, N, O]
      actions     : [B, N]         (int64)
      rewards     : [B, N]         (float32)  -- we sum over N to get team reward
      next_states : [B, N, O]
      dones       : [B]            (float32)
    """
    S, A, R, NS, D = batch_tuple
    B, N, O = S.shape

    S_t  = torch.as_tensor(S,  device=device, dtype=torch.float32).view(B * N, O)  # [B*N,O]
    NS_t = torch.as_tensor(NS, device=device, dtype=torch.float32).view(B * N, O)  # [B*N,O]
    A_t  = torch.as_tensor(A,  device=device, dtype=torch.long).view(B * N)        # [B*N]
    R_t  = torch.as_tensor(R,  device=device, dtype=torch.float32)                 # [B,N]
    D_t  = torch.as_tensor(D,  device=device, dtype=torch.float32)                 # [B]

    # Per-agent Q(s,·) and gather chosen actions
    q_all = online_q(S_t)                                      # [B*N, A]
    q_taken = q_all.gather(1, A_t.view(-1, 1)).squeeze(1)      # [B*N]
    q_taken = q_taken.view(B, N)                                # [B,N]
    q_tot = q_taken.sum(dim=1)                                  # [B]  (VDN sum)

    with torch.no_grad():
        # Next actions via online (Double-DQN), per agent
        q_next_online = online_q(NS_t).view(B, N, -1)           # [B,N,A]
        next_actions = q_next_online.argmax(dim=-1)             # [B,N]

        # Evaluate those actions on target net
        q_next_target = target_q(NS_t).view(B, N, -1)           # [B,N,A]
        q_next_a = q_next_target.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)  # [B,N]
        q_next_tot = q_next_a.sum(dim=1)                        # [B]

        # Team reward (sum of per-agent rewards)
        R_team = R_t.sum(dim=1)                                 # [B]
        target = R_team + (1.0 - D_t) * gamma * q_next_tot      # [B]

    loss = nn.functional.smooth_l1_loss(q_tot, target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_q.parameters(), grad_clip)
    optimizer.step()

    return float(loss.item())

# ================= CTDE-specific: VDN update for LSTM (Double-DQN) ===================

def vdn_update_lstm(
    device: torch.device,
    online_q: "RecurrentQNet",    # forward([B,T,O], hidden) -> ([B,T,A], new_hidden)
    target_q: "RecurrentQNet",
    optimizer: torch.optim.Optimizer,
    batch_tuple,                   # (S, A, R, NS, D, M) from JointSequenceReplay
    gamma: float = 0.99,
    burn_in: int = 6,
    double_dqn: bool = True,
    grad_clip: float = 1.0,
) -> float:
    S, A, R, NS, D, M = batch_tuple
    S  = torch.as_tensor(S,  device=device, dtype=torch.float32)  # [B,L,N,O]
    A  = torch.as_tensor(A,  device=device, dtype=torch.long)     # [B,L,N]
    R  = torch.as_tensor(R,  device=device, dtype=torch.float32)  # [B,L,N]
    NS = torch.as_tensor(NS, device=device, dtype=torch.float32)  # [B,L,N,O]
    D  = torch.as_tensor(D,  device=device, dtype=torch.float32)  # [B,L] (team)
    M  = torch.as_tensor(M,  device=device, dtype=torch.float32)  # [B,L] (mask)

    B, L, N, O = S.shape
    burn = int(burn_in)
    if burn >= L:
        burn = max(0, L - 1)

    # warm hidden with burn-in (vectorized across agents: B*N sequences)
    h_o = h_t = None
    if burn > 0:
        with torch.no_grad():
            _, h_o = online_q(S[:, :burn].reshape(B * N, burn, O), None)
            _, h_t = target_q(NS[:, :burn].reshape(B * N, burn, O), None)

    Tw = L - burn
    Q_team_list, Target_team_list = [], []

    for t in range(Tw):
        tt = burn + t

        s_t  = S[:, tt].reshape(B * N, 1, O)
        ns_t = NS[:, tt].reshape(B * N, 1, O)
        a_t  = A[:, tt].reshape(B * N)
        r_t  = R[:, tt]                 # [B,N]
        d_ts = D[:, tt]                 # [B] team done

        # online forward (with grad)
        q_now, h_o = online_q(s_t, h_o)            # [B*N,1,A]
        q_now = q_now.squeeze(1)                   # [B*N,A]
        q_taken_nodes = q_now.gather(1, a_t.unsqueeze(1)).squeeze(1)  # [B*N]
        q_taken_team  = q_taken_nodes.view(B, N).sum(dim=1)           # [B]
        Q_team_list.append(q_taken_team)

        with torch.no_grad():
            qn_o, _ = online_q(ns_t, h_o)         # argmax path (disposable)
            qn_t, h_t = target_q(ns_t, h_t)       # target path (stateful)
            qn_o = qn_o.squeeze(1); qn_t = qn_t.squeeze(1)           # [B*N,A]

            next_a = torch.argmax(qn_o if double_dqn else qn_t, dim=-1)     # [B*N]
            next_q_team = qn_t.gather(1, next_a.unsqueeze(1)).squeeze(1)    # [B*N]
            next_q_team = next_q_team.view(B, N).sum(dim=1)                  # [B]

            r_team = r_t.sum(dim=1)                                         # [B]
            target_team = r_team + (1.0 - d_ts) * gamma * next_q_team       # [B]
            Target_team_list.append(target_team)

        # reset hidden where team done
        if h_o is not None:
            keep = (1.0 - d_ts).view(1, B, 1, 1)
            h0 = h_o[0].view(1, B, N, -1); c0 = h_o[1].view(1, B, N, -1)
            h0.mul_(keep); c0.mul_(keep)
            h_o = (h0.view(1, B * N, -1), c0.view(1, B * N, -1))
        if h_t is not None:
            keep = (1.0 - d_ts).view(1, B, 1, 1)
            h1 = h_t[0].view(1, B, N, -1); c1 = h_t[1].view(1, B, N, -1)
            h1.mul_(keep); c1.mul_(keep)
            h_t = (h1.view(1, B * N, -1), c1.view(1, B * N, -1))

    Q_team  = torch.stack(Q_team_list,      dim=1)   # [B, Tw]
    TargetT = torch.stack(Target_team_list, dim=1)   # [B, Tw]
    M_w = M[:, burn:]                                  # [B, Tw]

    td = nn.functional.smooth_l1_loss(Q_team, TargetT, reduction='none') * M_w
    denom = M_w.sum().clamp_min(1.0)
    loss = td.sum() / denom

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_q.parameters(), grad_clip)
    optimizer.step()

    return float(loss.item())


# ================================ Soft Update ==================================

@torch.no_grad()
def soft_update(target: torch.nn.Module, online: torch.nn.Module, tau: float = 0.01):
    """Polyak averaging: target <- tau * online + (1 - tau) * target"""
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * op.data)
