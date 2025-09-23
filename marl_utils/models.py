#!/usr/bin/env python3
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple, Optional

# =============================== MLP Models ===============================
class QNetMLP(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_units: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, action_dim),
        )
    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.network(state_tensor)

class RecurrentQNet(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_units: int = 128):
        super().__init__()
        # Encoder to hidden size H
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_units),
            nn.ReLU(),
        )
        # NEW: LayerNorm over the last dim (H) before the LSTM
        self.pre_lstm_norm = nn.LayerNorm(hidden_units)

        self.lstm = nn.LSTM(hidden_units, hidden_units, batch_first=True)

        # OPTIONAL: LayerNorm after LSTM (helps if LSTM outputs drift)
        self.post_lstm_norm = nn.LayerNorm(hidden_units)

        self.q_head = nn.Linear(hidden_units, action_dim)

    def forward(
        self,
        observation_seq: torch.Tensor,  # [B, T, O]
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        # Encode per step to H
        z = self.encoder(observation_seq)           # [B, T, H]
        z = self.pre_lstm_norm(z)                   # LN over H

        # Recurrent pass
        z, hidden_state = self.lstm(z, hidden_state)  # [B, T, H]

        # (Optional) stabilize head input
        z = self.post_lstm_norm(z)                  # LN over H

        # Q-values per step
        q_values = self.q_head(z)                   # [B, T, A]
        return q_values, hidden_state


# class ActorMLP(nn.Module):
#     def __init__(self, observation_dim: int, action_dim: int, hidden_units: int = 128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(observation_dim, hidden_units), nn.ReLU(),
#             nn.Linear(hidden_units, hidden_units), nn.ReLU(),
#             nn.Linear(hidden_units, action_dim)
#         )
#     def forward(self, state_tensor: torch.Tensor):
#         return self.net(state_tensor)  # logits

# =============================== LSTM Models ===============================
# class ActorLSTM(nn.Module):
#     def __init__(self, observation_dim: int, action_dim: int, hidden_units: int = 128):
#         super().__init__()
#         self.fc = nn.Linear(observation_dim, hidden_units)
#         self.lstm = nn.LSTM(hidden_units, hidden_units, batch_first=True)
#         self.logits = nn.Linear(hidden_units, action_dim)
#         self.activation = nn.ReLU()
#     def forward(self, obs_seq: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
#         z = self.activation(self.fc(obs_seq))
#         z, hidden_state = self.lstm(z, hidden_state)
#         return self.logits(z), hidden_state  # [B,T,A]
#
# class CriticMLP(nn.Module):
#     def __init__(self, observation_dim: int, hidden_units: int = 128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(observation_dim, hidden_units), nn.ReLU(),
#             nn.Linear(hidden_units, hidden_units), nn.ReLU(),
#             nn.Linear(hidden_units, 1)
#         )
#     def forward(self, state_tensor: torch.Tensor):
#         return self.net(state_tensor).squeeze(-1)
#
# class CriticLSTM(nn.Module):
#     def __init__(self, observation_dim: int, hidden_units: int = 128):
#         super().__init__()
#         self.fc = nn.Linear(observation_dim, hidden_units)
#         self.lstm = nn.LSTM(hidden_units, hidden_units, batch_first=True)
#         self.value_head = nn.Linear(hidden_units, 1)
#         self.activation = nn.ReLU()
#     def forward(self, obs_seq: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
#         z = self.activation(self.fc(obs_seq))
#         z, hidden_state = self.lstm(z, hidden_state)
#         v = self.value_head(z).squeeze(-1)
#         return v, hidden_state

# ====================================== GNN models =========================================
class SimpleAttnMP(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int = 0, hidden: int = 128):
        super().__init__()
        self.phi = nn.Linear(node_dim, hidden)
        self.psi = nn.Linear(edge_dim, hidden) if edge_dim > 0 else None
        self.attn = nn.Linear(2 * hidden, 1)
        self.upd = nn.GRUCell(hidden, hidden)
    def forward(self, node_x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None):
        N = node_x.size(0)
        src, dst = edge_index[0], edge_index[1]
        h = torch.relu(self.phi(node_x))
        m_src = h[src]; m_dst = h[dst]
        m_e = torch.relu(self.psi(edge_attr)) if (self.psi is not None and edge_attr is not None) else torch.zeros_like(m_src)
        scores = self.attn(torch.cat([m_src + m_e, m_dst], dim=-1)).squeeze(-1)
        exp_scores = torch.exp(scores - scores.max())
        denom = torch.zeros(N, device=node_x.device); denom.index_add_(0, dst, exp_scores)
        alpha = exp_scores / (denom[dst] + 1e-9)
        aggregated = torch.zeros_like(h); aggregated.index_add_(0, dst, alpha.unsqueeze(-1) * (m_src + m_e))
        return self.upd(aggregated, h)

class GNNPolicyQ(nn.Module):
    def __init__(self, node_dim: int, actions: int = 4, edge_dim: int = 0, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleAttnMP(node_dim if i == 0 else hidden, edge_dim=edge_dim, hidden=hidden)
            for i in range(layers)
        ])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, actions))
    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None):
        z = node_feats
        for mp in self.layers:
            z = mp(z, edge_index, edge_attr)
        return self.head(z)


class GNNLSTMPolicyQ(nn.Module):
    """
    Spatio-temporal Q-network:
      - Per-timestep GNN message passing (vectorized over batch) → node embeddings z_t ∈ R^{B×N×H}
      - LayerNorm over embeddings (stabilizes scale/drift across nodes & time)
      - Per-node LSTM over time on z_{1:T} (nodes flattened into batch: B*N sequences)
      - Q-head maps LSTM outputs to per-node Q-values

    Inputs:
      X_seq    : [B, T, N, O]
      edge_index : [2, E]   (same topology for all items in the batch)
    Outputs:
      Q_seq    : [B, T, N, A]
      new_hidden: (h, c) with shape [1, B*N, H]
    """
    def __init__(self, node_dim: int, actions: int, hidden: int = 128, gnn_layers: int = 2):
        super().__init__()
        self.hidden = hidden
        self.actions = actions

        self.gnn_layers = nn.ModuleList([
            SimpleAttnMP(node_dim if i == 0 else hidden, edge_dim=0, hidden=hidden)
            for i in range(gnn_layers)
        ])

        # NEW: LayerNorm before the LSTM (applied over last dim H)
        self.pre_lstm_norm = nn.LayerNorm(hidden)

        self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, actions)
        )

    def _build_batched_edges(self, edge_index: torch.Tensor, B: int, N: int) -> torch.Tensor:
        Ei = edge_index
        device = Ei.device
        offsets = torch.arange(B, device=device, dtype=Ei.dtype) * N  # [B]
        src = Ei[0].unsqueeze(0) + offsets.unsqueeze(1)               # [B,E]
        dst = Ei[1].unsqueeze(0) + offsets.unsqueeze(1)               # [B,E]
        return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0) # [2,B*E]

    def forward(
        self,
        X_seq: torch.Tensor,                 # [B, T, N, O]
        edge_index: torch.Tensor,            # [2, E]
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        B, T, N, O = X_seq.shape
        device = X_seq.device
        Ei = edge_index.to(device)

        # ----- GNN per timestep (vectorized over batch) -----
        Ei_batched = self._build_batched_edges(Ei, B, N)              # [2,B*E]
        z_time = []
        for t in range(T):
            x_t = X_seq[:, t, :, :]                                   # [B,N,O]
            x_flat = x_t.reshape(B * N, O).contiguous()               # [B*N,O]
            z = x_flat
            for mp in self.gnn_layers:
                z = mp(z, Ei_batched, edge_attr=None)                 # [B*N,H]
            z_time.append(z.reshape(B, N, self.hidden).unsqueeze(1))  # [B,1,N,H]
        z_seq = torch.cat(z_time, dim=1)                              # [B,T,N,H]

        # ----- NEW: LayerNorm over H before LSTM -----
        # LayerNorm expects the last dimension (H); works on [B,T,N,H] directly
        z_seq = self.pre_lstm_norm(z_seq)

        # ----- LSTM over time, per node -----
        z_seq_bn = z_seq.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, self.hidden)  # [B*N,T,H]
        if hidden is not None:
            h0, c0 = hidden
            if h0.size(1) != B * N or c0.size(1) != B * N:
                raise RuntimeError(f"LSTM hidden batch mismatch: got {h0.size(1)}, expected {B*N}")
            if h0.size(2) != self.hidden or c0.size(2) != self.hidden:
                raise RuntimeError(f"LSTM hidden width mismatch: got {h0.size(2)}, expected {self.hidden}")

        q_h_seq, new_hidden = self.lstm(z_seq_bn, hidden)             # [B*N,T,H]

        # ----- Q head and reshape back to [B,T,N,A] -----
        q_seq_bn = self.head(q_h_seq)                                 # [B*N,T,A]
        BN, T_out, A = q_seq_bn.shape
        if A != self.actions:
            raise RuntimeError(f"Q-head actions mismatch: got {A}, expected {self.actions}")
        if BN != B * N:
            raise RuntimeError(f"Internal size mismatch: BN={BN} vs B*N={B*N}")

        q_seq = (
            q_seq_bn.reshape(B, N, T_out, A)   # [B,N,T,A]
                    .permute(0, 2, 1, 3)       # [B,T,N,A]
                    .contiguous()
        )
        return q_seq, new_hidden

    @torch.no_grad()
    def step(
        self,
        X_t: torch.Tensor,               # [B, N, O]
        edge_index: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        Q_seq, new_hidden = self.forward(X_t.unsqueeze(1), edge_index, hidden)  # -> [B,1,N,A]
        return Q_seq[:, 0], new_hidden
