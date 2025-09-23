
#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# =================================== Replay Buffer ===================================
@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: float

class ReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int, seed: int = 1234):
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.random_gen = np.random.default_rng(seed)
        self.states = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.size = 0
        self.write_index = 0
    def push(self, tr: Transition):
        i = self.write_index
        self.states[i] = tr.state; self.next_states[i] = tr.next_state
        self.actions[i] = tr.action; self.rewards[i] = tr.reward; self.dones[i] = tr.done
        self.write_index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def __len__(self):
        return self.size
    def sample(self, batch_size: int):
        idx = self.random_gen.integers(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx])

# =================================== Sequence Replay  ===================================
@dataclass
class AgentEpisode:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    next_observations: np.ndarray

class SequenceReplay:
    def __init__(self, capacity_steps: int, observation_dim: int, seed: int = 1234):
        self.capacity_steps = capacity_steps
        self.observation_dim = observation_dim
        self.random_gen = np.random.default_rng(seed)
        self.episodes: List[AgentEpisode] = []
        self.total_steps = 0
    def push_episode(self, ep: AgentEpisode):
        self.episodes.append(ep)
        self.total_steps += int(ep.observations.shape[0])
        while self.total_steps > self.capacity_steps and len(self.episodes) > 1:
            old = self.episodes.pop(0)
            self.total_steps -= int(old.observations.shape[0])
    def __len__(self):
        return len(self.episodes)
    def sample(self, batch_size: int, seq_len: int):
        assert len(self.episodes) > 0
        ep_idx = self.random_gen.choice(len(self.episodes), size=batch_size, replace=True)
        obs_b = np.zeros((batch_size, seq_len, self.observation_dim), dtype=np.float32)
        next_obs_b = np.zeros_like(obs_b)
        act_b = np.zeros((batch_size, seq_len), dtype=np.int64)
        rew_b = np.zeros((batch_size, seq_len), dtype=np.float32)
        done_b = np.ones((batch_size, seq_len), dtype=np.float32)
        for b, eidx in enumerate(ep_idx):
            ep = self.episodes[eidx]
            T = ep.observations.shape[0]
            if T <= seq_len:
                start = 0; end = T
            else:
                start = int(self.random_gen.integers(0, T - seq_len))
                end = start + seq_len
            sl = slice(start, end)
            L = ep.observations[sl].shape[0]
            obs_b[b, :L] = ep.observations[sl]
            next_obs_b[b, :L] = ep.next_observations[sl]
            act_b[b, :L] = ep.actions[sl]
            rew_b[b, :L] = ep.rewards[sl]
            done_b[b, :L] = ep.dones[sl]
            if L < seq_len:
                obs_b[b, L:] = ep.observations[sl][-1]
                next_obs_b[b, L:] = ep.next_observations[sl][-1]
                act_b[b, L:] = ep.actions[sl][-1]
                rew_b[b, L:] = ep.rewards[sl][-1]
                done_b[b, L:] = 1.0
        return {'obs': obs_b, 'next_obs': next_obs_b, 'actions': act_b, 'rewards': rew_b, 'dones': done_b}

# =============================== Global Replay (for GNN IDQN) ===============================

class GlobalReplayBuffer:
    """
    A replay buffer that stores *joint* transitions for all agents at once.
    Shapes:
      - state_all:      [num_agents, obs_dim]
      - actions_all:    [num_agents]          (int64)
      - rewards_all:    [num_agents]          (float32)
      - next_state_all: [num_agents, obs_dim]
      - done_flag:      scalar {0.0, 1.0}     (float32)

    Sampling returns a tuple:
      (states, actions, rewards, next_states, dones)
      with shapes:
        states      : [B, num_agents, obs_dim]
        actions     : [B, num_agents]
        rewards     : [B, num_agents]
        next_states : [B, num_agents, obs_dim]
        dones       : [B]
    """
    def __init__(self, capacity: int, num_agents: int, obs_dim: int, seed: int = 1234):
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self._rng = np.random.default_rng(seed)

        self.states      = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.actions     = np.zeros((capacity, num_agents), dtype=np.int64)
        self.rewards     = np.zeros((capacity, num_agents), dtype=np.float32)
        self.dones       = np.zeros((capacity,), dtype=np.float32)

        self._size = 0
        self._idx = 0

    def __len__(self) -> int:
        return self._size

    def push(self,
             state_all,
             actions_all,
             rewards_all,
             next_state_all,
             done_flag: float):
        """
        Add one joint transition (for all agents at a single timestep).
        Arrays can be any array-like; they’ll be converted with the correct dtypes.
        """
        i = self._idx

        s  = np.asarray(state_all, dtype=np.float32)
        ns = np.asarray(next_state_all, dtype=np.float32)
        a  = np.asarray(actions_all, dtype=np.int64)
        r  = np.asarray(rewards_all, dtype=np.float32)
        d  = np.float32(done_flag)

        # Basic shape checks (helpful during integration)
        assert s.shape  == (self.num_agents, self.obs_dim),  f"state_all shape {s.shape} != {(self.num_agents, self.obs_dim)}"
        assert ns.shape == (self.num_agents, self.obs_dim),  f"next_state_all shape {ns.shape} != {(self.num_agents, self.obs_dim)}"
        assert a.shape  == (self.num_agents,),               f"actions_all shape {a.shape} != {(self.num_agents,)}"
        assert r.shape  == (self.num_agents,),               f"rewards_all shape {r.shape} != {(self.num_agents,)}"

        self.states[i]      = s
        self.next_states[i] = ns
        self.actions[i]     = a
        self.rewards[i]     = r
        self.dones[i]       = d

        self._idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Uniformly sample B transitions.
        Returns (states, actions, rewards, next_states, dones) with shapes:
          [B,N,O], [B,N], [B,N], [B,N,O], [B]
        """
        assert self._size > 0, "Cannot sample from an empty GlobalReplayBuffer."
        idx = self._rng.integers(0, self._size, size=int(batch_size))
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


# =============================== Global Sequence Replay (for GNN+LSTM IDQN) ===============================
@dataclass
class GlobalEpisode:
    """
    One full-graph episode.
      states       : [T, N, O]
      actions      : [T, N]       (int64)
      rewards      : [T, N]       (float32)
      next_states  : [T, N, O]
      dones        : [T]          (float32; 1.0 at terminal step)
    """
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray

class GlobalSequenceReplay:
    """
    Replay buffer for spatio-temporal DRQN over graphs.
    Stores *episodes* (joint across all agents) and samples fixed-length sequences.

    Capacity measured in *steps*; older episodes are discarded when over capacity.
    """
    def __init__(self, capacity_steps: int, num_agents: int, obs_dim: int, seed: int = 1234):
        self.capacity_steps = int(capacity_steps)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.rng = np.random.default_rng(seed)

        self.episodes: List[GlobalEpisode] = []
        self.total_steps: int = 0

    def __len__(self) -> int:
        return len(self.episodes)

    # ------------- Push -------------
    def push_episode(
        self,
        states: np.ndarray,       # [T, N, O], float32
        actions: np.ndarray,      # [T, N],   int64
        rewards: np.ndarray,      # [T, N],   float32
        next_states: np.ndarray,  # [T, N, O], float32
        dones: np.ndarray,        # [T],      float32
    ):
        # Basic validation + dtype coercion
        s  = np.asarray(states, dtype=np.float32)
        a  = np.asarray(actions, dtype=np.int64)
        r  = np.asarray(rewards, dtype=np.float32)
        ns = np.asarray(next_states, dtype=np.float32)
        d  = np.asarray(dones, dtype=np.float32)

        assert s.ndim == 3 and s.shape[1] == self.num_agents and s.shape[2] == self.obs_dim, f"states shape {s.shape} != [T,N,O]"
        assert ns.shape == s.shape, f"next_states shape {ns.shape} != {s.shape}"
        assert a.shape == (s.shape[0], self.num_agents), f"actions shape {a.shape} != [T,N]"
        assert r.shape == a.shape, f"rewards shape {r.shape} != [T,N]"
        assert d.shape == (s.shape[0],), f"dones shape {d.shape} != [T]"

        ep = GlobalEpisode(s, a, r, ns, d)
        self.episodes.append(ep)
        self.total_steps += int(s.shape[0])

        # Evict oldest episodes to respect capacity (keep at least one)
        while self.total_steps > self.capacity_steps and len(self.episodes) > 1:
            old = self.episodes.pop(0)
            self.total_steps -= int(old.states.shape[0])

    # ------------- Sample -------------
    def sample(self, batch_size: int, seq_len: int):
        """
        Randomly sample B sequences of length L=seq_len.
        If an episode is shorter than L, we pad by repeating the final frame and set dones=1.0 for padded steps.

        Returns tuple:
          states      : [B, L, N, O]
          actions     : [B, L, N]
          rewards     : [B, L, N]
          next_states : [B, L, N, O]
          dones       : [B, L]
        """
        assert len(self.episodes) > 0, "GlobalSequenceReplay is empty."

        B = int(batch_size); L = int(seq_len)
        N = self.num_agents; O = self.obs_dim

        S  = np.zeros((B, L, N, O), dtype=np.float32)
        A  = np.zeros((B, L, N),    dtype=np.int64)
        R  = np.zeros((B, L, N),    dtype=np.float32)
        NS = np.zeros((B, L, N, O), dtype=np.float32)
        D  = np.ones((B, L),        dtype=np.float32)  # default to terminal; will overwrite with real flags

        # Choose episodes with replacement
        ep_indices = self.rng.integers(low=0, high=len(self.episodes), size=B)

        for b, eidx in enumerate(ep_indices):
            ep = self.episodes[eidx]
            T = ep.states.shape[0]

            if T >= L:
                # Random contiguous window
                start = int(self.rng.integers(low=0, high=T - L + 1))
                end = start + L
                S[b]  = ep.states[start:end]
                A[b]  = ep.actions[start:end]
                R[b]  = ep.rewards[start:end]
                NS[b] = ep.next_states[start:end]
                D[b]  = ep.dones[start:end]
            else:
                # Copy entire episode then pad by repeating last frame; mark pads as terminal
                S[b, :T]  = ep.states
                A[b, :T]  = ep.actions
                R[b, :T]  = ep.rewards
                NS[b, :T] = ep.next_states
                D[b, :T]  = ep.dones

                if T > 0:
                    # Repeat final frame
                    S[b, T:]  = ep.states[T-1:T]
                    A[b, T:]  = ep.actions[T-1:T]
                    R[b, T:]  = 0.0                            # no extra reward on pads
                    NS[b, T:] = ep.next_states[T-1:T]
                    D[b, T:]  = 1.0                             # padded steps are terminal
                else:
                    # Degenerate empty episode (shouldn't happen): keep zeros, dones=1
                    pass

        return (S, A, R, NS, D)

# =============================== CTDE: joint replay for centralized training ===============================
class JointReplayBuffer:
    """
    Stores joint transitions for VDN/QMIX-style CTDE.
      state_all      : [N, O]
      action_all     : [N]      (int64)
      reward_all     : [N]      (float32)  -> we'll sum to a team reward at train time
      next_state_all : [N, O]
      done_flag      : scalar {0.0, 1.0}
    Sampling returns:
      states      : [B, N, O]
      actions     : [B, N]
      rewards     : [B, N]   (we keep per-agent; reduce to team inside the update)
      next_states : [B, N, O]
      dones       : [B]
    """
    def __init__(self, capacity: int, num_agents: int, obs_dim: int, seed: int = 1234):
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self._rng = np.random.default_rng(seed)

        self.states      = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.actions     = np.zeros((capacity, num_agents), dtype=np.int64)
        self.rewards     = np.zeros((capacity, num_agents), dtype=np.float32)
        self.dones       = np.zeros((capacity,), dtype=np.float32)

        self._size = 0
        self._idx = 0

    def __len__(self) -> int:
        return self._size

    def push(self,
             state_all,
             actions_all,
             rewards_all,
             next_state_all,
             done_flag: float):
        i = self._idx
        s  = np.asarray(state_all, dtype=np.float32)           # [N,O]
        ns = np.asarray(next_state_all, dtype=np.float32)      # [N,O]
        a  = np.asarray(actions_all, dtype=np.int64)           # [N]
        r  = np.asarray(rewards_all, dtype=np.float32)         # [N]
        d  = np.float32(done_flag)

        assert s.shape  == (self.num_agents, self.obs_dim)
        assert ns.shape == (self.num_agents, self.obs_dim)
        assert a.shape  == (self.num_agents,)
        assert r.shape  == (self.num_agents,)

        self.states[i]      = s
        self.next_states[i] = ns
        self.actions[i]     = a
        self.rewards[i]     = r
        self.dones[i]       = d

        self._idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        assert self._size > 0, "Cannot sample from empty JointReplayBuffer."
        idx = self._rng.integers(0, self._size, size=int(batch_size))
        return (self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx])


# ================= CTDE: joint sequence replay for centralized training with LSTM =================

@dataclass
class JointEpisode:
    states: np.ndarray       # [T, N, O]
    actions: np.ndarray      # [T, N]   (int64)
    rewards: np.ndarray      # [T, N]   (float32)
    next_states: np.ndarray  # [T, N, O]
    dones: np.ndarray        # [T]      (float32; env-step terminal flag)

class JointSequenceReplay:
    """
    Stores full joint episodes and samples fixed-length sequences with masks.
    Returns (S, A, R, NS, D, M):
      S,NS: [B,L,N,O], A,R: [B,L,N], D: [B,L], M: [B,L] (1=real, 0=pad)
    """
    def __init__(self, capacity_steps: int, num_agents: int, obs_dim: int, seed: int = 1234):
        self.capacity_steps = int(capacity_steps)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.rng = np.random.default_rng(seed)
        self.episodes: List[JointEpisode] = []
        self.total_steps: int = 0

    def __len__(self) -> int:
        return len(self.episodes)

    def push_episode(
        self,
        states: np.ndarray,       # [T,N,O] float32
        actions: np.ndarray,      # [T,N]   int64
        rewards: np.ndarray,      # [T,N]   float32
        next_states: np.ndarray,  # [T,N,O] float32
        dones: np.ndarray,        # [T]     float32
    ):
        s  = np.asarray(states, dtype=np.float32)
        a  = np.asarray(actions, dtype=np.int64)
        r  = np.asarray(rewards, dtype=np.float32)
        ns = np.asarray(next_states, dtype=np.float32)
        d  = np.asarray(dones, dtype=np.float32)

        assert s.ndim == 3 and s.shape[1] == self.num_agents and s.shape[2] == self.obs_dim
        assert ns.shape == s.shape
        assert a.shape == (s.shape[0], self.num_agents)
        assert r.shape == a.shape
        assert d.shape == (s.shape[0],)

        ep = JointEpisode(s, a, r, ns, d)
        self.episodes.append(ep)
        self.total_steps += int(s.shape[0])

        # Evict oldest episodes to respect capacity (keep at least one)
        while self.total_steps > self.capacity_steps and len(self.episodes) > 1:
            old = self.episodes.pop(0)
            self.total_steps -= int(old.states.shape[0])

    def sample(self, batch_size: int, seq_len: int):
        """
        Returns:
          S, A, R, NS : [B, L, N, O], [B, L, N], [B, L, N], [B, L, N, O]
          D           : [B, L]   (team done per step; pads are 1)
          M           : [B, L]   (valid mask; real steps=1, pads=0)
        """
        assert len(self.episodes) > 0, "Replay is empty."

        B = int(batch_size); L = int(seq_len)
        N = self.num_agents; O = self.obs_dim

        S  = np.zeros((B, L, N, O), dtype=np.float32)
        A  = np.zeros((B, L, N),    dtype=np.int64)
        R  = np.zeros((B, L, N),    dtype=np.float32)
        NS = np.zeros((B, L, N, O), dtype=np.float32)
        D  = np.ones((B, L),        dtype=np.float32)   # terminals/pads default to 1
        M  = np.zeros((B, L),       dtype=np.float32)   # mask (1 real, 0 pad)

        ep_indices = self.rng.integers(low=0, high=len(self.episodes), size=B)
        for b, eidx in enumerate(ep_indices):
            ep = self.episodes[eidx]
            T = ep.states.shape[0]

            if T >= L:
                start = int(self.rng.integers(low=0, high=T - L + 1))
                end = start + L
                S[b]  = ep.states[start:end]
                A[b]  = ep.actions[start:end]
                R[b]  = ep.rewards[start:end]
                NS[b] = ep.next_states[start:end]
                D[b]  = ep.dones[start:end]
                M[b]  = 1.0
            else:
                S[b, :T]  = ep.states
                A[b, :T]  = ep.actions
                R[b, :T]  = ep.rewards
                NS[b, :T] = ep.next_states
                D[b, :T]  = ep.dones
                M[b, :T]  = 1.0

                if T > 0:
                    # pad tail by repeating last frame; reward=0; done=1 for pads; mask=0
                    S[b, T:]  = ep.states[T-1:T]
                    A[b, T:]  = ep.actions[T-1:T]
                    R[b, T:]  = 0.0
                    NS[b, T:] = ep.next_states[T-1:T]
                    D[b, T:]  = 1.0
                    # M[b, T:] stays 0

        return (S, A, R, NS, D, M)
