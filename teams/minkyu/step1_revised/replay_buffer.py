"""Prioritized Experience Replay for scalar afterstate TD.

Modes:
  legacy: original without-replacement numpy.choice and sample-max IS scaling;
          reproduces the original 50k configuration (biased IS approx).
  replacement: independent with-replacement draws, w_i proportional to
          (N * P(i))**(-beta), normalized by max over the *whole buffer*.
          This is mathematically consistent importance weighting up to a
          common scale, not an exact undoing of function approximation.
"""

from dataclasses import dataclass
import copy
import numpy as np
import torch


@dataclass
class Transition:
    state: np.ndarray
    next_states: np.ndarray
    next_rewards: np.ndarray
    next_dones: np.ndarray
    terminal: bool
    chains: int = 0            # chain size on the action producing this afterstate


class PrioritizedReplay:
    def __init__(self, capacity: int, alpha: float, seed: int, mode='legacy'):
        if capacity <= 0 or not (0 <= alpha <= 1):
            raise ValueError('capacity > 0 and 0 <= alpha <= 1 required')
        if mode not in ('legacy', 'replacement'):
            raise ValueError('mode must be legacy or replacement')
        self.capacity = capacity
        self.alpha = alpha
        self.mode = mode
        self.data = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data)

    def add(self, item: Transition):
        priority = float(np.max(self.priorities[:len(self.data)])) if self.data else 1.0
        if self.pos == len(self.data):
            self.data.append(item)
        else:
            self.data[self.pos] = item
        self.priorities[self.pos] = max(priority, 1e-3)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        if len(self) < batch_size:
            raise ValueError('Not enough experiences for a full batch')
        if not 0 <= beta <= 1:
            raise ValueError('beta must lie in [0,1]')
        probs = np.maximum(self.priorities[:len(self)], 1e-6).astype(np.float64) ** self.alpha
        probs /= probs.sum()
        indices = self.rng.choice(len(self), batch_size,
                                  replace=(self.mode == 'replacement'), p=probs)
        raw = (len(self) * probs[indices]) ** (-beta)
        if self.mode == 'legacy':
            denom = raw.max()
        else:
            # Max possible raw weight across all items; does not depend on
            # the particular mini-batch that was drawn.
            denom = (len(self) * probs.min()) ** (-beta)
        weights = (raw / denom).astype(np.float32)
        return [self.data[int(i)] for i in indices], indices, weights

    def update_priorities(self, indices, td_errors):
        # With-replacement batches can contain repeated indices.  The
        # priority for such an item should reflect its largest |TD error|.
        merged = {}
        for idx, err in zip(indices, td_errors):
            i = int(idx)
            merged[i] = max(merged.get(i, 0.0), abs(float(err)))
        for idx, err in merged.items():
            self.priorities[idx] = max(err + 1e-3, 1e-3)

    def chain_distribution(self):
        """Distribution of chain sizes stored in replay and sampled by PER.

        Buffer occupancy and prioritized replay probability are different:
        high-priority observations are sampled more often than their raw
        proportion in the ring buffer. No collection-policy performance
        inference can be made from either quantity alone.
        """
        n = len(self.data)
        if n == 0:
            return {'priority_mean': None, 'priority_p95': None,
                    'priority_max': None, 'fraction_chain_ge_5': None,
                    'sample_prob_chain_ge_5': None, 'mean_chains': None,
                    'fraction_zero_chain': None}
        active = self.priorities[:n]
        chains = np.array([t.chains for t in self.data])
        weights = np.maximum(active.astype(np.float64), 1e-6) ** self.alpha
        weights /= weights.sum()
        ge5 = chains >= 5
        return {
            'priority_mean': float(active.mean()),
            'priority_p95': float(np.percentile(active, 95)),
            'priority_max': float(active.max()),
            'fraction_chain_ge_5': float(np.mean(ge5)),
            'sample_prob_chain_ge_5': float(weights[ge5].sum()),
            'mean_chains': float(chains.mean()),
            'fraction_zero_chain': float(np.mean(chains == 0)),
        }

    def state_dict(self):
        """Safe-to-load torch-only compact columnar format for resume.

        Storing each of 10k numpy transitions individually in torch.save is
        slower and increases Python pickle overhead.  All columns here are
        tensors, lists or basic builtins; torch.load(weights_only=True) works.
        """
        n = len(self.data)
        if n:
            shape = tuple(self.data[0].state.shape)
            max_next = max(len(t.next_states) for t in self.data)
            states = np.stack([t.state for t in self.data]).astype(np.uint8)
            next_states = np.zeros((n, max_next) + shape, dtype=np.uint8)
            rewards = np.zeros((n, max_next), dtype=np.float32)
            dones = np.zeros((n, max_next), dtype=np.bool_)
            counts = np.zeros(n, dtype=np.int16)
            terminals = np.zeros(n, dtype=np.bool_)
            chains = np.zeros(n, dtype=np.int16)
            for i, t in enumerate(self.data):
                k = len(t.next_states)
                counts[i] = k
                terminals[i] = t.terminal
                chains[i] = t.chains
                if k:
                    next_states[i, :k] = t.next_states
                    rewards[i, :k] = t.next_rewards
                    dones[i, :k] = t.next_dones
        else:
            states = np.empty((0, 5, 14, 6), dtype=np.uint8)
            next_states = np.empty((0, 0, 5, 14, 6), dtype=np.uint8)
            rewards = np.empty((0, 0), dtype=np.float32)
            dones = np.empty((0, 0), dtype=np.bool_)
            counts = np.empty((0,), dtype=np.int16)
            terminals = np.empty((0,), dtype=np.bool_)
            chains = np.empty((0,), dtype=np.int16)
        # JSON-friendly clone of PCG64 state: contains only builtin dicts and
        # Python integers; not a numpy object in currently supported numpy.
        return {
            'capacity': self.capacity, 'alpha': self.alpha, 'mode': self.mode,
            'pos': self.pos, 'rng_state': copy.deepcopy(self.rng.bit_generator.state),
            'priorities': torch.from_numpy(self.priorities.copy()),
            'states': torch.from_numpy(states),
            'next_states': torch.from_numpy(next_states),
            'next_rewards': torch.from_numpy(rewards),
            'next_dones': torch.from_numpy(dones),
            'next_counts': torch.from_numpy(counts),
            'terminal': torch.from_numpy(terminals),
            'chains': torch.from_numpy(chains),
        }

    def load_state_dict(self, packed):
        if (packed['capacity'] != self.capacity or
            packed['alpha'] != self.alpha or packed['mode'] != self.mode):
            raise ValueError('Replay capacity/alpha/mode mismatch at resume')
        counts = packed['next_counts'].numpy()
        states = packed['states'].numpy()
        next_states = packed['next_states'].numpy()
        rewards = packed['next_rewards'].numpy()
        dones = packed['next_dones'].numpy()
        terminal = packed['terminal'].numpy()
        chains = (packed['chains'].numpy() if 'chains' in packed else
                  np.zeros(len(counts), dtype=np.int16))
        self.data = [
            Transition(states[i].copy(), next_states[i, :int(k)].copy(),
                       rewards[i, :int(k)].copy(), dones[i, :int(k)].copy(),
                       bool(terminal[i]), int(chains[i]))
            for i, k in enumerate(counts)
        ]
        self.priorities[:] = packed['priorities'].numpy()
        self.pos = int(packed['pos'])
        self.rng.bit_generator.state = copy.deepcopy(packed['rng_state'])
