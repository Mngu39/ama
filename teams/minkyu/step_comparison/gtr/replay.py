"""Unmodified original definitions moved from puyo_train.py; imports are wiring only."""

import torch
import numpy as np
from dataclasses import dataclass, field
import copy
from collections import deque

from .actions import predict_values
from .model import CHANNELS, HEIGHT, WIDTH

@dataclass
class Transition:
    state: np.ndarray
    next_states: np.ndarray
    next_rewards: np.ndarray
    next_dones: np.ndarray
    terminal: bool
    chains: int = 0                                                                
    future_rewards: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))


@dataclass
class _PendingAfterstate:
    state: np.ndarray
    terminal: bool
    chains: int
    future_rewards: list = field(default_factory=list)


class NStepAssembler:
    """Assemble sampled n-step returns for an afterstate value function.

    V(x_t) predicts rewards AFTER the placement creating x_t. Thus the
    current placement's reward r_t MUST NOT enter its own TD target.
    For nonterminal x_t, the n-step target is:
      r_(t+1) + gamma*r_(t+2) + ... + gamma**(n-2)*r_(t+n-1)
        + gamma**(n-1)*max_a[r_(t+n)(a)+gamma*V(x_(t+n)(a))].
    Intermediate observed rewards follow the actual epsilon-greedy actor;
    the last action is greedily bootstrapped using the target network.
    """

    def __init__(self, n_step: int):
        if n_step < 1:
            raise ValueError('n_step must be >= 1')
        self.n_step = n_step
        self.pending = deque()

    def observe(self, state, realized_reward, terminal, chains,
                next_states, next_rewards, next_dones):
        """Accept one completed placement; emit zero or more replay items."""
                                                                  
                                                         
        for item in self.pending:
            item.future_rewards.append(float(realized_reward))
        self.pending.append(_PendingAfterstate(np.asarray(state, dtype=np.uint8),
                                               bool(terminal), int(chains)))
        completed = []
        if terminal:
                                                                        
            while self.pending:
                item = self.pending.popleft()
                completed.append(self._finish(item, None, None, None))
        else:
            if len(next_states) == 0:
                raise ValueError('Nonterminal afterstate needs candidate next states')
                                                                       
                                                                               
            while self.pending and len(self.pending[0].future_rewards) >= self.n_step - 1:
                item = self.pending.popleft()
                completed.append(self._finish(item, next_states, next_rewards, next_dones))
        return completed

    @staticmethod
    def _finish(item, next_states, next_rewards, next_dones):
        if next_states is None:
            ns = np.empty((0, CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)
            nr = np.empty(0, dtype=np.float32)
            nd = np.empty(0, dtype=np.bool_)
        else:
            ns = np.asarray(next_states, dtype=np.uint8)
            nr = np.asarray(next_rewards, dtype=np.float32)
            nd = np.asarray(next_dones, dtype=np.bool_)
        return Transition(state=item.state, next_states=ns, next_rewards=nr,
                          next_dones=nd, terminal=item.terminal,
                          chains=item.chains,
                          future_rewards=np.asarray(item.future_rewards, dtype=np.float32))


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
                                                                          
                                                       
            denom = (len(self) * probs.min()) ** (-beta)
        weights = (raw / denom).astype(np.float32)
        return [self.data[int(i)] for i in indices], indices, weights

    def update_priorities(self, indices, td_errors):
                                                                     
                                                                          
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
            prefix_max = max(len(t.future_rewards) for t in self.data)
            future_rewards = np.zeros((n, prefix_max), dtype=np.float32)
            future_counts = np.zeros(n, dtype=np.int16)
            for i, t in enumerate(self.data):
                k = len(t.next_states)
                counts[i] = k
                terminals[i] = t.terminal
                chains[i] = t.chains
                prefix_len = len(t.future_rewards)
                future_counts[i] = prefix_len
                if prefix_len:
                    future_rewards[i, :prefix_len] = t.future_rewards
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
            future_rewards = np.empty((0, 0), dtype=np.float32)
            future_counts = np.empty((0,), dtype=np.int16)
                                                                             
                                                                           
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
            'future_rewards': torch.from_numpy(future_rewards),
            'future_counts': torch.from_numpy(future_counts),
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
        future_counts = packed['future_counts'].numpy()
        future_rewards = packed['future_rewards'].numpy()
        self.data = [
            Transition(states[i].copy(), next_states[i, :int(k)].copy(),
                       rewards[i, :int(k)].copy(), dones[i, :int(k)].copy(),
                       bool(terminal[i]), int(chains[i]),
                       future_rewards[i, :int(future_counts[i])].copy())
            for i, k in enumerate(counts)
        ]
        self.priorities[:] = packed['priorities'].numpy()
        self.pos = int(packed['pos'])
        self.rng.bit_generator.state = copy.deepcopy(packed['rng_state'])


def td_targets(experiences, target_model, device, gamma, chunk_size):
    """Afterstate n-step sampled return with a greedy final backup.

    Existing 1-step experiences have an empty future_rewards array and
    give EXACTLY the original target: max_a[r(next,a)+gamma*V(next_afterstate)].
    If a later placement ends the game, only actually observed future
    rewards are summed; there is no bootstrap.
    """
    all_next = []
    spans = []
    offset = 0
    for t in experiences:
        if t.terminal or not len(t.next_states):
            spans.append((0, 0))
            continue
        start = offset
        all_next.append(t.next_states)
        offset += len(t.next_states)
        spans.append((start, offset))

    next_v = (predict_values(target_model, np.concatenate(all_next, axis=0),
                             device, chunk_size) if all_next else
              np.empty(0, dtype=np.float32))
    targets = np.zeros(len(experiences), dtype=np.float32)
    for i, transition in enumerate(experiences):
        if transition.terminal:
            continue
        future = transition.future_rewards
        prefix = sum((gamma ** k) * float(r) for k, r in enumerate(future))
        if not len(transition.next_states):
            if not len(future):
                raise ValueError('Nonterminal afterstate without future or candidate backup')
            targets[i] = prefix                               
            continue
        start, end = spans[i]
        estimates = (transition.next_rewards + gamma *
                     np.where(transition.next_dones, 0.0, next_v[start:end]))
        targets[i] = prefix + (gamma ** len(future)) * float(np.max(estimates))
    return targets
