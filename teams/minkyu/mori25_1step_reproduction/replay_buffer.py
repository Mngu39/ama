"""Prioritized replay for one-step TD on afterstate values.

An item for x_t stores the next revealed pair's possible choices:
  V(x_t) ~= max_a [r_(t+1)(a) + gamma * V_target(x_(t+1)(a))].
This target is an off-policy Bellman backup, not a fixed reward allocation.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Transition:
    state: np.ndarray            # uint8 [5,14,6], afterstate x_t
    next_states: np.ndarray      # uint8 [K,5,14,6], K legal next actions
    next_rewards: np.ndarray     # float32 [K], scaled score + gameover penalty
    next_dones: np.ndarray       # bool [K], gameover OR turn 50
    terminal: bool               # current afterstate x_t is terminal


class PrioritizedReplay:
    def __init__(self, capacity: int, alpha: float, seed: int):
        if capacity <= 0 or not (0 <= alpha <= 1):
            raise ValueError("capacity > 0 and 0 <= alpha <= 1 required")
        self.capacity = capacity
        self.alpha = alpha
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
            raise ValueError("Not enough experiences to sample a full batch")
        p = np.maximum(self.priorities[:len(self)], 1e-6) ** self.alpha
        p /= p.sum()
        indices = self.rng.choice(len(self), batch_size, replace=False, p=p)
        weights = (len(self) * p[indices]) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        return [self.data[int(i)] for i in indices], indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[int(idx)] = max(abs(float(err)) + 1e-3, 1e-3)
