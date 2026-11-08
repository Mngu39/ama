"""뿌요뿌요 점수 기반 강화학습. VS Code에서 ama-main 폴더를 열고 실행하세요.

Linux/WSL 최초 준비: bash scripts/setup_wsl.sh && source .venv/bin/activate
의존성: python -m pip install torch numpy
학습: python puyo_train.py
평가: python puyo_train.py evaluate --checkpoint runs/<run>/best.pt
정형 보상 비교군: python puyo_train.py --form-weight 0
기본값: 5,000 episodes, 1-ply training, final exhaustive 3-ply evaluation on 250 games.
"""

import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
import copy
from typing import Any
import csv
import gzip
import json
from pathlib import Path
from collections import defaultdict
from math import sqrt
import os
import random
import uuid
import argparse
import time
from collections import deque
from datetime import datetime, timezone
import subprocess
import sys
import torch.nn.functional as F


# ===========================================================================
# 01. 보드 인코딩 · 다음 뿌요를 제외한 배치 후 판
# ===========================================================================

# Encode Ama afterstates without exposing current/NEXT/NEXT2 puyo pairs.
#
# Ama stores colors only in the bottom 13 rows. Its 14th row is a six-bit
# occupancy mask, not six color values. Therefore the input is 5 x 14 x 6:
# channels 0-3 are R/Y/G/B; channel 4 is garbage (rows 0-12) or row-14
# occupancy (row 13). This is NOT identical to the 2025 paper's 14x6x4.

COLOR_CHANNEL = {"R": 0, "Y": 1, "G": 2, "B": 3, "#": 4}
CHANNELS = 5
HEIGHT = 14
WIDTH = 6


# ===========================================================================
# 02. 가치 신경망 · 잔차 CNN
# ===========================================================================

# Scalar value network for Ama afterstates (paper-inspired residual CNN).
#
# The 2025 paper uses 14x6x4 color planes. This repository currently exposes
# 5x14x6 (four color planes plus row-14 occupancy / garbage); therefore the
# architecture is paper-inspired but not an exact reproduction.



class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class AfterstateValueNet(nn.Module):
    """[N,5,14,6] -> [N] future value in units of score_scale points."""

    def __init__(self, input_channels: int = 5, channels: int = 256,
                 blocks: int = 6):
        super().__init__()
        if blocks < 1 or channels < 1:
            raise ValueError("blocks and channels must be >= 1")
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )
        self.residuals = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.head_conv = nn.Conv2d(channels, 1, 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14 * 6, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.stem[0].in_channels, 14, 6):
            raise ValueError(f"Expected [N,{self.stem[0].in_channels},14,6], got {tuple(x.shape)}")
        x = self.stem(x)
        x = self.residuals(x)
        x = self.head_conv(x)
        return self.head(x).squeeze(-1)


# ===========================================================================
# 03. Ama 정형 점수와 보조 보상 · 원 게임 점수는 별도로 기록
# ===========================================================================

FORM_PATTERNS = (
    # GTR: rows from bottom to top, group weights, distinct-color pairs
    (
        ((2, 2, 5, 0, 0, 0), (1, 1, 2, 5, 0, 0), (1, 2, 5, 0, 0, 0), (3, 3, 3, 4, 0, 0), (4, 4, 4, 0, 0, 0), (0, 0, 0, 0, 0, 0)),
        (0, 2, 1, 2, 0, 0),
        ((1, 2), (1, 3), (2, 3), (2, 5), (3, 4), (3, 5)),
    ),
    # FRON: rows from bottom to top, group weights, distinct-color pairs
    (
        ((3, 3, 2, 3, 6, 0), (1, 2, 2, 7, 0, 0), (1, 1, 3, 6, 0, 0), (4, 4, 4, 5, 0, 0), (5, 5, 5, 0, 0, 0), (0, 0, 0, 0, 0, 0)),
        (0, 2, 1, 1, 2, 0, 0, 0),
        ((1, 2), (1, 3), (1, 4), (2, 3), (2, 7), (3, 4), (3, 6), (4, 5)),
    ),
    # SGTR: rows from bottom to top, group weights, distinct-color pairs
    (
        ((2, 3, 3, 6, 0, 0), (1, 2, 2, 3, 6, 0), (1, 1, 3, 6, 0, 0), (4, 4, 4, 5, 0, 0), (5, 5, 5, 0, 0, 0), (0, 0, 0, 0, 0, 0)),
        (0, 2, 1, 1, 2, 0, 0),
        ((1, 2), (1, 3), (1, 4), (2, 3), (3, 4), (3, 6), (4, 5)),
    ),
)

# Use Ama's column-first scan: zero-weight groups keep the first observed color.
def compile_pattern(grid, weights, distinct):
    cells = [(x, y, grid[y][x]) for x in range(6) for y in range(6) if grid[y][x]]
    maximum = sum(weights[group] for _, _, group in cells)
    return cells, weights, distinct, maximum


COMPILED_FORMS = tuple(compile_pattern(*pattern) for pattern in FORM_PATTERNS)


def form_potential(board):
    # Ama ignores forms when garbage occupies the bottom-left 4x4.
    if any('#' in row[:4] for row in board[:4]):
        return 0.0
    best = 0.0
    for cells, weights, distinct, maximum in COMPILED_FORMS:
        colors = [None] * len(weights)
        score = 0
        for x, y, group in cells:
            color = board[y][x]
            if color == '.':
                continue
            if colors[group] is None:
                colors[group] = color
            elif weights[group] and colors[group] != color:
                break
            score += weights[group]
        else:
            if all(colors[i] is None or colors[i] != colors[j] for i, j in distinct):
                best = max(best, score / maximum)
    return best


class RewardEnvironment:
    # Keep the shaped signal separate from raw engine points used by metrics.
    def __init__(self, engine, gamma, form_weight, score_scale):
        self.engine = engine
        self.gamma = gamma
        self.raw_weight = form_weight * score_scale
        self.potential = 0.0

    def reset(self, seed):
        obs = self.engine.reset(seed)
        self.potential = self._potential(obs)
        return obs

    def observation(self):
        return self.engine.observation()

    def done(self):
        return self.engine.done()

    def _potential(self, observation):
        return form_potential(observation['board']) if self.raw_weight else 0.0

    def _reward(self, result):
        terminal = result['terminated'] or result['truncated']
        after = 0.0 if terminal else self._potential(result['observation'])
        result['form_bonus'] = self.raw_weight * (self.gamma * after - self.potential)
        return after

    def step(self, action):
        result = self.engine.step(action)
        self.potential = self._reward(result)
        return result

    def preview_all(self):
        candidates = self.engine.preview_all()
        for candidate in candidates:
            self._reward(candidate)
        return candidates

    def clone(self):
        from copy import copy
        other = copy(self)
        other.engine = self.engine.clone()
        return other


def make_environment(gamma, weight, scale):
    from rl_env import puyo_cpp
    engine = puyo_cpp.PuyoEnv()
    if not hasattr(engine, 'clone'):
        raise RuntimeError('Rebuild rl_env/puyo_cpp with scripts/setup_wsl.sh for clone support.')
    return RewardEnvironment(engine, gamma, weight, scale)


# ===========================================================================
# 04. 합법 배치 비교와 탐험
# ===========================================================================

# Epsilon-greedy action selection via candidate afterstates.
#
# At a decision state s, choose argmax_a [r(s,a) + gamma V(x(s,a))].
# Value V excludes the score earned by the CURRENT placement.



def encode_candidates(candidates):
    """Batch-encode only board cells and row-14 masks; never read future pairs."""
    if not candidates:
        raise ValueError("No legal candidates")
    observations = [c["observation"] for c in candidates]
    cells = "".join("".join(obs["board"]) for obs in observations).encode("ascii")
    cells = np.frombuffer(cells, dtype=np.uint8).reshape(-1, 13, WIDTH)
    planes = np.zeros((len(candidates), CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)
    for color, channel in COLOR_CHANNEL.items():
        planes[:, channel, :13, :] = cells == ord(color)
    masks = np.asarray([obs["row14"] for obs in observations], dtype=np.uint8)
    planes[:, 4, 13, :] = (masks[:, None] >> np.arange(WIDTH)) & 1
    return planes


def scaled_immediate_rewards(candidates, scale: float, gameover_penalty: float):
    return np.asarray([
        (float(c["score"]) + c["form_bonus"] + (gameover_penalty if c["terminated"] else 0.0)) / scale
        for c in candidates
    ], dtype=np.float32)


def candidate_dones(candidates):
    return np.asarray([
        bool(c["terminated"] or c["truncated"]) for c in candidates
    ], dtype=np.bool_)


@torch.no_grad()
def predict_values(model, arrays: np.ndarray, device: torch.device, chunk_size: int = 128):
    """Inference in chunks; preserve model's previous train/eval mode."""
    was_training = model.training
    model.eval()
    outputs = []
    try:
        for start in range(0, len(arrays), chunk_size):
            batch = torch.as_tensor(arrays[start:start + chunk_size],
                                    device=device, dtype=torch.float32)
            outputs.append(model(batch).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
    finally:
        model.train(was_training)


def pick_action(candidates, model, device, rng, epsilon: float,
                gamma: float, score_scale: float, gameover_penalty: float,
                chunk_size: int = 128):
    """Return candidate index and encoded afterstates (not raw action ID)."""
    arrays = encode_candidates(candidates)
    if rng.random() < epsilon:
        return int(rng.integers(0, len(candidates))), arrays

    rewards = scaled_immediate_rewards(candidates, score_scale, gameover_penalty)
    dones = candidate_dones(candidates)
    values = predict_values(model, arrays, device, chunk_size)
    scores = rewards + gamma * np.where(dones, 0.0, values)
    return int(np.argmax(scores)), arrays


# ===========================================================================
# 05. 우선순위 경험 재생
# ===========================================================================

# Prioritized Experience Replay for scalar afterstate TD.
#
# Modes:
#   legacy: original without-replacement numpy.choice and sample-max IS scaling;
#           reproduces the original 50k configuration (biased IS approx).
#   replacement: independent with-replacement draws, w_i proportional to
#           (N * P(i))**(-beta), normalized by max over the *whole buffer*.
#           This is mathematically consistent importance weighting up to a
#           common scale, not an exact undoing of function approximation.



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


# ===========================================================================
# 06. 한 수 TD 학습 목표
# ===========================================================================

# One-step off-policy TD target for scalar afterstate values.




def td_targets(experiences, target_model, device, gamma, chunk_size):
    """Compute off-policy Bellman targets for afterstates, without gradients.

    The target network evaluates candidate afterstates produced from the
    NEXT observed piece. Immediate scores are already scaled and include
    gameover penalty. Terminal afterstates have target 0 by definition.
    """
    all_next = []
    spans = []
    offset = 0
    for t in experiences:
        if t.terminal:
            spans.append((0, 0))
            continue
        if not len(t.next_states):
            raise ValueError("Nonterminal afterstate has no next legal actions")
        start = offset
        all_next.append(t.next_states)
        offset += len(t.next_states)
        spans.append((start, offset))

    if all_next:
        flat = np.concatenate(all_next, axis=0)
        next_v = predict_values(target_model, flat, device, chunk_size)
    else:
        next_v = np.empty(0, dtype=np.float32)

    targets = np.zeros(len(experiences), dtype=np.float32)
    for i, transition in enumerate(experiences):
        if transition.terminal:
            continue
        start, end = spans[i]
        estimates = (transition.next_rewards + gamma *
                     np.where(transition.next_dones, 0.0, next_v[start:end]))
        targets[i] = float(np.max(estimates))
    return targets


# ===========================================================================
# 07. 세 수 탐색 · 최선 경로의 첫 수만 실행
# ===========================================================================

# Greedy afterstate search (1-ply and paper-inspired pruned 3-ply).
#
# Search uses only the current and the two visible next pairs.  It does not
# make use of the simulated queue past those three pairs, even though the
# C++ engine internally owns a complete random sequence.
#
# Value V(x) is discounted FUTURE reward after the action giving x; thus the
# 3-ply value is r0 + gamma*r1 + gamma**2*r2 + gamma**3*V(x2).
# Intermediate pruning uses a matching shorter-horizon bootstrap score.





@dataclass(frozen=True)
class SearchResult:
    action: int
    score: float                # scaled, predicted discounted total reward
    evaluated: int              # number of CNN-evaluated candidate afterstates
    simulated: int              # number of simulated placements
    depth: int
    root_candidates: int


def _scores(candidates, model, device, gamma, scale, penalty, chunk):
    if not candidates:
        return np.empty(0, dtype=np.float64)
    arr = encode_candidates(candidates)
    value = predict_values(model, arr, device, chunk)
    reward = scaled_immediate_rewards(candidates, scale, penalty)
    done = candidate_dones(candidates)
    # Keep the exact NumPy expression/dtypes of the original 1-ply policy,
    # so loading run_50000.pt does not introduce avoidable tie/rounding drift.
    return reward + gamma * np.where(done, 0.0, value)


def _rank_top(entries, keep, *, score_index=0):
    """Stable tie-break in generation order (same as 1-ply argmax)."""
    return sorted(entries, key=lambda v: -v[score_index])[:keep]


def exhaustive_three_ply(env, roots, model, device, gamma, scale, penalty, chunk_size):
    """Read A/B/C only. Batch one root at a time to bound memory on dense boards."""
    best_action = None
    best_score = -np.inf
    simulated = len(roots)
    evaluated = 0
    root_rewards = scaled_immediate_rewards(roots, scale, penalty)
    for root, first_reward in zip(roots, root_rewards):
        action = int(root["action"])
        score = float(first_reward)
        if not (root["terminated"] or root["truncated"]):
            after_a = env.clone()
            after_a.step(action)
            seconds = after_a.preview_all()
            simulated += len(seconds)
            second_rewards = scaled_immediate_rewards(seconds, scale, penalty)
            leaves, prefixes = [], []
            score = -np.inf
            for second, second_reward in zip(seconds, second_rewards):
                prefix = float(first_reward) + gamma * float(second_reward)
                if second["terminated"] or second["truncated"]:
                    score = max(score, prefix)
                    continue
                after_b = after_a.clone()
                after_b.step(int(second["action"]))
                thirds = after_b.preview_all()
                leaves.extend(thirds)
                prefixes.extend([prefix] * len(thirds))
            if leaves:
                simulated += len(leaves)
                evaluated += len(leaves)
                last = _scores(leaves, model, device, gamma, scale, penalty, chunk_size)
                score = max(score, float(np.max(np.asarray(prefixes) + gamma**2 * last)))
        if score > best_score:
            best_score, best_action = score, action
    if best_action is None:
        raise RuntimeError("No legal path in 3-ply search")
    return SearchResult(best_action, best_score, evaluated, simulated, 3, len(roots))


def greedy_search(env, model, device, gamma: float, score_scale: float,
                  gameover_penalty: float, ply: int = 3, root_keep: int = 22,
                  second_keep: int = 484, chunk_size: int = 128,
                  root_candidates=None) -> SearchResult:
    """Find a real-game action.  No model training or env stepping occurs.

    Default evaluation keeps all 22 roots and all 484 two-placement paths.
    Smaller explicit limits enable beam pruning.  Terminal paths are
    considered as complete paths and never expanded beyond termination.
    """
    if ply not in (1, 3):
        raise ValueError('ply must be 1 or 3')
    if not root_keep or not second_keep or chunk_size < 1:
        raise ValueError('root_keep, second_keep, chunk_size must be positive')
    roots = root_candidates if root_candidates is not None else env.preview_all()
    if not roots:
        raise RuntimeError('No legal actions in a nonterminal state')
    if ply == 3 and root_keep >= len(roots) and second_keep >= 22 * len(roots):
        return exhaustive_three_ply(env, roots, model, device, gamma, score_scale,
                                    gameover_penalty, chunk_size)
    root_values = _scores(roots, model, device, gamma, score_scale,
                          gameover_penalty, chunk_size)
    chosen = int(np.argmax(root_values))
    if ply == 1:
        return SearchResult(int(roots[chosen]['action']), float(root_values[chosen]),
                            len(roots), len(roots), 1, len(roots))
    if not hasattr(env, 'clone'):
        raise RuntimeError('puyo_cpp PuyoEnv.clone() unavailable; rebuild updated rl_env/bindings.cpp')

    # Entries: (prefix estimate, root_action, cloned_env, path_reward).
    root_nodes = [(float(root_values[i]), int(c['action']), i)
                  for i, c in enumerate(roots) if not (c['terminated'] or c['truncated'])]
    completed = [(float(root_values[i]), int(c['action']))
                 for i, c in enumerate(roots) if c['terminated'] or c['truncated']]
    selected_roots = _rank_top(root_nodes, root_keep)
    n_sim = n_eval = len(roots)

    # Batch ALL depth-2 candidate CNN evaluations, rather than issuing one
    # GPU launch per root. The 3-ply actor may call this millions of times.
    groups2 = []   # (root_action, depth1_env, r0, candidate_list)
    for _, root_action, i in selected_roots:
        state = env.clone()
        actual = state.step(root_action)
        if actual['terminated'] or actual['truncated']:
            completed.append((float(root_values[i]), root_action))
            continue
        second = state.preview_all()
        if not second:
            raise RuntimeError('Nonterminal cloned state with no legal actions')
        r0 = float(scaled_immediate_rewards([roots[i]], score_scale, gameover_penalty)[0])
        groups2.append((root_action, state, r0, second))
    flat2 = [candidate for _, _, _, cs in groups2 for candidate in cs]
    n_sim += len(flat2)
    n_eval += len(flat2)
    scores2 = _scores(flat2, model, device, gamma, score_scale,
                      gameover_penalty, chunk_size)
    depth2_entries = []  # (score, root_action, depth1_env, second_action, prefix)
    offset = 0
    for root_action, state, r0, second in groups2:
        for candidate in second:
            stage2_score = r0 + gamma * float(scores2[offset])
            offset += 1
            if candidate['terminated'] or candidate['truncated']:
                completed.append((stage2_score, root_action))
            else:
                r1 = float(scaled_immediate_rewards(
                    [candidate], score_scale, gameover_penalty)[0])
                depth2_entries.append((stage2_score, root_action, state,
                                       int(candidate['action']), r0 + gamma * r1))

    selected_depth2 = _rank_top(depth2_entries, second_keep)
    groups3 = []  # (root_action, prefix, candidate_list)
    for _, root_action, depth1_env, second_action, reward_prefix in selected_depth2:
        state = depth1_env.clone()
        actual = state.step(second_action)
        if actual['terminated'] or actual['truncated']:
            completed.append((reward_prefix, root_action))
            continue
        third = state.preview_all()
        if not third:
            raise RuntimeError('Nonterminal depth-2 state with no legal actions')
        groups3.append((root_action, reward_prefix, third))
    flat3 = [candidate for _, _, cs in groups3 for candidate in cs]
    n_sim += len(flat3)
    n_eval += len(flat3)
    scores3 = _scores(flat3, model, device, gamma, score_scale,
                      gameover_penalty, chunk_size)
    offset = 0
    for root_action, reward_prefix, third in groups3:
        count = len(third)
        # The terminal mask in _scores prevents value leakage after gameover
        # or the 50-placement time limit.
        total = reward_prefix + gamma ** 2 * float(np.max(scores3[offset:offset + count]))
        completed.append((total, root_action))
        offset += count
    if not completed:
        # A nonterminal branch with fewer than three steps should only occur
        # at the artificial turn limit, which is explicitly handled above.
        raise RuntimeError('No completed 3-ply path')
    best = max(completed, key=lambda x: x[0])
    return SearchResult(best[1], best[0], n_eval, n_sim, 3, len(roots))


# ===========================================================================
# 08. 기록 저장
# ===========================================================================

# Portable, lossless CSV and JSON output for independent evaluations.



EPISODE_COLUMNS = [
    'run_id', 'checkpoint', 'ply', 'episode', 'seed', 'total_score', 'max_chain',
    'turns', 'terminated', 'truncated', 'activations', 'simulated_candidates',
    'evaluated_candidates', 'decision_time_seconds',
]
EVAL_MOVE_COLUMNS = [
    'run_id', 'checkpoint', 'ply', 'episode', 'seed', 'turn', 'action',
    'visible_current', 'visible_next', 'visible_next2',
    'chains', 'score', 'cumulative_score', 'max_chain_so_far',
    'terminated', 'truncated', 'search_value_scaled', 'simulated_candidates',
    'evaluated_candidates', 'decision_time_seconds', 'board_before',
    'row14_before', 'board_after', 'row14_after',
]


def board_string(observation):
    """Lower row first; one slash-separated row per height, reversible."""
    return '/'.join(observation['board'])


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_move_csv_gz(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 would make the gzip bytes reproducible, but gzip.open does not
    # expose this argument on all Python releases; content is reproducible.
    with gzip.open(path, 'wt', newline='', encoding='utf-8', compresslevel=6) as stream:
        writer = csv.DictWriter(stream, fieldnames=EVAL_MOVE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


class EpisodeResults:
    """Per-run statistics, persisted after each completed training or evaluation game."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.counts = defaultdict(int)
        self.games = 0
        self.total_score = 0
        self.total_max_chain = 0

    def record(self, episode, seed, score, max_chain):
        append_csv(self.directory / 'episode_max_chains.csv',
                   dict(episode=episode, seed=seed, score=score, max_chain=max_chain),
                   ['episode', 'seed', 'score', 'max_chain'])
        self.games += 1
        self.total_score += score
        self.total_max_chain += max_chain
        self.counts[max_chain] += 1
        write_json(self.directory / 'episode_statistics.json', {
            'episodes': self.games,
            'mean_score': self.total_score / self.games,
            'mean_max_chain': self.total_max_chain / self.games,
            'max_chain_distribution': [
                {'max_chain': chain, 'games': count, 'percent': 100 * count / self.games}
                for chain, count in sorted(self.counts.items())
            ],
        })


def append_csv(path, row, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ===========================================================================
# 09. 원 게임 점수와 연쇄 통계
# ===========================================================================

# Evaluation metrics computed from episode and move records, not training loss.


THRESHOLDS = (3, 5, 7, 10)


def summarize(episodes, moves):
    if not episodes:
        raise ValueError('At least one episode is needed')
    n = len(episodes)
    scores = np.asarray([e['total_score'] for e in episodes], dtype=np.float64)
    max_chains = np.asarray([e['max_chain'] for e in episodes], dtype=np.float64)
    turns = np.asarray([e['turns'] for e in episodes], dtype=np.float64)
    positive = np.asarray([m['chains'] for m in moves if m['chains'] > 0], dtype=np.float64)
    chains_per_game = defaultdict(int)
    for m in moves:
        if m['chains'] > 0:
            chains_per_game[int(m['episode'])] += 1
    output = {
        'n_episodes': n,
        'mean_max_chain': float(max_chains.mean()),
        'std_max_chain_between_games': float(max_chains.std(ddof=1)) if n > 1 else None,
        'se_mean_max_chain_between_games': float(max_chains.std(ddof=1) / sqrt(n)) if n > 1 else None,
        'median_max_chain': float(np.median(max_chains)),
        'p10_max_chain': float(np.percentile(max_chains, 10)),
        'p90_max_chain': float(np.percentile(max_chains, 90)),
        'mean_score': float(scores.mean()),
        'std_score_between_games': float(scores.std(ddof=1)) if n > 1 else None,
        'median_score': float(np.median(scores)),
        'mean_turns': float(turns.mean()),
        'gameovers': int(sum(bool(e['terminated']) for e in episodes)),
        'turn_limits': int(sum(bool(e['truncated']) for e in episodes)),
        'gameover_rate': float(sum(bool(e['terminated']) for e in episodes) / n),
        'mean_chain_per_activation': float(positive.mean()) if len(positive) else 0.0,
        'chain_activation_count': int(len(positive)),
        'chain_activations_per_game': float(len(positive) / n),
        'max_chain_histogram': {str(int(k)): int(v) for k, v in
                                zip(*np.unique(max_chains.astype(np.int64), return_counts=True))},
    }
    for threshold in THRESHOLDS:
        output[f'p_max_chain_ge_{threshold}'] = float(np.mean(max_chains >= threshold))
        output[f'mean_activations_ge_{threshold}_per_game'] = (
            sum(int(m['chains'] >= threshold) for m in moves) / n)
    # These SEs quantify evaluation-game randomness for THESE trained weights.
    # They do NOT quantify between-training-seed uncertainty.
    output['uncertainty_note'] = ('Episode SD/SE condition on one checkpoint; '
                                  'independent training runs are needed to estimate training variance.')
    return output


# ===========================================================================
# 10. 체크포인트 및 재개
# ===========================================================================

# Atomic model checkpoints and complete, replay-inclusive training resumes.




FORMAT = 2


def _atomic_torch_save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        torch.save(data, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def model_snapshot(model, episode, env_steps, updates, elapsed, run_id, cfg, value_cfg):
    return {
        'format_version': FORMAT,
        'model': model.state_dict(),
        'episode': int(episode), 'env_steps': int(env_steps),
        'gradient_updates': int(updates), 'elapsed_seconds': float(elapsed),
        'model_config': dict(cfg), 'value_config': dict(value_cfg),
        'run_id': run_id,
    }


def save_model(path, model, episode, env_steps, updates, elapsed, run_id,
               model_cfg, value_cfg):
    snapshot = model_snapshot(model, episode, env_steps, updates, elapsed,
                              run_id, model_cfg, value_cfg)
    _atomic_torch_save(snapshot, path)


def _rng_state(policy_rng):
    return {
        'python': random.getstate(),
        # numpy legacy RNG state includes ndarray, which is disallowed by
        # weights_only=True; store its uint32 vector as torch tensor.
        'numpy_global': _numpy_global_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'policy_numpy': policy_rng.bit_generator.state,
    }


def _numpy_global_state():
    state = np.random.get_state()
    return (state[0], torch.as_tensor(state[1].copy()), state[2], state[3], state[4])


def _restore_rng(state, policy_rng):
    random.setstate(tuple(state['python']))
    np_state = state['numpy_global']
    np.random.set_state((np_state[0], np_state[1].numpy(), np_state[2],
                         np_state[3], np_state[4]))
    torch.set_rng_state(state['torch_cpu'])
    if torch.cuda.is_available() and state['torch_cuda']:
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    policy_rng.bit_generator.state = state['policy_numpy']


def save_resume(path, model, target_model, optimizer, replay, policy_rng,
                episode, env_steps, updates, elapsed, best_metric, run_id,
                model_cfg, value_cfg, train_cfg):
    data = model_snapshot(model, episode, env_steps, updates, elapsed,
                          run_id, model_cfg, value_cfg)
    data.update({
        'kind': 'full_resume', 'target_model': target_model.state_dict(),
        'optimizer': optimizer.state_dict(), 'replay': replay.state_dict(),
        'rng': _rng_state(policy_rng),
        'best_metric': float(best_metric) if best_metric is not None else None,
        'train_config': dict(train_cfg),
    })
    _atomic_torch_save(data, path)


def load_resume(path, model, target_model, optimizer, replay, policy_rng,
                train_cfg):
    packed = torch.load(path, map_location='cpu', weights_only=True)
    if packed.get('kind') != 'full_resume' or packed.get('format_version') != FORMAT:
        raise ValueError('Not a compatible full-resume checkpoint')
    for key, value in packed['train_config'].items():
        if key not in train_cfg or train_cfg[key] != value:
            raise ValueError(f'Resume training config mismatch: {key}: '
                             f'stored={value!r} requested={train_cfg.get(key)!r}')
    model.load_state_dict(packed['model'], strict=True)
    target_model.load_state_dict(packed['target_model'], strict=True)
    optimizer.load_state_dict(packed['optimizer'])
    replay.load_state_dict(packed['replay'])
    _restore_rng(packed['rng'], policy_rng)
    return packed


# ===========================================================================
# 11. 학습 진단
# ===========================================================================

# Fixed-probe diagnostics: interference, bootstrapping and BatchNorm.
#
# A change in any diagnostic is *evidence to investigate*, not proof of a
# specific cause of policy collapse.  Fixed probe boards are built using a
# separate deterministic random-rollout policy, not the current learner.





@dataclass
class Probe:
    arrays: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    transition: Transition


def make_probes(env_class, *, seed=500000, episodes=6, per_episode=4):
    rng = np.random.default_rng(seed + 123456)
    env = env_class()
    probes = []
    for i in range(episodes):
        env.reset(seed + i)
        for t in range(per_episode):
            if env.done():
                break
            candidates = env.preview_all()
            if not candidates:
                break
            action_i = int(rng.integers(len(candidates)))
            arrays = encode_candidates(candidates)
            step = env.step(int(candidates[action_i]['action']))
            following = [] if (step['terminated'] or step['truncated']) else env.preview_all()
            next_states = (encode_candidates(following) if following else
                           np.empty((0,) + arrays.shape[1:], dtype=np.uint8))
            probes.append(Probe(
                arrays=arrays,
                rewards=scaled_immediate_rewards(candidates, 1000.0, -1000.0),
                dones=candidate_dones(candidates),
                transition=Transition(
                    arrays[action_i], next_states,
                    (scaled_immediate_rewards(following, 1000.0, -1000.0)
                     if following else np.empty((0,), dtype=np.float32)),
                    (candidate_dones(following) if following else
                     np.empty(0, dtype=np.bool_)),
                    bool(step['terminated'] or step['truncated']),
                ),
            ))
    return probes


def _train_mode_predict_preserving_bn(model, arrays, device):
    # Probe inference in train mode MUST NOT alter the actual training state.
    # Restore BN running buffers immediately, even when forward fails.
    original_training = model.training
    buffers = {name: buf.detach().clone()
               for name, buf in model.named_buffers()
               if 'running_mean' in name or 'running_var' in name or
               'num_batches_tracked' in name}
    try:
        model.train(True)
        with torch.no_grad():
            return model(torch.as_tensor(arrays, dtype=torch.float32,
                                         device=device)).detach().cpu().numpy()
    finally:
        with torch.no_grad():
            all_buffers = dict(model.named_buffers())
            for name, copy in buffers.items():
                all_buffers[name].copy_(copy)
        model.train(original_training)


def measure_probes(probes, model, target_model, device, value_cfg,
                   previous=None, chunk_size=128):
    if not probes:
        return {}, None
    gamma = value_cfg['gamma']
    scale = value_cfg['score_scale']
    penalty = value_cfg['gameover_penalty']
    gaps = []
    target_gaps = []
    flips = 0
    action_indices = []
    action_scores = []
    bn_diffs = []
    margin = []
    # Probe rewards were created with default scale=1000/penalty=-1000.
    # Current training defaults match, but explicitly rescale them to avoid
    # silently mixing score units in a configurable run.
    ratio = 1000.0 / scale
    for probe in probes:
        eval_v = predict_values(model, probe.arrays, device, chunk_size)
        target_v = predict_values(target_model, probe.arrays, device, chunk_size)
        bn_v = _train_mode_predict_preserving_bn(model, probe.arrays, device)
        r = probe.rewards * ratio
        # The original stored reward includes a gameover penalty. Reconstruct
        # reward from raw scaled score under custom penalties is impossible
        # without metadata; reject nondefault penalty in the calling trainer.
        if penalty != -1000.0:
            raise ValueError('Probe diagnostics currently require --gameover-penalty -1000')
        action_eval = r + gamma * np.where(probe.dones, 0.0, eval_v)
        action_train = r + gamma * np.where(probe.dones, 0.0, bn_v)
        best = int(np.argmax(action_eval))
        action_indices.append(best)
        action_scores.append(float(action_eval[best]))
        flips += int(best != int(np.argmax(action_train)))
        gaps.extend(np.abs(eval_v - bn_v).tolist())
        target_gaps.extend(np.abs(eval_v - target_v).tolist())
        bn_diffs.extend((eval_v - bn_v).tolist())
        if len(action_eval) >= 2:
            sorted_scores = np.sort(action_eval)
            margin.append(float(sorted_scores[-1] - sorted_scores[-2]))

    # The fixed-probe TD error has a MOVING target network; it is a useful
    # diagnostic but not a stationary validation loss.
    transitions = [p.transition for p in probes]
    # Important: this is only fully consistent with the chosen scale if the
    # default gameover penalty is used; re-scale the reward arrays.
    from dataclasses import replace
    adjusted = [replace(t, next_rewards=t.next_rewards * ratio) for t in transitions]
    y = td_targets(adjusted, target_model, device, gamma, chunk_size)
    states = np.stack([t.state for t in transitions])
    predicted = predict_values(model, states, device, chunk_size)
    td_abs = np.abs(y - predicted)
    report = {
        'probe_count': len(probes),
        'probe_bn_eval_train_mae': float(np.mean(gaps)),
        'probe_bn_eval_train_signed_bias': float(np.mean(bn_diffs)),
        'probe_bn_action_flip_rate': float(flips / len(probes)),
        'probe_online_target_mae': float(np.mean(target_gaps)),
        'probe_median_action_margin': float(np.median(margin)) if margin else None,
        'probe_td_abs_mean': float(td_abs.mean()),
        'probe_td_abs_p95': float(np.percentile(td_abs, 95)),
        'probe_td_abs_max': float(td_abs.max()),
        'probe_mean_best_action_value': float(np.mean(action_scores)),
    }
    this = {'actions': action_indices, 'values': predicted.copy()}
    if previous is not None:
        report['probe_action_flip_since_prev'] = float(
            np.mean(np.asarray(previous['actions']) != action_indices))
        report['probe_value_mae_since_prev'] = float(
            np.abs(previous['values'] - predicted).mean())
    else:
        report['probe_action_flip_since_prev'] = None
        report['probe_value_mae_since_prev'] = None
    return report, this


# ===========================================================================
# 12. 독립 평가
# ===========================================================================

# Independent greedy evaluation with 1-ply / 3-ply search and raw records.





def load_evaluation_model(path, device):
    # Supports original 50k checkpoint and new, weights-only snapshots.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = AfterstateValueNet(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model, checkpoint['value_config'], checkpoint


def run_evaluation(model, value_config, device, *, n_episodes=1000,
                   first_seed=100000, ply=3, root_keep=22, second_keep=484,
                   chunk_size=128, output_dir=None, checkpoint_name='', run_id='',
                   write_moves=True, progress=False):
    """Return summary and optionally persist episode/move records.

    This function does not modify network weights, BatchNorm running stats,
    or training random-number generators. Each episode starts from its own
    deterministic queue seed; tie-breaks use deterministic action order.
    """
    if n_episodes < 1 or first_seed < 0:
        raise ValueError('n_episodes must be positive and first_seed >= 0')
    results = None
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=False)
        results = EpisodeResults(output_dir)
    was_training = model.training
    model.eval()
    env = make_environment(value_config['gamma'], value_config.get('form_weight', 0.0),
                           value_config['score_scale'])
    episodes, moves = [], []
    start = time.perf_counter()
    try:
        for ep in range(n_episodes):
            seed = first_seed + ep
            env.reset(seed)
            score = longest = turns = activations = 0
            evaluated = simulated = 0
            decision_seconds = 0.0
            ending = None
            while not env.done():
                before = env.observation()
                decision_started = time.perf_counter()
                search = greedy_search(
                    env, model, device,
                    gamma=value_config['gamma'],
                    score_scale=value_config['score_scale'],
                    gameover_penalty=value_config['gameover_penalty'],
                    ply=ply, root_keep=root_keep, second_keep=second_keep,
                    chunk_size=chunk_size,
                )
                elapsed = time.perf_counter() - decision_started
                result = env.step(search.action)
                ending = result
                turns += 1
                score += int(result['score'])
                chains = int(result['chains'])
                longest = max(longest, chains)
                activations += int(chains > 0)
                evaluated += search.evaluated
                simulated += search.simulated
                decision_seconds += elapsed
                if write_moves:
                    shown = before['visible']
                    moves.append({
                        'run_id': run_id, 'checkpoint': checkpoint_name, 'ply': ply,
                        'episode': ep, 'seed': seed, 'turn': turns,
                        'action': search.action,
                        'visible_current': ''.join(shown[0]),
                        'visible_next': ''.join(shown[1]),
                        'visible_next2': ''.join(shown[2]),
                        'chains': chains, 'score': int(result['score']),
                        'cumulative_score': score, 'max_chain_so_far': longest,
                        'terminated': bool(result['terminated']),
                        'truncated': bool(result['truncated']),
                        'search_value_scaled': search.score,
                        'simulated_candidates': search.simulated,
                        'evaluated_candidates': search.evaluated,
                        'decision_time_seconds': elapsed,
                        'board_before': board_string(before),
                        'row14_before': int(before['row14']),
                        'board_after': board_string(result['observation']),
                        'row14_after': int(result['observation']['row14']),
                    })
            if results is not None:
                results.record(ep, seed, score, longest)
            episodes.append({
                'run_id': run_id, 'checkpoint': checkpoint_name, 'ply': ply,
                'episode': ep, 'seed': seed, 'total_score': score,
                'max_chain': longest, 'turns': turns,
                'terminated': bool(ending['terminated']),
                'truncated': bool(ending['truncated']), 'activations': activations,
                'simulated_candidates': simulated, 'evaluated_candidates': evaluated,
                'decision_time_seconds': decision_seconds,
            })
            if progress and (ep + 1) % max(1, n_episodes // 10) == 0:
                print(f'eval {ep + 1}/{n_episodes}', flush=True)
    finally:
        model.train(was_training)

    # Compute activation statistics from episode aggregates when move logging
    # is switched off; full detailed statistics require the move records.
    if write_moves:
        summary = summarize(episodes, moves)
    else:
        summary = summarize(episodes, [])
        summary['chain_activation_count'] = sum(e['activations'] for e in episodes)
        summary['chain_activations_per_game'] = summary['chain_activation_count'] / n_episodes
        summary['mean_chain_per_activation'] = None
        for k in (3, 5, 7, 10):
            summary[f'mean_activations_ge_{k}_per_game'] = None
    summary.update({
        'ply': ply, 'first_seed': first_seed, 'last_seed': first_seed + n_episodes - 1,
        'run_id': run_id, 'checkpoint': checkpoint_name,
        'eval_wall_time_seconds': time.perf_counter() - start,
        'total_simulated_candidates': int(sum(e['simulated_candidates'] for e in episodes)),
        'total_evaluated_candidates': int(sum(e['evaluated_candidates'] for e in episodes)),
        'avg_decision_time_seconds': float(sum(e['decision_time_seconds'] for e in episodes) /
                                            max(1, sum(e['turns'] for e in episodes))),
        'root_keep': root_keep if ply == 3 else None,
        'second_keep': second_keep if ply == 3 else None,
    })
    if output_dir is not None:
        path = Path(output_dir)
        write_csv(path / 'episodes.csv', episodes, EPISODE_COLUMNS)
        if write_moves:
            write_move_csv_gz(path / 'moves.csv.gz', moves)
        write_json(path / 'summary.json', summary)
    return summary


def evaluation_cli():
    p = argparse.ArgumentParser(description='Independent greedy 1-ply / 3-ply evaluation')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=250)
    p.add_argument('--seed', type=int, default=100000)
    p.add_argument('--ply', type=int, choices=[1, 3], default=3)
    p.add_argument('--root-keep', type=int, default=22)
    p.add_argument('--second-keep', type=int, default=484)
    p.add_argument('--infer-chunk', type=int, default=128)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--out-dir', type=Path, default=None)
    p.add_argument('--summary-only', action='store_true',
                   help='Do not save per-placement CSV; loses chain-per-event metrics')
    args = p.parse_args(sys.argv[2:])
    device = select_device(args.device)
    model, value_config, checkpoint = load_evaluation_model(args.checkpoint, device)
    path = args.out_dir
    if path is None:
        path = args.checkpoint.parent / 'evaluations' / (
            f'{args.checkpoint.stem}_ply{args.ply}_seed{args.seed}_n{args.episodes}')
    result = run_evaluation(
        model, value_config, device, n_episodes=args.episodes,
        first_seed=args.seed, ply=args.ply, root_keep=args.root_keep,
        second_keep=args.second_keep, chunk_size=args.infer_chunk,
        output_dir=path, checkpoint_name=str(args.checkpoint),
        run_id=str(checkpoint.get('run_id', 'legacy')),
        write_moves=not args.summary_only, progress=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(f'files={path.resolve()}')


# ===========================================================================
# 13. 학습 설정 및 실행
# ===========================================================================

# 1-step afterstate TD training, evaluation, diagnostics and reproducible I/O.
#
# A v2 training run starts in its own directory.  The original 50k checkpoint
# is supported for evaluation only; it does not contain the optimizer / replay
# state required for exact training continuation.





EP_COLUMNS = [
    'episode', 'seed', 'turns', 'score', 'max_chain', 'activations',
    'activations_ge_3', 'activations_ge_5', 'activations_ge_7',
    'terminated', 'truncated', 'epsilon', 'replay_size', 'env_steps',
    'gradient_updates', 'avg_weighted_huber', 'avg_unweighted_huber',
    'avg_abs_td', 'episode_seconds', 'elapsed_seconds',
]
TRAIN_MOVE_COLUMNS = [
    'episode', 'seed', 'turn', 'action', 'visible_current', 'visible_next',
    'visible_next2', 'chains', 'score', 'cumulative_score',
    'epsilon', 'terminated', 'truncated',
]
EVAL_COLUMNS = [
    'episode', 'env_steps', 'gradient_updates', 'eval_ply', 'eval_episodes',
    'mean_max_chain', 'std_max_chain_between_games',
    'mean_score', 'mean_turns', 'p_max_chain_ge_3', 'p_max_chain_ge_5',
    'p_max_chain_ge_7', 'p_max_chain_ge_10',
    'mean_chain_per_activation', 'chain_activations_per_game',
    'gameover_rate', 'avg_decision_time_seconds', 'eval_wall_time_seconds',
    'new_best',
]
DIAG_COLUMNS = [
    'episode', 'gradient_updates', 'probe_count',
    'probe_bn_eval_train_mae', 'probe_bn_eval_train_signed_bias',
    'probe_bn_action_flip_rate', 'probe_online_target_mae',
    'probe_median_action_margin', 'probe_td_abs_mean', 'probe_td_abs_p95',
    'probe_td_abs_max', 'probe_mean_best_action_value',
    'probe_action_flip_since_prev', 'probe_value_mae_since_prev',
    'replay_priority_mean', 'replay_priority_p95', 'replay_priority_max',
    'replay_fraction_chain_ge_5', 'replay_sample_prob_chain_ge_5',
    'replay_mean_chains', 'replay_fraction_zero_chain',
]
UPDATE_COLUMNS = [
    'episode', 'env_steps', 'gradient_updates', 'weighted_huber',
    'unweighted_huber', 'abs_td_mean', 'abs_td_p95', 'pred_value_mean',
    'target_value_mean', 'grad_norm_preclip', 'beta',
]


def select_device(requested):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; install a compatible PyTorch CUDA build')
    return torch.device(requested)


def linear_schedule(start, end, step, duration):
    return start + min(1.0, step / max(1, duration)) * (end - start)


def format_duration(seconds):
    total = max(0, int(seconds + .5))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def optimize(model, target_model, optimizer, replay, device, args, beta):
    samples, indices, weights = replay.sample(args.batch_size, beta)
    states = torch.as_tensor(np.stack([t.state for t in samples]),
                             dtype=torch.float32, device=device)
    with torch.no_grad():
        targets_np = td_targets(samples, target_model, device, args.gamma, args.infer_chunk)
    targets = torch.as_tensor(targets_np, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    model.train()
    predictions = model(states)
    errors = targets - predictions
    per_item = F.smooth_l1_loss(predictions, targets, reduction='none')
    weighted = (per_item * weights_t).mean()
    if not torch.isfinite(weighted):
        raise FloatingPointError('Non-finite weighted Huber loss')
    optimizer.zero_grad(set_to_none=True)
    weighted.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    errors_numpy = errors.detach().cpu().numpy()
    replay.update_priorities(indices, errors_numpy)
    abs_error = np.abs(errors_numpy)
    return {
        'weighted_huber': float(weighted.item()),
        'unweighted_huber': float(per_item.detach().mean().item()),
        'abs_td_mean': float(abs_error.mean()),
        'abs_td_p95': float(np.percentile(abs_error, 95)),
        'pred_value_mean': float(predictions.detach().mean().item()),
        'target_value_mean': float(targets.detach().mean().item()),
        'grad_norm_preclip': float(gradient_norm.item()),
        'beta': float(beta),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='One-step afterstate TD experiment v2')
    p.add_argument('--form-weight', type=float, default=0.04,
                   help='Potential weight in scaled reward units; 0 disables shaping.')
    p.add_argument('--episodes', type=int, default=5000)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--blocks', type=int, default=6)
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--score-scale', type=float, default=1000.0)
    p.add_argument('--gameover-penalty', type=float, default=-1000.0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--capacity', type=int, default=10000)
    p.add_argument('--warmup', type=int, default=256)
    p.add_argument('--update-every', type=int, default=4)
    p.add_argument('--target-sync', type=int, default=500)
    p.add_argument('--per-alpha', type=float, default=0.6)
    p.add_argument('--per-beta-start', type=float, default=0.4)
    p.add_argument('--per-beta-steps', type=int, default=50000)
    p.add_argument('--per-mode', choices=['legacy', 'replacement'], default='replacement')
    p.add_argument('--epsilon-start', type=float, default=0.4)
    p.add_argument('--epsilon-final', type=float, default=0.02)
    p.add_argument('--epsilon-steps', type=int, default=500000)
    p.add_argument('--infer-chunk', type=int, default=128)
    p.add_argument('--save-every', type=int, default=2500,
                   help='Save evaluation-only ep_XXXXX.pt snapshots')
    p.add_argument('--resume-every', type=int, default=500,
                   help='Overwrite the full last_resume.pt (large, includes replay)')
    p.add_argument('--eval-every', type=int, default=5000)
    p.add_argument('--eval-episodes', type=int, default=250)
    p.add_argument('--eval-seed', type=int, default=100000)
    p.add_argument('--eval-ply', choices=[1, 3], type=int, default=3)
    p.add_argument('--train-ply', choices=[1, 3], type=int, default=1,
                   help='1-ply historical baseline or 3-ply behavior policy; '
                        'TD target remains strictly 1-step in either case')
    p.add_argument('--train-root-keep', type=int, default=5)
    p.add_argument('--train-second-keep', type=int, default=10)
    p.add_argument('--eval-root-keep', type=int, default=22)
    p.add_argument('--eval-second-keep', type=int, default=484)
    p.add_argument('--probe-seed', type=int, default=500000)
    p.add_argument('--no-probes', action='store_true')
    p.add_argument('--no-train-moves', action='store_true')
    p.add_argument('--no-eval-moves', action='store_true')
    p.add_argument('--log-update-every', type=int, default=500)
    p.add_argument('--print-every', type=int, default=25)
    p.add_argument('--run-dir', type=Path, default=None,
                   help='Unique output directory; must not contain an existing run')
    p.add_argument('--resume', type=Path, default=None,
                   help='Continue from a v2 full-resume checkpoint into NEW run directory')
    p.add_argument('--out', type=Path, default=None,
                   help='Optional legacy filename for a FINAL weights-only copy')
    return p.parse_args(argv)


def _train_config(args):
    # Everything that changes gradients or training RNG sequence. Exclude
    # output, evaluation, diagnostic and completion-budget settings.
    return {k: getattr(args, k) for k in (
        'seed', 'channels', 'blocks', 'gamma', 'score_scale',
        'form_weight', 'gameover_penalty', 'lr', 'batch_size', 'capacity', 'warmup',
        'update_every', 'target_sync', 'per_alpha', 'per_beta_start',
        'per_beta_steps', 'per_mode', 'epsilon_start', 'epsilon_final',
        'epsilon_steps', 'infer_chunk', 'train_ply', 'train_root_keep',
        'train_second_keep',
    )}


def _git_rev():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL, text=True,
                                       timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _validate(args):
    if not np.isfinite(args.form_weight) or args.form_weight < 0:
        raise ValueError('form-weight must be finite and nonnegative')
    for k in ('episodes', 'batch_size', 'capacity', 'update_every',
              'target_sync', 'infer_chunk', 'eval_episodes', 'print_every'):
        if getattr(args, k) < 1:
            raise ValueError(f'--{k.replace("_", "-")} must be >= 1')
    for k in ('save_every', 'resume_every', 'eval_every', 'log_update_every'):
        if getattr(args, k) < 0:
            raise ValueError(f'--{k.replace("_", "-")} must be >= 0')
    if not (0 < args.gamma <= 1 and args.score_scale > 0 and args.lr > 0):
        raise ValueError('gamma, score-scale and lr out of range')
    if not 0 <= args.per_alpha <= 1 or not 0 <= args.per_beta_start <= 1:
        raise ValueError('PER alpha/beta out of range')
    if args.no_probes is False and args.gameover_penalty != -1000:
        raise ValueError('Fixed-probe diagnostics currently require -1000 gameover penalty; '
                         'pass --no-probes for another penalty')
    if args.no_probes is False and args.score_scale <= 0:
        raise ValueError('Invalid scale')
    if (args.train_root_keep < 1 or args.train_second_keep < 1 or
        args.eval_root_keep < 1 or args.eval_second_keep < 1):
        raise ValueError('Search root/second keep must be >= 1')
    if args.eval_seed <= args.seed + args.episodes - 1:
        raise ValueError('Evaluation seeds must be disjoint and greater than training seeds')
    if args.probe_seed <= args.eval_seed + args.eval_episodes - 1:
        raise ValueError('Probe seeds must be greater and disjoint from evaluation seeds')


def train(args):
    _validate(args)
    run_started = time.perf_counter()
    unique = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'ama_form_td_{unique}_seed{args.seed}'
    run_dir = (args.run_dir if args.run_dir is not None else
               Path(__file__).resolve().parent / 'runs' / run_id)
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'Run directory is nonempty: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device(args.device)
    model = AfterstateValueNet(CHANNELS, args.channels, args.blocks).to(device)
    target_model = AfterstateValueNet(CHANNELS, args.channels, args.blocks).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    replay = PrioritizedReplay(args.capacity, args.per_alpha, args.seed + 1,
                               mode=args.per_mode)
    rng = np.random.default_rng(args.seed + 2)
    env = make_environment(args.gamma, args.form_weight, args.score_scale)
    model_cfg = {'channels': args.channels, 'blocks': args.blocks,
                 'input_channels': CHANNELS}
    value_cfg = {'gamma': args.gamma, 'score_scale': args.score_scale,
                 'gameover_penalty': args.gameover_penalty, 'form_weight': args.form_weight}
    cfg = _train_config(args)
    start_episode = 0
    env_steps = gradient_updates = 0
    elapsed_before_resume = 0.0
    best_score = None
    parent = None
    if args.resume is not None:
        loaded = load_resume(args.resume, model, target_model, optimizer,
                             replay, rng, cfg)
        start_episode = int(loaded['episode'])
        env_steps = int(loaded['env_steps'])
        gradient_updates = int(loaded['gradient_updates'])
        elapsed_before_resume = float(loaded['elapsed_seconds'])
        parent = str(Path(args.resume).resolve())
        # Track best within this continuation. The saved source best model
        # may be in a different directory and is not copied implicitly.
        best_score = None
        if start_episode >= args.episodes:
            raise ValueError('Requested episodes must exceed resume checkpoint episode')

    def elapsed():
        return elapsed_before_resume + time.perf_counter() - run_started

    manifest = {
        'run_id': run_id, 'start_from_episode': start_episode,
        'planned_total_episodes': args.episodes, 'train_config': cfg,
        'model_config': model_cfg, 'value_config': value_cfg,
        'evaluation_config': {
            'eval_every': args.eval_every, 'eval_episodes': args.eval_episodes,
            'eval_seed': args.eval_seed, 'eval_ply': args.eval_ply,
            'root_keep': args.eval_root_keep,
            'second_keep': args.eval_second_keep,
        },
        'parent_resume_checkpoint': parent,
        'git_commit': _git_rev(),
        'python': sys.version, 'torch': torch.__version__,
        'numpy': np.__version__, 'device': str(device),
        'cuda_device_name': (torch.cuda.get_device_name(device)
                             if device.type == 'cuda' else None),
        'important_caveats': [
            'Single actor; not the 1000-actor setup in Mori 2025.',
            '5x14x6 features, not paper exact 4x14x6.',
            'Afterstate board only; future pairs/remaining turns not network inputs.',
            'PER mode selected in train_config; replacement changes training from legacy.',
            'Validation-game SE is NOT between-training-run SE.',
            '3-ply search is action selection, not 3-step TD learning.',
            'Training actor ply and evaluation ply are separately configured.',
        ],
    }
    write_json(run_dir / 'manifest.json', manifest)

    training_results = EpisodeResults(run_dir)
    ep_path = run_dir / 'train_episodes.csv'
    move_stream = None
    move_writer = None
    if not args.no_train_moves:
        move_stream = gzip.open(run_dir / 'train_moves.csv.gz', 'wt',
                                encoding='utf-8', newline='', compresslevel=5)
        move_writer = csv.DictWriter(move_stream, fieldnames=TRAIN_MOVE_COLUMNS)
        move_writer.writeheader()

    probes = None
    if not args.no_probes:
        probes = make_probes(lambda: make_environment(args.gamma, args.form_weight, args.score_scale), seed=args.probe_seed)
    probe_previous = None
    recent_times = deque(maxlen=10)
    last_episode = start_episode
    print(f'device={device}, run_dir={run_dir}, PER={args.per_mode}, '
          f'resume_from={start_episode}, eval_ply={args.eval_ply}', flush=True)

    def save_weights(episode, label):
        save_model(run_dir / label, model, episode, env_steps, gradient_updates,
                   elapsed(), run_id, model_cfg, value_cfg)

    def full_save(episode):
        save_resume(run_dir / 'last_resume.pt', model, target_model, optimizer,
                    replay, rng, episode, env_steps, gradient_updates,
                    elapsed(), best_score, run_id, model_cfg, value_cfg, cfg)

    try:
        for episode in range(start_episode + 1, args.episodes + 1):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            ep_started = time.perf_counter()
            seed = args.seed + episode - 1
            env.reset(seed)
            candidates = env.preview_all()
            ep_score = ep_max_chain = turns = activations = 0
            n3 = n5 = n7 = 0
            step_stats = []
            last_ending = None
            epsilon = linear_schedule(args.epsilon_start, args.epsilon_final,
                                      env_steps, args.epsilon_steps)
            while not env.done():
                epsilon = linear_schedule(args.epsilon_start, args.epsilon_final,
                                          env_steps, args.epsilon_steps)
                shown = env.observation()['visible'] if move_writer else None
                if args.train_ply == 1:
                    picked, encoded = pick_action(
                        candidates, model, device, rng, epsilon, args.gamma,
                        args.score_scale, args.gameover_penalty, args.infer_chunk)
                    action = int(candidates[picked]['action'])
                else:
                    # The exploration roll is exactly once per placement.
                    # In random exploration, no expensive tree is constructed.
                    if rng.random() < epsilon:
                        picked = int(rng.integers(len(candidates)))
                    else:
                        choice = greedy_search(
                            env, model, device, args.gamma, args.score_scale,
                            args.gameover_penalty, ply=3,
                            root_keep=args.train_root_keep,
                            second_keep=args.train_second_keep,
                            chunk_size=args.infer_chunk,
                            root_candidates=candidates)
                        action = choice.action
                        picked = next((i for i, c in enumerate(candidates)
                                       if int(c['action']) == action), -1)
                        if picked < 0:
                            raise RuntimeError('3-ply selected an illegal root action')
                    encoded = encode_candidates(candidates)
                    action = int(candidates[picked]['action'])
                result = env.step(action)
                last_ending = result
                chains = int(result['chains'])
                score = int(result['score'])
                ep_score += score
                ep_max_chain = max(ep_max_chain, chains)
                turns += 1
                env_steps += 1
                activations += int(chains > 0)
                n3 += int(chains >= 3)
                n5 += int(chains >= 5)
                n7 += int(chains >= 7)
                terminal = bool(result['terminated'] or result['truncated'])
                following = [] if terminal else env.preview_all()
                replay.add(Transition(
                    state=encoded[picked],
                    next_states=(encode_candidates(following) if following else
                                 np.empty((0, CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)),
                    next_rewards=(scaled_immediate_rewards(
                        following, args.score_scale, args.gameover_penalty)
                        if following else np.empty(0, dtype=np.float32)),
                    next_dones=(candidate_dones(following) if following else
                                np.empty(0, dtype=np.bool_)),
                    terminal=terminal, chains=chains,
                ))
                if move_writer:
                    move_writer.writerow({
                        'episode': episode, 'seed': seed, 'turn': turns,
                        'action': action, 'visible_current': ''.join(shown[0]),
                        'visible_next': ''.join(shown[1]),
                        'visible_next2': ''.join(shown[2]),
                        'chains': chains, 'score': score,
                        'cumulative_score': ep_score, 'epsilon': epsilon,
                        'terminated': bool(result['terminated']),
                        'truncated': bool(result['truncated']),
                    })
                candidates = following
                if len(replay) >= max(args.warmup, args.batch_size) and env_steps % args.update_every == 0:
                    beta = linear_schedule(args.per_beta_start, 1.0,
                                           gradient_updates, args.per_beta_steps)
                    stat = optimize(model, target_model, optimizer, replay,
                                    device, args, beta)
                    step_stats.append(stat)
                    gradient_updates += 1
                    if args.log_update_every and gradient_updates % args.log_update_every == 0:
                        append_csv(run_dir / 'train_updates.csv',
                                   {'episode': episode, 'env_steps': env_steps,
                                    'gradient_updates': gradient_updates, **stat},
                                   UPDATE_COLUMNS)
                    if gradient_updates % args.target_sync == 0:
                        target_model.load_state_dict(model.state_dict())
                        target_model.eval()

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            ep_seconds = time.perf_counter() - ep_started
            recent_times.append(ep_seconds)
            last_episode = episode
            ep_stats = {
                'episode': episode, 'seed': seed, 'turns': turns,
                'score': ep_score, 'max_chain': ep_max_chain,
                'activations': activations, 'activations_ge_3': n3,
                'activations_ge_5': n5, 'activations_ge_7': n7,
                'terminated': bool(last_ending['terminated']),
                'truncated': bool(last_ending['truncated']),
                'epsilon': epsilon, 'replay_size': len(replay),
                'env_steps': env_steps, 'gradient_updates': gradient_updates,
                'avg_weighted_huber': (float(np.mean([s['weighted_huber'] for s in step_stats]))
                                       if step_stats else None),
                'avg_unweighted_huber': (float(np.mean([s['unweighted_huber'] for s in step_stats]))
                                         if step_stats else None),
                'avg_abs_td': (float(np.mean([s['abs_td_mean'] for s in step_stats]))
                               if step_stats else None),
                'episode_seconds': ep_seconds, 'elapsed_seconds': elapsed(),
            }
            training_results.record(episode, seed, ep_score, ep_max_chain)
            append_csv(ep_path, ep_stats, EP_COLUMNS)
            if move_stream and episode % 50 == 0:
                move_stream.flush()
            if episode % args.print_every == 0 or episode == args.episodes:
                eta = (sum(recent_times) / len(recent_times) * (args.episodes - episode))
                loss = (f"{ep_stats['avg_weighted_huber']:.5f}"
                        if ep_stats['avg_weighted_huber'] is not None else 'n/a')
                print(f'episode={episode}/{args.episodes} turns={turns} '
                      f'score={ep_score} max_chain={ep_max_chain} eps={epsilon:.3f} '
                      f'replay={len(replay)} updates={gradient_updates} '
                      f'avg_loss={loss} ep_time={format_duration(ep_seconds)} '
                      f'elapsed={format_duration(elapsed())} '
                      f'eta~{format_duration(eta)}', flush=True)

            is_end = episode == args.episodes
            if is_end or (args.save_every and episode % args.save_every == 0):
                save_weights(episode, f'ep_{episode:06d}.pt')
                save_weights(episode, 'last.pt')

            if args.eval_every and (is_end or episode % args.eval_every == 0):
                # Every evaluation result must refer to an existing, immutable
                # checkpoint even when --eval-every != --save-every.
                snapshot = run_dir / f'ep_{episode:06d}.pt'
                if not snapshot.exists():
                    save_weights(episode, snapshot.name)
                evaluation_name = f'ep_{episode:06d}_ply{args.eval_ply}'
                result = run_evaluation(
                    model, value_cfg, device, n_episodes=args.eval_episodes,
                    first_seed=args.eval_seed, ply=args.eval_ply,
                    root_keep=args.eval_root_keep,
                    second_keep=args.eval_second_keep,
                    chunk_size=args.infer_chunk,
                    output_dir=run_dir / 'evaluations' / evaluation_name,
                    checkpoint_name=f'ep_{episode:06d}.pt', run_id=run_id,
                    write_moves=not args.no_eval_moves,
                )
                perf = result['mean_score']
                improving = (best_score is None or perf > best_score)
                if improving:
                    best_score = perf
                    save_weights(episode, 'best.pt')
                append_csv(run_dir / 'eval_metrics.csv', {
                    'episode': episode, 'env_steps': env_steps,
                    'gradient_updates': gradient_updates,
                    'eval_ply': args.eval_ply, 'eval_episodes': args.eval_episodes,
                    'new_best': improving, **result,
                }, EVAL_COLUMNS)
                print(f'eval episode={episode} mean_max_chain={result["mean_max_chain"]:.3f} '
                      f'mean_score={result["mean_score"]:.2f} '
                      f'best={best_score:.3f}', flush=True)
                if probes:
                    diagnostic, probe_previous = measure_probes(
                        probes, model, target_model, device, value_cfg,
                        previous=probe_previous, chunk_size=args.infer_chunk)
                    diagnostic.update({
                        'episode': episode, 'gradient_updates': gradient_updates,
                        **replay.chain_distribution(),
                    })
                    # Prefix columns match JSON field names directly.
                    pr = replay.chain_distribution()
                    diagnostic['replay_priority_mean'] = pr['priority_mean']
                    diagnostic['replay_priority_p95'] = pr['priority_p95']
                    diagnostic['replay_priority_max'] = pr['priority_max']
                    diagnostic['replay_fraction_chain_ge_5'] = pr['fraction_chain_ge_5']
                    diagnostic['replay_sample_prob_chain_ge_5'] = pr['sample_prob_chain_ge_5']
                    diagnostic['replay_mean_chains'] = pr['mean_chains']
                    diagnostic['replay_fraction_zero_chain'] = pr['fraction_zero_chain']
                    append_csv(run_dir / 'diagnostics.csv', diagnostic, DIAG_COLUMNS)
            elif is_end and probes:
                # No separate evaluation configured; still create an end probe.
                diagnostic, probe_previous = measure_probes(
                    probes, model, target_model, device, value_cfg,
                    previous=probe_previous, chunk_size=args.infer_chunk)
                diagnostic.update({'episode': episode, 'gradient_updates': gradient_updates})
                pr = replay.chain_distribution()
                for field, key in (
                    ('replay_priority_mean', 'priority_mean'),
                    ('replay_priority_p95', 'priority_p95'),
                    ('replay_priority_max', 'priority_max'),
                    ('replay_fraction_chain_ge_5', 'fraction_chain_ge_5'),
                    ('replay_sample_prob_chain_ge_5', 'sample_prob_chain_ge_5'),
                    ('replay_mean_chains', 'mean_chains'),
                    ('replay_fraction_zero_chain', 'fraction_zero_chain'),
                ):
                    diagnostic[field] = pr[key]
                append_csv(run_dir / 'diagnostics.csv', diagnostic, DIAG_COLUMNS)

            if is_end or (args.resume_every and episode % args.resume_every == 0):
                full_save(episode)
        # Save a portable final snapshot if the caller requested a legacy path.
        if args.out is not None:
            save_model(args.out, model, last_episode, env_steps,
                       gradient_updates, elapsed(), run_id, model_cfg, value_cfg)
        final = {
            'run_id': run_id, 'run_dir': str(run_dir),
            'start_from_episode': start_episode, 'end_episode': last_episode,
            'env_steps': env_steps, 'gradient_updates': gradient_updates,
            'elapsed_seconds_including_resume_ancestor': elapsed(),
            'best_eval_mean_score_this_segment': best_score,
            'per_mode': args.per_mode,
        }
        write_json(run_dir / 'completed.json', final)
        print('TRAINING COMPLETED ' + json.dumps(final, ensure_ascii=False), flush=True)
        return final
    finally:
        if move_stream is not None:
            move_stream.close()


def main():
    train(parse_args())

if __name__ == '__main__':
    if sys.argv[1:2] == ['evaluate']:
        evaluation_cli()
    else:
        main()
