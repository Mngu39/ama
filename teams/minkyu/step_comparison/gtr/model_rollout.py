"""Construct *training* targets by executing imaginary future placements.

The actual state is the board after the CURRENT placement. Its future targets
count placements t+1,...,t+n. At decision t, CURRENT/NEXT/NEXT2 were visible,
so the first *unknown* pair is t+3 (the third following placement). The queue
fork occurs BEFORE the real current placement, avoiding the observation at
turn t+1 that would reveal one extra pair.

The first n-1 future actions are sampled from the same epsilon-greedy 1-ply
policy as training, using separate deterministic RNG streams. The n-th move
is an action-max (up to 22 candidates), exactly as in the legacy target.
No evaluation ply or search tree width is changed.
"""
from dataclasses import dataclass
import numpy as np

from .actions import (encode_candidates, pick_action, scaled_immediate_rewards,
                      candidate_dones)
from .model import CHANNELS, HEIGHT, WIDTH
from .replay import Transition


def _empty_candidates():
    return (np.empty((0, CHANNELS, HEIGHT, WIDTH), dtype=np.uint8),
            np.empty(0, dtype=np.float32), np.empty(0, dtype=np.bool_))


@dataclass
class VirtualTarget:
    transition: Transition
    first_score: int
    first_board: tuple
    first_hidden_pair: str
    simulated_actions: tuple
    simulated_rewards: tuple
    last_candidates: int
    terminated_before_last: bool


def simulate_target(fork, action, encoded_state, chains, *, n_step, model,
                    device, rng, epsilon, gamma, score_scale, gameover_penalty,
                    infer_chunk):
    """Run on an isolated virtual environment whose hidden queue was replaced.

    No real future transitions are passed to this function. Returns the same
    Transition structure consumed by the unmodified td_targets()/PER logic.
    """
    if n_step < 1:
        raise ValueError('n_step must be positive')
    first = fork.step(int(action))
    first_done = bool(first['terminated'] or first['truncated'])
    state = np.asarray(encoded_state, dtype=np.uint8)
    future_rewards = []
    actions = []
    if first_done:
        next_states, next_rewards, next_dones = _empty_candidates()
    else:
        for _ in range(n_step - 1):
            candidates = fork.preview_all()
            if not candidates:
                raise RuntimeError('Nonterminal fork has no legal candidates')
            index, _ = pick_action(candidates, model, device, rng, epsilon,
                                   gamma, score_scale, gameover_penalty,
                                   infer_chunk)
            chosen = candidates[index]
            actions.append(int(chosen['action']))
            result = fork.step(actions[-1])
            reward = (float(result['score']) + float(result['form_bonus']) +
                      (gameover_penalty if result['terminated'] else 0.0)) / score_scale
            future_rewards.append(float(reward))
            if result['terminated'] or result['truncated']:
                break
        if fork.done():
            next_states, next_rewards, next_dones = _empty_candidates()
        else:
            last = fork.preview_all()
            if not last:
                raise RuntimeError('Nonterminal fork has no final-step candidates')
            next_states = encode_candidates(last)
            next_rewards = scaled_immediate_rewards(last, score_scale, gameover_penalty)
            next_dones = candidate_dones(last)
    # In the existing td_targets(), terminal=True denotes the *initial*
    # value state is terminal. If a virtual future move terminates, retain
    # terminal=False and the truncated future_rewards, with zero next_states.
    transition = Transition(state, next_states, next_rewards, next_dones,
                            first_done, int(chains),
                            np.asarray(future_rewards, dtype=np.float32))
    return VirtualTarget(
        transition=transition, first_score=int(first['score']),
        first_board=tuple(first['observation']['board']),
        first_hidden_pair=''.join(first['observation']['visible'][2]),
        simulated_actions=tuple(actions),
        simulated_rewards=tuple(future_rewards),
        last_candidates=int(len(next_rewards)),
        terminated_before_last=bool(fork.done()),
    )
