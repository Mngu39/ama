"""Epsilon-greedy action selection via candidate afterstates.

At a decision state s, choose argmax_a [r(s,a) + gamma V(x(s,a))].
Value V excludes the score earned by the CURRENT placement.
"""

import numpy as np
import torch
from .features import encode_afterstate


def encode_candidates(candidates):
    if not candidates:
        raise ValueError("No legal candidates")
    return np.asarray(
        [encode_afterstate(c["observation"]) for c in candidates],
        dtype=np.uint8,
    )


def scaled_immediate_rewards(candidates, scale: float, gameover_penalty: float):
    return np.asarray([
        (float(c["score"]) + (gameover_penalty if c["terminated"] else 0.0)) / scale
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
