"""Unmodified original definitions moved from puyo_train.py; imports are wiring only."""

import torch
import numpy as np

from .model import CHANNELS, COLOR_CHANNEL, HEIGHT, WIDTH

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
