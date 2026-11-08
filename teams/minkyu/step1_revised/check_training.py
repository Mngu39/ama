"""Lightweight unit check for network, PER and one-step TD target.

No C++ environment import needed. This checks code paths before a real run.
"""

import numpy as np
import torch
from .network import AfterstateValueNet
from .replay_buffer import PrioritizedReplay, Transition
from .td import td_targets


def main():
    torch.set_num_threads(min(2, torch.get_num_threads()))
    device = torch.device("cpu")
    model = AfterstateValueNet(input_channels=5, channels=16, blocks=1).to(device)
    model.eval()
    states = np.zeros((2, 5, 14, 6), dtype=np.uint8)
    with torch.no_grad():
        values = model(torch.tensor(states, dtype=torch.float32))
    assert values.shape == (2,)
    terminal = Transition(states[0], states[:0], np.empty(0, dtype=np.float32),
                          np.empty(0, dtype=np.bool_), True)
    continuing = Transition(states[1], states,
                            np.asarray([1.0, 2.0], dtype=np.float32),
                            np.asarray([True, True]), False)
    targets = td_targets([terminal, continuing], model, device, 0.995, 32)
    assert np.allclose(targets, [0.0, 2.0]), targets

    replay = PrioritizedReplay(10, alpha=0.6, seed=42)
    replay.add(terminal)
    replay.add(continuing)
    experiences, indices, weights = replay.sample(2, beta=0.4)
    assert len(experiences) == len(indices) == len(weights) == 2
    assert np.all(np.isfinite(weights))
    replay.update_priorities(indices, np.asarray([0.5, 1.0]))
    print("ONE-STEP AFTERSTATE LEARNING CHECK PASSED")


if __name__ == "__main__":
    main()
