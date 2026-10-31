"""One-step off-policy TD target for scalar afterstate values."""

import numpy as np

from .policy import predict_values


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

