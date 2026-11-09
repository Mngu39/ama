"""Unit/regression checks; run: python -m pytest -q tests/test_nstep.py"""
import numpy as np
import torch
import pytest
from puyo_train import (
    NStepAssembler, PrioritizedReplay, Transition, td_targets,
    CHANNELS, HEIGHT, WIDTH,
)


class ConstantValue(torch.nn.Module):
    def forward(self, x):
        return x.new_full((len(x),), 2.0)


def state(value=0):
    return np.full((CHANNELS, HEIGHT, WIDTH), value, np.uint8)


def choices():
    return np.stack([state(0), state(1)]), np.array([4.0, 3.0], dtype=np.float32), np.array([False, True])


def feed(assembler, i, reward, terminal=False):
    ns, nr, nd = choices()
    return assembler.observe(state(i), reward, terminal, i,
                            ns if not terminal else ns[:0],
                            nr if not terminal else nr[:0],
                            nd if not terminal else nd[:0])


def target(items):
    return td_targets(items, ConstantValue(), torch.device('cpu'), .9, 128)


def test_one_step_matches_original_candidate_target():
    assembler = NStepAssembler(1)
    items = feed(assembler, 0, 123.0)
    assert len(items) == 1
    assert len(items[0].future_rewards) == 0
                                                                  
    assert target(items)[0] == pytest.approx(5.8)
    assert not assembler.pending


@pytest.mark.parametrize('n_step', [1, 3, 5])
def test_horizon_discount_and_bootstrap(n_step):
    assembler = NStepAssembler(n_step)
    all_out = []
    for i in range(n_step):
        all_out.extend(feed(assembler, i, i + 1))
    assert len(all_out) == 1
    first = all_out[0]
    assert len(first.future_rewards) == n_step - 1
    expected = sum(.9 ** j * (j + 2) for j in range(n_step - 1)) + .9 ** (n_step - 1) * 5.8
    assert target([first])[0] == pytest.approx(expected, rel=1e-6)


def test_terminal_flushes_partial_episode_and_never_bootstraps():
    assembler = NStepAssembler(5)
    assert feed(assembler, 0, 999.) == []
    assert feed(assembler, 1, 1.) == []
    out = feed(assembler, 2, 7., terminal=True)
    assert len(out) == 3
    assert not assembler.pending
    assert [len(x.future_rewards) for x in out] == [2, 1, 0]
    assert list(target(out)) == pytest.approx([1 + .9 * 7, 7., 0.])


def test_early_terminal_candidate_does_not_bootstrap():
    ns, nr, nd = choices()
    t = Transition(state(), ns, nr, nd, False, 0)
    assert target([t])[0] == pytest.approx(5.8)
                                                                                      
    t2 = Transition(state(), ns, nr, np.ones(2,dtype=np.bool_), False, 0)
    assert target([t2])[0] == pytest.approx(4.)


def test_prioritized_replay_roundtrip_preserves_nstep_rewards():
    assembler = NStepAssembler(3)
    items = []
    for i in range(4):
        items.extend(feed(assembler, i, i + 1, terminal=(i==3)))
    assert len(items) == 4
    replay = PrioritizedReplay(10, 0.6, 42, mode='replacement')
    for t in items:
        replay.add(t)
    packed = replay.state_dict()
    restored = PrioritizedReplay(10, 0.6, 42, mode='replacement')
    restored.load_state_dict(packed)
    assert len(restored) == 4
    for t1, t2 in zip(replay.data, restored.data):
        np.testing.assert_array_equal(t1.future_rewards, t2.future_rewards)
        np.testing.assert_array_equal(t1.next_states, t2.next_states)
        assert t1.terminal == t2.terminal
    np.testing.assert_allclose(target(replay.data), target(restored.data), rtol=1e-6)


def test_bad_n_step_rejected():
    with pytest.raises(ValueError):
        NStepAssembler(0)
