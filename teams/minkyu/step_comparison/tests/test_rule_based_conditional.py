"""Regression tests for the rule_based generator after applying its C++ method patch."""
import numpy as np
import pytest

from gtr.puyo_generators import make_generator, per_placement_rng
from gtr.reward import make_environment


def test_rule_based_preserves_visible_and_real_engine():
    env = make_environment(0.995, 0.04, 1000.0)
    initial = env.reset(12345)
    original_board = initial['board']
    first_three = tuple(initial['visible'])
    fork = make_generator('rule_based').fork(env, per_placement_rng(12345, 1, 0, 9001))
    assert tuple(fork.observation()['visible']) == first_three
    assert env.observation()['board'] == original_board
    assert tuple(env.observation()['visible']) == first_three


def test_rule_based_must_depend_on_visible_triple_not_real_hidden():
    env = make_environment(0.995, 0.04, 1000.0)
    env.reset(12345)
    altered = env.fork_with_hidden_pairs(['BB'] * 128)
    assert altered.observation() == env.observation()
    stream1 = per_placement_rng(12345, 3, 0, 9001)
    stream2 = per_placement_rng(12345, 3, 0, 9001)
    a = make_generator('rule_based').fork(env, stream1)
    b = make_generator('rule_based').fork(altered, stream2)
    for i in range(4):
        assert a.observation() == b.observation()
        if i < 3:
            actions = a.preview_all()
            if not actions:
                break
            action = int(actions[0]['action'])
            a.step(action)
            b.step(action)
