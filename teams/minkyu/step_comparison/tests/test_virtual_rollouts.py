"""Regressions for virtual piece generation and n-step training-target assembly."""
import numpy as np
import pytest
import torch

from puyo_train import CHANNELS, HEIGHT, WIDTH, td_targets
from gtr.model_rollout import simulate_target
from gtr.puyo_generators import make_generator, per_placement_rng
from gtr.reward import make_environment
from gtr.actions import encode_candidates


class ConstantValue(torch.nn.Module):
    def forward(self, x):
        return x.new_zeros(len(x))


def test_generator_rng_does_not_consume_actor_rng():
    real_actor = np.random.default_rng(12347)
    reference = np.random.default_rng(12347)
    gen = per_placement_rng(12345, 9, 18, 9001)
    rollout = per_placement_rng(12345, 9, 18, 9002)
    fresh = per_placement_rng(12345, 9, 18, 9001)
    assert [int(gen.integers(100000)) for _ in range(3)] == [
        int(fresh.integers(100000)) for _ in range(3)]
    assert int(rollout.integers(9999)) == int(per_placement_rng(12345, 9, 18, 9002).integers(9999))
    assert int(real_actor.integers(2**32)) == int(reference.integers(2**32))


def test_generator_factory_learned_requires_a_model():
    assert make_generator('observed') is None
    gen = make_generator('learned')
    assert gen.name == 'learned'
    with pytest.raises(ValueError, match='learned generator model'):
        gen.fork(object(), np.random.default_rng(10))


def test_native_engine_fork_and_virtual_target():
    try:
        from rl_env import puyo_cpp
    except ImportError:
        pytest.skip('C++ engine not compiled in this test environment')
    if not hasattr(puyo_cpp.PuyoEnv, 'fork_with_hidden_pairs'):
        pytest.skip('C++ engine requires minimal fork patch')
    env = make_environment(.995, .04, 1000.)
    initial = env.reset(12345)
    roots = env.preview_all()
    action = int(roots[0]['action'])
    state = encode_candidates(roots)[0]
    original = env.observation()
    model = ConstantValue()
    for name in ['uniform', 'rule_based']:
        gen = make_generator(name)
        fork = gen.fork(env, per_placement_rng(12345, 1, 0, 9001))
        assert fork.observation() == original
        virtual = simulate_target(
            fork, action, state, chains=0, n_step=5, model=model,
            device=torch.device('cpu'),
            rng=per_placement_rng(12345, 1, 0, 9002), epsilon=0.4,
            gamma=.995, score_scale=1000., gameover_penalty=-1000.,
            infer_chunk=64)
        np.testing.assert_array_equal(virtual.transition.state, state)
        assert len(virtual.simulated_actions) <= 4
        assert len(virtual.transition.future_rewards) <= 4
        assert env.observation() == original  # no real game advancement
        assert np.isfinite(td_targets([virtual.transition], model,
                                      torch.device('cpu'), .995, 64)[0])


def test_current_action_is_identical_on_virtual_branch():
    try:
        from rl_env import puyo_cpp
    except ImportError:
        pytest.skip('No C++ game engine')
    if not hasattr(puyo_cpp.PuyoEnv, 'fork_with_hidden_rule_seed'):
        pytest.skip('C++ engine requires minimal fork patch')
    env = make_environment(.995, .04, 1000.)
    env.reset(12345)
    roots = env.preview_all()
    action = int(roots[0]['action'])
    state = encode_candidates(roots)[0]
    fork = make_generator('rule_based').fork(env, per_placement_rng(1,1,0,9001))
    original = env.step(action)
    virtual = simulate_target(fork, action, state, 0, n_step=3,
                              model=ConstantValue(), device=torch.device('cpu'),
                              rng=per_placement_rng(1,1,0,9002), epsilon=0.,
                              gamma=.995, score_scale=1000.,
                              gameover_penalty=-1000., infer_chunk=64)
    assert virtual.first_board == tuple(original['observation']['board'])
    assert virtual.first_score == int(original['score'])

@pytest.mark.parametrize('name', ['uniform', 'rule_based'])
def test_real_hidden_queue_cannot_change_virtual_target(name):
    try:
        from rl_env import puyo_cpp
    except ImportError:
        pytest.skip('No compiled C++ engine')
    if not hasattr(puyo_cpp.PuyoEnv, 'fork_with_hidden_pairs'):
        pytest.skip('C++ engine requires minimal fork patch')
    env = make_environment(.995, .04, 1000.)
    env.reset(12345)
    altered_real_hidden = env.fork_with_hidden_pairs(['BB'] * 128)
    assert env.observation() == altered_real_hidden.observation()
    roots = env.preview_all()
    action = int(roots[0]['action'])
    state = encode_candidates(roots)[0]
    def do_fork(parent):
        fork = make_generator(name).fork(parent, per_placement_rng(42,1,0,9001))
        return simulate_target(fork, action, state, 0, n_step=5,
            model=ConstantValue(), device=torch.device('cpu'),
            rng=per_placement_rng(42,1,0,9002), epsilon=0.4,
            gamma=.995, score_scale=1000., gameover_penalty=-1000., infer_chunk=64)
    a, b = do_fork(env), do_fork(altered_real_hidden)
    assert a.first_hidden_pair == b.first_hidden_pair
    assert a.simulated_actions == b.simulated_actions
    assert a.simulated_rewards == b.simulated_rewards
    np.testing.assert_array_equal(a.transition.next_states, b.transition.next_states)
