"""Non-C++ unit / integration tests; run from repository root with pytest."""

import gzip
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

# Train/evaluate imports a compiled extension in normal use. For isolated
# logic tests we substitute ONLY its public API, not the RL algorithms.
if 'rl_env.puyo_cpp' not in sys.modules:
    from .fake_env import FakePuyoEnv
    module = types.ModuleType('rl_env.puyo_cpp')
    module.PuyoEnv = FakePuyoEnv
    parent = types.ModuleType('rl_env')
    parent.__path__ = []
    parent.puyo_cpp = module
    sys.modules['rl_env'] = parent
    sys.modules['rl_env.puyo_cpp'] = module

from .fake_env import FakePuyoEnv
from ..checkpoint import save_resume, load_resume
from ..diagnostics import make_probes, measure_probes
from ..metrics import summarize
from ..network import AfterstateValueNet
from ..replay_buffer import PrioritizedReplay, Transition
from ..search import greedy_search
from ..evaluate import run_evaluation, load_evaluation_model
from ..train import parse_args, train
from ..td import td_targets


class ZeroNet(torch.nn.Module):
    def forward(self, arrays):
        return torch.zeros(len(arrays), device=arrays.device)


def test_3ply_and_branch_integrity():
    env = FakePuyoEnv()
    env.reset(123)
    model = ZeroNet().eval()
    one = greedy_search(env, model, torch.device('cpu'), 1.0, 1.0, -1000, ply=1)
    three = greedy_search(env, model, torch.device('cpu'), 1.0, 1.0, -1000,
                          ply=3, root_keep=2, second_keep=4)
    assert one.action == 0
    assert three.action == 1
    assert three.score == pytest.approx(54.0)
    assert three.evaluated == 2 + 4 + 8
    assert env.turn == 0 and not env.done()
    env.step(0)
    env.step(0)
    near_end = greedy_search(env, model, torch.device('cpu'), 1.0, 1.0, -1000, ply=3)
    assert near_end.action == 0


def test_metrics_and_raw_records(tmp_path):
    model = ZeroNet().eval()
    out = tmp_path / 'eval'
    result = run_evaluation(model, {'gamma': 1., 'score_scale': 1.,
                                    'gameover_penalty': -1000.},
                            torch.device('cpu'), n_episodes=3, first_seed=1000,
                            ply=3, root_keep=2, second_keep=4,
                            output_dir=out, checkpoint_name='test.pt')
    assert result['n_episodes'] == 3
    assert result['mean_max_chain'] == 5.0
    assert result['p_max_chain_ge_5'] == 1.0
    assert result['mean_score'] == 54.0
    assert (out / 'episodes.csv').exists()
    assert (out / 'summary.json').exists()
    with gzip.open(out / 'moves.csv.gz', 'rt') as stream:
        lines = stream.readlines()
    assert len(lines) == 1 + 3 * 3
    assert 'visible_next2' in lines[0] and 'board_after' in lines[0]
    assert summarize([{'total_score': 5, 'max_chain': 2, 'turns': 3,
                       'terminated': False, 'truncated': True}], [])['mean_score'] == 5.


def test_per_and_safe_resume(tmp_path):
    state = np.zeros((5, 14, 6), np.uint8)
    terminal = Transition(state, np.empty((0, 5, 14, 6), np.uint8),
                          np.empty(0, np.float32), np.empty(0, np.bool_), True)
    for mode in ('legacy', 'replacement'):
        replay = PrioritizedReplay(8, 0.6, seed=2, mode=mode)
        for _ in range(6):
            replay.add(terminal)
        replay.update_priorities([0, 1], [100., 1.])
        before = replay.state_dict()
        new = PrioritizedReplay(8, 0.6, 99, mode=mode)
        new.load_state_dict(before)
        assert len(new) == 6
        a, ia, wa = replay.sample(4, 0.5)
        b, ib, wb = new.sample(4, 0.5)
        np.testing.assert_array_equal(ia, ib)
        np.testing.assert_allclose(wa, wb)
        assert wa.max() <= 1.00001
        if mode == 'replacement':
            replay.update_priorities([2, 2], [0.1, 5.])
            assert replay.priorities[2] == pytest.approx(5.001)

    model = AfterstateValueNet(5, 8, 1)
    target = AfterstateValueNet(5, 8, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    policy_rng = np.random.default_rng(20)
    cfg = {'seed': 2, 'per_mode': 'legacy'}
    path = tmp_path / 'full.pt'
    save_resume(path, model, target, optimizer, replay, policy_rng, 3, 6, 2,
                1.5, 4.0, 'unit', {'channels': 8, 'blocks': 1,
                                  'input_channels': 5},
                {'gamma': .995, 'score_scale': 1000,
                 'gameover_penalty': -1000}, cfg)
    m2 = AfterstateValueNet(5, 8, 1)
    t2 = AfterstateValueNet(5, 8, 1)
    opt2 = torch.optim.Adam(m2.parameters(), lr=1e-4)
    # saved 'replay' is now from replacement mode in the loop above
    r2 = PrioritizedReplay(8, .6, 123, mode='replacement')
    loaded = load_resume(path, m2, t2, opt2, r2, np.random.default_rng(9), cfg)
    assert loaded['episode'] == 3
    assert len(r2) == 6
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), m2.parameters()))


def test_bn_diagnostics_does_not_change_bn_buffers():
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    env_class = FakePuyoEnv
    model = AfterstateValueNet(5, 8, 1)
    target = AfterstateValueNet(5, 8, 1)
    target.load_state_dict(model.state_dict())
    model.train()
    before = {k: v.clone() for k, v in model.named_buffers()}
    probes = make_probes(env_class, episodes=2, per_episode=2)
    metric, current = measure_probes(probes, model, target, torch.device('cpu'),
                                     {'gamma': .995, 'score_scale': 1000.,
                                      'gameover_penalty': -1000.})
    assert metric['probe_count'] == 4
    assert all(torch.equal(before[k], v) for k, v in model.named_buffers())
    assert model.training
    newer, _ = measure_probes(probes, model, target, torch.device('cpu'),
                              {'gamma': .995, 'score_scale': 1000.,
                               'gameover_penalty': -1000.}, previous=current)
    assert newer['probe_action_flip_since_prev'] == 0


def test_training_smoke_and_resume(tmp_path):
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    run1 = tmp_path / 'first'
    args = parse_args([
        '--run-dir', str(run1), '--episodes', '3', '--seed', '12',
        '--channels', '8', '--blocks', '1', '--warmup', '2',
        '--batch-size', '2', '--capacity', '20', '--update-every', '2',
        '--target-sync', '2', '--save-every', '2', '--resume-every', '2',
        '--eval-every', '2', '--eval-episodes', '2', '--eval-seed', '1000',
        '--probe-seed', '2000', '--print-every', '1',
        '--log-update-every', '1', '--device', 'cpu',
    ])
    result = train(args)
    assert result['end_episode'] == 3
    assert (run1 / 'last_resume.pt').exists()
    assert (run1 / 'ep_000003.pt').exists()
    assert (run1 / 'best.pt').exists()
    assert (run1 / 'train_moves.csv.gz').exists()
    assert (run1 / 'diagnostics.csv').exists()
    model, value, chk = load_evaluation_model(run1 / 'last.pt', torch.device('cpu'))
    assert chk['episode'] == 3

    run2 = tmp_path / 'continued'
    args2 = parse_args([
        '--run-dir', str(run2), '--resume', str(run1 / 'last_resume.pt'),
        '--episodes', '5', '--seed', '12',
        '--channels', '8', '--blocks', '1', '--warmup', '2',
        '--batch-size', '2', '--capacity', '20', '--update-every', '2',
        '--target-sync', '2', '--save-every', '2', '--resume-every', '2',
        '--eval-every', '2', '--eval-episodes', '2', '--eval-seed', '1000',
        '--probe-seed', '2000', '--print-every', '1',
        '--log-update-every', '1', '--device', 'cpu',
    ])
    final = train(args2)
    assert final['start_from_episode'] == 3
    assert final['end_episode'] == 5


def test_3ply_actor_training_smoke_and_exact_resume(tmp_path):
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    common = [
        '--seed', '15', '--channels', '8', '--blocks', '1',
        '--warmup', '2', '--batch-size', '2', '--capacity', '20',
        '--update-every', '2', '--target-sync', '2',
        '--train-ply', '3', '--train-root-keep', '2',
        '--train-second-keep', '3',
        '--save-every', '0', '--resume-every', '1',
        '--eval-every', '0', '--eval-episodes', '1',
        '--eval-seed', '1000', '--probe-seed', '2000',
        '--print-every', '10', '--device', 'cpu', '--no-probes',
    ]
    uninterrupted = tmp_path / 'uninterrupted'
    train(parse_args(common + ['--run-dir', str(uninterrupted),
                              '--episodes', '4']))
    partial = tmp_path / 'partial'
    train(parse_args(common + ['--run-dir', str(partial),
                              '--episodes', '2']))
    resumed = tmp_path / 'resumed'
    train(parse_args(common + ['--run-dir', str(resumed),
                              '--episodes', '4', '--resume',
                              str(partial / 'last_resume.pt')]))
    c1 = torch.load(uninterrupted / 'last.pt', weights_only=True)
    c2 = torch.load(resumed / 'last.pt', weights_only=True)
    assert c1['env_steps'] == c2['env_steps']
    assert c1['gradient_updates'] == c2['gradient_updates']
    for key in c1['model']:
        torch.testing.assert_close(c1['model'][key], c2['model'][key], rtol=0, atol=0)


def test_between_training_run_aggregation(tmp_path):
    import csv
    from ..aggregate_runs import aggregate
    runs = []
    for i, scores in ((0, (4.0, 5.0)), (1, (3.0, 6.0))):
        root = tmp_path / f'seed{i}'
        root.mkdir()
        cfg = {'seed': 10 + i, 'per_mode': 'replacement', 'train_ply': 1}
        manifest = {
            'run_id': f'run{i}', 'start_from_episode': 0,
            'train_config': cfg, 'evaluation_config': {'eval_seed': 100000},
        }
        (root / 'manifest.json').write_text(json.dumps(manifest))
        with (root / 'eval_metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['episode', 'mean_max_chain',
                'mean_score', 'p_max_chain_ge_5', 'gameover_rate',
                'mean_chain_per_activation'])
            writer.writeheader()
            for ep, v in zip((1000, 2000), scores):
                writer.writerow({'episode': ep, 'mean_max_chain': v,
                    'mean_score': v * 200, 'p_max_chain_ge_5': .5,
                    'gameover_rate': 0, 'mean_chain_per_activation': 3.0})
        runs.append(root)
    report = aggregate(runs, tmp_path / 'result')
    assert report['number_of_training_runs'] == 2
    assert report['last_common_evaluation_episode'] == 2000
    assert report['last_common_evaluation_stats']['mean_max_chain_mean'] == 5.5
    assert report['last_common_evaluation_stats']['mean_max_chain_sd'] == pytest.approx(2 ** -.5)
