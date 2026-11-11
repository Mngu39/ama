import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gtr.learned_generator import PairNet
from gtr.learned_monitor import LearnedMonitor, PROB_COLUMNS, PAIRS


def _fixture(tmp_path, n=64):
    # 64 independent deterministic reference observations per position.
    contexts = []
    labels = []
    for pos in (0, 3, 10, 20):
        for i in range(n):
            contexts.append([pos, (i + 1) % 16, (i + 2) % 16, (i + 3) % 16])
            labels.append((i + pos) % 16)
    probes = [{'id':f't{pos:02d}_1', 'engine_seed':700000,
               'context': [pos, 1, 2, 3]} for pos in (0, 3, 10, 20)]
    return LearnedMonitor(tmp_path, contexts=contexts, labels=labels,
                          probes=probes, seed_start=700000, games=64, batch_size=128)


def _csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def test_initial_final_snapshots_are_valid_and_read_only(tmp_path):
    monitor = _fixture(tmp_path)
    model = PairNet()
    model.train()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    torch_before = torch.get_rng_state().clone()
    np_before = np.random.get_state()
    first = monitor.snapshot(model, 0, mode='online', observations=0, updates=0)
    assert model.training
    assert all(torch.equal(model.state_dict()[k], v) for k,v in state.items())
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert np.array_equal(np.random.get_state()[1], np_before[1])
    assert first['heldout_nll'] == pytest.approx(np.log(16), abs=1e-6)
    assert first['position_marginal_tv'] == pytest.approx(0, abs=1e-6)
    assert first['position_marginal_kl_ama_to_learned'] == pytest.approx(0, abs=1e-6)
    with torch.no_grad():
        model.position_logits.weight[0, 0] = 2.0
    second = monitor.snapshot(model, 500, mode='online', observations=23456, updates=2932)
    assert second['heldout_nll'] != pytest.approx(first['heldout_nll'])
    saved = _csv(tmp_path / 'learned_distribution_snapshots.csv')
    assert len(saved) == 2*4
    assert saved[0]['episode'] == '0' and saved[4]['episode'] == '500'
    assert sum(float(saved[0][c]) for c in PROB_COLUMNS) == pytest.approx(1)
    assert all(float(saved[0][c]) == pytest.approx(1/16) for c in PROB_COLUMNS)
    assert float(saved[4]['p_RR']) > float(saved[0]['p_RR'])
    positions = _csv(tmp_path / 'learned_position_distributions.csv')
    assert len(positions) == 2*4
    assert float(positions[0]['ama_RR']) == pytest.approx(1/16)
    assert float(positions[0]['learned_RR']) == pytest.approx(1/16)
    metrics = _csv(tmp_path / 'learned_distribution_metrics.csv')
    assert len(metrics) == 2 and metrics[0]['episode'] == '0' and metrics[1]['episode'] == '500'
    assert float(metrics[0]['uniform_nll']) == pytest.approx(np.log(16))
    meta = json.loads((tmp_path / 'learned_monitor_reference.json').read_text())
    assert meta['heldout_not_used_for_training']
    npz = np.load(tmp_path / 'learned_monitor_reference.npz')
    assert len(npz['contexts']) == 256 and len(npz['labels']) == 256


def test_probe_selection_and_engine_independence(monkeypatch, tmp_path):
    from gtr import learned_monitor
    calls = []
    def fake_engine(games, start):
        calls.append((games,start))
        for seed in range(start,start+games):
            for t in (0, 3, 10):
                yield (('engine',seed), t, ['RY','GB','BB'], 'RG')
    monkeypatch.setattr(learned_monitor, 'examples_from_engine', fake_engine)
    m = LearnedMonitor.from_engine(tmp_path, seed_start=700000, games=40)
    assert calls == [(40,700000)]
    assert {p['id'] for p in m.probes} == {
        't00_1','t00_2','t03_1','t03_2','t10_1','t10_2'}
    assert all(p['engine_seed'] in (700000,700001) for p in m.probes)
    assert m.snapshot(PairNet(), 0, mode='online', observations=0, updates=0)['reference_samples'] == 120


def test_missing_sparse_positions_do_not_crash(tmp_path):
    m = LearnedMonitor(tmp_path, contexts=[[0,1,2,3]]*4,labels=[1]*4,
                       probes=[{'id':'probe','context':[0,1,2,3]}],
                       seed_start=700000,games=2)
    result = m.snapshot(PairNet(), 0, mode='frozen', observations=0, updates=0)
    assert result['position_marginal_tv'] is None
    assert result['position_marginal_positions'] == 0


def test_backfill_uses_existing_checkpoints_without_editing_them(monkeypatch, tmp_path):
    from gtr import learned_monitor_backfill
    from gtr.learned_generator import save_learned
    run=tmp_path/'run'
    run.mkdir()
    (run/'manifest.json').write_text(json.dumps({'learned_generator':{
        'mode':'online','pretrained_checkpoint':None}}))
    m1=PairNet()
    save_learned(run/'learned_ep_000500.pt',m1,
                 {'episode':500,'mode':'online','observations':1000,'updates':100})
    with torch.no_grad():
        m1.position_logits.weight[20,7]=1
    save_learned(run/'learned_ep_001000.pt',m1,
                 {'episode':1000,'mode':'online','observations':2000,'updates':200})
    original=[(p.name,p.read_bytes()) for p in run.glob('*.pt')]
    # Avoid depending on C++ extension in this test.
    mock_monitor=_fixture(tmp_path/'from_ref')
    monkeypatch.setattr(learned_monitor_backfill.LearnedMonitor, 'from_engine',
                        lambda out, **kw: _fixture(out))
    result=learned_monitor_backfill.backfill(run,output_dir=tmp_path/'analyzed',reference_games=64)
    rows=_csv(result/'learned_distribution_metrics.csv')
    assert [int(r['episode']) for r in rows] == [0,500,1000]
    assert [(p.name,p.read_bytes()) for p in run.glob('*.pt')] == original
    assert json.loads((result/'backfill_provenance.json').read_text())['checkpoints_examined']==2
