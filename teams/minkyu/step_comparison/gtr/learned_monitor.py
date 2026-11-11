"""Read-only, fixed-reference monitoring for an online/frozen learned pair generator.

The reference uses *newly revealed* public engine observations from a separate,
fixed seed range. No reference labels are used for training or virtual rollout.
True Ama conditional P(next pair | turn, visible triple) is NOT generally known;
KL/TV here compare *position-marginal* probabilities, not that conditional.
"""
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .learned_generator import index_pair
from .learned_train import examples_from_engine

PAIRS = tuple(index_pair(i) for i in range(16))
PROBE_POSITIONS = (0, 3, 10, 20, 30, 40)
PROB_COLUMNS = tuple('p_' + pair for pair in PAIRS)
DISTRIBUTION_COLUMNS = (
    'episode', 'observations', 'updates', 'probe_id', 'turn', 'current', 'next',
    'next2', *PROB_COLUMNS,
)
POSITION_COLUMNS = (
    'episode', 'turn', 'reference_samples', 'kl_empirical_ama_to_learned', 'tv',
    *('ama_' + pair for pair in PAIRS),
    *('learned_' + pair for pair in PAIRS),
)
METRIC_COLUMNS = (
    'episode', 'observations', 'updates', 'mode', 'reference_games',
    'reference_samples', 'heldout_nll', 'heldout_accuracy',
    'uniform_nll', 'nll_gain_vs_uniform',
    'position_marginal_kl_ama_to_learned',
    'position_marginal_tv', 'position_marginal_positions',
    'inference_us_per_context',
)


def _append_row(path, columns, row):
    path = Path(path)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class LearnedMonitor:
    def __init__(self, run_dir, *, contexts, labels, probes, seed_start, games, batch_size=8192):
        self.run_dir = Path(run_dir)
        self.x = torch.as_tensor(np.asarray(contexts, dtype=np.int64).copy())
        self.y = torch.as_tensor(np.asarray(labels, dtype=np.int64).copy())
        if self.x.ndim != 2 or self.x.shape[1] != 4 or self.y.shape != (len(self.x),):
            raise ValueError('Expected reference contexts [N,4] and pair labels [N]')
        if len(self.y) == 0:
            raise ValueError('Empty learned reference dataset')
        self.probes = list(probes)
        self.seed_start = int(seed_start)
        self.games = int(games)
        self.batch_size = int(batch_size)
        self.group_index = {}
        for pos in sorted(set(int(x) for x in self.x[:,0].tolist())):
            indices = torch.nonzero(self.x[:,0] == pos).flatten()
            if len(indices) >= 32:  # Exclude unstable marginal estimates.
                self.group_index[pos] = indices
        self.run_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.run_dir / 'learned_monitor_reference.npz',
                            contexts=self.x.numpy(), labels=self.y.numpy())
        (self.run_dir / 'learned_monitor_reference.json').write_text(json.dumps({
            'origin': 'real newly revealed public NEXT2 from Ama engine',
            'seed_start': self.seed_start, 'seed_end': self.seed_start + self.games - 1,
            'games': self.games, 'examples': len(self.y),
            'probes': self.probes,
            'heldout_not_used_for_training': True,
            'position_marginal_kl_tv': (
                'Weighted comparison between empirical Ama P(Z|position) and '
                'average Learned P(Z|position,visible) over survived heldout contexts. '
                'NOT KL/TV to exact Ama P(Z|position,visible); the reference '
                'conditions on observations surviving under the reference policy.'
            ),
            'reference_is_fixed_across_checkpoints': True,
        }, ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def from_engine(cls, run_dir, *, seed_start=700000, games=512, batch_size=8192):
        if games < 2:
            raise ValueError('Need at least two independent reference games')
        rows = []
        by_pos = defaultdict(list)
        for key, turn, visible, label in examples_from_engine(int(games), int(seed_start)):
            from .learned_generator import encode_context, pair_index
            context = encode_context(turn, visible)
            rows.append((context, pair_index(label)))
            if turn in PROBE_POSITIONS and len(by_pos[turn]) < 2:
                by_pos[turn].append((key[1], context))
        if not rows:
            raise ValueError('No observed reference pairs from engine')
        probes = []
        for turn in PROBE_POSITIONS:
            for j, (seed, context) in enumerate(by_pos[turn]):
                probes.append({'id': f't{turn:02d}_{j+1}', 'engine_seed': int(seed),
                               'context': [int(c) for c in context]})
        if not probes:
            raise ValueError('No fixed probe contexts from reference games')
        return cls(run_dir, contexts=[r[0] for r in rows], labels=[r[1] for r in rows],
                   probes=probes, seed_start=seed_start, games=games, batch_size=batch_size)

    def snapshot(self, model, episode, *, mode, observations, updates):
        # Must not modify weights, optimizers, global RNG, or train/eval mode.
        was_training = model.training
        try:
            model.eval()
            with torch.inference_mode():
                total_loss = 0.
                correct = 0
                # Forward once per fixed reference chunk; no stochastic sampling.
                probability_chunks = []
                started = time.perf_counter()
                for begin in range(0, len(self.y), self.batch_size):
                    xb = self.x[begin:begin+self.batch_size]
                    yb = self.y[begin:begin+self.batch_size]
                    logits = model(xb)
                    total_loss += float(F.cross_entropy(logits, yb, reduction='sum'))
                    correct += int((logits.argmax(dim=1) == yb).sum())
                    probability_chunks.append(torch.softmax(logits, dim=-1).cpu())
                elapsed = time.perf_counter() - started
                probabilities = torch.cat(probability_chunks, dim=0)
                nll = total_loss / len(self.y)
                accuracy = correct / len(self.y)
                kl_weighted = tv_weighted = 0.
                count = 0
                for pos, indexes in self.group_index.items():
                    size = len(indexes)
                    true_counts = torch.bincount(self.y[indexes], minlength=16).double()
                    true_p = true_counts / size
                    predicted_p = probabilities[indexes].double().mean(dim=0).clamp_min(1e-15)
                    nonzero = true_p > 0
                    kl = float((true_p[nonzero] * (true_p[nonzero] / predicted_p[nonzero]).log()).sum())
                    tv = float((true_p - predicted_p).abs().sum() * .5)
                    position_row = {'episode': int(episode), 'turn': pos,
                                    'reference_samples': size,
                                    'kl_empirical_ama_to_learned': kl, 'tv': tv}
                    position_row.update(zip(('ama_' + p for p in PAIRS),
                                            (float(v) for v in true_p)))
                    position_row.update(zip(('learned_' + p for p in PAIRS),
                                            (float(v) for v in predicted_p)))
                    _append_row(self.run_dir / 'learned_position_distributions.csv',
                                POSITION_COLUMNS, position_row)
                    kl_weighted += size * kl
                    tv_weighted += size * tv
                    count += size
                pos_kl = kl_weighted / count if count else None
                pos_tv = tv_weighted / count if count else None
                probe_data = torch.tensor([p['context'] for p in self.probes], dtype=torch.long)
                probe_probabilities = torch.softmax(model(probe_data), dim=-1).cpu().numpy()
        finally:
            model.train(was_training)

        metadata = {'episode': int(episode), 'observations': int(observations),
                    'updates': int(updates)}
        for probe, probs in zip(self.probes, probe_probabilities):
            context = probe['context']
            row = {**metadata, 'probe_id': probe['id'], 'turn': context[0],
                   'current': index_pair(context[1]),
                   'next': index_pair(context[2]),
                   'next2': index_pair(context[3])}
            row.update(zip(PROB_COLUMNS, (float(p) for p in probs)))
            _append_row(self.run_dir / 'learned_distribution_snapshots.csv',
                        DISTRIBUTION_COLUMNS, row)
        metrics = {**metadata, 'mode': str(mode),
                   'reference_games': self.games, 'reference_samples': len(self.y),
                   'heldout_nll': nll, 'heldout_accuracy': accuracy,
                   'uniform_nll': float(np.log(16)),
                   'nll_gain_vs_uniform': float(np.log(16)) - nll,
                   'position_marginal_kl_ama_to_learned': pos_kl,
                   'position_marginal_tv': pos_tv,
                   'position_marginal_positions': len(self.group_index),
                   'inference_us_per_context': elapsed * 1e6 / len(self.y)}
        _append_row(self.run_dir / 'learned_distribution_metrics.csv',
                    METRIC_COLUMNS, metrics)
        return metrics
