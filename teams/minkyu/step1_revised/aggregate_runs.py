"""Aggregate independently trained runs; do NOT treat episodes as seeds.

Usage:
  python -m teams.minkyu.mori25_1step_reproduction.aggregate_runs \
    --runs path/to/seed1 path/to/seed2 path/to/seed3 \
    --out-dir path/to/aggregate

Outputs:
  across_runs.csv  one row per evaluated training episode count
  run_auc.csv      one AUC per independent training run (observed range)
  aggregate.json  summary of final-checkpoint performance and caveats
"""

import argparse
from collections import defaultdict
import csv
import json
from math import sqrt
from pathlib import Path

import numpy as np

from .recording import write_json


FIELDS = ('mean_max_chain', 'mean_score', 'p_max_chain_ge_5',
          'gameover_rate', 'mean_chain_per_activation')


def load_run(path):
    path = Path(path)
    with (path / 'manifest.json').open(encoding='utf-8') as f:
        manifest = json.load(f)
    with (path / 'eval_metrics.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'No evaluated checkpoints: {path}')
    return manifest, rows


def _mean_std_se(vals):
    z = np.asarray(vals, dtype=np.float64)
    n = len(z)
    return (float(z.mean()), float(z.std(ddof=1)) if n >= 2 else None,
            float(z.std(ddof=1) / sqrt(n)) if n >= 2 else None)


def aggregate(paths, output_dir):
    if not paths:
        raise ValueError('At least one run required')
    dataset = []
    ids = []
    for path in paths:
        manifest, rows = load_run(path)
        ids.append(manifest['run_id'])
        if manifest['start_from_episode'] != 0:
            raise ValueError('A resumed segment is not a complete independent run; '
                             'merge its ancestor first or use full original runs')
        dataset.append((manifest, rows))
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate run_id: runs must be independent')
    cfg0 = dataset[0][0]['train_config']
    for manifest, _ in dataset[1:]:
        # The seed is allowed to vary, while algorithm and hyperparameters
        # must otherwise match for a within-algorithm variance estimate.
        for key, val in cfg0.items():
            if key != 'seed' and val != manifest['train_config'].get(key):
                raise ValueError(f'Hyperparameter mismatch across runs: {key}')
        if manifest['evaluation_config'] != dataset[0][0]['evaluation_config']:
            raise ValueError('Evaluation settings differ across runs')

    schedules = [tuple(int(row['episode']) for row in rows)
                 for _, rows in dataset]
    if len(set(schedules)) > 1:
        raise ValueError('Evaluated checkpoint schedules differ between runs; '
                         'align them before comparing AUC or variance')

    by_episode = defaultdict(list)
    auc = []
    for manifest, rows in dataset:
        series = []
        for row in rows:
            ep = int(row['episode'])
            by_episode[ep].append(row)
            series.append((ep, float(row['mean_max_chain'])))
        series.sort()
        if len(series) >= 2:
            x, y = zip(*series)
            auc_val = float(np.trapezoid(y, x) / (x[-1] - x[0]))
        else:
            auc_val = None
        auc.append({'run_id': manifest['run_id'],
                    'seed': manifest['train_config'].get('seed'),
                    'auc_mean_max_chain_observed_range': auc_val,
                    'first_eval_episode': series[0][0],
                    'last_eval_episode': series[-1][0]})
    output_rows = []
    for ep, rows in sorted(by_episode.items()):
        rec = {'episode': ep, 'run_count': len(rows)}
        for field in FIELDS:
            vals = [float(r[field]) for r in rows
                    if r.get(field) not in ('', 'None', 'nan', None)]
            if not vals:
                rec[f'{field}_mean'] = rec[f'{field}_sd'] = rec[f'{field}_se'] = None
                continue
            mu, sd, se = _mean_std_se(vals)
            rec[f'{field}_mean'] = mu
            rec[f'{field}_sd'] = sd
            rec[f'{field}_se'] = se
        output_rows.append(rec)
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=False)
    with (path / 'across_runs.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    with (path / 'run_auc.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(auc[0]))
        writer.writeheader()
        writer.writerows(auc)
    common = set.intersection(*(set(int(row['episode']) for row in rows)
                                for _, rows in dataset))
    last_common = max(common) if common else None
    final = next((r for r in output_rows if r['episode'] == last_common), None)
    report = {
        'number_of_training_runs': len(dataset),
        'last_common_evaluation_episode': last_common,
        'last_common_evaluation_stats': final,
        'uncertainty_note': ('Between-run SD uses truly independent training seeds; '
                             'no CI is meaningful with n=1, and n=3/5 gives very '
                             'imprecise variance estimates. AUC covers only the '
                             'observed checkpoint range.'),
    }
    write_json(path / 'aggregate.json', report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(aggregate(args.runs, args.out_dir), indent=2,
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
