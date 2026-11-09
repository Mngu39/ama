"""Compare completed GTR n-step training runs; do not conflate train and evaluation scores.

Example:
python analysis/compare_runs.py --runs runs/n1 runs/n3 runs/n5 --out comparison
"""
from pathlib import Path
import argparse
import json
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_data(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf8'))
    train = pd.read_csv(run_dir / 'train_episodes.csv')
    if train.empty:
        raise ValueError(f'No training data at {run_dir}')
    eval_path = run_dir / 'eval_metrics.csv'
    evaluation = pd.read_csv(eval_path) if eval_path.exists() else None
    if evaluation is not None and evaluation.empty:
        evaluation = None
    step = manifest['train_config'].get('n_step', 1)
    return run_dir, manifest, train, evaluation, step


def plot_series(data, out, column, name, ylabel, rolling=500):
    fig, ax = plt.subplots(figsize=(9, 5))
    used = 0
    for directory, _, train, _, n in data:
        if column not in train or train[column].isna().all():
            continue
        series = train[column].rolling(min(rolling, len(train)), min_periods=1).mean()
        ax.plot(train['episode'], series, label=f'{directory.name} (n={n})')
        used += 1
    if used:
        ax.set(xlabel='Training episode', ylabel=ylabel, title=f'Training: {name}')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(out / f'{name}.png', dpi=160)
    plt.close(fig)


def plot_evaluation(data, out, key, name, ylabel):
    fig, ax = plt.subplots(figsize=(9, 5))
    used = 0
    for directory, _, _, evaluation, n in data:
        if evaluation is None or key not in evaluation:
            continue
        ax.plot(evaluation['episode'], evaluation[key], 'o-', label=f'{directory.name} (n={n})')
        used += 1
    if used:
        ax.set(xlabel='Checkpoint episode', ylabel=ylabel, title=f'Independent 3-ply evaluation: {name}')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(out / f'eval_{name}.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--window', type=int, default=500)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    data = [read_data(p) for p in args.runs]
    reference = data[0][1]['train_config']
    keys_to_match = ('seed', 'channels', 'blocks', 'gamma', 'score_scale', 'form_weight',
                     'gameover_penalty', 'lr', 'batch_size', 'capacity', 'warmup',
                     'update_every', 'target_sync', 'per_alpha', 'per_beta_start',
                     'per_beta_steps', 'per_mode', 'epsilon_start', 'epsilon_final',
                     'epsilon_steps', 'train_ply', 'train_root_keep', 'train_second_keep')
    mismatches = []
    for folder, manifest, *_ in data[1:]:
        config = manifest['train_config']
        for key in keys_to_match:
            if config.get(key) != reference.get(key):
                mismatches.append(f'{folder.name}: {key} = {config.get(key)!r} vs {reference.get(key)!r}')
                                           
    e0 = data[0][1]['evaluation_config']
    for folder, manifest, *_ in data[1:]:
        if manifest['evaluation_config'] != e0:
            mismatches.append(f'{folder.name}: evaluation_config differs from first run')
    results = []
    for directory, manifest, train, evaluation, n in data:
        last = train.tail(min(args.window, len(train)))
        row = {
            'run_dir': str(directory), 'n_step': n,
            'episodes_completed': int(train['episode'].iloc[-1]),
            'last_window_episodes': len(last),
            'train_mean_score_last_window': float(last['score'].mean()),
            'train_mean_max_chain_last_window': float(last['max_chain'].mean()),
            'train_rate_max_chain_ge5_last_window': float((last['max_chain'] >= 5).mean()),
            'train_rate_max_chain_ge7_last_window': float((last['max_chain'] >= 7).mean()),
            'train_total_seconds': float(train['elapsed_seconds'].iloc[-1]),
            'updates': int(train['gradient_updates'].iloc[-1]),
        }
        if evaluation is not None:
            last_eval = evaluation.iloc[-1]
            row.update({
                'eval_episode': int(last_eval['episode']),
                'eval_3ply_mean_score': float(last_eval['mean_score']),
                'eval_3ply_mean_max_chain': float(last_eval['mean_max_chain']),
                'eval_3ply_rate_ge7': float(last_eval['p_max_chain_ge_7']),
                'eval_wall_time_seconds': float(last_eval['eval_wall_time_seconds']),
            })
        results.append(row)
    summary = pd.DataFrame(results)
    summary.to_csv(args.out / 'nstep_comparison_summary.csv', index=False, encoding='utf-8-sig')
    with (args.out / 'fairness_check.txt').open('w', encoding='utf8') as stream:
        stream.write('Comparison config check (excluding only n_step):\n')
        stream.write('\n'.join(mismatches) if mismatches else 'PASS: compared configs match (except n_step).')
        stream.write('\n\nNOTE: Training averages contain epsilon exploration. Final 3-ply evaluation should be used for policy comparison.\n')
        stream.write('One training seed per method does NOT estimate between-seed variance.\n')
    plots = (('score', 'training_score', 'Score / episode'),
             ('max_chain', 'training_chain', 'Maximum chain / episode'),
             ('avg_abs_td', 'training_abs_td', 'Mean absolute TD error'),
             ('epsilon', 'training_epsilon', 'Exploration rate'))
    for col, name, ylabel in plots:
        plot_series(data, args.out, col, name, ylabel, max(1,args.window))
    plot_evaluation(data, args.out, 'mean_max_chain', 'mean_max_chain', 'Mean maximum chain')
    plot_evaluation(data, args.out, 'mean_score', 'mean_score', 'Mean score')
    print(summary.to_string(index=False))
    print('Fairness check:', ('PASS' if not mismatches else 'MISMATCHES'), *mismatches, sep='\n')
    print('Saved to',args.out)
    return 0 if not mismatches else 2


if __name__ == '__main__':
    sys.exit(main())
