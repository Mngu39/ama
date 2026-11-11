"""Unmodified original definitions moved from puyo_train.py; imports are wiring only."""

import numpy as np
import csv
import gzip
import json
from pathlib import Path
from collections import defaultdict
from math import sqrt


EPISODE_COLUMNS = [
    'run_id', 'checkpoint', 'ply', 'episode', 'seed', 'total_score', 'max_chain',
    'turns', 'terminated', 'truncated', 'activations', 'simulated_candidates',
    'evaluated_candidates', 'decision_time_seconds',
]


EVAL_MOVE_COLUMNS = [
    'run_id', 'checkpoint', 'ply', 'episode', 'seed', 'turn', 'action',
    'visible_current', 'visible_next', 'visible_next2',
    'chains', 'score', 'cumulative_score', 'max_chain_so_far',
    'terminated', 'truncated', 'search_value_scaled', 'simulated_candidates',
    'evaluated_candidates', 'decision_time_seconds', 'board_before',
    'row14_before', 'board_after', 'row14_after',
]


def board_string(observation):
    """Lower row first; one slash-separated row per height, reversible."""
    return '/'.join(observation['board'])


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_move_csv_gz(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
                                                                            
                                                                           
    with gzip.open(path, 'wt', newline='', encoding='utf-8', compresslevel=6) as stream:
        writer = csv.DictWriter(stream, fieldnames=EVAL_MOVE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


class EpisodeResults:
    """Per-run statistics, persisted after each completed training or evaluation game."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.counts = defaultdict(int)
        self.games = 0
        self.total_score = 0
        self.total_max_chain = 0

    def record(self, episode, seed, score, max_chain):
        append_csv(self.directory / 'episode_max_chains.csv',
                   dict(episode=episode, seed=seed, score=score, max_chain=max_chain),
                   ['episode', 'seed', 'score', 'max_chain'])
        self.games += 1
        self.total_score += score
        self.total_max_chain += max_chain
        self.counts[max_chain] += 1
        write_json(self.directory / 'episode_statistics.json', {
            'episodes': self.games,
            'mean_score': self.total_score / self.games,
            'mean_max_chain': self.total_max_chain / self.games,
            'max_chain_distribution': [
                {'max_chain': chain, 'games': count, 'percent': 100 * count / self.games}
                for chain, count in sorted(self.counts.items())
            ],
        })


def append_csv(path, row, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow(row)


THRESHOLDS = (3, 5, 7, 10)


def summarize(episodes, moves):
    if not episodes:
        raise ValueError('At least one episode is needed')
    n = len(episodes)
    scores = np.asarray([e['total_score'] for e in episodes], dtype=np.float64)
    max_chains = np.asarray([e['max_chain'] for e in episodes], dtype=np.float64)
    turns = np.asarray([e['turns'] for e in episodes], dtype=np.float64)
    positive = np.asarray([m['chains'] for m in moves if m['chains'] > 0], dtype=np.float64)
    chains_per_game = defaultdict(int)
    for m in moves:
        if m['chains'] > 0:
            chains_per_game[int(m['episode'])] += 1
    output = {
        'n_episodes': n,
        'mean_max_chain': float(max_chains.mean()),
        'std_max_chain_between_games': float(max_chains.std(ddof=1)) if n > 1 else None,
        'se_mean_max_chain_between_games': float(max_chains.std(ddof=1) / sqrt(n)) if n > 1 else None,
        'median_max_chain': float(np.median(max_chains)),
        'p10_max_chain': float(np.percentile(max_chains, 10)),
        'p90_max_chain': float(np.percentile(max_chains, 90)),
        'mean_score': float(scores.mean()),
        'std_score_between_games': float(scores.std(ddof=1)) if n > 1 else None,
        'median_score': float(np.median(scores)),
        'mean_turns': float(turns.mean()),
        'gameovers': int(sum(bool(e['terminated']) for e in episodes)),
        'turn_limits': int(sum(bool(e['truncated']) for e in episodes)),
        'gameover_rate': float(sum(bool(e['terminated']) for e in episodes) / n),
        'mean_chain_per_activation': float(positive.mean()) if len(positive) else 0.0,
        'chain_activation_count': int(len(positive)),
        'chain_activations_per_game': float(len(positive) / n),
        'max_chain_histogram': {str(int(k)): int(v) for k, v in
                                zip(*np.unique(max_chains.astype(np.int64), return_counts=True))},
    }
    for threshold in THRESHOLDS:
        output[f'p_max_chain_ge_{threshold}'] = float(np.mean(max_chains >= threshold))
        output[f'mean_activations_ge_{threshold}_per_game'] = (
            sum(int(m['chains'] >= threshold) for m in moves) / n)
                                                                              
                                                             
    output['uncertainty_note'] = ('Episode SD/SE condition on one checkpoint; '
                                  'independent training runs are needed to estimate training variance.')
    return output
