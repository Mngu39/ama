"""Evaluation metrics computed from episode and move records, not training loss."""

from collections import defaultdict
from math import sqrt
import numpy as np

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
    # These SEs quantify evaluation-game randomness for THESE trained weights.
    # They do NOT quantify between-training-seed uncertainty.
    output['uncertainty_note'] = ('Episode SD/SE condition on one checkpoint; '
                                  'independent training runs are needed to estimate training variance.')
    return output
