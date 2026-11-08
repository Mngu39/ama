"""Portable, lossless CSV and JSON output for independent evaluations."""

import csv
import gzip
import json
from pathlib import Path


EPISODE_COLUMNS = [
    'run_id', 'checkpoint', 'ply', 'episode', 'seed', 'total_score', 'max_chain',
    'turns', 'terminated', 'truncated', 'activations', 'simulated_candidates',
    'evaluated_candidates', 'decision_time_seconds',
]
MOVE_COLUMNS = [
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
    # mtime=0 would make the gzip bytes reproducible, but gzip.open does not
    # expose this argument on all Python releases; content is reproducible.
    with gzip.open(path, 'wt', newline='', encoding='utf-8', compresslevel=6) as stream:
        writer = csv.DictWriter(stream, fieldnames=MOVE_COLUMNS)
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


def append_csv(path, row, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow(row)
