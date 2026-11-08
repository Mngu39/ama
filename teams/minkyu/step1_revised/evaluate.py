"""Independent greedy evaluation with 1-ply / 3-ply search and raw records."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from rl_env import puyo_cpp
from .metrics import summarize
from .network import AfterstateValueNet
from .recording import (EPISODE_COLUMNS, board_string, write_csv,
                        write_json, write_move_csv_gz)
from .search import greedy_search


def load_evaluation_model(path, device):
    # Supports original 50k checkpoint and new, weights-only snapshots.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = AfterstateValueNet(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model, checkpoint['value_config'], checkpoint


def run_evaluation(model, value_config, device, *, n_episodes=100,
                   first_seed=100000, ply=1, root_keep=5, second_keep=10,
                   chunk_size=128, output_dir=None, checkpoint_name='', run_id='',
                   write_moves=True, progress=False):
    """Return summary and optionally persist episode/move records.

    This function does not modify network weights, BatchNorm running stats,
    or training random-number generators. Each episode starts from its own
    deterministic queue seed; tie-breaks use deterministic action order.
    """
    if n_episodes < 1 or first_seed < 0:
        raise ValueError('n_episodes must be positive and first_seed >= 0')
    was_training = model.training
    model.eval()
    env = puyo_cpp.PuyoEnv()
    episodes, moves = [], []
    start = time.perf_counter()
    try:
        for ep in range(n_episodes):
            seed = first_seed + ep
            env.reset(seed)
            score = longest = turns = activations = 0
            evaluated = simulated = 0
            decision_seconds = 0.0
            ending = None
            while not env.done():
                before = env.observation()
                decision_started = time.perf_counter()
                search = greedy_search(
                    env, model, device,
                    gamma=value_config['gamma'],
                    score_scale=value_config['score_scale'],
                    gameover_penalty=value_config['gameover_penalty'],
                    ply=ply, root_keep=root_keep, second_keep=second_keep,
                    chunk_size=chunk_size,
                )
                elapsed = time.perf_counter() - decision_started
                result = env.step(search.action)
                ending = result
                turns += 1
                score += int(result['score'])
                chains = int(result['chains'])
                longest = max(longest, chains)
                activations += int(chains > 0)
                evaluated += search.evaluated
                simulated += search.simulated
                decision_seconds += elapsed
                if write_moves:
                    shown = before['visible']
                    moves.append({
                        'run_id': run_id, 'checkpoint': checkpoint_name, 'ply': ply,
                        'episode': ep, 'seed': seed, 'turn': turns,
                        'action': search.action,
                        'visible_current': ''.join(shown[0]),
                        'visible_next': ''.join(shown[1]),
                        'visible_next2': ''.join(shown[2]),
                        'chains': chains, 'score': int(result['score']),
                        'cumulative_score': score, 'max_chain_so_far': longest,
                        'terminated': bool(result['terminated']),
                        'truncated': bool(result['truncated']),
                        'search_value_scaled': search.score,
                        'simulated_candidates': search.simulated,
                        'evaluated_candidates': search.evaluated,
                        'decision_time_seconds': elapsed,
                        'board_before': board_string(before),
                        'row14_before': int(before['row14']),
                        'board_after': board_string(result['observation']),
                        'row14_after': int(result['observation']['row14']),
                    })
            episodes.append({
                'run_id': run_id, 'checkpoint': checkpoint_name, 'ply': ply,
                'episode': ep, 'seed': seed, 'total_score': score,
                'max_chain': longest, 'turns': turns,
                'terminated': bool(ending['terminated']),
                'truncated': bool(ending['truncated']), 'activations': activations,
                'simulated_candidates': simulated, 'evaluated_candidates': evaluated,
                'decision_time_seconds': decision_seconds,
            })
            if progress and (ep + 1) % max(1, n_episodes // 10) == 0:
                print(f'eval {ep + 1}/{n_episodes}', flush=True)
    finally:
        model.train(was_training)

    # Compute activation statistics from episode aggregates when move logging
    # is switched off; full detailed statistics require the move records.
    if write_moves:
        summary = summarize(episodes, moves)
    else:
        summary = summarize(episodes, [])
        summary['chain_activation_count'] = sum(e['activations'] for e in episodes)
        summary['chain_activations_per_game'] = summary['chain_activation_count'] / n_episodes
        summary['mean_chain_per_activation'] = None
        for k in (3, 5, 7, 10):
            summary[f'mean_activations_ge_{k}_per_game'] = None
    summary.update({
        'ply': ply, 'first_seed': first_seed, 'last_seed': first_seed + n_episodes - 1,
        'run_id': run_id, 'checkpoint': checkpoint_name,
        'eval_wall_time_seconds': time.perf_counter() - start,
        'total_simulated_candidates': int(sum(e['simulated_candidates'] for e in episodes)),
        'total_evaluated_candidates': int(sum(e['evaluated_candidates'] for e in episodes)),
        'avg_decision_time_seconds': float(sum(e['decision_time_seconds'] for e in episodes) /
                                            max(1, sum(e['turns'] for e in episodes))),
        'root_keep': root_keep if ply == 3 else None,
        'second_keep': second_keep if ply == 3 else None,
    })
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=False)
        write_csv(path / 'episodes.csv', episodes, EPISODE_COLUMNS)
        if write_moves:
            write_move_csv_gz(path / 'moves.csv.gz', moves)
        write_json(path / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description='Independent greedy 1-ply / 3-ply evaluation')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=100000)
    p.add_argument('--ply', type=int, choices=[1, 3], default=1)
    p.add_argument('--root-keep', type=int, default=5)
    p.add_argument('--second-keep', type=int, default=10)
    p.add_argument('--infer-chunk', type=int, default=128)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--out-dir', type=Path, default=None)
    p.add_argument('--summary-only', action='store_true',
                   help='Do not save per-placement CSV; loses chain-per-event metrics')
    args = p.parse_args()
    from .train import select_device
    device = select_device(args.device)
    model, value_config, checkpoint = load_evaluation_model(args.checkpoint, device)
    path = args.out_dir
    if path is None:
        path = args.checkpoint.parent / 'evaluations' / (
            f'{args.checkpoint.stem}_ply{args.ply}_seed{args.seed}_n{args.episodes}')
    result = run_evaluation(
        model, value_config, device, n_episodes=args.episodes,
        first_seed=args.seed, ply=args.ply, root_keep=args.root_keep,
        second_keep=args.second_keep, chunk_size=args.infer_chunk,
        output_dir=path, checkpoint_name=str(args.checkpoint),
        run_id=str(checkpoint.get('run_id', 'legacy')),
        write_moves=not args.summary_only, progress=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(f'files={path.resolve()}')


if __name__ == '__main__':
    main()
