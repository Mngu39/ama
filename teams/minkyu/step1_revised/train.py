"""1-step afterstate TD training, evaluation, diagnostics and reproducible I/O.

A v2 training run starts in its own directory.  The original 50k checkpoint
is supported for evaluation only; it does not contain the optimizer / replay
state required for exact training continuation.
"""

import argparse
import csv
from collections import deque
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from rl_env import puyo_cpp
from .checkpoint import load_resume, save_model, save_resume
from .features import CHANNELS, HEIGHT, WIDTH
from .network import AfterstateValueNet
from .policy import (candidate_dones, encode_candidates, pick_action,
                     scaled_immediate_rewards)
from .recording import append_csv, write_json
from .replay_buffer import PrioritizedReplay, Transition
from .td import td_targets


EP_COLUMNS = [
    'episode', 'seed', 'turns', 'score', 'max_chain', 'activations',
    'activations_ge_3', 'activations_ge_5', 'activations_ge_7',
    'terminated', 'truncated', 'epsilon', 'replay_size', 'env_steps',
    'gradient_updates', 'avg_weighted_huber', 'avg_unweighted_huber',
    'avg_abs_td', 'episode_seconds', 'elapsed_seconds',
]
MOVE_COLUMNS = [
    'episode', 'seed', 'turn', 'action', 'visible_current', 'visible_next',
    'visible_next2', 'chains', 'score', 'cumulative_score',
    'epsilon', 'terminated', 'truncated',
]
EVAL_COLUMNS = [
    'episode', 'env_steps', 'gradient_updates', 'eval_ply', 'eval_episodes',
    'mean_max_chain', 'std_max_chain_between_games',
    'mean_score', 'mean_turns', 'p_max_chain_ge_3', 'p_max_chain_ge_5',
    'p_max_chain_ge_7', 'p_max_chain_ge_10',
    'mean_chain_per_activation', 'chain_activations_per_game',
    'gameover_rate', 'avg_decision_time_seconds', 'eval_wall_time_seconds',
    'new_best',
]
DIAG_COLUMNS = [
    'episode', 'gradient_updates', 'probe_count',
    'probe_bn_eval_train_mae', 'probe_bn_eval_train_signed_bias',
    'probe_bn_action_flip_rate', 'probe_online_target_mae',
    'probe_median_action_margin', 'probe_td_abs_mean', 'probe_td_abs_p95',
    'probe_td_abs_max', 'probe_mean_best_action_value',
    'probe_action_flip_since_prev', 'probe_value_mae_since_prev',
    'replay_priority_mean', 'replay_priority_p95', 'replay_priority_max',
    'replay_fraction_chain_ge_5', 'replay_sample_prob_chain_ge_5',
    'replay_mean_chains', 'replay_fraction_zero_chain',
]
UPDATE_COLUMNS = [
    'episode', 'env_steps', 'gradient_updates', 'weighted_huber',
    'unweighted_huber', 'abs_td_mean', 'abs_td_p95', 'pred_value_mean',
    'target_value_mean', 'grad_norm_preclip', 'beta',
]


def select_device(requested):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; install a compatible PyTorch CUDA build')
    return torch.device(requested)


def linear_schedule(start, end, step, duration):
    return start + min(1.0, step / max(1, duration)) * (end - start)


def format_duration(seconds):
    total = max(0, int(seconds + .5))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def optimize(model, target_model, optimizer, replay, device, args, beta):
    samples, indices, weights = replay.sample(args.batch_size, beta)
    states = torch.as_tensor(np.stack([t.state for t in samples]),
                             dtype=torch.float32, device=device)
    with torch.no_grad():
        targets_np = td_targets(samples, target_model, device, args.gamma, args.infer_chunk)
    targets = torch.as_tensor(targets_np, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    model.train()
    predictions = model(states)
    errors = targets - predictions
    per_item = F.smooth_l1_loss(predictions, targets, reduction='none')
    weighted = (per_item * weights_t).mean()
    if not torch.isfinite(weighted):
        raise FloatingPointError('Non-finite weighted Huber loss')
    optimizer.zero_grad(set_to_none=True)
    weighted.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    errors_numpy = errors.detach().cpu().numpy()
    replay.update_priorities(indices, errors_numpy)
    abs_error = np.abs(errors_numpy)
    return {
        'weighted_huber': float(weighted.item()),
        'unweighted_huber': float(per_item.detach().mean().item()),
        'abs_td_mean': float(abs_error.mean()),
        'abs_td_p95': float(np.percentile(abs_error, 95)),
        'pred_value_mean': float(predictions.detach().mean().item()),
        'target_value_mean': float(targets.detach().mean().item()),
        'grad_norm_preclip': float(gradient_norm.item()),
        'beta': float(beta),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='One-step afterstate TD experiment v2')
    p.add_argument('--episodes', type=int, default=20000)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--blocks', type=int, default=6)
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--score-scale', type=float, default=1000.0)
    p.add_argument('--gameover-penalty', type=float, default=-1000.0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--capacity', type=int, default=10000)
    p.add_argument('--warmup', type=int, default=256)
    p.add_argument('--update-every', type=int, default=4)
    p.add_argument('--target-sync', type=int, default=500)
    p.add_argument('--per-alpha', type=float, default=0.6)
    p.add_argument('--per-beta-start', type=float, default=0.4)
    p.add_argument('--per-beta-steps', type=int, default=50000)
    p.add_argument('--per-mode', choices=['legacy', 'replacement'], default='replacement')
    p.add_argument('--epsilon-start', type=float, default=0.4)
    p.add_argument('--epsilon-final', type=float, default=0.02)
    p.add_argument('--epsilon-steps', type=int, default=500000)
    p.add_argument('--infer-chunk', type=int, default=128)
    p.add_argument('--save-every', type=int, default=2500,
                   help='Save evaluation-only ep_XXXXX.pt snapshots')
    p.add_argument('--resume-every', type=int, default=2500,
                   help='Overwrite the full last_resume.pt (large, includes replay)')
    p.add_argument('--eval-every', type=int, default=2500)
    p.add_argument('--eval-episodes', type=int, default=100)
    p.add_argument('--eval-seed', type=int, default=100000)
    p.add_argument('--eval-ply', choices=[1, 3], type=int, default=1)
    p.add_argument('--train-ply', choices=[1, 3], type=int, default=1,
                   help='1-ply historical baseline or 3-ply behavior policy; '
                        'TD target remains strictly 1-step in either case')
    p.add_argument('--train-root-keep', type=int, default=5)
    p.add_argument('--train-second-keep', type=int, default=10)
    p.add_argument('--eval-root-keep', type=int, default=5)
    p.add_argument('--eval-second-keep', type=int, default=10)
    p.add_argument('--probe-seed', type=int, default=500000)
    p.add_argument('--no-probes', action='store_true')
    p.add_argument('--no-train-moves', action='store_true')
    p.add_argument('--no-eval-moves', action='store_true')
    p.add_argument('--log-update-every', type=int, default=500)
    p.add_argument('--print-every', type=int, default=25)
    p.add_argument('--run-dir', type=Path, default=None,
                   help='Unique output directory; must not contain an existing run')
    p.add_argument('--resume', type=Path, default=None,
                   help='Continue from a v2 full-resume checkpoint into NEW run directory')
    p.add_argument('--out', type=Path, default=None,
                   help='Optional legacy filename for a FINAL weights-only copy')
    return p.parse_args(argv)


def _train_config(args):
    # Everything that changes gradients or training RNG sequence. Exclude
    # output, evaluation, diagnostic and completion-budget settings.
    return {k: getattr(args, k) for k in (
        'seed', 'channels', 'blocks', 'gamma', 'score_scale',
        'gameover_penalty', 'lr', 'batch_size', 'capacity', 'warmup',
        'update_every', 'target_sync', 'per_alpha', 'per_beta_start',
        'per_beta_steps', 'per_mode', 'epsilon_start', 'epsilon_final',
        'epsilon_steps', 'infer_chunk', 'train_ply', 'train_root_keep',
        'train_second_keep',
    )}


def _git_rev():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL, text=True,
                                       timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _validate(args):
    for k in ('episodes', 'batch_size', 'capacity', 'update_every',
              'target_sync', 'infer_chunk', 'eval_episodes', 'print_every'):
        if getattr(args, k) < 1:
            raise ValueError(f'--{k.replace("_", "-")} must be >= 1')
    for k in ('save_every', 'resume_every', 'eval_every', 'log_update_every'):
        if getattr(args, k) < 0:
            raise ValueError(f'--{k.replace("_", "-")} must be >= 0')
    if not (0 < args.gamma <= 1 and args.score_scale > 0 and args.lr > 0):
        raise ValueError('gamma, score-scale and lr out of range')
    if not 0 <= args.per_alpha <= 1 or not 0 <= args.per_beta_start <= 1:
        raise ValueError('PER alpha/beta out of range')
    if args.no_probes is False and args.gameover_penalty != -1000:
        raise ValueError('Fixed-probe diagnostics currently require -1000 gameover penalty; '
                         'pass --no-probes for another penalty')
    if args.no_probes is False and args.score_scale <= 0:
        raise ValueError('Invalid scale')
    if (args.train_root_keep < 1 or args.train_second_keep < 1 or
        args.eval_root_keep < 1 or args.eval_second_keep < 1):
        raise ValueError('Search root/second keep must be >= 1')
    if args.eval_seed <= args.seed + args.episodes - 1:
        raise ValueError('Evaluation seeds must be disjoint and greater than training seeds')
    if args.probe_seed <= args.eval_seed + args.eval_episodes - 1:
        raise ValueError('Probe seeds must be greater and disjoint from evaluation seeds')


def train(args):
    _validate(args)
    run_started = time.perf_counter()
    unique = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'mori25_1step_{unique}_seed{args.seed}'
    run_dir = (args.run_dir if args.run_dir is not None else
               Path(__file__).resolve().parent / 'runs' / run_id)
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'Run directory is nonempty: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device(args.device)
    model = AfterstateValueNet(CHANNELS, args.channels, args.blocks).to(device)
    target_model = AfterstateValueNet(CHANNELS, args.channels, args.blocks).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    replay = PrioritizedReplay(args.capacity, args.per_alpha, args.seed + 1,
                               mode=args.per_mode)
    rng = np.random.default_rng(args.seed + 2)
    env = puyo_cpp.PuyoEnv()
    model_cfg = {'channels': args.channels, 'blocks': args.blocks,
                 'input_channels': CHANNELS}
    value_cfg = {'gamma': args.gamma, 'score_scale': args.score_scale,
                 'gameover_penalty': args.gameover_penalty}
    cfg = _train_config(args)
    start_episode = 0
    env_steps = gradient_updates = 0
    elapsed_before_resume = 0.0
    best_score = None
    parent = None
    if args.resume is not None:
        loaded = load_resume(args.resume, model, target_model, optimizer,
                             replay, rng, cfg)
        start_episode = int(loaded['episode'])
        env_steps = int(loaded['env_steps'])
        gradient_updates = int(loaded['gradient_updates'])
        elapsed_before_resume = float(loaded['elapsed_seconds'])
        parent = str(Path(args.resume).resolve())
        # Track best within this continuation. The saved source best model
        # may be in a different directory and is not copied implicitly.
        best_score = None
        if start_episode >= args.episodes:
            raise ValueError('Requested episodes must exceed resume checkpoint episode')

    def elapsed():
        return elapsed_before_resume + time.perf_counter() - run_started

    manifest = {
        'run_id': run_id, 'start_from_episode': start_episode,
        'planned_total_episodes': args.episodes, 'train_config': cfg,
        'model_config': model_cfg, 'value_config': value_cfg,
        'evaluation_config': {
            'eval_every': args.eval_every, 'eval_episodes': args.eval_episodes,
            'eval_seed': args.eval_seed, 'eval_ply': args.eval_ply,
            'root_keep': args.eval_root_keep,
            'second_keep': args.eval_second_keep,
        },
        'parent_resume_checkpoint': parent,
        'git_commit': _git_rev(),
        'python': sys.version, 'torch': torch.__version__,
        'numpy': np.__version__, 'device': str(device),
        'cuda_device_name': (torch.cuda.get_device_name(device)
                             if device.type == 'cuda' else None),
        'important_caveats': [
            'Single actor; not the 1000-actor setup in Mori 2025.',
            '5x14x6 features, not paper exact 4x14x6.',
            'Afterstate board only; future pairs/remaining turns not network inputs.',
            'PER mode selected in train_config; replacement changes training from legacy.',
            'Validation-game SE is NOT between-training-run SE.',
            '3-ply search is action selection, not 3-step TD learning.',
            'Training actor ply and evaluation ply are separately configured.',
        ],
    }
    write_json(run_dir / 'manifest.json', manifest)

    ep_path = run_dir / 'train_episodes.csv'
    move_stream = None
    move_writer = None
    if not args.no_train_moves:
        move_stream = gzip.open(run_dir / 'train_moves.csv.gz', 'wt',
                                encoding='utf-8', newline='', compresslevel=5)
        move_writer = csv.DictWriter(move_stream, fieldnames=MOVE_COLUMNS)
        move_writer.writeheader()

    probes = None
    if not args.no_probes:
        from .diagnostics import make_probes
        probes = make_probes(puyo_cpp.PuyoEnv, seed=args.probe_seed)
    probe_previous = None
    recent_times = deque(maxlen=10)
    last_episode = start_episode
    print(f'device={device}, run_dir={run_dir}, PER={args.per_mode}, '
          f'resume_from={start_episode}, eval_ply={args.eval_ply}', flush=True)

    def save_weights(episode, label):
        save_model(run_dir / label, model, episode, env_steps, gradient_updates,
                   elapsed(), run_id, model_cfg, value_cfg)

    def full_save(episode):
        save_resume(run_dir / 'last_resume.pt', model, target_model, optimizer,
                    replay, rng, episode, env_steps, gradient_updates,
                    elapsed(), best_score, run_id, model_cfg, value_cfg, cfg)

    try:
        for episode in range(start_episode + 1, args.episodes + 1):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            ep_started = time.perf_counter()
            seed = args.seed + episode - 1
            env.reset(seed)
            candidates = env.preview_all()
            ep_score = ep_max_chain = turns = activations = 0
            n3 = n5 = n7 = 0
            step_stats = []
            last_ending = None
            epsilon = linear_schedule(args.epsilon_start, args.epsilon_final,
                                      env_steps, args.epsilon_steps)
            while not env.done():
                epsilon = linear_schedule(args.epsilon_start, args.epsilon_final,
                                          env_steps, args.epsilon_steps)
                shown = env.observation()['visible'] if move_writer else None
                if args.train_ply == 1:
                    picked, encoded = pick_action(
                        candidates, model, device, rng, epsilon, args.gamma,
                        args.score_scale, args.gameover_penalty, args.infer_chunk)
                    action = int(candidates[picked]['action'])
                else:
                    # The exploration roll is exactly once per placement.
                    # In random exploration, no expensive tree is constructed.
                    if rng.random() < epsilon:
                        picked = int(rng.integers(len(candidates)))
                    else:
                        from .search import greedy_search
                        choice = greedy_search(
                            env, model, device, args.gamma, args.score_scale,
                            args.gameover_penalty, ply=3,
                            root_keep=args.train_root_keep,
                            second_keep=args.train_second_keep,
                            chunk_size=args.infer_chunk,
                            root_candidates=candidates)
                        action = choice.action
                        picked = next((i for i, c in enumerate(candidates)
                                       if int(c['action']) == action), -1)
                        if picked < 0:
                            raise RuntimeError('3-ply selected an illegal root action')
                    encoded = encode_candidates(candidates)
                    action = int(candidates[picked]['action'])
                result = env.step(action)
                last_ending = result
                chains = int(result['chains'])
                score = int(result['score'])
                ep_score += score
                ep_max_chain = max(ep_max_chain, chains)
                turns += 1
                env_steps += 1
                activations += int(chains > 0)
                n3 += int(chains >= 3)
                n5 += int(chains >= 5)
                n7 += int(chains >= 7)
                terminal = bool(result['terminated'] or result['truncated'])
                following = [] if terminal else env.preview_all()
                replay.add(Transition(
                    state=encoded[picked],
                    next_states=(encode_candidates(following) if following else
                                 np.empty((0, CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)),
                    next_rewards=(scaled_immediate_rewards(
                        following, args.score_scale, args.gameover_penalty)
                        if following else np.empty(0, dtype=np.float32)),
                    next_dones=(candidate_dones(following) if following else
                                np.empty(0, dtype=np.bool_)),
                    terminal=terminal, chains=chains,
                ))
                if move_writer:
                    move_writer.writerow({
                        'episode': episode, 'seed': seed, 'turn': turns,
                        'action': action, 'visible_current': ''.join(shown[0]),
                        'visible_next': ''.join(shown[1]),
                        'visible_next2': ''.join(shown[2]),
                        'chains': chains, 'score': score,
                        'cumulative_score': ep_score, 'epsilon': epsilon,
                        'terminated': bool(result['terminated']),
                        'truncated': bool(result['truncated']),
                    })
                candidates = following
                if len(replay) >= max(args.warmup, args.batch_size) and env_steps % args.update_every == 0:
                    beta = linear_schedule(args.per_beta_start, 1.0,
                                           gradient_updates, args.per_beta_steps)
                    stat = optimize(model, target_model, optimizer, replay,
                                    device, args, beta)
                    step_stats.append(stat)
                    gradient_updates += 1
                    if args.log_update_every and gradient_updates % args.log_update_every == 0:
                        append_csv(run_dir / 'train_updates.csv',
                                   {'episode': episode, 'env_steps': env_steps,
                                    'gradient_updates': gradient_updates, **stat},
                                   UPDATE_COLUMNS)
                    if gradient_updates % args.target_sync == 0:
                        target_model.load_state_dict(model.state_dict())
                        target_model.eval()

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            ep_seconds = time.perf_counter() - ep_started
            recent_times.append(ep_seconds)
            last_episode = episode
            ep_stats = {
                'episode': episode, 'seed': seed, 'turns': turns,
                'score': ep_score, 'max_chain': ep_max_chain,
                'activations': activations, 'activations_ge_3': n3,
                'activations_ge_5': n5, 'activations_ge_7': n7,
                'terminated': bool(last_ending['terminated']),
                'truncated': bool(last_ending['truncated']),
                'epsilon': epsilon, 'replay_size': len(replay),
                'env_steps': env_steps, 'gradient_updates': gradient_updates,
                'avg_weighted_huber': (float(np.mean([s['weighted_huber'] for s in step_stats]))
                                       if step_stats else None),
                'avg_unweighted_huber': (float(np.mean([s['unweighted_huber'] for s in step_stats]))
                                         if step_stats else None),
                'avg_abs_td': (float(np.mean([s['abs_td_mean'] for s in step_stats]))
                               if step_stats else None),
                'episode_seconds': ep_seconds, 'elapsed_seconds': elapsed(),
            }
            append_csv(ep_path, ep_stats, EP_COLUMNS)
            if move_stream and episode % 50 == 0:
                move_stream.flush()
            if episode % args.print_every == 0 or episode == args.episodes:
                eta = (sum(recent_times) / len(recent_times) * (args.episodes - episode))
                loss = (f"{ep_stats['avg_weighted_huber']:.5f}"
                        if ep_stats['avg_weighted_huber'] is not None else 'n/a')
                print(f'episode={episode}/{args.episodes} turns={turns} '
                      f'score={ep_score} max_chain={ep_max_chain} eps={epsilon:.3f} '
                      f'replay={len(replay)} updates={gradient_updates} '
                      f'avg_loss={loss} ep_time={format_duration(ep_seconds)} '
                      f'elapsed={format_duration(elapsed())} '
                      f'eta~{format_duration(eta)}', flush=True)

            is_end = episode == args.episodes
            if is_end or (args.save_every and episode % args.save_every == 0):
                save_weights(episode, f'ep_{episode:06d}.pt')
                save_weights(episode, 'last.pt')

            if args.eval_every and (is_end or episode % args.eval_every == 0):
                # Every evaluation result must refer to an existing, immutable
                # checkpoint even when --eval-every != --save-every.
                snapshot = run_dir / f'ep_{episode:06d}.pt'
                if not snapshot.exists():
                    save_weights(episode, snapshot.name)
                from .evaluate import run_evaluation
                evaluation_name = f'ep_{episode:06d}_ply{args.eval_ply}'
                result = run_evaluation(
                    model, value_cfg, device, n_episodes=args.eval_episodes,
                    first_seed=args.eval_seed, ply=args.eval_ply,
                    root_keep=args.eval_root_keep,
                    second_keep=args.eval_second_keep,
                    chunk_size=args.infer_chunk,
                    output_dir=run_dir / 'evaluations' / evaluation_name,
                    checkpoint_name=f'ep_{episode:06d}.pt', run_id=run_id,
                    write_moves=not args.no_eval_moves,
                )
                perf = result['mean_max_chain']
                improving = (best_score is None or perf > best_score)
                if improving:
                    best_score = perf
                    save_weights(episode, 'best.pt')
                append_csv(run_dir / 'eval_metrics.csv', {
                    'episode': episode, 'env_steps': env_steps,
                    'gradient_updates': gradient_updates,
                    'eval_ply': args.eval_ply, 'eval_episodes': args.eval_episodes,
                    'new_best': improving, **result,
                }, EVAL_COLUMNS)
                print(f'eval episode={episode} mean_max_chain={perf:.3f} '
                      f'mean_score={result["mean_score"]:.2f} '
                      f'best={best_score:.3f}', flush=True)
                if probes:
                    from .diagnostics import measure_probes
                    diagnostic, probe_previous = measure_probes(
                        probes, model, target_model, device, value_cfg,
                        previous=probe_previous, chunk_size=args.infer_chunk)
                    diagnostic.update({
                        'episode': episode, 'gradient_updates': gradient_updates,
                        **replay.chain_distribution(),
                    })
                    # Prefix columns match JSON field names directly.
                    pr = replay.chain_distribution()
                    diagnostic['replay_priority_mean'] = pr['priority_mean']
                    diagnostic['replay_priority_p95'] = pr['priority_p95']
                    diagnostic['replay_priority_max'] = pr['priority_max']
                    diagnostic['replay_fraction_chain_ge_5'] = pr['fraction_chain_ge_5']
                    diagnostic['replay_sample_prob_chain_ge_5'] = pr['sample_prob_chain_ge_5']
                    diagnostic['replay_mean_chains'] = pr['mean_chains']
                    diagnostic['replay_fraction_zero_chain'] = pr['fraction_zero_chain']
                    append_csv(run_dir / 'diagnostics.csv', diagnostic, DIAG_COLUMNS)
            elif is_end and probes:
                # No separate evaluation configured; still create an end probe.
                from .diagnostics import measure_probes
                diagnostic, probe_previous = measure_probes(
                    probes, model, target_model, device, value_cfg,
                    previous=probe_previous, chunk_size=args.infer_chunk)
                diagnostic.update({'episode': episode, 'gradient_updates': gradient_updates})
                pr = replay.chain_distribution()
                for field, key in (
                    ('replay_priority_mean', 'priority_mean'),
                    ('replay_priority_p95', 'priority_p95'),
                    ('replay_priority_max', 'priority_max'),
                    ('replay_fraction_chain_ge_5', 'fraction_chain_ge_5'),
                    ('replay_sample_prob_chain_ge_5', 'sample_prob_chain_ge_5'),
                    ('replay_mean_chains', 'mean_chains'),
                    ('replay_fraction_zero_chain', 'fraction_zero_chain'),
                ):
                    diagnostic[field] = pr[key]
                append_csv(run_dir / 'diagnostics.csv', diagnostic, DIAG_COLUMNS)

            if is_end or (args.resume_every and episode % args.resume_every == 0):
                full_save(episode)
        # Save a portable final snapshot if the caller requested a legacy path.
        if args.out is not None:
            save_model(args.out, model, last_episode, env_steps,
                       gradient_updates, elapsed(), run_id, model_cfg, value_cfg)
        final = {
            'run_id': run_id, 'run_dir': str(run_dir),
            'start_from_episode': start_episode, 'end_episode': last_episode,
            'env_steps': env_steps, 'gradient_updates': gradient_updates,
            'elapsed_seconds_including_resume_ancestor': elapsed(),
            'best_eval_mean_max_chain_this_segment': best_score,
            'per_mode': args.per_mode,
        }
        write_json(run_dir / 'completed.json', final)
        print('TRAINING COMPLETED ' + json.dumps(final, ensure_ascii=False), flush=True)
        return final
    finally:
        if move_stream is not None:
            move_stream.close()


def main():
    train(parse_args())


if __name__ == '__main__':
    main()
