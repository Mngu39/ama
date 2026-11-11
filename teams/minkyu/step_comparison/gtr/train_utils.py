"""Unmodified original definitions moved from puyo_train.py; imports are wiring only."""

import torch
import numpy as np
from pathlib import Path
import argparse
import subprocess
import torch.nn.functional as F

from .replay import td_targets

EP_COLUMNS = [
    'episode', 'seed', 'turns', 'score', 'max_chain', 'activations',
    'activations_ge_3', 'activations_ge_5', 'activations_ge_7',
    'terminated', 'truncated', 'epsilon', 'replay_size', 'env_steps',
    'gradient_updates', 'n_step', 'avg_weighted_huber', 'avg_unweighted_huber',
    'avg_abs_td', 'episode_seconds', 'elapsed_seconds',
]


TRAIN_MOVE_COLUMNS = [
    'episode', 'seed', 'turn', 'action', 'visible_current', 'visible_next',
    'visible_next2', 'chains', 'score', 'cumulative_score',
    'epsilon', 'terminated', 'truncated',
]


EVAL_COLUMNS = [
    'episode', 'env_steps', 'gradient_updates', 'n_step', 'eval_ply', 'eval_episodes',
    'mean_max_chain', 'std_max_chain_between_games',
    'mean_score', 'mean_turns', 'p_max_chain_ge_3', 'p_max_chain_ge_5',
    'p_max_chain_ge_7', 'p_max_chain_ge_10',
    'mean_chain_per_activation', 'chain_activations_per_game',
    'gameover_rate', 'avg_decision_time_seconds', 'eval_wall_time_seconds',
    'new_best', 'new_best_chain',
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
    p = argparse.ArgumentParser(description='GTR afterstate TD (1/3/5-step) comparison')
    p.add_argument('--form-weight', type=float, default=0.04,
                   help='Potential weight in scaled reward units; 0 disables shaping.')
    p.add_argument('--episodes', type=int, default=20000)
    p.add_argument('--n-step', type=int, default=1,
                   help='TD backup horizon in placements; NOT 1/3-ply search.')
    p.add_argument('--puyo-generator',
                   choices=['observed', 'uniform', 'rule_based', 'learned'],
                   default='observed',
                   help='observed=original real-trajectory n-step; otherwise use a '
                        'model-generated future on a forked engine')
    p.add_argument('--learned-mode', choices=['online','frozen'], default='online',
                   help='online=joint supervised training from newly revealed real pairs; '
                        'frozen=load separately trained weights and do not update')
    p.add_argument('--learned-checkpoint', type=Path, default=None,
                   help='Optional pretrained generator checkpoint; REQUIRED for frozen mode')
    p.add_argument('--learned-lr', type=float, default=0.001)
    p.add_argument('--learned-batch-size', type=int, default=128)
    p.add_argument('--learned-warmup', type=int, default=128)
    p.add_argument('--learned-update-every', type=int, default=8)
    p.add_argument('--learned-capacity', type=int, default=30000)
    p.add_argument('--learned-monitor-every', type=int, default=500,
                   help='Record fixed-input 16-pair probabilities and heldout scores every N episodes; 0 disables')
    p.add_argument('--learned-monitor-games', type=int, default=512,
                   help='Independent Ama games for fixed, read-only generator validation')
    p.add_argument('--learned-monitor-seed-start', type=int, default=700000,
                   help='Must not overlap RL training, model pretraining, or evaluation seeds')
    p.add_argument('--no-model-rollout-log', action='store_true',
                   help='Disable per-placement virtual-target audit CSV')
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
    p.add_argument('--save-every', type=int, default=500,
                   help='Save evaluation-only ep_XXXXX.pt snapshots')
    p.add_argument('--resume-every', type=int, default=500,
                   help='Overwrite the full last_resume.pt (large, includes replay)')
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--eval-episodes', type=int, default=100)
    p.add_argument('--eval-seed', type=int, default=200000)
    p.add_argument('--eval-ply', choices=[1, 3], type=int, default=1)
    p.add_argument('--train-ply', choices=[1, 3], type=int, default=1,
                   help='1-ply historical baseline or 3-ply behavior policy; '
                         'TD horizon is controlled independently by --n-step')
    p.add_argument('--train-root-keep', type=int, default=5)
    p.add_argument('--train-second-keep', type=int, default=10)
    p.add_argument('--eval-root-keep', type=int, default=22)
    p.add_argument('--eval-second-keep', type=int, default=484)
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


def is_better_chain(candidate, best):
    return best is None or tuple(candidate) > tuple(best)


def _train_config(args):
                                                                         
                                                                    
    config = {k: getattr(args, k) for k in (
        'seed', 'channels', 'blocks', 'gamma', 'score_scale',
        'form_weight', 'gameover_penalty', 'lr', 'batch_size', 'capacity', 'warmup',
        'update_every', 'target_sync', 'per_alpha', 'per_beta_start',
        'per_beta_steps', 'per_mode', 'epsilon_start', 'epsilon_final',
        'epsilon_steps', 'infer_chunk', 'train_ply', 'train_root_keep',
         'train_second_keep', 'n_step',
    )}
    if args.puyo_generator != 'observed':
        config['puyo_generator'] = args.puyo_generator
    if args.puyo_generator == 'learned':
        for key in ('learned_mode','learned_checkpoint','learned_lr',
                    'learned_batch_size','learned_warmup','learned_update_every',
                    'learned_capacity'):
            value=getattr(args,key)
            config[key]=str(value) if isinstance(value,Path) else value
    return config


def _git_rev():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL, text=True,
                                       timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _validate(args):
    if args.learned_monitor_every < 0 or args.learned_monitor_games < 2:
        raise ValueError('Learned monitor interval must be >=0 and games >=2')
    if args.puyo_generator == 'learned' and args.learned_monitor_every:
        monitored = set(range(args.learned_monitor_seed_start,
                              args.learned_monitor_seed_start + args.learned_monitor_games))
        training = range(args.seed, args.seed + args.episodes)
        validation = range(args.eval_seed, args.eval_seed + args.eval_episodes)
        if (any(seed in training or seed in validation for seed in monitored)):
            raise ValueError('Learned monitor reference seed range overlaps RL train/validation')
    if args.n_step < 1:
        raise ValueError('n-step must be >= 1')
    if args.puyo_generator == 'learned':
        if args.learned_mode == 'frozen' and args.learned_checkpoint is None:
            raise ValueError('--learned-mode frozen requires --learned-checkpoint')
        if args.learned_checkpoint is not None and not args.learned_checkpoint.is_file():
            raise FileNotFoundError(args.learned_checkpoint)
        if (args.learned_lr <= 0 or args.learned_batch_size < 1 or
                args.learned_warmup < 1 or args.learned_update_every < 1 or
                args.learned_capacity < args.learned_batch_size):
            raise ValueError('Invalid learned generator hyperparameters')
    if args.puyo_generator != 'observed' and args.train_ply != 1:
        raise ValueError('Virtual-rollout policy currently supports --train-ply 1 only')
    if not np.isfinite(args.form_weight) or args.form_weight < 0:
        raise ValueError('form-weight must be finite and nonnegative')
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
