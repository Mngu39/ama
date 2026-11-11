"""뿌요뿌요 점수 기반 강화학습. VS Code에서 ama-main 폴더를 열고 실행하세요.

Linux/WSL 최초 준비: bash scripts/setup_wsl.sh && source .venv/bin/activate
의존성: python -m pip install torch numpy
학습: python puyo_train.py
평가: python puyo_train.py evaluate --checkpoint runs/<run>/best.pt
정형 보상 비교군: python puyo_train.py --form-weight 0
기본값: 20,000 episodes, 1-ply training, greedy 1-ply validation every 500 episodes.
"""

import torch
from torch import nn
import numpy as np
from dataclasses import dataclass, field
import copy
from typing import Any
import csv
import gzip
import json
from pathlib import Path
from collections import defaultdict
from math import sqrt
import os
import random
import uuid
import argparse
import time
from collections import deque
from datetime import datetime, timezone
import subprocess
import sys
import shutil
import torch.nn.functional as F

if __package__:
    from .gtr.puyo_generators import make_generator, per_placement_rng
    from .gtr.learned_generator import LearnedGenerator, save_learned
    from .gtr.learned_monitor import LearnedMonitor
    from .gtr.model_rollout import simulate_target
    from .gtr.actions import encode_candidates, scaled_immediate_rewards, candidate_dones, predict_values, pick_action
    from .gtr.checkpoints import FORMAT, _atomic_torch_save, model_snapshot, save_model, _rng_state, _numpy_global_state, _restore_rng, save_resume, load_resume
    from .gtr.diagnostics import Probe, make_probes, _train_mode_predict_preserving_bn, measure_probes
    from .gtr.evaluation import load_evaluation_model, run_evaluation, evaluation_cli
    from .gtr.model import COLOR_CHANNEL, CHANNELS, HEIGHT, WIDTH, ResidualBlock, AfterstateValueNet
    from .gtr.records import EPISODE_COLUMNS, EVAL_MOVE_COLUMNS, board_string, write_csv, write_move_csv_gz, write_json, EpisodeResults, append_csv, THRESHOLDS, summarize
    from .gtr.replay import Transition, _PendingAfterstate, NStepAssembler, PrioritizedReplay, td_targets
    from .gtr.reward import FORM_PATTERNS, compile_pattern, COMPILED_FORMS, form_potential, RewardEnvironment, make_environment
    from .gtr.search import SearchResult, _scores, _rank_top, exhaustive_three_ply, greedy_search
    from .gtr.train_utils import EP_COLUMNS, TRAIN_MOVE_COLUMNS, EVAL_COLUMNS, DIAG_COLUMNS, UPDATE_COLUMNS, select_device, linear_schedule, format_duration, optimize, parse_args, is_better_chain, _train_config, _git_rev, _validate
else:
    from gtr.puyo_generators import make_generator, per_placement_rng
    from gtr.learned_generator import LearnedGenerator, save_learned
    from gtr.learned_monitor import LearnedMonitor
    from gtr.model_rollout import simulate_target
    from gtr.actions import encode_candidates, scaled_immediate_rewards, candidate_dones, predict_values, pick_action
    from gtr.checkpoints import FORMAT, _atomic_torch_save, model_snapshot, save_model, _rng_state, _numpy_global_state, _restore_rng, save_resume, load_resume
    from gtr.diagnostics import Probe, make_probes, _train_mode_predict_preserving_bn, measure_probes
    from gtr.evaluation import load_evaluation_model, run_evaluation, evaluation_cli
    from gtr.model import COLOR_CHANNEL, CHANNELS, HEIGHT, WIDTH, ResidualBlock, AfterstateValueNet
    from gtr.records import EPISODE_COLUMNS, EVAL_MOVE_COLUMNS, board_string, write_csv, write_move_csv_gz, write_json, EpisodeResults, append_csv, THRESHOLDS, summarize
    from gtr.replay import Transition, _PendingAfterstate, NStepAssembler, PrioritizedReplay, td_targets
    from gtr.reward import FORM_PATTERNS, compile_pattern, COMPILED_FORMS, form_potential, RewardEnvironment, make_environment
    from gtr.search import SearchResult, _scores, _rank_top, exhaustive_three_ply, greedy_search
    from gtr.train_utils import EP_COLUMNS, TRAIN_MOVE_COLUMNS, EVAL_COLUMNS, DIAG_COLUMNS, UPDATE_COLUMNS, select_device, linear_schedule, format_duration, optimize, parse_args, is_better_chain, _train_config, _git_rev, _validate

def train(args):
    _validate(args)
    run_started = time.perf_counter()
    unique = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'ama_form_td_n{args.n_step}_{unique}_seed{args.seed}'
    if args.puyo_generator != 'observed':
        run_id = f'ama_form_td_n{args.n_step}_{args.puyo_generator}_{unique}_seed{args.seed}'
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
    env = make_environment(args.gamma, args.form_weight, args.score_scale)
    learned = (LearnedGenerator(
        mode=args.learned_mode, seed=args.seed,
        init_checkpoint=args.learned_checkpoint, lr=args.learned_lr,
        batch_size=args.learned_batch_size, warmup=args.learned_warmup,
        update_every=args.learned_update_every, capacity=args.learned_capacity)
        if args.puyo_generator == 'learned' else None)
    generator = make_generator(args.puyo_generator, learned=learned,
                               n_step=args.n_step)
    if generator is not None:
        required = ('fork_with_hidden_rule_seed' if args.puyo_generator == 'rule_based'
                    else 'fork_with_hidden_pairs')
        if not hasattr(env.engine, required):
            raise RuntimeError(
                f'PuyoEnv.{required} unavailable; apply patches/ama_hidden_queue_fork.patch '
                'to Ama root and rebuild rl_env/puyo_cpp before model-based training')
    model_cfg = {'channels': args.channels, 'blocks': args.blocks,
                 'input_channels': CHANNELS}
    value_cfg = {'n_step': args.n_step, 'gamma': args.gamma, 'score_scale': args.score_scale,
                 'gameover_penalty': args.gameover_penalty, 'form_weight': args.form_weight}
    cfg = _train_config(args)
    start_episode = 0
    env_steps = gradient_updates = 0
    elapsed_before_resume = 0.0
    best_score = None
    best_chain_metric = None
    parent = None
    if args.resume is not None:
        loaded = load_resume(args.resume, model, target_model, optimizer,
                             replay, rng, cfg)
        if learned is not None:
            extra = loaded.get('extra_state')
            if not isinstance(extra, dict) or 'learned_generator' not in extra:
                raise ValueError('Learned run requires a learned resume state')
            learned.load_state_dict(extra['learned_generator'])
        start_episode = int(loaded['episode'])
        env_steps = int(loaded['env_steps'])
        gradient_updates = int(loaded['gradient_updates'])
        elapsed_before_resume = float(loaded['elapsed_seconds'])
        parent = str(Path(args.resume).resolve())
        previous_dir = Path(args.resume).parent
        previous_best_score = previous_dir / 'best.pt'
        if previous_best_score.is_file() and loaded.get('best_metric') is not None:
            shutil.copy2(previous_best_score, run_dir / 'best.pt')
            best_score = float(loaded['best_metric'])
            if learned is not None and (previous_dir / 'learned_best.pt').is_file():
                shutil.copy2(previous_dir / 'learned_best.pt',run_dir/'learned_best.pt')
        previous_best_chain = previous_dir / 'best_chain.pt'
        if previous_best_chain.is_file() and loaded.get('best_chain_metric') is not None:
            shutil.copy2(previous_best_chain, run_dir / 'best_chain.pt')
            best_chain_metric = tuple(float(x) for x in loaded['best_chain_metric'])
            if learned is not None and (previous_dir / 'learned_best_chain.pt').is_file():
                shutil.copy2(previous_dir/'learned_best_chain.pt',run_dir/'learned_best_chain.pt')
            previous_selection = previous_dir / 'best_chain_selection.json'
            if previous_selection.is_file():
                shutil.copy2(previous_selection, run_dir / 'best_chain_selection.json')
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
        'learned_generator': ({'mode':args.learned_mode,
                               'pretrained_checkpoint':str(args.learned_checkpoint) if args.learned_checkpoint else None,
                               'architecture':'position logits + 3-pair MLP (5872 parameters)',
                               'training_signal':'newly revealed real NEXT2 after current placement',
                               'sampling':'categorical stochastic autoregressive',
                               'offline_pretraining_cost_not_included_in_rl_elapsed':True}
                              if learned is not None else None),
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
             'n>1 targets sample intermediate actions under the behavior policy, then greedily bootstrap; no off-policy correction.',
             'Probe TD diagnostics retain the original 1-step definition for across-horizon comparison.',
            'Training actor ply and evaluation ply are separately configured.',
        ],
    }
    if generator is not None:
        value_cfg['puyo_generator'] = args.puyo_generator
        manifest['virtual_rollout'] = {
            'mode': 'model_based_sampled_actions_last_partial',
            'generator': args.puyo_generator,
            'first_unknown_pair_relative_to_current': 3,
            'actor_policy': 'epsilon_greedy_1ply_on_virtual_states',
            'generator_rng': 'SeedSequence(seed,episode,turn,9001)',
            'virtual_action_rng': 'SeedSequence(seed,episode,turn,9002)',
            'rule_based_conditioning': ('visible_3_pair_rejection_exact_given_visible_only'
                                        if args.puyo_generator == 'rule_based' else None),
        }
    if learned is not None:
        manifest['learned_generator']['distribution_monitor'] = {
            'every_episodes': args.learned_monitor_every,
            'reference_games': args.learned_monitor_games,
            'reference_seed_start': args.learned_monitor_seed_start,
            'fixed_probes': 'from independent reference engine games',
            'probabilities': 'all 16 pair combinations',
            'heldout_nll': 'fixed actual newly disclosed pairs',
            'kl_tv': 'position marginal, NOT exact visible-conditional',
        }
    write_json(run_dir / 'manifest.json', manifest)
    learned_monitor = None
    if learned is not None and args.learned_monitor_every:
        learned_monitor = LearnedMonitor.from_engine(
            run_dir, seed_start=args.learned_monitor_seed_start,
            games=args.learned_monitor_games)
        # Record the initial generator (episode 0 on a fresh run).
        initial_episode = start_episode
        save_learned(run_dir / f'learned_ep_{initial_episode:06d}.pt',
                     learned.model, {'episode': initial_episode,
                                     'mode': learned.mode, 'observations': learned.seen,
                                     'updates': learned.updates})
        learned_monitor.snapshot(learned.model, initial_episode,
                                 mode=learned.mode, observations=learned.seen,
                                 updates=learned.updates)

    training_results = EpisodeResults(run_dir)
    ep_path = run_dir / 'train_episodes.csv'
    move_stream = None
    move_writer = None
    if not args.no_train_moves:
        move_stream = gzip.open(run_dir / 'train_moves.csv.gz', 'wt',
                                encoding='utf-8', newline='', compresslevel=5)
        move_writer = csv.DictWriter(move_stream, fieldnames=TRAIN_MOVE_COLUMNS)
        move_writer.writeheader()

    virtual_stream = None
    virtual_writer = None
    if generator is not None and not args.no_model_rollout_log:
        virtual_stream = gzip.open(run_dir / 'model_rollouts.csv.gz', 'wt',
                                   encoding='utf-8', newline='', compresslevel=5)
        virtual_writer = csv.DictWriter(virtual_stream, fieldnames=[
            'episode','seed','turn','n_step','generator',
            'first_hidden_pair','simulated_actions','simulated_rewards','final_candidate_count',
            'terminated_early'])
        virtual_writer.writeheader()
    probes = None
    if not args.no_probes:
        probes = make_probes(lambda: make_environment(args.gamma, args.form_weight, args.score_scale), seed=args.probe_seed)
    probe_previous = None
    recent_times = deque(maxlen=10)
    last_episode = start_episode
    print(f'device={device}, run_dir={run_dir}, PER={args.per_mode}, '
          f'resume_from={start_episode}, n_step={args.n_step}, eval_ply={args.eval_ply}, '
          f'generator={args.puyo_generator}', flush=True)

    def save_weights(episode, label):
        save_model(run_dir / label, model, episode, env_steps, gradient_updates,
                   elapsed(), run_id, model_cfg, value_cfg)
        if learned is not None:
            save_learned(run_dir / ('learned_' + label), learned.model,
                         {'episode':int(episode), 'mode':args.learned_mode,
                          'observations':int(learned.seen), 'updates':int(learned.updates)})

    def full_save(episode):
        save_resume(run_dir / 'last_resume.pt', model, target_model, optimizer,
                    replay, rng, episode, env_steps, gradient_updates,
                    elapsed(), best_score, run_id, model_cfg, value_cfg, cfg, best_chain_metric,
                    extra_state=({'learned_generator':learned.state_dict()}
                                 if learned is not None else None))

    try:
        for episode in range(start_episode + 1, args.episodes + 1):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            ep_started = time.perf_counter()
            seed = args.seed + episode - 1
            env.reset(seed)
            candidates = env.preview_all()
            assembler = NStepAssembler(args.n_step) if generator is None else None
            ep_score = ep_max_chain = turns = activations = 0
            n3 = n5 = n7 = 0
            step_stats = []
            last_ending = None
            epsilon = linear_schedule(args.epsilon_start, args.epsilon_final,
                                      env_steps, args.epsilon_steps)
            while not env.done():
                epsilon = linear_schedule(args.epsilon_start, args.epsilon_final,
                                          env_steps, args.epsilon_steps)
                before_observation = env.observation() if learned is not None else None
                shown = (before_observation['visible'] if (move_writer and learned is not None)
                         else env.observation()['visible'] if move_writer else None)
                if args.train_ply == 1:
                    picked, encoded = pick_action(
                        candidates, model, device, rng, epsilon, args.gamma,
                        args.score_scale, args.gameover_penalty, args.infer_chunk)
                    action = int(candidates[picked]['action'])
                else:
                                                                         
                                                                              
                    if rng.random() < epsilon:
                        picked = int(rng.integers(len(candidates)))
                    else:
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
                # MUST fork before env.step(action): after stepping, the real
                # observation would disclose a previously hidden extra pair.
                virtual_fork = None
                if generator is not None:
                    virtual_fork = generator.fork(
                        env, per_placement_rng(args.seed, episode, turns, 9001))
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
                                                                      
                                                                              
                realized_reward = (score + float(result['form_bonus']) +
                                   (args.gameover_penalty if result['terminated'] else 0.0)) / args.score_scale
                if generator is None:
                    finished = assembler.observe(
                        state=encoded[picked], realized_reward=realized_reward,
                        terminal=terminal, chains=chains,
                        next_states=(encode_candidates(following) if following else
                                     np.empty((0, CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)),
                        next_rewards=(scaled_immediate_rewards(
                            following, args.score_scale, args.gameover_penalty)
                            if following else np.empty(0, dtype=np.float32)),
                        next_dones=(candidate_dones(following) if following else
                                    np.empty(0, dtype=np.bool_)),
                    )
                else:
                    virtual = simulate_target(
                        virtual_fork, action, encoded[picked], chains,
                        n_step=args.n_step, model=model, device=device,
                        rng=per_placement_rng(args.seed, episode, turns - 1, 9002),
                        epsilon=epsilon, gamma=args.gamma,
                        score_scale=args.score_scale,
                        gameover_penalty=args.gameover_penalty,
                        infer_chunk=args.infer_chunk)
                    # The virtual environment must match the REAL deterministic
                    # current placement before divergent hidden pairs are used.
                    if (virtual.first_score != score or
                            virtual.first_board != tuple(result['observation']['board'])):
                        raise RuntimeError('Virtual fork differs on the current known placement')
                    finished = [virtual.transition]
                    if virtual_writer is not None:
                        virtual_writer.writerow({
                            'episode': episode, 'seed': seed, 'turn': turns,
                            'n_step': args.n_step, 'generator': args.puyo_generator,
                            'first_hidden_pair': virtual.first_hidden_pair,
                            'simulated_actions': json.dumps(virtual.simulated_actions),
                            'simulated_rewards': json.dumps(virtual.simulated_rewards),
                            'final_candidate_count': virtual.last_candidates,
                            'terminated_early': virtual.terminated_before_last,
                        })
                # The pre-action prediction and virtual target are fixed BEFORE
                # the newly revealed real NEXT2 may supervise the generator.
                if learned is not None:
                    learned.observe(before_observation, result['observation'],
                                    terminal=terminal)
                for transition in finished:
                    replay.add(transition)
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

            if assembler is not None and assembler.pending:
                raise RuntimeError('Episode ended with unflushed n-step transitions')
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            ep_seconds = time.perf_counter() - ep_started
            recent_times.append(ep_seconds)
            last_episode = episode
            ep_stats = {
                'episode': episode, 'seed': seed, 'n_step': args.n_step, 'turns': turns,
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
            training_results.record(episode, seed, ep_score, ep_max_chain)
            append_csv(ep_path, ep_stats, EP_COLUMNS)
            if learned is not None:
                append_csv(run_dir / 'learned_metrics.csv', {
                    'episode':episode,'mode':learned.mode,
                    'observations':learned.seen,'updates':learned.updates,
                    'last_batch_loss':learned.last_loss,
                }, ['episode','mode','observations','updates','last_batch_loss'])
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

            if learned_monitor is not None and (is_end or episode % args.learned_monitor_every == 0):
                learned_checkpoint = run_dir / f'learned_ep_{episode:06d}.pt'
                if not learned_checkpoint.exists():
                    save_learned(learned_checkpoint, learned.model, {
                        'episode': int(episode), 'mode': learned.mode,
                        'observations': int(learned.seen), 'updates': int(learned.updates)})
                monitor_metrics = learned_monitor.snapshot(
                    learned.model, episode, mode=learned.mode,
                    observations=learned.seen, updates=learned.updates)
                print(f'learned-monitor episode={episode} '
                      f'heldout_nll={monitor_metrics["heldout_nll"]:.5f} '
                      f'marginal_tv={monitor_metrics["position_marginal_tv"]}',
                      flush=True)

            if args.eval_every and (is_end or episode % args.eval_every == 0):
                                                                              
                                                                    
                snapshot = run_dir / f'ep_{episode:06d}.pt'
                if not snapshot.exists():
                    save_weights(episode, snapshot.name)
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
                perf = float(result['mean_score'])
                chain_metric = (float(result['mean_max_chain']), perf)
                improving = (best_score is None or perf > best_score)
                improving_chain = (best_chain_metric is None or chain_metric > best_chain_metric)
                if improving:
                    best_score = perf
                    save_weights(episode, 'best.pt')
                if improving_chain:
                    best_chain_metric = chain_metric
                    save_weights(episode, 'best_chain.pt')
                    write_json(run_dir / 'best_chain_selection.json', {
                        'episode': episode, 'mean_max_chain': chain_metric[0],
                        'mean_score': chain_metric[1], 'validation_ply': args.eval_ply,
                        'validation_games': args.eval_episodes,
                        'validation_seed_start': args.eval_seed,
                        'selection_rule': 'mean_max_chain_then_mean_score',
                    })
                append_csv(run_dir / 'eval_metrics.csv', {
                    'episode': episode, 'env_steps': env_steps,
                    'gradient_updates': gradient_updates, 'n_step': args.n_step,
                    'eval_ply': args.eval_ply, 'eval_episodes': args.eval_episodes,
                    'new_best': improving, 'new_best_chain': improving_chain, **result,
                }, EVAL_COLUMNS)
                print(f'eval episode={episode} mean_max_chain={result["mean_max_chain"]:.3f} '
                      f'mean_score={result["mean_score"]:.2f} '
                      f'best={best_score:.3f}', flush=True)
                if probes:
                    diagnostic, probe_previous = measure_probes(
                        probes, model, target_model, device, value_cfg,
                        previous=probe_previous, chunk_size=args.infer_chunk)
                    diagnostic.update({
                        'episode': episode, 'gradient_updates': gradient_updates,
                        **replay.chain_distribution(),
                    })
                                                                     
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
                                                                               
        if args.out is not None:
            save_model(args.out, model, last_episode, env_steps,
                       gradient_updates, elapsed(), run_id, model_cfg, value_cfg)
        final = {
            'run_id': run_id, 'run_dir': str(run_dir),
            'start_from_episode': start_episode, 'end_episode': last_episode,
            'env_steps': env_steps, 'gradient_updates': gradient_updates,
            'elapsed_seconds_including_resume_ancestor': elapsed(),
            'best_eval_mean_score_this_segment': best_score,
            'best_eval_mean_max_chain_this_segment': (best_chain_metric[0] if best_chain_metric else None),
            'per_mode': args.per_mode, 'n_step': args.n_step,
            'puyo_generator': args.puyo_generator,
            'learned_updates':(learned.updates if learned is not None else None),
            'learned_observations':(learned.seen if learned is not None else None),
        }
        write_json(run_dir / 'completed.json', final)
        print('TRAINING COMPLETED ' + json.dumps(final, ensure_ascii=False), flush=True)
        return final
    finally:
        if move_stream is not None:
            move_stream.close()
        if virtual_stream is not None:
            virtual_stream.close()


def main():
    train(parse_args())

if __name__ == '__main__':
    if sys.argv[1:2] == ['evaluate']:
        evaluation_cli()
    else:
        main()
