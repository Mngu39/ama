"""Unmodified original definitions moved from puyo_train.py; imports are wiring only."""

import torch
import numpy as np
from dataclasses import dataclass, field
import copy

from .actions import candidate_dones, encode_candidates, predict_values, scaled_immediate_rewards
from .replay import Transition, td_targets

@dataclass
class Probe:
    arrays: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    transition: Transition


def make_probes(env_class, *, seed=500000, episodes=6, per_episode=4):
    rng = np.random.default_rng(seed + 123456)
    env = env_class()
    probes = []
    for i in range(episodes):
        env.reset(seed + i)
        for t in range(per_episode):
            if env.done():
                break
            candidates = env.preview_all()
            if not candidates:
                break
            action_i = int(rng.integers(len(candidates)))
            arrays = encode_candidates(candidates)
            step = env.step(int(candidates[action_i]['action']))
            following = [] if (step['terminated'] or step['truncated']) else env.preview_all()
            next_states = (encode_candidates(following) if following else
                           np.empty((0,) + arrays.shape[1:], dtype=np.uint8))
            probes.append(Probe(
                arrays=arrays,
                rewards=scaled_immediate_rewards(candidates, 1000.0, -1000.0),
                dones=candidate_dones(candidates),
                transition=Transition(
                    arrays[action_i], next_states,
                    (scaled_immediate_rewards(following, 1000.0, -1000.0)
                     if following else np.empty((0,), dtype=np.float32)),
                    (candidate_dones(following) if following else
                     np.empty(0, dtype=np.bool_)),
                    bool(step['terminated'] or step['truncated']),
                ),
            ))
    return probes


def _train_mode_predict_preserving_bn(model, arrays, device):
                                                                             
                                                                      
    original_training = model.training
    buffers = {name: buf.detach().clone()
               for name, buf in model.named_buffers()
               if 'running_mean' in name or 'running_var' in name or
               'num_batches_tracked' in name}
    try:
        model.train(True)
        with torch.no_grad():
            return model(torch.as_tensor(arrays, dtype=torch.float32,
                                         device=device)).detach().cpu().numpy()
    finally:
        with torch.no_grad():
            all_buffers = dict(model.named_buffers())
            for name, copy in buffers.items():
                all_buffers[name].copy_(copy)
        model.train(original_training)


def measure_probes(probes, model, target_model, device, value_cfg,
                   previous=None, chunk_size=128):
    if not probes:
        return {}, None
    gamma = value_cfg['gamma']
    scale = value_cfg['score_scale']
    penalty = value_cfg['gameover_penalty']
    gaps = []
    target_gaps = []
    flips = 0
    action_indices = []
    action_scores = []
    bn_diffs = []
    margin = []


    ratio = 1000.0 / scale
    for probe in probes:
        eval_v = predict_values(model, probe.arrays, device, chunk_size)
        target_v = predict_values(target_model, probe.arrays, device, chunk_size)
        bn_v = _train_mode_predict_preserving_bn(model, probe.arrays, device)
        r = probe.rewards * ratio


        if penalty != -1000.0:
            raise ValueError('Probe diagnostics currently require --gameover-penalty -1000')
        action_eval = r + gamma * np.where(probe.dones, 0.0, eval_v)
        action_train = r + gamma * np.where(probe.dones, 0.0, bn_v)
        best = int(np.argmax(action_eval))
        action_indices.append(best)
        action_scores.append(float(action_eval[best]))
        flips += int(best != int(np.argmax(action_train)))
        gaps.extend(np.abs(eval_v - bn_v).tolist())
        target_gaps.extend(np.abs(eval_v - target_v).tolist())
        bn_diffs.extend((eval_v - bn_v).tolist())
        if len(action_eval) >= 2:
            sorted_scores = np.sort(action_eval)
            margin.append(float(sorted_scores[-1] - sorted_scores[-2]))


    transitions = [p.transition for p in probes]
                                                                           
                                                                   
    from dataclasses import replace
    adjusted = [replace(t, next_rewards=t.next_rewards * ratio) for t in transitions]
    y = td_targets(adjusted, target_model, device, gamma, chunk_size)
    states = np.stack([t.state for t in transitions])
    predicted = predict_values(model, states, device, chunk_size)
    td_abs = np.abs(y - predicted)
    report = {
        'probe_count': len(probes),
        'probe_bn_eval_train_mae': float(np.mean(gaps)),
        'probe_bn_eval_train_signed_bias': float(np.mean(bn_diffs)),
        'probe_bn_action_flip_rate': float(flips / len(probes)),
        'probe_online_target_mae': float(np.mean(target_gaps)),
        'probe_median_action_margin': float(np.median(margin)) if margin else None,
        'probe_td_abs_mean': float(td_abs.mean()),
        'probe_td_abs_p95': float(np.percentile(td_abs, 95)),
        'probe_td_abs_max': float(td_abs.max()),
        'probe_mean_best_action_value': float(np.mean(action_scores)),
    }
    this = {'actions': action_indices, 'values': predicted.copy()}
    if previous is not None:
        report['probe_action_flip_since_prev'] = float(
            np.mean(np.asarray(previous['actions']) != action_indices))
        report['probe_value_mae_since_prev'] = float(
            np.abs(previous['values'] - predicted).mean())
    else:
        report['probe_action_flip_since_prev'] = None
        report['probe_value_mae_since_prev'] = None
    return report, this
