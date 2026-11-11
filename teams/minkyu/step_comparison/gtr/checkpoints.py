"""Unmodified original definitions moved from puyo_train.py; imports are wiring only."""

import torch
import numpy as np
from pathlib import Path
import os
import random
import uuid


FORMAT = 3


def _atomic_torch_save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        torch.save(data, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def model_snapshot(model, episode, env_steps, updates, elapsed, run_id, cfg, value_cfg):
    return {
        'format_version': FORMAT,
        'model': model.state_dict(),
        'episode': int(episode), 'env_steps': int(env_steps),
        'gradient_updates': int(updates), 'elapsed_seconds': float(elapsed),
        'model_config': dict(cfg), 'value_config': dict(value_cfg),
        'run_id': run_id,
    }


def save_model(path, model, episode, env_steps, updates, elapsed, run_id,
               model_cfg, value_cfg):
    snapshot = model_snapshot(model, episode, env_steps, updates, elapsed,
                              run_id, model_cfg, value_cfg)
    _atomic_torch_save(snapshot, path)


def _rng_state(policy_rng):
    return {
        'python': random.getstate(),
                                                                         
                                                                     
        'numpy_global': _numpy_global_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'policy_numpy': policy_rng.bit_generator.state,
    }


def _numpy_global_state():
    state = np.random.get_state()
    return (state[0], torch.as_tensor(state[1].copy()), state[2], state[3], state[4])


def _restore_rng(state, policy_rng):
    random.setstate(tuple(state['python']))
    np_state = state['numpy_global']
    np.random.set_state((np_state[0], np_state[1].numpy(), np_state[2],
                         np_state[3], np_state[4]))
    torch.set_rng_state(state['torch_cpu'])
    if torch.cuda.is_available() and state['torch_cuda']:
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    policy_rng.bit_generator.state = state['policy_numpy']


def save_resume(path, model, target_model, optimizer, replay, policy_rng,
                episode, env_steps, updates, elapsed, best_metric, run_id,
                model_cfg, value_cfg, train_cfg, best_chain_metric=None, extra_state=None):
    data = model_snapshot(model, episode, env_steps, updates, elapsed,
                          run_id, model_cfg, value_cfg)
    data.update({
        'kind': 'full_resume', 'target_model': target_model.state_dict(),
        'optimizer': optimizer.state_dict(), 'replay': replay.state_dict(),
        'rng': _rng_state(policy_rng),
        'best_metric': float(best_metric) if best_metric is not None else None,
        'best_chain_metric': list(best_chain_metric) if best_chain_metric is not None else None,
        'train_config': dict(train_cfg),
        'extra_state': extra_state,
    })
    _atomic_torch_save(data, path)


def load_resume(path, model, target_model, optimizer, replay, policy_rng,
                train_cfg):
    packed = torch.load(path, map_location='cpu', weights_only=True)
    if packed.get('kind') != 'full_resume' or packed.get('format_version') != FORMAT:
        raise ValueError('Not a compatible full-resume checkpoint')
    if packed['train_config'] != train_cfg:
        for key, value in packed['train_config'].items():
            if key not in train_cfg or train_cfg[key] != value:
                raise ValueError(f'Resume training config mismatch: {key}: '
                                 f'stored={value!r} requested={train_cfg.get(key)!r}')
        raise ValueError('Resume training config mismatch: keys differ')
    model.load_state_dict(packed['model'], strict=True)
    target_model.load_state_dict(packed['target_model'], strict=True)
    optimizer.load_state_dict(packed['optimizer'])
    replay.load_state_dict(packed['replay'])
    _restore_rng(packed['rng'], policy_rng)
    return packed
