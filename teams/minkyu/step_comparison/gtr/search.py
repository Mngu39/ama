"""Unmodified original definitions moved from puyo_train.py; imports are wiring only."""

import numpy as np
from dataclasses import dataclass, field

from .actions import candidate_dones, encode_candidates, predict_values, scaled_immediate_rewards

@dataclass(frozen=True)
class SearchResult:
    action: int
    score: float                                                           
    evaluated: int                                                             
    simulated: int                                              
    depth: int
    root_candidates: int


def _scores(candidates, model, device, gamma, scale, penalty, chunk):
    if not candidates:
        return np.empty(0, dtype=np.float64)
    arr = encode_candidates(candidates)
    value = predict_values(model, arr, device, chunk)
    reward = scaled_immediate_rewards(candidates, scale, penalty)
    done = candidate_dones(candidates)
                                                                          
                                                                              
    return reward + gamma * np.where(done, 0.0, value)


def _rank_top(entries, keep, *, score_index=0):
    """Stable tie-break in generation order (same as 1-ply argmax)."""
    return sorted(entries, key=lambda v: -v[score_index])[:keep]


def exhaustive_three_ply(env, roots, model, device, gamma, scale, penalty, chunk_size):
    """Read A/B/C only. Batch one root at a time to bound memory on dense boards."""
    best_action = None
    best_score = -np.inf
    simulated = len(roots)
    evaluated = 0
    root_rewards = scaled_immediate_rewards(roots, scale, penalty)
    for root, first_reward in zip(roots, root_rewards):
        action = int(root["action"])
        score = float(first_reward)
        if not (root["terminated"] or root["truncated"]):
            after_a = env.clone()
            after_a.step(action)
            seconds = after_a.preview_all()
            simulated += len(seconds)
            second_rewards = scaled_immediate_rewards(seconds, scale, penalty)
            leaves, prefixes = [], []
            score = -np.inf
            for second, second_reward in zip(seconds, second_rewards):
                prefix = float(first_reward) + gamma * float(second_reward)
                if second["terminated"] or second["truncated"]:
                    score = max(score, prefix)
                    continue
                after_b = after_a.clone()
                after_b.step(int(second["action"]))
                thirds = after_b.preview_all()
                leaves.extend(thirds)
                prefixes.extend([prefix] * len(thirds))
            if leaves:
                simulated += len(leaves)
                evaluated += len(leaves)
                last = _scores(leaves, model, device, gamma, scale, penalty, chunk_size)
                score = max(score, float(np.max(np.asarray(prefixes) + gamma**2 * last)))
        if score > best_score:
            best_score, best_action = score, action
    if best_action is None:
        raise RuntimeError("No legal path in 3-ply search")
    return SearchResult(best_action, best_score, evaluated, simulated, 3, len(roots))


def greedy_search(env, model, device, gamma: float, score_scale: float,
                  gameover_penalty: float, ply: int = 3, root_keep: int = 22,
                  second_keep: int = 484, chunk_size: int = 128,
                  root_candidates=None) -> SearchResult:
    """Find a real-game action.  No model training or env stepping occurs.

    Default evaluation keeps all 22 roots and all 484 two-placement paths.
    Smaller explicit limits enable beam pruning.  Terminal paths are
    considered as complete paths and never expanded beyond termination.
    """
    if ply not in (1, 3):
        raise ValueError('ply must be 1 or 3')
    if not root_keep or not second_keep or chunk_size < 1:
        raise ValueError('root_keep, second_keep, chunk_size must be positive')
    roots = root_candidates if root_candidates is not None else env.preview_all()
    if not roots:
        raise RuntimeError('No legal actions in a nonterminal state')
    if ply == 3 and root_keep >= len(roots) and second_keep >= 22 * len(roots):
        return exhaustive_three_ply(env, roots, model, device, gamma, score_scale,
                                    gameover_penalty, chunk_size)
    root_values = _scores(roots, model, device, gamma, score_scale,
                          gameover_penalty, chunk_size)
    chosen = int(np.argmax(root_values))
    if ply == 1:
        return SearchResult(int(roots[chosen]['action']), float(root_values[chosen]),
                            len(roots), len(roots), 1, len(roots))
    if not hasattr(env, 'clone'):
        raise RuntimeError('puyo_cpp PuyoEnv.clone() unavailable; rebuild updated rl_env/bindings.cpp')

                                                                       
    root_nodes = [(float(root_values[i]), int(c['action']), i)
                  for i, c in enumerate(roots) if not (c['terminated'] or c['truncated'])]
    completed = [(float(root_values[i]), int(c['action']))
                 for i, c in enumerate(roots) if c['terminated'] or c['truncated']]
    selected_roots = _rank_top(root_nodes, root_keep)
    n_sim = n_eval = len(roots)


    groups2 = []                                                  
    for _, root_action, i in selected_roots:
        state = env.clone()
        actual = state.step(root_action)
        if actual['terminated'] or actual['truncated']:
            completed.append((float(root_values[i]), root_action))
            continue
        second = state.preview_all()
        if not second:
            raise RuntimeError('Nonterminal cloned state with no legal actions')
        r0 = float(scaled_immediate_rewards([roots[i]], score_scale, gameover_penalty)[0])
        groups2.append((root_action, state, r0, second))
    flat2 = [candidate for _, _, _, cs in groups2 for candidate in cs]
    n_sim += len(flat2)
    n_eval += len(flat2)
    scores2 = _scores(flat2, model, device, gamma, score_scale,
                      gameover_penalty, chunk_size)
    depth2_entries = []                                                           
    offset = 0
    for root_action, state, r0, second in groups2:
        for candidate in second:
            stage2_score = r0 + gamma * float(scores2[offset])
            offset += 1
            if candidate['terminated'] or candidate['truncated']:
                completed.append((stage2_score, root_action))
            else:
                r1 = float(scaled_immediate_rewards(
                    [candidate], score_scale, gameover_penalty)[0])
                depth2_entries.append((stage2_score, root_action, state,
                                       int(candidate['action']), r0 + gamma * r1))

    selected_depth2 = _rank_top(depth2_entries, second_keep)
    groups3 = []                                         
    for _, root_action, depth1_env, second_action, reward_prefix in selected_depth2:
        state = depth1_env.clone()
        actual = state.step(second_action)
        if actual['terminated'] or actual['truncated']:
            completed.append((reward_prefix, root_action))
            continue
        third = state.preview_all()
        if not third:
            raise RuntimeError('Nonterminal depth-2 state with no legal actions')
        groups3.append((root_action, reward_prefix, third))
    flat3 = [candidate for _, _, cs in groups3 for candidate in cs]
    n_sim += len(flat3)
    n_eval += len(flat3)
    scores3 = _scores(flat3, model, device, gamma, score_scale,
                      gameover_penalty, chunk_size)
    offset = 0
    for root_action, reward_prefix, third in groups3:
        count = len(third)
                                                                            
                                         
        total = reward_prefix + gamma ** 2 * float(np.max(scores3[offset:offset + count]))
        completed.append((total, root_action))
        offset += count
    if not completed:
                                                                            
                                                                          
        raise RuntimeError('No completed 3-ply path')
    best = max(completed, key=lambda x: x[0])
    return SearchResult(best[1], best[0], n_eval, n_sim, 3, len(roots))
