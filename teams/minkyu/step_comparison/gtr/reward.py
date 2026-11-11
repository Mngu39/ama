"""Unmodified original definitions moved from puyo_train.py; imports are wiring only."""

import copy


FORM_PATTERNS = (
                                                                       
    (
        ((2, 2, 5, 0, 0, 0), (1, 1, 2, 5, 0, 0), (1, 2, 5, 0, 0, 0), (3, 3, 3, 4, 0, 0), (4, 4, 4, 0, 0, 0), (0, 0, 0, 0, 0, 0)),
        (0, 2, 1, 2, 0, 0),
        ((1, 2), (1, 3), (2, 3), (2, 5), (3, 4), (3, 5)),
    ),
                                                                        
    (
        ((3, 3, 2, 3, 6, 0), (1, 2, 2, 7, 0, 0), (1, 1, 3, 6, 0, 0), (4, 4, 4, 5, 0, 0), (5, 5, 5, 0, 0, 0), (0, 0, 0, 0, 0, 0)),
        (0, 2, 1, 1, 2, 0, 0, 0),
        ((1, 2), (1, 3), (1, 4), (2, 3), (2, 7), (3, 4), (3, 6), (4, 5)),
    ),
                                                                        
    (
        ((2, 3, 3, 6, 0, 0), (1, 2, 2, 3, 6, 0), (1, 1, 3, 6, 0, 0), (4, 4, 4, 5, 0, 0), (5, 5, 5, 0, 0, 0), (0, 0, 0, 0, 0, 0)),
        (0, 2, 1, 1, 2, 0, 0),
        ((1, 2), (1, 3), (1, 4), (2, 3), (3, 4), (3, 6), (4, 5)),
    ),
)


def compile_pattern(grid, weights, distinct):
    cells = [(x, y, grid[y][x]) for x in range(6) for y in range(6) if grid[y][x]]
    maximum = sum(weights[group] for _, _, group in cells)
    return cells, weights, distinct, maximum


COMPILED_FORMS = tuple(compile_pattern(*pattern) for pattern in FORM_PATTERNS)


def form_potential(board):
                                                                  
    if any('#' in row[:4] for row in board[:4]):
        return 0.0
    best = 0.0
    for cells, weights, distinct, maximum in COMPILED_FORMS:
        colors = [None] * len(weights)
        score = 0
        for x, y, group in cells:
            color = board[y][x]
            if color == '.':
                continue
            if colors[group] is None:
                colors[group] = color
            elif weights[group] and colors[group] != color:
                break
            score += weights[group]
        else:
            if all(colors[i] is None or colors[i] != colors[j] for i, j in distinct):
                best = max(best, score / maximum)
    return best


class RewardEnvironment:
                                                                             
    def __init__(self, engine, gamma, form_weight, score_scale):
        self.engine = engine
        self.gamma = gamma
        self.raw_weight = form_weight * score_scale
        self.potential = 0.0

    def reset(self, seed):
        obs = self.engine.reset(seed)
        self.potential = self._potential(obs)
        return obs

    def observation(self):
        return self.engine.observation()

    def done(self):
        return self.engine.done()

    def _potential(self, observation):
        return form_potential(observation['board']) if self.raw_weight else 0.0

    def _reward(self, result):
        terminal = result['terminated'] or result['truncated']
        after = 0.0 if terminal else self._potential(result['observation'])
        result['form_bonus'] = self.raw_weight * (self.gamma * after - self.potential)
        return after

    def step(self, action):
        result = self.engine.step(action)
        self.potential = self._reward(result)
        return result

    def preview_all(self):
        candidates = self.engine.preview_all()
        for candidate in candidates:
            self._reward(candidate)
        return candidates

    def clone(self):
        from copy import copy
        other = copy(self)
        other.engine = self.engine.clone()
        return other

    def fork_with_hidden_pairs(self, full_queue):
        # The C++ clone overwrites the hidden suffix only. The real environment
        # and its currently visible CURRENT/NEXT/NEXT2 remain unchanged.
        from copy import copy
        other = copy(self)
        other.engine = self.engine.fork_with_hidden_pairs(full_queue)
        return other

    def fork_with_hidden_rule_seed(self, seed):
        from copy import copy
        other = copy(self)
        other.engine = self.engine.fork_with_hidden_rule_seed(int(seed))
        return other


def make_environment(gamma, weight, scale):
    from rl_env import puyo_cpp
    engine = puyo_cpp.PuyoEnv()
    if not hasattr(engine, 'clone'):
        raise RuntimeError('Rebuild rl_env/puyo_cpp with scripts/setup_wsl.sh for clone support.')
    return RewardEnvironment(engine, gamma, weight, scale)
