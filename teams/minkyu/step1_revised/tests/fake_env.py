"""Deterministic test substitute for the C++ env (NOT a Puyo simulator)."""

import copy


class FakePuyoEnv:
    def __init__(self):
        self.turn = 0
        self.finish = False
        self.first_action = None
        self.score = 0
        self.seed = 0

    def clone(self):
        return copy.deepcopy(self)

    def reset(self, seed):
        self.__init__()
        self.seed = seed
        return self.observation()

    def observation(self):
        board = ['......'] * 13
        # Encode previous action and turn into board for deterministic probes.
        color = ('R', 'Y', 'G', 'B')[self.turn % 4]
        board[self.turn % 13] = color + '.....'
        return {
            'board': board, 'row14': 0,
            'visible': [('R', 'Y'), ('Y', 'G'), ('G', 'B')],
            'turn': self.turn,
        }

    def done(self):
        return self.finish

    def legal_actions(self):
        return [not self.finish, not self.finish]

    def step(self, action):
        if self.finish or action not in (0, 1):
            raise ValueError('invalid action')
        if self.turn == 0:
            self.first_action = action
            points = 4 if action == 0 else 3
        elif self.turn == 1:
            # Taking the slightly lower first-step reward (action 1)
            # enables a much larger future reward.
            points = 50 if self.first_action == 1 else 1
        else:
            points = 1
        self.turn += 1
        self.finish = self.turn >= 3
        self.score += points
        return {
            'observation': self.observation(),
            'chains': 5 if points >= 50 else 0,
            'score': points,
            'terminated': False,
            'truncated': self.finish,
        }

    def preview(self, action):
        return self.clone().step(action)

    def preview_all(self):
        if self.finish:
            return []
        return [dict(action=a, **self.preview(a)) for a in (0, 1)]
