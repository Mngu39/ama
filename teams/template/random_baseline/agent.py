import random


class Agent:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def select_action(self, observation, legal_mask):
        legal = [
            i for i, allowed in enumerate(legal_mask)
            if allowed
        ]
        if not legal:
            raise ValueError("No legal actions")

        return self.rng.choice(legal)