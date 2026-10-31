from .agent import Agent


def make_policy(checkpoint=None, seed=0):
    if checkpoint is not None:
        raise ValueError("Random baseline has no checkpoint")

    return Agent(seed=seed)