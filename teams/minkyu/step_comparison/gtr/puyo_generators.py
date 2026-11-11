"""Counterfactual hidden-pair generation, independent of the real queue.

At decision time t, the engine exposes only pairs t,t+1,t+2. All indices
>= t+3 on the fork are overwritten BEFORE simulating the selected action.

rule_based samples an independent full Ama queue using its real C++ RNG/shuffle
algorithm, then splices its hidden suffix. This is the correct UNCONDITIONAL
queue process, not the exact posterior conditioned on observed pairs/history.
"""
from dataclasses import dataclass
import numpy as np

COLORS = np.asarray(list('RYGB'))
QUEUE_PAIRS = 128


@dataclass(frozen=True)
class PuyoGenerator:
    name: str
    learned: object = None
    n_step: int = 1

    def fork(self, env, rng):
        if self.name == 'uniform':
            # Generate independently of the actual C++ queue. Pair order matters.
            draws = rng.integers(0, len(COLORS), size=(QUEUE_PAIRS, 2))
            full_queue = [''.join(COLORS[pair]) for pair in draws]
            return env.fork_with_hidden_pairs(full_queue)
        if self.name == 'rule_based':
            # Uses the SAME implementation cell::create_queue(seed) as Ama,
            # with an independent seed (not the environment's true seed).
            seed = int(rng.integers(0, 2**32, dtype=np.uint64))
            return env.fork_with_hidden_rule_seed(seed)
        if self.name == 'learned':
            if self.learned is None:
                raise ValueError('learned generator model must be supplied')
            observation=env.observation()
            turn=int(observation['turn'])
            # Only generate the hidden pairs that can affect n-step TD.
            needed=max(0, min(QUEUE_PAIRS-turn-3, 50-turn-3, self.n_step-2))
            learned_pairs=self.learned.sample_hidden(
                turn,observation['visible'],needed,rng)
            # Pad unused tail independently: engine's fixed queue storage needs
            # 128 pairs; these positions are never consumed by this rollout.
            pad=rng.integers(0,len(COLORS),size=(QUEUE_PAIRS,2))
            full_queue=[''.join(COLORS[pair]) for pair in pad]
            for idx,pair in enumerate(learned_pairs,start=turn+3):
                full_queue[idx]=pair
            return env.fork_with_hidden_pairs(full_queue)
        raise ValueError(f'Unknown model-based puyo generator: {self.name}')


def make_generator(name, learned=None, n_step=1):
    if name == 'observed':
        return None
    if name not in ('uniform', 'rule_based', 'learned'):
        raise ValueError(f'Unknown generator: {name}')
    return PuyoGenerator(name, learned=learned, n_step=n_step)


def per_placement_rng(base_seed, episode, turn, stream):
    # Stateless, replay/resume-safe, and does not consume the real actor RNG.
    return np.random.default_rng(np.random.SeedSequence(
        [int(base_seed), int(episode), int(turn), int(stream)]))
