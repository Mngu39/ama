import argparse

from rl_env import puyo_cpp
from .agent import Agent
from .reward import compute_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.seed < 0 or args.seed + args.episodes > 2**32:
        parser.error("seed range must fit uint32")

    env = puyo_cpp.PuyoEnv()
    agent = Agent(seed=args.seed)

    for episode in range(args.episodes):
        observation = env.reset(args.seed + episode)

        episode_score = 0
        episode_reward = 0.0
        max_chain = 0
        turns = 0

        while not env.done():
            mask = env.legal_actions()

            action = agent.select_action(observation, mask)
            result = env.step(action)

            reward = compute_reward(result)

            # Learning algorithms can update their models here.
            # This template's random agent does not learn.

            episode_score += result["score"]
            episode_reward += reward
            max_chain = max(max_chain, result["chains"])
            turns += 1

            observation = result["observation"]

        print(
            f"Episode {episode + 1}"
            f" | turns={turns}"
            f" | score={episode_score}"
            f" | max_chain={max_chain}"
            f" | reward={episode_reward:.2f}"
        )

    print("EXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()