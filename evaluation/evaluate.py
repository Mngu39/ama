import argparse
import importlib

from rl_env import puyo_cpp


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--policy",
        default="teams.template.random_baseline.eval_policy"
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100000)

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.seed < 0 or args.seed + args.episodes > 2**32:
        parser.error("seed range must fit uint32")

    module = importlib.import_module(args.policy)
    policy = module.make_policy(
        checkpoint=args.checkpoint,
        seed=args.seed
    )

    env = puyo_cpp.PuyoEnv()

    scores = []
    max_chains = []
    turns = []
    gameovers = 0
    truncations = 0

    for episode in range(args.episodes):
        observation = env.reset(args.seed + episode)

        reset_fn = getattr(policy, "reset_episode", None)
        if callable(reset_fn):
            reset_fn()

        episode_score = 0
        episode_max_chain = 0
        episode_turns = 0

        while not env.done():
            mask = env.legal_actions()

            action = policy.select_action(
                observation, mask
            )

            result = env.step(action)
            observation = result["observation"]

            episode_score += result["score"]
            episode_max_chain = max(
                episode_max_chain, result["chains"]
            )
            episode_turns += 1

            if result["terminated"]:
                gameovers += 1
            elif result["truncated"]:
                truncations += 1

        scores.append(episode_score)
        max_chains.append(episode_max_chain)
        turns.append(episode_turns)

    n = args.episodes

    print("\n=== Evaluation ===")
    print(f"Policy:          {args.policy}")
    print(f"Episodes:        {n}")
    print(f"Seed range:      {args.seed} - {args.seed+n-1}")
    print(f"Average score:   {sum(scores)/n:.2f}")
    print(f"Mean max chain:  {sum(max_chains)/n:.3f}")
    print(f"Average turns:   {sum(turns)/n:.2f}")
    print(f"Gameovers:       {gameovers}")
    print(f"Turn limits:     {truncations}")
    print("EVALUATION COMPLETED")


if __name__ == "__main__":
    main()