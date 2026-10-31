
import argparse
import importlib

from rl_env import puyo_cpp


ALPHABET = (
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[]"
)

COLOR_ID = {
    "R": 0,
    "G": 1,
    "B": 2,
    "Y": 3,
}


def decode_action(action):
    if 0 <= action <= 5:
        return action, 0

    if 6 <= action <= 10:
        return action - 6, 1

    if 11 <= action <= 16:
        return action - 11, 2

    if 17 <= action <= 21:
        return action - 16, 3

    raise ValueError(f"Invalid action: {action}")


def encode_placement(pair, action):
    x, rotation = decode_action(action)

    first, second = pair
    pair_code = (
        COLOR_ID[first] * 5
        + COLOR_ID[second]
    )

    placement_code = ((x + 1) << 2) + rotation
    code = pair_code | (placement_code << 7)

    return (
        ALPHABET[code & 0x3F]
        + ALPHABET[(code >> 6) & 0x3F]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        default="teams.template.random_baseline.eval_policy",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    if not 0 <= args.seed <= 0xFFFFFFFF:
        parser.error("seed must fit uint32")

    module = importlib.import_module(args.policy)

    policy = module.make_policy(
        checkpoint=args.checkpoint,
        seed=args.seed,
    )

    reset_fn = getattr(policy, "reset_episode", None)
    if callable(reset_fn):
        reset_fn()

    env = puyo_cpp.PuyoEnv()
    observation = env.reset(args.seed)

    encoded_moves = []
    total_score = 0
    max_chain = 0

    while not env.done():
        mask = env.legal_actions()

        action = policy.select_action(
            observation, mask
        )

        if (
            not isinstance(action, int)
            or action < 0
            or action >= puyo_cpp.ACTION_COUNT
            or not mask[action]
        ):
            raise ValueError(
                f"Policy selected illegal action: {action}"
            )

        pair = observation["visible"][0]
        encoded_moves.append(
            encode_placement(pair, action)
        )

        result = env.step(action)

        total_score += result["score"]
        max_chain = max(
            max_chain, result["chains"]
        )

        observation = result["observation"]

    replay_url = (
        "http://www.puyop.com/s/_"
        + "".join(encoded_moves)
    )

    print(f"Seed: {args.seed}")
    print(f"Turns: {observation['turn']}")
    print(f"Total score: {total_score}")
    print(f"Max chain: {max_chain}")
    print(f"Replay: {replay_url}")


if __name__ == "__main__":
    main()
