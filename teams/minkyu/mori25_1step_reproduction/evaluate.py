"""Greedy 1-ply evaluation of an afterstate model.

This experiment needs env.preview_all(), so the shared evaluation/evaluate.py
(policy receives only observation + mask) is insufficient without an adapter.
"""

import argparse
from pathlib import Path
import numpy as np
import torch

from rl_env import puyo_cpp
from .network import AfterstateValueNet
from .policy import pick_action
from .train import select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_config = checkpoint["model_config"]
    value_config = checkpoint["value_config"]
    model = AfterstateValueNet(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    env = puyo_cpp.PuyoEnv()
    rng = np.random.default_rng(args.seed)
    scores, max_chains, turns, gameovers = [], [], [], 0
    for episode in range(args.episodes):
        env.reset(args.seed + episode)
        score, longest, moves = 0, 0, 0
        while not env.done():
            candidates = env.preview_all()
            picked, _ = pick_action(
                candidates, model, device, rng, epsilon=0.0,
                gamma=value_config["gamma"],
                score_scale=value_config["score_scale"],
                gameover_penalty=value_config["gameover_penalty"],
            )
            result = env.step(int(candidates[picked]["action"]))
            score += int(result["score"])
            longest = max(longest, int(result["chains"]))
            moves += 1
            if result["terminated"]:
                gameovers += 1
        scores.append(score)
        max_chains.append(longest)
        turns.append(moves)
    print(f"episodes={args.episodes} seeds={args.seed}..{args.seed + args.episodes - 1}")
    print(f"avg_score={np.mean(scores):.2f}")
    print(f"mean_max_chain={np.mean(max_chains):.3f}")
    print(f"avg_turns={np.mean(turns):.2f}")
    print(f"gameovers={gameovers}, turn_limits={args.episodes - gameovers}")


if __name__ == "__main__":
    main()
