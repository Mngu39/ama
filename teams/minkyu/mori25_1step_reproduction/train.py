"""Paper-inspired 1-step afterstate TD training.

From the repository root:
  python -m teams.minkyu.mori25_1step_reproduction.train --episodes 10

Value definition: V(x_t) is the discounted *future* score AFTER placement t.
Immediate score from placement t is added separately when selecting action t.
Consequently the one-step TD target for x_t uses the NEXT placement:
  target(x_t) = max_a {reward_(t+1)(a) + gamma * V_target(x_(t+1)(a))}
The revealed next piece is part of the sampled transition, NOT network input.
"""

import argparse
from collections import deque
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from rl_env import puyo_cpp
from .features import CHANNELS, HEIGHT, WIDTH
from .network import AfterstateValueNet
from .policy import (candidate_dones, encode_candidates, pick_action,
                     scaled_immediate_rewards)
from .replay_buffer import PrioritizedReplay, Transition
from .td import td_targets


def select_device(requested: str):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Try --device cpu; check PyTorch CUDA installation.")
    return torch.device(requested)


def linear_schedule(start: float, end: float, step: int, duration: int):
    ratio = min(1.0, step / max(1, duration))
    return start + ratio * (end - start)


def format_duration(seconds: float) -> str:
    """Display elapsed wall time as HH:MM:SS (also supports runs over 24 h)."""
    total = max(0, int(seconds + 0.5))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def optimize(model, target_model, optimizer, replay, device, args, beta):
    experiences, indices, weights = replay.sample(args.batch_size, beta)
    states = torch.as_tensor(np.stack([t.state for t in experiences]),
                             dtype=torch.float32, device=device)
    targets_np = td_targets(experiences, target_model, device,
                            args.gamma, args.infer_chunk)
    targets = torch.as_tensor(targets_np, device=device)
    importance_weights = torch.as_tensor(weights, device=device)

    model.train()
    predictions = model(states)
    errors = targets - predictions
    per_item_loss = F.smooth_l1_loss(predictions, targets, reduction="none")
    loss = (per_item_loss * importance_weights).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    replay.update_priorities(indices, errors.detach().cpu().numpy())
    return float(loss.item())


def parse_args():
    p = argparse.ArgumentParser(description="Train 1-step TD scalar afterstate value")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--channels", type=int, default=256,
                   help="2025 paper-inspired network: 256")
    p.add_argument("--blocks", type=int, default=6,
                   help="2025 paper comparison option: 6 residual blocks")
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--score-scale", type=float, default=1000.0,
                   help="Neural network predicts reward in units of this many points")
    p.add_argument("--gameover-penalty", type=float, default=-1000.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--capacity", type=int, default=10000)
    p.add_argument("--warmup", type=int, default=256,
                   help="Number of saved afterstates before gradient updates")
    p.add_argument("--update-every", type=int, default=4,
                   help="Environment placements per gradient update")
    p.add_argument("--target-sync", type=int, default=500,
                   help="Gradient updates between target-network copies")
    p.add_argument("--per-alpha", type=float, default=0.6)
    p.add_argument("--per-beta-start", type=float, default=0.4)
    p.add_argument("--per-beta-steps", type=int, default=50000)
    p.add_argument("--epsilon-start", type=float, default=0.4)
    p.add_argument("--epsilon-final", type=float, default=0.02)
    p.add_argument("--epsilon-steps", type=int, default=20000)
    p.add_argument("--infer-chunk", type=int, default=128)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "checkpoints" / "last.pt")
    return p.parse_args()


def train(args):
    if args.episodes < 1 or args.batch_size < 1 or args.update_every < 1 or args.score_scale <= 0:
        raise ValueError("episodes, batch-size, update-every, score-scale must be positive")
    run_started = time.perf_counter()  # Includes setup, training, and checkpoint I/O.
    recent_episode_times = deque(maxlen=10)  # Rolling average for the ETA estimate.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    print(f"device={device}, model={args.blocks} residual blocks x {args.channels} channels")
    print("value=afterstate FUTURE returns; target=1-step Bellman max; replay=PER")

    model = AfterstateValueNet(CHANNELS, args.channels, args.blocks).to(device)
    target_model = AfterstateValueNet(CHANNELS, args.channels, args.blocks).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    replay = PrioritizedReplay(args.capacity, args.per_alpha, args.seed + 1)
    rng = np.random.default_rng(args.seed + 2)
    env = puyo_cpp.PuyoEnv()
    env_steps = 0
    gradient_updates = 0

    def save(episode):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "episode": episode,
            "env_steps": env_steps,
            "gradient_updates": gradient_updates,
            "elapsed_seconds": time.perf_counter() - run_started,
            "model_config": {"channels": args.channels, "blocks": args.blocks,
                             "input_channels": CHANNELS},
            "value_config": {"gamma": args.gamma, "score_scale": args.score_scale,
                             "gameover_penalty": args.gameover_penalty},
        }, args.out)
        print(f"checkpoint={args.out}", flush=True)

    for episode in range(1, args.episodes + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)  # Include completed GPU work in wall time.
        episode_started = time.perf_counter()
        env.reset(args.seed + episode - 1)
        candidates = env.preview_all()
        ep_score, ep_max_chain, turns, ep_loss = 0, 0, 0, []
        while not env.done():
            epsilon = linear_schedule(args.epsilon_start, args.epsilon_final,
                                      env_steps, args.epsilon_steps)
            picked, encoded = pick_action(
                candidates, model, device, rng, epsilon, args.gamma,
                args.score_scale, args.gameover_penalty, args.infer_chunk,
            )
            action = int(candidates[picked]["action"])
            result = env.step(action)
            ep_score += int(result["score"])
            ep_max_chain = max(ep_max_chain, int(result["chains"]))
            turns += 1
            env_steps += 1

            terminal = bool(result["terminated"] or result["truncated"])
            following = [] if terminal else env.preview_all()
            replay.add(Transition(
                state=encoded[picked],
                next_states=(encode_candidates(following) if following else
                             np.empty((0, CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)),
                next_rewards=(scaled_immediate_rewards(
                    following, args.score_scale, args.gameover_penalty)
                    if following else np.empty(0, dtype=np.float32)),
                next_dones=(candidate_dones(following) if following else
                            np.empty(0, dtype=np.bool_)),
                terminal=terminal,
            ))
            candidates = following
            if len(replay) >= max(args.warmup, args.batch_size) and env_steps % args.update_every == 0:
                beta = linear_schedule(args.per_beta_start, 1.0,
                                       gradient_updates, args.per_beta_steps)
                ep_loss.append(optimize(model, target_model, optimizer,
                                        replay, device, args, beta))
                gradient_updates += 1
                if gradient_updates % args.target_sync == 0:
                    target_model.load_state_dict(model.state_dict())
                    target_model.eval()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        episode_seconds = time.perf_counter() - episode_started
        recent_episode_times.append(episode_seconds)
        elapsed_seconds = time.perf_counter() - run_started
        eta_seconds = (sum(recent_episode_times) / len(recent_episode_times)
                       * (args.episodes - episode))
        loss_text = f"{float(np.mean(ep_loss)):.5f}" if ep_loss else "n/a"
        print(f"episode={episode}/{args.episodes} turns={turns} score={ep_score} "
              f"max_chain={ep_max_chain} eps={epsilon:.3f} "
              f"replay={len(replay)} updates={gradient_updates} "
              f"avg_loss={loss_text} "
              f"ep_time={format_duration(episode_seconds)} "
              f"elapsed={format_duration(elapsed_seconds)} "
              f"eta~{format_duration(eta_seconds)}", flush=True)
        if args.save_every > 0 and episode % args.save_every == 0:
            save(episode)
    save(args.episodes)
    total_seconds = time.perf_counter() - run_started
    print(f"TRAINING COMPLETED total_time={format_duration(total_seconds)} "
          f"seconds={total_seconds:.1f} episodes={args.episodes} "
          f"env_steps={env_steps} gradient_updates={gradient_updates} "
          f"steps_per_sec={env_steps / max(total_seconds, 1e-9):.2f}",
          flush=True)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
