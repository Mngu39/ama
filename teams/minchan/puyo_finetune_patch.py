"""Patch an existing generated train_full_resume.py without modifying its original.

Usage from repository root (after setup_puyo_safe_training.py has run):
 python /kaggle/input/<dataset>/puyo_finetune_patch.py --root /kaggle/working/ama

Emits: teams/minkyu/mori25_1step_reproduction/train_finetune.py
"""
import argparse
import ast
from pathlib import Path


def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Patch not applicable, expected 1 occurrence, found {count}: {old[:110]!r}")
    return text.replace(old, new, 1)


def patch(s):
    s = replace_once(
        s, '    return p.parse_args()',
        '''    p.add_argument("--eval-every", type=int, default=500,
                   help="Run greedy validation every N episodes; 0 disables")
    p.add_argument("--eval-games", type=int, default=100,
                   help="Number of fixed-seed greedy validation games")
    p.add_argument("--eval-seed", type=int, default=200000,
                   help="First validation game seed (separate from training seeds)")
    p.add_argument("--allow-lr-change", action="store_true",
                   help="Allow changing only --lr when full-state resuming")
    return p.parse_args()''')
    # Evaluation is separate from training RNG: separate env, separate numpy RNG.
    anchor = 'def train(args):'
    evaluation = '''@torch.no_grad()
def evaluate_greedy(model, device, args):
    """Fixed-seed one-ply greedy evaluation, without touching training env/RNG."""
    was_training = model.training
    model.eval()
    validation_env = puyo_cpp.PuyoEnv()
    validation_rng = np.random.default_rng(args.eval_seed + 99999)
    total_chains, total_scores = [], []
    try:
        for i in range(args.eval_games):
            validation_env.reset(args.eval_seed + i)
            candidates = validation_env.preview_all()
            score, maximum = 0, 0
            while not validation_env.done():
                picked, _ = pick_action(
                    candidates, model, device, validation_rng, 0.0,
                    args.gamma, args.score_scale, args.gameover_penalty,
                    args.infer_chunk,
                )
                result = validation_env.step(int(candidates[picked]["action"]))
                score += int(result["score"])
                maximum = max(maximum, int(result["chains"]))
                if bool(result["terminated"] or result["truncated"]):
                    break
                candidates = validation_env.preview_all()
            total_chains.append(maximum)
            total_scores.append(score)
    finally:
        model.train(was_training)
    return float(np.mean(total_chains)), float(np.mean(total_scores))


'''
    s = replace_once(s, anchor, evaluation + anchor)
    # extra flags are not part of model training hyperparameters and should not alter strict restore check.
    old_config_tail = '"save_every", "device")}'
    if s.count(old_config_tail) != 2:
        raise RuntimeError("Expected two training_config exclusions")
    s = s.replace(old_config_tail,
                  '"save_every", "device", "eval_every", "eval_games", "eval_seed", "allow_lr_change")}')
    s = replace_once(s,
        '''        if ck["training_config"] != config:
            raise ValueError("Training configuration differs from checkpoint")''',
        '''        previous_config = ck["training_config"]
        previous_except_lr = {k: v for k, v in previous_config.items() if k != "lr"}
        requested_except_lr = {k: v for k, v in config.items() if k != "lr"}
        if previous_except_lr != requested_except_lr:
            raise ValueError("Training configuration differs (other than learning rate)")
        if previous_config.get("lr") != config.get("lr") and not args.allow_lr_change:
            raise ValueError("Learning rate differs: pass --allow-lr-change explicitly")''')
    s = replace_once(s,
        '        optimizer.load_state_dict(ck["optimizer"])',
        '''        optimizer.load_state_dict(ck["optimizer"])
        # Adam optimizer checkpoint includes its *old* LR; override AFTER load.
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        print(f"FINETUNE lr={args.lr:.8g} (previous={previous_config.get('lr')})", flush=True)''')
    # Best model separate from full-state slots (lightweight eval-only checkpoint).
    s = replace_once(s,
        '    for episode in range(start_episode + 1, args.episodes + 1):',
        '''    if args.eval_every < 0 or args.eval_games < 1:
        raise ValueError("--eval-every must be >= 0 and --eval-games must be >= 1")
    best_chain = float("-inf")
    best_score = float("-inf")
    best_path = args.out.with_name(args.out.stem + "_best_model.pt")

    def validate_and_store(episode):
        nonlocal best_chain, best_score
        mean_chain, mean_score = evaluate_greedy(model, device, args)
        print(f"VALIDATION episode={episode} games={args.eval_games} "
              f"avg_max_chain={mean_chain:.5f} avg_score={mean_score:.2f}", flush=True)
        if (mean_chain, mean_score) > (best_chain, best_score):
            best_chain, best_score = mean_chain, mean_score
            best_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = best_path.with_suffix(best_path.suffix + ".tmp")
            try:
                torch.save({
                    "model": model.state_dict(), "episode": episode,
                    "avg_max_chain": mean_chain, "avg_score": mean_score,
                    "eval_games": args.eval_games, "eval_seed": args.eval_seed,
                    "model_config": {"channels": args.channels, "blocks": args.blocks,
                                     "input_channels": CHANNELS},
                    "value_config": {"gamma": args.gamma, "score_scale": args.score_scale,
                                     "gameover_penalty": args.gameover_penalty},
                }, tmp)
                os.replace(tmp, best_path)
            finally:
                if tmp.exists():
                    tmp.unlink()
            print(f"BEST MODEL saved={best_path}", flush=True)

    # Evaluate restored baseline before further training, to avoid picking a worse checkpoint.
    if args.eval_every > 0:
        validate_and_store(start_episode)

    for episode in range(start_episode + 1, args.episodes + 1):''')
    s = replace_once(s,
        '''        if args.save_every > 0 and episode % args.save_every == 0:
            save(episode)''',
        '''        if args.save_every > 0 and episode % args.save_every == 0:
            save(episode)
        if args.eval_every > 0 and episode % args.eval_every == 0:
            validate_and_store(episode)''')
    ast.parse(s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('/kaggle/working/ama'))
    args = ap.parse_args()
    base = args.root / 'teams/minkyu/mori25_1step_reproduction'
    source = base / 'train_full_resume.py'
    target = base / 'train_finetune.py'
    if not source.is_file():
        raise FileNotFoundError(f'{source} not found: run setup code cell first')
    target.write_text(patch(source.read_text(encoding='utf-8')), encoding='utf-8')
    print('CREATED:', target)


if __name__ == '__main__':
    main()
