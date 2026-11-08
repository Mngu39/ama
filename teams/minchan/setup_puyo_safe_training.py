from pathlib import Path
import ast

root = Path.home() / 'ama'
base = root / 'teams/minkyu/mori25_1step_reproduction'
src = base / 'train.py'
dst = base / 'train_full_resume.py'
if not src.exists():
    raise SystemExit(f'Cannot find {src}')
s = src.read_text(encoding='utf-8')

def change(old, new):
    global s
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f'Expected one match, found {n}: {old[:100]!r}')
    s = s.replace(old, new, 1)

change('import argparse\n', 'import argparse\nimport os\n')
change('    return p.parse_args()', '''    p.add_argument("--resume-from", type=Path, default=None)
    return p.parse_args()''')
change('    env_steps = 0\n    gradient_updates = 0', '''    env_steps = 0
    gradient_updates = 0
    start_episode = 0

    if args.resume_from is not None:
        ck = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if ck.get("format_version") != 1:
            raise ValueError("Not a full-state checkpoint from this training script")
        config = {k: v for k, v in vars(args).items()
                  if k not in ("episodes", "out", "resume_from", "save_every", "device")}
        if ck["training_config"] != config:
            raise ValueError("Training configuration differs from checkpoint")
        model.load_state_dict(ck["model"])
        target_model.load_state_dict(ck["target_model"])
        target_model.eval()
        optimizer.load_state_dict(ck["optimizer"])
        replay = ck["replay"]
        rng.bit_generator.state = ck["action_rng"]
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"])
        if device.type == "cuda" and ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"], device)
        start_episode = int(ck["episode"])
        env_steps = int(ck["env_steps"])
        gradient_updates = int(ck["gradient_updates"])
        if args.episodes <= start_episode:
            raise ValueError("--episodes must be the TOTAL target episode count, greater than saved episode")
        print(f"FULL RESUME episode={start_episode} env_steps={env_steps} "
              f"updates={gradient_updates}", flush=True)
''')
a = s.index('    def save(episode):')
b = s.index('    for episode in range(', a)
s = s[:a] + '''    def save(episode):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Alternate two checkpoint slots; atomic replacement protects the older slot.
        slot = (episode // max(1, args.save_every)) % 2
        path = args.out.with_name(f"{args.out.stem}_slot{slot}{args.out.suffix}")
        tmp = path.with_name(path.name + ".tmp")
        ck = {
            "format_version": 1,
            "model": model.state_dict(),
            "target_model": target_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "replay": replay,
            "episode": episode,
            "env_steps": env_steps,
            "gradient_updates": gradient_updates,
            "action_rng": rng.bit_generator.state,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "training_config": {k: v for k, v in vars(args).items()
                                if k not in ("episodes", "out", "resume_from", "save_every", "device")},
        }
        try:
            torch.save(ck, tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        print(f"checkpoint={path}", flush=True)

''' + s[b:]
change('    for episode in range(1, args.episodes + 1):', '    for episode in range(start_episode + 1, args.episodes + 1):')
ast.parse(s)
dst.write_text(s, encoding='utf-8')
print(f'CREATED: {dst}')
