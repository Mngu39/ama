"""Reconstruct fixed-context generator distributions from EXISTING learned_ep_*.pt.

No RL re-training or edits to checkpoint weights. Writes into a separate
analysis directory, leaving the original training run untouched.
"""
import argparse
import csv
import json
from pathlib import Path

from .learned_generator import PairNet, load_learned
from .learned_monitor import LearnedMonitor


def backfill(run_dir, *, reference_games=512, reference_seed_start=700000,
             output_dir=None, include_uniform_initial=True):
    run_dir = Path(run_dir)
    manifest_path = run_dir / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Missing training manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('learned_generator') is None:
        raise ValueError('Run was not trained with a learned puyo generator')
    generator = manifest['learned_generator']
    # Traverse parent resumes so interrupted/continued runs form one timeline.
    # Later checkpoints take precedence if the same episode exists in both.
    timeline = {}
    seen_dirs = set()
    def add_ancestors(directory):
        directory = directory.resolve()
        if directory in seen_dirs:
            raise ValueError('Circular parent resume chain')
        seen_dirs.add(directory)
        info_path = directory / 'manifest.json'
        if info_path.is_file():
            info = json.loads(info_path.read_text(encoding='utf-8'))
            parent = info.get('parent_resume_checkpoint')
            if parent:
                parent_file = Path(parent)
                if not parent_file.is_file():
                    raise FileNotFoundError(f'Parent resume checkpoint missing: {parent_file}')
                add_ancestors(parent_file.parent)
        for path in directory.glob('learned_ep_*.pt'):
            try:
                ep = int(path.stem.split('_')[-1])
            except ValueError:
                continue
            timeline[ep] = path
    add_ancestors(run_dir)
    checkpoints = sorted(timeline.items())
    if not checkpoints:
        raise FileNotFoundError(f'No learned_ep_*.pt snapshots in {run_dir}')
    out = Path(output_dir) if output_dir is not None else run_dir / 'learned_distribution_analysis'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Analysis output is nonempty: {out}; use another --output-dir')
    monitor = LearnedMonitor.from_engine(out, games=reference_games,
                                         seed_start=reference_seed_start)
    if include_uniform_initial and checkpoints[0][0] > 0 and not generator.get('pretrained_checkpoint'):
        # PairNet has IDENTICALLY ZERO output logits at initialization,
        # independently of its randomly initialized hidden weights.
        monitor.snapshot(PairNet(), 0, mode=generator['mode'], observations=0, updates=0)
        note = {'episode_zero': 'uniform distribution reconstructed from PairNet initialization; '
                               'not an original checkpoint; initial hidden weights not reconstructed'}
    else:
        note = {'episode_zero': 'only included if the original learned_ep_000000.pt exists'}
    model = PairNet()
    for episode, path in checkpoints:
        model, metadata = load_learned(path, model)
        results = monitor.snapshot(model, episode, mode=metadata.get('mode',generator['mode']),
                                   observations=metadata.get('observations',0),
                                   updates=metadata.get('updates',0))
        print(f'backfill episode={episode} heldout_nll={results["heldout_nll"]:.5f}', flush=True)
    (out / 'backfill_provenance.json').write_text(json.dumps({
        'source_run':str(run_dir),'source_checkpoints':[str(p) for _,p in checkpoints],
        'checkpoints_examined':len(checkpoints),'reference_games':reference_games,
        'reference_seed_start':reference_seed_start,**note,
    },ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'BACKFILL COMPLETED: {out}',flush=True)
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(description='Backfill learned generator distribution history from saved checkpoints')
    parser.add_argument('--run-dir', type=Path,required=True)
    parser.add_argument('--reference-games',type=int,default=512)
    parser.add_argument('--reference-seed-start',type=int,default=700000)
    parser.add_argument('--output-dir',type=Path,default=None)
    args=parser.parse_args(argv)
    backfill(args.run_dir, reference_games=args.reference_games,
             reference_seed_start=args.reference_seed_start, output_dir=args.output_dir)

if __name__=='__main__':
    main()
