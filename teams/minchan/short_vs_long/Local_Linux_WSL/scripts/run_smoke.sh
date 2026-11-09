#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1
DEVICE="${SMOKE_DEVICE:-cuda}"
STAMP="$(date -u +%Y%m%dT%H%M%S)_$$"
DEST="runs/DEBUG_B30k_E500k_n1_${STAMP}"
python - <<'PYT'
import torch
print('torch',torch.__version__,'cuda',torch.cuda.is_available())
PYT
python puyo_train.py   --episodes 12 --n-step 1 --seed 12345   --device "$DEVICE" --train-ply 1 --eval-ply 1   --channels 8 --blocks 1 --batch-size 4 --capacity 64 --warmup 4   --epsilon-start 0.4 --epsilon-final 0.02 --epsilon-steps 500000   --lr 0.0001 --gamma 0.995 --form-weight 0.04   --per-alpha 0.6 --per-beta-start 0.4 --per-beta-steps 50000   --eval-every 6 --eval-episodes 2 --eval-seed 200000   --save-every 6 --resume-every 6 --no-probes --no-train-moves   --run-dir "$DEST"
python - "$DEST" <<'PYT'
from pathlib import Path
import json,sys
p=Path(sys.argv[1]); complete=json.loads((p/'completed.json').read_text())
assert int(complete.get('gradient_updates',0))>0,complete
for name in ('best_chain.pt','best_chain_selection.json','eval_metrics.csv','train_episodes.csv','last_resume.pt','ep_000012.pt'):
    assert (p/name).is_file(),f'missing {name}'
print('SMOKE PASS',p.resolve(),'updates=',complete.get('gradient_updates'))
PYT
