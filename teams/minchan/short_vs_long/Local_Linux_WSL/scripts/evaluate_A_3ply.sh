#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
if [[ $# -ne 1 || ! -f "$1" ]]; then
  echo '사용법: bash scripts/evaluate_A_3ply.sh runs/<20k실험폴더>/best_chain.pt' >&2
  exit 2
fi
WEIGHTS="$(realpath "$1")"
STAMP="$(date -u +%Y%m%dT%H%M%S)_$$"
OUT="evaluations/A_3ply250_${STAMP}"
python -u puyo_train.py evaluate   --checkpoint "$WEIGHTS" --episodes 250 --seed 100000   --ply 3 --root-keep 22 --second-keep 484 --device cuda   --out-dir "$OUT"
printf '평가 결과: %s\n' "$OUT"
