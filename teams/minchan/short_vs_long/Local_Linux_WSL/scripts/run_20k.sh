#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1
python - <<'PYT'
import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA GPU를 찾을 수 없습니다. PyTorch CUDA 설치와 GPU 설정을 확인해 주십시오.')
print('CUDA GPU:', torch.cuda.get_device_name(0))
PYT
STAMP="$(date -u +%Y%m%dT%H%M%S)_$$"
DEST="runs/H2_GTR_20k_B30k_E500k_n1_${STAMP}"
CMD=(python -u puyo_train.py
  --episodes 20000 --n-step 1 --seed 12345 --device cuda
  --train-ply 1 --eval-ply 1
  --channels 256 --blocks 6 --lr 0.0001 --batch-size 16 --gamma 0.995
  --form-weight 0.04 --capacity 30000 --warmup 256 --update-every 4
  --target-sync 500 --per-alpha 0.6 --per-beta-start 0.4 --per-beta-steps 50000
  --per-mode replacement --epsilon-start 0.4 --epsilon-final 0.02
  --epsilon-steps 500000 --eval-every 500 --eval-episodes 100 --eval-seed 200000
  --save-every 500 --resume-every 500 --run-dir "$DEST")
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 1 || ! -f "$1" ]]; then
    echo '사용법: bash scripts/run_20k.sh [이전 last_resume.pt 파일 경로]' >&2
    exit 2
  fi
  CMD+=(--resume "$(realpath "$1")")
fi
mkdir -p runs
printf '출력 디렉터리: %s\n' "$DEST"
"${CMD[@]}" 2>&1 | tee "${DEST}_console.log"
printf '학습 결과: %s\n' "$DEST"
