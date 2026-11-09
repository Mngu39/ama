# Linux / WSL 로컬 20,000에피소드 실행

본 폴더는 `GTR_20k_Kaggle.ipynb`에 내장된 GTR 학습 소스와 **같은 학습기 및 실험 설정**을 사용합니다. Windows 네이티브 환경은 지원 검증 범위에 포함되지 않으며, Ubuntu 또는 Windows WSL Ubuntu를 기준으로 안내드립니다.

## 처음 1회 준비

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev g++
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pybind11 pytest pandas matplotlib
```

CUDA 지원 PyTorch는 사용하시는 NVIDIA 드라이버에 맞는 버전으로 설치해 주십시오. https://pytorch.org/get-started/locally/ 에서 설치 명령을 확인하실 수 있습니다.

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python build_engine.py
```

## 간단한 동작 테스트

```bash
bash scripts/run_smoke.sh
```

12에피소드 축소 모델을 학습하고 6·12에피소드에서 각 2게임씩 평가합니다. `SMOKE PASS`를 확인해 주십시오. 로컬 CUDA가 아직 없는 경우 `SMOKE_DEVICE=cpu bash scripts/run_smoke.sh`로 CPU 기능만 테스트할 수 있습니다.

## 20,000판 본 학습

```bash
bash scripts/run_20k.sh
```

새 모델을 0부터 학습하며 Buffer 30,000, Epsilon decay 500,000, 1-step TD, 학습 탐색 1-ply, 500판마다 1-ply 100게임 검증을 적용합니다. `runs/`에 실행 결과를 기록합니다. 같은 설정에서 중단된 학습을 이어가려면 다음과 같이 진행해 주십시오.

```bash
bash scripts/run_20k.sh runs/<이전실행폴더>/last_resume.pt
```

5,000판 파라미터 비교용 체크포인트는 본 학습의 `last_resume.pt`가 아니며, 학습 재개 입력으로 사용하시면 안 됩니다.

## 최종 A 평가

```bash
bash scripts/evaluate_A_3ply.sh runs/<20k실행폴더>/best_chain.pt
```

3-ply 250게임, 시드 100000~100249로 평가하며 B/C도 동일 모델 가중치와 평가 시드를 사용해 주십시오. `analysis/horizon2_compare.py`에서 A/B/C 로그를 비교할 수 있습니다.

## 검증 상황

- Python 기능 테스트 12개 통과 및 Kaggle Tesla T4의 축소 DEBUG 학습 성공을 확인하였습니다.
- 민규님 PC의 C++ 환경과 로컬 CUDA GPU는 아직 직접 검증하지 않았으므로, 본 학습 전 스모크 테스트를 실행해 주십시오.

## 실험 설계 확인

학습 중 `--eval-ply 1`은 최고 체크포인트를 고르기 위한 **내부 검증**입니다. 최종 A/B/C 발표용 평가는 3-ply 250판(시드 100000~100249)으로 별도 수행합니다. 따라서 1-ply 최고 체크포인트가 3-ply 기준 최고임을 보장하지 않습니다. 3-ply로 체크포인트를 선정해야 한다면 본 학습 이전에 팀에서 방식을 확정해 주십시오.

이 패키지는 `20,000판을 처음부터 새로 학습`하는 프로토콜입니다. 11월 7일 회의록에는 5k 사전 실험 후 20k 본 학습 방침이 기록되어 있지만, 새로 시작할지 5k에서 재개할지는 명시되어 있지 않아 이 점은 작업상 결정 사항임을 알려드립니다.
