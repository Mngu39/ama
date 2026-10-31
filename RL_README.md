# Puyo Puyo Reinforcement Learning Project

이 프로젝트는 오픈소스 Puyo Puyo Tsu AI **Ama**의 게임 시뮬레이터를 활용하여 다양한 강화학습 알고리즘, 신경망 구조, 보상함수 및 학습 전략을 실험하기 위한 팀프로젝트입니다.

**GitHub 저장소:** https://github.com/Mngu39/ama

## 프로젝트 기본 원칙

- **공통 게임 환경:** 뿌요 생성, 착수, 연쇄, 점수 계산 등은 동일한 C++ 시뮬레이터 사용
- **개별 학습 실험:** 각 팀원은 자신의 폴더에서 모델, 보상함수, 학습 방법, 하이퍼파라미터를 자유롭게 구성
- **공통 성능 비교:** 학습 보상과 별개로 동일한 시드와 환경에서 실제 연쇄·점수 등을 측정
- **다중 실험 허용:** 한 팀원이 여러 개의 모델과 아이디어를 독립적으로 관리 가능

## 실행 방법 선택

| 환경 | 권장 용도 | 준비 사항 |
|---|---|---|
| 로컬 Ubuntu / Windows WSL2 | 개발, 디버깅, 반복 학습 | Git, Python, C++ 컴파일러 |
| GitHub Actions | 간단한 실행·검증·평가 | GitHub 계정과 워크플로 설정 |
| Google Colab | 로컬 설치 없이 클라우드에서 실험, 선택적으로 GPU 사용 | Google 계정 |

**처음 시작한다면:** [설치 및 협업 안내](docs/SETUP.md)를 참고하세요.

## 프로젝트 구조

| 경로 | 용도 |
|---|---|
| `core/`, `ai/`, `puyop/` | 기존 Ama 소스 |
| `rl_env/` | C++ 기반 RL 게임 환경 및 Python 바인딩 |
| `teams/` | 팀원별 모델 및 학습 실험 |
| `teams/template/random_baseline/` | 무작위 정책 예제 |
| `evaluation/` | 공통 성능 평가 |
| `scripts/setup_wsl.sh` | WSL/Ubuntu 초기 설치 및 빌드 |
| `docs/` | 프로젝트 사용 설명서 |

## 빠른 시작 — Ubuntu / WSL2

```bash
sudo apt update
sudo apt install -y git build-essential python3-venv python3-dev

git clone https://github.com/Mngu39/ama.git
cd ama

bash scripts/setup_wsl.sh
source .venv/bin/activate
```

무작위 정책을 10에피소드 실행:

```bash
python -m teams.template.random_baseline.train \
  --episodes 10 --seed 12345
```

공통 평가:

```bash
python -m evaluation.evaluate \
  --policy teams.template.random_baseline.eval_policy \
  --episodes 100 --seed 100000
```

현재 무작위 정책은 RL 학습을 수행하지 않는 예제입니다.

## 팀원별 실험

예시 디렉터리:

```text
teams/
├── template/
│   └── random_baseline/
├── member_a/
│   ├── td_3step/
│   ├── dqn_score/
│   └── dqn_chain/
└── member_b/
    ├── actor_critic/
    └── custom_model/
```

각 실험에는 `agent.py`, `reward.py`, `train.py`, `eval_policy.py` 등을 자유롭게 작성할 수 있고, 여러 파일로 분리해도 됩니다.

공통 평가 프로그램과 연결하려면 평가용 정책 인터페이스만 준수하면 됩니다.

## 상세 문서

1. [SETUP.md](docs/SETUP.md) — 로컬·GitHub·Colab 실행 및 팀원별 파일 관리
2. [ENV_API.md](docs/ENV_API.md) — 게임 상태, 행동 공간, 관측값, 보상함수 설계
3. [EVALUATION.md](docs/EVALUATION.md) — 성능 지표, 비교 실험, 플레이 시각화

## 원본 Ama 프로젝트 및 출처

이 저장소는 **citrus610이 개발한 Ama (Puyo Puyo Tsu AI)**를 Fork하여 강화학습 실험 환경을 추가한 프로젝트입니다.

- **원본 저장소:** https://github.com/citrus610/ama
- **원본 개발자:** citrus610
- **원본 프로젝트명:** Ama — Puyo Puyo Tsu AI
- **라이선스:** MIT License
- **라이선스 원문:** [LICENSE](LICENSE)

게임 시뮬레이션과 기존 탐색 AI의 주요 구현은 원본 Ama 프로젝트를 기반으로 합니다. 본 팀프로젝트에서는 게임 환경의 Python 인터페이스, RL 실험 구조 및 평가 코드를 추가·확장합니다.

원본 프로젝트의 저작권 및 라이선스 고지를 유지하며, 외부 라이브러리·논문을 추가로 활용하는 경우 해당 출처도 별도로 명시합니다.