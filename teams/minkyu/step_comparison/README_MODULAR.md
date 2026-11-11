# GTR n-step 학습 코드 — 기능 변경 없는 모듈 분리본

원본: 업로드된 `minchan.zip`의 `minchan/short_vs_long/Local_Linux_WSL/`.

## 배치 방법

ZIP 안의 `gtr_step_comparison/` 폴더를 기존 Ama 저장소의 `teams/minkyu/` 아래에 추가하세요.
기존 루트 `core/`, `rl_env/`, `.venv/`, 그리고 다른 `teams/minkyu/` 프로젝트는 수정하거나 덮어쓰지 않습니다.

```text
ama/
├── core/                    # 기존 유지
├── rl_env/                  # 기존 유지
├── .venv/                   # 기존 유지
└── teams/minkyu/
    └── gtr_step_comparison/
        ├── puyo_train.py                 # 학습 루프 및 CLI 진입점
        ├── team_20k_config.json          # 원본 그대로
        ├── gtr/
        │   ├── model.py                 # CNN
        │   ├── reward.py                # GTR 보상·환경 래퍼
        │   ├── actions.py               # 상태 인코딩·후보 선택
        │   ├── replay.py                # n-step 경험·PER·학습 타깃
        │   ├── search.py                # 행동 탐색
        │   ├── records.py               # 결과·CSV·JSON 기록
        │   ├── checkpoints.py           # 모델 저장·재개
        │   ├── diagnostics.py           # 진단 프로브
        │   ├── evaluation.py            # 평가 실행
        │   └── train_utils.py           # CLI 인자·Optimizer·검증
        ├── analysis/
        │   ├── compare_runs.py
        │   └── horizon2_compare.py
        └── tests/
            ├── conftest.py              # 루트에서 pytest import 연결만 추가
            ├── test_nstep.py
            ├── test_best_chain_selection.py
            └── test_horizon2_compare.py
```

## 실행 및 테스트

Ama 저장소 루트에서 (기존 Python 환경을 활성화한 뒤):

```bash
python -m teams.minkyu.gtr_step_comparison.puyo_train --help
python -m teams.minkyu.gtr_step_comparison.puyo_train evaluate --help
python -m pytest -q teams/minkyu/gtr_step_comparison/tests
```

`cd teams/minkyu/gtr_step_comparison` 후 기존 방식의 `python puyo_train.py ...`도 지원합니다.
기존 `--n-step`, `--capacity`, `--epsilon-steps` 등의 실행 인자·기본값은 변경하지 않았습니다.

## 변경 범위·검증

- 원본 `puyo_train.py`의 함수·클래스·상수 정의 **69개 전부**, AST(추상 구문 트리) 기준으로 정확히 동일합니다. `train()` 본문과 CLI 처리 분기도 변경하지 않았습니다.
- 원본 `team_20k_config.json`, 분석 파일 2개, 테스트 파일 3개는 **바이트 단위로 그대로 복사**했습니다.
- 추가된 부분은 파일 간 import/공개 API 연결, `gtr/__init__.py`, 루트 pytest 실행을 위한 `tests/conftest.py`, 이 설명서뿐입니다.
- 기존 단위 테스트 12개 통과, `python -m ...` 및 직접 CLI의 도움말 동작 확인.
- 별도의 모의 게임 환경에서 **1/3/5-step** 각각 원본과 분리본의 CPU 학습·평가를 비교하여 가중치, Optimizer, Replay, RNG, 주요 CSV가 일치함을 확인했습니다(실행시간/실행 ID는 제외).
- 실제 Ama C++ 모듈과 사용자의 CUDA 환경에서 실행한 것은 아니므로, 실기기 학습 성공을 보장하는 검증은 아닙니다.

**의도적으로 수정하지 않은 기존 사항:** `analysis/compare_runs.py`에 남아 있는 평가 플라이 표기 문제, 학습 타깃의 1/3/5-step 정책 혼합, 미래 피스 생성모델 관련 미구현 부분. 후속 실험 수정은 이 원본 분리본과 별도로 진행해야 합니다.
