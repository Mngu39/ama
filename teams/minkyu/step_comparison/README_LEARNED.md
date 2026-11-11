# Learned stochastic puyo generator — 온라인·오프라인

## 적용 범위

이 패치를 Ama 저장소 루트에 풀어 기존 `teams/minkyu/step_comparison/`의 동일 파일만 덮어쓴다.
기존 `observed`/`uniform`/`rule_based` 경로 및 엔진 C++ 코드는 변경하지 않는다.
**학습 중 실행되는 Ama 프로세스가 있다면 완료된 다음 적용한다.**
기존 `rl_env.PuyoEnv.fork_with_hidden_pairs` 바인딩이 설치되어 있어야 한다.

새 기능은 `--puyo-generator learned`이고, 다음 두 모드를 지원한다.

- `--learned-mode online` (기본): RL과 생성기 지도학습을 **동시에** 진행. 실제 게임의 턴 t에서 공개된 CURRENT/NEXT/NEXT2로 가상 미래를 먼저 생성하고, 실제 `env.step` 뒤 새 NEXT2가 공개된 경우에만 생성기 정답으로 사용. 가상 경로의 실제 숨은 큐에 대한 접근 없음.
- `--learned-mode frozen --learned-checkpoint <path>`: 별도 지도학습한 생성기 가중치를 로드하고 RL 도중 생성기 가중치 고정.
- `--learned-mode online --learned-checkpoint <path>`: 사전학습 가중치로 시작하여 공동학습 (선택사항).

모델: 순번별 학습 로그잇 + 현재 공개된 3쌍의 8차원 임베딩 + 32→64→32→16 MLP, **5,872 파라미터**. 교차엔트로피로 학습하고 16종 뿌요쌍을 **확률적으로** 생성한다. 각 가상 쌍을 다음 autoregressive 입력으로 사용한다. MAP/argmax는 사용하지 않는다. 학습에서 실제 Ama 생성 규칙이나 시드는 NN의 입력이 아니다.

`fork_with_hidden_pairs()`가 전체 128쌍 인자를 요구하므로, 온라인에서는 **실제로 필요한 n-step 범위까지만 NN이 생성**하고 나머지는 균등 난수로 덮어쓴다. 보조 값은 가상 TD 계산에 사용되지 않으며, 실제 미래 피스는 어느 위치에서도 참조하지 않는다.

## A. 오프라인 생성기 사전학습 (엔진에서 독립 실험 데이터 생성)

Ama 루트에서 `.venv` 활성화 후:

```bash
python -m teams.minkyu.step_comparison.gtr.learned_train \
  --engine-games 20000 \
  --engine-seed-start 600000 \
  --output teams/minkyu/step_comparison/weights/learned_offline.pt \
  --epochs 15 --batch-size 1024
```

`--engine-games`는 엔진 `reset/observation/legal_actions/step`만 사용해 **새로 공개되는** NEXT2를 정답으로 얻는다. 숨은 큐를 직접 열지 않는다. 게임오버 시 해당 게임을 종료한다. 엔진 시뮬레이션의 생존 여부에 따른 자료 선택 편향은 있을 수 있다. `--engine-seed-start`는 RL 학습·검증·시험 시드와 겹치지 않게 설정한다.

혹은 이전에 실제 경험한 착수 기록으로 학습:

```bash
python -m teams.minkyu.step_comparison.gtr.learned_train \
  --moves teams/minkyu/step_comparison/runs/n1_observed_20k/train_moves.csv.gz \
  --output teams/minkyu/step_comparison/weights/learned_from_moves.pt
```

이때 각 `turn=t`의 레이블은 같은 행에 있는 미공개 피스가 아니라 **다음 행(turn=t+1)에서 처음 공개된 NEXT2**이다. 마지막 행에는 다음 관측이 없어 레이블을 만들지 않는다. 검증은 개별 착수가 아닌 **게임 시드 단위 분할**이다. 이미 실행한 RL 실험의 같은 시드 데이터로 사전학습하면 공정한 독립 학습 예산 비교가 어려우므로 최종 생성모델 비교는 독립 게임 시드에서 사전학습하는 편이 낫다.

학습 완료 시 `.pt` 생성기 가중치와 `.history.json`의 Train/Validation NLL이 저장된다. Uniform NLL 기준은 `ln(16)≈2.77259` nats.

## B. 오프라인 생성기를 고정하여 15-step RL 학습

```bash
python -m teams.minkyu.step_comparison.puyo_train \
  --episodes 20000 --n-step 15 --puyo-generator learned \
  --learned-mode frozen \
  --learned-checkpoint teams/minkyu/step_comparison/weights/learned_offline.pt \
  --seed 12345 --device cuda \
  --channels 256 --blocks 6 \
  --form-weight 0.04 --gamma 0.995 \
  --capacity 30000 --per-mode replacement \
  --epsilon-start 0.4 --epsilon-final 0.02 --epsilon-steps 500000 \
  --train-ply 1 \
  --eval-every 500 --eval-episodes 100 --eval-ply 1 --eval-seed 200000 \
  --save-every 500 --resume-every 500 \
  --run-dir teams/minkyu/step_comparison/runs/n15_learned_frozen_20k
```

## C. 온라인 공동학습: RL 15-step + 생성기

```bash
python -m teams.minkyu.step_comparison.puyo_train \
  --episodes 20000 --n-step 15 --puyo-generator learned \
  --learned-mode online \
  --learned-lr 0.001 --learned-batch-size 128 \
  --learned-warmup 128 --learned-update-every 8 \
  --learned-capacity 30000 \
  --seed 12345 --device cuda \
  --channels 256 --blocks 6 \
  --form-weight 0.04 --gamma 0.995 \
  --capacity 30000 --per-mode replacement \
  --epsilon-start 0.4 --epsilon-final 0.02 --epsilon-steps 500000 \
  --train-ply 1 \
  --eval-every 500 --eval-episodes 100 --eval-ply 1 --eval-seed 200000 \
  --save-every 500 --resume-every 500 \
  --run-dir teams/minkyu/step_comparison/runs/n15_learned_online_20k
```

## D. 저장 및 재개

각 RL 가중치 `ep_000500.pt`, `last.pt`, `best_chain.pt`와 짝지어 `learned_ep_000500.pt`, `learned_last.pt`, `learned_best_chain.pt`를 별도 저장한다.

`last_resume.pt`에는 생성기 가중치, 생성기 optimizer, replay minibatch를 위한 RNG, 지도 데이터 순환 버퍼, 지도 업데이트 수까지 **함께 저장**한다. 재개 시 기존 실행과 동일한 설정 및 `--resume <기존 run>/last_resume.pt`, 새로운 `--run-dir`을 지정한다. 오프라인 고정 모드에서는 pretrained checkpoint의 경로를 동일하게 지정한다.

`learned_metrics.csv`에 에피소드별 누적 지도관측 수, 지도 업데이트 수, 최근 minibatch CE를 남긴다. 최근 minibatch CE는 독립 Test NLL이 아니며 온라인 모드에서 고정된 검증분포 평가를 자동으로 제공하지 않는다.

## 필수 검증

```bash
python -m pytest -q teams/minkyu/step_comparison/tests
```

개발 컨테이너에서 native Ama를 연결한 테스트 23개 통과. CPU 스모크: n=5, 3에피소드 온라인 및 오프라인 동작, 온라인 Resume와 중단 없는 실행의 가중치 일치 확인. **장기 20,000ep CUDA 학습과 최종 성능은 아직 실행·검증하지 않았다.**

## 해석상 주의

- 오프라인 생성기 학습에 사용한 **별도의 엔진 게임·계산비용**을 RL 20,000ep와 분리 기록해야 한다.
- 온라인 모델은 처음에 Uniform 분포에서 시작하여 실제 게임에서 점진적으로 학습하고, 오프라인 모드는 이미 학습된 분포에서 시작한다. 이 둘을 동일한 정보 예산이라고 해석하면 안 된다.
- `Learned` 모델은 공개된 3쌍과 위치만 조건으로 사용하며 전체 과거 시퀀스의 정확한 조건부 분포를 보장하지 않는다.
- 15-step 실제 가상 행동은 기존 ε-greedy 중간 행동 1개씩 + 마지막 Partial을 그대로 사용한다.
- 실제 게임 평가의 3-ply Beam Search 설정은 기존과 같다. 이 패치는 평가 엔진을 바꾸지 않는다.
