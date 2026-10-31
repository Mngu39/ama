# Puyo RL Environment — API 및 변수 가이드

이 문서는 강화학습 모델과 보상함수 설계에 사용할 수 있는 게임 환경 정보를 설명합니다.

**중요:** 현재 Python 인터페이스에서 직접 제공되는 값과, 이를 가공하여 계산할 수 있는 값, 향후 C++ 연결이 필요한 값은 서로 구분합니다.

## 1. 기본 사용법

```python
from rl_env import puyo_cpp

env = puyo_cpp.PuyoEnv()
obs = env.reset(12345)

while not env.done():
    legal_mask = env.legal_actions()
    action = next(i for i, valid in enumerate(legal_mask) if valid)

    result = env.step(action)
    obs = result["observation"]
```

현재 공통 인터페이스의 핵심은 `PuyoEnv` 클래스입니다.

## 2. 함수 및 상수 전체 목록

| 이름 | 입력 | 반환 | 기능 |
|---|---|---|---|
| `puyo_cpp.PuyoEnv()` | 없음 | 환경 객체 | 게임 환경 생성 |
| `env.reset(seed)` | 정수 시드 | 관측값 `dict` | 게임 초기화 |
| `env.observation()` | 없음 | 관측값 `dict` | 현재 상태 조회 |
| `env.legal_actions()` | 없음 | 길이 22의 `bool` 목록 | 합법적 행동 확인 |
| `env.step(action)` | 행동 ID (0~21) | 결과 `dict` | 착수 실행 및 연쇄 계산 |
| `env.done()` | 없음 | `bool` | 에피소드 종료 여부 |
| `puyo_cpp.ACTION_COUNT` | — | `22` | 행동 공간 크기 |
| `puyo_cpp.MAX_TURNS` | — | `50` | 현재 임시 착수 제한 |

### `reset(seed)`

```python
obs = env.reset(12345)
```

게임판을 초기화하고 시드에 따른 뿌요 시퀀스를 생성합니다.

같은 시드와 같은 착수열을 적용하면 동일한 게임 진행을 재현할 수 있도록 설계되어 있습니다.

현재 C++ 구현의 `seed`는 32비트 부호 없는 정수 범위입니다.

### `observation()`

현재 판면과 공개된 뿌요 시퀀스 및 턴 번호를 반환합니다.

### `legal_actions()`

```python
mask = env.legal_actions()
```

항상 길이 22의 `bool` 목록을 반환합니다.

예를 들어 `mask[5] == True`이면 5번 착수가 가능하며, `False`이면 현재 선택할 수 없습니다.

### `step(action)`

```python
result = env.step(3)
```

해당 착수를 실행하고 연쇄 소멸 및 점수 계산을 수행한 후 다음 상태를 반환합니다.

불가능한 행동을 선택하면 예외가 발생합니다. 에피소드 종료 후 추가 착수도 허용되지 않습니다.

### `done()`

```python
if env.done():
    # Episode finished
    ...
```

게임오버 또는 착수 제한으로 에피소드가 끝난 상태를 나타냅니다.

---

## 3. 관측값 Observation

`reset()`, `observation()` 및 `step()`의 `result["observation"]`에서 얻을 수 있습니다.

```python
obs = env.observation()

board = obs["board"]
row14 = obs["row14"]
visible = obs["visible"]
turn = obs["turn"]
```

| 변수 | Python 형식 | 의미 |
|---|---|---|
| `board` | `list[str]` | 13행 × 6열 게임판 |
| `row14` | `int` | 14번째 행의 점유 상태 비트마스크 |
| `visible` | `list[tuple[str, str]]` | 현재 뿌요 및 NEXT/NEXT2 |
| `turn` | `int` | 현재까지 진행한 착수 수 |

### 3.1 `board` — 게임판

13개 문자열이며 각 문자열은 6개 문자로 구성됩니다.

```python
board = obs["board"]

print(board[0])       # 맨 아래 행
print(board[12])      # 가장 위의 관측 행
print(board[0][2])    # 아래에서 첫 행, 세 번째 열
```

좌표계:

- `x = 0~5`: 왼쪽에서 오른쪽
- `y = 0~12`: 아래에서 위
- `board[y][x]`: 해당 좌표의 셀

현재 가능한 셀 문자:

| 문자 | 의미 |
|---|---|
| `.` | 빈 칸 |
| `R` | 빨강 |
| `Y` | 노랑 |
| `G` | 초록 |
| `B` | 파랑 |
| `#` | 방해뿌요(Garbage) |

게임판을 사람이 보기 쉬운 순서로 출력하려면:

```python
for row in reversed(board):
    print(row)
```

### 3.2 `row14` — 상단 추가 행 정보

```python
row14 = obs["row14"]
```

6개 열의 추가 행 점유 상태를 비트로 표현하는 내부 값입니다.

특정 열의 비트를 확인하는 예:

```python
occupied = bool(row14 & (1 << x))
```

이 값은 `board`의 13개 행에 포함되지 않는 정보를 제공합니다. 단, 개별 색상은 제공하지 않습니다.

**주의:** 현재 게임판 관측을 13행으로 제한한 설계가 학습에 충분한지는 검토가 필요합니다. `row14`만으로 상단 셀의 색상을 복원할 수는 없습니다.

### 3.3 `visible` — 공개된 뿌요 시퀀스

```python
visible = obs["visible"]

current = visible[0]
next_pair = visible[1]
next2_pair = visible[2]
```

각 원소는 두 뿌요 색상을 나타내는 튜플입니다.

예:

```python
("R", "Y")
```

첫 번째 값은 축 뿌요(axis), 두 번째 값은 자식 뿌요(child)입니다.

현재 관측 가능한 시퀀스는 현재 쌍과 앞으로 두 쌍, 총 3쌍입니다.

**공정한 정보 조건:** 현재 에이전트의 정책이 미래에 생성될 뿌요 전체를 미리 읽지 않도록 주의하세요. 시드는 실험 재현을 위한 값이지, 비공개 미래 시퀀스를 정책에 제공하기 위한 값이 아닙니다.

### 3.4 `turn` — 착수 횟수

```python
turn = obs["turn"]
```

게임 시작 직후는 `0`, 첫 번째 착수 후는 `1`입니다.

현재 기본 환경은 50착수 도달 시 종료됩니다.

---

## 4. 행동 공간 Action Space

행동은 정수 `0~21`로 고정됩니다.

축 뿌요를 어느 열에 두고 자식 뿌요를 어느 방향에 배치할지 나타냅니다.

| 행동 ID | 축 뿌요의 열 (0-based) | 자식 뿌요의 방향 |
|---|---|---|
| 0~5 | 0~5 | 위 (UP) |
| 6~10 | 0~4 | 오른쪽 (RIGHT) |
| 11~16 | 0~5 | 아래 (DOWN) |
| 17~21 | 1~5 | 왼쪽 (LEFT) |

총 22가지입니다.

### 행동의 의미

- UP: 축 뿌요의 위에 자식 뿌요
- RIGHT: 축 뿌요의 오른쪽에 자식 뿌요
- DOWN: 축 뿌요의 아래에 자식 뿌요
- LEFT: 축 뿌요의 왼쪽에 자식 뿌요

판면 높이나 착수 접근 가능성 때문에 일부 행동은 불가능할 수 있습니다.

또한 같은 색 두 뿌요의 경우 중복 배치를 제거하므로, 빈 판에서는 UP 6개와 RIGHT 5개, 총 11개만 유효합니다.

### 행동 마스킹

```python
mask = env.legal_actions()

legal = [
    action
    for action, valid in enumerate(mask)
    if valid
]
```

DQN 등에서 행동별 Q값을 계산할 경우 불가능한 착수를 제외하고 최대값을 선택해야 합니다.

예시:

```python
import numpy as np

q_values = np.asarray(q_values, dtype=float)
masked_q = np.where(mask, q_values, -np.inf)
action = int(np.argmax(masked_q))
```

에피소드가 종료된 상태에서는 다음 행동을 선택하지 않습니다.

---

## 5. 착수 결과 `step()` 전체 반환값

```python
result = env.step(action)
```

| 변수 | 형식 | 의미 |
|---|---|---|
| `observation` | `dict` | 착수 후 다음 관측값 |
| `chains` | `int` | 이번 착수에서 발생한 연쇄 수 |
| `score` | `int` | 이번 착수에서 획득한 점수 |
| `terminated` | `bool` | 게임 진행 불가능으로 종료 |
| `truncated` | `bool` | 임시 착수 제한으로 종료 |

**`score`와 `chains`는 에피소드 전체의 누적값이 아닙니다.** 이번 착수에서 발생한 결과입니다.

누적 점수와 에피소드 최대 연쇄는 직접 집계합니다.

```python
total_score = 0
max_chain = 0

while not env.done():
    mask = env.legal_actions()
    action = next(i for i, valid in enumerate(mask) if valid)
    result = env.step(action)

    total_score += result["score"]
    max_chain = max(max_chain, result["chains"])
```

### 종료 조건의 차이

`terminated`: 게임 엔진의 착수 가능 조건상 더 이상 진행할 수 없음

`truncated`: 현재 실험에서 정한 50착수 제한에 도달

이 둘을 구분해야 합니다. 특히 TD 학습에서 시간제한으로 끝난 상태의 bootstrap 처리 방식은 실제 terminal state와 다를 수 있습니다.

---

## 6. 관측값에서 직접 계산할 수 있는 파생 변수

아래 값은 현재 API가 별도 함수로 제공하는 것은 아니지만, `board`와 `result`에서 계산할 수 있습니다.

| 파생 변수 | 계산 방법 | 연구 활용 예시 |
|---|---|---|
| 열별 높이 | 각 열의 가장 높은 점유 칸 계산 | 높이 페널티 |
| 최대 높이 | 열별 높이의 최댓값 | 게임오버 위험 |
| 평균 높이 | 열별 높이의 평균 | 판면 관리 |
| 높이 편차 | 열별 높이의 분산·표준편차 | 판면 평탄도 |
| 인접 열 높이 차이 | 이웃 열 높이의 절댓값 합 | 표면 불균형 |
| 색상별 뿌요 수 | 보드 문자별 빈도 | 색상 분포 |
| 총 점유 칸 수 | 빈 칸이 아닌 셀 수 | 판면 밀도 |
| 빈 공간 수 | 전체 셀 수 − 점유 칸 수 | 잔여 공간 |
| 연결 요소 수 | 같은 색의 상하좌우 연결 그룹 계산 | 잠재 연쇄 구조 |
| 연결 그룹 크기 | 같은 색으로 연결된 그룹의 크기 | 소멸 후보 탐색 |
| 크기 2·3 연결 그룹 수 | 그룹 크기별 집계 | 연쇄 잠재력 |
| 게임오버 위험 지표 | 중앙 열 높이·합법 행동 수 등 | 생존 중심 학습 |
| 합법적 행동 수 | `sum(env.legal_actions())` | 선택 가능성 |
| 이번 착수의 연쇄 수 | `result["chains"]` | 연쇄 보상 |
| 이번 착수의 점수 | `result["score"]` | 점수 보상 |
| 에피소드 최대 연쇄 | 각 착수 연쇄의 최댓값 | 핵심 평가 지표 |
| 에피소드 누적 점수 | 착수별 점수의 합 | 누적 성능 |

예를 들어 열 높이는 다음처럼 계산할 수 있습니다.

```python
board = obs["board"]

heights = [
    max(
        (y + 1 for y in range(13)
         if board[y][x] != "."),
        default=0
    )
    for x in range(6)
]
```

이 예시는 `board`에 보이는 13개 행만 사용하므로 `row14` 점유 상태는 별도로 고려해야 합니다.

### 상태 인코딩

신경망 입력은 팀원별로 자유롭게 설계할 수 있습니다.

예시:

1. 셀 문자별 정수 인코딩
2. 색상별 One-hot 채널
3. 색상 및 열 높이를 함께 입력
4. 현재/NEXT/NEXT2 뿌요 색상 인코딩 추가
5. 연결 그룹·잠재 연쇄 등 파생 특징 추가
6. CNN 기반 2차원 판면 처리
7. Flatten 후 MLP 입력
8. 순서 정보를 활용하는 시퀀스 모델

다만 **입력 인코딩 방식은 게임 환경의 정보 공개 범위를 넘지 않아야 합니다.**

---

## 7. 보상함수 설계

현재 C++ 게임 환경은 보상함수를 강제하지 않습니다.

팀원은 `result`와 전후 관측값을 사용해 자유롭게 보상을 계산할 수 있습니다.

### 점수 중심

```python
def compute_reward(result):
    return float(result["score"])
```

### 연쇄 수 중심

```python
def compute_reward(result):
    return float(result["chains"])
```

### 연쇄 비선형 보상

```python
def compute_reward(result):
    return float(result["chains"] ** 2)
```

### 연쇄와 점수 혼합

```python
def compute_reward(result):
    return (
        result["chains"] ** 2
        + 0.001 * result["score"]
    )
```

### 판면 높이 페널티 혼합

```python
def compute_reward(result):
    board = result["observation"]["board"]
    height = max(
        (y + 1 for y, row in enumerate(board)
         if any(c != "." for c in row)),
        default=0
    )
    return (
        float(result["chains"] ** 2)
        - 0.1 * height
    )
```

모두 예시이며, 최적의 보상함수로 확정된 것은 아닙니다.

이외에도 생존 보상, 게임오버 페널티, 연쇄 형성 잠재력, 점수 로그 변환, 시간 페널티, potential-based shaping 등을 연구할 수 있습니다.

**주의:** 보상 정의가 달라지면 RL의 최적화 목표 자체가 달라질 수 있습니다. 최종 성능은 공통 평가 지표로 별도 비교해야 합니다.

---

## 8. 현재 Python API에 직접 노출되지 않은 기능

아래 항목은 원본 Ama의 C++ 코드에서 일부 계산할 수 있거나 향후 확장 가능한 기능이지만, **현재 Python 바인딩에서 직접 사용할 수 있는 공개 메서드는 아닙니다.**

| 기능 | 현재 상태 |
|---|---|
| 현재 상태 복제 `clone()` | 미구현 |
| 임의 판면 지정 `set_board()` | 미구현 |
| 임의 뿌요 시퀀스 지정 | 미구현 |
| 다른 행동을 가정한 다음 상태 일괄 계산 | 미구현 |
| 에피소드별 최대 착수 수 설정 | 미구현 |
| 연쇄 단계별 상세 소멸 기록 | 미노출 |
| 원본 C++의 열 높이 조회 | Python에 직접 미노출 |
| 원본 C++의 연결 마스크·그룹 분석 | Python에 직접 미노출 |
| 방해뿌요 생성 및 대전 로직 | 현재 RL 환경에 미연결 |
| 상대방 상태 및 공격 정보 | 현재 RL 환경에 없음 |
| 플레이 시각화 URL 직접 생성 | 현재 RL 환경에 미구현 |
| GPU 기반 게임 시뮬레이션 | 미구현 |
| 학습 보상함수 자동 계산 | 미구현 |

특히 3-ply 탐색, MCTS 또는 모델 기반 강화학습을 실험하려면 상태 복제와 가상 착수 기능이 중요할 수 있습니다.

그런 기능이 필요하다면 프로젝트 관리자에게 공통 환경 확장 사항으로 제안해 주세요. 동일한 기능을 팀원별로 중복 구현하는 것보다 공통 인터페이스로 제공하는 편이 일관성과 검증 측면에서 유리합니다.

---

## 9. 원본 Ama와 현재 RL 환경의 차이

원본 Ama는 단순한 뿌요 시뮬레이터뿐 아니라 Beam Search 등 기존 탐색 AI를 포함합니다.

그러나 현재 `PuyoEnv`는 **1인용 독립 착수 및 연쇄 실험 환경**에 초점을 두고 있습니다.

현재 조건:

- 6열 게임판
- 네 가지 일반 뿌요 색상
- 현재/NEXT/NEXT2 관측
- 최대 22개 고정 착수
- 에피소드 최대 50착수
- 별도 상대방 및 방해뿌요 공격 없음
- 기존 Ama의 탐색 AI는 RL 환경에서 사용하지 않음

따라서 대전 성능, 오쟈마 대응, 상대 AI에 대한 승률을 평가하는 환경과는 구분해야 합니다.

원본 C++ 구현을 직접 살펴보고 싶은 경우:

- `core/cell.h`: 뿌요 색상 및 시퀀스
- `core/field.h`: 게임판
- `core/move.h`, `core/move.cpp`: 착수 생성과 검증
- `core/chain.h`: 연쇄 점수
- `rl_env/puyo_env.h`: 현재 공통 RL 환경
- `rl_env/bindings.cpp`: Python에 노출하는 인터페이스

**주의:** 원본 C++ 클래스의 모든 메서드가 Python으로 공개되는 것은 아닙니다.