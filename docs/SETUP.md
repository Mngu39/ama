# 설치, 실행 및 GitHub 협업 안내

## 1. GitHub 저장소와 실행 환경의 관계

프로젝트 저장소:

https://github.com/Mngu39/ama

GitHub에는 소스코드가 저장됩니다. 팀원은 다음 중 하나를 선택해 실행할 수 있습니다.

| 방식 | 코드 저장 위치 | 실제 실행 위치 |
|---|---|---|
| 로컬 Clone | GitHub + 개인 PC | 개인 PC |
| GitHub Actions | GitHub | GitHub 제공 Runner |
| Google Colab | GitHub + Colab VM | Google Colab VM |

GitHub에서 소스를 수정했다고 개인 PC 파일이 자동으로 바뀌지는 않습니다. 반대로 개인 PC에서 파일을 수정하더라도 Commit 및 Push를 해야 GitHub에 반영됩니다.

---

## 2. 방법 A — 자신의 PC에서 실행

### 2.1 Windows 사용자: WSL2 권장

Windows에서는 Ubuntu WSL2 사용을 권장합니다.

1. Windows에 WSL2 및 Ubuntu 24.04 설치
2. VS Code, Git 또는 GitHub Desktop 설치
3. GitHub 저장소 Clone
4. Ubuntu 터미널에서 프로젝트를 열고 자동 설치 스크립트 실행

WSL Ubuntu에서 저장소를 처음 Clone하는 방법:

```bash
sudo apt update
sudo apt install -y git build-essential python3-venv python3-dev

git clone https://github.com/Mngu39/ama.git
cd ama

bash scripts/setup_wsl.sh
source .venv/bin/activate
```

GitHub Desktop을 사용하는 경우에는 다음 방법도 가능합니다.

1. GitHub Desktop 실행
2. `File > Clone repository > URL`
3. `https://github.com/Mngu39/ama.git` 입력
4. 로컬 폴더 선택 후 Clone
5. 해당 폴더를 VS Code에서 열고 WSL 터미널로 접속
6. 프로젝트 최상위 폴더에서 `bash scripts/setup_wsl.sh` 실행

Windows에 Clone한 저장소를 WSL에서 열 경우 드라이브는 `/mnt/c/` 등으로 표시됩니다. 예를 들어 Windows의 `C:\Users\...\Documents\GitHub\ama`는 WSL에서 `/mnt/c/Users/.../Documents/GitHub/ama`입니다.

**GitHub Desktop과 WSL이 동일한 저장소를 사용한다면 두 번 Clone할 필요는 없습니다.**

Linux 사용자도 같은 Ubuntu 설치 절차를 사용할 수 있습니다.

현재 빌드는 x86-64와 SSE4.1, BMI2를 사용하는 구성에서 검증되었습니다. ARM 기반 Mac 및 기타 아키텍처는 직접 지원이 확인되지 않았습니다.

### 2.2 이후 저장소 업데이트

다른 팀원이 GitHub에 변경사항을 올렸다면 해당 브랜치에서:

```bash
git pull origin main
```

GitHub Desktop에서는 `Fetch origin` 후 필요한 경우 `Pull origin`을 선택합니다.

로컬에서 수정 중인 파일이 있다면 먼저 Commit하거나 별도 브랜치에 저장해야 충돌 위험을 줄일 수 있습니다.

### 2.3 설치 및 실행 확인

```bash
source .venv/bin/activate

python -m teams.template.random_baseline.train \
  --episodes 10 --seed 12345

python -m evaluation.evaluate \
  --policy teams.template.random_baseline.eval_policy \
  --episodes 100 --seed 100000
```

각 실행이 정상 종료되면 프로젝트 기본 환경을 사용할 수 있습니다.

---

## 3. 방법 B — Google Colab에서 실행

Google 계정으로 https://colab.research.google.com/ 에 접속합니다.

`새 노트북`을 만들고 다음을 순서대로 실행합니다.

### 3.1 GitHub 코드 가져오기

첫 번째 코드 셀:

```python
!git clone https://github.com/Mngu39/ama.git
%cd /content/ama
```

이미 Clone했으며 새로운 GitHub 변경사항을 가져오려면:

```python
!git pull origin main
```

단, Colab의 기본 런타임은 일시적인 가상 머신이므로 런타임이 초기화되면 다시 Clone해야 할 수 있습니다.

### 3.2 pybind11 설치

```python
!python -m pip install pybind11==3.1.0
```

### 3.3 C++ 게임 엔진 컴파일

먼저 CPU 명령어 지원 확인:

```python
!uname -m
!grep -m1 -o 'sse4_1' /proc/cpuinfo
```

x86-64 및 SSE4.1 환경이라면 다음과 같은 **BMI2 비활성화 빌드**를 시도할 수 있습니다.

```python
!g++ -std=c++20 -O2 -msse4.1 -shared -fPIC \
  $(python -m pybind11 --includes) \
  core/*.cpp rl_env/bindings.cpp \
  -o rl_env/puyo_cpp$(python -m pybind11 --extension-suffix)
```

이는 현재 WSL에서 검증된 `-march=native -DPEXT` 빌드와 다른 CPU 호환성용 빌드입니다. Colab에서 별도로 검증해야 하며, 런타임 CPU가 SSE4.1을 제공하지 않으면 사용할 수 없습니다.

### 3.4 실험 실행

```python
!python -m teams.template.random_baseline.train --episodes 10
```

평가:

```python
!python -m evaluation.evaluate \
  --policy teams.template.random_baseline.eval_policy \
  --episodes 100 --seed 100000
```

### 3.5 GPU 사용

Colab 상단 `런타임 > 런타임 유형 변경`에서 GPU를 선택할 수 있습니다.

주의할 점:

- Ama의 C++ 시뮬레이터가 GPU에서 실행되는 것은 아닙니다.
- PyTorch 등으로 구현한 신경망 연산에 GPU를 활용할 수 있습니다.
- CUDA를 사용하려면 해당 팀원의 학습 코드가 GPU 연산을 지원해야 합니다.
- GPU 사용 여부와 별개로 C++ 엔진은 해당 런타임의 CPU에서 실행됩니다.

### 3.6 학습 결과 저장

Colab의 VM 파일은 영구 저장소가 아닙니다. 필요한 체크포인트나 결과를 Google Drive에 저장할 수 있습니다.

```python
from google.colab import drive
drive.mount("/content/drive")
```

이후 `/content/drive/MyDrive/` 아래의 팀원별 폴더에 모델 가중치와 실험 결과를 저장합니다.

학습된 모델이 큰 경우 GitHub에 직접 Commit하지 말고 Google Drive 등 별도의 저장소를 사용합니다.

---

## 4. 방법 C — GitHub Actions에서 실행

GitHub Actions를 사용하면 개인 PC 없이 GitHub의 Ubuntu Runner에서 빌드·테스트·평가를 실행할 수 있습니다.

단, 실행하려면 먼저 `.github/workflows/` 아래에 프로젝트용 워크플로 파일을 추가해야 합니다. 기존 저장소에 있는 워크플로와는 별개입니다.

워크플로를 추가한 후:

1. GitHub 저장소 접속
2. 상단 `Actions` 선택
3. 왼쪽에서 `RL Evaluation` 워크플로 선택
4. `Run workflow` 클릭
5. 브랜치, 평가 정책, 에피소드 수, 시드 입력
6. 실행 후 로그에서 결과 확인

`Run workflow`는 워크플로가 `workflow_dispatch`를 지원하고 기본 브랜치에 등록된 경우 사용할 수 있습니다. 실행하려면 저장소의 쓰기 권한도 필요합니다.

**GitHub Actions 권장 용도**

- 새 코드가 정상 컴파일되는지 확인
- 공통 환경 회귀 테스트
- 무작위 정책 또는 학습된 모델의 짧은 평가
- 반복 가능한 소규모 실험

장시간 GPU 학습이나 대규모 하이퍼파라미터 탐색은 로컬 머신 또는 Colab 등 별도 학습 환경 사용을 권장합니다.

GitHub Runner는 실행할 때마다 새 환경을 만들 수 있으므로, 학습된 가중치와 실험 결과가 자동으로 영구 보존되는 것은 아닙니다.

---

## 5. GitHub에서 팀원별 실험 파일 만들기

### 방법 A — GitHub 웹사이트에서 직접 작성

1. https://github.com/Mngu39/ama 접속
2. `Code` 탭에서 `teams/` 폴더 열기
3. `Add file > Create new file` 클릭
4. 파일 이름에 경로까지 입력

예:

```text
teams/member_a/td_3step/train.py
```

이렇게 입력하면 하위 폴더를 포함해 파일을 만들 수 있습니다.

5. 코드 작성
6. `Commit changes` 클릭
7. 본인 브랜치에 저장하거나 새 브랜치를 만들고 Pull Request 제출

여러 파일을 작성하려면 동일한 방식으로 `agent.py`, `reward.py`, `eval_policy.py` 등을 생성합니다.

### 방법 B — 파일을 로컬에서 만들어 GitHub에 업로드

1. 자신의 PC에서 실험 폴더와 파일 작성
2. GitHub 저장소의 해당 폴더로 이동
3. `Add file > Upload files` 선택
4. 파일 업로드 후 Commit

또는 GitHub Desktop에서 변경사항을 Commit하고 Push합니다.

### 방법 C — 카카오톡으로 전달

GitHub 직접 수정이 어렵다면 실험 폴더를 `.zip`으로 압축하거나 개별 `.py` 파일을 카카오톡으로 프로젝트 관리자에게 전달해도 됩니다.

전달 시 함께 알려주세요.

- 팀원명
- 실험명
- 파일을 넣을 상대 경로
- 필요한 라이브러리
- 실행 명령
- 모델 가중치가 있다면 그 파일 또는 별도 공유 링크

관리자가 해당 경로에 파일을 추가하고 Commit/Push할 수 있습니다.

### 권장 실험 구성

```text
teams/member_a/td_3step/
├── __init__.py
├── agent.py
├── reward.py
├── train.py
├── eval_policy.py
├── network.py
└── config.py
```

필요 없는 파일은 생략해도 됩니다.

실험별 실행 예시:

```bash
python -m teams.member_a.td_3step.train
```

`-m`으로 실행하려면 Python 패키지 경로와 모듈 구성이 맞아야 하며, 관련 폴더에 `__init__.py`를 두는 것을 권장합니다.

---

## 6. GitHub 브랜치 협업

공통 코드가 변경되는 것을 줄이기 위해 팀원별 브랜치 사용을 권장합니다.

예:

```bash
git switch -c member-a-td3
git add teams/member_a/td_3step/
git commit -m "Add member A 3-step TD experiment"
git push -u origin member-a-td3
```

GitHub에서 Pull Request를 작성하면 관리자가 변경사항을 확인하고 `main`에 병합할 수 있습니다.

저장소에 직접 Push하려면 Collaborator 권한이 필요합니다. 권한이 없는 공개 저장소 사용자라면 자신의 Fork를 만든 뒤 Pull Request를 보내는 방식도 가능합니다.

**주의:** 팀원별 브랜치는 코드 충돌을 줄이기 위한 작업 방식이지, 다른 팀원의 파일을 수정하지 못하게 하는 접근 권한 기능은 아닙니다.

---

## 7. 외부 라이브러리와 실험 파일

현재 공통 Python 환경은 `pybind11`을 사용합니다.

PyTorch, NumPy 등의 패키지는 모든 실험에 필수인 것은 아니므로 각 팀원이 자신에게 필요한 의존성을 관리할 수 있습니다.

예:

```text
teams/member_a/td_3step/requirements.txt
```

사용 시:

```bash
python -m pip install -r teams/member_a/td_3step/requirements.txt
```

패키지 버전은 가능하면 고정하고, 같은 실험을 재현할 수 있도록 설정 파일을 함께 공유합니다.

GitHub에 API 키, 비밀번호, 개인 인증 파일을 Commit하지 마세요.