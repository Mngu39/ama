
#!/usr/bin/env bash
set -euo pipefail

# Move to the repository root.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Check required commands.
for cmd in python3 g++; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Missing dependency: $cmd"
        exit 1
    fi
done

echo "[1/4] Creating Python virtual environment"
python3 -m venv .venv
source .venv/bin/activate

echo "[2/4] Installing Python dependencies"
python -m pip install "pybind11==3.1.0"

echo "[3/4] Building Ama Python extension"
EXT="$(python -m pybind11 --extension-suffix)"

g++ -std=c++20 -O2 -march=native -DPEXT \
    -shared -fPIC \
    $(python -m pybind11 --includes) \
    core/*.cpp rl_env/bindings.cpp \
    -o "rl_env/puyo_cpp${EXT}"

echo "[4/4] Testing Python environment"
python - <<'PY'
from rl_env import puyo_cpp

env = puyo_cpp.PuyoEnv()
obs = env.reset(12345)

assert puyo_cpp.ACTION_COUNT == 22
assert obs["visible"][0] == ("R", "Y")
assert len(env.legal_actions()) == 22

result = env.step(0)
assert result["observation"]["turn"] == 1

print("PYTHON ENVIRONMENT PASSED")
PY

echo "SETUP COMPLETED"
