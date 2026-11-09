from pathlib import Path
import importlib.util
import os
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parent
if sys.platform != 'linux':
    raise RuntimeError('Linux 또는 Windows WSL Ubuntu에서 실행해 주십시오.')
if importlib.util.find_spec('pybind11') is None:
    raise RuntimeError('pybind11이 없습니다. python -m pip install pybind11을 실행해 주십시오.')
import pybind11
suffix = sysconfig.get_config_var('EXT_SUFFIX')
if not suffix:
    raise RuntimeError('현재 Python의 EXT_SUFFIX를 확인할 수 없습니다.')
result = ROOT / 'rl_env' / ('puyo_cpp' + suffix)
command = [
    os.environ.get('CXX', 'g++'), '-std=c++20', '-O2', '-shared', '-fPIC',
    '-I' + pybind11.get_include(), '-I' + sysconfig.get_path('include'),
    *[str(x) for x in sorted((ROOT / 'core').glob('*.cpp'))],
    str(ROOT / 'rl_env' / 'bindings.cpp'), '-o', str(result),
]
print('Building:', result, flush=True)
subprocess.run(command, cwd=ROOT, check=True)
subprocess.run([sys.executable, '-c',
                'from rl_env import puyo_cpp; e=puyo_cpp.PuyoEnv(); e.reset(12345); '
                'assert e.clone().preview_all(); print("Puyo C++ engine OK")'],
               cwd=ROOT, check=True)
