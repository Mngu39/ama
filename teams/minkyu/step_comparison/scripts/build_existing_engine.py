"""Rebuild ONLY the existing Ama-root rl_env/puyo_cpp extension in Linux/WSL.

Requires the root C++ API patch first. Does not create a duplicate engine.
"""
import importlib.util
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if sys.platform != 'linux':
    raise RuntimeError('Run inside WSL/Linux with the existing Ama .venv')
if importlib.util.find_spec('pybind11') is None:
    raise RuntimeError('Missing pybind11. Install in current .venv: pip install pybind11')
import pybind11

suffix = sysconfig.get_config_var('EXT_SUFFIX')
if not suffix:
    raise RuntimeError('Cannot determine current Python extension suffix')
output = ROOT / 'rl_env' / ('puyo_cpp' + suffix)
command = [
    os.environ.get('CXX', 'g++'), '-std=c++20', '-O2', '-msse4.1',
    '-shared', '-fPIC',
    '-I' + pybind11.get_include(), '-I' + sysconfig.get_path('include'),
    *map(str, sorted((ROOT / 'core').glob('*.cpp'))),
    str(ROOT / 'rl_env' / 'bindings.cpp'), '-o', str(output),
]
print('Rebuilding Ama-root engine:', output, flush=True)
subprocess.run(command, cwd=ROOT, check=True)
subprocess.run([
    sys.executable, '-c',
    ('from rl_env import puyo_cpp; e=puyo_cpp.PuyoEnv(); e.reset(12345); '
     'assert e.clone().preview_all(); '
     'assert e.fork_with_hidden_pairs(["RR"]*128).preview_all(); '
     'assert e.fork_with_hidden_rule_seed(54321).preview_all(); '
     'print("Ama hidden-queue fork API OK")')
], cwd=ROOT, check=True)
