#!/usr/bin/env python3
"""Compile and run Ama's exact rule_based fast candidate parity + benchmark."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'teams/minkyu/step_comparison/tests/check_rule_based_fast.cpp'
with tempfile.TemporaryDirectory(prefix='ama-rule-fast-') as tmp:
    exe = Path(tmp) / 'check_rule_fast'
    subprocess.run(['g++', '-std=c++20', '-O2', '-msse4.1', '-I', str(ROOT),
                    str(SRC), '-o', str(exe)], check=True, cwd=ROOT)
    subprocess.run([str(exe)], check=True, cwd=ROOT)
