"""Preserve the original standalone test imports when invoked from the Ama repo root."""
from pathlib import Path
import sys

_project_dir = str(Path(__file__).resolve().parents[1])
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)
