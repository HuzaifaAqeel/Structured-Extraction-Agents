"""Makes the repo root importable as ``src.*`` when running pytest from anywhere."""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
