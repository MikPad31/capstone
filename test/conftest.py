"""
Make the repo root importable so tests can `from src import ...` from anywhere.

`__init__.py` already covers the default import mode, but not `--import-mode=importlib`,
which skips that path insertion; this is what keeps imports working under both. No absolute
paths: the root is derived from this file's own location.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
