"""
Shared test setup.

Adds the repository root to sys.path so tests can import top-level
packages (dataset, evaluator, parsers, runners, stats, report) regardless
of whether the project has been pip-installed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
