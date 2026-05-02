"""
scripts/snapshot_run.py

Copy a run directory's verifiable artifacts into results/<model_name>/ so they
can be committed to the repo. Reproducible re-runs with the same seed produce
identical outputs, so the snapshot is the public leaderboard receipt.

Files copied:
    report.html        Inline self-contained HTML report.
    summary.json       Machine-readable metric summary.
    run_config.json    Frozen config snapshot (model, backend, filters, seed).

Usage:
    python scripts/snapshot_run.py runs/always_null_20260502_134730
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

ARTIFACTS = ("report.html", "summary.json", "run_config.json")


def snapshot(run_dir: Path) -> Path:
    """Copy artifacts from run_dir to results/<model>/. Returns the destination."""
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"run_config.json missing in {run_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_name = config.get("model")
    if not model_name:
        raise ValueError(f"run_config.json in {run_dir} has no 'model' field")

    safe_name = model_name.replace("/", "-").replace(":", "-")
    dest = RESULTS_DIR / safe_name
    dest.mkdir(parents=True, exist_ok=True)

    for artifact in ARTIFACTS:
        src = run_dir / artifact
        if not src.exists():
            print(f"  WARN: {artifact} not found in {run_dir} - skipping")
            continue
        shutil.copy2(src, dest / artifact)
        print(f"  copied {artifact} -> results/{safe_name}/{artifact}")

    return dest


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    run_dir = Path(sys.argv[1]).resolve()
    snapshot(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
