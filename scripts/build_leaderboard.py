"""
scripts/build_leaderboard.py

Aggregate every results/<model>/summary.json into a single Markdown leaderboard
table sorted by primary metric (exact match rate, descending).

Usage:
    python scripts/build_leaderboard.py

Writes:
    results/leaderboard.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

PRIMARY_METRIC = "exact_match_rate"
COLUMNS: list[tuple[str, str]] = [
    ("model",                "Model"),
    ("backend",              "Backend"),
    ("split",                "Split"),
    ("n_records",            "N"),
    ("exact_match_rate",     "Exact match"),
    ("mean_ned",             "Mean NED"),
    ("required_f1",          "Required F1"),
    ("hallucination_rate",   "Hallucination"),
    ("parse_success",        "Parse"),
]


def load_summaries() -> list[dict]:
    """Find every summary.json under results/ and load it."""
    summaries: list[dict] = []
    if not RESULTS_DIR.exists():
        return summaries
    for path in sorted(RESULTS_DIR.glob("*/summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"  WARN: skipping unreadable {path}: {e}")
            continue
        data["_path"] = str(path.relative_to(ROOT))
        summaries.append(data)
    return summaries


def fmt(value: float | None, places: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{places}f}"


def _row_value(summary: dict, key: str) -> str:
    """Pull a column value out of a summary dict, formatted as markdown text."""
    if key in summary:
        v = summary[key]
        return "-" if v is None else str(v)
    metrics = summary.get("metrics", {})
    if key in metrics:
        return fmt(metrics[key])
    return "-"


def render_markdown(summaries: list[dict]) -> str:
    """Render summaries as a Markdown leaderboard table."""
    # Sort by primary metric, descending. None values go last.
    summaries = sorted(
        summaries,
        key=lambda s: (s.get("metrics", {}).get(PRIMARY_METRIC) or -1),
        reverse=True,
    )

    lines: list[str] = []
    lines.append("# LORE leaderboard")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(
        "Sorted by exact match rate. All entries are reproducible from the "
        "frozen seed in `config/generation_config.json` plus the run config "
        "snapshot in each entry's `results/<model>/run_config.json`."
    )
    lines.append("")

    if not summaries:
        lines.append("_No runs in `results/` yet._")
        lines.append("")
        return "\n".join(lines)

    header = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    sep    = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines.append(header)
    lines.append(sep)

    for s in summaries:
        cells = [_row_value(s, key) for key, _ in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    summaries = load_summaries()
    md = render_markdown(summaries)
    out_path = RESULTS_DIR / "leaderboard.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {len(summaries)} entries to {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
