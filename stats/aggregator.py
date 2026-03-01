"""
stats/aggregator.py

Takes a list of evaluated prediction records (after all 5 evaluator
modules have run) and computes dataset-level statistics.

Grouping levels (from eval_config.json):
    1. overall              — all domains, all difficulties
    2. domain               — per domain
    3. difficulty           — per difficulty level
    4. domain_x_difficulty  — 12 cells (3 domains × 4 difficulties)

Statistics computed per metric per group:
    - mean
    - median
    - 10th percentile
    - 90th percentile
    - std deviation
    - count (n)

Also computes:
    - failure mode breakdown (counts per failure mode)
    - parse failure rate
    - schema validity rate
"""

from __future__ import annotations
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

_CONFIG_DIR = Path(__file__).parent.parent / "config"

def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

_EVAL_CFG = _load_json(_CONFIG_DIR / "eval_config.json")


# ══════════════════════════════════════════════════════════════════════════════
# METRICS KEYS — what we extract from each evaluated prediction record
# These are the scalar metrics written by the full evaluator pipeline
# ══════════════════════════════════════════════════════════════════════════════

SCALAR_METRICS = [
    "field_f1",
    "required_f1",
    "optional_f1",
    "exact_match_rate",
    "mean_ned",
    "numeric_tolerance_rate",
    "mean_correction_gain",
    "hallucination_rate",
    "schema_valid",          # bool → treated as 0/1
    "parse_success",         # derived: parse_status == "success"
]


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL HELPERS — pure Python, no numpy
# ══════════════════════════════════════════════════════════════════════════════

def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _percentile(values: list[float], p: float) -> float | None:
    """Linear interpolation percentile."""
    if not values:
        return None
    s     = sorted(values)
    n     = len(s)
    idx   = (p / 100) * (n - 1)
    lo    = int(idx)
    hi    = lo + 1
    frac  = idx - lo
    if hi >= n:
        return s[-1]
    return s[lo] + frac * (s[hi] - s[lo])


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _summarize(values: list[float]) -> dict:
    """Compute all summary stats for a list of values."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0, "mean": None, "median": None,
                "p10": None, "p90": None, "std": None}
    return {
        "n":      len(clean),
        "mean":   round(_mean(clean),          4),
        "median": round(_median(clean),        4),
        "p10":    round(_percentile(clean, 10), 4),
        "p90":    round(_percentile(clean, 90), 4),
        "std":    round(_std(clean),           4) if _std(clean) else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# METRIC EXTRACTION — pull scalar metrics from one evaluated record
# ══════════════════════════════════════════════════════════════════════════════

def _extract_metrics(record: dict) -> dict[str, float | None]:
    """
    Extract scalar metric values from one fully-evaluated prediction record.

    Expects the record to have been enriched by the evaluator pipeline:
        record["metrics"]["field_metrics"]
        record["metrics"]["normalization_metrics"]
        record["metrics"]["correction_metrics"]
        record["metrics"]["hallucination"]
        record["metrics"]["schema"]
    """
    m = record.get("metrics", {})

    fm   = m.get("field_metrics", {})
    nm   = m.get("normalization_metrics", {})
    cm   = m.get("correction_metrics", {})
    hall = m.get("hallucination", {})
    sch  = m.get("schema", {})

    return {
        "field_f1":               fm.get("field_f1"),
        "required_f1":            fm.get("required_f1"),
        "optional_f1":            fm.get("optional_f1"),
        "exact_match_rate":       fm.get("exact_match_rate"),
        "mean_ned":               nm.get("mean_ned"),
        "numeric_tolerance_rate": nm.get("numeric_tolerance_rate"),
        "mean_correction_gain":   cm.get("mean_correction_gain"),
        "hallucination_rate":     hall.get("hallucination_rate"),
        "schema_valid":           float(sch.get("schema_valid", False)),
        "parse_success":          float(record.get("parse_status") == "success"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FAILURE MODE COUNTER
# ══════════════════════════════════════════════════════════════════════════════

def _count_failure_modes(records: list[dict]) -> dict[str, int]:
    """
    Count how many records exhibit each failure mode.
    A record can have multiple failure modes.
    """
    counts: dict[str, int] = defaultdict(int)

    for rec in records:
        m    = rec.get("metrics", {})
        fm   = m.get("field_metrics", {})
        hall = m.get("hallucination", {})
        sch  = m.get("schema", {})
        cm   = m.get("correction_metrics", {})
        parse = rec.get("parse_status", "failure")

        if parse == "failure":
            counts["parse_failure"] += 1
        elif parse == "partial":
            counts["partial_parse"] += 1

        if not sch.get("schema_valid", True):
            counts["schema_mismatch"] += 1

        if fm.get("required_f1", 1.0) < 1.0:
            counts["missing_required"] += 1

        if fm.get("exact_match_rate", 1.0) < 1.0 and fm.get("field_f1", 0) > 0:
            counts["wrong_value"] += 1

        if hall.get("n_hallucinated", 0) > 0:
            counts["hallucination"] += 1

        gain = cm.get("mean_correction_gain")
        if gain is not None and gain < 0:
            counts["correction_regression"] += 1

    return dict(counts)


# ══════════════════════════════════════════════════════════════════════════════
# GROUPING
# ══════════════════════════════════════════════════════════════════════════════

def _group_records(
    records: list[dict],
    levels:  list[str],
) -> dict[str, list[dict]]:
    """
    Group records by the specified grouping levels.
    Returns dict mapping group_key → list of records in that group.

    Supported levels:
        "overall", "domain", "difficulty", "domain_x_difficulty", "task"
    """
    groups: dict[str, list[dict]] = defaultdict(list)

    for rec in records:
        keys = []
        for level in levels:
            if level == "overall":
                keys.append("overall")
            elif level == "domain":
                keys.append(rec.get("domain", "unknown"))
            elif level == "difficulty":
                keys.append(rec.get("difficulty", "unknown"))
            elif level == "domain_x_difficulty":
                keys.append(
                    f"{rec.get('domain','unknown')}__{rec.get('difficulty','unknown')}"
                )
            elif level == "task":
                keys.append(rec.get("task", "unknown"))

        group_key = " | ".join(keys)
        groups[group_key].append(rec)

    return dict(groups)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AGGREGATOR
# ══════════════════════════════════════════════════════════════════════════════

def aggregate(
    records: list[dict],
    grouping_levels: list[str] | None = None,
) -> dict:
    """
    Compute aggregated statistics from evaluated prediction records.

    Args:
        records:         List of fully-evaluated prediction records.
                         Each record must have a "metrics" key populated
                         by the evaluator pipeline.
        grouping_levels: Which grouping levels to compute.
                         Defaults to eval_config.json settings.

    Returns:
        Nested dict:
            {
              "overall": { metric: {n, mean, median, p10, p90, std}, ... },
              "by_domain": { domain: { metric: {...} } },
              "by_difficulty": { difficulty: { metric: {...} } },
              "by_domain_x_difficulty": { "domain__diff": { metric: {...} } },
              "failure_modes": { mode: count },
              "n_records": int,
            }
    """
    if grouping_levels is None:
        grouping_levels = _EVAL_CFG["aggregation"]["grouping_levels"]

    if not records:
        return {"n_records": 0, "error": "No records to aggregate"}

    # ── Overall ───────────────────────────────────────────────────────────
    all_metrics: dict[str, list] = defaultdict(list)
    for rec in records:
        extracted = _extract_metrics(rec)
        for metric, value in extracted.items():
            if value is not None:
                all_metrics[metric].append(value)

    overall = {metric: _summarize(vals) for metric, vals in all_metrics.items()}

    # ── By domain ─────────────────────────────────────────────────────────
    domain_groups = _group_records(records, ["domain"])
    by_domain = {}
    for group_key, group_recs in domain_groups.items():
        group_metrics: dict[str, list] = defaultdict(list)
        for rec in group_recs:
            for metric, value in _extract_metrics(rec).items():
                if value is not None:
                    group_metrics[metric].append(value)
        by_domain[group_key] = {m: _summarize(v) for m, v in group_metrics.items()}

    # ── By difficulty ─────────────────────────────────────────────────────
    diff_groups = _group_records(records, ["difficulty"])
    by_difficulty = {}
    for group_key, group_recs in diff_groups.items():
        group_metrics = defaultdict(list)
        for rec in group_recs:
            for metric, value in _extract_metrics(rec).items():
                if value is not None:
                    group_metrics[metric].append(value)
        by_difficulty[group_key] = {m: _summarize(v) for m, v in group_metrics.items()}

    # ── By domain × difficulty ─────────────────────────────────────────────
    cross_groups = _group_records(records, ["domain_x_difficulty"])
    by_cross = {}
    for group_key, group_recs in cross_groups.items():
        group_metrics = defaultdict(list)
        for rec in group_recs:
            for metric, value in _extract_metrics(rec).items():
                if value is not None:
                    group_metrics[metric].append(value)
        by_cross[group_key] = {m: _summarize(v) for m, v in group_metrics.items()}

    # ── Failure modes ──────────────────────────────────────────────────────
    failure_modes = _count_failure_modes(records)

    return {
        "n_records":               len(records),
        "overall":                 overall,
        "by_domain":               by_domain,
        "by_difficulty":           by_difficulty,
        "by_domain_x_difficulty":  by_cross,
        "failure_modes":           failure_modes,
    }


def print_summary(agg: dict, metric: str = "field_f1") -> None:
    """
    Print a human-readable summary table for one metric across all groupings.
    """
    print(f"\nMetric: {metric}")
    print(f"{'Group':<35} {'n':>5}  {'mean':>7}  {'median':>7}  {'p10':>7}  {'p90':>7}")
    print("─" * 75)

    def _row(label: str, stats: dict) -> None:
        s = stats.get(metric, {})
        if not s or s.get("n", 0) == 0:
            return
        print(
            f"  {label:<33} {s['n']:>5}  "
            f"{str(s['mean']):>7}  {str(s['median']):>7}  "
            f"{str(s['p10']):>7}  {str(s['p90']):>7}"
        )

    _row("OVERALL", agg["overall"])
    print()

    for domain, stats in sorted(agg["by_domain"].items()):
        _row(domain, stats)
    print()

    for diff in ["easy", "medium", "hard", "extreme"]:
        if diff in agg["by_difficulty"]:
            _row(diff, agg["by_difficulty"][diff])
    print()

    for cross_key in sorted(agg["by_domain_x_difficulty"]):
        _row(cross_key.replace("__", " / "), agg["by_domain_x_difficulty"][cross_key])


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random

    # ── Generate synthetic evaluated records for testing ──────────────────
    rng = random.Random(42)
    domains     = ["receipts", "insurance", "hospital"]
    difficulties = ["easy", "medium", "hard", "extreme"]
    tasks       = ["extraction", "normalization", "correction", "hallucination", "schema"]

    # Simulate degrading performance as difficulty increases
    difficulty_penalty = {"easy": 0.0, "medium": 0.08, "hard": 0.18, "extreme": 0.32}

    records = []
    for i in range(120):
        domain     = rng.choice(domains)
        difficulty = rng.choice(difficulties)
        task       = rng.choice(tasks)
        penalty    = difficulty_penalty[difficulty]

        base_f1   = rng.uniform(0.75, 0.98)
        f1        = max(0.0, base_f1 - penalty + rng.gauss(0, 0.05))
        hall_rate = max(0.0, rng.uniform(0, 0.1) + penalty * 0.3)
        parsed    = rng.random() > (penalty * 0.4)
        valid     = parsed and rng.random() > (penalty * 0.2)

        records.append({
            "id":          f"sample_{i:04d}",
            "domain":      domain,
            "difficulty":  difficulty,
            "task":        task,
            "parse_status": "success" if parsed else "failure",
            "metrics": {
                "field_metrics": {
                    "field_f1":         round(f1, 4),
                    "required_f1":      round(min(1.0, f1 + 0.05), 4),
                    "optional_f1":      round(max(0.0, f1 - 0.1), 4),
                    "exact_match_rate": round(f1 * 0.9, 4),
                },
                "normalization_metrics": {
                    "mean_ned":               round(max(0, 0.15 - f1 * 0.1), 4),
                    "numeric_tolerance_rate": round(f1 * 0.95, 4),
                },
                "correction_metrics": {
                    "mean_correction_gain": round(f1 - 0.1 + rng.gauss(0, 0.05), 4),
                },
                "hallucination": {
                    "hallucination_rate": round(hall_rate, 4),
                    "n_hallucinated":     int(hall_rate * 5),
                },
                "schema": {
                    "schema_valid": valid,
                },
            },
        })

    print("=" * 60)
    print("PART 1 — Aggregate stats")
    print("=" * 60)
    agg = aggregate(records)
    print(f"Total records: {agg['n_records']}")

    print_summary(agg, "field_f1")

    print("\n" + "=" * 60)
    print("PART 2 — Failure mode breakdown")
    print("=" * 60)
    for mode, count in sorted(agg["failure_modes"].items(), key=lambda x: -x[1]):
        print(f"  {mode:<30} {count:>4}")

    print("\n" + "=" * 60)
    print("PART 3 — Hallucination rate by difficulty")
    print("=" * 60)
    for diff in ["easy", "medium", "hard", "extreme"]:
        stats = agg["by_difficulty"].get(diff, {}).get("hallucination_rate", {})
        if stats.get("n", 0) > 0:
            print(f"  {diff:<10}  mean={stats['mean']}  p90={stats['p90']}")