"""
stats/bootstrap.py

Computes confidence intervals and statistical significance tests.

Functions:
    bootstrap_ci()        - 95% bootstrap CI for a list of values
    compare_models()      - Wilcoxon signed-rank test for paired model scores
    is_significant()      - Simple CI overlap check for quick screening
    full_comparison()     - Full comparison report between two model result sets
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_CONFIG_DIR = Path(__file__).parent.parent / "config"

def _load_json(path: Path) -> dict:
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)

_EVAL_CFG = _load_json(_CONFIG_DIR / "eval_config.json")

N_RESAMPLES  = _EVAL_CFG["aggregation"]["bootstrap_resamples"]   # 10000
CI_LEVEL     = _EVAL_CFG["aggregation"]["confidence_interval"]   # 0.95
SIG_THRESH   = _EVAL_CFG["statistical_tests"]["significance_threshold"]  # 0.05


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP CONFIDENCE INTERVAL
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    values:      list[float],
    n_resamples: int   = N_RESAMPLES,
    ci_level:    float = CI_LEVEL,
    seed:        int   = 42,
) -> dict:
    """
    Compute a bootstrap confidence interval for the mean of values.

    Args:
        values:      List of per-sample metric scores.
        n_resamples: Number of bootstrap resamples.
        ci_level:    Confidence level (e.g. 0.95 for 95% CI).
        seed:        Random seed for reproducibility.

    Returns:
        Dict with: mean, ci_lower, ci_upper, ci_width, n
    """
    clean = [v for v in values if v is not None]
    n = len(clean)

    if n == 0:
        return {"mean": None, "ci_lower": None, "ci_upper": None,
                "ci_width": None, "n": 0}
    if n == 1:
        return {"mean": clean[0], "ci_lower": clean[0], "ci_upper": clean[0],
                "ci_width": 0.0, "n": 1}

    rng           = random.Random(seed)
    sample_means  = []

    for _ in range(n_resamples):
        resample = [rng.choice(clean) for _ in range(n)]
        sample_means.append(sum(resample) / n)

    sample_means.sort()
    alpha     = 1 - ci_level
    lo_idx    = int(math.floor(alpha / 2 * n_resamples))
    hi_idx    = int(math.ceil((1 - alpha / 2) * n_resamples)) - 1
    hi_idx    = min(hi_idx, n_resamples - 1)

    ci_lower  = sample_means[lo_idx]
    ci_upper  = sample_means[hi_idx]
    true_mean = sum(clean) / n

    return {
        "mean":      round(true_mean, 4),
        "ci_lower":  round(ci_lower,  4),
        "ci_upper":  round(ci_upper,  4),
        "ci_width":  round(ci_upper - ci_lower, 4),
        "n":         n,
        "ci_level":  ci_level,
    }


def ci_overlaps(ci_a: dict, ci_b: dict) -> bool:
    """Return True if two CIs overlap - indicates no clear winner."""
    if None in (ci_a.get("ci_lower"), ci_a.get("ci_upper"),
                ci_b.get("ci_lower"), ci_b.get("ci_upper")):
        return True
    return ci_a["ci_lower"] <= ci_b["ci_upper"] and ci_b["ci_lower"] <= ci_a["ci_upper"]


# ══════════════════════════════════════════════════════════════════════════════
# WILCOXON SIGNED-RANK TEST
# Paired non-parametric test for comparing two models on the same samples
# ══════════════════════════════════════════════════════════════════════════════

def wilcoxon_signed_rank(
    scores_a: list[float],
    scores_b: list[float],
) -> dict:
    """
    Wilcoxon signed-rank test for paired samples.

    Tests H0: the two distributions are identical (no difference between models).
    Uses the normal approximation for p-value (valid for n >= 10).

    Args:
        scores_a: Per-sample metric scores for model A.
        scores_b: Per-sample metric scores for model B.
                  Must be the same length as scores_a (paired).

    Returns:
        Dict with: W_statistic, z_score, p_value, significant, effect_size
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"scores_a and scores_b must have the same length "
            f"(got {len(scores_a)} and {len(scores_b)})"
        )

    # Compute differences, exclude zeros (ties)
    diffs = [a - b for a, b in zip(scores_a, scores_b, strict=True) if a != b]
    n = len(diffs)

    if n == 0:
        return {
            "W_statistic": 0.0, "z_score": 0.0, "p_value": 1.0,
            "significant": False, "effect_size": 0.0, "n_pairs": 0,
            "note": "All differences are zero - models are identical on these samples",
        }

    if n < 10:
        return {
            "W_statistic": None, "z_score": None, "p_value": None,
            "significant": None, "effect_size": None, "n_pairs": n,
            "note": f"n={n} is too small for reliable Wilcoxon approximation (need >= 10)",
        }

    # Rank the absolute differences
    abs_diffs  = [(abs(d), i) for i, d in enumerate(diffs)]
    abs_diffs.sort(key=lambda x: x[0])

    # Assign ranks (average rank for ties)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs_diffs[j][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2  # average of 1-based ranks
        for k in range(i, j):
            ranks[abs_diffs[k][1]] = avg_rank
        i = j

    # Sum of positive and negative ranks
    W_plus  = sum(ranks[i] for i, d in enumerate(diffs) if d > 0)
    W_minus = sum(ranks[i] for i, d in enumerate(diffs) if d < 0)
    W       = min(W_plus, W_minus)

    # Normal approximation
    mean_W = n * (n + 1) / 4
    std_W  = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z      = (W - mean_W) / std_W if std_W > 0 else 0.0

    # Two-tailed p-value using standard normal approximation
    p_value = 2 * _normal_sf(abs(z))

    # Effect size r = z / sqrt(n) (matched pairs)
    effect_size = abs(z) / math.sqrt(n) if n > 0 else 0.0

    return {
        "W_statistic": round(W, 2),
        "z_score":     round(z, 4),
        "p_value":     round(p_value, 6),
        "significant": p_value < SIG_THRESH,
        "effect_size": round(effect_size, 4),
        "n_pairs":     n,
        "W_plus":      round(W_plus, 2),
        "W_minus":     round(W_minus, 2),
    }


def _normal_sf(z: float) -> float:
    """
    Survival function (1 - CDF) of the standard normal distribution.
    Uses math.erfc for accuracy without scipy.
    """
    return 0.5 * math.erfc(z / math.sqrt(2))


# ══════════════════════════════════════════════════════════════════════════════
# FULL MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def full_comparison(
    records_a:   list[dict],
    records_b:   list[dict],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    metrics:     list[str] | None = None,
) -> dict:
    """
    Full statistical comparison between two models on paired samples.

    Matches records by sample ID, then runs:
    - Bootstrap CI for each model on each metric
    - CI overlap check
    - Wilcoxon signed-rank test on paired differences

    Args:
        records_a:    Evaluated prediction records for model A.
        records_b:    Evaluated prediction records for model B.
        model_a_name: Human-readable name for model A.
        model_b_name: Human-readable name for model B.
        metrics:      Which metrics to compare. Defaults to key metrics.

    Returns:
        Nested dict: { metric: { ci_a, ci_b, overlaps, wilcoxon, winner } }
    """
    if metrics is None:
        metrics = ["field_f1", "required_f1", "exact_match_rate",
                   "hallucination_rate", "mean_correction_gain", "schema_valid"]

    # ── Match records by sample ID ─────────────────────────────────────────
    from stats.aggregator import _extract_metrics

    a_by_id = {r["id"]: r for r in records_a}
    b_by_id = {r["id"]: r for r in records_b}
    shared_ids = sorted(set(a_by_id) & set(b_by_id))

    if not shared_ids:
        return {"error": "No shared sample IDs between the two record sets"}

    results = {
        "model_a":      model_a_name,
        "model_b":      model_b_name,
        "n_paired":     len(shared_ids),
        "metrics":      {},
    }

    for metric in metrics:
        scores_a = []
        scores_b = []

        for sid in shared_ids:
            ma = _extract_metrics(a_by_id[sid]).get(metric)
            mb = _extract_metrics(b_by_id[sid]).get(metric)
            if ma is not None and mb is not None:
                scores_a.append(ma)
                scores_b.append(mb)

        if not scores_a:
            results["metrics"][metric] = {"error": "No paired values found"}
            continue

        ci_a      = bootstrap_ci(scores_a)
        ci_b      = bootstrap_ci(scores_b)
        overlaps  = ci_overlaps(ci_a, ci_b)
        wilcoxon  = wilcoxon_signed_rank(scores_a, scores_b)

        # Determine winner
        mean_diff = (ci_a["mean"] or 0) - (ci_b["mean"] or 0)
        if wilcoxon.get("significant") and not overlaps:
            winner = model_a_name if mean_diff > 0 else model_b_name
        elif overlaps:
            winner = "tie (CIs overlap)"
        else:
            winner = "inconclusive (CI gap but not Wilcoxon significant)"

        results["metrics"][metric] = {
            "ci_a":       ci_a,
            "ci_b":       ci_b,
            "mean_diff":  round(mean_diff, 4),
            "ci_overlaps": overlaps,
            "wilcoxon":   wilcoxon,
            "winner":     winner,
        }

    return results


def print_comparison(comparison: dict) -> None:
    """Pretty-print a full_comparison() result."""
    a = comparison["model_a"]
    b = comparison["model_b"]
    n = comparison["n_paired"]

    print(f"\n{'═'*65}")
    print(f"Model comparison: {a} vs {b}  (n={n} paired samples)")
    print(f"{'═'*65}")
    print(f"{'Metric':<28} {a[:12]:>12}  {b[:12]:>12}  {'Δmean':>8}  {'Winner'}")
    print("─" * 65)

    for metric, result in comparison["metrics"].items():
        if "error" in result:
            print(f"  {metric:<26} ERROR: {result['error']}")
            continue
        ca = result["ci_a"]
        cb = result["ci_b"]
        print(
            f"  {metric:<26} "
            f"{str(ca['mean']):>7} ±{str(ca['ci_width']):>5}  "
            f"{str(cb['mean']):>7} ±{str(cb['ci_width']):>5}  "
            f"{result['mean_diff']:>+8.4f}  "
            f"{result['winner']}"
        )
        w = result["wilcoxon"]
        if w.get("p_value") is not None:
            sig = "yes" if w["significant"] else "no"
            print(f"    {'':26} Wilcoxon: p={w['p_value']}  sig={sig}  "
                  f"effect_size={w['effect_size']}")


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random as _rnd

    rng = _rnd.Random(99)
    N   = 60

    # Simulate two models with slightly different performance
    def make_records(n: int, base: float, noise: float, seed: int) -> list[dict]:
        r = _rnd.Random(seed)
        records = []
        domains     = ["receipts", "insurance", "hospital"]
        difficulties = ["easy", "medium", "hard", "extreme"]
        dp = {"easy": 0.0, "medium": 0.07, "hard": 0.16, "extreme": 0.28}

        for i in range(n):
            d   = r.choice(difficulties)
            f1  = max(0, min(1, base - dp[d] + r.gauss(0, noise)))
            hall = max(0, r.uniform(0, 0.1) + dp[d] * 0.2)
            parsed = r.random() > dp[d] * 0.3
            valid  = parsed and r.random() > dp[d] * 0.15

            records.append({
                "id": f"sample_{i:04d}",
                "domain":      r.choice(domains),
                "difficulty":  d,
                "task":        "extraction",
                "parse_status": "success" if parsed else "failure",
                "metrics": {
                    "field_metrics": {
                        "field_f1":         round(f1, 4),
                        "required_f1":      round(min(1, f1 + 0.05), 4),
                        "optional_f1":      round(max(0, f1 - 0.1), 4),
                        "exact_match_rate": round(f1 * 0.9, 4),
                    },
                    "normalization_metrics": {
                        "mean_ned": round(max(0, 0.2 - f1 * 0.15), 4),
                        "numeric_tolerance_rate": round(f1 * 0.95, 4),
                    },
                    "correction_metrics": {"mean_correction_gain": round(f1 - 0.1, 4)},
                    "hallucination":      {"hallucination_rate": round(hall, 4), "n_hallucinated": 0},
                    "schema":             {"schema_valid": valid},
                },
            })
        return records

    records_gpt   = make_records(N, base=0.88, noise=0.07, seed=1)  # strong model
    records_llama = make_records(N, base=0.74, noise=0.09, seed=2)  # weaker model

    print("=" * 65)
    print("PART 1 - Bootstrap CI for GPT-4o field_f1")
    print("=" * 65)
    from stats.aggregator import _extract_metrics
    scores_gpt = [_extract_metrics(r)["field_f1"] for r in records_gpt if _extract_metrics(r)["field_f1"] is not None]
    ci = bootstrap_ci(scores_gpt)
    print(f"  mean  : {ci['mean']}")
    print(f"  95% CI: [{ci['ci_lower']}, {ci['ci_upper']}]")
    print(f"  width : {ci['ci_width']}")
    print(f"  n     : {ci['n']}")

    print("\n" + "=" * 65)
    print("PART 2 - Full model comparison")
    print("=" * 65)
    comparison = full_comparison(records_gpt, records_llama, "GPT-4o", "Llama-8B")
    print_comparison(comparison)
