"""
evaluator/normalization_metrics.py

Measures how close predicted values are to ground truth beyond binary exact match.

Metrics produced per field:
    - exact_match      : bool — normalized pred == normalized gt
    - edit_distance    : int  — raw Levenshtein distance (strings only)
    - ned              : float 0–1 — normalized edit distance (strings only)
    - relative_error   : float — |pred - gt| / |gt| (numbers only)
    - within_tolerance : bool — relative_error <= threshold (numbers only)

Aggregate metrics:
    - mean_ned              : mean NED across all string fields
    - exact_match_rate      : fraction of fields with exact match
    - numeric_tolerance_rate: fraction of numeric fields within tolerance
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any


# ── Config ─────────────────────────────────────────────────────────────────────

_CONFIG_DIR = Path(__file__).parent.parent / "config"

def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

_EVAL_CFG  = _load_json(_CONFIG_DIR / "eval_config.json")
_DOMAINS   = _load_json(_CONFIG_DIR / "domains.json")

NUMERIC_TOLERANCE = _EVAL_CFG["thresholds"]["numeric_relative_error_tolerance"]


# ══════════════════════════════════════════════════════════════════════════════
# LEVENSHTEIN DISTANCE (no external dependency)
# ══════════════════════════════════════════════════════════════════════════════

def levenshtein(a: str, b: str) -> int:
    """
    Classic dynamic-programming Levenshtein edit distance.
    O(len(a) * len(b)) time, O(len(b)) space.
    """
    a, b = str(a), str(b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,            # deletion
                curr[j-1] + 1,          # insertion
                prev[j-1] + (ca != cb)  # substitution
            )
        prev = curr
    return prev[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    """
    Levenshtein distance normalized by the length of the longer string.
    Range: 0.0 (identical) – 1.0 (completely different).
    """
    if not a and not b:
        return 0.0
    dist = levenshtein(str(a), str(b))
    return dist / max(len(str(a)), len(str(b)))


# ══════════════════════════════════════════════════════════════════════════════
# FIELD TYPE LOOKUP
# ══════════════════════════════════════════════════════════════════════════════

def _get_field_type(field_path: str, domain: str) -> str:
    """Get the schema type for a dotted field path."""
    schema = _DOMAINS[domain]["schema"]
    parts  = field_path.split(".")
    node   = schema
    for part in parts:
        node = node.get(part, {})
    fmt = node.get("format", "")
    return fmt if fmt else node.get("type", "string")


def _is_numeric(field_type: str) -> bool:
    return field_type.lower() == "number"


def _is_string(field_type: str) -> bool:
    return field_type.lower() not in ("number",)


# ══════════════════════════════════════════════════════════════════════════════
# PER-FIELD METRIC
# ══════════════════════════════════════════════════════════════════════════════

def compute_field_normalization(
    pred_value: Any,
    gt_value:   Any,
    field_type: str,
) -> dict:
    """
    Compute normalization metrics for a single field.

    Returns a dict with relevant metrics for the field type.
    Always returns exact_match regardless of type.
    """
    result: dict[str, Any] = {"field_type": field_type}

    # Handle None predictions
    if pred_value is None and gt_value is None:
        return {**result, "exact_match": True,  "ned": 0.0}
    if pred_value is None or gt_value is None:
        return {**result, "exact_match": False, "ned": 1.0,
                "edit_distance": -1, "within_tolerance": False}

    pred_str = str(pred_value).strip().lower()
    gt_str   = str(gt_value).strip().lower()
    exact    = pred_str == gt_str

    if _is_numeric(field_type):
        try:
            pred_f = float(pred_value)
            gt_f   = float(gt_value)
            if gt_f == 0:
                rel_err   = 0.0 if pred_f == 0 else float("inf")
            else:
                rel_err   = abs(pred_f - gt_f) / abs(gt_f)
            within = rel_err <= NUMERIC_TOLERANCE
            return {
                **result,
                "exact_match":      exact,
                "relative_error":   round(rel_err, 6),
                "within_tolerance": within,
            }
        except (TypeError, ValueError):
            return {**result, "exact_match": False,
                    "relative_error": None, "within_tolerance": False}
    else:
        dist = levenshtein(pred_str, gt_str)
        ned  = normalized_edit_distance(pred_str, gt_str)
        return {
            **result,
            "exact_match":   exact,
            "edit_distance": dist,
            "ned":           round(ned, 4),
        }


# ══════════════════════════════════════════════════════════════════════════════
# STRUCT-LEVEL METRIC
# ══════════════════════════════════════════════════════════════════════════════

def _flatten(struct: dict, prefix: str = "") -> dict[str, Any]:
    flat = {}
    for key, value in struct.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=path))
        else:
            flat[path] = value
    return flat


def compute_normalization_metrics(
    pred_struct: dict,
    gt_struct:   dict,
    domain:      str,
) -> dict:
    """
    Compute normalization metrics across all fields in a sample.

    Only evaluates fields present in GT (with non-None values).
    Fields in pred but not GT are ignored here (handled by hallucination_detector).

    Returns:
        Dict with aggregate metrics and per_field breakdown.
    """
    pred_flat = _flatten(pred_struct)
    gt_flat   = {k: v for k, v in _flatten(gt_struct).items() if v is not None}

    per_field:       dict[str, dict] = {}
    ned_values:      list[float]     = []
    exact_matches:   int             = 0
    within_tol:      int             = 0
    numeric_count:   int             = 0

    for field_path, gt_val in gt_flat.items():
        pred_val   = pred_flat.get(field_path)
        field_type = _get_field_type(field_path, domain)
        metrics    = compute_field_normalization(pred_val, gt_val, field_type)

        per_field[field_path] = metrics

        if metrics.get("exact_match"):
            exact_matches += 1

        if _is_numeric(field_type):
            numeric_count += 1
            if metrics.get("within_tolerance", False):
                within_tol += 1
        else:
            ned = metrics.get("ned")
            if ned is not None:
                ned_values.append(ned)

    n_fields      = len(gt_flat)
    mean_ned      = sum(ned_values) / len(ned_values) if ned_values else 0.0
    exact_rate    = exact_matches / n_fields          if n_fields   else 0.0
    num_tol_rate  = within_tol    / numeric_count     if numeric_count else None

    return {
        "domain":                   domain,
        "n_fields_evaluated":       n_fields,
        "exact_match_rate":         round(exact_rate, 4),
        "mean_ned":                 round(mean_ned, 4),
        "numeric_tolerance_rate":   round(num_tol_rate, 4) if num_tol_rate is not None else None,
        "n_exact_matches":          exact_matches,
        "n_numeric_within_tol":     within_tol,
        "per_field":                per_field,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("PART 1 — Levenshtein sanity checks")
    print("=" * 60)
    cases = [
        ("kitten", "sitting", 3),
        ("ashwin", "ashwln",  1),
        ("P1-093482", "P1-O93482", 1),
        ("hello", "hello", 0),
        ("", "abc", 3),
    ]
    for a, b, expected in cases:
        dist   = levenshtein(a, b)
        status = "✓" if dist == expected else f"✗ got {dist}"
        print(f"  '{a}' vs '{b}' → {dist}  {status}")

    print("\n" + "=" * 60)
    print("PART 2 — Per-field normalization metrics")
    print("=" * 60)

    field_cases = [
        # (pred, gt, type, label)
        ("2002-08-12",  "2002-08-12",  "date",   "Date — exact match"),
        ("2002-08-11",  "2002-08-12",  "date",   "Date — off by one day"),
        ("ashwin shetty", "ashwin shetty", "string", "Name — exact"),
        ("ashwln shetty", "ashwin shetty", "string", "Name — 1 char error"),
        (5000.0,   5000.0,  "number", "Number — exact"),
        (5050.0,   5000.0,  "number", "Number — 1% off (within tol)"),
        (5100.0,   5000.0,  "number", "Number — 2% off (outside tol)"),
        (None,     "INR",   "string", "Currency — missing prediction"),
    ]
    for pred, gt, ftype, label in field_cases:
        m = compute_field_normalization(pred, gt, ftype)
        em = "✓" if m["exact_match"] else "✗"
        extra = ""
        if "ned" in m:
            extra = f"NED={m['ned']}"
        if "relative_error" in m:
            extra = f"rel_err={m['relative_error']}  within_tol={m['within_tolerance']}"
        print(f"  {label:<38} exact={em}  {extra}")

    print("\n" + "=" * 60)
    print("PART 3 — Full struct normalization metrics")
    print("=" * 60)

    gt = {
        "policyholder": {
            "name": "ashwin shetty", "dob": "2002-08-12", "gender": "male"
        },
        "policy": {
            "policy_number": "p1-093482", "policy_type": "term life",
        },
        "premium": {"amount": 5000.0, "currency": "INR", "payment_frequency": "yearly"},
        "agent":   {"name": "rahul verma", "agent_id": "rv-221"},
    }

    pred_good = {
        "policyholder": {
            "name": "ashwin shetty", "dob": "2002-08-12", "gender": "male"
        },
        "policy": {
            "policy_number": "p1-093482", "policy_type": "term life",
        },
        "premium": {"amount": 5000.0, "currency": "INR", "payment_frequency": "yearly"},
        "agent":   {"name": "rahul verma", "agent_id": "rv-221"},
    }

    pred_noisy = {
        "policyholder": {
            "name": "ashwln 5hetty",   # 2 char errors
            "dob":  "2002-08-11",      # 1 day off
            "gender": "male"
        },
        "policy": {
            "policy_number": "p1-O93482",  # O/0 error
        },
        "premium": {"amount": 5100.0, "currency": "INR"},  # 2% off, missing freq
    }

    for label, pred in [("Perfect", pred_good), ("Noisy OCR prediction", pred_noisy)]:
        m = compute_normalization_metrics(pred, gt, "insurance")
        print(f"\n  [{label}]")
        print(f"    exact_match_rate      : {m['exact_match_rate']}")
        print(f"    mean_ned              : {m['mean_ned']}")
        print(f"    numeric_tolerance_rate: {m['numeric_tolerance_rate']}")
        print(f"    n_exact_matches       : {m['n_exact_matches']} / {m['n_fields_evaluated']}")