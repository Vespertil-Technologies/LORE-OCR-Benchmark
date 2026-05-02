"""
evaluator/field_metrics.py

Computes precision, recall, and F1 for field presence and value correctness.

Operates on normalized pred_struct and gt_struct - call normalizers.py first.
Works on both flat (receipts) and nested (insurance, hospital) structures.

Metrics produced:
    - field_precision     : fraction of predicted fields that exist in GT
    - field_recall        : fraction of GT fields that were predicted
    - field_f1            : harmonic mean of precision and recall
    - required_f1         : F1 computed only over required fields
    - optional_f1         : F1 computed only over optional fields
    - exact_match_rate    : fraction of matched fields with correct value
    - per_field           : per-field breakdown dict
"""

from __future__ import annotations

from typing import Any

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _flatten(struct: dict, prefix: str = "") -> dict[str, Any]:
    """
    Flatten a nested dict into dotted-path keys.
    {"policyholder": {"name": "X"}} → {"policyholder.name": "X"}
    """
    flat = {}
    for key, value in struct.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=path))
        else:
            flat[path] = value
    return flat


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# Fields whose values must match exactly (case-sensitive, no fuzzy)
# These are identifiers - a model cannot guess them from context
_EXACT_ID_FIELDS = {
    "receipt_number", "policy.policy_number", "insurance.policy_number",
    "attending_physician.id", "agent.agent_id",
}


def _values_match(pred_val: Any, gt_val: Any, field_path: str = "") -> bool:
    """
    Check if a predicted value matches ground truth after normalization.

    Strictness rules:
    - ID fields (policy numbers, receipt numbers, doctor IDs):
        case-sensitive exact string match - no tolerance at all
    - Numbers: exact match after normalization - no percentage tolerance
    - Strings: case-insensitive, whitespace-normalized
    """
    if gt_val is None and pred_val is None:
        return True
    if gt_val is None or pred_val is None:
        return False

    # ID fields - case-sensitive exact match, no tolerance
    if field_path in _EXACT_ID_FIELDS:
        return str(pred_val).strip() == str(gt_val).strip()

    # Numeric comparison - exact after normalization (no % tolerance)
    if isinstance(gt_val, (int, float)) and isinstance(pred_val, (int, float)):
        return round(float(pred_val), 2) == round(float(gt_val), 2)

    # String comparison - case-insensitive, whitespace-normalized
    pred_s = " ".join(str(pred_val).strip().lower().split())
    gt_s   = " ".join(str(gt_val).strip().lower().split())
    return pred_s == gt_s


# ══════════════════════════════════════════════════════════════════════════════
# MAIN METRIC FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_field_metrics(
    pred_struct:     dict,
    gt_struct:       dict,
    required_fields: list[str],
    domain:          str,
) -> dict:
    """
    Compute field-level metrics for one sample.

    Args:
        pred_struct:     Normalized prediction dict (from normalizers.py).
        gt_struct:       Normalized ground truth dict.
        required_fields: List of dotted field paths that are required.
        domain:          Domain name (for context in output).

    Returns:
        Dict with overall metrics and per-field breakdown.
    """
    pred_flat = _flatten(pred_struct)
    gt_flat   = _flatten(gt_struct)

    # Remove None values from GT - optional fields absent from GT
    # are not expected to be predicted
    gt_present = {k: v for k, v in gt_flat.items() if v is not None}
    required_set = set(required_fields)

    # ── Field presence sets ────────────────────────────────────────────────
    pred_keys    = set(pred_flat.keys())
    gt_keys      = set(gt_present.keys())
    matched_keys = pred_keys & gt_keys

    # ── Overall precision / recall / F1 ───────────────────────────────────
    precision = len(matched_keys) / len(pred_keys) if pred_keys else 0.0
    recall    = len(matched_keys) / len(gt_keys)   if gt_keys   else 0.0
    f1        = _f1(precision, recall)

    # ── Required fields only ───────────────────────────────────────────────
    req_gt   = {k: v for k, v in gt_present.items() if k in required_set}
    req_pred = {k: v for k, v in pred_flat.items()  if k in required_set}
    req_matched = set(req_pred.keys()) & set(req_gt.keys())

    req_precision = len(req_matched) / len(req_pred) if req_pred else 0.0
    req_recall    = len(req_matched) / len(req_gt)   if req_gt   else 0.0
    required_f1   = _f1(req_precision, req_recall)

    # ── Optional fields only ───────────────────────────────────────────────
    opt_gt   = {k: v for k, v in gt_present.items() if k not in required_set}
    opt_pred = {k: v for k, v in pred_flat.items()  if k not in required_set}
    opt_matched = set(opt_pred.keys()) & set(opt_gt.keys())

    opt_precision = len(opt_matched) / len(opt_pred) if opt_pred else 0.0
    opt_recall    = len(opt_matched) / len(opt_gt)   if opt_gt   else 1.0  # no optionals in GT = perfect
    optional_f1   = _f1(opt_precision, opt_recall)

    # ── Value exact match ──────────────────────────────────────────────────
    exact_matches = 0
    per_field     = {}

    # Evaluate every GT field
    for field_path, gt_val in gt_present.items():
        pred_val   = pred_flat.get(field_path)
        present    = field_path in pred_flat
        match      = _values_match(pred_val, gt_val, field_path) if present else False
        is_required = field_path in required_set

        per_field[field_path] = {
            "predicted":    present,
            "exact_match":  match,
            "is_required":  is_required,
            "gt_value":     gt_val,
            "pred_value":   pred_val,
        }
        if match:
            exact_matches += 1

    # Also flag hallucinated fields (predicted but not in GT)
    for field_path in pred_keys - gt_keys:
        per_field[field_path] = {
            "predicted":    True,
            "exact_match":  False,
            "is_required":  field_path in required_set,
            "gt_value":     None,
            "pred_value":   pred_flat[field_path],
            "hallucinated": True,
        }

    exact_match_rate = exact_matches / len(gt_present) if gt_present else 0.0

    return {
        "domain":           domain,
        "field_precision":  round(precision, 4),
        "field_recall":     round(recall, 4),
        "field_f1":         round(f1, 4),
        "required_f1":      round(required_f1, 4),
        "optional_f1":      round(optional_f1, 4),
        "exact_match_rate": round(exact_match_rate, 4),
        "n_gt_fields":      len(gt_present),
        "n_pred_fields":    len(pred_keys),
        "n_matched_fields": len(matched_keys),
        "n_exact_matches":  exact_matches,
        "per_field":        per_field,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    required = [
        "policyholder.name", "policyholder.dob",
        "policy.policy_number", "premium.amount", "premium.currency"
    ]

    gt = {
        "policyholder": {"name": "ashwin shetty", "dob": "2002-08-12", "gender": "male"},
        "policy":       {"policy_number": "p1-093482", "policy_type": "term life"},
        "premium":      {"amount": 5000.0, "currency": "INR", "payment_frequency": "yearly"},
        "agent":        {"name": "rahul verma", "agent_id": "rv-221"},
    }

    cases = [
        ("Perfect prediction", {
            "policyholder": {"name": "ashwin shetty", "dob": "2002-08-12", "gender": "male"},
            "policy":       {"policy_number": "p1-093482", "policy_type": "term life"},
            "premium":      {"amount": 5000.0, "currency": "INR", "payment_frequency": "yearly"},
            "agent":        {"name": "rahul verma", "agent_id": "rv-221"},
        }),
        ("Missing optional fields (agent)", {
            "policyholder": {"name": "ashwin shetty", "dob": "2002-08-12"},
            "policy":       {"policy_number": "p1-093482"},
            "premium":      {"amount": 5000.0, "currency": "INR"},
        }),
        ("Wrong values on required fields", {
            "policyholder": {"name": "wrong name",   "dob": "1999-01-01"},
            "policy":       {"policy_number": "XX-000"},
            "premium":      {"amount": 9999.0, "currency": "USD"},
        }),
        ("Hallucinated extra fields + missing required", {
            "policyholder": {"name": "ashwin shetty", "dob": "2002-08-12"},
            "policy":       {"policy_number": "p1-093482"},
            "premium":      {"amount": 5000.0, "currency": "INR"},
            "extra_section":{"fake_field": "invented value"},
        }),
        ("Empty prediction (parse failure)", {}),
    ]

    for label, pred in cases:
        metrics = compute_field_metrics(pred, gt, required, "insurance")
        print(f"\n{'─'*55}")
        print(f"Case: {label}")
        print(f"  field_f1        : {metrics['field_f1']}")
        print(f"  required_f1     : {metrics['required_f1']}")
        print(f"  optional_f1     : {metrics['optional_f1']}")
        print(f"  exact_match_rate: {metrics['exact_match_rate']}")
        print(f"  matched {metrics['n_matched_fields']}/{metrics['n_gt_fields']} GT fields, "
              f"{metrics['n_exact_matches']} exact matches")
