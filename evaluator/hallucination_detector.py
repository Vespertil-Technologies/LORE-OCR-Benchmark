"""
evaluator/hallucination_detector.py

Identifies fields in pred_struct that the model invented —
not derivable from ocr_text and not in gt_struct.

A predicted value is considered "derivable from OCR" if a
fuzzy substring search finds it (or something very close to it)
within the OCR text. The threshold is configurable in eval_config.json.

Hallucination rate = hallucinated_fields / total_predicted_fields

Lower is better. 0.0 = model never invented anything.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

_CONFIG_DIR = Path(__file__).parent.parent / "config"

def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

_EVAL_CFG = _load_json(_CONFIG_DIR / "eval_config.json")
FUZZY_THRESHOLD = _EVAL_CFG["thresholds"]["hallucination_fuzzy_match_threshold"]


# ══════════════════════════════════════════════════════════════════════════════
# FUZZY SUBSTRING MATCH (no external dependency)
# ══════════════════════════════════════════════════════════════════════════════

def _fuzzy_ratio(a: str, b: str) -> float:
    """
    Compute a similarity ratio between two strings using a sliding window.
    Approximates rapidfuzz.partial_ratio — finds the best matching substring
    of the longer string against the shorter string.

    Returns a score 0–100 (matches rapidfuzz convention).
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 100.0

    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    s_len = len(shorter)

    if s_len == 0:
        return 0.0

    best = 0.0
    for i in range(len(longer) - s_len + 1):
        window = longer[i:i + s_len]
        # Count matching characters positionally
        matches = sum(c1 == c2 for c1, c2 in zip(shorter, window))
        ratio = (2.0 * matches) / (s_len + s_len) * 100
        if ratio > best:
            best = ratio
        if best == 100.0:
            break

    return best


def _value_in_ocr(value: Any, ocr_text: str, threshold: float) -> bool:
    """
    Check if a predicted value is derivable from the OCR text.
    Uses fuzzy substring matching — threshold from eval_config.json.
    """
    if value is None:
        return True   # None is never hallucinated
    val_str = str(value).strip().lower()
    ocr_str = ocr_text.lower()

    if not val_str:
        return True

    # Direct substring check first (fast path)
    if val_str in ocr_str:
        return True

    # Fuzzy check for OCR-corrupted values
    score = _fuzzy_ratio(val_str, ocr_str)
    return score >= threshold


# ══════════════════════════════════════════════════════════════════════════════
# FLATTEN
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def detect_hallucinations(
    pred_struct: dict,
    gt_struct:   dict,
    ocr_text:    str,
    threshold:   float = FUZZY_THRESHOLD,
) -> dict:
    """
    Detect hallucinated fields in pred_struct.

    A field is hallucinated if:
        - Its value is NOT None
        - Its path is absent from GT or its GT value is None
        - Its predicted value cannot be found (fuzzily) in the OCR text

    Args:
        pred_struct: The parsed prediction dict (normalized).
        gt_struct:   The ground truth dict.
        ocr_text:    The raw OCR text the model received.
        threshold:   Fuzzy match score (0–100) to consider a value "in OCR".

    Returns:
        Dict with hallucination_rate, hallucinated_fields, and per_field detail.
    """
    pred_flat = _flatten(pred_struct)
    gt_flat   = _flatten(gt_struct)

    per_field:           dict[str, dict] = {}
    hallucinated_fields: list[str]       = []
    total_predicted = 0

    for field_path, pred_val in pred_flat.items():
        if pred_val is None:
            continue  # None predictions are not hallucinations

        total_predicted += 1
        gt_val = gt_flat.get(field_path)

        # Determine if the value is derivable from OCR
        in_ocr = _value_in_ocr(pred_val, ocr_text, threshold)

        # Determine if it matches GT
        gt_match = False
        if gt_val is not None:
            pred_str = str(pred_val).strip().lower()
            gt_str   = str(gt_val).strip().lower()
            gt_match = pred_str == gt_str

        # Hallucinated: not in GT AND not derivable from OCR
        is_hallucinated = (
            (gt_val is None or not gt_match) and not in_ocr
        )

        field_result = {
            "pred_value":       pred_val,
            "gt_value":         gt_val,
            "in_gt":            gt_val is not None,
            "matches_gt":       gt_match,
            "found_in_ocr":     in_ocr,
            "hallucinated":     is_hallucinated,
        }

        per_field[field_path] = field_result
        if is_hallucinated:
            hallucinated_fields.append(field_path)

    hallucination_rate = (
        len(hallucinated_fields) / total_predicted if total_predicted > 0 else 0.0
    )

    return {
        "hallucination_rate":  round(hallucination_rate, 4),
        "n_hallucinated":      len(hallucinated_fields),
        "n_predicted":         total_predicted,
        "hallucinated_fields": hallucinated_fields,
        "per_field":           per_field,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    ocr_text = (
        "Namc: Ashwln 5hetty\n"
        "D08: 12/08/2002\n"
        "Gender: Male\n"
        "Polcy No: P1-0934B2\n"
        "Amount: 5,O00\n"
        "Currency: INR\n"
        "Nominee: N/A\n"         # extraneous field — in OCR but not in schema
        "Branch Code: 4421\n"    # extraneous field — in OCR but not in schema
    )

    gt = {
        "policyholder": {
            "name": "ashwin shetty", "dob": "2002-08-12",
            "gender": "male", "contact_number": None, "address": None
        },
        "policy":  {"policy_number": "p1-093482"},
        "premium": {"amount": 5000.0, "currency": "INR"},
        "agent":   {"name": None, "agent_id": None},
    }

    cases = [
        ("Clean prediction — no hallucinations", {
            "policyholder": {"name": "ashwin shetty", "dob": "2002-08-12", "gender": "male"},
            "policy":       {"policy_number": "p1-093482"},
            "premium":      {"amount": 5000.0, "currency": "INR"},
        }),
        ("Hallucinated field from extraneous OCR text", {
            "policyholder": {"name": "ashwin shetty", "dob": "2002-08-12"},
            "policy":       {"policy_number": "p1-093482"},
            "premium":      {"amount": 5000.0, "currency": "INR"},
            "nominee":      "N/A",          # in OCR but not in schema → found in OCR, not hallucinated
        }),
        ("Fully invented field — not in OCR or GT", {
            "policyholder": {"name": "ashwin shetty", "dob": "2002-08-12"},
            "policy":       {"policy_number": "p1-093482"},
            "premium":      {"amount": 5000.0, "currency": "INR"},
            "policyholder.contact_number": "9999999999",  # invented — not in OCR
        }),
        ("Ghost value — wrong value for real field", {
            "policyholder": {"name": "ashwin shetty", "dob": "2002-08-12"},
            "policy":       {"policy_number": "p1-000000"},  # not in OCR, not in GT
            "premium":      {"amount": 5000.0, "currency": "INR"},
        }),
    ]

    for label, pred in cases:
        result = detect_hallucinations(pred, gt, ocr_text)
        print(f"\n{'─'*55}")
        print(f"Case: {label}")
        print(f"  hallucination_rate : {result['hallucination_rate']}")
        print(f"  hallucinated       : {result['hallucinated_fields']}")
        print(f"  n_predicted        : {result['n_predicted']}")