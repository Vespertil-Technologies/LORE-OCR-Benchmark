"""
evaluator/correction_metrics.py

Measures correction gain: how much did the LLM improve the OCR value
toward ground truth, compared to just using the raw OCR text as-is?

Formula per field:
    ocr_distance = levenshtein(raw_ocr_value, gt_value)
    llm_distance = levenshtein(pred_value, gt_value)
    correction_gain = (ocr_distance - llm_distance) / ocr_distance

Range:
     1.0 → perfect correction (pred exactly matches GT)
     0.0 → no improvement (LLM output as far from GT as raw OCR)
    <0.0 → regression (LLM made it worse than raw OCR)

Fields where ocr_distance == 0 (OCR was already correct) are excluded
from the mean - the LLM had nothing to correct.

This metric requires raw_text from generation_meta (stored in prediction records).
"""

from __future__ import annotations
import re
from typing import Any

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))
from evaluator.normalization_metrics import levenshtein, _flatten


# ══════════════════════════════════════════════════════════════════════════════
# RAW OCR VALUE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

def _extract_ocr_value(
    field_path:   str,
    ocr_text:     str,
    key_variants: list[str],
) -> str | None:
    """
    Find the value for a field in the raw OCR text by scanning for
    its key label variants.

    Rules:
    - Key must appear at the start of a line (with optional leading whitespace)
      to avoid matching "Name" inside "Agent Name".
    - Variants are tried longest-first so specific labels win over generic ones.
    - Separator and any trailing whitespace are stripped from the captured value.

    Returns the raw string value as it appears in the OCR text,
    or None if the field cannot be located.
    """
    sep_pattern   = r"[ \t]*[:\-–\/][ \t]*"
    value_pattern = r"(.+?)[ \t]*(?:\n|$)"

    # Sort longest-first so "Agent Name" is tried before "Name"
    sorted_variants = sorted(key_variants, key=len, reverse=True)

    for variant in sorted_variants:
        escaped = re.escape(variant)
        # Anchor to start of line (with optional leading whitespace)
        pattern = re.compile(
            r"(?:^|\n)[ \t]*" + escaped + r"\b" + sep_pattern + value_pattern,
            re.IGNORECASE
        )
        m = pattern.search(ocr_text)
        if m:
            val = m.group(1).strip()
            if val:
                return val

    return None


# ══════════════════════════════════════════════════════════════════════════════
# PER-FIELD CORRECTION METRIC
# ══════════════════════════════════════════════════════════════════════════════

def compute_field_correction(
    raw_ocr_value: str | None,
    pred_value:    Any,
    gt_value:      Any,
) -> dict:
    """
    Compute correction gain for a single field.

    Args:
        raw_ocr_value: The raw value extracted from OCR text (before LLM).
        pred_value:    The LLM's predicted value (after normalization).
        gt_value:      The ground truth value.

    Returns:
        Dict with correction_gain and supporting distances.
    """
    # Skip fields where GT is None (optional field absent)
    if gt_value is None:
        return {"skipped": True, "reason": "gt_value is None"}

    gt_str  = str(gt_value).strip().lower()
    pred_str = str(pred_value).strip().lower() if pred_value is not None else ""

    llm_distance = levenshtein(pred_str, gt_str)

    if raw_ocr_value is None:
        # Can't compute OCR distance - field not found in OCR text
        return {
            "skipped":      True,
            "reason":       "field not found in ocr_text",
            "llm_distance": llm_distance,
        }

    ocr_str      = str(raw_ocr_value).strip().lower()
    ocr_distance = levenshtein(ocr_str, gt_str)

    if ocr_distance == 0:
        # OCR was already correct - no correction needed, skip from mean
        return {
            "skipped":        True,
            "reason":         "ocr_already_correct",
            "ocr_distance":   0,
            "llm_distance":   llm_distance,
            "correction_gain": 1.0 if llm_distance == 0 else None,
        }

    gain = (ocr_distance - llm_distance) / ocr_distance

    return {
        "skipped":        False,
        "ocr_value":      raw_ocr_value,
        "ocr_distance":   ocr_distance,
        "llm_distance":   llm_distance,
        "correction_gain": round(gain, 4),
        "improved":       gain > 0,
        "regressed":      gain < 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STRUCT-LEVEL CORRECTION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_correction_metrics(
    ocr_text:     str,
    raw_text:     str,
    pred_struct:  dict,
    gt_struct:    dict,
    domain:       str,
    domains_cfg:  dict,
) -> dict:
    """
    Compute correction metrics across all fields in a sample.

    Args:
        ocr_text:    Corrupted OCR text (what the model received).
        raw_text:    Clean serialized text before noise injection
                     (from generation_meta.raw_text in prediction records).
        pred_struct: Normalized prediction dict.
        gt_struct:   Normalized ground truth dict.
        domain:      Domain name.
        domains_cfg: Loaded domains.json.

    Returns:
        Dict with mean_correction_gain and per_field breakdown.
    """
    pred_flat = _flatten(pred_struct)
    gt_flat   = {k: v for k, v in _flatten(gt_struct).items() if v is not None}
    key_variants = domains_cfg[domain]["key_label_variants"]

    per_field:  dict[str, dict] = {}
    gains:      list[float]     = []
    n_improved: int             = 0
    n_regressed: int            = 0

    for field_path, gt_val in gt_flat.items():
        # Get key label variants for this field
        variants = key_variants.get(field_path, [field_path.split(".")[-1]])

        # Extract raw OCR value for this field
        raw_ocr_val = _extract_ocr_value(field_path, ocr_text, variants)

        pred_val = pred_flat.get(field_path)
        metrics  = compute_field_correction(raw_ocr_val, pred_val, gt_val)

        per_field[field_path] = metrics

        if not metrics.get("skipped"):
            gain = metrics["correction_gain"]
            gains.append(gain)
            if metrics.get("improved"):
                n_improved += 1
            if metrics.get("regressed"):
                n_regressed += 1

    mean_gain = sum(gains) / len(gains) if gains else None
    min_gain  = min(gains)              if gains else None
    max_gain  = max(gains)              if gains else None

    return {
        "domain":              domain,
        "mean_correction_gain": round(mean_gain, 4) if mean_gain is not None else None,
        "min_correction_gain":  round(min_gain, 4)  if min_gain  is not None else None,
        "max_correction_gain":  round(max_gain, 4)  if max_gain  is not None else None,
        "n_fields_evaluated":  len(gains),
        "n_improved":          n_improved,
        "n_regressed":         n_regressed,
        "per_field":           per_field,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    with open(Path(__file__).parent.parent / "config" / "domains.json") as f:
        domains_cfg = json.load(f)

    # Simulated OCR text (what the model received - noisy)
    ocr_text = (
        "Namc: Ashwln 5hetty\n"
        "D08: 12/08/2002\n"
        "Gender: Male\n"
        "Polcy No: P1-0934B2\n"
        "Policy Type: Term Llfe\n"
        "Amount: 5,O00\n"
        "Currency: INR\n"
        "Pay Freq: yearly\n"
        "Agent: Rahul Verma\n"
        "ID: RV-221\n"
    )

    # Raw text (clean, before noise - stored in generation_meta)
    raw_text = (
        "Name: Ashwin Shetty\n"
        "DOB: 12/08/2002\n"
        "Gender: Male\n"
        "Policy No: P1-093482\n"
        "Policy Type: Term Life\n"
        "Amount: 5,000\n"
        "Currency: INR\n"
        "Pay Freq: yearly\n"
        "Agent: Rahul Verma\n"
        "ID: RV-221\n"
    )

    # GT (normalized)
    gt = {
        "policyholder": {"name": "ashwin shetty", "dob": "2002-08-12", "gender": "male"},
        "policy":       {"policy_number": "p1-093482", "policy_type": "term life"},
        "premium":      {"amount": 5000.0, "currency": "INR", "payment_frequency": "yearly"},
        "agent":        {"name": "rahul verma", "agent_id": "rv-221"},
    }

    cases = [
        ("Perfect correction", {
            "policyholder": {"name": "ashwin shetty", "dob": "2002-08-12", "gender": "male"},
            "policy":       {"policy_number": "p1-093482", "policy_type": "term life"},
            "premium":      {"amount": 5000.0, "currency": "INR", "payment_frequency": "yearly"},
            "agent":        {"name": "rahul verma", "agent_id": "rv-221"},
        }),
        ("Partial correction", {
            "policyholder": {"name": "ashwln shetty", "dob": "2002-08-12"},  # name still wrong
            "policy":       {"policy_number": "p1-093482"},
            "premium":      {"amount": 5000.0, "currency": "INR"},
        }),
        ("Regression - LLM made it worse", {
            "policyholder": {"name": "xyz completely wrong", "dob": "1900-01-01"},
            "policy":       {"policy_number": "xx-000000"},
            "premium":      {"amount": 9999.0, "currency": "USD"},
        }),
    ]

    for label, pred in cases:
        m = compute_correction_metrics(
            ocr_text, raw_text, pred, gt, "insurance", domains_cfg
        )
        print(f"\n{'─'*55}")
        print(f"Case: {label}")
        print(f"  mean_correction_gain : {m['mean_correction_gain']}")
        print(f"  improved / regressed : {m['n_improved']} / {m['n_regressed']}")
        print(f"  fields evaluated     : {m['n_fields_evaluated']}")