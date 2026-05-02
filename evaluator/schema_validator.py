"""
evaluator/schema_validator.py

Checks whether pred_struct conforms to the domain's schema from domains.json.

Checks performed:
    1. parse_status == "failure" → immediately invalid, no further checks
    2. Top-level structure matches (flat vs nested as expected by domain)
    3. Required fields are present (not missing entirely)
    4. No unexpected top-level sections (unknown keys)
    5. Leaf value types are correct (number field should not be a string)
    6. Nested sections that are present must be dicts, not scalars

Does NOT use jsonschema library (no network). Pure Python validation
against the frozen domains.json schema.
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

DOMAINS = _load_json(_CONFIG_DIR / "domains.json")


# ══════════════════════════════════════════════════════════════════════════════
# TYPE CHECKING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _expected_python_type(field_type: str):
    """Map schema field type to Python type(s) for isinstance check."""
    ft = field_type.lower()
    if ft == "number":
        return (int, float)
    elif ft in ("date", "time", "string", "phone", "iso-4217", "yyyy-mm-dd", "hh:mm"):
        return str
    else:
        return str  # default


def _check_leaf_type(value: Any, field_type: str) -> bool:
    """Return True if value matches the expected Python type for field_type."""
    if value is None:
        return True   # null is always acceptable (required fields can be null)
    expected = _expected_python_type(field_type)
    return isinstance(value, expected)


# ══════════════════════════════════════════════════════════════════════════════
# RECURSIVE SCHEMA WALKER
# ══════════════════════════════════════════════════════════════════════════════

def _validate_node(
    pred_node:   Any,
    schema_node: dict,
    path:        str,
    violations:  list[str],
) -> None:
    """
    Recursively validate pred_node against schema_node.
    Appends human-readable violation strings to violations list.
    """
    if not isinstance(pred_node, dict):
        violations.append(
            f"Expected a dict at '{path}' but got {type(pred_node).__name__}"
        )
        return

    for key, schema_value in schema_node.items():
        if key.startswith("_"):
            continue

        field_path = f"{path}.{key}" if path else key
        pred_value = pred_node.get(key)

        if isinstance(schema_value, dict) and "type" in schema_value:
            # Leaf field
            field_type = schema_value.get("format") or schema_value["type"]
            if pred_value is not None and not _check_leaf_type(pred_value, field_type):
                violations.append(
                    f"Field '{field_path}': expected {field_type} "
                    f"but got {type(pred_value).__name__} (value: {repr(pred_value)[:40]})"
                )
        elif isinstance(schema_value, dict):
            # Nested section
            if pred_value is not None:
                if not isinstance(pred_value, dict):
                    violations.append(
                        f"Section '{field_path}' should be a dict "
                        f"but got {type(pred_value).__name__}"
                    )
                else:
                    _validate_node(pred_value, schema_value, field_path, violations)

    # Check for unknown top-level keys in pred (one level only)
    if not path:
        schema_keys = {k for k in schema_node if not k.startswith("_")}
        pred_keys   = set(pred_node.keys())
        unknown     = pred_keys - schema_keys
        for k in unknown:
            violations.append(f"Unknown top-level key '{k}' not in schema")


# ══════════════════════════════════════════════════════════════════════════════
# REQUIRED FIELD CHECKER
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


def _check_required_fields(
    pred_struct:     dict,
    required_fields: list[str],
    violations:      list[str],
) -> None:
    """
    Check that all required fields are present in pred_struct.
    They may be null - but they must be present as keys.
    """
    pred_flat = _flatten(pred_struct)

    for field_path in required_fields:
        # Check if the field exists anywhere in the flat pred
        if field_path not in pred_flat:
            violations.append(
                f"Required field '{field_path}' is missing from prediction"
            )


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def validate_schema(
    pred_struct:  dict,
    domain:       str,
    parse_status: str,
) -> dict:
    """
    Validate pred_struct against the domain schema.

    Args:
        pred_struct:  The parsed prediction dict (from json_coercion.py).
                      Not necessarily normalized - schema check happens first.
        domain:       Domain name.
        parse_status: 'success', 'partial', or 'failure' (from json_coercion.py).

    Returns:
        Dict with schema_valid (bool), violations (list of strings),
        and required_fields_present (bool).
    """
    # Immediate fail on parse failure
    if parse_status == "failure":
        return {
            "schema_valid":            False,
            "required_fields_present": False,
            "parse_status":            parse_status,
            "n_violations":            1,
            "violations":              ["Model output could not be parsed as JSON"],
        }

    if not isinstance(pred_struct, dict):
        return {
            "schema_valid":            False,
            "required_fields_present": False,
            "parse_status":            parse_status,
            "violations":              ["Prediction is not a dict"],
        }

    violations: list[str] = []
    domain_cfg   = DOMAINS[domain]
    schema       = domain_cfg["schema"]
    required     = domain_cfg["required_fields"]

    # Run structural + type checks
    _validate_node(pred_struct, schema, path="", violations=violations)

    # Run required field presence checks
    req_violations_before = len(violations)
    _check_required_fields(pred_struct, required, violations)
    required_fields_present = len(violations) == req_violations_before

    schema_valid = len(violations) == 0

    return {
        "schema_valid":            schema_valid,
        "required_fields_present": required_fields_present,
        "parse_status":            parse_status,
        "n_violations":            len(violations),
        "violations":              violations,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    cases = [
        ("Valid nested insurance prediction", "insurance", "success", {
            "policyholder": {"name": "Ashwin Shetty", "dob": "2002-08-12"},
            "policy":       {"policy_number": "P1-093482"},
            "premium":      {"amount": 5000.0, "currency": "INR"},
        }),
        ("Missing required field (policy_number)", "insurance", "success", {
            "policyholder": {"name": "Ashwin Shetty", "dob": "2002-08-12"},
            "premium":      {"amount": 5000.0, "currency": "INR"},
        }),
        ("Wrong type - amount as string", "insurance", "success", {
            "policyholder": {"name": "Ashwin Shetty", "dob": "2002-08-12"},
            "policy":       {"policy_number": "P1-093482"},
            "premium":      {"amount": "five thousand", "currency": "INR"},
        }),
        ("Flattened output for nested domain (structure error)", "insurance", "success", {
            "policyholder.name":    "Ashwin Shetty",
            "policy.policy_number": "P1-093482",
            "premium.amount":       5000.0,
            "premium.currency":     "INR",
        }),
        ("Unknown top-level key", "insurance", "success", {
            "policyholder": {"name": "Ashwin Shetty", "dob": "2002-08-12"},
            "policy":       {"policy_number": "P1-093482"},
            "premium":      {"amount": 5000.0, "currency": "INR"},
            "nominee":      "N/A",
        }),
        ("Parse failure - no further checks", "insurance", "failure", {}),
        ("Valid flat receipts prediction", "receipts", "success", {
            "vendor_name":  "SwiggyMart",
            "date":         "2024-03-14",
            "total_amount": 368.16,
            "currency":     "INR",
        }),
    ]

    for label, domain, parse_status, pred in cases:
        assert isinstance(pred, dict)  # __main__ test cases all use dicts
        result = validate_schema(pred, domain, parse_status)
        status = "VALID" if result["schema_valid"] else "INVALID"
        print(f"\n{'─'*55}")
        print(f"Case: {label}")
        print(f"  {status}  ({result['n_violations']} violation(s))")
        for v in result["violations"]:
            print(f"    → {v}")
