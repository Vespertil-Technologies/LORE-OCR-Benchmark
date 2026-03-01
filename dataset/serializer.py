"""
dataset/serializer.py

Converts a clean gt_struct dict into a flat multiline text string that
resembles raw form output before OCR noise is applied.

Responsibilities:
- Flatten nested dicts into key: value lines
- Randomize key label selection (from domains.json key_label_variants)
- Randomize line ordering within each section
- Write dates in a non-ISO format (noise engine may corrupt further)
- Write numbers with thousand-separator commas
- Never apply noise — that is strictly the noise engine's job

The output of this module is what the noise engine receives as input.
"""

import json
import random
from pathlib import Path
from datetime import datetime, date as date_type
from typing import Any

# ── Config loading ─────────────────────────────────────────────────────────────

_CONFIG_DIR = Path(__file__).parent.parent / "config"

def _load_domains() -> dict:
    with open(_CONFIG_DIR / "domains.json", encoding="utf-8") as f:
        return json.load(f)

DOMAINS = _load_domains()

# ── Date pre-serialization formats ────────────────────────────────────────────
# These are the "clean" non-ISO formats the serializer writes.
# The noise engine's date_format_vary and date_partial functions may corrupt them further.
# We intentionally avoid ISO here so the model always has to normalize.

_DATE_PREFORMATS = [
    "%d/%m/%Y",    # 14/03/2024  — most common in India
    "%d-%m-%Y",    # 14-03-2024
    "%d %b %Y",    # 14 Mar 2024
    "%d/%m/%y",    # 14/03/24
]


def _format_date(iso_date_str: str, rng: random.Random) -> str:
    """Convert an ISO date string to a randomly chosen non-ISO format."""
    try:
        dt = datetime.strptime(iso_date_str, "%Y-%m-%d")
        fmt = rng.choice(_DATE_PREFORMATS)
        return dt.strftime(fmt)
    except (ValueError, TypeError):
        return str(iso_date_str)  # Return as-is if unparseable


def _format_number(value: Any) -> str:
    """Format a numeric value with thousand-separator commas."""
    try:
        f = float(value)
        if f == int(f):
            # Whole number — no decimal
            return f"{int(f):,}"
        else:
            return f"{f:,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _pick_label(field_path: str, domain: str, rng: random.Random) -> str:
    """
    Pick a random key label from the domain's key_label_variants for this field.
    Falls back to the last segment of the dotted path if no variant is defined.
    """
    variants = DOMAINS[domain]["key_label_variants"].get(field_path)
    if variants:
        return rng.choice(variants)
    # Fallback: humanize the field name from the last path segment
    return field_path.split(".")[-1].replace("_", " ").title()


def _pick_separator(rng: random.Random) -> str:
    """
    Choose a key-value separator. Heavily weighted toward colon
    since the noise engine is responsible for removing/swapping it.
    """
    # 85% colon, 10% colon+space already done, 5% other (pre-noise variation)
    return rng.choices([": ", ":", " : "], weights=[70, 20, 10])[0]


# ── Core flattening logic ──────────────────────────────────────────────────────

def _flatten_struct(
    gt_struct: dict,
    domain: str,
    schema: dict,
    rng: random.Random,
    prefix: str = ""
) -> list[tuple[str, str, str]]:
    """
    Recursively walk gt_struct and produce a list of (field_path, label, value_str) tuples.
    Skips fields whose value is None (optional fields absent from OCR).
    Returns only leaf-level fields.
    """
    rows = []
    for key, value in gt_struct.items():
        field_path = f"{prefix}.{key}" if prefix else key
        schema_node = schema.get(key, {})

        if value is None:
            # Optional field absent — do not emit a line for it
            continue

        if isinstance(value, dict):
            # Nested section — recurse
            child_rows = _flatten_struct(value, domain, schema_node, rng, prefix=field_path)
            rows.extend(child_rows)
        else:
            # Leaf field — determine field type from schema and format value
            field_type = schema_node.get("type", "string")
            label = _pick_label(field_path, domain, rng)

            if field_type == "date" and isinstance(value, str):
                value_str = _format_date(value, rng)
            elif field_type == "number":
                value_str = _format_number(value)
            else:
                value_str = str(value)

            rows.append((field_path, label, value_str))

    return rows


# ── Section grouping ───────────────────────────────────────────────────────────

def _group_by_section(rows: list[tuple[str, str, str]]) -> dict[str, list]:
    """
    Group (field_path, label, value) rows by their top-level section.
    E.g. 'policyholder.name' goes into the 'policyholder' section.
    Flat domains (receipts) go into a single '__root__' section.
    """
    sections: dict[str, list] = {}
    for field_path, label, value_str in rows:
        parts = field_path.split(".")
        section = parts[0] if len(parts) > 1 else "__root__"
        sections.setdefault(section, []).append((field_path, label, value_str))
    return sections


# ── Main serializer ────────────────────────────────────────────────────────────

def serialize(
    gt_struct: dict,
    domain: str,
    rng: random.Random,
) -> str:
    """
    Convert a clean gt_struct into a multiline OCR-like text string.

    Args:
        gt_struct: Clean ground-truth dict matching the domain schema.
        domain:    One of 'receipts', 'insurance', 'hospital'.
        rng:       Seeded random.Random instance for reproducibility.

    Returns:
        A multiline string where each line is 'Label: value'.
        Section ordering is fixed (matches schema order).
        Line ordering within each section is slightly randomized.
    """
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'. Must be one of: {list(DOMAINS.keys())}")

    schema = DOMAINS[domain]["schema"]
    rows = _flatten_struct(gt_struct, domain, schema, rng)
    sections = _group_by_section(rows)

    lines = []
    for section, section_rows in sections.items():
        # Shuffle lines within a section to simulate real form variation
        # but keep sections in their defined order (patient before visit, etc.)
        rng.shuffle(section_rows)

        for _, label, value_str in section_rows:
            sep = _pick_separator(rng)
            lines.append(f"{label}{sep}{value_str}")

        # Occasionally add a blank line between sections (realistic OCR behaviour)
        if section != "__root__" and rng.random() < 0.3:
            lines.append("")

    return "\n".join(lines).strip()


# ── Usage example ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint

    # ── Example 1: Receipts (flat) ─────────────────────────────────────────
    receipt_gt = {
        "vendor_name":    "SwiggyMart",
        "date":           "2024-03-14",
        "time":           "14:32",
        "receipt_number": "RCT-20240314-007",
        "subtotal":       312.00,
        "tax_amount":     56.16,
        "total_amount":   368.16,
        "currency":       "INR",
        "payment_method": "UPI",
        "line_items":     "2x Bread 45.00, 1x Milk 222.00, 1x Eggs 45.00"
    }

    rng = random.Random(42)
    print("=" * 60)
    print("DOMAIN: receipts")
    print("=" * 60)
    print(serialize(receipt_gt, "receipts", rng))

    # ── Example 2: Insurance (nested, some optional fields absent) ─────────
    insurance_gt = {
        "policyholder": {
            "name":           "Ashwin Shetty",
            "dob":            "2002-08-12",
            "gender":         "Male",
            "contact_number": None,   # absent — will be omitted
            "address":        None    # absent — will be omitted
        },
        "policy": {
            "policy_number": "P1-093482",
            "policy_type":   "Term Life",
            "start_date":    "2023-01-01",
            "end_date":      None     # absent
        },
        "premium": {
            "amount":            5000,
            "currency":          "INR",
            "payment_frequency": "yearly"
        },
        "agent": {
            "name":     "Rahul Verma",
            "agent_id": "RV-221"
        }
    }

    rng2 = random.Random(42)
    print("\n" + "=" * 60)
    print("DOMAIN: insurance")
    print("=" * 60)
    print(serialize(insurance_gt, "insurance", rng2))

    # ── Example 3: Hospital (nested, many optional fields absent) ──────────
    hospital_gt = {
        "patient": {
            "name":           "Priya Sharma",
            "dob":            "1995-11-07",
            "gender":         "Female",
            "blood_group":    "B+",
            "contact_number": None,
            "address":        None
        },
        "visit": {
            "date":             "2024-03-14",
            "time":             "09:15",
            "department":       "Cardiology",
            "reason_for_visit": "Chest pain and shortness of breath"
        },
        "vitals": {
            "blood_pressure": "120/80",
            "pulse":          "78",
            "temperature":    "98.6F",
            "weight":         None,
            "height":         None
        },
        "insurance": {
            "provider":      "StarHealth",
            "policy_number": "SH-4521"
        },
        "attending_physician": {
            "name": "Dr Mehta",
            "id":   "MH-0042"
        }
    }

    rng3 = random.Random(42)
    print("\n" + "=" * 60)
    print("DOMAIN: hospital")
    print("=" * 60)
    print(serialize(hospital_gt, "hospital", rng3))