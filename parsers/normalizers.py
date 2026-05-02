"""
parsers/normalizers.py

Standardizes extracted field values into canonical form so they
can be compared fairly against ground truth in gt_struct.

Normalizers are called per-field, per-type. They never raise -
they always return the best attempt or None if normalization fails.

Field types (from domains.json):
    date    → YYYY-MM-DD
    time    → HH:MM (24h)
    number  → float
    phone   → 10-digit string
    string  → lowercase, stripped, abbreviations expanded
"""

import json
import re
from pathlib import Path
from typing import Any

# ── Config ─────────────────────────────────────────────────────────────────────

_CONFIG_DIR = Path(__file__).parent.parent / "config"

def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

_EVAL_CONFIG = _load_json(_CONFIG_DIR / "eval_config.json")
_NORM_CFG    = _EVAL_CONFIG["normalization"]

ABBREV_EXPANSIONS: dict[str, str] = {
    k: v for k, v in _NORM_CFG["abbreviation_expansions"].items()
    if not k.startswith("_")
}
PHONE_DIGIT_COUNT = _NORM_CFG["phone_digit_count"]


# ══════════════════════════════════════════════════════════════════════════════
# DATE NORMALIZER
# ══════════════════════════════════════════════════════════════════════════════

# Month name → zero-padded number
_MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

# Ordered list of (pattern, extractor_fn) pairs
# Each extractor returns (year, month, day) as zero-padded strings or None
_DATE_PATTERNS: list[tuple[re.Pattern, Any]] = []


def _reg(pattern: str):
    """Register a date pattern with a named-group extractor."""
    def decorator(fn):
        _DATE_PATTERNS.append((re.compile(pattern, re.IGNORECASE), fn))
        return fn
    return decorator


@_reg(r"^(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})$")
def _iso_like(m):
    return m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)


@_reg(r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})$")
def _dmy_4year(m):
    # Ambiguous - assume DD/MM/YYYY (Indian standard)
    return m.group(3), m.group(2).zfill(2), m.group(1).zfill(2)


@_reg(r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2})$")
def _dmy_2year(m):
    year = int(m.group(3))
    full_year = str(2000 + year if year < 50 else 1900 + year)
    return full_year, m.group(2).zfill(2), m.group(1).zfill(2)


@_reg(r"^(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})$")
def _dmonthname_y(m):
    month = _MONTH_MAP.get(m.group(2)[:3].lower())
    if not month:
        return None
    return m.group(3), month, m.group(1).zfill(2)


@_reg(r"^([A-Za-z]{3,9})\s+(\d{1,2})[,\s]+(\d{4})$")
def _monthname_dy(m):
    month = _MONTH_MAP.get(m.group(1)[:3].lower())
    if not month:
        return None
    return m.group(3), month, m.group(2).zfill(2)


@_reg(r"^(\d{8})$")
def _compact(m):
    # DDMMYYYY
    s = m.group(1)
    return s[4:8], s[2:4], s[0:2]


def normalize_date(value: Any) -> str | None:
    """
    Convert any date-like string to ISO 8601 YYYY-MM-DD.
    Returns None if conversion fails.
    """
    if value is None:
        return None
    text = str(value).strip()

    for pattern, extractor in _DATE_PATTERNS:
        m = pattern.match(text)
        if m:
            result = extractor(m)
            if result is None:
                continue
            year, month, day = result
            # Basic sanity check
            try:
                y, mo, d = int(year), int(month), int(day)
                if 1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
                    return f"{year}-{month}-{day}"
            except ValueError:
                continue

    return None  # Unparseable


# ══════════════════════════════════════════════════════════════════════════════
# TIME NORMALIZER
# ══════════════════════════════════════════════════════════════════════════════

_TIME_PATTERN = re.compile(
    r"(\d{1,2})[:\.](\d{2})(?:[:\.](\d{2}))?\s*(am|pm)?",
    re.IGNORECASE
)

def normalize_time(value: Any) -> str | None:
    """
    Convert a time string to HH:MM 24-hour format.
    Returns None if conversion fails.
    """
    if value is None:
        return None
    text = str(value).strip()
    m = _TIME_PATTERN.search(text)
    if not m:
        return None

    hour   = int(m.group(1))
    minute = int(m.group(2))
    ampm   = (m.group(4) or "").lower()

    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0

    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return f"{hour:02d}:{minute:02d}"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# NUMBER NORMALIZER
# ══════════════════════════════════════════════════════════════════════════════

# Currency symbols to strip before parsing
_CURRENCY_SYMBOLS = re.compile(r"[₹$€£¥₩₫฿]")
# Common Indian number suffixes
_LAKH  = re.compile(r"(\d[\d,\.]*)\s*(?:lakh|lac)", re.IGNORECASE)
_CRORE = re.compile(r"(\d[\d,\.]*)\s*crore",         re.IGNORECASE)

def normalize_number(value: Any) -> float | None:
    """
    Convert a numeric string to a float.
    Handles: commas, currency symbols, INR/USD/etc suffixes, lakh/crore.
    Returns None if conversion fails.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    # Handle lakh / crore
    m = _CRORE.search(text)
    if m:
        try:
            return float(m.group(1).replace(",", "")) * 1_00_00_000
        except ValueError:
            pass

    m = _LAKH.search(text)
    if m:
        try:
            return float(m.group(1).replace(",", "")) * 1_00_000
        except ValueError:
            pass

    # Strip everything that isn't a digit, dot, or minus
    text = _CURRENCY_SYMBOLS.sub("", text)
    text = re.sub(r"[^\d\.\-]", "", text)

    if not text or text in (".", "-"):
        return None

    try:
        return float(text)
    except ValueError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PHONE NORMALIZER
# ══════════════════════════════════════════════════════════════════════════════

def normalize_phone(value: Any) -> str | None:
    """
    Strip all non-digit characters and return a 10-digit phone string.
    Returns None if result is not exactly 10 digits.
    """
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    # Strip leading country code (91 for India)
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) == PHONE_DIGIT_COUNT:
        return digits
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STRING NORMALIZER
# ══════════════════════════════════════════════════════════════════════════════

def normalize_string(value: Any) -> str | None:
    """
    Lowercase, strip whitespace, collapse internal spaces,
    and expand known abbreviations.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)         # collapse multiple spaces
    text = ABBREV_EXPANSIONS.get(text, text)  # expand if exact abbrev match
    return text if text else None


# ══════════════════════════════════════════════════════════════════════════════
# CURRENCY CODE NORMALIZER
# ══════════════════════════════════════════════════════════════════════════════

# Map common variants to ISO 4217
_CURRENCY_MAP = {
    "rs":      "INR",
    "inr":     "INR",
    "rupee":   "INR",
    "rupees":  "INR",
    "₹":       "INR",
    "usd":     "USD",
    "dollar":  "USD",
    "dollars": "USD",
    "$":       "USD",
    "eur":     "EUR",
    "euro":    "EUR",
    "euros":   "EUR",
    "€":       "EUR",
    "gbp":     "GBP",
    "pound":   "GBP",
    "pounds":  "GBP",
    "£":       "GBP",
}

def normalize_currency_code(value: Any) -> str | None:
    """
    Normalize a currency string to ISO 4217 code.
    Returns the uppercased value as-is if not in the map (might already be ISO).
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _CURRENCY_MAP:
        return _CURRENCY_MAP[text]
    upper = text.upper()
    # Return as-is if it looks like a 3-letter ISO code
    if re.match(r"^[A-Z]{3}$", upper):
        return upper
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PAYMENT FREQUENCY NORMALIZER
# ══════════════════════════════════════════════════════════════════════════════

_FREQ_MAP = {
    "yearly":      "yearly",
    "annual":      "yearly",
    "annually":    "yearly",
    "per year":    "yearly",
    "p.a.":        "yearly",
    "pa":          "yearly",
    "/yr":         "yearly",
    "/year":       "yearly",
    "monthly":     "monthly",
    "per month":   "monthly",
    "p.m.":        "monthly",
    "pm":          "monthly",
    "/month":      "monthly",
    "/mo":         "monthly",
    "quarterly":   "quarterly",
    "per quarter": "quarterly",
    "/quarter":    "quarterly",
    "half-yearly": "half-yearly",
    "half yearly": "half-yearly",
    "semi-annual": "half-yearly",
    "biannual":    "half-yearly",
    "/half year":  "half-yearly",
}

def normalize_payment_frequency(value: Any) -> str | None:
    """Normalize payment frequency strings to a canonical form."""
    if value is None:
        return None
    text = str(value).strip().lower()
    return _FREQ_MAP.get(text, text)  # return as-is if not in map


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCHER - normalize by field type
# ══════════════════════════════════════════════════════════════════════════════

def normalize_value(value: Any, field_type: str) -> Any:
    """
    Normalize a single field value given its type from domains.json.

    Args:
        value:      The raw extracted value (string, number, None, etc.)
        field_type: Type string from domains.json schema
                    (date, time, number, phone, string, ISO-4217)

    Returns:
        Normalized value, or None if normalization fails.
    """
    if value is None:
        return None

    ft = field_type.lower()

    if ft in ("date", "yyyy-mm-dd"):
        return normalize_date(value)
    elif ft in ("time", "hh:mm"):
        return normalize_time(value)
    elif ft == "number":
        return normalize_number(value)
    elif ft == "phone":
        return normalize_phone(value)
    elif ft in ("iso-4217", "currency"):
        return normalize_currency_code(value)
    elif ft == "payment_frequency":
        return normalize_payment_frequency(value)
    elif ft == "string":
        return normalize_string(value)
    else:
        return normalize_string(value)


# ══════════════════════════════════════════════════════════════════════════════
# STRUCT NORMALIZER - normalize an entire pred_struct
# ══════════════════════════════════════════════════════════════════════════════

def _get_field_type(field_path: str, domain: str, domains_cfg: dict) -> str:
    """
    Walk the schema to find the effective normalizer type for a dotted field path.
    Checks 'format' first (e.g. ISO-4217) then falls back to 'type'.
    Also handles special field paths that need their own normalizer.
    """
    # Path-based overrides for fields whose schema type is "string"
    # but that need a specialised normalizer
    _PATH_OVERRIDES = {
        "payment_frequency":         "payment_frequency",
        "premium.payment_frequency": "payment_frequency",
    }
    leaf = field_path.split(".")[-1]
    if field_path in _PATH_OVERRIDES:
        return _PATH_OVERRIDES[field_path]
    if leaf in _PATH_OVERRIDES:
        return _PATH_OVERRIDES[leaf]

    schema = domains_cfg[domain]["schema"]
    parts  = field_path.split(".")
    node   = schema
    for part in parts:
        node = node.get(part, {})

    # Prefer format over type when present (e.g. format: ISO-4217)
    fmt = node.get("format", "")
    if fmt:
        return fmt
    return node.get("type", "string")


def normalize_struct(
    pred_struct: dict,
    domain: str,
    domains_cfg: dict,
    prefix: str = "",
) -> dict:
    """
    Recursively normalize all values in pred_struct based on their
    field type in domains.json.

    Args:
        pred_struct:  The parsed prediction dict.
        domain:       Domain name.
        domains_cfg:  Loaded domains.json dict.
        prefix:       Internal - dotted path prefix for recursion.

    Returns:
        A new dict with all values normalized.
    """
    result = {}
    for key, value in pred_struct.items():
        field_path = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            result[key] = normalize_struct(value, domain, domains_cfg, prefix=field_path)
        else:
            field_type = _get_field_type(field_path, domain, domains_cfg)
            result[key] = normalize_value(value, field_type)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("=" * 60)
    print("PART 1 - Date normalization")
    print("=" * 60)

    date_cases = [
        ("14/03/2024",    "2024-03-14"),
        ("14-03-2024",    "2024-03-14"),
        ("14 Mar 2024",   "2024-03-14"),
        ("Mar 14, 2024",  "2024-03-14"),
        ("14/03/24",      "2024-03-14"),
        ("2024-03-14",    "2024-03-14"),
        ("14032024",      "2024-03-14"),
        ("12/0B/2002",    None),          # corrupted - cannot parse
        ("not a date",    None),
    ]
    for raw, expected in date_cases:
        result = normalize_date(raw)
        status = "PASS" if result == expected else f"FAIL got {result}"
        print(f"  {raw:<20} → {str(result):<14} {status}")

    print("\n" + "=" * 60)
    print("PART 2 - Number normalization")
    print("=" * 60)

    num_cases = [
        ("5,000",      5000.0),
        ("5,000 INR",  5000.0),
        ("₹ 5000",     5000.0),
        ("368.16",     368.16),
        ("S,OOO",      None),     # OCR-corrupted - can't parse
        ("2 lakh",     200000.0),
        ("1.5 crore",  15000000.0),
        (5000,         5000.0),
    ]
    for raw, expected in num_cases:
        result = normalize_number(raw)
        status = "PASS" if result == expected else f"FAIL got {result}"
        print(f"  {str(raw):<20} → {str(result):<14} {status}")

    print("\n" + "=" * 60)
    print("PART 3 - Currency code normalization")
    print("=" * 60)

    curr_cases = [
        ("INR",     "INR"),
        ("inr",     "INR"),
        ("Rs",      "INR"),
        ("₹",       "INR"),
        ("USD",     "USD"),
        ("dollars", "USD"),
        ("EUR",     "EUR"),
        ("1RN",     None),   # OCR-corrupted
    ]
    for raw, expected in curr_cases:
        result = normalize_currency_code(raw)
        status = "PASS" if result == expected else f"FAIL got {result}"
        print(f"  {raw:<12} → {str(result):<8} {status}")

    print("\n" + "=" * 60)
    print("PART 4 - Full struct normalization (insurance)")
    print("=" * 60)

    import json as _json
    with open(Path(__file__).parent.parent / "config" / "domains.json") as f:
        domains_cfg = _json.load(f)

    pred_struct = {
        "policyholder": {
            "name": "Ashwin Shetty",
            "dob":  "12/08/2002",        # needs normalization
            "gender": "Male",
        },
        "policy": {
            "policy_number": "P1-093482",
            "policy_type":   "Term Life",
            "start_date":    "01 Jan 2023",  # needs normalization
        },
        "premium": {
            "amount":            "5,000",    # needs normalization
            "currency":          "Rs",       # needs normalization
            "payment_frequency": "annual",   # needs normalization
        },
    }

    normalized = normalize_struct(pred_struct, "insurance", domains_cfg)
    print(_json.dumps(normalized, indent=2))

    print("\n" + "=" * 60)
    print("PART 5 - Time and phone normalization")
    print("=" * 60)

    time_cases = [("09:15", "09:15"), ("9:15 AM", "09:15"), ("21:30", "21:30"), ("9.15pm", "21:15")]
    for raw, expected in time_cases:
        result = normalize_time(raw)
        status = "PASS" if result == expected else f"FAIL got {result}"
        print(f"  time  {raw:<12} → {str(result):<8} {status}")

    phone_cases = [("9876543210", "9876543210"), ("+919876543210", "9876543210"), ("98765", None)]
    for raw, expected in phone_cases:
        result = normalize_phone(raw)
        status = "PASS" if result == expected else f"FAIL got {result}"
        print(f"  phone {raw:<16} → {str(result):<12} {status}")
