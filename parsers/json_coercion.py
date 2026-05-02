"""
parsers/json_coercion.py

Robustly extracts a valid Python dict from raw LLM output strings.
Models often wrap JSON in markdown, add commentary, or produce
slightly malformed output. This module handles all of it gracefully.

Attempt order (stops at first success):
    1. Direct json.loads() on the full string
    2. Strip markdown code fences (```json ... ```) then json.loads()
    3. Extract first {...} block and json.loads()
    4. Strip common prose prefixes, retry json.loads()
    5. Regex-based key-value extraction (partial parse)
    6. Return ({}, "failure")

Returns:
    (parsed_dict, parse_status)
    parse_status is one of: "success" | "partial" | "failure"

Never raises - always returns something.
"""

import json
import re
from typing import Any

# ══════════════════════════════════════════════════════════════════════════════
# ATTEMPT 1 - Direct parse
# ══════════════════════════════════════════════════════════════════════════════

def _try_direct(text: str) -> dict | None:
    """Try json.loads() on the raw string."""
    try:
        result = json.loads(text.strip())
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ATTEMPT 2 - Strip markdown code fences
# ══════════════════════════════════════════════════════════════════════════════

_CODE_FENCE_PATTERN = re.compile(
    r"```(?:json)?\s*([\s\S]*?)\s*```",
    re.IGNORECASE
)

def _try_strip_fences(text: str) -> dict | None:
    """Extract content inside ```json ... ``` or ``` ... ``` blocks."""
    matches = _CODE_FENCE_PATTERN.findall(text)
    for block in matches:
        try:
            result = json.loads(block.strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            continue
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ATTEMPT 3 - Extract first {...} block
# ══════════════════════════════════════════════════════════════════════════════

def _find_json_object(text: str) -> str | None:
    """
    Find the outermost {...} block in a string by counting braces.
    Handles nested objects correctly.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]

    return None


def _try_extract_block(text: str) -> dict | None:
    """Extract the first JSON object block from anywhere in the text."""
    block = _find_json_object(text)
    if block:
        try:
            result = json.loads(block)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ATTEMPT 4 - Strip prose prefixes
# ══════════════════════════════════════════════════════════════════════════════

_PROSE_PREFIXES = [
    r"^here(?:'s| is) the (?:json|extracted|structured).*?:\s*",
    r"^(?:sure|certainly|of course)[,!]?\s*",
    r"^based on the ocr text.*?:\s*",
    r"^the extracted (?:data|fields|information).*?:\s*",
    r"^(?:json|output|result)[:\s]+",
]
_PROSE_PREFIX_RE = re.compile(
    "|".join(_PROSE_PREFIXES),
    re.IGNORECASE | re.DOTALL
)

def _try_strip_prefix(text: str) -> dict | None:
    """Remove common prose preambles then retry parsing."""
    cleaned = _PROSE_PREFIX_RE.sub("", text.strip(), count=1).strip()
    if cleaned != text.strip():
        result = _try_direct(cleaned) or _try_extract_block(cleaned)
        return result
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ATTEMPT 5 - Partial parse via regex key-value extraction
# ══════════════════════════════════════════════════════════════════════════════

_KV_PATTERN = re.compile(
    r'"([\w\.]+)"\s*:\s*'           # key in double quotes
    r'('
    r'"(?:[^"\\]|\\.)*"'            # string value
    r'|true|false|null'             # boolean / null
    r'|-?\d+(?:\.\d+)?'             # number
    r')',
    re.DOTALL
)

def _try_regex_extract(text: str) -> dict | None:
    """
    Last resort: extract individual key-value pairs using regex.
    Only captures flat key-value pairs - nested structure is lost.
    Returns None if fewer than 2 pairs found (not worth calling partial).
    """
    matches = _KV_PATTERN.findall(text)
    if len(matches) < 2:
        return None

    result: dict[str, Any] = {}
    for key, raw_value in matches:
        try:
            value = json.loads(raw_value)
        except (json.JSONDecodeError, ValueError):
            value = raw_value.strip('"')
        # Flatten dotted keys into nested dicts
        parts = key.split(".")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    return result if result else None


# ══════════════════════════════════════════════════════════════════════════════
# FIXUP - Clean up common JSON issues before re-attempting
# ══════════════════════════════════════════════════════════════════════════════

def _fix_common_issues(text: str) -> str:
    """
    Apply lightweight fixes for common LLM JSON formatting mistakes.
    Applied before each attempt so the attempt functions stay pure.
    """
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Replace single quotes with double quotes (only outside existing double-quote strings)
    # Simple heuristic - only safe for flat structures
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    # Normalize None → null (Python repr leakage)
    text = re.sub(r"\bNone\b", "null", text)
    # Normalize True/False → true/false
    text = re.sub(r"\bTrue\b",  "true",  text)
    text = re.sub(r"\bFalse\b", "false", text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

ParseStatus = str  # "success" | "partial" | "failure"

def coerce(raw_text: str) -> tuple[dict, ParseStatus]:
    """
    Extract a dict from raw LLM output. Never raises.

    Args:
        raw_text: The raw string returned by the model.

    Returns:
        (parsed_dict, parse_status)
        - parsed_dict:  The extracted dict, or {} on total failure.
        - parse_status: "success" | "partial" | "failure"
    """
    if not raw_text or not raw_text.strip():
        return {}, "failure"

    fixed = _fix_common_issues(raw_text)

    # Attempt 1 - Direct
    result = _try_direct(fixed)
    if result is not None:
        return result, "success"

    # Attempt 2 - Strip markdown fences
    result = _try_strip_fences(fixed)
    if result is not None:
        return result, "success"

    # Attempt 3 - Extract {...} block
    result = _try_extract_block(fixed)
    if result is not None:
        return result, "success"

    # Attempt 4 - Strip prose prefix
    result = _try_strip_prefix(fixed)
    if result is not None:
        return result, "success"

    # Attempt 5 - Partial regex extraction
    result = _try_regex_extract(fixed)
    if result is not None:
        return result, "partial"

    return {}, "failure"


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    test_cases = [
        # (label, raw_text, expected_status)

        ("Clean JSON",
         '{"vendor_name": "SwiggyMart", "date": "2024-03-14", "total_amount": 368.16, "currency": "INR"}',
         "success"),

        ("Markdown fenced JSON",
         '```json\n{"vendor_name": "SwiggyMart", "total_amount": 368.16}\n```',
         "success"),

        ("Prose prefix + JSON",
         'Here is the extracted JSON:\n{"vendor_name": "SwiggyMart", "total_amount": 368.16}',
         "success"),

        ("JSON buried in explanation",
         'Based on the OCR text, I found the following fields:\n{"vendor_name": "SwiggyMart"}\nNote: some fields were unclear.',
         "success"),

        ("Nested JSON (insurance)",
         '{"policyholder": {"name": "Ashwin Shetty", "dob": "2002-08-12"}, "policy": {"policy_number": "P1-093482"}, "premium": {"amount": 5000, "currency": "INR"}}',
         "success"),

        ("Trailing comma (common LLM mistake)",
         '{"vendor_name": "SwiggyMart", "total_amount": 368.16,}',
         "success"),

        ("Python None/True/False leakage",
         '{"vendor_name": "SwiggyMart", "time": None, "success": True}',
         "success"),

        ("Partial - only key-value pairs, no braces",
         '"vendor_name": "SwiggyMart", "total_amount": 368.16, "currency": "INR"',
         "partial"),

        ("Complete failure - pure prose",
         'I was unable to extract any structured data from this OCR text.',
         "failure"),

        ("Empty string",
         "",
         "failure"),
    ]

    print(f"{'Label':<45} {'Status':<10} {'Keys found'}")
    print("-" * 75)

    all_passed = True
    for label, raw, expected_status in test_cases:
        result, status = coerce(raw)
        match = "PASS" if status == expected_status else "FAIL MISMATCH"
        if status != expected_status:
            all_passed = False
        keys = list(result.keys()) if result else []
        print(f"{label:<45} {status:<10} {keys}  {match}")

    print()
    print("All tests passed" if all_passed else "Some tests FAILED")
