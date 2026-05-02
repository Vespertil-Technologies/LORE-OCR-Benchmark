"""
runners/baselines.py

Hardcoded, non-LLM baselines that anchor the lower end of the leaderboard.
Every model the benchmark evaluates should beat these. If it does not,
something is wrong with the prompt, the parser, or the model.

Baselines defined here:
    - always_null  : returns "{}" for every sample. The minimum-effort floor.
    - regex_rules  : extracts key:value pairs from the OCR text using the
                     domain's known label variants. No LLM, no normalization
                     beyond regex matching, no schema validation.

Both baselines run instantly (no API calls, no network). They are dispatched
from runners/llm_adapter.call() when the model_cfg's backend is "hardcoded".
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

_CONFIG_DIR = Path(__file__).parent.parent / "config"

with open(_CONFIG_DIR / "domains.json", encoding="utf-8") as f:
    _DOMAINS: dict = json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE: always_null
# ══════════════════════════════════════════════════════════════════════════════

def _always_null(sample: dict) -> str:
    """Return an empty JSON object regardless of input.

    Schema validator will mark every required field as missing and the
    parser will report 'success' with an empty dict, giving a clear floor
    against which every other model must improve.
    """
    return "{}"


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE: regex_rules
# ══════════════════════════════════════════════════════════════════════════════

# Captures: <label> <separator> <value>
# Allows colon, equals, dash, or pipe as the key/value separator.
_KEY_VALUE_LINE = re.compile(
    r"^\s*([A-Za-z][\w\s./()'-]*?)\s*[:=|]\s*(.+?)\s*$"
)


def _build_label_index(domain: str) -> dict[str, str]:
    """Return a dict mapping lowercased label variant to dotted field path."""
    domain_cfg = _DOMAINS[domain]
    label_to_path: dict[str, str] = {}
    for field_path, variants in domain_cfg["key_label_variants"].items():
        for variant in variants:
            label_to_path[variant.strip().lower()] = field_path
    return label_to_path


def _set_nested(target: dict, dotted_path: str, value: str) -> None:
    """Place value at target[a][b][c] for path 'a.b.c'."""
    parts = dotted_path.split(".")
    node = target
    for p in parts[:-1]:
        existing = node.get(p)
        if not isinstance(existing, dict):
            existing = {}
            node[p] = existing
        node = existing
    node[parts[-1]] = value


def _regex_rules(sample: dict) -> str:
    """Extract fields by matching OCR labels against known per-domain label variants.

    Uses domains.json key_label_variants as the source of truth for which
    raw text labels map to which canonical field paths. The first matching
    line wins for each field. No type coercion, no normalization. Schema
    structure is preserved for nested domains.
    """
    domain   = sample.get("domain", "")
    ocr_text = sample.get("ocr_text", "")
    if not domain or domain not in _DOMAINS:
        return "{}"

    label_to_path = _build_label_index(domain)
    result: dict = {}
    seen: set[str] = set()

    for line in ocr_text.split("\n"):
        m = _KEY_VALUE_LINE.match(line)
        if not m:
            continue
        label = m.group(1).strip().lower()
        value = m.group(2).strip()
        if label not in label_to_path:
            continue
        field_path = label_to_path[label]
        if field_path in seen:
            continue
        _set_nested(result, field_path, value)
        seen.add(field_path)

    return json.dumps(result, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

_BASELINE_DISPATCH: dict[str, Callable[[dict], str]] = {
    "always_null":  _always_null,
    "regex_rules":  _regex_rules,
}


def call_baseline(sample: dict, model_cfg: dict) -> str:
    """Run the configured hardcoded baseline against one sample.

    Args:
        sample:    Sample dict with at least 'domain' and 'ocr_text' keys.
        model_cfg: Model configuration. The model name is used to pick the baseline.

    Returns:
        The model output string. Always valid JSON.
    """
    name = model_cfg.get("name", "")
    fn = _BASELINE_DISPATCH.get(name)
    if fn is None:
        raise ValueError(
            f"Unknown hardcoded baseline '{name}'. "
            f"Known baselines: {sorted(_BASELINE_DISPATCH)}"
        )
    return fn(sample)


def list_baselines() -> list[str]:
    """Return the names of all registered baselines."""
    return sorted(_BASELINE_DISPATCH)
