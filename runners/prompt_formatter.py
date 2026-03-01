"""
runners/prompt_formatter.py

Builds the exact prompt string sent to the LLM for a given sample.

Responsibilities:
    - Read the frozen domain template
    - Inject {OCR_TEXT}, {SCHEMA}, {TASK_INSTRUCTION} placeholders
    - Build a clean JSON schema string from domains.json (what the model sees)
    - Return the prompt string + a hash of it (for run logging & reproducibility)
    - Never modify template files

Does NOT:
    - Call any LLM
    - Parse model output
    - Know about evaluation metrics
"""

import hashlib
import json
from pathlib import Path
from typing import Any

# ── Config ─────────────────────────────────────────────────────────────────────

_BASE_DIR     = Path(__file__).parent.parent
_CONFIG_DIR   = _BASE_DIR / "config"
_TEMPLATE_DIR = _BASE_DIR / "prompts" / "templates"

def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

DOMAINS           = _load_json(_CONFIG_DIR / "domains.json")
TASK_INSTRUCTIONS = _load_json(_TEMPLATE_DIR / "task_instructions.json")

# Required placeholders every template must contain
_REQUIRED_PLACEHOLDERS = {"{OCR_TEXT}", "{SCHEMA}", "{TASK_INSTRUCTION}"}


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA BUILDER
# Converts the domains.json schema into a clean JSON string the model can read
# ══════════════════════════════════════════════════════════════════════════════

def _build_schema_string(domain: str) -> str:
    """
    Build a human-readable JSON schema string from domains.json.
    Shows the model exactly what fields to return and their types.

    For required fields: shows the field with its type.
    For optional fields: shows the field with its type + "(optional — omit if absent)".
    Null values are shown as null.
    """
    domain_cfg    = DOMAINS[domain]
    schema_def    = domain_cfg["schema"]
    required      = set(domain_cfg["required_fields"])
    optional      = set(domain_cfg["optional_fields"])

    def _render_node(node: dict, prefix: str = "") -> dict:
        """Recursively render a schema node into a display dict."""
        result = {}
        for key, value in node.items():
            if key.startswith("_"):
                continue
            field_path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict) and "type" in value:
                # Leaf field
                field_type = value["type"]
                fmt        = value.get("format", "")
                fmt_note   = f" ({fmt})" if fmt else ""

                if field_path in required:
                    result[key] = f"{field_type}{fmt_note} | null"
                else:
                    result[key] = f"{field_type}{fmt_note} | omit if absent"
            elif isinstance(value, dict):
                # Nested section — recurse
                result[key] = _render_node(value, prefix=field_path)

        return result

    rendered = _render_node(schema_def)
    return json.dumps(rendered, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _load_template(domain: str) -> str:
    """Load the frozen prompt template for a domain."""
    template_path = _TEMPLATE_DIR / f"{domain}.txt"
    if not template_path.exists():
        raise FileNotFoundError(
            f"No prompt template found for domain '{domain}' at {template_path}"
        )
    return template_path.read_text(encoding="utf-8")


def _validate_template(template: str, domain: str) -> None:
    """Ensure all required placeholders are present in the template."""
    missing = [p for p in _REQUIRED_PLACEHOLDERS if p not in template]
    if missing:
        raise ValueError(
            f"Template for '{domain}' is missing required placeholders: {missing}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT HASH
# ══════════════════════════════════════════════════════════════════════════════

def _hash_prompt(prompt: str) -> str:
    """Return an 8-character SHA256 hash of the prompt string."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(sample: dict) -> tuple[str, str]:
    """
    Build the complete prompt string for a sample.

    Args:
        sample: A sample dict as produced by sample_builder.py.
                Must contain 'domain', 'task', and 'ocr_text'.

    Returns:
        (prompt_string, prompt_hash) — the full prompt and its 8-char hash.

    Raises:
        FileNotFoundError: If the domain template file does not exist.
        ValueError: If the template is missing required placeholders,
                    or if the task is not recognized.
    """
    domain   = sample["domain"]
    task     = sample["task"]
    ocr_text = sample["ocr_text"]

    # Validate task
    if task not in TASK_INSTRUCTIONS:
        raise ValueError(
            f"Unknown task '{task}'. Must be one of: {list(TASK_INSTRUCTIONS.keys())}"
        )

    # Load + validate template
    template = _load_template(domain)
    _validate_template(template, domain)

    # Build schema string
    schema_str = _build_schema_string(domain)

    # Get task instruction
    task_instruction = TASK_INSTRUCTIONS[task]

    # Inject all placeholders
    prompt = (
        template
        .replace("{TASK_INSTRUCTION}", task_instruction)
        .replace("{SCHEMA}", schema_str)
        .replace("{OCR_TEXT}", ocr_text)
    )

    return prompt, _hash_prompt(prompt)


def build_prompt_from_parts(
    domain: str,
    task: str,
    ocr_text: str,
) -> tuple[str, str]:
    """
    Build a prompt directly from parts — useful for testing or ablation runs
    where you don't have a full sample dict.

    Returns:
        (prompt_string, prompt_hash)
    """
    sample = {"domain": domain, "task": task, "ocr_text": ocr_text}
    return build_prompt(sample)


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_BASE_DIR))
    from dataset.loader import load_samples

    DATA_DIR = _BASE_DIR / "data"

    print("=" * 60)
    print("PART 1 — Schema strings (what the model sees per domain)")
    print("=" * 60)

    for domain in ["receipts", "insurance", "hospital"]:
        print(f"\n── {domain.upper()} ──")
        print(_build_schema_string(domain))

    print("\n" + "=" * 60)
    print("PART 2 — Full prompt for one sample (insurance / hard)")
    print("=" * 60)

    samples = load_samples(DATA_DIR, domain="insurance", difficulty="hard", split="test")
    if samples:
        sample = samples[0]
        prompt, prompt_hash = build_prompt(sample)
        print(f"Sample ID   : {sample['id']}")
        print(f"Task        : {sample['task']}")
        print(f"Prompt hash : {prompt_hash}")
        print(f"\n{'-' * 40}")
        print(prompt)
    else:
        print("No samples found — run sample_builder.py first.")

    print("\n" + "=" * 60)
    print("PART 3 — Same OCR text, all 5 task variants")
    print("=" * 60)

    if samples:
        ocr_text = samples[0]["ocr_text"]
        for task in ["extraction", "normalization", "correction", "hallucination", "schema"]:
            _, h = build_prompt_from_parts("insurance", task, ocr_text)
            print(f"  task={task:<14}  prompt_hash={h}")