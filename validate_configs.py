"""
validate_configs.py

Loads all 4 config files and runs cross-validation checks to catch
inconsistencies before any code downstream tries to use them.

Usage:
    python validate_configs.py

Expected output (if everything is correct):
    [OK] domains.json loaded - 3 domains found
    [OK] noise_weights.json loaded
    [OK] generation_config.json loaded
    [OK] eval_config.json loaded
    [OK] All noise tags in domain_applicability exist in weight tables
    [OK] All required_fields paths exist in domain schemas
    [OK] Task distribution sums to 1.0 for all difficulties
    [OK] Split ratios sum to 1.0
    [OK] Abbreviation expansions are consistent across domains and eval config

    All config checks passed.
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "config"

# ── Loaders ──────────────────────────────────────────────────────────────────

def load(filename: str) -> dict:
    path = CONFIG_DIR / filename
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[OK] {filename} loaded")
    return data


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_all_noise_tags(nw: dict) -> set[str]:
    """Return every noise tag name defined across all tier weight tables."""
    tags: set[str] = set()
    for tier_key in ["tier_1_weights", "tier_2_weights", "tier_3_weights", "tier_4_weights"]:
        tags.update(k for k in nw[tier_key] if not k.startswith("_"))
    return tags


def get_schema_leaf_paths(schema: dict, prefix: str = "") -> set:
    """
    Recursively walk a nested schema dict and return all dotted leaf paths.
    E.g. {"policyholder": {"name": {...}}} -> {"policyholder.name"}
    """
    paths = set()
    for key, value in schema.items():
        if key.startswith("_"):
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and "type" in value:
            # It's a leaf field definition
            paths.add(full_key)
        elif isinstance(value, dict):
            # It's a nested section - recurse
            paths.update(get_schema_leaf_paths(value, full_key))
    return paths


# ── Checks ────────────────────────────────────────────────────────────────────

def check_domains_loaded(domains: dict) -> None:
    domain_names = [k for k in domains if not k.startswith("_")]
    assert len(domain_names) == 3, f"Expected 3 domains, found {len(domain_names)}"
    print(f"[OK] domains.json loaded - {len(domain_names)} domains found: {domain_names}")


def check_noise_applicability(nw: dict) -> None:
    """Every tag in domain_applicability must exist in a tier weight table."""
    all_tags = get_all_noise_tags(nw)
    applicability_tags = {k for k in nw["domain_applicability"] if not k.startswith("_")}
    missing = applicability_tags - all_tags
    assert not missing, f"Tags in domain_applicability but not in weight tables: {missing}"
    print("[OK] All noise tags in domain_applicability exist in weight tables")


def check_required_fields(domains: dict) -> None:
    """Every dotted path in required_fields must resolve to a leaf in the schema."""
    for domain_name, domain_cfg in domains.items():
        if domain_name.startswith("_"):
            continue
        schema_paths = get_schema_leaf_paths(domain_cfg["schema"])
        for field_path in domain_cfg["required_fields"]:
            assert field_path in schema_paths, (
                f"[{domain_name}] required_field '{field_path}' "
                f"not found in schema. Valid paths: {sorted(schema_paths)}"
            )
        for field_path in domain_cfg["optional_fields"]:
            assert field_path in schema_paths, (
                f"[{domain_name}] optional_field '{field_path}' "
                f"not found in schema."
            )
    print("[OK] All required_fields and optional_fields paths exist in domain schemas")


def check_task_distributions(gen: dict) -> None:
    """Task distribution fractions must sum to 1.0 for each difficulty."""
    for difficulty, dist in gen["task_distribution"].items():
        if difficulty.startswith("_"):
            continue
        total = sum(v for k, v in dist.items() if not k.startswith("_"))
        assert abs(total - 1.0) < 1e-9, (
            f"Task distribution for '{difficulty}' sums to {total}, expected 1.0"
        )
    print("[OK] Task distribution sums to 1.0 for all difficulties")


def check_split_ratios(gen: dict) -> None:
    ratios = gen["split_ratios"]
    total = sum(ratios.values())
    assert abs(total - 1.0) < 1e-9, f"Split ratios sum to {total}, expected 1.0"
    print("[OK] Split ratios sum to 1.0")


def check_abbreviation_consistency(domains: dict, ev: dict) -> None:
    """
    The abbreviation_expansions in eval_config are also used by the noise
    engine's key_abbrev function. Check that every abbreviated key label
    found in domains key_label_variants has a corresponding expansion entry
    (if it's more than one word it's OK to omit - only short abbrevs matter).
    """
    eval_abbrevs = set(ev["normalization"]["abbreviation_expansions"].keys())
    # Collect all short (<=8 char, lowercase) label variants across all domains
    short_variants = set()
    for domain_name, domain_cfg in domains.items():
        if domain_name.startswith("_"):
            continue
        for _field, variants in domain_cfg["key_label_variants"].items():
            for v in variants:
                if len(v) <= 8:
                    short_variants.add(v.lower())

    # Warn (not fail) about missing expansions - some short labels are proper words
    missing = short_variants - eval_abbrevs
    genuine_abbrevs = {m for m in missing if not m.replace(" ", "").isalpha() or len(m) <= 4}
    if genuine_abbrevs:
        print(f"[WARN] Short label variants without abbreviation expansions: {genuine_abbrevs}")
        print("       These may be fine if they are proper words (e.g. 'type', 'name').")
    else:
        print("[OK] Abbreviation expansions are consistent across domains and eval config")


def check_noise_rates_coverage(nw: dict) -> None:
    """Every noise tag with weights should also have an entry in noise_application_rates."""
    all_tags = get_all_noise_tags(nw)
    rate_tags = {k for k in nw["noise_application_rates"] if not k.startswith("_")}
    missing = all_tags - rate_tags
    assert not missing, f"Noise tags missing from noise_application_rates: {missing}"
    print("[OK] All noise tags have application rate definitions")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    errors = []

    try:
        domains = load("domains.json")
        nw      = load("noise_weights.json")
        gen     = load("generation_config.json")
        ev      = load("eval_config.json")
    except FileNotFoundError as e:
        print(f"[FAIL] Could not load config file: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[FAIL] JSON parse error: {e}")
        sys.exit(1)

    checks: list[tuple[Callable[..., None], tuple]] = [
        (check_domains_loaded,         (domains,)),
        (check_noise_applicability,    (nw,)),
        (check_required_fields,        (domains,)),
        (check_task_distributions,     (gen,)),
        (check_split_ratios,           (gen,)),
        (check_abbreviation_consistency,(domains, ev)),
        (check_noise_rates_coverage,   (nw,)),
    ]

    for fn, args in checks:
        try:
            fn(*args)
        except AssertionError as e:
            print(f"[FAIL] {e}")
            errors.append(str(e))

    print()
    if errors:
        print(f"{len(errors)} check(s) failed. Fix the above before proceeding.")
        sys.exit(1)
    else:
        print("All config checks passed.")


if __name__ == "__main__":
    main()


# ── Usage example ─────────────────────────────────────────────────────────────
#
# From the project root (ocr_benchmark/):
#
#   python validate_configs.py
#
# You should see:
#   [OK] domains.json loaded - 3 domains found: ['receipts', 'insurance', 'hospital']
#   [OK] noise_weights.json loaded
#   [OK] generation_config.json loaded
#   [OK] eval_config.json loaded
#   [OK] All noise tags in domain_applicability exist in weight tables
#   [OK] All required_fields and optional_fields paths exist in domain schemas
#   [OK] Task distribution sums to 1.0 for all difficulties
#   [OK] Split ratios sum to 1.0
#   [OK] Abbreviation expansions are consistent across domains and eval config
#   [OK] All noise tags have application rate definitions
#
#   All config checks passed.
#
# Run this every time you modify a config file.
