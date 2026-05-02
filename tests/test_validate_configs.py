"""
Smoke tests for validate_configs.py.

The script's individual checks are covered by exercising the helpers
directly. The end-to-end run is covered by importing main() and
asserting it does not raise SystemExit with a non-zero code.
"""

import json

import pytest

import validate_configs as vc


def test_main_exits_clean_on_unmodified_repo():
    """validate_configs.main() should not call sys.exit with non-zero on a healthy repo."""
    try:
        vc.main()
    except SystemExit as exc:
        assert exc.code in (None, 0), f"validate_configs.main() exited non-zero: {exc.code}"


class TestHelpers:
    def test_get_schema_leaf_paths_handles_nested(self):
        schema = {
            "policyholder": {"name": {"type": "string"}, "dob": {"type": "date"}},
            "policy":       {"policy_number": {"type": "string"}},
        }
        paths = vc.get_schema_leaf_paths(schema)
        assert paths == {"policyholder.name", "policyholder.dob", "policy.policy_number"}

    def test_get_schema_leaf_paths_skips_underscore_keys(self):
        schema = {
            "_comment": "ignored",
            "name":     {"type": "string"},
        }
        assert vc.get_schema_leaf_paths(schema) == {"name"}


class TestCheckTaskDistributions:
    def test_distributions_summing_to_one_pass(self):
        gen = {"task_distribution": {
            "easy": {"extraction": 0.5, "normalization": 0.3, "correction": 0.2},
        }}
        vc.check_task_distributions(gen)

    def test_distribution_not_summing_to_one_fails(self):
        gen = {"task_distribution": {
            "easy": {"extraction": 0.5, "normalization": 0.3, "correction": 0.1},
        }}
        with pytest.raises(AssertionError):
            vc.check_task_distributions(gen)


class TestCheckSplitRatios:
    def test_ratios_summing_to_one_pass(self):
        gen = {"split_ratios": {"train": 0.6, "dev": 0.2, "test": 0.2}}
        vc.check_split_ratios(gen)

    def test_ratios_not_summing_to_one_fail(self):
        gen = {"split_ratios": {"train": 0.6, "dev": 0.2, "test": 0.1}}
        with pytest.raises(AssertionError):
            vc.check_split_ratios(gen)


def test_config_files_are_present_and_valid_json():
    """All four config files must exist and be parseable as JSON."""
    config_dir = vc.CONFIG_DIR
    for filename in ("domains.json", "noise_weights.json", "generation_config.json", "eval_config.json"):
        path = config_dir / filename
        assert path.exists(), f"Missing config file: {filename}"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
