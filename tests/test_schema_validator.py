"""
Tests for evaluator/schema_validator.py.

Verifies the public validate_schema() entrypoint correctly classifies
common prediction shapes against the frozen domain schemas.
"""

import pytest

from evaluator.schema_validator import validate_schema

VALID_INSURANCE = {
    "policyholder": {"name": "Ashwin Shetty", "dob": "2002-08-12"},
    "policy":       {"policy_number": "P1-093482"},
    "premium":      {"amount": 5000.0, "currency": "INR"},
}

VALID_RECEIPT = {
    "vendor_name":  "SwiggyMart",
    "date":         "2024-03-14",
    "total_amount": 368.16,
    "currency":     "INR",
}


class TestParseFailureShortCircuit:
    def test_parse_failure_is_invalid_immediately(self):
        result = validate_schema({}, "insurance", parse_status="failure")
        assert result["schema_valid"] is False
        assert result["required_fields_present"] is False
        assert result["n_violations"] == 1


class TestNonDictPrediction:
    def test_string_prediction_is_invalid(self):
        result = validate_schema("not a dict", "insurance", parse_status="success")  # type: ignore[arg-type]
        assert result["schema_valid"] is False

    def test_list_prediction_is_invalid(self):
        result = validate_schema(["item"], "insurance", parse_status="success")  # type: ignore[arg-type]
        assert result["schema_valid"] is False


class TestValidPredictions:
    def test_valid_nested_insurance(self):
        result = validate_schema(VALID_INSURANCE, "insurance", parse_status="success")
        assert result["schema_valid"] is True
        assert result["n_violations"] == 0
        assert result["required_fields_present"] is True

    def test_valid_flat_receipt(self):
        result = validate_schema(VALID_RECEIPT, "receipts", parse_status="success")
        assert result["schema_valid"] is True
        assert result["required_fields_present"] is True


class TestStructuralViolations:
    def test_flattened_output_for_nested_domain_is_invalid(self):
        flat = {
            "policyholder.name":    "Ashwin Shetty",
            "policy.policy_number": "P1-093482",
            "premium.amount":       5000.0,
            "premium.currency":     "INR",
        }
        result = validate_schema(flat, "insurance", parse_status="success")
        assert result["schema_valid"] is False

    def test_nested_section_as_scalar_is_invalid(self):
        bad = dict(VALID_INSURANCE)
        bad["premium"] = "5000 INR"
        result = validate_schema(bad, "insurance", parse_status="success")
        assert result["schema_valid"] is False
        assert any("premium" in v.lower() for v in result["violations"])

    def test_unknown_top_level_key_is_violation(self):
        bad = dict(VALID_INSURANCE)
        bad["nominee"] = "N/A"
        result = validate_schema(bad, "insurance", parse_status="success")
        assert result["schema_valid"] is False
        assert any("nominee" in v.lower() for v in result["violations"])


class TestRequiredFields:
    def test_missing_required_field_marks_required_present_false(self):
        bad = {
            "policyholder": {"name": "Ashwin Shetty", "dob": "2002-08-12"},
            "premium":      {"amount": 5000.0, "currency": "INR"},
        }
        result = validate_schema(bad, "insurance", parse_status="success")
        assert result["required_fields_present"] is False
        assert result["schema_valid"] is False


class TestTypeViolations:
    def test_number_field_with_string_value_is_violation(self):
        bad = {
            "policyholder": {"name": "Ashwin Shetty", "dob": "2002-08-12"},
            "policy":       {"policy_number": "P1-093482"},
            "premium":      {"amount": "five thousand", "currency": "INR"},
        }
        result = validate_schema(bad, "insurance", parse_status="success")
        assert result["schema_valid"] is False
        assert any("amount" in v.lower() for v in result["violations"])

    def test_null_leaf_value_is_acceptable(self):
        """Required fields can be null as long as the key is present."""
        ok = {
            "policyholder": {"name": "Ashwin Shetty", "dob": "2002-08-12"},
            "policy":       {"policy_number": "P1-093482"},
            "premium":      {"amount": None, "currency": "INR"},
        }
        result = validate_schema(ok, "insurance", parse_status="success")
        # Null leaf does not break type check (None is always acceptable).
        # If amount is required, presence is what matters, not non-null.
        assert "amount" not in [v.split("'")[1] for v in result["violations"] if "'" in v]


@pytest.mark.parametrize("domain", ["receipts", "insurance", "hospital"])
def test_unknown_domain_does_not_crash(domain):
    """Each declared domain accepts a well-formed empty dict without raising."""
    result = validate_schema({}, domain, parse_status="success")
    assert "schema_valid" in result
