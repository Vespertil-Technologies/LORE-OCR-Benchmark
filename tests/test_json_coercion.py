"""
Tests for parsers/json_coercion.py.

Covers each step of the 5-attempt cascade plus the failure path.
LLMs return badly-formatted JSON in many ways. These tests pin the
behaviour for each known failure mode.
"""

import pytest

from parsers.json_coercion import coerce


class TestEmptyAndWhitespace:
    @pytest.mark.parametrize("raw", ["", "   ", "\n\n", None])
    def test_empty_inputs_return_failure(self, raw):
        result, status = coerce(raw if raw is not None else "")
        assert result == {}
        assert status == "failure"


class TestAttempt1DirectParse:
    def test_clean_json(self):
        raw = '{"vendor_name": "SwiggyMart", "total_amount": 368.16}'
        result, status = coerce(raw)
        assert status == "success"
        assert result["vendor_name"] == "SwiggyMart"
        assert result["total_amount"] == 368.16

    def test_clean_json_with_whitespace(self):
        raw = '   {"a": 1}   '
        result, status = coerce(raw)
        assert status == "success"
        assert result == {"a": 1}

    def test_nested_json(self):
        raw = '{"policy": {"policy_number": "P1-093482"}}'
        result, status = coerce(raw)
        assert status == "success"
        assert result["policy"]["policy_number"] == "P1-093482"


class TestAttempt2MarkdownFences:
    def test_json_fence(self):
        raw = '```json\n{"vendor_name": "SwiggyMart"}\n```'
        result, status = coerce(raw)
        assert status == "success"
        assert result["vendor_name"] == "SwiggyMart"

    def test_plain_fence(self):
        raw = '```\n{"a": 1}\n```'
        result, status = coerce(raw)
        assert status == "success"
        assert result == {"a": 1}

    def test_fence_with_uppercase_language(self):
        raw = '```JSON\n{"a": 1}\n```'
        result, status = coerce(raw)
        assert status == "success"


class TestAttempt3ExtractBlock:
    def test_json_buried_in_explanation(self):
        raw = (
            "Based on the OCR text, I found the following fields:\n"
            '{"vendor_name": "SwiggyMart"}\n'
            "Note: some fields were unclear."
        )
        result, status = coerce(raw)
        assert status == "success"
        assert result["vendor_name"] == "SwiggyMart"

    def test_json_with_braces_in_strings(self):
        """The brace-counter must respect string boundaries."""
        raw = '{"note": "contains } brace", "valid": true}'
        result, status = coerce(raw)
        assert status == "success"
        assert result["valid"] is True

    def test_picks_first_complete_object(self):
        raw = 'prefix {"a": 1} middle {"b": 2} suffix'
        result, status = coerce(raw)
        assert status == "success"
        assert result == {"a": 1}


class TestAttempt4ProsePrefix:
    def test_here_is_the_json_prefix(self):
        raw = 'Here is the extracted JSON:\n{"vendor_name": "SwiggyMart"}'
        result, status = coerce(raw)
        assert status == "success"
        assert result["vendor_name"] == "SwiggyMart"

    def test_sure_prefix(self):
        raw = 'Sure! {"a": 1}'
        result, status = coerce(raw)
        assert status == "success"


class TestFixupCommonIssues:
    def test_trailing_comma_is_repaired(self):
        raw = '{"a": 1, "b": 2,}'
        result, status = coerce(raw)
        assert status == "success"
        assert result == {"a": 1, "b": 2}

    def test_python_none_is_normalized(self):
        raw = '{"a": None, "b": 1}'
        result, status = coerce(raw)
        assert status == "success"
        assert result["a"] is None
        assert result["b"] == 1

    def test_python_true_false_is_normalized(self):
        raw = '{"a": True, "b": False}'
        result, status = coerce(raw)
        assert status == "success"
        assert result["a"] is True
        assert result["b"] is False


class TestAttempt5RegexPartial:
    def test_no_braces_partial_extraction(self):
        raw = '"vendor_name": "SwiggyMart", "total_amount": 368.16'
        result, status = coerce(raw)
        assert status == "partial"
        assert result["vendor_name"] == "SwiggyMart"
        assert result["total_amount"] == 368.16

    def test_dotted_keys_become_nested(self):
        raw = '"policy.policy_number": "P1-093482", "premium.amount": 5000.0'
        result, status = coerce(raw)
        assert status == "partial"
        assert result["policy"]["policy_number"] == "P1-093482"
        assert result["premium"]["amount"] == 5000.0


class TestTotalFailure:
    def test_pure_prose_returns_failure(self):
        raw = "I cannot extract any fields from this text."
        result, status = coerce(raw)
        assert status == "failure"
        assert result == {}

    def test_garbage_returns_failure(self):
        raw = "<<>><><><>"
        result, status = coerce(raw)
        assert status == "failure"


class TestNeverRaises:
    """Public API contract: coerce() must never raise."""

    @pytest.mark.parametrize("raw", [
        "",
        "{",
        "}",
        "{{{}}}",
        '{"key": "value with \\"escaped quote"}',
        "definitely not json",
        '{"unbalanced": "string',
        "[1, 2, 3]",  # array, not dict
        '{"a":}',
    ])
    def test_does_not_raise(self, raw):
        # Should always return a 2-tuple, never propagate an exception.
        result, status = coerce(raw)
        assert isinstance(result, dict)
        assert status in ("success", "partial", "failure")
