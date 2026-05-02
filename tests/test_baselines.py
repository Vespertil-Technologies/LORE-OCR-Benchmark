"""
Tests for runners/baselines.py.

Locks in the contract that the two non-LLM baselines must satisfy:
    - always_null returns "{}" for every sample.
    - regex_rules returns valid JSON, never raises, places known field labels
      into the right (possibly nested) field paths from domains.json.
"""

import json

import pytest

from runners.baselines import call_baseline, list_baselines


class TestAlwaysNull:
    def test_returns_empty_object(self):
        sample = {"id": "s1", "domain": "receipts", "ocr_text": "anything goes here"}
        assert call_baseline(sample, {"name": "always_null"}) == "{}"

    def test_ignores_ocr_content(self):
        """Same output regardless of OCR text or domain."""
        for domain in ("receipts", "insurance", "hospital"):
            for text in ("", "Vendor: Foo\nTotal: 100", "garbage 12345"):
                out = call_baseline(
                    {"id": "s", "domain": domain, "ocr_text": text},
                    {"name": "always_null"},
                )
                assert out == "{}"


class TestRegexRulesReceipts:
    def test_extracts_known_receipt_labels(self):
        sample = {
            "id":      "s1",
            "domain":  "receipts",
            "ocr_text": "Vendor: SwiggyMart\nDate: 2024-03-14\nTotal: 368.16\nCurrency: INR",
        }
        out = json.loads(call_baseline(sample, {"name": "regex_rules"}))
        assert out.get("vendor_name") == "SwiggyMart"
        assert out.get("date")        == "2024-03-14"
        assert out.get("total_amount") == "368.16"
        assert out.get("currency")    == "INR"

    def test_unknown_labels_are_dropped(self):
        sample = {
            "id":      "s1",
            "domain":  "receipts",
            "ocr_text": "RandomLabel: hello\nVendor: Foo",
        }
        out = json.loads(call_baseline(sample, {"name": "regex_rules"}))
        assert "randomlabel" not in out
        assert out.get("vendor_name") == "Foo"


class TestRegexRulesNested:
    def test_inserts_into_nested_section_for_insurance(self):
        sample = {
            "id":      "s1",
            "domain":  "insurance",
            "ocr_text": "Policy Number: P1-093482\nName: Ashwin Shetty",
        }
        out = json.loads(call_baseline(sample, {"name": "regex_rules"}))
        assert isinstance(out, dict)
        # Policy number lives under policy.policy_number per domains.json
        assert out.get("policy", {}).get("policy_number") == "P1-093482"

    def test_returns_valid_json_for_hospital(self):
        sample = {
            "id":      "s1",
            "domain":  "hospital",
            "ocr_text": "Patient Name: Priya Iyer\nDate: 2024-04-02",
        }
        raw = call_baseline(sample, {"name": "regex_rules"})
        out = json.loads(raw)  # must always be parseable
        assert isinstance(out, dict)


class TestRegexRulesEdgeCases:
    def test_empty_ocr_returns_empty_dict(self):
        out = json.loads(call_baseline(
            {"id": "s", "domain": "receipts", "ocr_text": ""},
            {"name": "regex_rules"},
        ))
        assert out == {}

    def test_unknown_domain_returns_empty_dict(self):
        out = json.loads(call_baseline(
            {"id": "s", "domain": "made_up_domain", "ocr_text": "Vendor: X"},
            {"name": "regex_rules"},
        ))
        assert out == {}

    def test_first_match_wins(self):
        """If a label appears twice, the first occurrence is kept."""
        sample = {
            "id":      "s",
            "domain":  "receipts",
            "ocr_text": "Vendor: First\nVendor: Second",
        }
        out = json.loads(call_baseline(sample, {"name": "regex_rules"}))
        assert out["vendor_name"] == "First"


class TestDispatch:
    def test_unknown_baseline_raises(self):
        with pytest.raises(ValueError):
            call_baseline({"domain": "receipts", "ocr_text": ""}, {"name": "not_a_baseline"})

    def test_list_baselines_returns_known(self):
        names = list_baselines()
        assert "always_null" in names
        assert "regex_rules" in names
