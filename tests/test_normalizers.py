"""
Tests for parsers/normalizers.py.

Locks in the per-field normalization contracts for date, time, number,
phone, currency, and string. Indian-context conventions matter here:
DD/MM/YYYY is the assumed default for ambiguous numeric dates, lakh /
crore are recognized number suffixes, INR is the canonical currency.
"""

import pytest

from parsers.normalizers import (
    normalize_currency_code,
    normalize_date,
    normalize_number,
    normalize_payment_frequency,
    normalize_phone,
    normalize_string,
    normalize_time,
)


class TestNormalizeDate:
    @pytest.mark.parametrize("raw, expected", [
        ("2024-03-14",      "2024-03-14"),
        ("2024/03/14",      "2024-03-14"),
        ("14/03/2024",      "2024-03-14"),  # DD/MM/YYYY (Indian default)
        ("14-03-2024",      "2024-03-14"),
        ("14 Mar 2024",     "2024-03-14"),
        ("14 March 2024",   "2024-03-14"),
        ("Mar 14, 2024",    "2024-03-14"),
        ("14032024",        "2024-03-14"),  # compact DDMMYYYY
    ])
    def test_known_formats_become_iso(self, raw, expected):
        assert normalize_date(raw) == expected

    @pytest.mark.parametrize("raw", [
        "12/0B/2002",   # OCR-corrupted month
        "not a date",
        "32/01/2024",   # invalid day
        "01/13/2024",   # invalid month
    ])
    def test_unparseable_returns_none(self, raw):
        assert normalize_date(raw) is None

    def test_none_input_returns_none(self):
        assert normalize_date(None) is None


class TestNormalizeTime:
    @pytest.mark.parametrize("raw, expected", [
        ("09:15",    "09:15"),
        ("9:15 AM",  "09:15"),
        ("21:30",    "21:30"),
        ("9.15pm",   "21:15"),
        ("12:00 AM", "00:00"),
        ("12:00 PM", "12:00"),
    ])
    def test_known_formats(self, raw, expected):
        assert normalize_time(raw) == expected

    def test_invalid_returns_none(self):
        assert normalize_time("not a time") is None
        assert normalize_time(None) is None


class TestNormalizeNumber:
    @pytest.mark.parametrize("raw, expected", [
        ("5,000",       5000.0),
        ("5,000 INR",   5000.0),
        ("₹ 5000", 5000.0),   # rupee symbol
        ("368.16",      368.16),
        ("2 lakh",      200000.0),
        ("1.5 crore",   15000000.0),
        (5000,          5000.0),
        (5000.5,        5000.5),
    ])
    def test_known_inputs(self, raw, expected):
        assert normalize_number(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", [
        "S,OOO",   # OCR letters substituted for digits
        "abc",
        "",
        ".",
    ])
    def test_garbage_returns_none(self, raw):
        assert normalize_number(raw) is None

    def test_none_returns_none(self):
        assert normalize_number(None) is None


class TestNormalizeCurrencyCode:
    @pytest.mark.parametrize("raw, expected", [
        ("INR",       "INR"),
        ("inr",       "INR"),
        ("Rs",        "INR"),
        ("rupees",    "INR"),
        ("₹",    "INR"),
        ("USD",       "USD"),
        ("dollars",   "USD"),
        ("EUR",       "EUR"),
        ("euros",     "EUR"),
        ("XYZ",       "XYZ"),    # unknown but ISO-shaped passes through
    ])
    def test_known_codes(self, raw, expected):
        assert normalize_currency_code(raw) == expected

    def test_unparseable_returns_none(self):
        assert normalize_currency_code("1RN") is None

    def test_none_returns_none(self):
        assert normalize_currency_code(None) is None


class TestNormalizePhone:
    @pytest.mark.parametrize("raw, expected", [
        ("9876543210",       "9876543210"),
        ("+919876543210",    "9876543210"),
        ("+91 98765 43210",  "9876543210"),
        ("98765-43210",      "9876543210"),
    ])
    def test_known_phone_inputs(self, raw, expected):
        assert normalize_phone(raw) == expected

    def test_too_short_returns_none(self):
        assert normalize_phone("98765") is None

    def test_none_returns_none(self):
        assert normalize_phone(None) is None


class TestNormalizeString:
    def test_lowercases_and_strips(self):
        assert normalize_string("  Hello World  ") == "hello world"

    def test_collapses_internal_whitespace(self):
        assert normalize_string("hello    world") == "hello world"

    def test_empty_returns_none(self):
        assert normalize_string("") is None
        assert normalize_string("   ") is None

    def test_none_returns_none(self):
        assert normalize_string(None) is None


class TestNormalizePaymentFrequency:
    @pytest.mark.parametrize("raw, expected", [
        ("annual",       "yearly"),
        ("annually",     "yearly"),
        ("yearly",       "yearly"),
        ("p.a.",         "yearly"),
        ("monthly",      "monthly"),
        ("per month",    "monthly"),
        ("quarterly",    "quarterly"),
        ("half-yearly",  "half-yearly"),
        ("semi-annual",  "half-yearly"),
    ])
    def test_known_inputs(self, raw, expected):
        assert normalize_payment_frequency(raw) == expected

    def test_unknown_passes_through_lowercased(self):
        assert normalize_payment_frequency("WHATEVER") == "whatever"

    def test_none_returns_none(self):
        assert normalize_payment_frequency(None) is None
