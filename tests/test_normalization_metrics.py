"""
Golden-file tests for evaluator/normalization_metrics.py.

These pin the math of Levenshtein distance, normalized edit distance,
and per-field exact-match metrics. Test inputs use textbook examples
and edge cases so failures clearly indicate which property regressed.
"""

import pytest

from evaluator.normalization_metrics import (
    compute_field_normalization,
    levenshtein,
    normalized_edit_distance,
)


class TestLevenshtein:
    @pytest.mark.parametrize("a, b, expected", [
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2),
        ("intention", "execution", 5),
        ("", "abc", 3),
        ("abc", "", 3),
        ("hello", "hello", 0),
        ("a", "b", 1),
        ("abc", "abd", 1),
    ])
    def test_known_pairs(self, a, b, expected):
        assert levenshtein(a, b) == expected

    def test_symmetry(self):
        """Levenshtein distance is symmetric: d(a, b) == d(b, a)."""
        for a, b in [("kitten", "sitting"), ("foo", "bar"), ("abc", "xyzabc")]:
            assert levenshtein(a, b) == levenshtein(b, a)

    def test_triangle_inequality(self):
        """d(a, c) <= d(a, b) + d(b, c)."""
        a, b, c = "kitten", "sitting", "kettle"
        assert levenshtein(a, c) <= levenshtein(a, b) + levenshtein(b, c)

    def test_coerces_non_string_inputs(self):
        """levenshtein casts inputs to str so int/float comparisons do not crash."""
        assert levenshtein(123, "123") == 0  # type: ignore[arg-type]
        assert levenshtein(5000, "5000.0") > 0  # type: ignore[arg-type]


class TestNormalizedEditDistance:
    def test_identical_is_zero(self):
        assert normalized_edit_distance("hello", "hello") == 0.0

    def test_empty_strings_is_zero(self):
        assert normalized_edit_distance("", "") == 0.0

    def test_completely_different_is_one(self):
        """Two non-overlapping equal-length strings have NED 1.0."""
        assert normalized_edit_distance("aaa", "bbb") == 1.0

    def test_one_char_off_in_short_string(self):
        # 1 substitution / 5 chars = 0.2
        assert normalized_edit_distance("ashwin", "ashwln") == pytest.approx(1 / 6)

    @pytest.mark.parametrize("a, b", [
        ("kitten", "sitting"),
        ("hello world", "hello"),
        ("abc", "xyz"),
    ])
    def test_range_zero_to_one(self, a, b):
        ned = normalized_edit_distance(a, b)
        assert 0.0 <= ned <= 1.0

    def test_normalizes_by_longer_string(self):
        """NED divides by max(len(a), len(b))."""
        # "abc" vs "abcdefghij": 7 inserts, divided by 10 = 0.7
        assert normalized_edit_distance("abc", "abcdefghij") == pytest.approx(0.7)


class TestComputeFieldNormalization:
    def test_string_exact_match(self):
        result = compute_field_normalization("hello", "hello", "string")
        assert result["exact_match"] is True
        assert result["edit_distance"] == 0
        assert result["ned"] == 0.0

    def test_string_off_by_one(self):
        result = compute_field_normalization("ashwln shetty", "ashwin shetty", "string")
        assert result["exact_match"] is False
        assert result["edit_distance"] == 1
        assert 0.0 < result["ned"] < 1.0

    def test_number_exact_match(self):
        result = compute_field_normalization(5000.0, 5000.0, "number")
        assert result["exact_match"] is True
        assert result["relative_error"] == 0.0
        assert result["within_tolerance"] is True

    def test_number_within_tolerance(self):
        """1% off should be within the default 1% relative-error tolerance."""
        result = compute_field_normalization(5050.0, 5000.0, "number")
        assert result["exact_match"] is False
        assert result["relative_error"] == pytest.approx(0.01)
        assert result["within_tolerance"] is True

    def test_number_outside_tolerance(self):
        """2% off should fail tolerance."""
        result = compute_field_normalization(5100.0, 5000.0, "number")
        assert result["exact_match"] is False
        assert result["within_tolerance"] is False

    def test_missing_prediction_string(self):
        result = compute_field_normalization(None, "INR", "string")
        assert result["exact_match"] is False

    def test_missing_prediction_number(self):
        result = compute_field_normalization(None, 5000.0, "number")
        assert result["exact_match"] is False
