"""
Tests for stats/bootstrap.py.

Covers bootstrap CI properties (mean inside CI, CI shrinks with more
samples, deterministic under seed) and the Wilcoxon signed-rank
implementation against textbook properties.
"""

import math

import pytest

from stats.bootstrap import (
    bootstrap_ci,
    ci_overlaps,
    wilcoxon_signed_rank,
)


class TestBootstrapCI:
    def test_empty_returns_nones(self):
        result = bootstrap_ci([])
        assert result["n"] == 0
        assert result["mean"] is None
        assert result["ci_lower"] is None

    def test_single_value_collapses_ci(self):
        result = bootstrap_ci([0.5])
        assert result["n"] == 1
        assert result["mean"] == 0.5
        assert result["ci_lower"] == 0.5
        assert result["ci_upper"] == 0.5
        assert result["ci_width"] == 0.0

    def test_mean_is_inside_ci(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        result = bootstrap_ci(values, n_resamples=2000, seed=1)
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_seed_is_deterministic(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        a = bootstrap_ci(values, n_resamples=500, seed=42)
        b = bootstrap_ci(values, n_resamples=500, seed=42)
        assert a == b

    def test_ci_shrinks_with_more_samples(self):
        small = bootstrap_ci([0.4, 0.5, 0.6] * 3, n_resamples=2000, seed=7)
        large = bootstrap_ci([0.4, 0.5, 0.6] * 100, n_resamples=2000, seed=7)
        assert large["ci_width"] < small["ci_width"]

    def test_filters_none_values(self):
        result = bootstrap_ci([1.0, 2.0, None, 3.0], n_resamples=200, seed=0)
        assert result["n"] == 3
        assert result["mean"] == pytest.approx(2.0, abs=0.01)


class TestCIOverlaps:
    def test_clear_overlap(self):
        a = {"ci_lower": 0.4, "ci_upper": 0.6}
        b = {"ci_lower": 0.5, "ci_upper": 0.7}
        assert ci_overlaps(a, b) is True

    def test_no_overlap(self):
        a = {"ci_lower": 0.1, "ci_upper": 0.3}
        b = {"ci_lower": 0.5, "ci_upper": 0.7}
        assert ci_overlaps(a, b) is False

    def test_touching_endpoints_overlap(self):
        a = {"ci_lower": 0.1, "ci_upper": 0.5}
        b = {"ci_lower": 0.5, "ci_upper": 0.9}
        assert ci_overlaps(a, b) is True

    def test_none_inputs_overlap_defensively(self):
        a = {"ci_lower": None, "ci_upper": None}
        b = {"ci_lower": 0.0, "ci_upper": 1.0}
        assert ci_overlaps(a, b) is True


class TestWilcoxonSignedRank:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            wilcoxon_signed_rank([1.0, 2.0], [1.0])

    def test_all_zero_diffs_returns_no_difference(self):
        result = wilcoxon_signed_rank([0.5] * 10, [0.5] * 10)
        assert result["p_value"] == 1.0
        assert result["significant"] is False
        assert result["n_pairs"] == 0

    def test_too_few_samples_returns_none(self):
        """Fewer than 10 non-zero diffs should not produce a real p-value."""
        a = [0.1, 0.2, 0.3, 0.4, 0.5]
        b = [0.2, 0.3, 0.4, 0.5, 0.6]
        result = wilcoxon_signed_rank(a, b)
        assert result["p_value"] is None
        assert result["significant"] is None

    def test_clearly_better_model_is_significant(self):
        """Model A consistently scores higher than B by 0.1 on every sample."""
        a = [0.7 + 0.01 * i for i in range(20)]
        b = [0.6 + 0.01 * i for i in range(20)]
        result = wilcoxon_signed_rank(a, b)
        assert result["p_value"] < 0.05
        assert result["significant"] is True

    def test_random_noise_is_not_significant(self):
        """Two models with identical scores plus tiny equal-and-opposite noise."""
        a = [0.5] * 20
        b = [0.5] * 20
        # Inject equal and opposite differences
        for i in range(0, 20, 2):
            a[i]   = 0.51
            b[i+1] = 0.51
        result = wilcoxon_signed_rank(a, b)
        assert result["p_value"] > 0.05
        assert result["significant"] is False

    def test_p_value_is_two_tailed(self):
        """Reversing roles of A and B gives the same p-value (two-tailed)."""
        a = [0.7 + 0.01 * i for i in range(15)]
        b = [0.6 + 0.01 * i for i in range(15)]
        forward = wilcoxon_signed_rank(a, b)
        reverse = wilcoxon_signed_rank(b, a)
        assert math.isclose(forward["p_value"], reverse["p_value"], rel_tol=1e-9)

    def test_effect_size_is_in_unit_range(self):
        a = [0.7 + 0.01 * i for i in range(20)]
        b = [0.6 + 0.01 * i for i in range(20)]
        result = wilcoxon_signed_rank(a, b)
        assert 0.0 <= result["effect_size"] <= 1.0
