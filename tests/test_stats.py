"""Unit tests for the cohort statistics layer."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.stats import (
    benjamini_hochberg,
    cohens_dz,
    icc3_1,
    kruskal_wallis,
    paired_comparison,
)


def test_icc_identical_measurements_is_one():
    values = np.array([45.0, 52.0, 38.0, 61.0, 47.0, 55.0])
    assert icc3_1(values, values) == pytest.approx(1.0)


def test_icc_offset_measurements_stay_high():
    values = np.array([45.0, 52.0, 38.0, 61.0, 47.0, 55.0])
    assert icc3_1(values, values + 3.0) == pytest.approx(1.0)


def test_icc_uncorrelated_measurements_is_low():
    rng = np.random.default_rng(0)
    a = rng.normal(50, 5, 60)
    b = rng.normal(50, 5, 60)
    assert icc3_1(a, b) < 0.5


def test_icc_requires_enough_pairs():
    assert np.isnan(icc3_1([1.0, 2.0], [1.0, 2.0]))


def test_icc_ignores_missing_pairs():
    a = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 6.0])
    b = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1])
    assert np.isfinite(icc3_1(a, b))


def test_cohens_dz():
    differences = np.array([1.0, 2.0, 3.0])
    assert cohens_dz(differences) == pytest.approx(2.0 / 1.0)


def test_cohens_dz_handles_zero_variance():
    assert np.isnan(cohens_dz(np.ones(5)))


def test_benjamini_hochberg_uniform_p_values():
    adjusted = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
    assert adjusted == pytest.approx([0.05] * 5)


def test_benjamini_hochberg_preserves_order_and_is_monotone():
    adjusted = benjamini_hochberg([0.5, 0.001])
    assert adjusted[1] == pytest.approx(0.002)
    assert adjusted[0] == pytest.approx(0.5)

    monotone = benjamini_hochberg([0.001, 0.009, 0.2, 0.3])
    assert np.all(np.diff(monotone) >= -1e-12)


def test_benjamini_hochberg_keeps_nan():
    adjusted = benjamini_hochberg([0.01, np.nan, 0.04])
    assert np.isnan(adjusted[1])
    assert np.isfinite(adjusted[[0, 2]]).all()


def test_paired_comparison_reports_effect_size():
    a = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    b = a - np.array([1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0])
    result = paired_comparison(a, b, "metric", "a vs b", {"activity": "baseline"})

    assert result["n"] == 7
    assert result["median_diff"] == pytest.approx(2.0)
    assert result["activity"] == "baseline"
    assert np.isfinite(result["p_value"])
    assert np.isfinite(result["cohens_dz"])


def test_paired_comparison_constant_difference_has_no_effect_size():
    a = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    result = paired_comparison(a, a - 2.0, "metric", "a vs b")

    assert result["median_diff"] == pytest.approx(2.0)
    assert np.isnan(result["cohens_dz"])


def test_paired_comparison_without_variance_is_skipped():
    a = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert paired_comparison(a, a, "metric", "a vs a") is None


def test_paired_comparison_requires_six_pairs():
    a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    assert paired_comparison(a, a - 1.0, "metric", "a vs b") is None


def test_kruskal_wallis_separated_groups():
    rng = np.random.default_rng(1)
    groups = [rng.normal(0, 1, 20), rng.normal(10, 1, 20)]
    result = kruskal_wallis(groups)
    assert result["p_value"] < 0.001


def test_kruskal_wallis_needs_two_groups():
    result = kruskal_wallis([np.arange(10.0)])
    assert np.isnan(result["p_value"])


def test_vectorised_cluster_bootstrap_matches_estimate():
    import pandas as pd

    from analysis.agreement import cluster_bootstrap, cluster_bootstrap_mean

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"participant": np.repeat(np.arange(15), 4), "x": rng.normal(3, 1, 60)})
    fast = cluster_bootstrap_mean(frame, "x")
    slow = cluster_bootstrap(frame, lambda f: f["x"].mean())
    assert fast[0] == pytest.approx(slow[0])
    assert fast[1] == pytest.approx(slow[1], abs=0.1) and fast[2] == pytest.approx(slow[2], abs=0.1)
