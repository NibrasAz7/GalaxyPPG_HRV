"""Tests for the agreement statistics."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.agreement import bland_altman, icc_a1, log_ratio_bland_altman, tost_paired


def test_icc_a1_penalises_systematic_bias():
    x = np.array([45, 52, 38, 61, 47, 55, 60, 43.0])
    assert icc_a1(x, x)["icc"] == pytest.approx(1.0)
    assert icc_a1(x, x + 10)["icc"] < 0.7  # consistency ICC would still be 1.0


def test_icc_a1_bootstrap_interval_brackets_estimate():
    rng = np.random.default_rng(0)
    truth = rng.normal(70, 10, 40)
    result = icc_a1(truth + rng.normal(0, 3, 40), truth)
    assert result["ci_low"] < result["icc"] < result["ci_high"] < 1.0
    assert 0.85 < result["icc"] < 0.97


def test_repeated_measures_limits_exceed_naive_limits_for_clustered_differences():
    rng = np.random.default_rng(1)
    subject = np.repeat(np.arange(12), 6)
    diff = np.repeat(rng.normal(0, 5, 12), 6) + rng.normal(0, 1, 72)
    naive = bland_altman(diff, np.zeros(72))
    clustered = bland_altman(diff, np.zeros(72), subject)
    assert clustered["n_subjects"] == 12
    assert abs(naive["sd"] - clustered["sd"]) < 2.0
    assert clustered["loa_high"] > clustered["bias"] > clustered["loa_low"]


def test_log_ratio_limits_are_multiplicative():
    rng = np.random.default_rng(2)
    reference = rng.uniform(20, 60, 50)
    result = log_ratio_bland_altman(reference * 1.2 * np.exp(rng.normal(0, 0.05, 50)), reference)
    assert result["ratio"] == pytest.approx(1.2, abs=0.03)
    assert result["ratio_loa_low"] < result["ratio"] < result["ratio_loa_high"]


def test_tost_declares_equivalence_only_inside_bounds():
    rng = np.random.default_rng(3)
    inside = tost_paired(rng.normal(0.01, 0.05, 30), -0.1, 0.1)
    outside = tost_paired(rng.normal(0.15, 0.05, 30), -0.1, 0.1)
    assert inside["equivalent"] and not outside["equivalent"]


def test_generated_manuscript_macro_names_are_valid_latex():
    """Control-sequence names may contain letters only (no digits); guards the generator."""
    import os
    import re

    path = os.path.join("..", "Assessment-of-PPG-Pipline-Paper", "generated", "numbers.tex")
    if not os.path.exists(path):
        pytest.skip("manuscript not present")
    names = re.findall(r"\\newcommand\{\\([^}]*)\}", open(path, encoding="utf-8").read())
    assert names and all(re.fullmatch(r"[A-Za-z]+", n) for n in names)
