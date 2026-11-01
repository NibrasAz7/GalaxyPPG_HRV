"""Unit tests for the validation layer (peak matching and NN agreement)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation import (
    align_interval_sequences,
    beat_interval_agreement,
    estimate_lag,
    match_peaks,
    register_vendor_runs,
    sequence_interval_agreement,
    surrogate_recall,
    time_aligned_agreement,
)


def _times(seconds: np.ndarray) -> pd.DatetimeIndex:
    origin = pd.Timestamp("2024-03-19 00:00:00")
    return pd.DatetimeIndex(origin + pd.to_timedelta(seconds, unit="s"))


def test_match_peaks_perfect_recall_and_precision():
    reference = _times(np.arange(0, 60, 0.8))
    detected = reference.copy()

    result = match_peaks(reference, detected, tolerance_s=0.5)
    assert result["recall"] == pytest.approx(100.0)
    assert result["precision"] == pytest.approx(100.0)
    assert result["f1"] == pytest.approx(100.0)


def test_match_peaks_detects_missing_and_spurious_peaks():
    regular = np.arange(0, 60, 0.8)
    spurious = np.arange(100, 110, 0.8)
    reference = _times(regular)
    detected = _times(np.sort(np.concatenate([regular[:40], spurious])))

    result = match_peaks(reference, detected, tolerance_s=0.5)
    assert 40 <= result["recall"] < 100
    assert result["precision"] < 100
    assert result["n_reference"] == len(regular)
    assert result["n_detected"] == len(regular[:40]) + len(spurious)


def test_match_peaks_handles_empty_input():
    empty = np.array([])
    result = match_peaks(empty, empty, tolerance_s=0.5)
    assert result["recall"] == 0.0 and result["n_matched"] == 0


def test_time_aligned_agreement_identical_series():
    nn = np.full(200, 800.0) + np.sin(np.linspace(0, 8 * np.pi, 200)) * 20.0
    times = _times(np.cumsum(nn) / 1000.0)

    agreement = time_aligned_agreement(nn, times, nn, times, resample_hz=4.0)
    assert agreement["bias"] == pytest.approx(0.0, abs=1e-6)
    assert agreement["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert agreement["correlation"] == pytest.approx(1.0, abs=1e-6)


def test_time_aligned_agreement_handles_missing_series():
    nn = np.full(20, 800.0)
    times = _times(np.cumsum(nn) / 1000.0)

    assert time_aligned_agreement(None, None, nn, times) == {}
    assert time_aligned_agreement(nn, times, None, None) == {}


def test_time_aligned_agreement_detects_bias():
    phase = np.linspace(0, 12 * np.pi, 200)
    nn_a = 800.0 + 15.0 * np.sin(phase)
    nn_b = 780.0 + 15.0 * np.sin(phase)
    times_a = _times(np.cumsum(nn_a) / 1000.0)
    times_b = _times(np.cumsum(nn_b) / 1000.0)

    agreement = time_aligned_agreement(nn_a, times_a, nn_b, times_b, resample_hz=4.0)
    assert agreement["bias"] == pytest.approx(20.0, abs=2.0)
    assert agreement["loa_upper"] > agreement["bias"] >= agreement["loa_lower"]


def test_match_peaks_reports_timing_offsets():
    reference = _times(np.arange(0, 30, 0.8))
    detected = _times(np.arange(0, 30, 0.8) + 0.05)  # constant +50 ms lag

    result = match_peaks(reference, detected, tolerance_s=0.5)
    assert result["median_absolute_offset_ms"] == pytest.approx(50.0, abs=1.0)
    assert result["median_signed_offset_ms"] == pytest.approx(50.0, abs=1.0)


def test_match_peaks_perfect_timing_has_zero_offset():
    reference = _times(np.arange(0, 30, 0.8))
    result = match_peaks(reference, reference.copy(), tolerance_s=0.5)
    assert result["median_absolute_offset_ms"] == pytest.approx(0.0, abs=1e-9)


# --- regression tests for the validation redesign -------------------------------------


def test_time_aligned_agreement_uses_absolute_time():
    """Series starting at different absolute times must not be re-referenced (bug B1)."""
    rng = np.random.default_rng(3)
    nn = 800.0 + np.cumsum(rng.normal(0, 8, 400))
    times = _times(np.cumsum(nn) / 1000.0)
    late = 150  # second copy starts 150 beats (~2 min) later
    agreement = time_aligned_agreement(nn[late:], times[late:], nn, times, resample_hz=4.0)
    # identical except PCHIP end-slope effects at the start of the shorter copy
    assert agreement["rmse"] < 1.0
    assert agreement["correlation"] > 0.999
    # the previous self-referenced implementation compared unrelated beats
    shifted = time_aligned_agreement(nn[late:], times[late:] - (times[late] - times[0]), nn, times)
    assert shifted["rmse"] > 10 * agreement["rmse"]


def test_wide_tolerance_cannot_distinguish_phase_but_tight_can():
    """+-500 ms matching scores ~100 % for a half-beat-shifted train (bug B3)."""
    reference = _times(np.arange(0, 120, 0.75))
    shifted = _times(np.arange(0, 120, 0.75) + 0.30)  # 40 % of a beat out of phase
    assert match_peaks(reference, shifted, tolerance_s=0.5)["recall"] > 95
    assert match_peaks(reference, shifted, tolerance_s=0.1)["recall"] == 0
    assert surrogate_recall(reference, reference, tolerance_s=0.1) == 0


def test_estimate_lag_recovers_pulse_arrival_time():
    rng = np.random.default_rng(0)
    ibi = 0.8 + 0.05 * rng.standard_normal(200)
    reference = np.cumsum(ibi)
    detected = reference + 0.30 + rng.normal(0, 0.01, len(reference))
    lag = estimate_lag(reference, detected, max_lag_s=1.0)
    assert lag["lag_s"] == pytest.approx(0.30, abs=0.01)
    assert lag["prominence"] > 10
    result = match_peaks(reference, detected, tolerance_s=0.1, lag_s=lag["lag_s"])
    assert result["recall"] > 99
    assert result["timing_sd_ms"] == pytest.approx(10, abs=4)


def test_estimate_lag_respects_search_centre():
    reference = np.arange(0, 100, 0.8)
    detected = reference + 2.5
    assert estimate_lag(reference, detected, max_lag_s=0.3, center_s=2.4)["lag_s"] == pytest.approx(2.5, abs=0.01)


def test_beat_interval_agreement_only_uses_matched_consecutive_beats():
    reference = np.arange(0, 60, 0.8)
    detected = np.delete(reference, [10, 30]) + 0.2  # two missed beats, 200 ms delay
    result = beat_interval_agreement(reference, detected, tolerance_s=0.1, lag_s=0.2)
    assert result["mae"] == pytest.approx(0.0, abs=1e-6)
    assert result["n"] == len(reference) - 1 - 4


def test_sequence_alignment_tolerates_dropped_values():
    rng = np.random.default_rng(5)
    reference = 800 + rng.normal(0, 40, 120)
    other = np.delete(reference, [7, 50, 51]) + rng.normal(0, 1, 117)
    other = np.insert(other, 90, 400.0)  # spurious value
    pairs = align_interval_sequences(reference, other)
    assert len(pairs) >= 115
    result = sequence_interval_agreement(reference, other)
    assert result["mae"] < 1.5
    assert result["coverage_pct"] > 95


def test_vendor_run_registration_recovers_true_values_and_rejects_shuffled():
    rng = np.random.default_rng(7)
    ibi = 0.8 + 0.05 * np.sin(np.arange(400) / 3.0) + rng.normal(0, 0.02, 400)
    ref = np.r_[0.0, np.cumsum(ibi)]
    vendor_values = ibi * 1000 + rng.normal(0, 3, 400)
    vendor_ends = ref[1:] + 2.5 + rng.normal(0, 0.8, 400)  # timestamps off by ~seconds
    vendor_ends = np.sort(vendor_ends)
    good = register_vendor_runs(ref, ref[1:] + 2.5, vendor_values, contiguity_ms=60.0)
    assert good["coverage_pct"] > 80 and good["mae"] < 5
    shuffled = register_vendor_runs(ref, ref[1:] + 2.5, rng.permutation(vendor_values))
    assert shuffled["coverage_pct"] < 10
