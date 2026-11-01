"""Unit tests for the HRV metric and artifact-correction layer.

The tests use synthetic NN interval series with known properties, so they run
without the GalaxyPPG dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.hrv_metrics import (
    SEGMENT_METRIC_KEYS,
    aggregate_hrv_metrics,
    approximate_entropy,
    build_nn_segments,
    calculate_ibi,
    compute_frequency_domain,
    compute_poincare,
    compute_time_domain,
    detect_gaps,
    detect_ibi_artifacts,
    robust_hr,
)


def _timestamps_from_nn(nn_ms: np.ndarray) -> pd.DatetimeIndex:
    """Build beat timestamps from an interval series."""
    origin = pd.Timestamp("2024-03-19 00:00:00")
    cumulative_ms = np.concatenate([[0.0], np.cumsum(nn_ms)])
    return pd.DatetimeIndex(origin + pd.to_timedelta(cumulative_ms, unit="ms"))


def _clean_nn(n_beats: int = 240, mean_ms: float = 800.0, amplitude_ms: float = 25.0) -> np.ndarray:
    """Respiratory-like periodic NN series."""
    phase = np.linspace(0, 8 * np.pi, n_beats)
    return mean_ms + amplitude_ms * np.sin(phase)


@pytest.fixture
def config() -> dict:
    return {
        "processing": {
            "ibi_range": [300, 2000],
            "missed_beat_factor": 1.5,
            "extra_beat_factor": 0.6,
            "median_window_beats": 11,
            "gap_threshold_ms": 3000.0,
        },
        "hrv": {
            "min_segment_duration_s": 30.0,
            "min_segment_beats": 20,
            "max_interpolated_fraction": 0.25,
            "max_jitter_percent": 15.0,
            "min_hf_duration_s": 60.0,
            "min_lf_duration_s": 120.0,
            "resample_hz": 4.0,
            "apen_m": 2,
            "apen_r_factor": 0.2,
        },
    }


def test_calculate_ibi():
    peaks = np.array([0, 64, 128, 192])
    ibis = calculate_ibi(peaks, fs=64)
    assert np.allclose(ibis, [1000.0, 1000.0, 1000.0])


def test_clean_series_has_no_artifacts():
    nn = _clean_nn()
    invalid = detect_ibi_artifacts(nn)
    assert invalid.sum() == 0
    assert detect_gaps(nn, invalid).sum() == 0


def test_missed_and_extra_beats_are_flagged():
    nn = _clean_nn(n_beats=200)
    nn[100] *= 2.0  # missed beat
    nn[150] *= 0.5  # extra detected peak

    invalid = detect_ibi_artifacts(nn)
    assert invalid[100]
    assert invalid[150]
    assert invalid.sum() <= 8  # only the neighbourhood of the two faults


def test_long_invalid_run_becomes_gap():
    nn = _clean_nn(n_beats=200)
    nn[80:84] = np.array([1900.0, 1900.0, 1800.0, 1850.0])  # sustained dropout

    invalid = detect_ibi_artifacts(nn)
    gap = detect_gaps(nn, invalid)
    assert gap[80:84].all()


def test_segmentation_splits_at_gaps(config):
    nn = _clean_nn(n_beats=300)
    nn[150:154] = 1900.0

    beat_times = _timestamps_from_nn(nn)
    segments, reasons = build_nn_segments(nn, beat_times, config)

    assert reasons["total"] == 0
    assert len(segments) == 2
    for segment in segments:
        assert segment.duration_s >= config["hrv"]["min_segment_duration_s"]
        assert segment.n_beats >= config["hrv"]["min_segment_beats"]


def test_interpolation_restores_near_clean_statistics(config):
    clean = _clean_nn(n_beats=300)
    corrupted = clean.copy()
    corrupted[100] *= 2.0  # single missed beat

    beat_times = _timestamps_from_nn(corrupted)
    segments, _ = build_nn_segments(corrupted, beat_times, config)

    assert len(segments) == 1
    corrected = segments[0].nn
    assert segments[0].interpolated_fraction > 0

    clean_sdnn = np.std(clean, ddof=1)
    corrected_sdnn = np.std(corrected, ddof=1)
    assert abs(corrected_sdnn - clean_sdnn) / clean_sdnn < 0.25


def test_consecutive_missed_beats_split_segment(config):
    nn = _clean_nn(n_beats=300)
    nn[100] *= 2.0
    nn[101] *= 2.0  # 3200 ms of missing data -> gap

    beat_times = _timestamps_from_nn(nn)
    segments, _ = build_nn_segments(nn, beat_times, config)

    assert len(segments) == 2


def test_segment_with_excessive_interpolation_is_rejected(config):
    nn = _clean_nn(n_beats=100)
    nn[10:60] *= 0.4  # half the recording is corrupted

    beat_times = _timestamps_from_nn(nn)
    segments, reasons = build_nn_segments(nn, beat_times, config)

    assert all(segment.duration_s < 60 for segment in segments)
    assert reasons["total"] >= 1


def test_device_hr_gate_is_off_by_default(config):
    """The device-HR gate is circular and disabled unless explicitly enabled."""
    nn = _clean_nn(n_beats=300)
    beat_times = _timestamps_from_nn(nn)

    passed, _ = build_nn_segments(nn, beat_times, config, reference_hr=120.0)
    assert len(passed) == 1

    enabled = {**config, "hrv": {**config["hrv"], "device_hr_gate": True}}
    rejected, reasons = build_nn_segments(nn, beat_times, enabled, reference_hr=120.0)
    assert rejected == []
    assert reasons["hr_deviation"] == 1


def test_rhythm_gate_rejects_half_rate_chain(config):
    """A clean-looking half-rate beat chain is rejected by the spectral rhythm gate."""
    nn = _clean_nn(n_beats=300) * 2.0  # every other beat missed, consistently
    nn = np.clip(nn, None, 1990.0)
    beat_times = _timestamps_from_nn(nn)
    segments, reasons = build_nn_segments(nn, beat_times, config, rhythm_hr=75.0)
    assert segments == []
    assert reasons["rhythm"] >= 1


def test_cadence_gate_rejects_motion_locked_segment(config):
    nn = np.full(200, 1000.0) + np.random.default_rng(1).normal(0, 20, 200)  # 60 bpm
    beat_times = _timestamps_from_nn(nn)
    segments, reasons = build_nn_segments(nn, beat_times, config, cadence_hz=2.0)  # 120 steps/min
    assert segments == []
    assert reasons["cadence"] >= 1
    kept, _ = build_nn_segments(nn, beat_times, config, cadence_hz=1.37)
    assert len(kept) == 1


def test_time_domain_metrics_are_consistent():
    nn = np.array([800.0, 810.0, 790.0, 805.0, 795.0])
    metrics = compute_time_domain(nn)

    assert metrics["mean_nn"] == pytest.approx(800.0, abs=1.0)
    assert metrics["mean_hr"] == pytest.approx(60000 / 800.0, abs=1.0)
    assert metrics["sdnn"] == pytest.approx(np.std(nn, ddof=1))
    assert metrics["rmssd"] > 0
    assert 0 <= metrics["pnn50"] <= 100


def test_poincare_descriptors():
    nn = _clean_nn(n_beats=100)
    poincare = compute_poincare(nn)
    assert poincare["sd1"] == pytest.approx(np.std(np.diff(nn), ddof=1) / np.sqrt(2))
    assert poincare["sd2"] > poincare["sd1"]


def test_approximate_entropy_range():
    rng = np.random.default_rng(0)
    phase = np.linspace(0, 20 * np.pi, 200)
    periodic = 800.0 + 40.0 * np.sin(phase)
    noisy = 800.0 + rng.normal(0, 40.0, 200)

    assert approximate_entropy(periodic) < approximate_entropy(noisy)
    assert approximate_entropy(noisy) > 0


def test_frequency_domain_duration_gating():
    short = _clean_nn(n_beats=40)  # ~32 s
    assert compute_frequency_domain(short, min_hf_duration_s=60.0) == {}

    long = _clean_nn(n_beats=120, amplitude_ms=30.0)  # ~96 s
    spectra = compute_frequency_domain(long, min_hf_duration_s=60.0, min_lf_duration_s=120.0)
    assert np.isfinite(spectra["hf_power"])
    assert np.isnan(spectra["lf_power"])


def test_aggregate_hrv_metrics_reports_yield(config):
    nn = _clean_nn(n_beats=200)
    beat_times = _timestamps_from_nn(nn)
    segments, reasons = build_nn_segments(nn, beat_times, config)
    metrics = aggregate_hrv_metrics(segments, reasons, config)

    assert metrics["n_segments"] == 1
    assert metrics["n_segments_excluded"] == 0
    assert metrics["n_beats"] == 201
    assert metrics["n_intervals"] == 200
    assert np.isfinite(metrics["sdnn"])
    assert np.isfinite(metrics["rmssd"])
    assert metrics["pct_interpolated"] == 0.0


def test_aggregation_handles_mixed_segment_durations(config):
    """Segments shorter than the HF gate must not break the aggregation."""
    nn = _clean_nn(n_beats=200)
    nn[40:44] = 1900.0  # 7.6 s gap -> splits into a ~32 s and a ~125 s segment

    beat_times = _timestamps_from_nn(nn)
    segments, reasons = build_nn_segments(nn, beat_times, config)
    assert len(segments) == 2

    metrics = aggregate_hrv_metrics(segments, reasons, config)
    assert metrics["n_segments"] == 2
    assert np.isfinite(metrics["sdnn"])
    assert np.isfinite(metrics["hf_power"])
    for key in SEGMENT_METRIC_KEYS:
        assert key in metrics


def test_robust_hr_ignores_outliers():
    nn = np.full(100, 800.0)
    nn[5] = 1500.0
    assert robust_hr(nn) == pytest.approx(75.0)
    assert np.isnan(robust_hr(np.array([100.0, 5000.0])))
