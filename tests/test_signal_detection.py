"""Tests for filtering, signal repair and beat detection."""

from __future__ import annotations

import numpy as np
import pytest

from src.hrv_metrics import compute_frequency_domain
from src.peak_detection import (
    detect_peaks_scipy,
    detect_peaks_slope_sum,
    ensemble_peaks,
    select_optimal_peaks,
    template_matching,
)
from src.signal_processing import adaptive_bandpass_filter, bandpass_filter, repair_signal
from src.utils import load_config


@pytest.fixture(scope="module")
def config():
    return load_config("config.yaml")


@pytest.mark.parametrize("device", ["E4", "GalaxyWatch", "PolarH10"])
@pytest.mark.parametrize("motion", ["low", "moderate", "severe"])
def test_filters_are_stable_for_every_configuration(config, device, motion):
    """SOS design: finite output for every device x motion state (bug B5)."""
    fs = config["devices"][device]["fs"]
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, int(fs * 120))
    y = adaptive_bandpass_filter(x, device, motion, config=config)
    assert np.all(np.isfinite(y))
    assert np.std(y) < 10 * np.std(x)


def test_band_above_nyquist_is_rejected():
    """Silent clamping of an impossible band is now an error (bug B4)."""
    with pytest.raises(ValueError):
        bandpass_filter(np.zeros(1000), 130.0, 0.05, 100.0, 4)


def test_processing_ignores_activity_label(config):
    x = np.random.default_rng(1).normal(0, 1, 25 * 60)
    a = adaptive_bandpass_filter(x, "GalaxyWatch", "low", "baseline", config)
    b = adaptive_bandpass_filter(x, "GalaxyWatch", "low", "rest-1", config)
    assert np.array_equal(a, b)


def test_repair_signal_removes_level_shifts_and_nans():
    fs = 25.0
    t = np.arange(0, 60, 1 / fs)
    pulse = np.sin(2 * np.pi * 1.2 * t)
    x = pulse.copy()
    x[700:] += 40.0  # gain reset
    x[300:305] = np.nan
    repaired, unreliable = repair_signal(x, fs)
    assert np.all(np.isfinite(repaired))
    assert unreliable[300:305].all() and unreliable[699:701].any()
    assert np.std(np.diff(repaired)) < 0.5


def _pulse_train(fs, hr_bpm, seconds=60):
    t = np.arange(0, seconds, 1 / fs)
    phase = 2 * np.pi * hr_bpm / 60 * t
    return np.maximum(np.sin(phase), 0) ** 2, t


@pytest.mark.parametrize("hr", [60, 120, 175, 195])
def test_detectors_resolve_exercise_heart_rates(config, hr):
    """No detector may impose a half-rate ceiling below 200 bpm (bug B10)."""
    fs = 64.0
    x, t = _pulse_train(fs, hr)
    expected = len(t) / fs * hr / 60
    scipy_peaks = detect_peaks_scipy(x, fs, "E4", "severe", {"snr_db": 20}, config)
    slope_peaks = detect_peaks_slope_sum(x, fs)
    assert len(scipy_peaks) == pytest.approx(expected, rel=0.05)
    assert len(slope_peaks) == pytest.approx(expected, rel=0.05)


def test_template_fiducial_matches_systolic_peak():
    """Template beats must sit on the systolic maximum, not half a template later (bug B9)."""
    fs = 64.0
    x, _ = _pulse_train(fs, 72)
    true = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    beats = template_matching(x, fs, None, anchors=true)
    offsets = [np.min(np.abs(true - b)) for b in beats]
    assert np.median(offsets) <= 1


def test_ensemble_requires_consensus():
    peaks_a = np.array([100, 200, 300, 400])
    peaks_b = np.array([101, 201, 299, 401, 555])   # 555 is a false positive
    peaks_c = np.array([102, 199, 300, 650])          # 650 is a false positive
    result = ensemble_peaks(None, 100.0, {"neurokit2": peaks_a, "scipy": peaks_b, "slope": peaks_c})
    assert list(result) == [100, 200, 300, 400]


def test_priority_selection_depends_only_on_device_and_motion(config):
    results = {"neurokit2": np.arange(0, 1000, 50), "scipy": np.arange(0, 1000, 40),
               "ensemble": np.arange(0, 1000, 45)}
    priority = config["detection"]["priority"]
    a = select_optimal_peaks(results, "GalaxyWatch", "low", {}, "baseline", priority=priority)
    b = select_optimal_peaks(results, "GalaxyWatch", "low", {}, "rest-1", priority=priority)
    assert a[1] == b[1] == priority["GalaxyWatch"]["low"][0]


@pytest.mark.parametrize("duration", [60, 120, 180, 300])
@pytest.mark.parametrize("ibi", [0.6, 0.85])
def test_spectral_power_is_unbiased_and_length_independent(duration, ibi):
    """Sum(PSD) x df integration on a fixed Welch window (bug B6)."""
    t = np.arange(0, duration, ibi)
    nn = ibi * 1000 + 30 * np.sin(2 * np.pi * 0.25 * t)  # HF power = 450 ms^2
    result = compute_frequency_domain(nn, min_hf_duration_s=50, min_lf_duration_s=50)
    assert result["hf_power"] == pytest.approx(450, rel=0.06)


def _asymmetric_pulse(fs, hr_bpm=72.0, seconds=60.0):
    """Fast upstroke (25 % of the cycle), slow decay: a realistic PPG shape."""
    t = np.arange(0, seconds, 1 / fs)
    phase = (t * hr_bpm / 60.0) % 1.0
    up = phase < 0.25
    x = np.where(up, np.sin(np.pi / 2 * phase / 0.25), np.exp(-(phase - 0.25) / 0.3))
    onsets = np.where(np.diff(phase) < 0)[0] + 1
    return x - x.mean(), onsets


def test_configured_polarity_is_applied(config):
    from src.signal_processing import pulse_asymmetry
    x, _ = _asymmetric_pulse(25.0)
    raw_reflective = -x  # Galaxy Watch raw is inverted
    upright = adaptive_bandpass_filter(raw_reflective, "GalaxyWatch", "low", config=config)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(upright, distance=10)
    assert pulse_asymmetry(upright, peaks) < 0.45


def test_onset_anchor_tangent_precedes_systolic_peak():
    from src.peak_detection import locate_fiducial
    fs = 64.0
    x, onsets = _asymmetric_pulse(fs)
    onsets = onsets[(onsets > 5) & (onsets < len(x) - 40)]
    tangent = locate_fiducial(x, onsets, fs, "tangent", anchor="onset")
    peak = locate_fiducial(x, onsets, fs, "peak", anchor="onset")
    onset = locate_fiducial(x, onsets, fs, "onset", anchor="onset")
    assert np.all(peak - tangent > 0)
    assert np.median(np.abs(tangent - onset)) < 0.15 * fs
    period = fs * 60 / 72
    assert np.median(peak - onset) == pytest.approx(0.25 * period, abs=2)
