"""Unit tests for sub-sample peak refinement."""

from __future__ import annotations

import numpy as np
import pytest

from src.peak_detection import refine_peaks_parabolic


def test_symmetric_peak_keeps_integer_position():
    signal = np.array([0.0, 0.5, 0.9, 1.0, 0.9, 0.5, 0.0])
    positions = refine_peaks_parabolic(signal, np.array([3]))
    assert positions[0] == pytest.approx(3.0)


def test_asymmetric_peak_shifts_towards_true_maximum():
    signal = np.array([0.0, 0.5, 1.0, 0.9, 0.4, 0.1])
    # curvature = 0.5 - 2*1.0 + 0.9 = -0.6 ; offset = 0.5*(0.5-0.9)/-0.6 = 0.333
    positions = refine_peaks_parabolic(signal, np.array([2]))
    assert positions[0] == pytest.approx(2.3333, abs=1e-3)


def test_offsets_are_bounded_by_half_a_sample():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=50)
    peaks = np.array([5, 10, 20, 30, 40])
    positions = refine_peaks_parabolic(signal, peaks)

    offsets = positions - peaks
    assert np.all(np.abs(offsets) <= 0.5 + 1e-9)


def test_edge_peaks_are_left_unchanged():
    signal = np.array([1.0, 0.8, 0.5, 0.2, 0.1])
    positions = refine_peaks_parabolic(signal, np.array([0, 4]))
    assert positions.tolist() == [0.0, 4.0]


def test_empty_peak_array():
    positions = refine_peaks_parabolic(np.zeros(10), np.array([], dtype=int))
    assert len(positions) == 0


def test_refinement_reduces_interval_quantisation():
    """Sub-sample peaks recover intervals that integer sampling cannot."""
    rng = np.random.default_rng(0)
    fs = 25.0
    true_positions = np.cumsum(rng.uniform(15.0, 22.0, 40))  # samples
    grid = np.arange(int(true_positions[-1]) + 10)
    signal = np.zeros_like(grid, dtype=float)
    for centre in true_positions:
        signal += np.exp(-0.5 * ((grid - centre) / 2.0) ** 2)

    detected = np.round(true_positions).astype(int)
    refined = refine_peaks_parabolic(signal, detected)

    true_ibis = np.diff(true_positions) / fs * 1000.0
    integer_ibis = np.diff(detected) / fs * 1000.0
    refined_ibis = np.diff(refined) / fs * 1000.0

    integer_rmse = float(np.sqrt(np.mean((integer_ibis - true_ibis) ** 2)))
    refined_rmse = float(np.sqrt(np.mean((refined_ibis - true_ibis) ** 2)))

    assert refined_rmse < integer_rmse
    assert refined_rmse < 20.0


def test_harmonic_sum_finds_heart_rate():
    from src.peak_detection import harmonic_sum_hr

    fs = 64.0
    t = np.arange(0, 60, 1 / fs)
    # Fundamental at 1.2 Hz plus harmonics, with a stronger 2nd harmonic
    signal = 0.4 * np.sin(2 * np.pi * 1.2 * t) + 1.0 * np.sin(2 * np.pi * 2.4 * t)

    estimate = harmonic_sum_hr(signal, fs)
    assert estimate == pytest.approx(72.0, abs=6.0)


def test_estimate_rhythm_interval_prefers_device_when_plausible():
    from src.peak_detection import estimate_rhythm_interval

    fs = 64.0
    t = np.arange(0, 30, 1 / fs)
    signal = np.sin(2 * np.pi * 1.2 * t)

    assert estimate_rhythm_interval(signal, fs, device_hr=90.0, prior="auto") == pytest.approx(666.7, abs=1.0)
    interval = estimate_rhythm_interval(signal, fs, device_hr=np.nan, prior="auto")
    assert 60000.0 / interval == pytest.approx(72.0, abs=6.0)  # 1.2 Hz within one bin


def test_select_peaks_with_rhythm_rejects_off_rhythm_spikes():
    from src.peak_detection import select_peaks_with_rhythm

    fs = 64.0
    interval_samples = 0.8 * fs  # 800 ms -> 75 bpm
    true_peaks = np.round(np.arange(2, 60) * interval_samples).astype(int)
    spikes = np.array([int(2.4 * interval_samples), int(7.4 * interval_samples)])

    rng = np.random.default_rng(0)
    signal = 0.05 * rng.normal(size=true_peaks.max() + 10)
    signal[true_peaks] = 1.0
    signal[spikes] = 1.2  # spurious peaks are even taller

    candidates = np.sort(np.concatenate([true_peaks, spikes]))
    selected = select_peaks_with_rhythm(signal, candidates, fs, 800.0)

    assert not set(selected.tolist()) & set(spikes.tolist())
    assert len(selected) >= len(true_peaks) - 2


def test_select_peaks_with_rhythm_tolerates_missed_beats():
    from src.peak_detection import select_peaks_with_rhythm

    fs = 64.0
    interval_samples = 0.8 * fs
    beats = np.round(np.arange(2, 40) * interval_samples).astype(int)
    missing = beats[10]

    rng = np.random.default_rng(1)
    signal = 0.05 * rng.normal(size=beats.max() + 10)
    signal[beats] = 1.0
    candidates = beats[beats != missing]

    selected = select_peaks_with_rhythm(signal, candidates, fs, 800.0)
    # The beat after the hole is still accepted (spacing 2x the interval)
    assert missing + int(interval_samples) in selected.tolist() or len(selected) >= len(candidates) - 1


def test_select_peaks_with_rhythm_without_prior_returns_candidates():
    from src.peak_detection import select_peaks_with_rhythm

    signal = np.ones(500)
    candidates = np.array([10, 20, 30, 40])
    result = select_peaks_with_rhythm(signal, candidates, 64.0, np.nan)
    assert np.array_equal(result, candidates)


def test_complete_rhythm_intervals_inserts_missed_beats():
    from src.peak_detection import complete_rhythm_intervals

    fs = 64.0
    interval_samples = 0.8 * fs
    # Beats at 0, 1x, 2x, 4x, 5x -> two missed beats after the third
    positions = np.array([0, 1, 2, 4, 5], dtype=float) * interval_samples
    completed, synthetic = complete_rhythm_intervals(positions, fs, 800.0)

    assert len(completed) == len(positions) + 1
    assert synthetic.sum() == 1
    assert synthetic[3]  # inserted between the beats at 2x and 4x
    assert completed[3] == pytest.approx(3 * interval_samples)


def test_complete_rhythm_intervals_leaves_regular_rhythm_untouched():
    from src.peak_detection import complete_rhythm_intervals

    fs = 64.0
    positions = np.arange(10) * 0.8 * fs
    completed, synthetic = complete_rhythm_intervals(positions, fs, 800.0)

    assert np.allclose(completed, positions)
    assert not synthetic.any()


def test_complete_rhythm_intervals_requires_prior():
    from src.peak_detection import complete_rhythm_intervals

    positions = np.array([0.0, 100.0])
    completed, synthetic = complete_rhythm_intervals(positions, 64.0, np.nan)
    assert np.allclose(completed, positions)
    assert not synthetic.any()


def test_rhythm_tracking_plus_completion_recovers_half_rate_series():
    """A half-rate selection must not halve the HR estimate after completion."""
    from src.peak_detection import complete_rhythm_intervals, select_peaks_with_rhythm

    fs = 64.0
    true_interval = 0.6 * fs  # 600 ms -> 100 bpm
    beats = np.round(np.arange(2, 60) * true_interval).astype(int)
    # Detector missed every other beat: only the tall ones are candidates
    tall = beats[::2]
    signal = np.zeros(beats.max() + 10)
    signal[beats] = 1.0
    signal[tall] = 1.6
    candidates = tall

    selected = select_peaks_with_rhythm(signal, candidates, fs, 600.0)
    completed, synthetic = complete_rhythm_intervals(selected, fs, 600.0)

    ibis = np.diff(completed) / fs * 1000.0
    hr = 60000.0 / np.median(ibis)
    assert hr == pytest.approx(100.0, abs=8.0)
    assert synthetic.sum() > 0


def _pulse_train_signal(fs: float, interval_s: float, n_beats: int) -> tuple[np.ndarray, np.ndarray]:
    """Band-limited pulse train with systolic peaks at ``interval_s`` spacing."""
    rate = 1.0 / interval_s
    t = np.arange(0, n_beats * interval_s, 1.0 / fs)
    signal = (
        np.sin(2 * np.pi * rate * t)
        + 0.4 * np.sin(4 * np.pi * rate * t)
        + 0.15 * np.sin(6 * np.pi * rate * t)
    )
    beats = np.round((np.arange(n_beats) + 0.25) / rate * fs).astype(int)
    return signal, beats[beats < len(signal)]


def test_spectral_rhythm_peaks_keeps_full_rate_chain():
    """With candidates at the true rate, the spectral prior keeps all beats."""
    from src.peak_detection import spectral_rhythm_peaks

    fs = 64.0
    signal, beats = _pulse_train_signal(fs, 0.6, 40)  # 100 bpm
    results = {"neurokit2": beats, "scipy": beats[::2]}

    selected = spectral_rhythm_peaks(results, signal, fs)

    assert len(selected) >= len(beats) - 2


def test_spectral_rhythm_peaks_requires_candidates():
    from src.peak_detection import spectral_rhythm_peaks

    selected = spectral_rhythm_peaks({"scipy": np.array([1, 2, 3])}, np.ones(500), 64.0)
    assert len(selected) == 0
