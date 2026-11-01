"""Tests for sample clocks, clock synchronisation and vendor timestamp conventions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sync import estimate_clock_offset, estimate_local_offset, resolve_window_offsets
from src.timebase import fit_sample_clock


def test_sample_clock_recovers_true_rate_from_quantised_stamps():
    """Polar-like stream: 130.28 Hz with millisecond-quantised timestamps (bug B2)."""
    true_fs = 130.28
    n = 130 * 180
    seconds = np.arange(n) / true_fs
    stamps = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.floor(seconds * 1000) / 1000, unit="s")
    clock = fit_sample_clock(pd.DatetimeIndex(stamps))
    assert clock.fs == pytest.approx(true_fs, rel=1e-4)
    # nominal 130 Hz would drift ~0.39 s by the end; the fitted clock stays < 1 ms
    assert abs(clock.to_seconds(np.array([n - 1]))[0] - seconds[-1]) < 1e-3
    assert abs((n - 1) / 130.0 - seconds[-1]) > 0.35


def test_sample_clock_preserves_packet_loss_gaps():
    fs = 25.0
    k = np.r_[np.arange(0, 500), np.arange(550, 1000)]  # 50 samples (2 s) lost
    stamps = pd.Timestamp("2024-01-01") + pd.to_timedelta(k / fs, unit="s")
    clock = fit_sample_clock(pd.DatetimeIndex(stamps))
    assert clock.n_gaps == 1
    assert clock.to_seconds(np.array([500]))[0] == pytest.approx(550 / fs, abs=1e-3)


def _acc(times_s: np.ndarray, activity: np.ndarray) -> pd.DataFrame:
    idx = pd.Timestamp("2024-01-01") + pd.to_timedelta(times_s, unit="s")
    return pd.DataFrame({"acc_x_g": 1.0 + activity, "acc_y_g": 0.0, "acc_z_g": 0.0}, index=idx)


def _activity(t: np.ndarray, rng) -> np.ndarray:
    bursts = np.zeros_like(t)
    for start in rng.uniform(0, t[-1] - 60, 25):
        bursts += ((t > start) & (t < start + rng.uniform(5, 40))) * rng.uniform(0.2, 1.0)
    return bursts * (1 + 0.5 * np.sin(2 * np.pi * 1.8 * t)) + rng.normal(0, 0.02, len(t))


def test_clock_offset_from_accelerometer_activity():
    rng = np.random.default_rng(2)
    t = np.arange(0, 1800, 1 / 50)
    activity = _activity(t, rng)
    reference = _acc(t, activity)
    device = _acc(t + 7.3, activity)  # device clock 7.3 s ahead
    offset = estimate_clock_offset(reference, device, "Dev")
    assert offset.reliable
    assert offset.offset_s == pytest.approx(7.3, abs=0.15)


def test_local_offset_tracks_clock_jump():
    rng = np.random.default_rng(4)
    t = np.arange(0, 3600, 1 / 50)
    activity = _activity(t, rng)
    shift = np.where(t < 1800, 2.0, 4.5)  # device re-synchronised mid-session
    reference = _acc(t, activity)
    device = _acc(t + shift, activity)
    start = pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=2500)
    end = start + pd.Timedelta(seconds=180)
    local = estimate_local_offset(reference, device, "Dev", start, end, session_offset_s=2.0)
    assert local.reliable
    assert local.offset_s == pytest.approx(4.5, abs=0.15)


def test_isolated_mislocks_are_rejected():
    # true offset about -5 s; one mis-lock at 2.4 s, one at -12 s, two unreliable windows
    local = [-5.0, -5.1, -4.9, 2.4, -5.0, None, -12.0, -5.05, None, -4.95]
    times = list(range(10))
    offsets, sources = resolve_window_offsets(local, times, session_offset_s=-7.8)
    assert all(abs(o + 5.0) < 0.2 for o in offsets)
    assert sources[3] == sources[6] == sources[5] == sources[8] == "neighbour"
    assert sources[0] == "local"


def test_real_mid_session_clock_step_is_preserved():
    """A step supported by several windows (a clock resynchronisation) must not be averaged away."""
    local = [-5.0, -5.1, -5.0, -5.2, -8.1, -8.0, -8.2, -8.1, -8.0]
    offsets, sources = resolve_window_offsets(local, list(range(9)), session_offset_s=-6.5)
    assert offsets == local and set(sources) == {"local"}


def test_session_offset_is_the_fallback_without_supported_estimates():
    offsets, sources = resolve_window_offsets([None, -3.0, None], [0, 1, 2], session_offset_s=-1.2)
    assert offsets == [-1.2, -1.2, -1.2] and set(sources) == {"session"}
