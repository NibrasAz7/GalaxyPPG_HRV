"""Tests for beat SQI and Lipponen-Tarvainen correction."""

from __future__ import annotations

import numpy as np
import pytest

from src.artifacts import beat_template_sqi, classify_artifacts, correct_beat_times


def _beats(n=300, seed=0):
    rng = np.random.default_rng(seed)
    ibi = 0.8 + 0.04 * np.sin(np.arange(n) / 4) + rng.normal(0, 0.015, n)
    return np.r_[0.0, np.cumsum(ibi)]


def test_no_false_positives_on_clean_series():
    result = correct_beat_times(_beats())
    assert sum(result.counts.values()) == 0
    assert not result.corrected.any()


def test_missed_extra_and_ectopic_are_corrected_conserving_time():
    truth = _beats()
    observed = np.delete(truth, 100)                                   # missed
    observed = np.sort(np.r_[observed, 0.5 * (observed[199] + observed[200]) - 0.05])  # extra
    observed[250] -= 0.15                                               # ectopic
    result = correct_beat_times(observed)
    assert result.counts["missed"] >= 1 and result.counts["extra"] >= 1
    assert len(result.times) == len(truth)                 # beat count restored (bug B7)
    assert result.times[-1] == pytest.approx(observed[-1])  # total time conserved
    assert np.max(np.abs(result.times - truth)) < 0.1


def test_long_gaps_are_never_split():
    beats = _beats()
    beats[150:] += 6.0  # 6-s dropout
    result = correct_beat_times(beats, max_gap_s=3.0)
    assert result.counts["missed"] == 0
    assert np.max(np.diff(result.times)) > 6.0


def test_classify_artifacts_flags_single_missed_beat():
    rr = np.diff(_beats())
    rr[100] += rr[101]
    rr = np.delete(rr, 101)
    found = classify_artifacts(rr)
    assert 100 in found["missed"]


def test_template_sqi_penalises_corrupted_beats():
    fs = 64.0
    t = np.arange(0, 120, 1 / fs)
    signal = np.sin(2 * np.pi * 1.2 * t) ** 3
    peaks = np.round((np.arange(0.2, 119, 1 / 1.2)) * fs).astype(int)
    rng = np.random.default_rng(0)
    corrupted = signal.copy()
    bad = peaks[50]
    corrupted[bad - 20: bad + 20] = rng.normal(0, 1, 40)
    sqi = beat_template_sqi(corrupted, peaks, fs)
    assert np.nanmedian(sqi) > 0.95
    assert sqi[50] < 0.6
