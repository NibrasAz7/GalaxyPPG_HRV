"""Unit tests for accelerometer-referenced adaptive noise cancellation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signal_processing import acc_referenced_anc, normalized_lms


def _config(**anc_overrides) -> dict:
    parameters = {"enabled": True, "order": 32, "mu": 0.01, "delay_ms": 0.0,
                  "reference_band": [0.4, 5.0], "motion_states": ["moderate", "severe"]}
    parameters.update(anc_overrides)
    return {"processing": {"anc": parameters},
            "devices": {"E4": {"fs": 64, "bandpass": [0.5, 4.0], "filter_order": 4}}}


def test_normalized_lms_cancels_correlated_noise():
    rng = np.random.default_rng(0)
    fs = 64.0
    t = np.arange(0, 60, 1 / fs)
    cardiac = np.sin(2 * np.pi * 1.2 * t)
    noise = rng.normal(0, 1, len(t))
    noise = np.convolve(noise, np.ones(5) / 5, mode="same")
    noisy = cardiac + 0.8 * noise

    cleaned = normalized_lms(noisy, noise, order=32, mu=0.01)

    tail = slice(len(t) // 2, None)
    residual_before = np.std(noisy[tail] - cardiac[tail])
    residual_after = np.std(cleaned[tail] - cardiac[tail])
    assert residual_after < 0.75 * residual_before


def test_normalized_lms_short_signal_is_returned_unchanged():
    signal = np.arange(10.0)
    assert np.allclose(normalized_lms(signal, np.ones(10), order=32), signal)


def test_acc_referenced_anc_disabled_returns_input():
    signal = np.arange(100.0)
    config = _config(enabled=False)
    acc = pd.DataFrame({"acc_x_g": np.zeros(100), "acc_y_g": np.zeros(100), "acc_z_g": np.zeros(100)},
                       index=pd.date_range("2024-01-01", periods=100, freq="10ms"))
    result = acc_referenced_anc(signal, acc, acc.index, 100.0, config)
    assert np.allclose(result, signal)


def test_acc_referenced_anc_requires_accelerometer_data():
    signal = np.arange(100.0)
    timestamps = pd.date_range("2024-01-01", periods=100, freq="10ms")
    result = acc_referenced_anc(signal, pd.DataFrame(), timestamps, 100.0, _config())
    assert np.allclose(result, signal)


def test_acc_referenced_anc_reduces_correlated_artifact():
    fs = 64.0
    t = np.arange(0, 40, 1 / fs)
    artifact = np.sin(2 * np.pi * 1.8 * t)
    cardiac = np.sin(2 * np.pi * 1.1 * t)
    signal = cardiac + 1.5 * artifact

    acc = pd.DataFrame({
        "acc_x_g": artifact,
        "acc_y_g": np.zeros_like(artifact),
        "acc_z_g": np.zeros_like(artifact),
    }, index=pd.date_range("2024-01-01", periods=len(t), freq=pd.Timedelta(seconds=1 / fs)))

    cleaned = acc_referenced_anc(signal, acc, acc.index, fs, _config())

    tail = slice(len(t) // 2, None)
    before = np.std(signal[tail] - cardiac[tail])
    after = np.std(cleaned[tail] - cardiac[tail])
    assert after < 0.25 * before


def test_acc_referenced_anc_rejects_missing_columns():
    signal = np.arange(100.0)
    acc = pd.DataFrame({"acc_x_g": np.zeros(100)},
                       index=pd.date_range("2024-01-01", periods=100, freq="10ms"))
    result = acc_referenced_anc(signal, acc, acc.index, 100.0, _config())
    assert np.allclose(result, signal)
