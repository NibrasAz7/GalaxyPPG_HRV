"""Unit tests for the pipeline-level NeuroKit2 input gate."""

from __future__ import annotations

import numpy as np

from run_pipeline import _raw_neurokit2_signal
from src.utils import load_config


def _gate_config():
    return {
        "processing": {
            "neurokit2_raw": {
                "enabled": True,
                "devices": ["E4"],
                "motion_states": ["severe"],
            }
        }
    }


def test_raw_neurokit2_gate_matches_device_and_motion():
    raw = np.arange(10.0)
    config = _gate_config()

    assert _raw_neurokit2_signal("E4", "severe", raw, config) is raw
    assert _raw_neurokit2_signal("E4", "low", raw, config) is None
    assert _raw_neurokit2_signal("GalaxyWatch", "severe", raw, config) is None


def test_raw_neurokit2_gate_disabled():
    raw = np.arange(10.0)
    config = _gate_config()
    config["processing"]["neurokit2_raw"]["enabled"] = False

    assert _raw_neurokit2_signal("E4", "severe", raw, config) is None


def test_multi_method_honours_neurokit2_signal_override(monkeypatch):
    import src.peak_detection as peak_module

    config = load_config("config.yaml")
    captured: dict = {}

    def fake_detector(signal, fs, device, motion_class=None):
        captured["signal"] = np.asarray(signal)
        return np.arange(20, 900, 40)

    monkeypatch.setattr(peak_module, "detect_peaks_neurokit2", fake_detector)

    filtered = np.sin(np.linspace(0.0, 50.0, 2000))
    override = np.full(2000, 0.5)
    results = peak_module.multi_method_peak_detection(
        filtered, "E4", "severe", np.ones(2000, dtype=bool),
        {"snr_db": 10.0}, "jogging", config, neurokit2_signal=override,
    )

    assert np.array_equal(captured["signal"], override)
    assert "neurokit2" in results
