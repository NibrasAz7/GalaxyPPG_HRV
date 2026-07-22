#!/usr/bin/env python
"""Single-participant entry point for the GalaxyPPG HRV pipeline.

Processes one participant through the full pipeline:

1. Load Event.csv to get activity sessions
2. Process PolarH10 (ECG) as ground truth for all activities
3. Process E4 and GalaxyWatch (PPG) against ECG reference
4. Save results to ``paper_results/<participant>_results.csv``

Usage::

    python run_pipeline.py --participant P02
    python run_pipeline.py --participant P02 --data-root /path/to/GalaxyPPG/Dataset
    python run_pipeline.py --participant P02 --device E4 --activity baseline
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config
from src.data_loader import load_device_data, extract_device_ibis
from src.signal_processing import adaptive_bandpass_filter, estimate_signal_quality
from src.motion_detection import calculate_motion
from src.peak_detection import multi_method_peak_detection, select_optimal_peaks
from src.hrv_metrics import calculate_ibi, compute_rolling_hr, compute_hrv_metrics
from src.validation import validate_peaks_physiologically, calculate_pmr, bland_altman_analysis

logger = logging.getLogger(__name__)


def _estimate_seed_hr(device_data: dict, start: pd.Timestamp, end: pd.Timestamp, fallback: float = 80.0) -> float:
    """Estimate seed heart rate from device-reported HR data.

    Used to set a dynamic refractory period for peak detection.

    Args:
        device_data: Output of ``load_device_data``.
        start: Window start time.
        end: Window end time.
        fallback: Default HR if insufficient data.

    Returns:
        Median HR in bpm, or *fallback*.
    """
    hr_df = device_data.get("hr", pd.DataFrame())
    if hr_df.empty:
        return fallback
    seg = hr_df[start:end]
    vals = seg["hr"].values if "hr" in seg.columns else np.array([])
    vals = vals[(vals > 30) & (vals < 220)]
    return float(np.median(vals)) if len(vals) >= 5 else fallback


def process_activity(
    user_folder: str,
    device: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    activity_name: str,
    config: dict,
    ecg_reference: dict | None = None,
) -> dict:
    """Process one device × activity combination through the full pipeline.

    Orchestrates all stages: data loading → motion detection → filtering →
    peak detection → IBI calculation → HRV metrics → validation.

    Args:
        user_folder: Path to participant data directory.
        device: Device identifier (``"E4"``, ``"GalaxyWatch"``, ``"PolarH10"``).
        start_time: Activity start timestamp.
        end_time: Activity end timestamp.
        activity_name: Activity name from Event.csv.
        config: Parsed configuration dictionary.
        ecg_reference: Optional dict of ECG ground-truth HR/IBIs per activity.

    Returns:
        Results dictionary (empty if processing fails).
    """
    dev_cfg = config["devices"][device]
    fs = dev_cfg["fs"]
    signal_col = dev_cfg["signal_col"]

    # Load device data
    device_data = load_device_data(user_folder, device, config)
    if "signal" not in device_data:
        logger.warning("%s: no signal data found", device)
        return {}

    # Extract signal segment
    signal_segment = device_data["signal"][start_time:end_time]
    if len(signal_segment) < fs * 5:
        logger.warning("%s: insufficient data (%d samples)", device, len(signal_segment))
        return {}

    signal_data = np.nan_to_num(signal_segment[signal_col].values)

    # Motion analysis
    clean_mask, motion_levels, motion_metrics = calculate_motion(
        device_data.get("acc", pd.DataFrame()), signal_segment.index, device, config
    )

    # Signal quality
    quality_metrics = estimate_signal_quality(signal_data, fs)
    seed_hr = _estimate_seed_hr(device_data, start_time, end_time, 80.0)
    quality_metrics["seed_hr"] = seed_hr

    # Adaptive filtering
    signal_filtered = adaptive_bandpass_filter(
        signal_data, device, motion_metrics["motion_level"], activity_name, config
    )

    # Peak detection
    peak_results = multi_method_peak_detection(
        signal_filtered, device, motion_metrics["motion_level"],
        clean_mask, quality_metrics, activity_name, config
    )
    optimal_peaks, method_used = select_optimal_peaks(
        peak_results, device, motion_metrics["motion_level"],
        quality_metrics, activity_name
    )
    validated_peaks = validate_peaks_physiologically(optimal_peaks, fs, config)

    # Custom IBIs
    custom_ibis = np.array([])
    custom_ibi_times = np.array([])
    if len(validated_peaks) >= 2:
        ibis = calculate_ibi(validated_peaks, fs)
        ibi_times = signal_segment.index[validated_peaks[1:]]
        ibi_range = config["processing"]["ibi_range"]
        valid = (ibis >= ibi_range[0]) & (ibis <= ibi_range[1])
        custom_ibis = ibis[valid]
        custom_ibi_times = ibi_times[valid]

    # Device IBIs
    device_ibis, device_ibi_times = extract_device_ibis(
        device_data.get("ibi", pd.DataFrame()), device, start_time, end_time, config
    )

    # Assemble results
    results = {
        "device": device,
        "activity": activity_name,
        "method_used": method_used,
        "duration_sec": (end_time - start_time).total_seconds(),
        "n_peaks_detected": len(validated_peaks),
        "n_peaks_raw": {k: len(v) for k, v in peak_results.items()},
        **motion_metrics,
        **quality_metrics,
    }

    # Custom HR / HRV
    if len(custom_ibis) > 0:
        results["custom_hr"] = 60000 / np.mean(custom_ibis)
        results["custom_mean_ibi"] = float(np.mean(custom_ibis))
        results["custom_n_ibis"] = len(custom_ibis)
        hrv = compute_hrv_metrics(custom_ibis)
        results["custom_sdnn"] = hrv["sdnn"]
        results["custom_rmssd"] = hrv["rmssd"]

    # Device HR
    if "hr" in device_data and not device_data["hr"].empty:
        hr_seg = device_data["hr"][start_time:end_time]
        if not hr_seg.empty:
            hr_vals = hr_seg["hr"].values
            valid_hr = hr_vals[(hr_vals > 0) & (hr_vals < 300)]
            if len(valid_hr) > 0:
                results["device_hr"] = float(np.mean(valid_hr))
                results["device_hr_std"] = float(np.std(valid_hr))

    if len(device_ibis) > 0:
        results["device_mean_ibi"] = float(np.mean(device_ibis))
        results["device_hr_from_ibi"] = 60000 / np.mean(device_ibis)
        results["device_sdnn"] = float(np.std(device_ibis))
        results["device_n_ibis"] = len(device_ibis)

    # HR error vs device
    if "custom_hr" in results and "device_hr" in results:
        results["hr_error"] = abs(results["custom_hr"] - results["device_hr"])
        results["hr_error_percent"] = results["hr_error"] / results["device_hr"] * 100

    # ECG ground truth comparison
    if ecg_reference and activity_name in ecg_reference:
        ecg_hr = ecg_reference[activity_name]["hr"]
        ecg_ibis = ecg_reference[activity_name]["ibis"]

        if "custom_hr" in results:
            results["hr_error_vs_ecg"] = abs(results["custom_hr"] - ecg_hr)
            results["hr_error_percent_vs_ecg"] = results["hr_error_vs_ecg"] / ecg_hr * 100

        if len(custom_ibis) > 2 and len(ecg_ibis) > 2:
            ba = bland_altman_analysis(custom_ibis, ecg_ibis)
            results["bland_altman_vs_ecg"] = ba
            if ba:
                results["ba_correlation_vs_ecg"] = ba["correlation"]

    # PMR
    if len(device_ibi_times) > 1 and len(validated_peaks) > 1:
        results["pmr"] = calculate_pmr(
            validated_peaks, device_ibi_times, signal_segment.index.values, config
        )

    # Bland-Altman vs device
    if len(custom_ibis) > 2 and len(device_ibis) > 2:
        ba = bland_altman_analysis(custom_ibis, device_ibis)
        results["bland_altman"] = ba
        if ba:
            results["ba_mean_diff"] = ba["mean_difference"]
            results["ba_std_diff"] = ba["std_difference"]
            results["ba_correlation"] = ba["correlation"]

    # Rolling HR trend
    if len(custom_ibis) >= 2:
        hr_ts = compute_rolling_hr(custom_ibis, custom_ibi_times, activity_name, device, config)
        if hr_ts:
            results["hr_timeseries"] = hr_ts

    logger.info(
        "%s %s: method=%s peaks=%d motion=%s snr=%.1f dB",
        device, activity_name, method_used, len(validated_peaks),
        motion_metrics["motion_level"], quality_metrics["snr_db"],
    )

    return results


def run_pipeline(user_folder: str, config: dict) -> tuple[pd.DataFrame, dict]:
    """Run the full validation pipeline for one participant.

    Processes all activities from ``Event.csv``, with ECG (PolarH10) as
    ground truth, then E4 and GalaxyWatch as PPG devices under test.

    Args:
        user_folder: Path to participant data directory.
        config: Parsed configuration dictionary.

    Returns:
        Tuple of ``(results_df, ecg_reference)``.
    """
    events_path = os.path.join(user_folder, "Event.csv")
    if not os.path.exists(events_path):
        logger.error("Events file not found: %s", events_path)
        return pd.DataFrame(), {}

    events = pd.read_csv(events_path)
    events["datetime"] = pd.to_datetime(events["timestamp"], unit="ms")

    activities = config.get("activities", ["baseline", "keyboard-typing", "rest-1", "walking", "jogging"])
    all_results: list[dict] = []
    ecg_reference: dict = {}

    # First pass: ECG ground truth
    logger.info("Processing ECG ground truth (PolarH10)")
    for activity in activities:
        session = events[events["session"] == activity].sort_values("datetime")
        enter = session[session["status"] == "ENTER"]
        exit_ = session[session["status"] == "EXIT"]
        if len(enter) == 0 or len(exit_) == 0:
            continue

        start_time = enter.iloc[0]["datetime"]
        end_time = exit_.iloc[0]["datetime"]

        ecg_results = process_activity(user_folder, "PolarH10", start_time, end_time, activity, config)
        if ecg_results and "custom_hr" in ecg_results:
            ecg_reference[activity] = {
                "hr": ecg_results["custom_hr"],
                "ibis": ecg_results.get("custom_ibis", []),
                "quality": ecg_results.get("snr_db", 0),
            }
            all_results.append(ecg_results)

    # Second pass: PPG devices
    logger.info("Processing PPG devices (E4, GalaxyWatch)")
    for activity in activities:
        session = events[events["session"] == activity].sort_values("datetime")
        enter = session[session["status"] == "ENTER"]
        exit_ = session[session["status"] == "EXIT"]
        if len(enter) == 0 or len(exit_) == 0:
            continue

        start_time = enter.iloc[0]["datetime"]
        end_time = exit_.iloc[0]["datetime"]

        for device in ["E4", "GalaxyWatch"]:
            results = process_activity(
                user_folder, device, start_time, end_time, activity, config, ecg_reference
            )
            if results:
                all_results.append(results)

    return pd.DataFrame(all_results), ecg_reference


def main():
    parser = argparse.ArgumentParser(description="GalaxyPPG HRV Pipeline — single participant")
    parser.add_argument("--participant", required=True, help="Participant ID (e.g. P02)")
    parser.add_argument("--data-root", default=None, help="Dataset root directory")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    config_path = args.config or os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)

    data_root = args.data_root or os.environ.get("PPG_DATA_ROOT") or config["dataset"]["root"]
    if not data_root:
        logger.error("No data root specified. Use --data-root, PPG_DATA_ROOT env var, or config.yaml.")
        sys.exit(1)

    user_folder = os.path.join(data_root, args.participant)
    if not os.path.exists(user_folder):
        logger.error("Participant folder not found: %s", user_folder)
        sys.exit(1)

    logger.info("Processing participant %s from %s", args.participant, user_folder)
    results_df, _ = run_pipeline(user_folder, config)

    if results_df.empty:
        logger.warning("No results generated for %s", args.participant)
        sys.exit(1)

    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{args.participant}_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info("Results saved to %s (%d rows)", csv_path, len(results_df))


if __name__ == "__main__":
    main()
