#!/usr/bin/env python
"""Tune the accelerometer-referenced adaptive noise canceller (ANC).

Performs a coarse-to-fine sweep of the NLMS parameters (step size, filter
order, reference delay) on a small tuning subset and reports HR accuracy,
peak recall and timing error for every candidate.  The final candidate is then
validated on walking *and* jogging so that a walking-only gain cannot silently
degrade other conditions.

Usage::

    uv run python run_anc_tuning.py
    uv run python run_anc_tuning.py --participants P04 P10 P11 P21 --jobs 4
"""

from __future__ import annotations

import argparse
import copy
import logging
import os

import numpy as np
import pandas as pd

from analysis.common import PPG_DEVICES
from src.data_loader import load_device_data
from src.hrv_metrics import calculate_ibi, robust_hr
from src.motion_detection import calculate_motion
from src.peak_detection import multi_method_peak_detection, refine_peaks_parabolic, select_optimal_peaks
from src.signal_processing import acc_referenced_anc, adaptive_bandpass_filter, estimate_signal_quality
from src.utils import load_config
from src.validation import match_peaks

logger = logging.getLogger(__name__)

TUNING_ACTIVITY = "walking"
VALIDATION_ACTIVITIES = ["walking", "jogging"]


def activity_window(participant: str, activity: str, data_root: str):
    """Return the ENTER/EXIT window of an activity for one participant."""
    events = pd.read_csv(os.path.join(data_root, participant, "Event.csv"))
    events["datetime"] = pd.to_datetime(events["timestamp"], unit="ms")
    session = events[events["session"] == activity]
    return session[session["status"] == "ENTER"]["datetime"].min(), \
        session[session["status"] == "EXIT"]["datetime"].max()


def build_chain_cache(
    participants: list[str],
    activities: list[str],
    data_root: str,
    config: dict,
) -> list[dict]:
    """Load and pre-compute everything that does not depend on ANC parameters."""
    cache: list[dict] = []
    for participant in participants:
        for activity in activities:
            try:
                start, end = activity_window(participant, activity, data_root)
            except (FileNotFoundError, KeyError):
                logger.warning("%s/%s: no event window", participant, activity)
                continue

            reference = _ecg_reference(participant, activity, data_root, config, start, end)
            if reference is None:
                logger.warning("%s/%s: no ECG reference", participant, activity)
                continue

            for device in PPG_DEVICES:
                device_data = load_device_data(os.path.join(data_root, participant), device, config)
                if "signal" not in device_data:
                    continue
                segment = device_data["signal"][start:end]
                if len(segment) < 100:
                    continue

                fs = config["devices"][device]["fs"]
                raw = np.nan_to_num(segment[config["devices"][device]["signal_col"]].values)
                clean, _, motion_metrics = calculate_motion(
                    device_data.get("acc", pd.DataFrame()), segment.index, device, config
                )
                cache.append({
                    "participant": participant,
                    "activity": activity,
                    "device": device,
                    "fs": fs,
                    "raw": raw,
                    "acc": device_data.get("acc", pd.DataFrame()),
                    "timestamps": segment.index,
                    "clean": clean,
                    "motion_level": motion_metrics["motion_level"],
                    "quality": estimate_signal_quality(raw, fs),
                    "ecg_hr": reference["hr"],
                    "ecg_peaks": reference["peaks"],
                })
    return cache


def _ecg_reference(participant, activity, data_root, config, start, end):
    """Detect ECG reference peaks and HR for one participant and activity."""
    device_data = load_device_data(os.path.join(data_root, participant), "PolarH10", config)
    if "signal" not in device_data:
        return None
    segment = device_data["signal"][start:end]
    fs = config["devices"]["PolarH10"]["fs"]
    raw = np.nan_to_num(segment[config["devices"]["PolarH10"]["signal_col"]].values)
    filtered = adaptive_bandpass_filter(raw, "PolarH10", "severe", activity, config)
    peaks = multi_method_peak_detection(
        filtered, "PolarH10", "severe", np.ones(len(raw), bool), {"snr_db": 0}, activity, config
    )
    selected, _ = select_optimal_peaks(peaks, "PolarH10", "severe", {"snr_db": 0}, activity)
    if len(selected) < 5:
        return None
    positions = refine_peaks_parabolic(filtered, selected)
    peak_times = segment.index[0] + pd.to_timedelta(positions / fs, unit="s")
    return {"hr": robust_hr(calculate_ibi(positions, fs)), "peaks": peak_times}


def evaluate_candidate(cache: list[dict], config: dict) -> pd.DataFrame:
    """Run the detection chain for every cached recording with one ANC config."""
    rows = []
    for item in cache:
        signal = item["raw"]
        anc_config = config["processing"].get("anc", {})
        states = anc_config.get("motion_states", ["moderate", "severe"])
        if anc_config.get("enabled", False) and item["motion_level"] in states:
            signal = acc_referenced_anc(
                signal, item["acc"], item["timestamps"], item["fs"], config
            )

        filtered = adaptive_bandpass_filter(
            signal, item["device"], item["motion_level"], item["activity"], config
        )
        results = multi_method_peak_detection(
            filtered, item["device"], item["motion_level"], item["clean"],
            item["quality"], item["activity"], config,
        )
        peaks, _ = select_optimal_peaks(
            results, item["device"], item["motion_level"], item["quality"], item["activity"]
        )
        if len(peaks) < 5:
            continue
        positions = refine_peaks_parabolic(filtered, peaks)
        peak_times = item["timestamps"][0] + pd.to_timedelta(positions / item["fs"], unit="s")
        hr = robust_hr(calculate_ibi(positions, item["fs"]))
        matched = match_peaks(item["ecg_peaks"], peak_times, tolerance_s=0.3)

        rows.append({
            "participant": item["participant"],
            "activity": item["activity"],
            "device": item["device"],
            "motion_level": item["motion_level"],
            "hr": hr,
            "ecg_hr": item["ecg_hr"],
            "hr_error": abs(hr - item["ecg_hr"]),
            "recall": matched["recall"],
            "timing_ms": matched["median_absolute_offset_ms"],
        })
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame) -> dict:
    """Aggregate one candidate's per-recording results."""
    if frame.empty:
        return {"n": 0, "hr_mae": np.nan, "recall": np.nan, "timing_ms": np.nan}
    return {
        "n": int(len(frame)),
        "hr_mae": float(frame["hr_error"].mean()),
        "recall": float(frame["recall"].mean()),
        "timing_ms": float(frame["timing_ms"].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="GalaxyPPG HRV — ANC parameter tuning")
    parser.add_argument("--participants", nargs="+",
                        default=["P04", "P10", "P11", "P21", "P02", "P17"])
    parser.add_argument("--data-root", default="Dataset")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="paper_results/anc_tuning.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    base_config = load_config(args.config)

    logger.info("Building cache for %d participants", len(args.participants))
    tuning_cache = build_chain_cache(args.participants, [TUNING_ACTIVITY], args.data_root, base_config)
    validation_cache = build_chain_cache(
        args.participants, VALIDATION_ACTIVITIES, args.data_root, base_config
    )
    logger.info("Cache: %d tuning recordings, %d validation recordings",
                len(tuning_cache), len(validation_cache))

    def variant(**overrides) -> dict:
        config = copy.deepcopy(base_config)
        config["processing"]["anc"].update(overrides)
        return config

    candidates: list[tuple[str, dict]] = [("off", variant(enabled=False))]

    # Stage 1: step size
    step_sizes = [0.002, 0.005, 0.01, 0.02, 0.05]
    for mu in step_sizes:
        candidates.append((f"mu={mu}", variant(enabled=True, mu=mu)))

    results = []
    for label, config in candidates:
        summary = summarise(evaluate_candidate(tuning_cache, config))
        results.append({"stage": "mu", "candidate": label, **summary})
        logger.info("stage=mu %-12s %s", label, summary)

    stage1 = pd.DataFrame(results)
    best_mu = stage1[stage1["candidate"] != "off"].sort_values(["hr_mae", "recall"],
                                                              ascending=[True, False]).iloc[0]["candidate"]
    best_mu_value = float(best_mu.split("=")[1])
    logger.info("Best step size: %s", best_mu)

    # Stage 2: filter order
    for order in [8, 16, 32, 64]:
        config = variant(enabled=True, mu=best_mu_value, order=order)
        summary = summarise(evaluate_candidate(tuning_cache, config))
        results.append({"stage": "order", "candidate": f"order={order}", **summary})
        logger.info("stage=order %-12s %s", f"order={order}", summary)

    stage2 = pd.DataFrame([r for r in results if r["stage"] == "order"])
    best_order = int(stage2.sort_values(["hr_mae", "recall"], ascending=[True, False]).iloc[0]["candidate"].split("=")[1])
    logger.info("Best order: %d", best_order)

    # Stage 3: reference delay
    for delay_ms in [0.0, 40.0, 80.0, 150.0, 250.0]:
        config = variant(enabled=True, mu=best_mu_value, order=best_order, delay_ms=delay_ms)
        summary = summarise(evaluate_candidate(tuning_cache, config))
        results.append({"stage": "delay", "candidate": f"delay={delay_ms}", **summary})
        logger.info("stage=delay %-12s %s", f"delay={delay_ms}", summary)

    stage3 = pd.DataFrame([r for r in results if r["stage"] == "delay"])
    best_delay = float(stage3.sort_values(["hr_mae", "recall"], ascending=[True, False]).iloc[0]["candidate"].split("=")[1])
    logger.info("Best delay: %.0f ms", best_delay)

    # Final validation on walking + jogging
    final_config = variant(enabled=True, mu=best_mu_value, order=best_order, delay_ms=best_delay)
    baseline_frame = evaluate_candidate(validation_cache, variant(enabled=False))
    final_frame = evaluate_candidate(validation_cache, final_config)

    for activity in VALIDATION_ACTIVITIES:
        baseline_activity = baseline_frame[baseline_frame["activity"] == activity]
        final_activity = final_frame[final_frame["activity"] == activity]
        results.append({"stage": f"validation:{activity}", "candidate": "off",
                        **summarise(baseline_activity)})
        results.append({"stage": f"validation:{activity}",
                        "candidate": f"mu={best_mu_value},order={best_order},delay={best_delay}",
                        **summarise(final_activity)})

    table = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    table.to_csv(args.output, index=False)
    logger.info("Wrote tuning results → %s", args.output)

    validation = table[table["stage"].str.startswith("validation")]
    print(validation.to_string(index=False))
    print()
    print(f"Suggested config: anc: enabled: true, mu: {best_mu_value}, "
          f"order: {best_order}, delay_ms: {best_delay}")


if __name__ == "__main__":
    main()
