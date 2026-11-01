#!/usr/bin/env python
"""PPG fiducial-point study (development participants, resting windows).

For the beats selected by the pipeline, five fiducial definitions are
located on the filtered PPG and compared with the ECG R-peaks after removing
the recording's delay: parabolic systolic peak (pipeline default),
spline-refined systolic peak, maximum first derivative (max slope),
tangent-intersection foot, and maximum second derivative.  Reported per
window: beat recall (+-100 ms), timing SD, beat-to-beat IBI MAE and the
RMSSD ratio over matched intervals.

Usage::

    .venv/bin/python run_fiducial_study.py --jobs 15
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from analysis.common import CONDITIONS
from run_pipeline import process_participant
from src.peak_detection import FIDUCIALS, locate_fiducial
from src.utils import load_config
from src.validation import beat_interval_agreement, estimate_lag, match_peaks

REST = CONDITIONS["rest"] + ["standing"]
ANCHOR = "systolic"


def locate_fiducials(signal: np.ndarray, peaks: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Positions of every fiducial (see :func:`src.peak_detection.locate_fiducial`)."""
    return {kind: locate_fiducial(signal, peaks, fs, kind, anchor=ANCHOR) for kind in FIDUCIALS}


def analyse(task: tuple[str, str, dict]) -> list[dict]:
    """Fiducial comparison on the shared pipeline outputs of one participant."""
    data_root, participant, config = task
    global ANCHOR
    ANCHOR = config.get("processing", {}).get("timing", {}).get("anchor", "systolic")
    out = process_participant(os.path.join(data_root, participant), config, activities=REST)
    if not out:
        return []
    tolerance = float(config["validation"]["match_tolerance_s"])
    rows = []
    for (device, activity), (_, extras) in out["ppg"].items():
        if activity not in out["reference"] or "lag_s" not in extras:
            continue
        fs = float(config["devices"][device]["fs"])
        filtered = extras["signal_timing"]  # the signal beat times are taken from
        peaks = np.asarray(extras["timing_peaks"])
        seconds = extras["clock_seconds"]
        ecg = out["reference"][activity]["beats_s"]
        for name, positions in locate_fiducials(filtered, peaks, fs).items():
            ok = np.isfinite(positions)
            beats = extras["clock_t0_s"] + np.interp(positions[ok], np.arange(len(seconds)), seconds)
            # The delay differs per fiducial (the foot precedes the peak),
            # so it is re-estimated around the pipeline delay.
            fiducial_lag = estimate_lag(ecg, beats, max_lag_s=0.4, center_s=extras["lag_s"])["lag_s"]
            match = match_peaks(ecg, beats, tolerance, fiducial_lag)
            ibi = beat_interval_agreement(ecg, beats, tolerance, fiducial_lag)
            if ibi.get("n", 0) < 20:
                continue
            ratio = (np.sqrt(np.mean(np.diff(ibi["ibi_detected"]) ** 2))
                     / np.sqrt(np.mean(np.diff(ibi["ibi_reference"]) ** 2)))
            rows.append({"participant": participant, "device": device, "activity": activity,
                         "fiducial": name, "recall": match["recall"], "timing_sd_ms": match["timing_sd_ms"],
                         "ibi_mae_ms": ibi["mae"], "ibi_sd_ms": ibi["sd"], "rmssd_ratio": ratio,
                         "delay_ms": fiducial_lag * 1000.0})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="PPG fiducial-point study (dev split)")
    parser.add_argument("--data-root", default="Dataset")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="paper_results/fiducial_study.csv")
    parser.add_argument("--jobs", type=int, default=15)
    parser.add_argument("--anchor", choices=["systolic", "onset", "both"], default="both",
                        help="Beat detection anchor(s) to evaluate")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    config = load_config(args.config)
    participants = [p for p in config["evaluation"]["dev_participants"]
                    if os.path.exists(os.path.join(args.data_root, p, "Event.csv"))]
    import copy
    frames = []
    for anchor in (["systolic", "onset"] if args.anchor == "both" else [args.anchor]):
        cfg = copy.deepcopy(config)
        cfg.setdefault("processing", {}).setdefault("timing", {})["anchor"] = anchor
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            chunks = list(pool.map(analyse, [(args.data_root, p, cfg) for p in participants]))
        frames.append(pd.DataFrame(r for chunk in chunks for r in chunk).assign(anchor=anchor))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(args.output, index=False)
    summary = frame.groupby(["device", "anchor", "fiducial"])[
        ["recall", "timing_sd_ms", "ibi_mae_ms", "rmssd_ratio", "delay_ms"]].median()
    summary.round(2).to_csv(os.path.join(os.path.dirname(args.output), "tables", "fiducial_summary.csv"))
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
