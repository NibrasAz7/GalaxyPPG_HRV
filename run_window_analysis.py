#!/usr/bin/env python
"""Window-length study: HRV agreement versus analysis window length.

Accepted PPG NN segments (main pipeline, all quality gates) are cut into
non-overlapping sub-windows of 30, 60 and 120 s.  For every sub-window the
same indices are computed from the ECG over the *identical* span (shifted by
the recording's PPG-ECG delay), so the comparison isolates the effect of
window length on agreement.  Errors are averaged per participant before
summarising, so that participants with more windows do not dominate
(no pseudo-replication).

Outputs::

    paper_results/window_analysis/windows.csv          one row per sub-window
    paper_results/window_analysis/window_summary.csv   split x device x length x metric

Usage::

    .venv/bin/python run_window_analysis.py --jobs 20
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from run_all_participants import discover_participants, participant_split
from run_pipeline import run_pipeline
from src.hrv_metrics import segment_metrics
from src.utils import load_config
from src.validation import to_seconds

logger = logging.getLogger("window_analysis")

LENGTHS_S = (30.0, 60.0, 120.0)
METRICS = ("mean_nn", "sdnn", "rmssd", "sd1", "hf_power", "sampen")


def _windows_for_participant(task: tuple[str, str, dict]) -> list[dict]:
    data_root, participant, config = task
    frame, reference, extras = run_pipeline(os.path.join(data_root, participant), config,
                                            return_extras=True)
    if frame.empty:
        return []
    split = participant_split(participant, config)
    rows: list[dict] = []
    ppg = frame[frame["device"] != "PolarH10"].set_index(["device", "activity"])
    for (device, activity), ext in extras.items():
        if activity not in reference or (device, activity) not in ppg.index:
            continue
        lag_ms = ppg.loc[(device, activity), "lag_ms"]
        lag = float(lag_ms) / 1000.0 if np.isfinite(lag_ms) else 0.0
        ecg_beats = reference[activity]["beats_s"]
        for segment in ext.get("segments", []):
            end_s = to_seconds(segment["interval_end_times"])
            for length in LENGTHS_S:
                start = end_s[0] - segment["nn"][0] / 1000.0
                while start + length <= end_s[-1] + 1e-9:
                    stop = start + length
                    sel = (end_s > start) & (end_s <= stop)
                    nn = segment["nn"][sel]
                    synthetic = segment.get("synthetic")
                    synthetic = synthetic[sel] if synthetic is not None else None
                    e0 = np.searchsorted(ecg_beats, start - lag - 0.15)
                    e1 = np.searchsorted(ecg_beats, stop - lag + 0.15, side="right") - 1
                    ecg_nn = np.diff(ecg_beats[e0:e1 + 1]) * 1000.0
                    if len(nn) >= 10 and len(ecg_nn) >= 10 and np.all((ecg_nn > 300) & (ecg_nn < 2000)):
                        own = segment_metrics(nn, config, synthetic)
                        ref = segment_metrics(ecg_nn, config)
                        row = {"participant": participant, "split": split, "device": device,
                               "activity": activity, "window_s": length}
                        for metric in METRICS:
                            row[f"ppg_{metric}"] = own.get(metric, np.nan)
                            row[f"ecg_{metric}"] = ref.get(metric, np.nan)
                        rows.append(row)
                    start = stop
    return rows


def summarise(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    splits = {"all": windows, **{s: windows[windows["split"] == s] for s in ("dev", "test")}}
    for split, data in splits.items():
        for (device, length), g in data.groupby(["device", "window_s"]):
            for metric in METRICS:
                own, ref = g[f"ppg_{metric}"], g[f"ecg_{metric}"]
                ok = np.isfinite(own) & np.isfinite(ref)
                if metric != "sampen":
                    ok &= (own > 0) & (ref > 0)
                h = g[ok]
                if h.empty:
                    continue
                if metric in ("mean_nn", "sampen"):
                    err = (h[f"ppg_{metric}"] - h[f"ecg_{metric}"]).abs()
                    label = "abs_error"
                else:
                    err = np.log(h[f"ppg_{metric}"] / h[f"ecg_{metric}"]).abs()
                    label = "abs_log_ratio"
                per_participant = err.groupby(h["participant"]).mean()
                rows.append({
                    "split": split, "device": device, "window_s": length, "metric": metric,
                    "error_type": label, "n_windows": int(len(h)),
                    "n_participants": int(per_participant.size),
                    "error_median": float(per_participant.median()),
                    "error_q25": float(per_participant.quantile(0.25)),
                    "error_q75": float(per_participant.quantile(0.75)),
                    "within_10pct": float(100 * (err <= np.log(1.1)).mean()) if label == "abs_log_ratio" else np.nan,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="HRV agreement vs window length")
    parser.add_argument("--data-root", default="Dataset")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="paper_results/window_analysis")
    parser.add_argument("--jobs", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    config = load_config(args.config)
    participants = [p for p in discover_participants(args.data_root)
                    if os.path.exists(os.path.join(args.data_root, p, "Event.csv"))]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        chunks = list(pool.map(_windows_for_participant, [(args.data_root, p, config) for p in participants]))
    windows = pd.DataFrame(row for chunk in chunks for row in chunk)
    os.makedirs(args.output_dir, exist_ok=True)
    windows.to_csv(os.path.join(args.output_dir, "windows.csv"), index=False)
    summary = summarise(windows)
    summary.to_csv(os.path.join(args.output_dir, "window_summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary[summary["split"] == "all"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
