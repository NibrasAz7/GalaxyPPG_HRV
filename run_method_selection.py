#!/usr/bin/env python
"""Detector selection study (development participants only).

Every individual detector (NeuroKit2, SciPy adaptive threshold, slope-sum,
template matching) and the consensus ensemble is scored against the ECG for
every recording of the *development* split, using exactly the validation of
the main pipeline: session clock alignment, the anchored PPG-ECG delay of the
recording, one-to-one beat matching (+-100 ms) and beat-to-beat IBI error.

The priority table in ``config.yaml`` (``detection.priority``) is derived
from this study per device x accelerometer motion state: methods are ranked
by mean F1, ties broken by beat-IBI MAE.  Test participants are never used.

Usage::

    .venv/bin/python run_method_selection.py --data-root Dataset --jobs 16
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

from run_pipeline import process_participant
from src.hrv_metrics import robust_hr
from src.utils import load_config
from src.validation import beat_interval_agreement, estimate_lag, match_peaks

logger = logging.getLogger("method_selection")


def analyse_participant(task: tuple[str, str, dict]) -> list[dict]:
    """Score every detector for every PPG window of one participant.

    Uses the shared :func:`run_pipeline.process_participant`, so clock
    alignment (session + per-window), the ECG reference and the delay prior
    are identical to the main pipeline.  Each detector's delay is
    re-estimated within the window's prior (as the pipeline does for the
    selected detector) before matching.
    """
    data_root, participant, config = task
    out = process_participant(os.path.join(data_root, participant), config)
    if not out:
        return []
    tolerance = float(config.get("validation", {}).get("match_tolerance_s", 0.10))
    rows: list[dict] = []
    for (device, activity), (results, extras) in out["ppg"].items():
        if activity not in out["reference"] or "lag_prior" not in extras:
            continue
        ecg_s = out["reference"][activity]["beats_s"]
        center, halfwidth = extras["lag_prior"]
        seconds = extras["clock_seconds"]
        for method, peaks in extras.get("method_peaks", {}).items():
            if method.endswith("_all") or len(peaks) < 3:
                continue
            beats = extras["clock_t0_s"] + np.interp(np.asarray(peaks, dtype=float),
                                                     np.arange(len(seconds)), seconds)
            lag = estimate_lag(ecg_s, beats, max_lag_s=halfwidth, center_s=center)["lag_s"]
            score = match_peaks(ecg_s, beats, tolerance, lag)
            ibi = beat_interval_agreement(ecg_s, beats, tolerance, lag)
            hr = robust_hr(np.diff(beats) * 1000.0)
            rows.append({
                "participant": participant, "device": device, "activity": activity,
                "motion_level": results.get("motion_level"), "method": method,
                "selected": method == extras.get("method_used"), "lag_source": results.get("lag_source"),
                "recall": score["recall"], "precision": score["precision"], "f1": score["f1"],
                "timing_sd_ms": score["timing_sd_ms"],
                "beat_ibi_mae_ms": ibi.get("mae", np.nan),
                "hr_error": abs(hr - out["reference"][activity]["hr_median"]) if np.isfinite(hr) else np.nan,
            })
    return rows


def derive_priority(summary: pd.DataFrame, current: dict) -> dict:
    """Rank detectors per device x motion by F1 (ties: lower beat-IBI MAE)."""
    table = {device: dict(states) for device, states in current.items()}
    for (device, motion), group in summary.groupby(["device", "motion_level"]):
        ranked = group.sort_values(["f1", "beat_ibi_mae_ms"], ascending=[False, True])["method"].tolist()
        table.setdefault(device, {})[motion] = ranked
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Detector selection study (dev split)")
    parser.add_argument("--data-root", default="Dataset")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="paper_results/method_selection.csv")
    parser.add_argument("--summary", default="paper_results/tables/method_selection_summary.csv")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_config(args.config)
    evaluation = config.get("evaluation", {})
    if args.split == "dev":
        participants = evaluation["dev_participants"]
    elif args.split == "test":
        participants = evaluation["test_participants"]
    else:
        participants = evaluation["dev_participants"] + evaluation["test_participants"]
    participants = [p for p in participants
                    if os.path.exists(os.path.join(args.data_root, p, "Event.csv"))]

    tasks = [(args.data_root, p, config) for p in participants]
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            chunks = list(pool.map(analyse_participant, tasks))
    else:
        chunks = [analyse_participant(t) for t in tasks]
    frame = pd.DataFrame(row for chunk in chunks for row in chunk)
    frame["split"] = args.split
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    frame.to_csv(args.output, index=False)

    summary = frame.groupby(["device", "motion_level", "method"], observed=True).agg(
        n=("f1", "size"), recall=("recall", "mean"), precision=("precision", "mean"),
        f1=("f1", "mean"), beat_ibi_mae_ms=("beat_ibi_mae_ms", "median"),
        timing_sd_ms=("timing_sd_ms", "median"), hr_mae=("hr_error", "mean"),
        selected_pct=("selected", lambda s: 100.0 * s.mean()),
    ).round(2).reset_index()
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    summary.to_csv(args.summary, index=False)
    print(summary.to_string(index=False))
    priority = derive_priority(summary, config.get("detection", {}).get("priority", {}))
    print("\nDerived detection.priority (" + args.split + "):")
    print(yaml.safe_dump({"priority": priority}, default_flow_style=None, sort_keys=False))


if __name__ == "__main__":
    main()
