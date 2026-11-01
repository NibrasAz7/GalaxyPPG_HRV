#!/usr/bin/env python
"""Evaluate peak-detection variants (rhythm-constrained selection).

Runs the full pipeline for each variant on a tuning subset and reports HR
accuracy, peak recall and peak timing error per activity, including resting
activities as a regression check.

Usage::

    uv run python run_detector_tuning.py --jobs 5
    uv run python run_detector_tuning.py --participants P04 P10 --jobs 2
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from analysis.common import PPG_DEVICES
from run_pipeline import run_pipeline
from src.utils import load_config

logger = logging.getLogger(__name__)

TUNING_PARTICIPANTS = ["P02", "P04", "P10", "P11", "P17", "P21"]
MOTION_ACTIVITIES = ["walking", "jogging"]
REST_ACTIVITIES = ["baseline", "rest-1"]

VARIANTS: dict[str, dict] = {
    "baseline": {"enabled": False},
    "prior_hps": {"enabled": True, "prior": "hps"},
    "prior_device": {"enabled": True, "prior": "device"},
    "prior_auto": {"enabled": True, "prior": "auto"},
}


def _run_one(task: tuple[str, str, str, dict]) -> tuple[str, str, pd.DataFrame]:
    """Process one (variant, participant) pair."""
    variant, participant, data_root, config = task
    results, _ = run_pipeline(os.path.join(data_root, participant), config)
    if not results.empty:
        results["participant"] = participant
    return variant, participant, results


def main():
    parser = argparse.ArgumentParser(description="GalaxyPPG HRV — detector tuning")
    parser.add_argument("--participants", nargs="+", default=TUNING_PARTICIPANTS)
    parser.add_argument("--data-root", default="Dataset")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--output", default="paper_results/detector_tuning.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    base_config = load_config(args.config)

    tasks = []
    for variant, overrides in VARIANTS.items():
        config = copy.deepcopy(base_config)
        config["processing"]["rhythm_tracking"].update(overrides)
        # Disable ANC so the comparison isolates the detector change
        config["processing"]["anc"]["enabled"] = False
        for participant in args.participants:
            tasks.append((variant, participant, args.data_root, config))

    frames: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_run_one, task): task[:2] for task in tasks}
        for future in as_completed(futures):
            variant, participant = futures[future]
            try:
                _, _, results = future.result()
            except Exception as exc:
                logger.error("%s/%s failed: %s", variant, participant, exc)
                continue
            if results.empty:
                continue
            results["variant"] = variant
            frames.append(results)
            logger.info("%s/%s: %d rows", variant, participant, len(results))

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.output, index=False)
    logger.info("Wrote per-recording results → %s", args.output)

    ppg = combined[combined["device"].isin(PPG_DEVICES)]
    summary = ppg.groupby(["variant", "activity"]).agg(
        n=("hr_error_vs_ecg", "size"),
        hr_mae=("hr_error_vs_ecg", "mean"),
        recall=("pmr_recall_vs_ecg", "mean"),
        timing_ms=("peak_timing_ms_vs_ecg", "mean"),
    ).reset_index()

    summary["severe_hr_error_pct"] = (
        ppg.assign(severe=ppg["hr_error_vs_ecg"] > 10)
        .groupby(["variant", "activity"])["severe"].mean().to_numpy() * 100
    )
    summary_path = args.output.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote summary → %s", summary_path)

    pd.set_option("display.width", 200)
    print(summary.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
