#!/usr/bin/env python
"""Ablation and sensitivity study.

Every design choice of the pipeline is expressed as a configuration override
(a *variant*).  Each variant is run on the whole cohort (in-process,
parallel across participants) and summarised **per split**, so that

* choices are justified on the development participants only, and
* their effect on the untouched test participants can be reported without
  having influenced the choice.

The study replaces the former ad-hoc scripts: detector tuning (rhythm
tracking prior), adaptive-noise-cancellation tuning and the QC-threshold
sensitivity analysis are all variants here.

Outputs::

    paper_results/ablation/<variant>.csv      per-window results
    paper_results/ablation_summary.csv        variant x split x device x condition

Usage::

    .venv/bin/python run_ablation.py --jobs 20
    .venv/bin/python run_ablation.py --variant base --variant no_sqi --jobs 20
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from analysis.common import CONDITIONS
from run_all_participants import discover_participants, participant_split
from run_pipeline import run_pipeline
from src.utils import load_config

logger = logging.getLogger("ablation")

# Each variant: list of (dotted config path, value).
VARIANTS: dict[str, list[tuple[str, object]]] = {
    "base": [],
    # --- beat QC -----------------------------------------------------------------
    "no_sqi": [("devices.E4.sqi_beat_threshold", None), ("devices.GalaxyWatch.sqi_beat_threshold", None),
               ("hrv.min_segment_sqi", None)],
    "sqi_0.70": [("devices.E4.sqi_beat_threshold", 0.70), ("devices.GalaxyWatch.sqi_beat_threshold", 0.70)],
    "sqi_0.90": [("devices.E4.sqi_beat_threshold", 0.90), ("devices.GalaxyWatch.sqi_beat_threshold", 0.90)],
    "rhythm_gate": [("hrv.rhythm_gate", True)],
    "no_cadence_gate": [("hrv.cadence_lock.enabled", False)],
    "no_beat_recovery": [("processing.beat_recovery.enabled", False)],
    "no_timing_refinement": [("processing.timing.enabled", False)],
    "timing_low_motion_only": [("processing.timing.motion_states", ["low"])],
    "recovered_in_differences": [("processing.beat_recovery.exclude_from_differences", False)],
    "recovery_corr_0.9": [("processing.beat_recovery.min_correlation", 0.9)],
    "burden_10pct": [("hrv.max_interpolated_fraction", 0.10)],
    "device_hr_gate": [("hrv.device_hr_gate", True)],
    # --- correction burden / jitter (former sensitivity analysis) -----------------
    "strict": [("hrv.max_interpolated_fraction", 0.02), ("hrv.max_jitter_percent", 10.0)],
    "permissive": [("hrv.max_interpolated_fraction", 0.15), ("hrv.max_jitter_percent", 25.0)],
    "min_segment_60s": [("hrv.min_segment_duration_s", 60.0)],
    # --- detection ----------------------------------------------------------------
    "ensemble_2votes": [("detection.ensemble_min_votes", 2)],
    "neurokit2_only": [("detection.priority.E4", {m: ["neurokit2"] for m in ("low", "moderate", "severe")}),
                       ("detection.priority.GalaxyWatch", {m: ["neurokit2"] for m in ("low", "moderate", "severe")})],
    # --- motion handling (former ANC / detector tuning) ----------------------------
    "anc_off": [("processing.anc.enabled", False)],
    "anc_all_motion": [("processing.anc.motion_states", ["low", "moderate", "severe"])],
    "anc_all_motion_both": [("processing.anc.motion_states", ["low", "moderate", "severe"]),
                            ("processing.anc.devices", ["E4", "GalaxyWatch"])],
    "anc_both_devices": [("processing.anc.devices", ["E4", "GalaxyWatch"])],
    "anc_mu0.005": [("processing.anc.mu", 0.005)],
    "anc_order16": [("processing.anc.order", 16)],
    "tracking_hps": [("processing.rhythm_tracking.prior", "hps")],
    # --- clock synchronisation ----------------------------------------------------
    "raw_local_sync": [("validation.local_sync_min_support", 0)],
    "session_sync_only": [("validation.local_sync_enabled", False)],
    # --- validation sensitivity ---------------------------------------------------
    "match_50ms": [("validation.match_tolerance_s", 0.05)],
    "match_150ms": [("validation.match_tolerance_s", 0.15)],
}



def apply_overrides(config: dict, overrides: list[tuple[str, object]]) -> dict:
    """Deep-copy *config* and set each dotted path to its value."""
    out = copy.deepcopy(config)
    for path, value in overrides:
        node = out
        keys = path.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return out


def fingerprint(config: dict) -> str:
    """Short hash of a full configuration (provenance of a summary row)."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _run(task: tuple[str, str, str, dict]) -> pd.DataFrame:
    variant, participant, data_root, config = task
    frame, _ = run_pipeline(os.path.join(data_root, participant), config)
    if not frame.empty:
        frame.insert(0, "split", participant_split(participant, config))
        frame.insert(0, "participant", participant)
        frame.insert(0, "variant", variant)
    return frame


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Per variant x split x device x condition: yield, HRV and beat agreement."""
    ppg = frame[frame["device"] != "PolarH10"]
    rows = []
    splits = {"all": ppg, **{s: ppg[ppg["split"] == s] for s in ("dev", "test")}}
    for split, data in splits.items():
        for (variant, device), g in data.groupby(["variant", "device"]):
            for condition, activities in CONDITIONS.items():
                c = g[g["activity"].isin(activities)]
                if c.empty:
                    continue
                accepted = c[c["n_segments"].fillna(0) > 0]
                paired = accepted.dropna(subset=["rmssd", "ecg_rmssd"])
                paired = paired[(paired["rmssd"] > 0) & (paired["ecg_rmssd"] > 0)]
                log_ratio = np.log(paired["rmssd"] / paired["ecg_rmssd"])
                rows.append({
                    "variant": variant, "split": split, "device": device, "condition": condition,
                    "n_windows": int(len(c)),
                    "hrv_yield_pct": float(100 * len(accepted) / len(c)),
                    "n_paired": int(len(paired)),
                    "rmssd_ratio_gm": float(np.exp(log_ratio.mean())) if len(paired) else np.nan,
                    "rmssd_abs_log_error": float(np.abs(log_ratio).mean()) if len(paired) else np.nan,
                    "rmssd_within_10pct": float(100 * (np.abs(log_ratio) <= np.log(1.1)).mean()) if len(paired) else np.nan,
                    "sdnn_ratio_gm": float(np.exp(np.log(paired["sdnn"] / paired["ecg_sdnn"]).mean()))
                    if len(paired) else np.nan,
                    "hr_mae_bpm": float(c["hr_error_vs_ecg"].mean()),
                    "f1_pct": float(c["f1_vs_ecg"].mean()),
                    "recall_pct": float(c["recall_vs_ecg"].mean()),
                    "timing_sd_ms": float(c["timing_sd_ms_vs_ecg"].median()),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation / sensitivity study")
    parser.add_argument("--data-root", default="Dataset")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--variant", action="append", dest="variants", choices=sorted(VARIANTS))
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--output-dir", default="paper_results/ablation")
    parser.add_argument("--summary", default="paper_results/ablation_summary.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    base = load_config(args.config)
    participants = [p for p in discover_participants(args.data_root)
                    if os.path.exists(os.path.join(args.data_root, p, "Event.csv"))]
    variants = args.variants or list(VARIANTS)
    if "base" not in variants:
        variants = ["base", *variants]  # the reference must always be re-computed alongside
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = [(v, p, args.data_root, apply_overrides(base, VARIANTS[v])) for v in variants for p in participants]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        frames = [f for f in pool.map(_run, tasks) if not f.empty]
    combined = pd.concat(frames, ignore_index=True)
    for variant, group in combined.groupby("variant"):
        group.to_csv(os.path.join(args.output_dir, f"{variant}.csv"), index=False)

    summary = summarise(combined)
    summary["config_hash"] = summary["variant"].map(
        {v: fingerprint(apply_overrides(base, VARIANTS[v])) for v in variants})
    if os.path.exists(args.summary) and args.variants:
        previous = pd.read_csv(args.summary)
        current = {v: fingerprint(apply_overrides(base, VARIANTS[v])) for v in VARIANTS}
        # Keep earlier rows only when they were computed with today's
        # configuration for that variant and the variant still exists.
        keep = previous["variant"].isin(set(VARIANTS) - set(variants))
        if "config_hash" in previous:
            keep &= previous["config_hash"] == previous["variant"].map(current)
        else:
            keep &= False
        dropped = sorted(set(previous.loc[~keep & ~previous["variant"].isin(variants), "variant"]))
        if dropped:
            logger.warning("Dropping stale ablation rows (config changed or variant removed): %s", dropped)
        summary = pd.concat([previous[keep], summary], ignore_index=True)
    summary.to_csv(args.summary, index=False)
    view = summary[(summary["split"] == "dev")].round(2)
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print(view.to_string(index=False))


if __name__ == "__main__":
    main()
