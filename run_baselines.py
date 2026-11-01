#!/usr/bin/env python
"""External baselines on identical windows and identical validation.

Every PPG window of every participant is processed by

* ``pipeline``       - this pipeline (all detected beats);
* ``heartpy``        - HeartPy (van Gent et al., 2019) after a 0.5-4 Hz
  band-pass, the method used by the dataset authors (Park et al., 2025);
  systolic peaks of the upright pulse (all three baselines detect systolic
  peaks; the pipeline times beats at the intersecting-tangents foot);
* ``nk_elgendi``     - NeuroKit2 default PPG path (``ppg_clean`` + Elgendi
  peaks) with NeuroKit2's Kubios artifact correction;
* ``nk_msptdfast``   - MSPTDfast (Charlton et al., 2024) via NeuroKit2;
* For HRV, every baseline is reported raw and after the same
  Lipponen-Tarvainen correction as the pipeline (``*_corrected``), with
  synthetic beats excluded from successive differences.
* ``vendor``         - the device's own IBI stream (reported from the main
  results; registered by value because its timestamps are unreliable).

All beat trains are timed on the same fitted sample clock and validated
against the same ECG reference with the same procedure (delay re-estimated
per method within the window's prior, one-to-one matching +-100 ms, beat-
to-beat IBI agreement).  HRV agreement is compared on the windows the
pipeline accepted, with every method's RMSSD computed over the same accepted
spans (beats inside the spans, physiological range 300-2000 ms only).

Usage::

    .venv/bin/python run_baselines.py --jobs 23
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from analysis.common import ACTIVITY_CONDITION
from run_all_participants import discover_participants, participant_split
from run_pipeline import process_participant
from src.artifacts import correct_beat_times
from src.hrv_metrics import robust_hr
from src.signal_processing import bandpass_filter
from src.utils import load_config
from src.validation import beat_interval_agreement, estimate_lag, match_peaks, to_seconds

logger = logging.getLogger("baselines")
METHODS = ("pipeline", "heartpy", "nk_elgendi", "nk_msptdfast")


def _oriented(raw: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return raw if np.dot(raw - raw.mean(), reference) >= 0 else -raw


def baseline_peaks(raw: np.ndarray, fs: float, polarity_ref: np.ndarray) -> dict[str, np.ndarray]:
    """Peak indices of every external method on one raw (repaired) window."""
    import heartpy as hp
    import neurokit2 as nk

    out: dict[str, np.ndarray] = {}
    signal = _oriented(raw, polarity_ref)
    empty = np.array([], dtype=int)
    try:
        filtered = bandpass_filter(signal, fs, 0.5, min(4.0, 0.45 * fs), 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wd, _ = hp.process(hp.scale_data(filtered), fs, bpmmax=220)
        peaks = np.asarray(wd["peaklist"], dtype=int)
        removed = set(np.asarray(wd.get("removed_beats", []), dtype=int).tolist())
        out["heartpy"] = np.array([p for p in peaks if p not in removed], dtype=int)
    except Exception:
        out["heartpy"] = empty
    try:
        cleaned = nk.ppg_clean(signal, sampling_rate=fs)
    except Exception:
        cleaned = None
    for name, kwargs in (("nk_elgendi", {"method": "elgendi", "correct_artifacts": True}),
                         ("nk_msptdfast", {"method": "charlton"})):
        try:
            # MSPTDfast requires an integer sampling rate (64 and 25 Hz are integers)
            _, info = nk.ppg_peaks(cleaned, sampling_rate=int(round(fs)), **kwargs)
            out[name] = np.asarray(info["PPG_Peaks"], dtype=int)
        except Exception:
            out[name] = empty
    return out


def _rmssd_in_spans(beats: np.ndarray, spans_or_synthetic, spans=None) -> float:
    """RMSSD of beats inside the spans (300-2000 ms; synthetic beats excluded)."""
    if spans is None:
        spans, synthetic = spans_or_synthetic, np.zeros(len(beats), dtype=bool)
    else:
        synthetic = np.asarray(spans_or_synthetic, dtype=bool)
    diffs = []
    for a, b in spans:
        inside = (beats >= a - 1e-6) & (beats <= b + 1e-6)
        t, syn = beats[inside], synthetic[inside]
        ibi = np.diff(t) * 1000.0
        ok = (ibi >= 300) & (ibi <= 2000) & ~syn[:-1] & ~syn[1:] if len(t) > 1 else np.array([], bool)
        d = np.diff(ibi)[ok[:-1] & ok[1:]] if len(ibi) > 2 else np.array([])
        diffs.append(d)
    d = np.concatenate(diffs) if diffs else np.array([])
    return float(np.sqrt(np.mean(d ** 2))) if len(d) >= 10 else np.nan


def analyse(task: tuple[str, str, dict]) -> list[dict]:
    data_root, participant, config = task
    out = process_participant(os.path.join(data_root, participant), config)
    if not out:
        return []
    tolerance = float(config["validation"]["match_tolerance_s"])
    split = participant_split(participant, config)
    rows = []
    for (device, activity), (results, extras) in out["ppg"].items():
        reference = out["reference"].get(activity)
        if reference is None or "lag_prior" not in extras:
            continue
        fs = float(config["devices"][device]["fs"])
        ecg = reference["beats_s"]
        center, halfwidth = extras["lag_prior"]
        index = np.arange(len(extras["clock_seconds"]), dtype=float)
        gap_s = float(config.get("processing", {}).get("gap_threshold_ms", 3000.0)) / 1000.0

        def to_abs(positions, _x=extras, _index=index):
            return _x["clock_t0_s"] + np.interp(np.asarray(positions, dtype=float), _index,
                                                _x["clock_seconds"])

        trains = {"pipeline": extras["detected_s"]}
        corrected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, peaks in baseline_peaks(extras["signal_repaired"], fs, extras["signal_filtered"]).items():
            trains[name] = to_abs(peaks) if len(peaks) else np.array([])
            if len(trains[name]) >= 5:
                fix = correct_beat_times(trains[name], max_gap_s=gap_s)
                synthetic = fix.altered_beats if fix.altered_beats is not None else np.zeros(len(fix.times), bool)
                corrected[name] = (fix.times, synthetic)

        spans = [(to_seconds([s["start"]])[0], to_seconds([s["end"]])[0]) for s in extras.get("segments", [])]
        lag_pipeline = extras.get("lag_s", np.nan)
        ecg_rmssd_spans = np.nan
        if spans and np.isfinite(lag_pipeline):
            ecg_rmssd_spans = _rmssd_in_spans(ecg, [(a - lag_pipeline, b - lag_pipeline) for a, b in spans])

        for name, beats in trains.items():
            row = {"participant": participant, "split": split, "device": device, "activity": activity,
                   "condition": ACTIVITY_CONDITION.get(activity), "method": name,
                   "motion_level": results.get("motion_level"), "n_beats": int(len(beats))}
            if len(beats) >= 5:
                lag = estimate_lag(ecg, beats, max_lag_s=max(halfwidth, 0.25), center_s=center)["lag_s"]
                match = match_peaks(ecg, beats, tolerance, lag)
                ibi = beat_interval_agreement(ecg, beats, tolerance, lag)
                hr = robust_hr(np.diff(beats) * 1000.0)
                row.update({
                    "recall": match["recall"], "precision": match["precision"], "f1": match["f1"],
                    "timing_sd_ms": match["timing_sd_ms"], "beat_ibi_mae_ms": ibi.get("mae", np.nan),
                    "hr_error_bpm": abs(hr - reference["hr_median"]) if np.isfinite(hr) else np.nan,
                })
                if spans and np.isfinite(ecg_rmssd_spans) and np.isfinite(lag):
                    if name == "pipeline":
                        rmssd = results.get("rmssd", np.nan)
                        rmssd_corr = rmssd
                    else:
                        rmssd = _rmssd_in_spans(beats, spans)
                        rmssd_corr = (_rmssd_in_spans(*corrected[name], spans) if name in corrected else np.nan)
                    row.update({"rmssd": rmssd, "rmssd_corrected": rmssd_corr,
                                "ecg_rmssd_spans": ecg_rmssd_spans})
            rows.append(row)
        rows.append({"participant": participant, "split": split, "device": device, "activity": activity,
                     "condition": ACTIVITY_CONDITION.get(activity), "method": "vendor",
                     "motion_level": results.get("motion_level"),
                     "beat_ibi_mae_ms": results.get("vendor_ibi_mae_vs_ecg", np.nan),
                     "vendor_coverage_pct": results.get("vendor_ibi_coverage_pct", 0.0),
                     "hr_error_bpm": results.get("vendor_hr_error_vs_ecg", np.nan)})
    return rows


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (device, condition, method), g in frame.groupby(["device", "condition", "method"]):
        paired = g.dropna(subset=["rmssd", "ecg_rmssd_spans"]) if "rmssd" in g else g.iloc[0:0]
        paired = paired[(paired["rmssd"] > 0) & (paired["ecg_rmssd_spans"] > 0)]
        log_ratio = np.log(paired["rmssd"] / paired["ecg_rmssd_spans"])
        corr = paired.dropna(subset=["rmssd_corrected"]) if "rmssd_corrected" in paired else paired.iloc[0:0]
        corr = corr[corr["rmssd_corrected"] > 0]
        log_ratio_corr = np.log(corr["rmssd_corrected"] / corr["ecg_rmssd_spans"])
        rows.append({
            "device": device, "condition": condition, "method": method, "n_windows": int(len(g)),
            "f1_pct": float(g["f1"].median()) if "f1" in g else np.nan,
            "recall_pct": float(g["recall"].median()) if "recall" in g else np.nan,
            "precision_pct": float(g["precision"].median()) if "precision" in g else np.nan,
            "timing_sd_ms": float(g["timing_sd_ms"].median()) if "timing_sd_ms" in g else np.nan,
            "beat_ibi_mae_ms": float(g["beat_ibi_mae_ms"].median()),
            "hr_mae_bpm": float(g["hr_error_bpm"].mean()),
            "vendor_coverage_pct": float(g["vendor_coverage_pct"].mean()) if "vendor_coverage_pct" in g else np.nan,
            "n_hrv_windows": int(len(paired)),
            "rmssd_ratio_gm": float(np.exp(log_ratio.mean())) if len(paired) else np.nan,
            "rmssd_abs_log_error": float(log_ratio.abs().median()) if len(paired) else np.nan,
            "rmssd_within_10pct": float(100 * (log_ratio.abs() <= np.log(1.1)).mean()) if len(paired) else np.nan,
            "rmssd_ratio_gm_corrected": float(np.exp(log_ratio_corr.mean())) if len(corr) else np.nan,
            "rmssd_abs_log_error_corrected": float(log_ratio_corr.abs().median()) if len(corr) else np.nan,
            "rmssd_within_10pct_corrected": float(100 * (log_ratio_corr.abs() <= np.log(1.1)).mean())
            if len(corr) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="External baselines on identical windows")
    parser.add_argument("--data-root", default="Dataset")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="paper_results/baselines.csv")
    parser.add_argument("--summary", default="paper_results/tables/baselines_summary.csv")
    parser.add_argument("--jobs", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    config = load_config(args.config)
    participants = [p for p in discover_participants(args.data_root)
                    if os.path.exists(os.path.join(args.data_root, p, "Event.csv"))]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        chunks = list(pool.map(analyse, [(args.data_root, p, config) for p in participants]))
    frame = pd.DataFrame(r for chunk in chunks for r in chunk)
    frame.to_csv(args.output, index=False)
    summary = summarise(frame)
    summary.to_csv(args.summary, index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
