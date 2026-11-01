"""Cohort-level statistics for the GalaxyPPG HRV validation study.

Builds every summary table from ``paper_results/combined_results.csv``.
Every table carries a ``split`` column:

* ``test`` - participants never used for any tuning decision (primary);
* ``dev`` - development participants (all tuning happened here);
* ``all`` - full cohort (secondary).

Agreement statistics use :mod:`analysis.agreement` (ICC(A,1) with CI,
repeated-measures and log-ratio Bland-Altman, TOST, participant-clustered
bootstrap CIs, Friedman tests).  Pooled "rest" rows combine baseline and
rest-1 with repeated-measures limits.

Usage::

    uv run python run_stats.py --csv paper_results/combined_results.csv
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr, wilcoxon

from analysis.agreement import (
    bland_altman,
    cluster_bootstrap,
    cluster_bootstrap_mean,
    friedman,
    icc_a1,
    log_ratio_bland_altman,
    tost_paired,
)
from analysis.common import (
    CONDITION_ORDER,
    CONDITIONS,
    PPG_DEVICES,
    load_combined,
    ppg_only,
    qc_passed,
)

logger = logging.getLogger(__name__)

SPLITS = ("test", "dev", "all")
REST = tuple(CONDITIONS["rest"])

# A priori equivalence bounds (TOST).  HR: +-5 bpm (ANSI/CTA-2065 accuracy
# band).  Log-scale HRV: +-0.10 (~ +-10 %) for amplitude-like indices and
# +-0.20 for power indices (power scales with amplitude squared).
DEFAULT_BOUNDS = {"hr": 5.0, "log_amplitude": 0.10, "log_power": 0.20, "mean_nn_ms": 10.0}

HRV_METRICS = {
    # metric: (scale, kind) - kind selects log-ratio (positive indices) or plain BA
    "mean_nn": ("ms", "plain"),
    "sdnn": ("ms", "log_amplitude"),
    "rmssd": ("ms", "log_amplitude"),
    "pnn50": ("%", "plain"),
    "sd1": ("ms", "log_amplitude"),
    "sd2": ("ms", "log_amplitude"),
    "hf_power": ("ms2", "log_power"),
    "lf_power": ("ms2", "log_power"),
    "lf_hf_ratio": ("-", "log_power"),
    "sampen": ("-", "plain"),
}


# ---------------------------------------------------------------------------
# Backwards-compatible helpers (used by tests, figures and window analysis)
# ---------------------------------------------------------------------------

def _clean_pair(a, b) -> tuple[np.ndarray, np.ndarray]:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    return a_arr[mask], b_arr[mask]


def icc3_1(measurement_a, measurement_b) -> float:
    """ICC(3,1), two-way mixed, consistency (kept for reference only).

    Consistency ICC ignores systematic bias; agreement tables use
    :func:`analysis.agreement.icc_a1`.
    """
    a, b = _clean_pair(measurement_a, measurement_b)
    n = len(a)
    if n < 3:
        return np.nan
    k = 2
    values = np.column_stack([a, b])
    grand_mean = values.mean()
    ss_total = ((values - grand_mean) ** 2).sum()
    ss_rows = k * ((values.mean(axis=1) - grand_mean) ** 2).sum()
    ss_columns = n * ((values.mean(axis=0) - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_columns
    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error
    if denominator <= 0:
        return np.nan
    return float((ms_rows - ms_error) / denominator)


def cohens_dz(differences) -> float:
    """Paired Cohen's d_z (mean difference standardised by its own SD)."""
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    sd = np.std(values, ddof=1)
    if sd == 0:
        return np.nan
    return float(np.mean(values) / sd)


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values, preserving input order."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return adjusted
    order = np.argsort(p[finite])
    ranked = p[finite][order]
    m = len(ranked)
    scaled = ranked * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(monotone, 0, 1)
    adjusted[finite] = out
    return adjusted


def paired_comparison(a, b, metric: str, comparison: str, group: dict | None = None) -> dict | None:
    """Wilcoxon signed-rank test with effect size for a paired comparison."""
    a_values, b_values = _clean_pair(a, b)
    if len(a_values) < 6 or not np.any(a_values != b_values):
        return None
    differences = a_values - b_values
    statistic, p_value = wilcoxon(a_values, b_values)
    return {
        "metric": metric, "comparison": comparison, **(group or {}),
        "n": int(len(differences)),
        "median_a": float(np.median(a_values)), "median_b": float(np.median(b_values)),
        "median_diff": float(np.median(differences)), "cohens_dz": cohens_dz(differences),
        "statistic": float(statistic), "p_value": float(p_value),
    }


def kruskal_wallis(groups: list[np.ndarray]) -> dict:
    """Kruskal-Wallis omnibus test (independent groups; kept for compatibility)."""
    cleaned = [np.asarray(g, dtype=float) for g in groups]
    cleaned = [g[np.isfinite(g)] for g in cleaned]
    cleaned = [g for g in cleaned if len(g) > 0]
    if len(cleaned) < 2 or sum(len(g) for g in cleaned) < 6:
        return {"statistic": np.nan, "p_value": np.nan, "n_groups": len(cleaned)}
    statistic, p_value = kruskal(*cleaned)
    return {"statistic": float(statistic), "p_value": float(p_value), "n_groups": len(cleaned)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {"all": df}
    if "split" in df.columns:
        for split in ("test", "dev"):
            frames[split] = df[df["split"] == split]
    return frames


def _conditions(ppg: pd.DataFrame):
    """Yield ``(label, frame)`` for each reporting condition.

    Conditions pool the protocol sessions of comparable motion/physiology
    (``analysis.common.CONDITIONS``); participants contribute several
    windows, handled by repeated-measures statistics.  The 3-min seated
    ``baseline`` is also reported alone for comparison with Park et al.
    (2025, Table 5).
    """
    yield "baseline", ppg[ppg["activity"] == "baseline"]
    for condition in CONDITION_ORDER:
        yield condition, ppg[ppg["condition"] == condition]


def _participant_means(frame: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Per-participant means of two paired columns (independent units for ICC)."""
    return frame.groupby("participant", observed=True)[[a, b]].mean().dropna()


def _boot_mean(frame: pd.DataFrame, column: str) -> tuple[float, float, float]:
    return cluster_bootstrap_mean(frame, column)


def _fmt_ci(prefix: str, values: tuple[float, float, float]) -> dict[str, float]:
    return {prefix: values[0], f"{prefix}_ci_low": values[1], f"{prefix}_ci_high": values[2]}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def cohort_flow_table(df: pd.DataFrame) -> pd.DataFrame:
    """CONSORT-style flow: windows -> valid reference -> accepted PPG HRV."""
    rows = []
    ppg = ppg_only(df)
    for (device, activity), g in ppg.groupby(["device", "condition"], observed=True):
        accepted = g["n_segments"].fillna(0) > 0
        rows.append({
            "device": device, "condition": activity,
            "n_windows": int(len(g)),
            "n_reference_valid": int(g["ecg_hr"].notna().sum()),
            "local_clock_sync_pct": float(100 * g["clock_sync_local"].mean()) if "clock_sync_local" in g else np.nan,
            "n_hrv_accepted": int(accepted.sum()),
            "hrv_yield_pct": float(100 * accepted.mean()),
            "n_hrv_paired_with_ecg": int((g["n_segments_paired"].fillna(0) > 0).sum())
            if "n_segments_paired" in g else 0,
            "accepted_duration_pct": float(100 * (g["duration_s"].fillna(0) / g["duration_sec"]).mean()),
            "beats_sqi_rejected_pct": float(g["pct_beats_sqi_rejected"].mean()),
            **{f"excluded_{gate}": int(g.get(f"excluded_{gate}", pd.Series(dtype=float)).fillna(0).sum())
               for gate in ("duration", "interpolation", "jitter", "sqi", "rhythm", "cadence")},
        })
    return pd.DataFrame(rows)


def hr_agreement_table(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Window HR (median of raw IBIs) vs ECG: MAE/MAPE/BA/ICC/TOST."""
    rows = []
    ppg = ppg_only(df)
    for device in PPG_DEVICES:
        dev = ppg[ppg["device"] == device]
        for condition, g in _conditions(dev):
            g = g[np.isfinite(g["hr_median"]) & np.isfinite(g["ecg_hr"])]
            if len(g) < 3:
                continue
            diff = g["hr_median"] - g["ecg_hr"]
            ba = bland_altman(g["hr_median"], g["ecg_hr"], g["participant"])
            means = _participant_means(g, "hr_median", "ecg_hr")
            icc = icc_a1(means["hr_median"], means["ecg_hr"])
            participant_diff = (g.assign(d=diff).groupby("participant", observed=True)["d"].mean())
            tost = tost_paired(participant_diff, -bounds["hr"], bounds["hr"])
            mae = _boot_mean(g.assign(ae=diff.abs()), "ae")
            rows.append({
                "device": device, "condition": condition, "n": int(len(g)),
                "n_participants": int(g["participant"].nunique()),
                **_fmt_ci("mae_bpm", mae),
                "mape_pct": float(np.mean(diff.abs() / g["ecg_hr"]) * 100),
                "bias_bpm": ba["bias"], "loa_low_bpm": ba["loa_low"], "loa_high_bpm": ba["loa_high"],
                "within_5bpm_pct": float(100 * np.mean(diff.abs() <= 5)),
                "icc_a1": icc["icc"], "icc_ci_low": icc["ci_low"], "icc_ci_high": icc["ci_high"],
                "tost_p": tost["p_tost"], "equivalent_5bpm": tost["equivalent"],
                "vendor_mae_bpm": float(g["vendor_hr_error_vs_ecg"].mean()) if "vendor_hr_error_vs_ecg" in g else np.nan,
                "vendor_n": int(g["vendor_hr_error_vs_ecg"].notna().sum()) if "vendor_hr_error_vs_ecg" in g else 0,
                "assisted_mae_bpm": float(g["hr_error_vs_ecg_tracked"].mean()) if "hr_error_vs_ecg_tracked" in g else np.nan,
                "spectral_mae_bpm": float(g["hr_error_vs_ecg_spectral"].mean()) if "hr_error_vs_ecg_spectral" in g else np.nan,
            })
    return pd.DataFrame(rows)


def beat_detection_table(df: pd.DataFrame) -> pd.DataFrame:
    """Beat-level detection vs ECG (+-100 ms after delay removal), with chance level."""
    rows = []
    ppg = ppg_only(df)
    for device in PPG_DEVICES:
        dev = ppg[ppg["device"] == device]
        for condition, g in _conditions(dev):
            g = g[g["recall_vs_ecg"].notna()]
            if len(g) < 3:
                continue
            row = {"device": device, "condition": condition, "n": int(len(g))}
            for column, name in (("recall_vs_ecg", "recall_pct"), ("precision_vs_ecg", "precision_pct"),
                                 ("f1_vs_ecg", "f1_pct"), ("recall_chance_vs_ecg", "chance_recall_pct")):
                row.update(_fmt_ci(name, _boot_mean(g, column)))
            row.update({
                "recall_tol50_pct": float(g["recall_vs_ecg_tol50"].mean()),
                "legacy_pmr_pct": float(g["recall_vs_ecg_tol500_unaligned"].mean()),
                "timing_sd_ms": float(g["timing_sd_ms_vs_ecg"].median()),
                "beat_ibi_mae_ms": float(g["beat_ibi_mae_vs_ecg"].median()),
                "vendor_ibi_coverage_pct": float(g["vendor_ibi_coverage_pct"].mean()) if "vendor_ibi_coverage_pct" in g else np.nan,
                "vendor_ibi_mae_ms": float(g["vendor_ibi_mae_vs_ecg"].median()) if "vendor_ibi_mae_vs_ecg" in g else np.nan,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def hrv_agreement_table(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """HRV indices vs ECG over identical spans (accepted PPG segments)."""
    rows = []
    ppg = qc_passed(ppg_only(df))
    for device in PPG_DEVICES:
        dev = ppg[ppg["device"] == device]
        for condition, g in _conditions(dev):
            for metric, (unit, kind) in HRV_METRICS.items():
                ref_col = f"ecg_{metric}"
                if metric not in g or ref_col not in g:
                    continue
                h = g[np.isfinite(g[metric]) & np.isfinite(g[ref_col])]
                if len(h) < 3:
                    continue
                subjects = h["participant"]
                means = _participant_means(h, metric, ref_col)
                icc = icc_a1(means[metric], means[ref_col])
                row = {"device": device, "condition": condition, "metric": metric, "unit": unit,
                       "n": int(len(h)), "n_participants": int(h["participant"].nunique()),
                       "ppg_median": float(h[metric].median()), "ecg_median": float(h[ref_col].median()),
                       "icc_a1": icc["icc"], "icc_ci_low": icc["ci_low"], "icc_ci_high": icc["ci_high"]}
                if kind == "plain":
                    ba = bland_altman(h[metric], h[ref_col], subjects)
                    bound = bounds["mean_nn_ms"] if metric == "mean_nn" else np.nan
                    tost = tost_paired(h[metric] - h[ref_col], -bound, bound) if np.isfinite(bound) else {}
                    row.update({"bias": ba["bias"], "loa_low": ba["loa_low"], "loa_high": ba["loa_high"],
                                "ratio": np.nan, "ratio_loa_low": np.nan, "ratio_loa_high": np.nan,
                                "equivalence_bound": bound, "tost_p": tost.get("p_tost", np.nan),
                                "equivalent": tost.get("equivalent", False)})
                else:
                    lr = log_ratio_bland_altman(h[metric], h[ref_col], subjects)
                    bound = bounds[kind]
                    positive = (h[metric] > 0) & (h[ref_col] > 0)
                    log_ratio = np.log(h.loc[positive, metric] / h.loc[positive, ref_col])
                    # Windows of one participant are not independent: equivalence is tested on
                    # participant means of the log ratio.
                    per_participant = log_ratio.groupby(h.loc[positive, "participant"], observed=True).mean()
                    tost = tost_paired(per_participant, -bound, bound)
                    row.update({"bias": float((h[metric] - h[ref_col]).mean()),
                                "loa_low": np.nan, "loa_high": np.nan,
                                "ratio": lr["ratio"], "ratio_loa_low": lr["ratio_loa_low"],
                                "ratio_loa_high": lr["ratio_loa_high"],
                                "equivalence_bound": bound, "tost_p": tost["p_tost"],
                                "equivalent": tost["equivalent"], "n_tost": tost["n"],
                                "ratio_ci90_low": float(np.exp(tost["ci90_low"])),
                                "ratio_ci90_high": float(np.exp(tost["ci90_high"]))})
                rows.append(row)
    return pd.DataFrame(rows)


def beat_ibi_agreement_table(df: pd.DataFrame) -> pd.DataFrame:
    """Beat-to-beat IBI agreement of HRV-accepted beats (repeated-measures BA)."""
    rows = []
    ppg = ppg_only(df)
    for device in PPG_DEVICES:
        dev = ppg[ppg["device"] == device]
        for condition, g in _conditions(dev):
            g = g[g["hrv_ibi_n_vs_ecg"].fillna(0) >= 10] if "hrv_ibi_n_vs_ecg" in g else g.iloc[0:0]
            if len(g) < 3:
                continue
            rows.append({
                "device": device, "condition": condition, "n_windows": int(len(g)),
                "n_beats": int(g["hrv_ibi_n_vs_ecg"].sum()),
                "bias_ms": float(np.average(g["hrv_ibi_bias_vs_ecg"], weights=g["hrv_ibi_n_vs_ecg"])),
                "sd_ms_median": float(g["hrv_ibi_sd_vs_ecg"].median()),
                "mae_ms_median": float(g["hrv_ibi_mae_vs_ecg"].median()),
                "r_median": float(g["hrv_ibi_r_vs_ecg"].median()),
                "timing_sd_ms_median": float(g["timing_sd_ms_vs_ecg"].median()),
            })
    return pd.DataFrame(rows)


def timing_jitter_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per accepted window: PPG and ECG RMSSD with the beat-timing SD vs ECG.

    The decomposition of the RMSSD inflation into timing-error terms needs the
    beat-level error series and is computed in ``analysis.timing_error``.
    """
    ppg = qc_passed(ppg_only(df))
    cols = ["participant", "split", "device", "activity", "rmssd", "ecg_rmssd", "timing_sd_ms_vs_ecg"]
    g = ppg[[c for c in cols if c in ppg]].dropna()
    if g.empty:
        return g
    return g.assign(rmssd_ratio=g["rmssd"] / g["ecg_rmssd"])


def motion_effect_table(df: pd.DataFrame) -> pd.DataFrame:
    """Friedman tests across activities and motion/periodicity associations."""
    rows = []
    ppg = ppg_only(df)
    for device in PPG_DEVICES:
        g = ppg[ppg["device"] == device].copy()
        g["hr_abs_error"] = g["hr_error_vs_ecg"]
        g = g[g["condition"].isin(["rest", "seated_task", "hand_activity", "standing", "locomotion"])]
        for value in ("hr_abs_error", "recall_vs_ecg", "f1_vs_ecg"):
            result = friedman(g, value, "condition")
            rows.append({"device": device, "test": "friedman", "outcome": value, **result})
        for predictor in ("mean_motion", "acc_periodicity"):
            sub = g[["participant", predictor, "f1_vs_ecg"]].dropna()
            if len(sub) < 10:
                continue
            est = cluster_bootstrap(sub, lambda f, p=predictor: spearmanr(f[p], f["f1_vs_ecg"])[0])
            rows.append({"device": device, "test": "spearman (cluster bootstrap)",
                         "outcome": f"f1_vs_ecg ~ {predictor}", "rho": est[0],
                         "rho_ci_low": est[1], "rho_ci_high": est[2], "n": int(len(sub))})
    return pd.DataFrame(rows)


def device_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """E4 vs Galaxy Watch paired per participant, BH-corrected within outcome."""
    rows = []
    ppg = ppg_only(df)
    for outcome in ("hr_error_vs_ecg", "f1_vs_ecg", "timing_sd_ms_vs_ecg"):
        for condition in CONDITION_ORDER:
            g = ppg[ppg["condition"] == condition]
            wide = g.pivot_table(index="participant", columns="device", values=outcome, observed=True)
            if {"E4", "GalaxyWatch"}.issubset(wide.columns):
                result = paired_comparison(wide["GalaxyWatch"], wide["E4"], outcome,
                                           "GalaxyWatch vs E4", {"condition": condition})
                if result is not None:
                    rows.append(result)
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_adj_bh"] = np.nan
        for _, group in table.groupby("metric"):
            table.loc[group.index, "p_adj_bh"] = benjamini_hochberg(group["p_value"])
    return table


def pipeline_vs_vendor_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline beats vs the vendor's own IBI stream, paired per window."""
    rows = []
    ppg = ppg_only(df)
    for device in PPG_DEVICES:
        dev = ppg[ppg["device"] == device]
        for condition, g in _conditions(dev):
            if len(g) < 3:
                continue
            both = g[["beat_ibi_mae_vs_ecg", "vendor_ibi_mae_vs_ecg"]].dropna()
            row = {
                "device": device, "condition": condition, "n_windows": int(len(g)),
                "pipeline_recall_pct": float(g["recall_vs_ecg"].mean()),
                "vendor_coverage_pct": float(g["vendor_ibi_coverage_pct"].fillna(0).mean())
                if "vendor_ibi_coverage_pct" in g else np.nan,
                "pipeline_ibi_mae_ms": float(g["beat_ibi_mae_vs_ecg"].median()),
                "vendor_ibi_mae_ms": float(g["vendor_ibi_mae_vs_ecg"].median()) if "vendor_ibi_mae_vs_ecg" in g else np.nan,
                "n_paired": int(len(both)),
            }
            if len(both) >= 6 and np.any(both.iloc[:, 0] != both.iloc[:, 1]):
                row["wilcoxon_p"] = float(wilcoxon(both.iloc[:, 0], both.iloc[:, 1])[1])
            rows.append(row)
    return pd.DataFrame(rows)


def reference_validity_table(df: pd.DataFrame) -> pd.DataFrame:
    """ECG reference (NeuroKit2 R-peaks) vs the Polar H10 firmware R-R values."""
    ecg = df[df["device"] == "PolarH10"]
    rows = []
    for activity, g in ecg.groupby("condition", observed=True):
        g = g[g["reference_ibi_n"].fillna(0) > 0] if "reference_ibi_n" in g else g.iloc[0:0]
        if g.empty:
            continue
        rows.append({
            "condition": activity, "n": int(len(g)),
            "mae_ms_median": float(g["reference_ibi_mae_ms"].median()),
            "mae_ms_max": float(g["reference_ibi_mae_ms"].max()),
            "within_5ms_pct": float(g["reference_within_5ms_pct"].mean()),
            "coverage_pct": float(g["reference_coverage_pct"].mean()),
            "ecg_hrv_yield_pct": float(100 * (g["n_segments"].fillna(0) > 0).mean()),
            "effective_fs_hz": float(g["fs_effective"].median()) if "fs_effective" in g else np.nan,
        })
    return pd.DataFrame(rows)


def clock_sync_table(df: pd.DataFrame) -> pd.DataFrame:
    """Clock offsets and residual PPG-ECG delay per device."""
    rows = []
    ppg = ppg_only(df)
    for device in PPG_DEVICES:
        g = ppg[ppg["device"] == device]
        per_participant = g.groupby("participant")
        rest = g[g["condition"] == "rest"]
        session = per_participant["clock_offset_session_s"].first()
        applied_shift = (g["clock_offset_s"] - g["clock_offset_session_s"]).abs()
        step = per_participant["clock_offset_s"].agg(lambda x: x.max() - x.min())
        rows.append({
            "device": device,
            "session_offset_median_s": float(session.median()),
            "session_offset_range_s": f"{session.min():.1f} to {session.max():.1f}",
            "session_sync_r_min": float(per_participant["clock_sync_session_r"].first().min()),
            "local_refinement_pct": float(100 * g["clock_sync_local"].mean()),
            "applied_vs_session_p90_s": float(applied_shift.quantile(0.9)),
            "applied_vs_session_max_s": float(applied_shift.max()),
            "participants_with_step_over_1s": int((step > 1.0).sum()),
            "neighbour_windows_pct": float(100 * (g["clock_offset_source"] == "neighbour").mean()),
            "within_session_drift_s_max": float((per_participant["clock_offset_s"].max()
                                                 - per_participant["clock_offset_s"].min()).max()),
            "within_session_drift_s_median": float((per_participant["clock_offset_s"].max()
                                                    - per_participant["clock_offset_s"].min()).median()),
            "unsynchronised_windows_pct": float(100 * (g["clock_offset_source"] == "none").mean())
            if "clock_offset_source" in g else np.nan,
            "delay_rest_mean_ms": float(rest["lag_ms"].mean()),
            "delay_rest_sd_between_ms": float(rest.groupby("participant")["lag_ms"].mean().std()),
            "delay_anchor_pct": float(100 * (g["lag_source"] == "anchor").mean()),
            "delay_free_pct": float(100 * (g["lag_source"] == "free").mean()),
        })
    return pd.DataFrame(rows)


def build_tables(csv_path: str, output_dir: str, bounds: dict | None = None) -> dict[str, pd.DataFrame]:
    """Build every manuscript summary table (one CSV per table, all splits)."""
    df = load_combined(csv_path)
    bounds = {**DEFAULT_BOUNDS, **(bounds or {})}
    builders = {
        "cohort_flow": cohort_flow_table,
        "hr_agreement": lambda d: hr_agreement_table(d, bounds),
        "beat_detection": beat_detection_table,
        "hrv_agreement": lambda d: hrv_agreement_table(d, bounds),
        "beat_ibi_agreement": beat_ibi_agreement_table,
        "pipeline_vs_vendor": pipeline_vs_vendor_table,
        "motion_effect": motion_effect_table,
        "device_comparison": device_comparison_table,
        "reference_validity": reference_validity_table,
        "clock_sync": clock_sync_table,
        "timing_jitter": timing_jitter_table,
    }
    os.makedirs(output_dir, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    for name, builder in builders.items():
        parts = []
        for split, frame in _split_frames(df).items():
            if frame.empty:
                continue
            table = builder(frame)
            if not table.empty:
                parts.append(table.assign(split=split))
        tables[name] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        path = os.path.join(output_dir, f"{name}.csv")
        tables[name].to_csv(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(tables[name]))
    return tables


def main():
    parser = argparse.ArgumentParser(description="GalaxyPPG HRV - cohort statistics")
    parser.add_argument("--csv", default="paper_results/combined_results.csv")
    parser.add_argument("--output-dir", default="paper_results/tables")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    build_tables(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
