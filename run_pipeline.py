#!/usr/bin/env python
"""Single-participant entry point for the GalaxyPPG HRV pipeline.

Processing chain for every device x activity window:

1. **Load & align** - device data are loaded once per participant and every
   device clock is aligned to the Polar H10 from accelerometer activity
   (``src.sync``); sample times come from a fitted per-sample clock
   (``src.timebase``), never from ``index / nominal_fs``.
2. **Repair & filter** - missing/sentinel samples and level shifts are
   repaired, then a zero-phase SOS band-pass chosen from the accelerometer
   motion state (no protocol labels) is applied.
3. **Detect** - NeuroKit2, SciPy, slope-sum and template detectors with a
   consensus ensemble; one output is selected by device x motion priority.
4. **Beat QC** - template-matching SQI discards corrupted beats;
   Lipponen-Tarvainen correction repairs missed/extra/ectopic beats on the
   beat-time axis (time-conserving).
5. **Segment & HRV** - gap-aware segmentation with correction-burden, jitter,
   SQI and rhythm-consistency gates, then time, frequency and non-linear
   indices.
6. **Validate** (PPG) - against the ECG reference on the same absolute time
   base: beat cross-correlogram delay, one-to-one beat matching (+-100 ms),
   chance-level surrogate, beat-to-beat IBI agreement, and ECG HRV computed
   over exactly the accepted PPG spans.  The vendors' own IBI streams are
   evaluated as a baseline by time-constrained run registration
   (``register_vendor_runs``), because their timestamps are unreliable.

Usage::

    python run_pipeline.py --participant P02 --data-root Dataset
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src import __version__
from src.artifacts import beat_template_sqi, correct_beat_times, make_beat_locator
from src.data_loader import extract_device_ibis, load_device_data
from src.hrv_metrics import (
    SEGMENT_METRIC_KEYS,
    aggregate_hrv_metrics,
    aggregate_segment_metrics,
    build_nn_segments,
    robust_hr,
    segment_metrics,
)
from src.motion_detection import acc_periodicity, calculate_motion
from src.peak_detection import (
    complete_rhythm_intervals,
    estimate_rhythm_interval,
    harmonic_sum_hr,
    locate_fiducial,
    multi_method_peak_detection,
    refine_peaks_parabolic,
    refine_timing,
    select_optimal_peaks,
    select_peaks_with_rhythm,
    spectral_rhythm_peaks,
)
from src.signal_processing import (
    acc_referenced_anc,
    adaptive_bandpass_filter,
    bandpass_filter,
    estimate_signal_quality,
    pulse_asymmetry,
    repair_signal,
)
from src.sync import estimate_clock_offset, estimate_local_offset, resolve_window_offsets
from src.timebase import fit_sample_clock
from src.utils import load_config
from src.validation import (
    beat_interval_agreement,
    estimate_lag,
    match_peaks,
    register_vendor_runs,
    sequence_interval_agreement,
    surrogate_recall,
    time_aligned_agreement,
    to_seconds,
)

logger = logging.getLogger(__name__)

MIN_ACTIVITY_DURATION_S = 30.0
PPG_DEVICES = ("E4", "GalaxyWatch")


def _window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Select rows inside a closed time window."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    mask = (df.index >= start) & (df.index <= end)
    return df.loc[mask]


def _device_hr(device_data: dict, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, float]:
    """Mean and SD of valid device-reported HR in a window (diagnostic only)."""
    segment = _window(device_data.get("hr", pd.DataFrame()), start, end)
    if segment.empty or "hr" not in segment.columns:
        return np.nan, np.nan
    values = segment["hr"].to_numpy(dtype=float)
    valid = values[(values > 0) & (values < 300)]
    if len(valid) == 0:
        return np.nan, np.nan
    return float(np.mean(valid)), float(np.std(valid))


def extract_activity_windows(events: pd.DataFrame, activities: list[str]) -> dict:
    """Extract ``(start, end)`` windows for each configured activity."""
    windows: dict = {}
    for activity in activities:
        session = events[events["session"] == activity]
        if session.empty:
            continue
        enter = session[session["status"] == "ENTER"]["datetime"]
        exit_ = session[session["status"] == "EXIT"]["datetime"]
        if enter.empty or exit_.empty:
            continue
        start, end = enter.min(), exit_.max()
        if (end - start).total_seconds() >= MIN_ACTIVITY_DURATION_S:
            windows[activity] = (start, end)
        else:
            logger.warning("%s: invalid window (%s -> %s), skipping", activity, start, end)
    return windows


def _raw_neurokit2_signal(device: str, motion_level: str,
                          raw_signal: np.ndarray, config: dict) -> np.ndarray | None:
    """Raw signal for NeuroKit2 when the configured device x motion gate matches."""
    gate = config.get("processing", {}).get("neurokit2_raw", {})
    if (gate.get("enabled", False)
            and device in gate.get("devices", [])
            and motion_level in gate.get("motion_states", [])):
        return raw_signal
    return None


def shift_device_clock(device_data: dict, offset_s: float) -> dict:
    """Return *device_data* with every time index moved onto the Polar clock."""
    if not offset_s:
        return device_data
    shift = pd.Timedelta(seconds=float(offset_s))
    shifted = dict(device_data)
    for key, value in device_data.items():
        if isinstance(value, pd.DataFrame) and isinstance(value.index, pd.DatetimeIndex):
            frame = value.copy()
            frame.index = frame.index - shift
            shifted[key] = frame
    return shifted


def _interval_sqi(corrected_s: np.ndarray, kept_s: np.ndarray, kept_sqi: np.ndarray) -> np.ndarray:
    """Per-interval SQI: mean SQI of the two bounding beats (inserted beats = NaN)."""
    if len(corrected_s) < 2:
        return np.array([])
    beat_sqi = np.full(len(corrected_s), np.nan)
    if len(kept_s):
        idx = np.clip(np.searchsorted(kept_s, corrected_s), 1, max(len(kept_s) - 1, 1))
        left = np.clip(idx - 1, 0, len(kept_s) - 1)
        right = np.clip(idx, 0, len(kept_s) - 1)
        nearest = np.where(np.abs(kept_s[left] - corrected_s) <= np.abs(kept_s[right] - corrected_s), left, right)
        close = np.abs(kept_s[nearest] - corrected_s) <= 0.05
        beat_sqi[close] = kept_sqi[nearest[close]]
    pair = np.vstack([beat_sqi[:-1], beat_sqi[1:]])
    finite = np.isfinite(pair)
    counts = finite.sum(axis=0)
    sums = np.where(finite, pair, 0.0).sum(axis=0)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def _fiducial_position_fn(upright: np.ndarray, peaks: np.ndarray, fs: float,
                          fiducial: str, anchor: str):
    """Map a recovered beat (integer maximum of the detection signal) to the
    configured fiducial, using the median detected interval as local context
    so recovered beats are timed exactly like detected ones."""
    peaks = np.asarray(peaks, dtype=int)
    m = int(np.median(np.diff(peaks))) if len(peaks) > 2 else int(fs)

    def position(k: int) -> float:
        return float(locate_fiducial(upright, np.array([k - m, k, k + m]), fs, fiducial, anchor=anchor)[1])

    return position


def _uncertain_intervals(correction, config: dict) -> np.ndarray | None:
    """Intervals bounded by a signal-recovered beat (excluded from differences)."""
    recovery = config.get("processing", {}).get("beat_recovery", {})
    beats = correction.recovered_beats
    if not recovery.get("exclude_from_differences", True) or beats is None or len(beats) < 2:
        return None
    return beats[:-1] | beats[1:]


def _segment_dict(segment) -> dict:
    end = segment.interval_end_times[-1]
    return {"start": segment.start_time, "end": end, "duration_s": segment.duration_s,
            "nn": segment.nn, "interval_end_times": segment.interval_end_times,
            "synthetic": segment.synthetic}


def _paired_reference_metrics(ppg_segments: list[dict], ecg_beats_s: np.ndarray,
                              ecg_corrected: np.ndarray, lag_s: float,
                              config: dict) -> tuple[dict[str, float], int]:
    """ECG HRV over exactly the time spans of the accepted PPG segments.

    Each PPG segment ``[start, end]`` (Polar time base) maps to ECG beats in
    ``[start - lag, end - lag]``.  The ECG intervals inside the span are
    taken from the (Lipponen-Tarvainen corrected) reference beat series.  A
    span is not paired when it contains an ECG gap or out-of-range interval,
    or when its ECG correction burden exceeds ``hrv.max_interpolated_fraction``
    (the same gate the PPG segments passed), so both sides receive equal QC.
    """
    per_segment, weights = [], []
    if len(ecg_beats_s) < 3 or not ppg_segments:
        return {key: np.nan for key in SEGMENT_METRIC_KEYS}, 0
    lag = 0.0 if not np.isfinite(lag_s) else lag_s
    ecg_ibis = np.diff(ecg_beats_s) * 1000.0
    max_burden = float(config.get("hrv", {}).get("max_interpolated_fraction", 0.05))
    corrected = (np.asarray(ecg_corrected, dtype=bool)
                 if ecg_corrected is not None and len(ecg_corrected) == len(ecg_ibis) else None)
    for seg in ppg_segments:
        a = to_seconds([seg["start"]])[0] - lag
        b = to_seconds([seg["end"]])[0] - lag
        i0 = int(np.searchsorted(ecg_beats_s, a - 0.15))
        i1 = int(np.searchsorted(ecg_beats_s, b + 0.15, side="right")) - 1
        if i1 - i0 < 10:
            continue
        nn = ecg_ibis[i0:i1]
        if np.any(nn > 2000.0) or np.any(nn < 300.0):
            continue
        if corrected is not None and float(np.mean(corrected[i0:i1])) > max_burden:
            continue
        per_segment.append(segment_metrics(nn, config))
        weights.append(seg["duration_s"])
    if not per_segment:
        return {key: np.nan for key in SEGMENT_METRIC_KEYS}, 0
    return aggregate_segment_metrics(per_segment, weights), len(per_segment)


def process_activity(
    user_folder: str,
    device: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    activity_name: str,
    config: dict,
    ecg_reference: dict | None = None,
    device_data: dict | None = None,
) -> tuple[dict, dict]:
    """Process one device x activity window through the full pipeline.

    Args:
        user_folder: Participant data directory.
        device: ``"E4"``, ``"GalaxyWatch"`` or ``"PolarH10"``.
        start_time: Window start (Polar time base).
        end_time: Window end (Polar time base).
        activity_name: Protocol label - used for reporting only.
        config: Parsed configuration dictionary.
        ecg_reference: Reference beats/segments for this activity (PPG only).
        device_data: Pre-loaded, clock-aligned device data (loaded and left
            unaligned when omitted).

    Returns:
        Tuple of ``(results, extras)``: scalar metrics for the results table
        and dense series kept in memory (beat times, segments, IBI pairs).
    """
    dev_cfg = config["devices"][device]
    fs = float(dev_cfg["fs"])
    signal_col = dev_cfg["signal_col"]
    is_ppg = device != "PolarH10"

    if device_data is None:
        device_data = load_device_data(user_folder, device, config)
    if "signal" not in device_data:
        logger.warning("%s: no signal data found", device)
        return {}, {}
    segment_frame = _window(device_data["signal"], start_time, end_time)
    if len(segment_frame) < fs * 5:
        logger.warning("%s %s: insufficient data (%d samples)", device, activity_name, len(segment_frame))
        return {}, {}

    # --- time base, repair, motion -------------------------------------------------
    clock = fit_sample_clock(segment_frame.index)
    raw_values = segment_frame[signal_col].to_numpy(dtype=float)
    signal_data, unreliable = repair_signal(raw_values, fs) if is_ppg else repair_signal(raw_values, fs, shift_mad_factor=np.inf)
    clean_mask, _, motion_metrics = calculate_motion(
        device_data.get("acc", pd.DataFrame()), segment_frame.index, device, config
    )
    clean_mask = clean_mask & ~unreliable
    motion_level = motion_metrics["motion_level"]

    quality_metrics = estimate_signal_quality(signal_data, fs)
    device_hr, device_hr_std = _device_hr(device_data, start_time, end_time)

    anc_cfg = config.get("processing", {}).get("anc", {})
    anc_states = anc_cfg.get("motion_states", ["moderate", "severe"])
    signal_clean, anc_applied = signal_data, False
    if is_ppg and motion_level in anc_states and device in anc_cfg.get("devices", list(PPG_DEVICES)):
        signal_clean = acc_referenced_anc(signal_data, device_data.get("acc", pd.DataFrame()),
                                          segment_frame.index, fs, config)
        anc_applied = signal_clean is not signal_data

    signal_filtered = adaptive_bandpass_filter(signal_clean, device, motion_level, config=config)
    # Beats are detected as maxima of the detection signal: the upright pulse
    # (anchor "systolic") or the inverted pulse (anchor "onset"), whose sharp
    # onset corner is detected more reliably than the often rounded or
    # double-humped systolic apex.  The beat time is then placed at the
    # configured fiducial on the upright signal.
    timing_cfg = config.get("processing", {}).get("timing", {})
    anchor = timing_cfg.get("anchor", "systolic") if is_ppg else "systolic"
    fiducial = timing_cfg.get("fiducial", "peak") if is_ppg else "peak"
    signal_upright = signal_filtered
    if anchor == "onset":
        signal_filtered = -signal_filtered

    # --- detection ------------------------------------------------------------------
    peak_results = multi_method_peak_detection(
        signal_filtered, device, motion_level, clean_mask, quality_metrics, config=config,
        neurokit2_signal=_raw_neurokit2_signal(device, motion_level, signal_data, config),
    )
    optimal_peaks, method_used = select_optimal_peaks(
        peak_results, device, motion_level,
        priority=config.get("detection", {}).get("priority"),
    )
    optimal_peaks = np.asarray(optimal_peaks, dtype=int)

    # Standalone spectral HR (E4, severe motion) and device-assisted tracked HR.
    # Both use interval completion (model-derived beats) and never feed HRV.
    hr_spectral = np.nan
    spectral_cfg = config.get("processing", {}).get("spectral_rhythm", {})
    if (spectral_cfg.get("enabled", False) and device in spectral_cfg.get("devices", ["E4"])
            and motion_level in spectral_cfg.get("motion_states", ["severe"])):
        anc_section = dict(config.get("processing", {}).get("anc", {}), enabled=True)
        motion_cleaned = acc_referenced_anc(signal_data, device_data.get("acc", pd.DataFrame()),
                                            segment_frame.index, fs, {"processing": {"anc": anc_section}})
        spectral_signal = adaptive_bandpass_filter(motion_cleaned, device, motion_level, config=config)
        spectral_interval = estimate_rhythm_interval(spectral_signal, fs, device_hr, spectral_cfg.get("prior", "hps"))
        spectral_results = multi_method_peak_detection(spectral_signal, device, motion_level,
                                                       clean_mask, quality_metrics, config=config)
        spectral_peaks = spectral_rhythm_peaks(spectral_results, spectral_signal, fs, device_hr,
                                               tracking=config.get("processing", {}).get("rhythm_tracking", {}),
                                               prior=spectral_cfg.get("prior", "hps"))
        if len(spectral_peaks) >= 5 and np.isfinite(spectral_interval):
            positions = refine_peaks_parabolic(spectral_signal, spectral_peaks)
            positions, _ = complete_rhythm_intervals(positions, fs, spectral_interval)
            hr_spectral = robust_hr(np.diff(clock.to_seconds(positions)) * 1000.0)

    hr_tracked = np.nan
    tracking = config.get("processing", {}).get("rhythm_tracking", {})
    if (tracking.get("enabled", False) and is_ppg
            and motion_level in tracking.get("motion_states", ["low", "moderate"])):
        expected = estimate_rhythm_interval(signal_filtered, fs, device_hr, tracking.get("prior", "device"))
        if np.isfinite(expected):
            candidates = np.unique(np.concatenate([
                np.asarray(p, dtype=int) for m, p in peak_results.items()
                if not m.endswith("_all") and len(p) > 0])) if peak_results else optimal_peaks
            if len(candidates) >= 5:
                tracked = select_peaks_with_rhythm(
                    signal_filtered, candidates, fs, expected,
                    tolerance=float(tracking.get("tolerance", 0.30)),
                    penalty_weight=float(tracking.get("penalty_weight", 1.0)),
                    max_interval_factor=float(tracking.get("max_interval_factor", 2.5)),
                    amplitude_weight=float(tracking.get("amplitude_weight", 1.0)),
                )
                if len(tracked) >= 5:
                    positions = refine_peaks_parabolic(signal_filtered, tracked)
                    positions, _ = complete_rhythm_intervals(positions, fs, expected)
                    hr_tracked = robust_hr(np.diff(clock.to_seconds(positions)) * 1000.0)

    # --- beat timing: re-localise on a wide-band signal, then the fitted clock ----------
    timing_signal, timing_peaks = signal_filtered, optimal_peaks
    timing_highcut = (timing_cfg.get("highcut_hz", {}) or {}).get(device)
    if (is_ppg and timing_cfg.get("enabled", True) and timing_highcut
            and motion_level in timing_cfg.get("motion_states", ["low", "moderate", "severe"])
            and len(optimal_peaks)):
        low = float(dev_cfg["bandpass"][0])
        highcut = min(float(timing_highcut), 0.45 * fs)
        wide = bandpass_filter(signal_clean, fs, low, highcut, int(timing_cfg.get("order", 2)))
        # keep the polarity chosen for the detection signal
        timing_signal = wide if np.dot(wide, signal_filtered) >= 0 else -wide
        positions, timing_peaks = refine_timing(timing_signal, optimal_peaks, fs,
                                                float(timing_cfg.get("search_s", 0.12)), fiducial,
                                                anchor, -timing_signal if anchor == "onset" else timing_signal)
    elif is_ppg and len(optimal_peaks):
        positions = locate_fiducial(signal_upright, optimal_peaks, fs, fiducial, anchor=anchor)
    else:
        positions = refine_peaks_parabolic(signal_filtered, optimal_peaks)
    t0_s = to_seconds([clock.t0])[0]
    detected_s = t0_s + clock.to_seconds(positions)
    detected_times = clock.to_times(positions)
    raw_ibis = np.diff(detected_s) * 1000.0

    # --- beat QC: SQI screen, then Lipponen-Tarvainen ----------------------------------
    sqi = beat_template_sqi(signal_filtered, optimal_peaks, fs)
    sqi_threshold = dev_cfg.get("sqi_beat_threshold", 0.8)
    sqi_threshold = -np.inf if sqi_threshold is None else float(sqi_threshold)
    in_unreliable = unreliable[np.clip(optimal_peaks, 0, len(unreliable) - 1)] if len(optimal_peaks) else np.array([], dtype=bool)
    keep = (np.nan_to_num(sqi, nan=1.0) >= sqi_threshold) & ~in_unreliable
    kept_s, kept_sqi = detected_s[keep], np.nan_to_num(sqi[keep], nan=1.0)
    gap_s = float(config.get("processing", {}).get("gap_threshold_ms", 3000.0)) / 1000.0
    locator = None
    if config.get("processing", {}).get("beat_recovery", {}).get("enabled", True):
        locator = make_beat_locator(
            timing_signal, clock.seconds, t0_s, timing_peaks[keep],
            position_fn=_fiducial_position_fn(-timing_signal if anchor == "onset" else timing_signal,
                                              timing_peaks[keep], fs, fiducial, anchor) if is_ppg else None,
            min_correlation=float(config["processing"].get("beat_recovery", {}).get("min_correlation")
                                  or (sqi_threshold if np.isfinite(sqi_threshold) else 0.7)),
        )
    correction = correct_beat_times(kept_s, max_gap_s=gap_s, locator=locator)
    beats_s = correction.times
    beat_times = pd.to_datetime(np.round(beats_s * 1e9).astype("int64"), unit="ns") if len(beats_s) else pd.DatetimeIndex([])
    ibis = np.diff(beats_s) * 1000.0
    # Segment-level SQI gate only where the beat SQI screen is enabled.
    interval_sqi = _interval_sqi(beats_s, kept_s, kept_sqi) if np.isfinite(sqi_threshold) else None

    rhythm_hr = (harmonic_sum_hr(signal_filtered, fs)
                 if is_ppg and config.get("hrv", {}).get("rhythm_gate", True) else None)
    acc_peak_hz, acc_concentration = acc_periodicity(device_data.get("acc"), start_time, end_time)
    cadence_cfg = config.get("hrv", {}).get("cadence_lock", {})
    cadence_hz = (acc_peak_hz if is_ppg and cadence_cfg.get("enabled", True)
                  and acc_concentration >= float(cadence_cfg.get("min_concentration", 0.30)) else None)
    segments, exclusion_reasons = build_nn_segments(
        ibis, beat_times, config, reference_hr=device_hr,
        corrected=correction.corrected, interval_sqi=interval_sqi, rhythm_hr=rhythm_hr,
        corrected_beats=correction.altered_beats, cadence_hz=cadence_hz,
        uncertain=_uncertain_intervals(correction, config),
    )
    hrv = aggregate_hrv_metrics(segments, exclusion_reasons, config)
    hrv["hr_median"] = robust_hr(raw_ibis, tuple(config["processing"]["ibi_range"]))
    hrv["hr_tracked"] = hr_tracked
    hrv["hr_spectral"] = hr_spectral
    hrv["hr_rhythm_prior"] = rhythm_hr if rhythm_hr is not None else np.nan

    extras = {
        "detected_s": detected_s,
        "peak_times": detected_times,
        "beats_s": beats_s,
        "beats_corrected": correction.corrected,
        "segments": [_segment_dict(s) for s in segments],
        "method_peaks": {name: np.asarray(p) for name, p in peak_results.items()},
        "method_used": method_used,
        "clock_t0_s": t0_s,
        "clock_seconds": clock.seconds,
        "signal_filtered": signal_upright,
        "timing_anchor": anchor,
        "signal_repaired": signal_data,
        "signal_timing": -timing_signal if anchor == "onset" else timing_signal,
        "timing_peaks": timing_peaks,
    }

    results = {
        "pipeline_version": __version__,
        "device": device,
        "activity": activity_name,
        "method_used": method_used,
        "duration_sec": (end_time - start_time).total_seconds(),
        "fs_effective": clock.fs,
        "n_clock_gaps": clock.n_gaps,
        "n_peaks_detected": int(len(optimal_peaks)),
        "n_beats_sqi_rejected": int((~keep).sum()),
        "pct_beats_sqi_rejected": float(100.0 * np.mean(~keep)) if len(keep) else np.nan,
        "median_beat_sqi": float(np.nanmedian(sqi)) if np.any(np.isfinite(sqi)) else np.nan,
        "n_corrected_ectopic": correction.counts.get("ectopic", 0),
        "n_corrected_missed": correction.counts.get("missed", 0),
        "n_corrected_extra": correction.counts.get("extra", 0),
        "n_corrected_longshort": correction.counts.get("longshort", 0),
        "n_beats_recovered": correction.counts.get("recovered", 0),
        "n_beats_synthetic": int(np.sum(correction.altered_beats)) if correction.altered_beats is not None else 0,
        "pct_unreliable_samples": float(100.0 * np.mean(unreliable)),
        "acc_peak_hz": acc_peak_hz,
        "pulse_rise_fraction": pulse_asymmetry(
            signal_upright, np.round(locate_fiducial(signal_upright, optimal_peaks, fs, "peak", anchor=anchor)).astype(int))
        if is_ppg and len(optimal_peaks) > 5 else np.nan,
        "acc_periodicity": acc_concentration,
        "anc_applied": anc_applied,
        **motion_metrics,
        **quality_metrics,
        **hrv,
    }
    if np.isfinite(device_hr):
        results["device_hr"] = device_hr
        results["device_hr_std"] = device_hr_std
        if np.isfinite(results.get("mean_hr", np.nan)):
            results["hr_error_vs_device"] = abs(results["mean_hr"] - device_hr)

    # --- vendor IBI stream ---------------------------------------------------------------
    vendor_ibis, vendor_times = extract_device_ibis(
        device_data.get("ibi", pd.DataFrame()), device, start_time, end_time, config
    )
    vendor_ibis = np.asarray(vendor_ibis, dtype=float)
    vendor_s = to_seconds(vendor_times) if len(vendor_ibis) else np.array([])
    if len(vendor_ibis):
        order = np.argsort(vendor_s, kind="stable")
        vendor_s, vendor_ibis = vendor_s[order], vendor_ibis[order]
        unique = np.r_[True, np.diff(vendor_s) > 1e-6]
        vendor_s, vendor_ibis = vendor_s[unique], vendor_ibis[unique]
        results["vendor_n_ibis"] = int(len(vendor_ibis))
        results["vendor_hr_median"] = robust_hr(vendor_ibis)
    extras["vendor_s"] = vendor_s
    extras["vendor_ibis"] = vendor_ibis

    if device == "PolarH10" and len(vendor_ibis) > 5 and len(beats_s) > 5:
        # Reference validity: our NeuroKit2-derived R-R intervals versus the
        # Polar H10 firmware R-R values (an independent on-device detector).
        # The firmware values are exact but their timestamps are re-anchored
        # to BLE arrival, so the sequences are aligned by value.
        consistency = sequence_interval_agreement(np.diff(beats_s) * 1000.0, vendor_ibis)
        results.update({
            "reference_ibi_mae_ms": consistency.get("mae", np.nan),
            "reference_ibi_bias_ms": consistency.get("bias", np.nan),
            "reference_ibi_r": consistency.get("r", np.nan),
            "reference_within_5ms_pct": consistency.get("within_5ms_pct", np.nan),
            "reference_coverage_pct": consistency.get("coverage_pct", np.nan),
            "reference_ibi_n": consistency.get("n", 0),
        })

    return results, extras


def validate_against_ecg(results: dict, extras: dict, reference: dict, config: dict,
                         lag_center_s: float, lag_halfwidth_s: float) -> None:
    """Validate one PPG window against the ECG reference (updates *results*).

    The PPG-ECG delay is searched within ``lag_center_s +- lag_halfwidth_s``
    (see :func:`_anchored_lags`), so low-quality windows cannot select an
    arbitrary delay that maximises chance matches, nor slip by one beat.
    """
    validation_cfg = config.get("validation", {})
    tolerance = float(validation_cfg.get("match_tolerance_s", 0.10))
    detected_s = extras["detected_s"]
    beats_s = extras["beats_s"]
    vendor_s, vendor_ibis = extras.get("vendor_s", np.array([])), extras.get("vendor_ibis", np.array([]))
    ecg_s = reference["beats_s"]
    ecg_hr = reference["hr_median"]

    results["ecg_hr"] = ecg_hr
    if np.isfinite(results.get("hr_median", np.nan)) and np.isfinite(ecg_hr):
        results["hr_error_vs_ecg"] = abs(results["hr_median"] - ecg_hr)
        results["hr_signed_error_vs_ecg"] = results["hr_median"] - ecg_hr
        results["hr_error_percent_vs_ecg"] = results["hr_error_vs_ecg"] / ecg_hr * 100.0
        for key in ("hr_tracked", "hr_spectral"):
            if np.isfinite(results.get(key, np.nan)):
                results[f"hr_error_vs_ecg_{key.split('_')[1]}"] = abs(results[key] - ecg_hr)

    lag = estimate_lag(ecg_s, detected_s, max_lag_s=lag_halfwidth_s, center_s=lag_center_s)
    lag_s = lag["lag_s"]
    extras["lag_s"] = lag_s
    extras["lag_prior"] = (lag_center_s, lag_halfwidth_s)
    results["lag_ms"] = lag_s * 1000.0 if np.isfinite(lag_s) else np.nan
    results["lag_prominence"] = lag["prominence"]
    match = match_peaks(ecg_s, detected_s, tolerance, lag_s)
    results.update({
        "recall_vs_ecg": match["recall"],
        "precision_vs_ecg": match["precision"],
        "f1_vs_ecg": match["f1"],
        "timing_mad_ms_vs_ecg": match["median_absolute_offset_ms"],
        "timing_sd_ms_vs_ecg": match["timing_sd_ms"],
        "recall_chance_vs_ecg": surrogate_recall(ecg_s, detected_s, tolerance, lag_s),
        "recall_vs_ecg_tol50": match_peaks(ecg_s, detected_s, 0.05, lag_s)["recall"],
        "recall_vs_ecg_tol500_unaligned": match_peaks(ecg_s, detected_s, 0.5, 0.0)["recall"],
    })
    detection_ibi = beat_interval_agreement(ecg_s, detected_s, tolerance, lag_s)
    for key in ("bias", "sd", "mae", "rmse", "r", "n"):
        results[f"beat_ibi_{key}_vs_ecg"] = detection_ibi.get(key, np.nan)
    extras["ibi_pairs"] = detection_ibi if detection_ibi.get("n", 0) else None

    segments = extras.get("segments", [])
    if segments:
        in_segments = np.zeros(len(beats_s), dtype=bool)
        for seg in segments:
            a, b = to_seconds([seg["start"]])[0], to_seconds([seg["end"]])[0]
            in_segments |= (beats_s >= a - 1e-6) & (beats_s <= b + 1e-6)
        hrv_ibi = beat_interval_agreement(ecg_s, beats_s[in_segments], tolerance, lag_s)
        for key in ("bias", "sd", "mae", "rmse", "r", "n"):
            results[f"hrv_ibi_{key}_vs_ecg"] = hrv_ibi.get(key, np.nan)
        extras["hrv_ibi_pairs"] = hrv_ibi if hrv_ibi.get("n", 0) else None

    paired, n_paired = _paired_reference_metrics(segments, ecg_s, reference.get("beats_corrected"), lag_s, config)
    results["n_segments_paired"] = n_paired
    for metric in SEGMENT_METRIC_KEYS:
        value = paired.get(metric, np.nan)
        results[f"ecg_{metric}"] = value
        results[f"ecg_window_{metric}"] = reference.get("metrics", {}).get(metric, np.nan)
        own = results.get(metric, np.nan)
        if np.isfinite(own) and np.isfinite(value) and value != 0:
            results[f"{metric}_ratio_vs_ecg"] = own / value

    longest = max(segments, key=lambda s: len(s["nn"])) if segments else None
    if longest is not None and reference.get("nn") is not None:
        agreement = time_aligned_agreement(longest["nn"], longest["interval_end_times"],
                                           reference["nn"], reference["nn_times"],
                                           resample_hz=config["hrv"]["resample_hz"], lag_s=lag_s)
        for key in ("bias", "loa_lower", "loa_upper", "correlation", "rmse", "n_pairs"):
            results[f"nn_grid_{key}_vs_ecg"] = agreement.get(key, np.nan)

    results["vendor_ibi_coverage_pct"] = 0.0
    if len(vendor_s) > 5:
        # Vendor values are registered run-by-run (timestamps of the Galaxy
        # Watch batches are unreliable); the same procedure is used for both
        # vendors.  Coverage = registered vendor intervals / ECG intervals.
        vibi = register_vendor_runs(ecg_s, vendor_s, vendor_ibis)
        results.update({
            "vendor_ibi_mae_vs_ecg": vibi["mae"],
            "vendor_ibi_bias_vs_ecg": vibi["bias"],
            "vendor_ibi_r_vs_ecg": vibi["r"],
            "vendor_ibi_coverage_pct": vibi["coverage_pct"],
            "vendor_runs": vibi["n_runs"],
            "vendor_runs_accepted": vibi["n_runs_accepted"],
        })
        if np.isfinite(results.get("vendor_hr_median", np.nan)) and np.isfinite(ecg_hr):
            results["vendor_hr_error_vs_ecg"] = abs(results["vendor_hr_median"] - ecg_hr)


def _anchored_lags(windows: list[tuple[str, dict, dict]], ecg_reference: dict, key: str,
                   config: dict) -> list[tuple[float, float, str]]:
    """Per-window lag priors anchored on high-confidence windows.

    For each window the free correlogram delay (+-``max_lag_s``) is
    computed; windows whose correlogram prominence reaches
    ``validation.anchor_prominence`` are *anchors*.  Every window then gets a
    prior interpolated in time between anchors (constant beyond them) and a
    search half-width of ``min(anchor_halfwidth_s, 0.4 x ECG IBI)``, which
    prevents both noise-fitted delays and one-beat slips.  Without any
    anchor the free delay is used and flagged.

    Returns:
        List of ``(center_s, halfwidth_s, source)`` aligned with *windows*.
    """
    validation_cfg = config.get("validation", {})
    max_lag = float(validation_cfg.get("max_lag_s", 1.0))
    min_prominence = float(validation_cfg.get("anchor_prominence", 10.0))
    max_halfwidth = float(validation_cfg.get("anchor_halfwidth_s", 0.25))
    free, mids, anchors_t, anchors_lag = [], [], [], []
    for activity, _, extras in windows:
        reference = ecg_reference.get(activity)
        beats = extras.get(key, np.array([]))
        if reference is None or len(beats) < 5:
            free.append((np.nan, np.nan))
            mids.append(np.nan)
            continue
        lag = estimate_lag(reference["beats_s"], beats, max_lag_s=max_lag)
        free.append((lag["lag_s"], lag["prominence"]))
        mid = float(np.median(reference["beats_s"])) if len(reference["beats_s"]) else np.nan
        mids.append(mid)
        if np.isfinite(lag["lag_s"]) and lag["prominence"] >= min_prominence:
            anchors_t.append(mid)
            anchors_lag.append(lag["lag_s"])
    order = np.argsort(anchors_t)
    anchors_t, anchors_lag = np.asarray(anchors_t)[order], np.asarray(anchors_lag)[order]
    out = []
    for (activity, _, _), (free_lag, prominence), mid in zip(windows, free, mids, strict=False):
        reference = ecg_reference.get(activity)
        ibi = float(np.median(np.diff(reference["beats_s"]))) if reference is not None and len(reference["beats_s"]) > 2 else 0.8
        halfwidth = min(max_halfwidth, 0.4 * ibi)
        if len(anchors_t) and np.isfinite(mid):
            out.append((float(np.interp(mid, anchors_t, anchors_lag)), halfwidth,
                        "anchor" if np.isfinite(prominence) and prominence >= min_prominence else "interpolated"))
        else:
            out.append((free_lag if np.isfinite(free_lag) else 0.0, max_lag, "free"))
    return out


def _reference_from_ecg(results: dict, extras: dict) -> dict:
    """Pack what PPG validation needs from one ECG window."""
    segments = extras.get("segments", [])
    longest = max(segments, key=lambda s: len(s["nn"])) if segments else None
    return {
        "hr_median": results.get("hr_median", np.nan),
        "beats_s": extras.get("beats_s", np.array([])),
        "beats_corrected": extras.get("beats_corrected"),
        "metrics": {key: results.get(key, np.nan) for key in SEGMENT_METRIC_KEYS},
        "nn": longest["nn"] if longest else None,
        "nn_times": longest["interval_end_times"] if longest else None,
    }


def load_participant(user_folder: str, config: dict, return_raw: bool = False):
    """Load all devices once and align their clocks to the Polar H10.

    Returns:
        ``(device_data_by_device, clock_offsets)``; with *return_raw* also the
        unshifted device data (``shift_device_clock`` copies, so no reload).
    """
    data = {device: load_device_data(user_folder, device, config)
            for device in ("PolarH10", *PPG_DEVICES)}
    raw = {device: data[device] for device in PPG_DEVICES}
    validation_cfg = config.get("validation", {})
    offsets = {"PolarH10": {"offset_s": 0.0, "r": np.nan, "reliable": True}}
    for device in PPG_DEVICES:
        offset = estimate_clock_offset(
            data["PolarH10"].get("acc"), data[device].get("acc"), device,
            max_lag_s=float(validation_cfg.get("clock_sync_max_lag_s", 120.0)),
            min_correlation=float(validation_cfg.get("clock_sync_min_correlation", 0.3)),
        )
        offsets[device] = {"offset_s": offset.offset_s, "r": offset.correlation, "reliable": offset.reliable}
        if offset.reliable:
            data[device] = shift_device_clock(data[device], offset.offset_s)
    return (data, offsets, raw) if return_raw else (data, offsets)


def _task(args: tuple) -> tuple[dict, dict]:
    return process_activity(*args)


def process_participant(user_folder: str, config: dict, jobs: int = 1,
                        activities: list[str] | None = None) -> dict:
    """Run every window of one participant through the full chain.

    This is the single orchestration used by the cohort runner and by every
    study script (detector selection, fiducial study, window analysis,
    figures), so all of them share identical clock alignment, ECG reference
    and delay estimation.

    Args:
        user_folder: Participant directory.
        config: Parsed configuration.
        jobs: Worker processes within the participant.
        activities: Optional subset of activities (default: configured).

    Returns:
        Dict with ``windows`` (activity -> (start, end)), ``ecg`` (activity ->
        (results, extras)), ``reference`` (activity -> reference dict),
        ``ppg`` (``(device, activity)`` -> (results, extras), validated) and
        ``offsets`` (session offsets).  Empty dict when ``Event.csv`` is
        missing.
    """
    events_path = os.path.join(user_folder, "Event.csv")
    if not os.path.exists(events_path):
        logger.error("Events file not found: %s", events_path)
        return {}
    events = pd.read_csv(events_path)
    events["datetime"] = pd.to_datetime(events["timestamp"], unit="ms")
    activities = activities or config.get(
        "activities", ["baseline", "keyboard-typing", "rest-1", "walking", "jogging"])
    windows = extract_activity_windows(events, activities)

    data, offsets, raw = load_participant(user_folder, config, return_raw=True)
    validation_cfg = config.get("validation", {})
    local: dict[tuple[str, str], dict] = {}
    local_enabled = bool(validation_cfg.get("local_sync_enabled", True))
    for device in PPG_DEVICES:
        for activity, (start, end) in windows.items():
            if not local_enabled:  # ablation: a single session offset for every window
                local[(device, activity)] = {"offset_s": offsets[device]["offset_s"], "r": np.nan, "reliable": False}
                continue
            estimate = estimate_local_offset(
                data["PolarH10"].get("acc"), raw[device].get("acc"), device, start, end,
                offsets[device]["offset_s"],
                context_s=float(validation_cfg.get("local_sync_context_s", 300.0)),
                search_s=float(validation_cfg.get("local_sync_search_s", 10.0)),
                min_correlation=float(validation_cfg.get("clock_sync_min_correlation", 0.3)),
            )
            local[(device, activity)] = {"offset_s": estimate.offset_s, "r": estimate.correlation,
                                         "reliable": estimate.reliable}
    # Robust per-window offsets: a local estimate is kept when other windows of
    # the participant agree with it (a real mid-session clock step is supported
    # by every later window; a mis-lock is isolated), otherwise the nearest
    # accepted window is used.
    tolerance = float(validation_cfg.get("local_sync_tolerance_s", 1.0))
    support = int(validation_cfg.get("local_sync_min_support", 2))
    midpoints = {a: ((start - pd.Timestamp("1970-01-01")).total_seconds()
                     + (end - pd.Timestamp("1970-01-01")).total_seconds()) / 2.0
                 for a, (start, end) in windows.items()}
    for device in PPG_DEVICES:
        keys_d = [k for k in local if k[0] == device]
        resolved, sources = resolve_window_offsets(
            [local[k]["offset_s"] if local[k]["reliable"] else None for k in keys_d],
            [midpoints[k[1]] for k in keys_d], offsets[device]["offset_s"],
            tolerance_s=tolerance, min_support=support)
        for k, value, source in zip(keys_d, resolved, sources, strict=True):
            local[k].update({"applied_s": value, "source": source, "used_local": source == "local"})

    def _run(tasks: list[tuple]) -> list[tuple[dict, dict]]:
        if jobs > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                return list(pool.map(_task, tasks))
        return [_task(t) for t in tasks]

    ecg_out = dict(zip(windows, _run([(user_folder, "PolarH10", s, e, a, config, None, data["PolarH10"])
                                       for a, (s, e) in windows.items()]), strict=True))
    reference = {a: _reference_from_ecg(r, x) for a, (r, x) in ecg_out.items()
                 if r and np.isfinite(r.get("hr_median", np.nan))}

    def _applied_offset(device: str, activity: str) -> tuple[float, str]:
        entry = local[(device, activity)]
        if offsets[device]["reliable"] or entry["reliable"]:
            return entry["applied_s"], entry["source"]
        return 0.0, "none"

    def _aligned(device: str, activity: str) -> dict:
        applied, source = _applied_offset(device, activity)
        return shift_device_clock(raw[device], applied) if source != "none" else data[device]

    keys = [(d, a) for a in windows for d in PPG_DEVICES]
    ppg_out = _run([(user_folder, d, *windows[a], a, config, None, _aligned(d, a)) for d, a in keys])
    ppg = {key: out for key, out in zip(keys, ppg_out, strict=True) if out[0]}

    for device in PPG_DEVICES:
        entries = [(a, *ppg[(d, a)]) for d, a in keys if d == device and (d, a) in ppg]
        priors = _anchored_lags(entries, reference, "detected_s", config)
        for (activity, results, extras), prior in zip(entries, priors, strict=True):
            results["lag_source"] = prior[2]
            results["lag_prior_ms"] = prior[0] * 1000.0
            if activity in reference:
                validate_against_ecg(results, extras, reference[activity], config, prior[0], prior[1])
            applied, source = _applied_offset(device, activity)
            results["clock_offset_s"] = applied
            results["clock_offset_source"] = source
            results["clock_offset_session_s"] = offsets[device]["offset_s"]
            results["clock_sync_session_r"] = offsets[device]["r"]
            results["clock_sync_session_reliable"] = offsets[device]["reliable"]
            results["clock_sync_local"] = local[(device, activity)]["used_local"]
            results["clock_sync_local_r"] = local[(device, activity)]["r"]
            results["clock_local_estimate_s"] = (local[(device, activity)]["offset_s"]
                                                 if local[(device, activity)]["reliable"] else np.nan)

    return {"windows": windows, "ecg": ecg_out, "reference": reference, "ppg": ppg, "offsets": offsets,
            "order": keys}


def run_pipeline(user_folder: str, config: dict, jobs: int = 1,
                 return_extras: bool = False):
    """Run the full validation pipeline for one participant.

    Result rows are returned in serial order (ECG activities, then PPG
    activity-major / device-minor) regardless of ``jobs``.

    Args:
        user_folder: Participant directory.
        config: Parsed configuration.
        jobs: Worker processes within the participant.
        return_extras: Also return the dense per-window PPG outputs keyed by
            ``(device, activity)``.

    Returns:
        ``(results_df, ecg_reference)`` or, with *return_extras*,
        ``(results_df, ecg_reference, ppg_extras)``.
    """
    out = process_participant(user_folder, config, jobs=jobs)
    if not out:
        return (pd.DataFrame(), {}, {}) if return_extras else (pd.DataFrame(), {})
    rows = [r for a, (r, _) in out["ecg"].items() if a in out["reference"]]
    rows += [out["ppg"][key][0] for key in out["order"] if key in out["ppg"]]
    frame = pd.DataFrame(rows)
    if return_extras:
        return frame, out["reference"], {key: x for key, (_, x) in out["ppg"].items()}
    return frame, out["reference"]


def main():
    parser = argparse.ArgumentParser(description="GalaxyPPG HRV Pipeline - single participant")
    parser.add_argument("--participant", required=True, help="Participant ID (e.g. P02)")
    parser.add_argument("--data-root", default=None, help="Dataset root directory")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--jobs", type=int, default=1, help="Worker processes within a participant")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = load_config(args.config or os.path.join(os.path.dirname(__file__), "config.yaml"))
    data_root = args.data_root or os.environ.get("PPG_DATA_ROOT") or config["dataset"]["root"]
    if not data_root:
        logger.error("No data root specified. Use --data-root, PPG_DATA_ROOT env var, or config.yaml.")
        sys.exit(1)
    user_folder = os.path.join(data_root, args.participant)
    if not os.path.exists(user_folder):
        logger.error("Participant folder not found: %s", user_folder)
        sys.exit(1)

    results_df, _ = run_pipeline(user_folder, config, jobs=args.jobs)
    if results_df.empty:
        logger.warning("No results generated for %s", args.participant)
        sys.exit(1)
    from run_all_participants import participant_split
    results_df.insert(0, "split", participant_split(args.participant, config))
    results_df.insert(0, "participant", args.participant)
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{args.participant}_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info("Results saved to %s (%d rows)", csv_path, len(results_df))


if __name__ == "__main__":
    main()
