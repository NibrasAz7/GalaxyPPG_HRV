"""Beat-level validation against the ECG reference.

All comparisons are made on a common *absolute* time base:

1. Device clocks are first aligned to the Polar H10 at session level from
   the accelerometers (``src.sync``).
2. Within each recording, the residual delay between ECG R-peaks and PPG
   systolic peaks - the pulse arrival time (PAT, ~150-350 ms at the wrist)
   plus any residual clock error - is estimated with the **beat
   cross-correlogram**: the distribution of all PPG-minus-ECG beat-time
   differences within +-``max_lag_s``.  True beat pairs pile up at the PAT;
   unrelated beats are spread uniformly, so the correlogram peak is the lag
   and its prominence (peak / background) measures alignment confidence.
3. Beats are paired one-to-one within a tolerance that is well below half
   an inter-beat interval (default +-100 ms after lag removal).  With the
   previous +-500 ms window, any beat train at the right rate scored
   ~100 % recall regardless of phase; ``surrogate_recall`` quantifies that
   chance level explicitly.
4. Beat-to-beat interval agreement uses only intervals whose two bounding
   beats are both matched, so every compared pair spans the same heartbeat.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

_EPOCH = pd.Timestamp("1970-01-01")


def to_seconds(times) -> np.ndarray:
    """Absolute timestamps -> float seconds since the Unix epoch."""
    if times is None:
        return np.array([], dtype=float)
    if isinstance(times, np.ndarray) and times.dtype.kind == "f":
        return times.astype(float)
    idx = pd.DatetimeIndex(times)
    return (idx - _EPOCH).total_seconds().to_numpy(dtype=float)


def estimate_lag(
    reference_times,
    detected_times,
    max_lag_s: float = 1.0,
    resolution_s: float = 0.005,
    bandwidth_s: float = 0.015,
    center_s: float = 0.0,
) -> dict[str, float]:
    """Estimate the delay of detected beats relative to reference beats.

    Args:
        reference_times: Reference (ECG R-peak) times.
        detected_times: Detected (PPG) beat times on the same time base.
        max_lag_s: Search half-range in seconds around ``center_s``.
        resolution_s: Lag grid resolution.
        bandwidth_s: Gaussian kernel width of the correlogram.
        center_s: Centre of the search range (a prior lag).

    Returns:
        Dict with ``lag_s`` (detected minus reference), ``prominence``
        (correlogram peak over its median) and ``n_pairs`` (differences used).
    """
    ref = np.sort(to_seconds(reference_times))
    det = np.sort(to_seconds(detected_times))
    out = {"lag_s": np.nan, "prominence": np.nan, "n_pairs": 0}
    if len(ref) < 5 or len(det) < 5:
        return out

    center = 0.0 if not np.isfinite(center_s) else float(center_s)
    lo = np.searchsorted(ref, det - center - max_lag_s)
    hi = np.searchsorted(ref, det - center + max_lag_s)
    diffs = np.concatenate([det[i] - ref[a:b] for i, (a, b) in enumerate(zip(lo, hi, strict=False)) if b > a]) \
        if np.any(hi > lo) else np.array([])
    if len(diffs) < 5:
        return out

    edges = center + np.arange(-max_lag_s, max_lag_s + resolution_s, resolution_s)
    hist, _ = np.histogram(diffs, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    sigma = bandwidth_s / resolution_s
    k = np.arange(-int(4 * sigma), int(4 * sigma) + 1)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    density = np.convolve(hist.astype(float), kernel / kernel.sum(), "same")
    best = int(np.argmax(density))
    background = float(np.median(density))
    out["lag_s"] = float(centres[best])
    out["prominence"] = float(density[best] / background) if background > 0 else np.inf
    out["n_pairs"] = int(len(diffs))
    return out


def pair_beats(reference_s: np.ndarray, detected_s: np.ndarray, tolerance_s: float) -> np.ndarray:
    """One-to-one nearest pairing within a tolerance.

    Each detected beat proposes its nearest reference beat; a reference beat
    claimed by several detected beats keeps the closest.  Because the
    tolerance is below half an inter-beat interval, this greedy-nearest
    rule yields the optimal assignment.

    Returns:
        Array of shape ``(n_pairs, 2)`` with ``(ref_index, det_index)``.
    """
    if len(reference_s) == 0 or len(detected_s) == 0:
        return np.empty((0, 2), dtype=int)
    idx = np.searchsorted(reference_s, detected_s)
    left = np.clip(idx - 1, 0, len(reference_s) - 1)
    right = np.clip(idx, 0, len(reference_s) - 1)
    d_left = np.abs(detected_s - reference_s[left])
    d_right = np.abs(detected_s - reference_s[right])
    nearest = np.where(d_left <= d_right, left, right)
    distance = np.minimum(d_left, d_right)
    candidates = np.where(distance <= tolerance_s)[0]
    best: dict[int, tuple[float, int]] = {}
    for j in candidates:
        r = int(nearest[j])
        if r not in best or distance[j] < best[r][0]:
            best[r] = (float(distance[j]), int(j))
    pairs = np.array(sorted((r, j) for r, (_, j) in best.items()), dtype=int)
    return pairs.reshape(-1, 2)


def match_peaks(
    reference_times,
    detected_times,
    tolerance_s: float = 0.1,
    lag_s: float = 0.0,
) -> dict[str, float]:
    """Match detected beats to reference beats after removing a known lag.

    Args:
        reference_times: Reference beat timestamps (ECG R-peaks).
        detected_times: Detected beat timestamps (same time base).
        tolerance_s: Matching tolerance after lag removal (seconds).
        lag_s: Delay of detected beats relative to reference beats.

    Returns:
        Dict with ``recall``, ``precision``, ``f1`` (percent),
        ``median_absolute_offset_ms`` (after lag removal),
        ``median_signed_offset_ms`` (raw detected-minus-reference delay),
        ``timing_sd_ms`` (robust SD of residual timing error), and counts.
    """
    ref = np.sort(to_seconds(reference_times))
    det = np.sort(to_seconds(detected_times))
    n_ref, n_det = len(ref), len(det)
    if n_ref == 0 or n_det == 0:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0,
                "median_absolute_offset_ms": np.nan, "median_signed_offset_ms": np.nan,
                "timing_sd_ms": np.nan, "n_reference": n_ref, "n_detected": n_det, "n_matched": 0}

    lag = 0.0 if not np.isfinite(lag_s) else float(lag_s)
    pairs = pair_beats(ref, det - lag, tolerance_s)
    n_matched = len(pairs)
    recall = 100.0 * n_matched / n_ref
    precision = 100.0 * n_matched / n_det
    f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
    residual = (det[pairs[:, 1]] - lag - ref[pairs[:, 0]]) if n_matched else np.array([])
    signed = residual + lag
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "median_absolute_offset_ms": float(np.median(np.abs(residual)) * 1000) if n_matched else np.nan,
        "median_signed_offset_ms": float(np.median(signed) * 1000) if n_matched else np.nan,
        "timing_sd_ms": float(1.4826 * np.median(np.abs(residual - np.median(residual))) * 1000)
        if n_matched > 2 else np.nan,
        "n_reference": n_ref,
        "n_detected": n_det,
        "n_matched": n_matched,
    }


def surrogate_recall(reference_times, detected_times, tolerance_s: float = 0.1,
                     lag_s: float = 0.0) -> float:
    """Chance-level recall: detected beats shifted by half a median interval.

    A detector that outputs a beat train at the correct rate but random
    phase would score this recall; real performance must exceed it.
    """
    det = np.sort(to_seconds(detected_times))
    if len(det) < 3:
        return np.nan
    half = 0.5 * float(np.median(np.diff(det)))
    return match_peaks(reference_times, det + half, tolerance_s, lag_s)["recall"]


def beat_interval_agreement(
    reference_times,
    detected_times,
    tolerance_s: float = 0.1,
    lag_s: float = 0.0,
) -> dict:
    """Beat-to-beat IBI agreement on intervals bounded by two matched beats.

    Returns:
        Dict with ``bias``, ``sd``, ``loa_lower``, ``loa_upper``, ``mae``,
        ``rmse``, ``r`` (ms), ``n`` and the paired arrays ``ibi_reference``
        / ``ibi_detected`` (ms) and ``times`` (reference interval end, s).
    """
    ref = np.sort(to_seconds(reference_times))
    det = np.sort(to_seconds(detected_times))
    lag = 0.0 if not np.isfinite(lag_s) else float(lag_s)
    pairs = pair_beats(ref, det - lag, tolerance_s)
    if len(pairs) < 3:
        return {"n": 0}
    consecutive = (np.diff(pairs[:, 0]) == 1) & (np.diff(pairs[:, 1]) == 1)
    if consecutive.sum() < 3:
        return {"n": int(consecutive.sum())}
    r0 = pairs[:-1, 0][consecutive]
    d0 = pairs[:-1, 1][consecutive]
    ibi_ref = (ref[r0 + 1] - ref[r0]) * 1000.0
    ibi_det = (det[d0 + 1] - det[d0]) * 1000.0
    diff = ibi_det - ibi_ref
    sd = float(np.std(diff, ddof=1))
    r = float(pearsonr(ibi_det, ibi_ref)[0]) if np.std(ibi_ref) > 0 and np.std(ibi_det) > 0 else np.nan
    return {
        "bias": float(np.mean(diff)),
        "sd": sd,
        "loa_lower": float(np.mean(diff) - 1.96 * sd),
        "loa_upper": float(np.mean(diff) + 1.96 * sd),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "r": r,
        "n": int(len(diff)),
        "ibi_reference": ibi_ref,
        "ibi_detected": ibi_det,
        "times": ref[r0 + 1],
    }


def align_interval_sequences(
    reference_ms: np.ndarray,
    other_ms: np.ndarray,
    cap_ms: float = 50.0,
    gap_cost: float = 0.6,
) -> np.ndarray:
    """Align two interval sequences by value (dynamic programming).

    A global alignment (Needleman-Wunsch) with substitution cost
    ``min(|a - b|, cap_ms) / cap_ms`` and a constant gap cost pairs the two
    series without using timestamps, tolerating dropped or extra intervals
    on either side.  Used for the Polar H10 firmware R-R stream, whose
    values are exact but whose timestamps are re-anchored to BLE arrival.

    Returns:
        Array ``(n_pairs, 2)`` of ``(reference_index, other_index)``.
    """
    a = np.asarray(reference_ms, dtype=float)
    b = np.asarray(other_ms, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.empty((0, 2), dtype=int)
    cost = np.minimum(np.abs(a[:, None] - b[None, :]), cap_ms) / cap_ms
    score = np.zeros((n + 1, m + 1))
    score[1:, 0] = gap_cost * np.arange(1, n + 1)
    score[0, 1:] = gap_cost * np.arange(1, m + 1)
    move = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        diag = score[i - 1, :-1] + cost[i - 1]
        up = score[i - 1, 1:] + gap_cost
        for j in range(1, m + 1):
            left = score[i, j - 1] + gap_cost
            best = diag[j - 1]
            choice = 0
            if up[j - 1] < best:
                best, choice = up[j - 1], 1
            if left < best:
                best, choice = left, 2
            score[i, j] = best
            move[i, j] = choice
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        if move[i, j] == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return np.array(pairs[::-1], dtype=int).reshape(-1, 2)


def sequence_interval_agreement(reference_ms: np.ndarray, other_ms: np.ndarray,
                                cap_ms: float = 50.0) -> dict:
    """Value-aligned interval agreement (see :func:`align_interval_sequences`).

    Returns:
        Dict with ``mae``, ``bias``, ``r``, ``within_5ms_pct``,
        ``within_10ms_pct`` (over aligned pairs whose difference is below
        ``cap_ms``), ``n`` and ``coverage_pct`` (aligned pairs / reference).
    """
    pairs = align_interval_sequences(reference_ms, other_ms, cap_ms=cap_ms)
    if len(pairs) < 3:
        return {"n": int(len(pairs))}
    a = np.asarray(reference_ms, dtype=float)[pairs[:, 0]]
    b = np.asarray(other_ms, dtype=float)[pairs[:, 1]]
    diff = b - a
    ok = np.abs(diff) < cap_ms
    a, b, diff = a[ok], b[ok], diff[ok]
    if len(diff) < 3:
        return {"n": int(len(diff))}
    r = float(pearsonr(a, b)[0]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
    return {"mae": float(np.mean(np.abs(diff))), "bias": float(np.mean(diff)), "r": r,
            "within_5ms_pct": float(100 * np.mean(np.abs(diff) <= 5)),
            "within_10ms_pct": float(100 * np.mean(np.abs(diff) <= 10)),
            "n": int(len(diff)), "coverage_pct": float(100 * len(diff) / len(reference_ms))}


def register_vendor_runs(
    reference_times,
    vendor_end_times,
    vendor_ibis_ms: np.ndarray,
    search_s: float = 8.0,
    min_run: int = 15,
    max_mae_ms: float = 30.0,
    distinct_ratio: float = 0.6,
    contiguity_ms: float = 60.0,
    chunk: int = 20,
) -> dict:
    """Register vendor IBI runs to the reference by value, within a time window.

    Vendor IBI streams carry exact interval values but unreliable timestamps
    (the Galaxy Watch stamps whole batches; the Polar stamps BLE arrival).
    Each contiguous vendor run (consecutive end times separated by their
    interval, within ``contiguity_ms``) is cut into chunks of ``chunk``
    intervals (the last chunk absorbs a remainder; chunks shorter than
    ``min_run`` are skipped), and each chunk is slid along the reference
    interval series, start beats within
    ``+-search_s`` of its nominal start.  A run is accepted when the best
    shift has MAE below ``max_mae_ms`` *and* is clearly better than the
    second-best shift (``best <= distinct_ratio x second``), which rejects
    ambiguous registrations; a shuffled-control run is essentially never
    accepted.

    Returns:
        Dict with ``mae``, ``bias``, ``rmse``, ``r``, ``n`` (registered
        intervals), ``coverage_pct`` (of reference intervals), ``n_runs`` and
        ``n_runs_accepted``.
    """
    ref = np.sort(to_seconds(reference_times))
    out = {"mae": np.nan, "bias": np.nan, "rmse": np.nan, "r": np.nan, "n": 0,
           "coverage_pct": 0.0, "n_runs": 0, "n_runs_accepted": 0}
    if len(ref) < min_run + 2:
        return out
    ref_ibi = np.diff(ref) * 1000.0
    end_s = to_seconds(vendor_end_times)
    values = np.asarray(vendor_ibis_ms, dtype=float)
    order = np.argsort(end_s)
    end_s, values = end_s[order], values[order]
    if len(values) < min_run:
        return out
    breaks = np.where(np.abs(np.diff(end_s) * 1000.0 - values[1:]) > contiguity_ms)[0] + 1
    runs = []
    for run in np.split(np.arange(len(values)), breaks):
        pieces = [run[i:i + chunk] for i in range(0, len(run), chunk)]
        if len(pieces) > 1 and len(pieces[-1]) < min_run:
            pieces[-2] = np.concatenate([pieces[-2], pieces[-1]])
            pieces = pieces[:-1]
        runs.extend(pieces)
    used = np.zeros(len(ref_ibi), dtype=bool)
    v_all, r_all = [], []
    for run in runs:
        if len(run) < min_run:
            continue
        out["n_runs"] += 1
        v = values[run]
        start_nominal = end_s[run[0]] - v[0] / 1000.0
        candidates = np.where(np.abs(ref[:-1] - start_nominal) <= search_s)[0]
        candidates = candidates[candidates + len(v) <= len(ref_ibi)]
        if len(candidates) < 2:
            continue
        maes = np.array([np.mean(np.abs(v - ref_ibi[j: j + len(v)])) for j in candidates])
        best = int(np.argmin(maes))
        # Smooth (autocorrelated) IBI series make the adjacent shifts nearly
        # as good; distinctness is therefore tested against shifts at least
        # two beats away, and the best of the neighbourhood is kept.
        far = np.abs(candidates - candidates[best]) >= 2
        second = float(np.min(maes[far])) if far.any() else np.inf
        if maes[best] <= max_mae_ms and maes[best] <= distinct_ratio * second:
            j = candidates[best]
            out["n_runs_accepted"] += 1
            span = slice(j, j + len(v))
            fresh = ~used[span]
            used[span] = True
            v_all.append(v[fresh])
            r_all.append(ref_ibi[span][fresh])
    if not v_all:
        return out
    v_all, r_all = np.concatenate(v_all), np.concatenate(r_all)
    diff = v_all - r_all
    r = float(pearsonr(v_all, r_all)[0]) if len(diff) > 3 and np.std(v_all) > 0 and np.std(r_all) > 0 else np.nan
    out.update({"mae": float(np.mean(np.abs(diff))), "bias": float(np.mean(diff)),
                "rmse": float(np.sqrt(np.mean(diff ** 2))), "r": r, "n": int(len(diff)),
                "coverage_pct": float(100.0 * len(diff) / len(ref_ibi))})
    return out


def time_aligned_agreement(
    nn_a: np.ndarray,
    times_a,
    nn_b: np.ndarray,
    times_b,
    resample_hz: float = 4.0,
    lag_s: float = 0.0,
) -> dict[str, float]:
    """Agreement between two NN series on a common absolute time grid.

    Both series are interpolated onto a uniform grid covering their *true*
    temporal overlap (absolute times; ``times_b`` is shifted by ``lag_s``
    before comparison).  The previous implementation re-referenced each
    series to its own first beat, so series starting at different times
    were compared out of phase.

    Args:
        nn_a: First NN series (ms).
        times_a: Timestamps of *nn_a* (interval end times).
        nn_b: Second NN series (ms).
        times_b: Timestamps of *nn_b*.
        resample_hz: Uniform grid rate in Hz.
        lag_s: Delay of series *a* relative to series *b* (seconds).

    Returns:
        Dict with ``bias``, ``std_diff``, ``loa_upper``, ``loa_lower``,
        ``correlation``, ``rmse`` and ``n_pairs``; empty when the series do
        not overlap sufficiently.
    """
    if nn_a is None or nn_b is None or times_a is None or times_b is None:
        return {}
    if len(nn_a) < 4 or len(nn_b) < 4:
        return {}
    seconds_a = to_seconds(times_a) - (0.0 if not np.isfinite(lag_s) else lag_s)
    seconds_b = to_seconds(times_b)
    start = max(seconds_a[0], seconds_b[0])
    end = min(seconds_a[-1], seconds_b[-1])
    if end - start < 1.0:
        return {}
    grid = np.arange(start, end, 1.0 / resample_hz)
    if len(grid) < 8:
        return {}
    try:
        values_a = PchipInterpolator(seconds_a, nn_a)(grid)
        values_b = PchipInterpolator(seconds_b, nn_b)(grid)
    except ValueError as exc:
        logger.debug("Time-aligned agreement failed: %s", exc)
        return {}
    differences = values_a - values_b
    bias = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))
    correlation = float(pearsonr(values_a, values_b)[0]) if np.std(values_a) > 0 and np.std(values_b) > 0 else np.nan
    return {
        "bias": bias,
        "std_diff": std_diff,
        "loa_upper": bias + 1.96 * std_diff,
        "loa_lower": bias - 1.96 * std_diff,
        "correlation": correlation,
        "rmse": float(np.sqrt(np.mean(differences ** 2))),
        "n_pairs": len(grid),
    }
