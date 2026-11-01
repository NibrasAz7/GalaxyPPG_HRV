"""Beat-timing error series of PPG against the ECG reference.

For every window, detected PPG beats are paired one-to-one with ECG R-peaks
(after removing the window's delay).  The pairing error
``e_i = t_ppg,i - t_ecg,i - delay`` (ms) is a *timing error series*:

* ``sigma`` - its SD within the window (static offset removed);
* ``rho1`` - lag-1 autocorrelation (smoothness of the error);
* ``eps_rmssd`` - RMS of its second differences ``e[i+2] - 2 e[i+1] + e[i]``.
  This is exactly the contribution of the timing error to the successive
  differences of the PPG inter-beat intervals, so for uncorrelated
  (white) timing errors ``eps_rmssd = sqrt(6) * sigma``;
* ``whiteness`` - ``eps_rmssd / (sqrt(6) * sigma)``: 1 for white errors,
  <1 when errors drift smoothly (e.g. beat-to-beat pulse arrival time
  variation), because smooth errors cancel in second differences.

The decomposition ``RMSSD_ppg^2 ~ RMSSD_ecg^2 + eps_rmssd^2`` holds when the
timing error is uncorrelated with the ECG's own successive differences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation import pair_beats


def _runs(ref_idx: np.ndarray, det_idx: np.ndarray) -> list[np.ndarray]:
    """Index groups of pairs that are consecutive in both beat trains."""
    if len(ref_idx) == 0:
        return []
    breaks = np.where((np.diff(ref_idx) != 1) | (np.diff(det_idx) != 1))[0] + 1
    return np.split(np.arange(len(ref_idx)), breaks)


def window_timing_error(detected_s: np.ndarray, ecg_s: np.ndarray, lag_s: float,
                        segments: list[tuple[float, float]] | None = None,
                        tolerance_s: float = 0.10, min_run: int = 8) -> dict[str, float]:
    """Timing-error statistics of one window (restricted to accepted segments if given)."""
    det = np.sort(np.asarray(detected_s, dtype=float))
    ecg = np.sort(np.asarray(ecg_s, dtype=float))
    out = {"n_beats": 0, "sigma_ms": np.nan, "rho1": np.nan, "eps_rmssd_ms": np.nan, "whiteness": np.nan}
    if len(det) < 10 or len(ecg) < 10 or not np.isfinite(lag_s):
        return out
    pairs = pair_beats(ecg, det - lag_s, tolerance_s)
    if len(pairs) < min_run:
        return out
    if segments:
        t = det[pairs[:, 1]]
        keep = np.zeros(len(pairs), dtype=bool)
        for a, b in segments:
            keep |= (t >= a - 1e-6) & (t <= b + 1e-6)
        pairs = pairs[keep]
    if len(pairs) < min_run:
        return out
    err = (det[pairs[:, 1]] - lag_s - ecg[pairs[:, 0]]) * 1000.0
    second, products, firsts, count = [], [], [], 0
    centred = err - np.mean(err)
    for run in _runs(pairs[:, 0], pairs[:, 1]):
        if len(run) < min_run:
            continue
        e = err[run]
        count += len(run)
        second.append(e[2:] - 2 * e[1:-1] + e[:-2])
        c = centred[run]
        products.append(c[1:] * c[:-1])
        firsts.append(c[:-1] ** 2)
    if not second:
        return out
    second = np.concatenate(second)
    sigma = float(np.std(centred, ddof=1))
    eps = float(np.sqrt(np.mean(second ** 2)))
    out.update({
        "n_beats": int(count), "sigma_ms": sigma, "eps_rmssd_ms": eps,
        "rho1": float(np.sum(np.concatenate(products)) / max(np.sum(np.concatenate(firsts)), 1e-9)),
        "whiteness": eps / (np.sqrt(6.0) * sigma) if sigma > 0 else np.nan,
    })
    return out


def cache_timing_errors(cache: dict, combined: pd.DataFrame) -> pd.DataFrame:
    """Timing-error table for every cached window, joined with the HRV results.

    Args:
        cache: Beat cache from ``analysis.figures.load_beat_cache``.
        combined: ``combined_results.csv`` (``load_combined``).

    Returns:
        One row per participant x device x activity with timing-error
        statistics and the PPG / ECG RMSSD of the accepted spans.
    """
    rows = []
    for participant, data in cache.items():
        for (device, activity), w in data["windows"].items():
            stats = window_timing_error(w["detected_s"], w["ecg_s"], w["lag_s"], w.get("segments") or None)
            rows.append({"participant": participant, "split": data["split"], "device": device,
                         "activity": activity, **stats})
    frame = pd.DataFrame(rows)
    keep = combined[combined["device"].isin(["E4", "GalaxyWatch"])][
        ["participant", "device", "activity", "condition", "rmssd", "ecg_rmssd", "n_segments"]]
    return frame.merge(keep, on=["participant", "device", "activity"], how="left")
