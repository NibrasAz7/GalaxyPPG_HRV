"""Validation metrics: physiological peak validation, PMR, and Bland-Altman.

Provides ground-truth comparison utilities used to assess the pipeline's
output against ECG-derived reference data:

- **Physiological peak validation** — removes peaks that produce IBIs
  outside the 30–160 bpm range.
- **Peak matching rate (PMR)** — percentage of pipeline-detected peaks
  that match a device-reported IBI timestamp within a tolerance window.
- **Bland-Altman analysis** — bias, limits of agreement, and correlation
  between two measurement methods.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def validate_peaks_physiologically(
    peaks: np.ndarray,
    fs: float,
    config: dict,
) -> np.ndarray:
    """Remove peaks that produce physiologically implausible IBIs.

    Computes successive IBIs from peak indices and keeps only those
    falling within the configured bpm range (default 30–160 bpm).

    Args:
        peaks: Array of peak sample indices.
        fs: Sampling frequency in Hz.
        config: Parsed configuration dictionary.

    Returns:
        Filtered array of peak indices (always keeps the first peak).
    """
    if len(peaks) < 2:
        return peaks

    bpm_min, bpm_max = config["processing"]["peak_bpm_range"]
    ibi_min_ms = 60000 / bpm_max  # shortest IBI (fastest HR)
    ibi_max_ms = 60000 / bpm_min  # longest IBI (slowest HR)

    ibis_ms = np.diff(peaks) / fs * 1000
    valid = (ibis_ms >= ibi_min_ms) & (ibis_ms <= ibi_max_ms)

    result = [peaks[0]]
    for i in range(len(valid)):
        if valid[i]:
            result.append(peaks[i + 1])

    return np.array(result)


def calculate_pmr(
    custom_peaks: np.ndarray,
    device_ibi_times: np.ndarray,
    signal_timestamps: np.ndarray,
    config: dict,
) -> float:
    """Compute the peak matching rate (PMR) between pipeline and device peaks.

    For each pipeline-detected peak, checks whether a device-reported IBI
    timestamp exists within a tolerance window (default 500 ms).  PMR is
    the percentage of matched peaks.

    Args:
        custom_peaks: Pipeline-detected peak indices.
        device_ibi_times: Device-reported IBI timestamps.
        signal_timestamps: Full signal timestamp array (for index → time mapping).
        config: Parsed configuration dictionary.

    Returns:
        PMR as a percentage (0–100).  Returns 0.0 if either input is empty.
    """
    if len(custom_peaks) == 0 or len(device_ibi_times) == 0:
        return 0.0

    try:
        if len(signal_timestamps) <= max(custom_peaks):
            return 0.0

        custom_times = pd.to_datetime(signal_timestamps[custom_peaks])
        if not isinstance(device_ibi_times, pd.DatetimeIndex):
            device_ibi_times = pd.to_datetime(device_ibi_times)

        tolerance = pd.Timedelta(seconds=config["processing"]["pmr_tolerance"])
        matched = sum(
            1 for ct in custom_times
            if np.any(np.abs(device_ibi_times - ct) <= tolerance)
        )
        return matched / len(custom_peaks) * 100

    except Exception as e:
        logger.warning("PMR calculation error: %s", e)
        return 0.0


def bland_altman_analysis(method1: np.ndarray, method2: np.ndarray) -> Dict:
    """Perform Bland-Altman agreement analysis between two measurement methods.

    Computes:

    - **Mean difference (bias)**: systematic offset between methods.
    - **Std of differences**: random error.
    - **Limits of agreement**: mean ± 1.96 × std (95% CI).
    - **Pearson correlation** and p-value.

    Args:
        method1: Values from method 1 (e.g. pipeline IBIs).
        method2: Values from method 2 (e.g. ECG IBIs).

    Returns:
        Dict with ``mean_difference``, ``std_difference``, ``loa_upper``,
        ``loa_lower``, ``correlation``, ``p_value``, ``differences``,
        ``means``, ``n_pairs``.  Empty dict if either input is empty.
    """
    if len(method1) == 0 or len(method2) == 0:
        return {}

    n = min(len(method1), len(method2))
    m1, m2 = method1[:n], method2[:n]

    diffs = m1 - m2
    means = (m1 + m2) / 2

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs))

    corr, p_val = pearsonr(m1, m2) if n > 2 else (0.0, 1.0)

    return {
        "mean_difference": mean_diff,
        "std_difference": std_diff,
        "loa_upper": mean_diff + 1.96 * std_diff,
        "loa_lower": mean_diff - 1.96 * std_diff,
        "correlation": float(corr),
        "p_value": float(p_val),
        "differences": diffs,
        "means": means,
        "n_pairs": n,
    }
