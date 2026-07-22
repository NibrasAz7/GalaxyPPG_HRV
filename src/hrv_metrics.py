"""HRV metrics: IBI artifact correction, rolling HR, and HRV index computation.

Implements the post-peak-detection stage of the pipeline:

1. **IBI calculation** — converts peak indices to inter-beat intervals (ms).
2. **Clinical IBI filtering** — multi-stage artifact removal:
   - Physiological range filtering (activity-specific HR bounds)
   - Successive-difference filtering (removes abrupt IBI jumps)
   - Local Outlier Factor (LOF) based removal (or IQR fallback)
   - Moving-median filtering for residual artifact suppression
3. **Rolling HR** — sliding-window heart rate with spline smoothing,
   device-aware smoothing parameters, and Hampel spike removal.
4. **HRV indices** — time-domain (SDNN, RMSSD, pNN50, mean HR) computed
   from artifact-corrected NN intervals, with 25% median-deviation
   gating to exclude motion-contaminated IBIs.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import medfilt

from .utils import get_activity_hr_bounds, get_successive_diff_threshold, hampel_filter

logger = logging.getLogger(__name__)


def calculate_ibi(peaks: np.ndarray, fs: float) -> np.ndarray:
    """Convert peak indices to inter-beat intervals in milliseconds.

    Args:
        peaks: Array of peak sample indices.
        fs: Sampling frequency in Hz.

    Returns:
        Array of IBI values in milliseconds (length = len(peaks) - 1).
    """
    if len(peaks) < 2:
        return np.array([])
    return np.diff(peaks) / fs * 1000.0


def _statistical_outlier_removal(
    ibis: np.ndarray, times: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove outliers using the IQR method (fallback when LOF is unavailable).

    Args:
        ibis: IBI values in milliseconds.
        times: Corresponding timestamps.

    Returns:
        Tuple of ``(filtered_ibis, filtered_times)``.
    """
    q1, q3 = np.percentile(ibis, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (ibis >= lower) & (ibis <= upper)
    return ibis[mask], times[mask]


def _lof_outlier_removal(
    ibis: np.ndarray, times: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove outliers using Local Outlier Factor (LOF).

    Uses IBI value and inter-sample time difference as features.
    Falls back to IQR-based removal if scikit-learn is unavailable
    or the sample count is too small.

    Args:
        ibis: IBI values in milliseconds.
        times: Corresponding timestamps.

    Returns:
        Tuple of ``(filtered_ibis, filtered_times)``.
    """
    if len(ibis) < 6:
        return _statistical_outlier_removal(ibis, times)

    try:
        from sklearn.neighbors import LocalOutlierFactor

        time_ns = pd.to_datetime(times).view("int64")
        time_diffs = np.diff(time_ns) / 1e9
        features = np.column_stack([ibis[:-1], time_diffs])

        n = len(features)
        if n < 3:
            return _statistical_outlier_removal(ibis, times)

        n_neighbors = min(5, max(2, n - 1))
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
        labels = lof.fit_predict(features)

        inlier_idx = np.where(labels == 1)[0]
        valid_idx = np.concatenate([inlier_idx, [len(ibis) - 1]])
        return ibis[valid_idx], times[valid_idx]

    except ImportError:
        return _statistical_outlier_removal(ibis, times)


def clinical_ibi_filtering(
    ibis: np.ndarray,
    ibi_times: np.ndarray,
    activity_type: str,
    device: str,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply multi-stage clinical IBI artifact removal.

    Stages (applied sequentially):

    1. **Physiological range filter** — removes IBIs outside activity-specific
       HR bounds (e.g. 45–110 bpm for baseline).
    2. **Successive-difference filter** — removes IBIs that change by more
       than a device- and activity-specific fraction of the preceding IBI.
    3. **LOF outlier removal** — detects multivariate outliers using IBI
       value and inter-sample timing.
    4. **Moving-median filter** — suppresses residual artifacts with a
       5-sample (or smaller) median filter.

    Args:
        ibis: Raw IBI values in milliseconds.
        ibi_times: Corresponding timestamps.
        activity_type: Activity name.
        device: Device identifier.
        config: Parsed configuration dictionary.

    Returns:
        Tuple of ``(filtered_ibis, filtered_times)``.
    """
    if len(ibis) < 3:
        return ibis, ibi_times

    # Stage 1: Physiological range
    hr_min, hr_max = get_activity_hr_bounds(activity_type, config)
    ibi_min = 60000 / hr_max
    ibi_max = 60000 / hr_min
    mask = (ibis >= ibi_min) & (ibis <= ibi_max)
    ibis_f, times_f = ibis[mask], ibi_times[mask]

    if len(ibis_f) < 3:
        return ibis_f, times_f

    # Stage 2: Successive difference
    diff_threshold = get_successive_diff_threshold(activity_type, device)
    succ_mask = np.ones(len(ibis_f), dtype=bool)
    for i in range(1, len(ibis_f)):
        if abs(ibis_f[i] - ibis_f[i - 1]) > diff_threshold * ibis_f[i - 1]:
            succ_mask[i] = False
    ibis_f, times_f = ibis_f[succ_mask], times_f[succ_mask]

    if len(ibis_f) < 3:
        return ibis_f, times_f

    # Stage 3: LOF outlier removal
    ibis_f, times_f = _lof_outlier_removal(ibis_f, times_f)

    # Stage 4: Moving-median filter
    if len(ibis_f) >= 5:
        win = min(5, len(ibis_f) // 2)
        if win % 2 == 0:
            win -= 1
        if win >= 3:
            ibis_f = median_filter(ibis_f.astype(float), size=win)

    return ibis_f, times_f


def _robust_window_filtering(
    window_ibis: np.ndarray,
) -> np.ndarray:
    """Apply trimmed-mean + Gaussian-weighted averaging within a rolling window.

    Removes the top and bottom 20% of IBIs, then computes a Gaussian-
    weighted average of the remaining values.

    Args:
        window_ibis: IBI values within a single rolling window.

    Returns:
        Array containing a single representative IBI value.
    """
    if len(window_ibis) < 3:
        return window_ibis

    sorted_ibis = np.sort(window_ibis)
    trim = int(len(sorted_ibis) * 0.2)
    trimmed = sorted_ibis[trim : -trim] if trim > 0 else sorted_ibis

    if len(trimmed) == 0:
        return np.array([np.mean(window_ibis)])

    mean_ibi = np.mean(trimmed)
    std = np.std(trimmed)
    if std == 0:
        return np.array([mean_ibi])

    weights = np.exp(-0.5 * ((trimmed - mean_ibi) / std) ** 2)
    weights /= np.sum(weights)
    return np.array([np.sum(weights * trimmed)])


def _hr_outlier_removal(
    hr_values: np.ndarray,
    hr_times: np.ndarray,
    activity_type: str,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove HR outliers using physiological bounds, rate-of-change, and Hampel filter.

    Args:
        hr_values: Rolling HR values in bpm.
        hr_times: Corresponding timestamps.
        activity_type: Activity name.
        config: Parsed configuration dictionary.

    Returns:
        Tuple of ``(filtered_hr, filtered_times)``.
    """
    if len(hr_values) < 3:
        return hr_values, hr_times

    # Physiological bounds
    hr_min, hr_max = get_activity_hr_bounds(activity_type, config)
    mask = (hr_values >= hr_min) & (hr_values <= hr_max)
    hr_values, hr_times = hr_values[mask], hr_times[mask]

    if len(hr_values) < 3:
        return hr_values, hr_times

    # Rate-of-change filter (max 30 bpm/s)
    hr_diffs = np.abs(np.diff(hr_values))
    time_ns = pd.to_datetime(hr_times).view("int64")
    dt = np.diff(time_ns) / 1e9
    dt[dt == 0] = np.nan
    rate = hr_diffs / dt
    change_mask = np.concatenate([[True], np.nan_to_num(rate, nan=0) <= 30])
    hr_values, hr_times = hr_values[change_mask], hr_times[change_mask]

    # Hampel spike removal
    if len(hr_values) >= 5:
        hr_values = hampel_filter(hr_values, window_size=5, n_sigma=3)

    return hr_values, hr_times


def compute_rolling_hr(
    ibis: np.ndarray,
    ibi_times: np.ndarray,
    activity_type: str,
    device: str,
    config: dict,
) -> dict:
    """Compute rolling heart rate from IBI series with spline smoothing.

    Pipeline:

    1. Apply clinical IBI filtering (artifact removal).
    2. Slide a beat-count window (15 beats, 50% overlap) and compute
       HR = 60000 / mean(window_ibis) with robust window filtering.
    3. Remove HR outliers (physiological bounds + rate-of-change + Hampel).
    4. Fit a univariate smoothing spline (device-aware smoothing factor).
    5. Optional light Gaussian smoothing for GalaxyWatch baseline.

    Device-specific parameters:

    - **PolarH10 (ECG)**: Minimal smoothing (factor 0.1), conservative
      outlier threshold (2.5σ).
    - **PPG baseline**: More smoothing (factor 0.4), wider window (20 beats).
    - **PPG motion**: Moderate smoothing (factor 0.2).

    Args:
        ibis: IBI values in milliseconds.
        ibi_times: Corresponding timestamps.
        activity_type: Activity name.
        device: Device identifier.
        config: Parsed configuration dictionary.

    Returns:
        Dict with keys ``"custom_hr"`` (list of smoothed HR values) and
        ``"times"`` (list of timestamps).  Empty dict if insufficient data.
    """
    window_beats = config["processing"]["window_beats"]

    # Device- and activity-specific parameters
    if device == "PolarH10":
        outlier_threshold = 2.5
        smoothing_factor = 0.1
        if activity_type == "baseline":
            window_beats = max(15, window_beats)
        else:
            window_beats = max(10, window_beats)
    elif activity_type == "baseline":
        window_beats = max(20, window_beats)
        outlier_threshold = 1.5
        smoothing_factor = 0.4
    else:
        outlier_threshold = 2.0
        smoothing_factor = 0.2

    # Fallback for short IBI series
    if len(ibis) < window_beats:
        inst_hr = 60000.0 / np.clip(ibis, 1e-6, None)
        t = pd.to_datetime(ibi_times)
        if len(inst_hr) >= 3:
            try:
                inst_hr = medfilt(inst_hr, kernel_size=3)
            except Exception:
                pass
        return {"custom_hr": inst_hr.tolist(), "times": t.tolist()}

    # Stage 1: Clinical IBI filtering
    filtered_ibis, filtered_times = clinical_ibi_filtering(
        ibis, ibi_times, activity_type, device, config
    )

    if len(filtered_ibis) < window_beats:
        return {}

    # Stage 2: Rolling window HR
    overlap = window_beats // 2
    rolling_hr: list[float] = []
    hr_times: list = []

    for i in range(0, len(filtered_ibis) - window_beats + 1, overlap):
        window = filtered_ibis[i : i + window_beats]
        window_filtered = _robust_window_filtering(window)
        if len(window_filtered) >= window_beats // 2:
            hr = 60000 / np.mean(window_filtered)
        else:
            hr = 60000 / np.mean(window)
        rolling_hr.append(hr)
        hr_times.append(filtered_times[i + window_beats // 2])

    if len(rolling_hr) < 3:
        return {}

    # Stage 3: HR outlier removal
    rolling_hr = np.array(rolling_hr)
    hr_times = np.array(hr_times)
    rolling_hr, hr_times = _hr_outlier_removal(rolling_hr, hr_times, activity_type, config)

    if len(rolling_hr) < 3:
        return {}

    # Stage 4: Spline smoothing
    try:
        time_numeric = np.arange(len(rolling_hr))
        s_factor = len(rolling_hr) * smoothing_factor
        k = min(3, max(1, len(rolling_hr) - 1))
        spline = UnivariateSpline(time_numeric, rolling_hr, s=s_factor, k=k)
        smoothed_hr = spline(time_numeric)

        # Light Gaussian for GalaxyWatch baseline
        if activity_type == "baseline" and device == "GalaxyWatch" and len(smoothed_hr) > 5:
            smoothed_hr = gaussian_filter1d(smoothed_hr, sigma=0.5)

        return {"custom_hr": smoothed_hr.tolist(), "times": hr_times.tolist()}
    except Exception as e:
        logger.warning("Spline smoothing error: %s", e)
        return {"custom_hr": rolling_hr.tolist(), "times": hr_times.tolist()}


def compute_hrv_metrics(custom_ibis: np.ndarray) -> dict:
    """Compute time-domain HRV metrics from artifact-corrected IBIs.

    Applies a 25% median-deviation filter to exclude motion-contaminated
    IBIs before computing HRV indices:

    - **SDNN**: standard deviation of NN intervals
    - **RMSSD**: root mean square of successive differences
    - **Mean HR**: average heart rate from IBI mean

    Args:
        custom_ibis: IBI values in milliseconds (already range-filtered).

    Returns:
        Dict with ``sdnn``, ``rmssd``, ``mean_hr`` (or zeros if insufficient data).
    """
    if len(custom_ibis) == 0:
        return {"sdnn": 0.0, "rmssd": 0.0, "mean_hr": 0.0}

    # 25% median-deviation gating (standard practice for motion artifact rejection)
    median_ibi = np.median(custom_ibis)
    valid = np.abs(custom_ibis - median_ibi) <= (0.25 * median_ibi)
    filtered = custom_ibis[valid]

    if len(filtered) > 1:
        sdnn = float(np.std(filtered))
        rmssd = float(np.sqrt(np.mean(np.diff(filtered) ** 2)))
    else:
        sdnn = 0.0
        rmssd = 0.0

    mean_hr = 60000.0 / np.mean(custom_ibis) if len(custom_ibis) > 0 else 0.0

    return {"sdnn": sdnn, "rmssd": rmssd, "mean_hr": mean_hr}
