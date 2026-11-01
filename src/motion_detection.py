"""Motion artifact detection and classification from accelerometer data.

Computes the tri-axial acceleration magnitude, removes the gravity component,
interpolates the resulting motion signal onto the PPG time grid, smooths it,
and classifies every sample as low / moderate / severe motion according to
device-specific thresholds.  A boolean clean mask marks samples free of
severe motion artifacts.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch

logger = logging.getLogger(__name__)


def classify_motion(
    motion: np.ndarray,
    thresholds: dict,
) -> tuple[np.ndarray, str, dict[str, float]]:
    """Classify per-sample motion and derive the recording-level label.

    Samples are classified following the manuscript definition:

    - ``low`` if ``motion <= theta_moderate``
    - ``moderate`` if ``theta_moderate < motion <= theta_severe``
    - ``severe`` if ``motion > theta_severe``

    The recording-level label is ``severe`` when more than 10 % of samples are
    severe, ``moderate`` when more than 20 % are moderate, and ``low``
    otherwise.

    Args:
        motion: Smoothed motion magnitude (g, gravity removed) per sample.
        thresholds: Dict with ``"moderate"`` and ``"severe"`` thresholds.

    Returns:
        Tuple of ``(classification, motion_level, fractions)`` where
        *classification* is an integer array (0 = low, 1 = moderate,
        2 = severe) and *fractions* holds the per-class sample proportions.
    """
    moderate_threshold = thresholds["moderate"]
    severe_threshold = thresholds["severe"]

    classification = np.zeros(len(motion), dtype=int)
    classification[(motion > moderate_threshold) & (motion <= severe_threshold)] = 1
    classification[motion > severe_threshold] = 2

    severe_fraction = float(np.mean(classification == 2))
    moderate_fraction = float(np.mean(classification == 1))

    if severe_fraction > 0.10:
        motion_level = "severe"
    elif moderate_fraction > 0.20:
        motion_level = "moderate"
    else:
        motion_level = "low"

    return classification, motion_level, {
        "still_time": float(np.mean(classification == 0)),
        "moderate_motion_time": moderate_fraction,
        "severe_motion_time": severe_fraction,
    }


def calculate_motion(
    acc_df: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    device: str,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute motion metrics and a clean-samples mask from accelerometer data.

    Pipeline:

    1. Compute 3-D acceleration magnitude: ``sqrt(x² + y² + z²)``.
    2. Remove gravity: ``|magnitude - 1.0|`` (1 g = Earth gravity).
    3. Interpolate to signal timestamps (accelerometer may run at a different rate).
    4. Gaussian-smooth the motion envelope (σ = 1.0 sample).
    5. Classify every sample and build the clean mask (non-severe samples).

    Args:
        acc_df: DataFrame with ``acc_x_g``, ``acc_y_g``, ``acc_z_g`` columns,
            indexed by datetime.  May be empty if no accelerometer data.
        timestamps: Signal timestamps (for interpolation target).
        device: Device identifier (used for threshold lookup).
        config: Parsed configuration dictionary.

    Returns:
        Tuple of:

        - ``clean_mask`` — boolean array (True = clean sample)
        - ``motion_levels`` — smoothed motion magnitude per sample
        - ``motion_metrics`` — dict with ``motion_level``, ``mean_motion``,
          ``max_motion``, ``clean_ratio``, ``still_time``,
          ``moderate_motion_time``, ``severe_motion_time``
    """
    thresholds = config["devices"][device]["motion_thresholds"]

    # Handle missing accelerometer data
    if acc_df.empty:
        logger.debug("%s: no ACC data — assuming low motion", device)
        return (
            np.ones(len(timestamps), dtype=bool),
            np.zeros(len(timestamps)),
            {
                "motion_level": "low",
                "mean_motion": 0.0,
                "max_motion": 0.0,
                "clean_ratio": 1.0,
                "still_time": 1.0,
                "moderate_motion_time": 0.0,
                "severe_motion_time": 0.0,
            },
        )

    # 3-D magnitude with gravity removed
    acc_mag = np.sqrt(acc_df["acc_x_g"] ** 2 + acc_df["acc_y_g"] ** 2 + acc_df["acc_z_g"] ** 2)
    acc_no_gravity = np.abs(acc_mag - 1.0)

    # Interpolate to signal timestamps (nanosecond integers for both axes)
    t_sig = timestamps.astype("datetime64[ns]").asi8
    t_acc = acc_df.index.astype("datetime64[ns]").asi8
    values = acc_no_gravity.fillna(0).values
    motion_levels = np.zeros(len(t_sig))
    inside = (t_sig >= t_acc.min()) & (t_sig <= t_acc.max())
    motion_levels[inside] = np.interp(t_sig[inside], t_acc, values)

    # Gaussian smoothing for stable classification
    if len(motion_levels) > 5:
        motion_smooth = gaussian_filter1d(motion_levels, sigma=1.0)
    else:
        motion_smooth = motion_levels

    classification, motion_class, fractions = classify_motion(motion_smooth, thresholds)
    clean_mask = classification != 2

    metrics = {
        "motion_level": motion_class,
        "mean_motion": float(np.mean(motion_smooth)),
        "max_motion": float(np.max(motion_levels)) if len(motion_levels) else 0.0,
        "clean_ratio": float(np.mean(clean_mask)),
        **fractions,
    }

    return clean_mask, motion_smooth, metrics


def acc_periodicity(
    acc_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    band: tuple[float, float] = (0.5, 4.0),
    peak_halfwidth_hz: float = 0.1,
) -> tuple[float, float]:
    """Dominant accelerometer frequency and its spectral concentration.

    Locomotion makes the wrist accelerometer strongly periodic (step and
    arm-swing frequencies).  A PPG beat detector can lock onto these
    harmonics and produce a regular, template-consistent "beat" train that
    passes interval- and morphology-based quality checks.

    Args:
        acc_df: Accelerometer frame (``acc_*_g`` columns).
        start: Window start.
        end: Window end.
        band: Frequency band searched (Hz).
        peak_halfwidth_hz: Half-width around the peak used for concentration.

    Returns:
        Tuple ``(peak_hz, concentration)`` where concentration is the fraction
        of in-band power within ``peak_halfwidth_hz`` of the peak (0-1);
        ``(nan, 0)`` when unavailable.
    """
    if acc_df is None or acc_df.empty:
        return np.nan, 0.0
    segment = acc_df.loc[(acc_df.index >= start) & (acc_df.index <= end)]
    if len(segment) < 64:
        return np.nan, 0.0
    steps = pd.Series(segment.index).diff().dt.total_seconds().dropna()
    fs = 1.0 / float(np.median(steps)) if len(steps) and np.median(steps) > 0 else np.nan
    if not np.isfinite(fs) or fs < 2 * band[1]:
        return np.nan, 0.0
    mag = np.sqrt(segment["acc_x_g"] ** 2 + segment["acc_y_g"] ** 2 + segment["acc_z_g"] ** 2).to_numpy(dtype=float)
    mag = np.nan_to_num(mag - np.nanmean(mag))
    nperseg = int(min(len(mag), round(20 * fs)))
    frequencies, power = welch(mag, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    in_band = (frequencies >= band[0]) & (frequencies <= band[1])
    if not np.any(in_band) or power[in_band].sum() <= 0:
        return np.nan, 0.0
    peak = float(frequencies[in_band][np.argmax(power[in_band])])
    near = in_band & (np.abs(frequencies - peak) <= peak_halfwidth_hz)
    return peak, float(power[near].sum() / power[in_band].sum())
