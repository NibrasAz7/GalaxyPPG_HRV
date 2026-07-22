"""Motion artifact detection and classification from accelerometer data.

Computes a 3-D acceleration magnitude, removes the gravity component,
interpolates to signal timestamps, and classifies each window as
low / moderate / severe motion.  Produces a boolean clean-mask that
marks samples contaminated by severe motion artifacts.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


def classify_motion(mean_motion: float, thresholds: dict) -> str:
    """Classify a mean motion level into low / moderate / severe.

    Args:
        mean_motion: Mean acceleration magnitude (g-force, gravity removed).
        thresholds: Dict with keys ``"low"``, ``"moderate"``, ``"severe"``.

    Returns:
        One of ``"low"``, ``"moderate"``, ``"severe"``.
    """
    if mean_motion < thresholds["low"]:
        return "low"
    elif mean_motion < thresholds["moderate"]:
        return "moderate"
    else:
        return "severe"


def calculate_motion(
    acc_df: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    device: str,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Compute motion metrics and a clean-samples mask from accelerometer data.

    Pipeline:

    1. Compute 3-D acceleration magnitude: ``sqrt(x² + y² + z²)``.
    2. Remove gravity: ``|magnitude - 1.0|`` (1 g = Earth gravity).
    3. Interpolate to signal timestamps (accelerometer may run at a different rate).
    4. Gaussian-smooth the motion envelope (σ = 1.0 sample).
    5. Slide a 3-second window and classify each window as clean or contaminated.
    6. Compute overall motion level (low / moderate / severe).

    Args:
        acc_df: DataFrame with ``acc_x_g``, ``acc_y_g``, ``acc_z_g`` columns,
            indexed by datetime.  May be empty if no accelerometer data.
        timestamps: Signal timestamps (for interpolation target).
        device: Device identifier (used for threshold lookup).
        config: Parsed configuration dictionary.

    Returns:
        Tuple of:

        - ``clean_mask`` — boolean array (True = clean sample)
        - ``motion_levels`` — float array of motion magnitude per sample
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

    # Interpolate to signal timestamps
    t_sig = timestamps.view("int64")
    t_acc = acc_df.index.view("int64")
    vals = acc_no_gravity.fillna(0).values
    motion_levels = np.zeros(len(t_sig))
    inside = (t_sig >= t_acc.min()) & (t_sig <= t_acc.max())
    motion_levels[inside] = np.interp(t_sig[inside], t_acc, vals)

    # Gaussian smoothing for stable classification
    if len(motion_levels) > 5:
        motion_smooth = gaussian_filter1d(motion_levels, sigma=1.0)
    else:
        motion_smooth = motion_levels

    # Windowed classification
    duration_sec = (timestamps[-1] - timestamps[0]).total_seconds()
    window_samples = max(1, int(3 * len(timestamps) / duration_sec))
    clean_mask = np.ones(len(timestamps), dtype=bool)
    classification = np.zeros(len(timestamps))

    for i in range(0, len(motion_smooth) - window_samples, max(1, window_samples // 4)):
        window = motion_smooth[i : i + window_samples]
        if np.max(window) > thresholds["severe"]:
            clean_mask[i : i + window_samples] = False
            classification[i : i + window_samples] = 2
        elif np.std(window) > thresholds["moderate"]:
            classification[i : i + window_samples] = 1

    mean_motion = float(np.mean(motion_smooth))
    motion_class = classify_motion(mean_motion, thresholds)

    metrics = {
        "motion_level": motion_class,
        "mean_motion": mean_motion,
        "max_motion": float(np.max(motion_levels)),
        "clean_ratio": float(np.mean(clean_mask)),
        "still_time": float(np.mean(classification == 0)),
        "moderate_motion_time": float(np.mean(classification == 1)),
        "severe_motion_time": float(np.mean(classification == 2)),
    }

    return clean_mask, motion_levels, metrics
