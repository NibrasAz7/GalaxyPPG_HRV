"""Shared utility functions for the GalaxyPPG HRV pipeline.

Provides:
- Configuration loading from YAML
- Activity-specific heart-rate bounds for IBI filtering
- Hampel filter for spike removal
- Successive-difference threshold lookup
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict:
    """Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to the ``config.yaml`` file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_activity_hr_bounds(activity: str, config: dict) -> Tuple[float, float]:
    """Return (hr_min, hr_max) in bpm for a given activity type.

    Values are used to constrain IBI physiological range filtering.
    Falls back to ``config["activity_hr_bounds"]["default"]`` when the
    activity is not explicitly listed.

    Args:
        activity: Activity name (e.g. ``"baseline"``, ``"walking"``).
        config: Parsed configuration dictionary.

    Returns:
        Tuple of ``(hr_min_bpm, hr_max_bpm)``.
    """
    bounds = config.get("activity_hr_bounds", {})
    return bounds.get(activity, bounds.get("default", (40, 200)))


def get_successive_diff_threshold(activity: str, device: str) -> float:
    """Return the successive-difference threshold for IBI filtering.

    PPG signals are noisier than ECG, so a wider threshold is used.
    Activity-specific multipliers further adjust the base value.

    Args:
        activity: Activity name.
        device: Device identifier (``"E4"``, ``"GalaxyWatch"``, ``"PolarH10"``).

    Returns:
        Threshold as a fraction of the preceding IBI (e.g. 0.25 = 25%).
    """
    base = 0.25 if device == "PolarH10" else 0.35

    multipliers = {
        "baseline": 0.8,
        "rest-1": 0.8,
        "keyboard-typing": 1.0,
        "walking": 1.2,
        "jogging": 1.5,
        "running": 1.8,
    }
    return base * multipliers.get(activity, 1.0)


def hampel_filter(data: np.ndarray, window_size: int = 5, n_sigma: float = 3.0) -> np.ndarray:
    """Apply a Hampel filter for univariate spike removal.

    Each sample is compared to the median of a sliding window.  If it
    deviates by more than *n_sigma* × MAD (median absolute deviation),
    it is replaced by the window median.

    Args:
        data: 1-D input signal.
        window_size: Half-width of the sliding window (total width = 2×window_size+1).
        n_sigma: Number of MADs beyond which a sample is treated as an outlier.

    Returns:
        Filtered copy of *data* with outliers replaced.
    """
    filtered = data.copy()
    n = len(data)

    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        window = data[start:end]

        median = np.median(window)
        mad = np.median(np.abs(window - median))

        if mad > 0 and abs(data[i] - median) > n_sigma * mad:
            filtered[i] = median

    return filtered
