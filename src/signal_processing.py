"""Signal processing: bandpass filtering, polarity correction, and quality assessment.

Provides the filtering stage of the pipeline:

1. **Auto-polarity** — inverts PPG signals where valleys dominate so that
   systolic peaks become positive.
2. **Adaptive bandpass filter** — applies a zero-phase Butterworth bandpass
   whose cutoff frequencies and order adapt to the device, motion level,
   and activity type.  ECG receives minimal filtering to preserve clinical
   detail; PPG receives tighter bands under motion.
3. **Signal quality estimation** — computes SNR (cardiac band vs. noise
   band), kurtosis, skewness, and HR-band power ratio from the FFT.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.stats import kurtosis

logger = logging.getLogger(__name__)


def auto_polarity(signal: np.ndarray) -> np.ndarray:
    """Invert a PPG signal if valleys dominate over peaks.

    Compares the 5th and 95th percentiles.  If the negative excursion is
    larger, the signal is inverted so that systolic peaks are positive.

    Args:
        signal: 1-D PPG signal.

    Returns:
        Signal with correct polarity (same shape).
    """
    p95 = np.percentile(signal, 95)
    p05 = np.percentile(signal, 5)
    return -signal if abs(p05) > abs(p95) else signal


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    Uses ``scipy.signal.filtfilt`` for zero-phase (forward-backward) filtering,
    which preserves peak timing accuracy.

    Args:
        signal: 1-D input signal.
        fs: Sampling frequency in Hz.
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        order: Butterworth filter order.

    Returns:
        Filtered signal (same length as input).
    """
    nyq = fs / 2
    low = max(0.01, lowcut / nyq)
    high = min(highcut / nyq, 0.95)
    if high <= low + 1e-3:
        high = min(0.95, low + 0.2)

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def adaptive_bandpass_filter(
    signal: np.ndarray,
    device: str,
    motion_class: str,
    activity_type: str,
    config: dict,
) -> np.ndarray:
    """Apply motion- and device-aware adaptive bandpass filtering.

    Filtering strategy:

    - **PolarH10 (ECG)**: Minimal filtering — preserves clinical QRS detail.
      Order is slightly reduced for baseline, standard otherwise.
    - **PPG devices (E4, GalaxyWatch)**: Bandpass widens for baseline
      (more signal), tightens for moderate/severe motion (reject artifacts).
      Filter order increases with motion severity.
    - **GalaxyWatch baseline**: Light Savitzky-Golay smoothing blend (20%)
      to reduce high-frequency noise without distorting the waveform.

    Args:
        signal: 1-D raw signal.
        device: Device identifier.
        motion_class: ``"low"``, ``"moderate"``, or ``"severe"``.
        activity_type: Activity name (e.g. ``"baseline"``).
        config: Parsed configuration dictionary with device parameters.

    Returns:
        Filtered signal (same length as input).
    """
    dev_cfg = config["devices"][device]
    fs = dev_cfg["fs"]
    low, high = dev_cfg["bandpass"]
    order = dev_cfg["filter_order"]

    try:
        if device == "PolarH10":
            # ECG: preserve clinical details, minimal modification
            if activity_type == "baseline":
                order = max(order - 1, 3)
        else:
            # PPG: adapt bandpass to motion level
            if activity_type == "baseline":
                low = max(low * 0.8, 0.3)
                high = min(high * 1.2, 5.0)
                order = max(order - 1, 2)
            elif motion_class == "severe":
                low = max(low, 0.9)
                high = min(high, 3.0)
                order = order + 2
            elif motion_class == "moderate":
                low = max(low, 0.8)
                high = min(high, 3.2)
                order = order + 1

        filtered = bandpass_filter(signal, fs, low, high, order)

        # Light Savitzky-Golay smoothing for GalaxyWatch baseline only
        if activity_type == "baseline" and device == "GalaxyWatch":
            win = min(int(fs * 0.2), len(filtered) // 10)
            if win >= 3:
                if win % 2 == 0:
                    win += 1
                try:
                    smoothed = savgol_filter(filtered, win, 2)
                    filtered = filtered * 0.8 + smoothed * 0.2
                except Exception:
                    pass

        # Apply polarity correction for PPG devices only
        if device != "PolarH10":
            filtered = auto_polarity(filtered)

        return filtered

    except Exception as e:
        logger.warning("Filter error for %s: %s", device, e)
        return signal


def estimate_signal_quality(signal: np.ndarray, fs: float) -> Dict:
    """Compute frequency-domain signal quality metrics.

    Calculates:

    - **SNR (dB)**: ratio of cardiac-band power (0.5–4 Hz) to noise-band
      power (>8 Hz), clamped to [0, 50] dB.
    - **Kurtosis / Skewness**: distribution shape of the time-domain signal.
    - **HR power ratio**: fraction of total spectral power in the cardiac band.

    Args:
        signal: 1-D signal (raw or filtered).
        fs: Sampling frequency in Hz.

    Returns:
        Dictionary with keys ``snr_db``, ``kurtosis``, ``skewness``,
        ``hr_power_ratio``, ``signal_std``, ``signal_range``.
    """
    try:
        sig_kurt = kurtosis(signal)
        sig_skew = pd.Series(signal).skew()
        sig_std = np.std(signal)
        sig_range = np.ptp(signal)

        freqs = fftfreq(len(signal), 1 / fs)
        fft_mag = np.abs(fft(signal))

        # Keep non-negative frequencies, exclude DC
        pos = freqs >= 0
        freqs = freqs[pos]
        fft_mag = fft_mag[pos]

        hr_band = (freqs >= 0.5) & (freqs <= 4.0)
        total_power = np.sum(fft_mag ** 2)
        hr_power = np.sum(fft_mag[hr_band] ** 2)
        hr_power_ratio = hr_power / total_power if total_power > 0 else 0.0

        # SNR: cardiac band vs. noise band
        if np.any(hr_band) and len(freqs) > 10:
            noise_band = (freqs >= 8) & (freqs <= fs / 2)
            if np.any(noise_band):
                sig_power = np.mean(fft_mag[hr_band] ** 2)
                noise_power = np.mean(fft_mag[noise_band] ** 2)
                snr_db = 10 * np.log10(sig_power / (noise_power + 1e-10))
                snr_db = max(0, min(50, snr_db))
            else:
                snr_db = 10.0
        else:
            snr_db = 5.0

        return {
            "snr_db": snr_db,
            "kurtosis": sig_kurt,
            "skewness": sig_skew,
            "hr_power_ratio": hr_power_ratio,
            "signal_std": sig_std,
            "signal_range": sig_range,
        }

    except Exception as e:
        logger.warning("Signal quality error: %s", e)
        return {
            "snr_db": 5.0,
            "kurtosis": 0.0,
            "skewness": 0.0,
            "hr_power_ratio": 0.0,
            "signal_std": 1.0,
            "signal_range": 1.0,
        }
