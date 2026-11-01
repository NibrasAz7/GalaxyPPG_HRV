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

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import kurtosis

logger = logging.getLogger(__name__)


def auto_polarity(signal: np.ndarray) -> np.ndarray:
    """Invert a PPG signal if valleys dominate over peaks.

    Compares the 5th and 95th percentiles.  If the negative excursion is
    larger, the signal is inverted so that systolic peaks are positive.

    A polarity-sensitive pulse-template test was evaluated as a refinement but
    rejected: under locomotion the band-passed segment is dominated by motion
    noise, so the template correlation became unreliable and degraded jogging
    peak recall.  The always-available percentile heuristic is therefore
    retained.

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

    The filter is designed in second-order sections and applied forward and
    backward with ``sosfiltfilt``.  The transfer-function (b, a) form used
    previously is numerically ill-conditioned for the 8th-16th order
    band-pass designs at low normalised frequencies used here, and
    ``filtfilt`` silently returns garbage rather than raising when the
    filter is unstable.

    Args:
        signal: 1-D input signal.
        fs: Sampling frequency in Hz.
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        order: Butterworth filter order (per band edge).

    Returns:
        Filtered signal (same length as input).

    Raises:
        ValueError: If the band is not strictly inside (0, Nyquist).
    """
    nyq = fs / 2.0
    if not (0.0 < lowcut < highcut < nyq):
        raise ValueError(
            f"Invalid band [{lowcut}, {highcut}] Hz for fs={fs} Hz (Nyquist {nyq} Hz)"
        )
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    filtered = sosfiltfilt(sos, np.asarray(signal, dtype=float))
    if not np.all(np.isfinite(filtered)):
        raise FloatingPointError("Band-pass filter produced non-finite output")
    return filtered


def repair_signal(
    signal: np.ndarray,
    fs: float,
    shift_mad_factor: float = 20.0,
    guard_s: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair missing samples and level shifts before filtering.

    * ``NaN`` samples (device-flagged or sentinel values) are bridged by
      linear interpolation instead of being replaced with zero, which would
      inject steps that ring through the IIR filter.
    * Abrupt level shifts (the Galaxy Watch re-centres its optical front end
      after every gain reset, producing DC steps of several pulse
      amplitudes) are removed by subtracting each step whose first
      difference exceeds ``shift_mad_factor`` robust SDs of the first
      difference.

    Both repairs are marked as unreliable, together with ``guard_s`` seconds
    on either side, so that beats in the repaired regions are excluded from
    the clean mask.

    Args:
        signal: Raw 1-D signal (may contain NaN).
        fs: Sampling frequency in Hz.
        shift_mad_factor: Threshold for level-shift detection.
        guard_s: Unreliable margin around each repair in seconds.

    Returns:
        Tuple of ``(repaired, unreliable_mask)``.
    """
    x = np.asarray(signal, dtype=float).copy()
    n = len(x)
    unreliable = np.zeros(n, dtype=bool)
    if n == 0:
        return x, unreliable

    missing = ~np.isfinite(x)
    if missing.all():
        return np.zeros(n), np.ones(n, dtype=bool)
    if missing.any():
        idx = np.arange(n)
        x[missing] = np.interp(idx[missing], idx[~missing], x[~missing])
        unreliable |= missing

    d = np.diff(x)
    if len(d) > 10:
        mad = 1.4826 * float(np.median(np.abs(d - np.median(d))))
        if mad > 0:
            shifts = np.where(np.abs(d) > shift_mad_factor * mad)[0]
            if len(shifts):
                correction = np.zeros(n)
                for i in shifts:
                    correction[i + 1:] += d[i]
                x = x - correction
                unreliable[shifts] = True
                unreliable[np.minimum(shifts + 1, n - 1)] = True

    guard = int(round(guard_s * fs))
    if guard > 0 and unreliable.any():
        unreliable = np.convolve(unreliable.astype(float), np.ones(2 * guard + 1), "same") > 0
    return x, unreliable


def motion_band(device_cfg: dict, motion_class: str) -> tuple[float, float, int]:
    """Return the ``(low, high, order)`` band-pass design for a motion state.

    Low motion uses the device default band.  Under moderate/severe motion
    the band is narrowed to the exercise cardiac range (0.7-3.5 Hz, i.e.
    42-210 bpm), rejecting slow sway and the harmonics of arm swing without
    clipping the fundamental at jogging heart rates (the previous 3.0 Hz
    ceiling sat below 180 bpm).  The design depends only on the measured
    accelerometer state, never on the protocol activity label.
    """
    low, high = device_cfg["bandpass"]
    order = int(device_cfg["filter_order"])
    if motion_class in ("moderate", "severe"):
        motion_cfg = device_cfg.get("motion_bandpass", [0.7, 3.5])
        low, high = max(low, motion_cfg[0]), min(high, motion_cfg[1])
    return float(low), float(high), order


def adaptive_bandpass_filter(
    signal: np.ndarray,
    device: str,
    motion_class: str,
    activity_type: str | None = None,
    config: dict | None = None,
) -> np.ndarray:
    """Apply motion- and device-aware zero-phase band-pass filtering.

    * **ECG (Polar H10)**: fixed QRS-preserving band from the configuration.
    * **PPG (E4, Galaxy Watch)**: device default band under low motion,
      exercise cardiac band under moderate/severe motion (see
      :func:`motion_band`), followed by automatic polarity correction.

    Args:
        signal: 1-D raw (repaired) signal.
        device: Device identifier.
        motion_class: ``"low"``, ``"moderate"`` or ``"severe"`` (from the
            accelerometer).
        activity_type: Ignored.  Retained only for call-site compatibility:
            processing never depends on the protocol activity label, which
            is unavailable in free-living use.
        config: Parsed configuration dictionary with device parameters.

    Returns:
        Filtered signal (same length as input).
    """
    if config is None:  # legacy positional call: (signal, device, motion, activity, config)
        raise TypeError("adaptive_bandpass_filter requires a config")
    dev_cfg = config["devices"][device]
    fs = dev_cfg["fs"]
    if device == "PolarH10":
        low, high = dev_cfg["bandpass"]
        order = int(dev_cfg["filter_order"])
    else:
        low, high, order = motion_band(dev_cfg, motion_class)

    filtered = bandpass_filter(signal, fs, low, high, order)
    if device != "PolarH10":
        polarity = dev_cfg.get("polarity")
        filtered = filtered * float(polarity) if polarity in (1, -1) else auto_polarity(filtered)
    return filtered


def pulse_asymmetry(signal: np.ndarray, peaks: np.ndarray) -> float:
    """Median fraction of the cardiac cycle spent rising to each detected maximum.

    A correctly oriented PPG rises quickly (systolic upstroke, ~20-35 % of
    the cycle) and decays slowly; values above 0.5 indicate an inverted
    signal whose maxima are pulse onsets.
    """
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) < 5:
        return np.nan
    fractions = []
    for a, b, c in zip(peaks[:-2], peaks[1:-1], peaks[2:], strict=False):
        if b - a < 3 or c - b < 3:
            continue
        rise = b - (a + int(np.argmin(signal[a:b])))
        fall = (b + int(np.argmin(signal[b:c]))) - b
        if rise + fall > 0:
            fractions.append(rise / (rise + fall))
    return float(np.median(fractions)) if fractions else np.nan


def normalized_lms(
    signal: np.ndarray,
    reference: np.ndarray,
    order: int = 32,
    mu: float = 0.25,
    eps: float = 1e-6,
) -> np.ndarray:
    """Adaptive noise cancellation with the normalised LMS algorithm.

    A tapped-delay-line filter is adapted to predict the motion component of
    *signal* from the accelerometer *reference* channels; the prediction is
    subtracted, leaving the cardiac component.

    Args:
        signal: Noisy signal (e.g. raw PPG).
        reference: Reference channels of shape ``(n_samples,)`` or
            ``(n_samples, n_channels)``, aligned with *signal*.
        order: Number of filter taps per reference channel.
        mu: Normalised step size (0 < mu < 2).
        eps: Regularisation term for the normalisation.

    Returns:
        Cleaned signal with the same length as *signal*.
    """
    reference = np.asarray(reference, dtype=float)
    if reference.ndim == 1:
        reference = reference[:, None]
    if len(signal) <= order:
        return np.array(signal, dtype=float)

    cleaned = np.array(signal, dtype=float)
    weights = np.zeros((order, reference.shape[1]))
    for index in range(order, len(signal)):
        window = reference[index - order:index][::-1]
        prediction = float(np.sum(weights * window))
        error = signal[index] - prediction
        cleaned[index] = error
        weights += mu * error * window / (np.sum(window * window) + eps)

    return cleaned


def acc_referenced_anc(
    signal: np.ndarray,
    acc_df: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    fs: float,
    config: dict,
) -> np.ndarray:
    """Remove motion artifacts using the accelerometer as a noise reference.

    The three accelerometer axes (gravity removed, band-pass filtered to the
    reference band) are interpolated onto the signal time grid and fed to an
    adaptive noise canceller.

    Args:
        signal: Raw signal.
        acc_df: Accelerometer DataFrame (may be empty).
        timestamps: Signal timestamps.
        fs: Sampling frequency in Hz.
        config: Parsed configuration dictionary (``processing.anc`` section).

    Returns:
        Cleaned signal, or the input signal when the stage is disabled or no
        accelerometer data are available.
    """
    parameters = config.get("processing", {}).get("anc", {})
    if not parameters.get("enabled", False) or acc_df is None or acc_df.empty:
        return signal

    order = int(parameters.get("order", 32))
    mu = float(parameters.get("mu", 0.01))
    band = parameters.get("reference_band", [0.4, 5.0])
    delay = int(round(float(parameters.get("delay_ms", 0.0)) / 1000.0 * fs))
    if order >= len(signal):
        return signal

    t_signal = timestamps.astype("datetime64[ns]").asi8
    t_acc = acc_df.index.astype("datetime64[ns]").asi8
    if len(t_acc) < 2:
        return signal

    channels = []
    for column in ("acc_x_g", "acc_y_g", "acc_z_g"):
        if column not in acc_df.columns:
            return signal
        values = np.interp(t_signal, t_acc, acc_df[column].fillna(0).to_numpy(dtype=float))
        channels.append(bandpass_filter(values, fs, band[0], band[1], 4))
    reference = np.stack(channels, axis=1)

    if delay > 0:
        reference = np.vstack([np.zeros((delay, reference.shape[1])), reference[:-delay]])

    return normalized_lms(signal, reference, order=order, mu=mu)


def estimate_signal_quality(signal: np.ndarray, fs: float) -> dict:
    """Compute frequency-domain signal quality metrics.

    The power spectral density is estimated with Welch's method (Hann window,
    50 % overlap, ~8 s segments).  The in-band signal-to-noise ratio is

    .. math::
        \\mathrm{SNR_{dB}} = 10 \\log_{10}
        \\left( \\frac{P_{HR}}{P_{total} - P_{HR}} \\right),

    where :math:`P_{HR}` is the power in the cardiac band (0.5–4.0 Hz) and
    :math:`P_{total} - P_{HR}` the residual (out-of-band) power.  This bounded
    definition avoids the arbitrary high-frequency cut-off of the previous
    estimator and is comparable across devices with different sampling rates.

    Calculates:

    - **SNR (dB)**: in-band cardiac power ratio, clamped to [-10, 30] dB.
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

        nperseg = int(min(len(signal), max(32, 8 * fs)))
        frequencies, psd = welch(
            signal,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="linear",
        )

        total_power = float(trapezoid(psd, frequencies))
        hr_band = (frequencies >= 0.5) & (frequencies <= 4.0)
        hr_power = float(trapezoid(psd[hr_band], frequencies[hr_band]))

        if total_power > 0:
            hr_power_ratio = hr_power / total_power
            residual_power = max(total_power - hr_power, 1e-12)
            snr_db = 10 * np.log10(max(hr_power, 1e-12) / residual_power)
            snr_db = float(np.clip(snr_db, -10, 30))
        else:
            hr_power_ratio = 0.0
            snr_db = -10.0

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
            "snr_db": 0.0,
            "kurtosis": 0.0,
            "skewness": 0.0,
            "hr_power_ratio": 0.0,
            "signal_std": 1.0,
            "signal_range": 1.0,
        }
