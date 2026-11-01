"""Inter-device clock synchronisation from accelerometer activity.

The three devices in GalaxyPPG keep independent clocks.  Their offsets
relative to the Polar H10 range from a few seconds (Empatica E4) to more
than a minute (Galaxy Watch 5, some participants), which is far larger than
a heartbeat: beat-level validation is impossible until the clocks are
aligned.

The offset is estimated from a signal that is *independent of the heart
signals under evaluation*: the accelerometer activity envelope.  Every
device carries an accelerometer, and the protocol's posture changes and
locomotion blocks (standing, walking, jogging, transitions between tasks)
produce sharp, shared activity edges.  For each device the gravity-removed
acceleration magnitude is converted to a 1-s RMS envelope sampled at 10 Hz,
log-compressed and z-scored, and the lag maximising the normalised
cross-correlation with the Polar H10 envelope (searched over
``max_lag_s``) is taken as the session clock offset.

Convention: ``offset_s`` is the amount by which the device clock is *ahead*
of the Polar clock, i.e. an event at Polar time ``t`` is stamped
``t + offset_s`` by the device.  Device times are aligned to the Polar time
base with ``t_device - offset_s``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ENVELOPE_HZ = 10.0


@dataclass
class ClockOffset:
    """Session-level clock offset of one device relative to the Polar H10."""

    device: str
    offset_s: float
    correlation: float
    reliable: bool


def activity_envelope(acc: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp,
                      rate_hz: float = ENVELOPE_HZ) -> np.ndarray:
    """Log-RMS activity envelope on a uniform grid ``[t0, t1)`` at *rate_hz*.

    Args:
        acc: Accelerometer frame with ``acc_x_g``/``acc_y_g``/``acc_z_g``.
        t0: Grid start.
        t1: Grid end.
        rate_hz: Envelope sampling rate.

    Returns:
        Envelope array (NaN outside the accelerometer coverage).
    """
    n = int((t1 - t0).total_seconds() * rate_hz)
    out = np.full(n, np.nan)
    if acc is None or acc.empty or n <= 0:
        return out
    t = (acc.index - t0).total_seconds().to_numpy(dtype=float)
    mag = np.sqrt(acc["acc_x_g"] ** 2 + acc["acc_y_g"] ** 2 + acc["acc_z_g"] ** 2).to_numpy(dtype=float)
    dynamic = (mag - np.nanmedian(mag)) ** 2
    bins = np.floor(t * rate_hz).astype(np.int64)
    ok = (bins >= 0) & (bins < n) & np.isfinite(dynamic)
    sums = np.bincount(bins[ok], weights=dynamic[ok], minlength=n)
    counts = np.bincount(bins[ok], minlength=n)
    has = counts > 0
    power = np.full(n, np.nan)
    power[has] = sums[has] / counts[has]
    # 1-s moving RMS (window = rate_hz bins), log-compressed
    window = int(rate_hz)
    filled = np.where(has, power, 0.0)
    kernel = np.ones(window)
    num = np.convolve(filled, kernel, "same")
    den = np.convolve(has.astype(float), kernel, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        rms = np.sqrt(num / den)
    out = np.log10(rms + 1e-3)
    out[den == 0] = np.nan
    return out


def _normalised_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (samples) maximising corr(a(t), b(t + lag)); NaNs ignored."""
    ma, mb = np.isfinite(a), np.isfinite(b)
    a = np.where(ma, a - np.nanmean(a), 0.0)
    b = np.where(mb, b - np.nanmean(b), 0.0)
    n = len(a) + len(b)
    size = 1 << (n - 1).bit_length()
    fa, fb = np.fft.rfft(a, size), np.fft.rfft(b, size)
    cross = np.fft.irfft(np.conj(fa) * fb, size)
    energy_a = np.fft.irfft(np.conj(np.fft.rfft(a * a, size)) * np.fft.rfft(mb.astype(float), size), size)
    energy_b = np.fft.irfft(np.conj(np.fft.rfft(ma.astype(float), size)) * np.fft.rfft(b * b, size), size)
    lags = np.r_[np.arange(0, max_lag + 1), np.arange(-max_lag, 0)]
    idx = lags % size
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cross[idx] / np.sqrt(np.maximum(energy_a[idx], 1e-12) * np.maximum(energy_b[idx], 1e-12))
    best = int(np.nanargmax(corr))
    return int(lags[best]), float(corr[best])


def estimate_clock_offset(
    reference_acc: pd.DataFrame,
    device_acc: pd.DataFrame,
    device: str,
    max_lag_s: float = 120.0,
    min_correlation: float = 0.3,
) -> ClockOffset:
    """Estimate a device's session clock offset relative to the reference.

    Args:
        reference_acc: Polar H10 accelerometer frame.
        device_acc: Device accelerometer frame.
        device: Device identifier (for reporting).
        max_lag_s: Search range in seconds.
        min_correlation: Minimum peak correlation for a reliable estimate.

    Returns:
        ``ClockOffset``; ``offset_s = 0`` and ``reliable = False`` when no
        estimate is possible.
    """
    if reference_acc is None or device_acc is None or reference_acc.empty or device_acc.empty:
        return ClockOffset(device, 0.0, np.nan, False)
    t0 = min(reference_acc.index[0], device_acc.index[0]) - pd.Timedelta(seconds=max_lag_s)
    t1 = max(reference_acc.index[-1], device_acc.index[-1]) + pd.Timedelta(seconds=max_lag_s)
    ref = activity_envelope(reference_acc, t0, t1)
    dev = activity_envelope(device_acc, t0, t1)
    if np.isfinite(ref).sum() < 600 or np.isfinite(dev).sum() < 600:
        return ClockOffset(device, 0.0, np.nan, False)
    lag, corr = _normalised_xcorr(ref, dev, int(max_lag_s * ENVELOPE_HZ))
    offset = lag / ENVELOPE_HZ
    reliable = bool(np.isfinite(corr) and corr >= min_correlation
                    and abs(offset) < max_lag_s - 1.0)
    if not reliable:
        logger.warning("%s: unreliable clock offset (r=%.2f, %.1f s)", device, corr, offset)
    return ClockOffset(device, float(offset), float(corr), reliable)


def estimate_local_offset(
    reference_acc: pd.DataFrame,
    device_acc: pd.DataFrame,
    device: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    session_offset_s: float,
    context_s: float = 300.0,
    search_s: float = 10.0,
    min_correlation: float = 0.3,
) -> ClockOffset:
    """Clock offset around one window, refined from the session offset.

    Consumer-device clocks drift and are occasionally re-synchronised
    mid-session (NTP), so a single session offset can be wrong by seconds
    late in the protocol.  The activity envelopes are cross-correlated over
    ``[start - context_s, end + context_s]`` (device side pre-shifted by the
    session offset) within ``+-search_s``; the refined offset is used when
    the local correlation is reliable, otherwise the session offset.

    Returns:
        ``ClockOffset`` with the total (session + local) offset.
    """
    fallback = ClockOffset(device, float(session_offset_s), np.nan, False)
    if reference_acc is None or device_acc is None or reference_acc.empty or device_acc.empty:
        return fallback
    t0 = start - pd.Timedelta(seconds=context_s)
    t1 = end + pd.Timedelta(seconds=context_s)
    shift = pd.Timedelta(seconds=float(session_offset_s))
    ref = activity_envelope(reference_acc.loc[(reference_acc.index >= t0) & (reference_acc.index <= t1)], t0, t1)
    dev_window = device_acc.loc[(device_acc.index >= t0 + shift - pd.Timedelta(seconds=search_s))
                                & (device_acc.index <= t1 + shift + pd.Timedelta(seconds=search_s))]
    shifted = dev_window.copy()
    shifted.index = shifted.index - shift
    dev = activity_envelope(shifted, t0, t1)
    if np.isfinite(ref).sum() < 600 or np.isfinite(dev).sum() < 600:
        return fallback
    lag, corr = _normalised_xcorr(ref, dev, int(search_s * ENVELOPE_HZ))
    local = lag / ENVELOPE_HZ
    reliable = bool(np.isfinite(corr) and corr >= min_correlation and abs(local) < search_s - 0.5)
    if not reliable:
        return fallback
    return ClockOffset(device, float(session_offset_s + local), float(corr), True)


def resolve_window_offsets(
    local_offsets: list[float | None],
    window_times_s: list[float],
    session_offset_s: float,
    tolerance_s: float = 1.0,
    min_support: int = 2,
) -> tuple[list[float], list[str]]:
    """Robust per-window clock offsets from noisy local estimates.

    Local (per-window) cross-correlation occasionally mis-locks on
    low-activity windows.  A mis-lock is an *isolated* value, whereas a real
    change of the device clock (a resynchronisation in mid-session) is
    supported by every later window.  A reliable local estimate is therefore
    accepted when at least ``min_support`` other reliable estimates of the
    same device and participant agree with it within ``tolerance_s``.
    Every other window takes the offset of the accepted window nearest in
    time, and the session offset when none is accepted.

    Args:
        local_offsets: Local offset per window (``None`` when unreliable).
        window_times_s: Time of every window (any consistent unit, e.g. the
            window midpoint in seconds).
        session_offset_s: Whole-session offset (fallback).
        tolerance_s: Agreement tolerance between estimates.
        min_support: Number of agreeing other estimates required.

    Returns:
        Tuple ``(offsets, sources)``; *sources* is ``"local"`` for accepted
        windows and ``"neighbour"`` or ``"session"`` otherwise.
    """
    values = np.array([np.nan if o is None else float(o) for o in local_offsets])
    times = np.asarray(window_times_s, dtype=float)
    reliable = np.isfinite(values)
    support = np.zeros(len(values), dtype=int)
    for i in np.where(reliable)[0]:
        agree = reliable & (np.abs(values - values[i]) <= tolerance_s)
        support[i] = int(agree.sum()) - 1
    accepted = reliable & (support >= min_support)
    offsets: list[float] = []
    sources: list[str] = []
    for i in range(len(values)):
        if accepted[i]:
            offsets.append(float(values[i]))
            sources.append("local")
        elif accepted.any():
            nearest = np.where(accepted)[0][np.argmin(np.abs(times[accepted] - times[i]))]
            offsets.append(float(values[nearest]))
            sources.append("neighbour")
        else:
            offsets.append(float(session_offset_s))
            sources.append("session")
    return offsets, sources
