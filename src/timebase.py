"""Sample clocks: exact per-sample times from device timestamps.

Device exports never sample at exactly the nominal rate: the Polar H10 runs
at ~130.28 Hz instead of 130 Hz, and the Galaxy Watch 5 at ~24.8 Hz instead of
25 Hz, with millisecond-quantised, jittered timestamps and occasional
dropped packets.  Converting a peak *index* to a time with ``index / fs``
therefore accumulates drift (0.3 s over three minutes for the Polar) and
shifts every beat after a lost packet.

``fit_sample_clock`` replaces the nominal clock with a piecewise-linear fit
of the recorded timestamps against the sample index: each contiguous run
(split at gaps larger than ``gap_factor`` median sample periods) receives a
least-squares line, which removes timestamp quantisation and jitter while
preserving the true sampling rate and every genuine gap.  Fractional peak
positions are then mapped to absolute times by linear interpolation of the
fitted clock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SampleClock:
    """Fitted per-sample time base for one signal window.

    Attributes:
        t0: Absolute time of sample 0 (timezone-naive).
        seconds: Fitted time of every sample, in seconds relative to ``t0``.
        fs: Effective sampling rate (samples per second over the window,
            excluding gaps).
        gap_after: Boolean array marking samples followed by a gap.
    """

    t0: pd.Timestamp
    seconds: np.ndarray
    fs: float
    gap_after: np.ndarray

    def to_seconds(self, positions: np.ndarray) -> np.ndarray:
        """Map fractional sample positions to seconds relative to ``t0``."""
        positions = np.asarray(positions, dtype=float)
        if len(positions) == 0:
            return positions
        return np.interp(positions, np.arange(len(self.seconds)), self.seconds)

    def to_times(self, positions: np.ndarray) -> pd.DatetimeIndex:
        """Map fractional sample positions to absolute timestamps."""
        return self.t0 + pd.to_timedelta(self.to_seconds(positions), unit="s")

    @property
    def n_gaps(self) -> int:
        return int(self.gap_after.sum())


def fit_sample_clock(index: pd.DatetimeIndex, gap_factor: float = 2.5) -> SampleClock:
    """Fit a piecewise-linear clock to a signal's timestamps.

    Args:
        index: Timestamps of consecutive samples (sorted, unique).
        gap_factor: A step larger than this multiple of the median sample
            period starts a new linear run (dropped packets / pauses).

    Returns:
        ``SampleClock`` for the window.
    """
    n = len(index)
    if n == 0:
        return SampleClock(pd.Timestamp(0), np.array([]), np.nan, np.array([], dtype=bool))
    t0 = pd.Timestamp(index[0])
    raw = (pd.DatetimeIndex(index) - t0).total_seconds().to_numpy(dtype=float)
    if n < 3:
        fs = 1.0 / np.median(np.diff(raw)) if n == 2 and raw[1] > raw[0] else np.nan
        return SampleClock(t0, raw, fs, np.zeros(n, dtype=bool))

    steps = np.diff(raw)
    period = float(np.median(steps[steps > 0])) if np.any(steps > 0) else np.nan
    gap_after = np.zeros(n, dtype=bool)
    if np.isfinite(period):
        gap_after[:-1] = steps > gap_factor * period

    fitted = np.empty(n)
    starts = np.r_[0, np.where(gap_after[:-1])[0] + 1]
    ends = np.r_[starts[1:], n]
    total_samples = 0
    total_time = 0.0
    for a, b in zip(starts, ends, strict=False):
        k = np.arange(b - a, dtype=float)
        if b - a >= 3:
            slope, intercept = np.polyfit(k, raw[a:b], 1)
            fitted[a:b] = intercept + slope * k
            total_samples += b - a - 1
            total_time += slope * (b - a - 1)
        else:
            fitted[a:b] = raw[a:b]
    # Anchor to the recorded start so absolute times stay unbiased.
    fitted -= fitted[0]
    fs = total_samples / total_time if total_time > 0 else (1.0 / period if period else np.nan)
    return SampleClock(t0, fitted, float(fs), gap_after)
