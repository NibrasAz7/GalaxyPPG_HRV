"""HRV metrics: gap-aware segmentation, artifact correction, and HRV indices.

Implements the post-peak-detection stage of the pipeline:

1. **IBI calculation** — peak indices → inter-beat intervals (ms).
2. **Artifact detection** — physiological range enforcement
   (300–2000 ms, i.e. 30–200 bpm) plus a rolling-median criterion: an
   interval deviating from the local median by more than 1.5x (missed beat)
   or falling below 0.6x (extra beat) is flagged invalid.
3. **Gap-aware segmentation** — contiguous runs of invalid intervals whose
   cumulative duration exceeds ``gap_threshold_ms`` are treated as genuine
   signal gaps and are never bridged; the NN series is split at each gap.
4. **Interpolation** — short invalid runs (isolated missed beats or ectopics)
   are reconstructed with a monotone PCHIP interpolant evaluated on the
   beat-time axis, preserving the original beat timing.
5. **HRV indices** — time-domain indices, Poincaré descriptors, approximate
   entropy, and duration-gated frequency-domain indices computed with Welch's
   method on a uniformly resampled tachogram.
6. **Aggregation** — indices are averaged across valid segments using
   duration-based weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import welch

logger = logging.getLogger(__name__)

# Frequency bands (Hz) used for spectral HRV indices.
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)

# Complete set of per-segment indices aggregated by ``aggregate_hrv_metrics``.
# Every segment contributes one (possibly NaN) value for each key so that the
# duration-weighted averages are always computed over aligned arrays.
TIME_DOMAIN_KEYS = (
    "mean_nn",
    "sdnn",
    "rmssd",
    "sdsd",
    "pnn20",
    "pnn50",
    "pct_jitter",
    "cvnn",
    "mean_hr",
    "min_hr",
    "max_hr",
)
POINCARE_KEYS = ("sd1", "sd2", "sd2_sd1")
FREQUENCY_KEYS = ("lf_power", "hf_power", "total_power", "lf_hf_ratio", "lf_nu", "hf_nu")
SEGMENT_METRIC_KEYS = TIME_DOMAIN_KEYS + POINCARE_KEYS + ("apen", "sampen") + FREQUENCY_KEYS


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


def robust_hr(
    ibis: np.ndarray,
    ibi_range: tuple[float, float] = (300.0, 2000.0),
) -> float:
    """Estimate heart rate from the median of in-range raw intervals.

    The median is used instead of the mean because it is insensitive to the
    isolated missed or extra beats that dominate motion-contaminated
    recordings.  This estimator is therefore computed for *every* recording
    and provides a like-for-like comparison against the reference.

    Args:
        ibis: Raw inter-beat intervals in milliseconds.
        ibi_range: ``(min_ms, max_ms)`` physiological NN range.

    Returns:
        Heart rate in bpm, or ``NaN`` when no interval is in range.
    """
    valid = ibis[(ibis >= ibi_range[0]) & (ibis <= ibi_range[1])]
    if len(valid) == 0:
        return np.nan
    return float(60000.0 / np.median(valid))


def _relative_seconds(times) -> np.ndarray:
    """Convert timestamps to seconds elapsed since the first timestamp.

    Uses pandas timedelta arithmetic so that the result is independent of the
    underlying datetime resolution (ms, us or ns).

    Args:
        times: Array-like of datetime values.

    Returns:
        Float array of relative times in seconds.
    """
    idx = pd.DatetimeIndex(times)
    return (idx - idx[0]).total_seconds().to_numpy(dtype=float)


def detect_ibi_artifacts(
    ibis: np.ndarray,
    ibi_range: tuple[float, float] = (300.0, 2000.0),
    missed_beat_factor: float = 1.5,
    extra_beat_factor: float = 0.6,
    median_window_beats: int = 11,
) -> np.ndarray:
    """Flag physiologically implausible or misdetected intervals.

    An interval is marked invalid when it falls outside *ibi_range*, when it
    exceeds the local median by more than *missed_beat_factor* (a missed or
    extra systolic peak), or when it falls below *extra_beat_factor* times the
    local median.  The local median is computed over a centred rolling window,
    which makes the criterion robust to isolated detection errors while
    preserving genuine beat-to-beat variability such as respiratory sinus
    arrhythmia.

    Args:
        ibis: Raw inter-beat intervals in milliseconds.
        ibi_range: ``(min_ms, max_ms)`` physiological NN range.
        missed_beat_factor: Factor above the local median taken as a missed beat.
        extra_beat_factor: Factor below the local median taken as an extra beat.
        median_window_beats: Width of the centred rolling median window.

    Returns:
        Boolean array where ``True`` marks an invalid interval.
    """
    invalid = ~np.isfinite(ibis) | (ibis < ibi_range[0]) | (ibis > ibi_range[1])
    if len(ibis) < 3:
        return invalid

    width = max(3, int(median_window_beats))
    if width % 2 == 0:
        width += 1
    local_median = (
        pd.Series(ibis)
        .rolling(window=width, center=True, min_periods=3)
        .median()
        .to_numpy()
    )

    with np.errstate(invalid="ignore"):
        missed = ibis > missed_beat_factor * local_median
        extra = ibis < extra_beat_factor * local_median

    return invalid | np.nan_to_num(missed, nan=False) | np.nan_to_num(extra, nan=False)


def detect_gaps(
    ibis: np.ndarray,
    invalid: np.ndarray,
    gap_threshold_ms: float = 3000.0,
) -> np.ndarray:
    """Identify invalid runs that represent genuine data gaps.

    Consecutive invalid intervals are grouped into runs.  A run whose
    cumulative duration reaches *gap_threshold_ms* cannot be attributed to a
    single missed beat and is therefore treated as a signal gap rather than
    being interpolated.

    Args:
        ibis: Raw inter-beat intervals in milliseconds.
        invalid: Boolean mask from :func:`detect_ibi_artifacts`.
        gap_threshold_ms: Cumulative duration (ms) above which an invalid run
            is considered a gap.

    Returns:
        Boolean array where ``True`` marks intervals belonging to a gap.
    """
    gap = np.zeros(len(ibis), dtype=bool)
    i = 0
    while i < len(ibis):
        if not invalid[i]:
            i += 1
            continue
        j = i
        while j < len(ibis) and invalid[j]:
            j += 1
        if float(np.sum(ibis[i:j])) >= gap_threshold_ms:
            gap[i:j] = True
        i = j
    return gap


@dataclass
class NNSegment:
    """A contiguous, artifact-corrected NN interval segment.

    Attributes:
        nn: Corrected NN intervals (ms).
        interval_end_times: Timestamp of the end beat of each interval.
        duration_s: Segment duration in seconds.
        n_beats: Number of beats spanned by the segment.
        interpolated_fraction: Fraction of intervals that were corrected
            (Lipponen-Tarvainen) or interpolated (residual artifacts) - the
            correction burden.
        start_time: Time of the first beat of the segment.
        mean_sqi: Mean template-matching SQI of the segment's intervals.
        synthetic: Boolean per interval: bounded by a synthetic (interpolated)
            beat or itself interpolated.  Excluded from successive-difference
            indices.
    """

    nn: np.ndarray
    interval_end_times: pd.DatetimeIndex
    duration_s: float
    n_beats: int
    interpolated_fraction: float
    start_time: pd.Timestamp | None = None
    mean_sqi: float = np.nan
    synthetic: np.ndarray | None = None


def _interpolate_invalid(
    nn: np.ndarray,
    invalid: np.ndarray,
    interval_end_times: pd.DatetimeIndex,
    method: str = "pchip",
) -> tuple[np.ndarray, float] | None:
    """Replace invalid intervals with interpolated values.

    Args:
        nn: NN intervals (ms) of a single segment.
        invalid: Boolean mask marking invalid intervals.
        interval_end_times: Timestamps associated with each interval.
        method: Interpolation family — ``"pchip"`` (monotone cubic, default),
            ``"linear"`` or ``"cubic"`` (natural cubic spline).

    Returns:
        Tuple of ``(corrected_nn, interpolated_fraction)``, or ``None`` when
        fewer than two valid intervals are available.
    """
    if not np.any(invalid):
        return nn.copy(), 0.0

    valid = ~invalid
    if int(valid.sum()) < 2:
        return None

    t = _relative_seconds(interval_end_times)
    if method == "linear":
        corrected_values = np.interp(t[invalid], t[valid], nn[valid])
    elif method == "cubic":
        from scipy.interpolate import CubicSpline

        corrected_values = CubicSpline(t[valid], nn[valid])(t[invalid])
    else:
        corrected_values = PchipInterpolator(t[valid], nn[valid])(t[invalid])
    corrected = nn.copy()
    corrected[invalid] = corrected_values
    return corrected, float(np.mean(invalid))


def segment_nn_series(
    ibis: np.ndarray,
    beat_times: pd.DatetimeIndex,
    invalid: np.ndarray,
    gap: np.ndarray,
    min_duration_s: float = 30.0,
    min_beats: int = 20,
    max_interpolated_fraction: float = 0.5,
    max_jitter_percent: float = 15.0,
    reference_hr: float | None = None,
    max_hr_deviation_percent: float = 25.0,
    interpolation_method: str = "linear",
    corrected: np.ndarray | None = None,
    interval_sqi: np.ndarray | None = None,
    min_segment_sqi: float | None = None,
    rhythm_hr: float | None = None,
    rhythm_ratio_range: tuple[float, float] = (0.67, 1.5),
    corrected_beats: np.ndarray | None = None,
    cadence_hz: float | None = None,
    uncertain: np.ndarray | None = None,
    cadence_tolerance: float = 0.08,
    cadence_multiples: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> tuple[list[NNSegment], dict[str, int]]:
    """Split an IBI series into valid, artifact-corrected NN segments.

    The series is cut at every gap interval.  Each resulting run is
    retained only when it passes, in order:

    * ``duration`` - at least ``min_duration_s`` and ``min_beats``;
    * ``interpolation`` - correction burden (Lipponen-Tarvainen corrections
      plus residual interpolated intervals) at most
      ``max_interpolated_fraction``;
    * ``jitter`` - residual successive-difference jitter;
    * ``sqi`` - mean template-matching SQI at least ``min_segment_sqi``;
    * ``rhythm`` - segment HR within ``rhythm_ratio_range`` of the
      standalone spectral HR of the same PPG (rejects half- and double-rate
      beat chains, which no interval-based rule can detect);
    * ``cadence`` - when the accelerometer is strongly periodic
      (``cadence_hz`` given), segment HR within ``cadence_tolerance`` of
      ``k x cadence`` (k in ``cadence_multiples``) is rejected as locked onto
      the step / arm-swing artifact;
    * ``hr_deviation`` - optional device-HR cross-check (disabled by default
      because it conditions acceptance on the vendor algorithm).

    Args:
        ibis: Inter-beat intervals in milliseconds.
        beat_times: Beat timestamps (length = ``len(ibis) + 1``).
        invalid: Boolean mask of residual invalid intervals (interpolated).
        gap: Boolean mask of gap intervals (never bridged).
        min_duration_s: Minimum segment duration in seconds.
        min_beats: Minimum number of beats per segment.
        max_interpolated_fraction: Maximum correction burden.
        max_jitter_percent: Maximum percentage of successive differences
            exceeding 20 % of the median NN after correction.
        reference_hr: Optional device HR for the (non-default) cross-check.
        max_hr_deviation_percent: Tolerance of the device-HR cross-check.
        interpolation_method: ``"linear"``, ``"pchip"`` or ``"cubic"``.
        corrected: Boolean mask of intervals altered by artifact correction.
        interval_sqi: Template-matching SQI per interval.
        min_segment_sqi: Minimum mean SQI per segment.
        rhythm_hr: Standalone spectral HR (bpm) for the rhythm gate.
        rhythm_ratio_range: Accepted ratio segment HR / ``rhythm_hr``.
        corrected_beats: Boolean per beat (``len(ibis) + 1``) marking beats
            altered by correction.  When given, the burden is expressed in
            beats (corrected beats + residual interpolated intervals, over
            beats), matching the Kubios definition; otherwise in intervals.

    Returns:
        Tuple of ``(segments, reasons)``; *reasons* counts rejected candidate
        segments per gate and their ``total``.
    """
    segments: list[NNSegment] = []
    reasons = {"duration": 0, "interpolation": 0, "jitter": 0, "sqi": 0,
               "rhythm": 0, "cadence": 0, "hr_deviation": 0}
    corrected = (np.zeros(len(ibis), dtype=bool) if corrected is None
                 else np.asarray(corrected, dtype=bool))

    def _collect(i0: int, i1: int) -> None:
        if i1 <= i0:
            return
        duration_s = (beat_times[i1] - beat_times[i0]).total_seconds()
        n_beats = i1 - i0 + 1
        if duration_s < min_duration_s or n_beats < min_beats:
            reasons["duration"] += 1
            return

        nn_seg = ibis[i0:i1]
        times_seg = beat_times[i0 + 1 : i1 + 1]
        interpolated = _interpolate_invalid(nn_seg, invalid[i0:i1], times_seg,
                                            method=interpolation_method)
        if interpolated is None:
            reasons["interpolation"] += 1
            return
        nn_corrected, _ = interpolated
        if corrected_beats is not None and len(corrected_beats) == len(ibis) + 1:
            burden = float((np.sum(corrected_beats[i0:i1 + 1]) + np.sum(invalid[i0:i1] & ~corrected[i0:i1]))
                           / n_beats)
        else:
            burden = float(np.mean(invalid[i0:i1] | corrected[i0:i1]))
        if burden > max_interpolated_fraction:
            reasons["interpolation"] += 1
            return

        median_nn = float(np.median(nn_corrected))
        jitter = (
            float(np.mean(np.abs(np.diff(nn_corrected)) > 0.20 * median_nn) * 100.0)
            if median_nn > 0 else 0.0
        )
        if jitter > max_jitter_percent:
            reasons["jitter"] += 1
            return

        mean_sqi = np.nan
        if interval_sqi is not None:
            values = np.asarray(interval_sqi[i0:i1], dtype=float)
            mean_sqi = float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan
            if min_segment_sqi is not None and not (np.isfinite(mean_sqi) and mean_sqi >= min_segment_sqi):
                reasons["sqi"] += 1
                return

        segment_hr = 60000.0 / median_nn if median_nn > 0 else np.nan
        if rhythm_hr is not None and np.isfinite(rhythm_hr) and rhythm_hr > 0 and np.isfinite(segment_hr):
            ratio = segment_hr / rhythm_hr
            if not (rhythm_ratio_range[0] <= ratio <= rhythm_ratio_range[1]):
                reasons["rhythm"] += 1
                return

        if cadence_hz is not None and np.isfinite(cadence_hz) and cadence_hz > 0 and np.isfinite(segment_hr):
            segment_hz = segment_hr / 60.0
            if any(abs(segment_hz - k * cadence_hz) <= cadence_tolerance * k * cadence_hz
                   for k in cadence_multiples):
                reasons["cadence"] += 1
                return

        if reference_hr is not None and np.isfinite(reference_hr) and reference_hr > 0:
            deviation = abs(segment_hr - reference_hr) / reference_hr * 100.0
            if np.isfinite(deviation) and deviation > max_hr_deviation_percent:
                reasons["hr_deviation"] += 1
                return

        segments.append(
            NNSegment(
                nn=nn_corrected,
                interval_end_times=times_seg,
                duration_s=duration_s,
                n_beats=n_beats,
                interpolated_fraction=burden,
                start_time=beat_times[i0],
                mean_sqi=mean_sqi,
                synthetic=(invalid[i0:i1] | corrected[i0:i1]
                           | (uncertain[i0:i1] if uncertain is not None else False)).copy(),
            )
        )

    start = 0
    for i in range(len(ibis)):
        if gap[i]:
            _collect(start, i)
            start = i + 1
    _collect(start, len(ibis))

    reasons["total"] = sum(reasons.values())
    return segments, reasons


def successive_differences(nn: np.ndarray, synthetic: np.ndarray | None = None) -> np.ndarray:
    """Successive NN differences, excluding pairs that involve a synthetic interval.

    Following the NN-only convention (Task Force, 1996), differences are
    taken only between two adjacent intervals that are both bounded by
    detected (or signal-recovered) beats.
    """
    differences = np.diff(nn)
    if synthetic is None or len(synthetic) != len(nn):
        return differences
    synthetic = np.asarray(synthetic, dtype=bool)
    keep = ~(synthetic[:-1] | synthetic[1:])
    return differences[keep] if keep.sum() >= 2 else differences


def compute_time_domain(nn: np.ndarray, synthetic: np.ndarray | None = None) -> dict[str, float]:
    """Compute standard time-domain HRV indices.

    Args:
        nn: Cleaned NN intervals in milliseconds.
        synthetic: Optional per-interval mask of interpolated intervals,
            excluded from successive-difference indices.

    Returns:
        Dictionary with ``mean_nn``, ``sdnn``, ``rmssd``, ``sdsd``, ``pnn20``,
        ``pnn50``, ``cvnn``, ``mean_hr``, ``min_hr`` and ``max_hr``.
    """
    if len(nn) < 2:
        return {}

    differences = successive_differences(nn, synthetic)
    mean_nn = float(np.mean(nn))
    sdnn = float(np.std(nn, ddof=1))
    median_nn = float(np.median(nn))

    return {
        "mean_nn": mean_nn,
        "sdnn": sdnn,
        "rmssd": float(np.sqrt(np.mean(differences ** 2))),
        "sdsd": float(np.std(differences, ddof=1)),
        "pnn20": float(np.mean(np.abs(differences) > 20.0) * 100.0),
        "pnn50": float(np.mean(np.abs(differences) > 50.0) * 100.0),
        "pct_jitter": float(np.mean(np.abs(differences) > 0.20 * median_nn) * 100.0)
        if median_nn > 0
        else np.nan,
        "cvnn": float(sdnn / mean_nn * 100.0) if mean_nn > 0 else np.nan,
        "mean_hr": float(60000.0 / mean_nn) if mean_nn > 0 else np.nan,
        "min_hr": float(60000.0 / np.max(nn)) if np.max(nn) > 0 else np.nan,
        "max_hr": float(60000.0 / np.min(nn)) if np.min(nn) > 0 else np.nan,
    }


def compute_poincare(nn: np.ndarray, synthetic: np.ndarray | None = None) -> dict[str, float]:
    """Compute Poincaré plot descriptors (SD1, SD2, SD2/SD1).

    Args:
        nn: Cleaned NN intervals in milliseconds.
        synthetic: Optional per-interval mask (see :func:`compute_time_domain`).

    Returns:
        Dictionary with ``sd1``, ``sd2`` and ``sd2_sd1``.
    """
    if len(nn) < 3:
        return {}

    differences = successive_differences(nn, synthetic)
    sdnn = float(np.std(nn, ddof=1))
    sd_successive = float(np.std(differences, ddof=1))

    sd1 = sd_successive / np.sqrt(2.0)
    sd2_squared = 2.0 * sdnn ** 2 - 0.5 * sd_successive ** 2
    sd2 = float(np.sqrt(sd2_squared)) if sd2_squared > 0 else np.nan

    return {
        "sd1": float(sd1),
        "sd2": sd2,
        "sd2_sd1": float(sd2 / sd1) if sd1 > 0 and np.isfinite(sd2) else np.nan,
    }


def approximate_entropy(
    nn: np.ndarray,
    m: int = 2,
    r_factor: float = 0.2,
) -> float:
    """Compute approximate entropy (ApEn) of the NN series.

    Args:
        nn: Cleaned NN intervals in milliseconds.
        m: Embedding dimension.
        r_factor: Tolerance as a multiple of the series standard deviation.

    Returns:
        ApEn value, or ``NaN`` if it cannot be computed.
    """
    n = len(nn)
    if n < m + 2:
        return np.nan

    tolerance = r_factor * float(np.std(nn, ddof=1))
    if tolerance <= 0:
        return np.nan

    def _phi(embedding: int) -> float:
        windows = np.array([nn[i : i + embedding] for i in range(n - embedding + 1)])
        counts = np.sum(
            np.max(np.abs(windows[:, None, :] - windows[None, :, :]), axis=2) <= tolerance,
            axis=1,
        )
        return float(np.mean(np.log(counts / (n - embedding + 1))))

    return float(_phi(m) - _phi(m + 1))


def band_power(frequencies: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    """Integrate a one-sided PSD over ``[band[0], band[1])`` as sum(PSD) x df.

    The PSD is evaluated on a zero-padded (fine) frequency grid, so the
    rectangle rule integrates the full band width.  The trapezoid over the
    in-band bins used previously covered only ``(n_bins - 1) x df`` and
    under-estimated power by 6-20 %, by an amount that depended on segment
    length (i.e. differently for PPG and ECG).
    """
    mask = (frequencies >= band[0]) & (frequencies < band[1])
    if not np.any(mask):
        return np.nan
    df = float(frequencies[1] - frequencies[0])
    return float(np.sum(psd[mask]) * df)


def compute_frequency_domain(
    nn: np.ndarray,
    resample_hz: float = 4.0,
    min_hf_duration_s: float = 60.0,
    min_lf_duration_s: float = 120.0,
    nperseg_s: float = 64.0,
) -> dict[str, float]:
    """Compute spectral HRV indices from the NN tachogram.

    The tachogram (beat times = cumulative NN) is resampled with a cubic
    spline onto a uniform grid spanning the observed beats only (no
    extrapolation), linearly
    detrended, and analysed with Welch's method using a *fixed* window
    length (``nperseg_s``, Hann, 50 % overlap; shorter segments use a single
    full-length window) and a fine zero-padded grid.  Band power is
    integrated with :func:`band_power` in ms^2.

    The LF band is reported only for segments of at least
    ``min_lf_duration_s`` and the HF band for at least
    ``min_hf_duration_s``; VLF requires 5 min and is not reported.

    Args:
        nn: Cleaned NN intervals in milliseconds.
        resample_hz: Uniform resampling rate in Hz.
        min_hf_duration_s: Minimum segment duration for HF estimation.
        min_lf_duration_s: Minimum segment duration for LF estimation.
        nperseg_s: Welch window length in seconds.

    Returns:
        Dictionary with ``lf_power``, ``hf_power``, ``total_power``,
        ``lf_hf_ratio``, ``lf_nu`` and ``hf_nu``.
    """
    if len(nn) < 8:
        return {}

    beat_t = np.cumsum(nn) / 1000.0
    duration_s = float(beat_t[-1] - beat_t[0])
    if duration_s < min_hf_duration_s:
        return {}

    grid = np.arange(beat_t[0], beat_t[-1], 1.0 / resample_hz)
    if len(grid) < 16:
        return {}
    try:
        # Cubic spline, as in Kubios HRV: the monotone PCHIP interpolant used
        # previously clips the extrema of the tachogram and attenuated HF
        # power by ~8 % at typical heart rates.
        tachogram = CubicSpline(beat_t, nn)(grid)
    except ValueError:
        return {}

    slopes = np.polyfit(grid, tachogram, 1)
    detrended = tachogram - np.polyval(slopes, grid)

    nperseg = int(min(len(detrended), round(nperseg_s * resample_hz)))
    nfft = max(4096, 1 << (nperseg - 1).bit_length())
    frequencies, psd = welch(detrended, fs=resample_hz, window="hann", nperseg=nperseg,
                             noverlap=nperseg // 2, nfft=nfft, detrend=False)

    hf_power = band_power(frequencies, psd, HF_BAND)
    lf_power = band_power(frequencies, psd, LF_BAND) if duration_s >= min_lf_duration_s else np.nan
    if not np.isfinite(hf_power):
        return {}
    return {"lf_power": lf_power, "hf_power": hf_power, **_spectral_ratios(lf_power, hf_power)}


def _spectral_ratios(lf_power: float, hf_power: float) -> dict[str, float]:
    """Total power, LF/HF and normalised units from band powers."""
    if not (np.isfinite(lf_power) and np.isfinite(hf_power)) or (lf_power + hf_power) <= 0:
        return {"total_power": np.nan, "lf_hf_ratio": np.nan, "lf_nu": np.nan, "hf_nu": np.nan}
    total = lf_power + hf_power
    return {
        "total_power": float(total),
        "lf_hf_ratio": float(lf_power / hf_power) if hf_power > 0 else np.nan,
        "lf_nu": float(100.0 * lf_power / total),
        "hf_nu": float(100.0 * hf_power / total),
    }


def sample_entropy(nn: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Sample entropy (Richman & Moorman, 2000); less length-biased than ApEn.

    Args:
        nn: Cleaned NN intervals in milliseconds.
        m: Embedding dimension.
        r_factor: Tolerance as a multiple of the series SD.

    Returns:
        SampEn, or ``NaN`` when undefined (no template matches).
    """
    n = len(nn)
    if n < m + 3:
        return np.nan
    tolerance = r_factor * float(np.std(nn, ddof=1))
    if tolerance <= 0:
        return np.nan

    def _count(dim: int) -> int:
        windows = np.array([nn[i: i + dim] for i in range(n - m)])
        dist = np.max(np.abs(windows[:, None, :] - windows[None, :, :]), axis=2)
        return int((np.sum(dist <= tolerance) - len(windows)) // 2)

    b = _count(m)
    a = _count(m + 1)
    if a == 0 or b == 0:
        return np.nan
    return float(-np.log(a / b))


def _duration_weighted_average(values: list[float], weights: list[float]) -> float:
    """Average finite values using duration-based weights.

    Args:
        values: Metric value per segment (may contain ``NaN``).
        weights: Segment durations in seconds.

    Returns:
        Weighted average, or ``NaN`` when no finite value is available.

    Raises:
        ValueError: If *values* and *weights* have different lengths.
    """
    values_arr = np.asarray(values, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    if values_arr.shape != weights_arr.shape:
        raise ValueError(
            f"values and weights must have matching shapes, got {values_arr.shape} and {weights_arr.shape}"
        )
    finite = np.isfinite(values_arr)
    if not np.any(finite):
        return np.nan
    return float(np.sum(values_arr[finite] * weights_arr[finite]) / np.sum(weights_arr[finite]))


def build_nn_segments(
    ibis: np.ndarray,
    beat_times: pd.DatetimeIndex,
    config: dict,
    reference_hr: float | None = None,
    extra_invalid: np.ndarray | None = None,
    interpolation_method: str | None = None,
    corrected: np.ndarray | None = None,
    interval_sqi: np.ndarray | None = None,
    rhythm_hr: float | None = None,
    corrected_beats: np.ndarray | None = None,
    cadence_hz: float | None = None,
    uncertain: np.ndarray | None = None,
) -> tuple[list[NNSegment], dict[str, int]]:
    """Detect residual artifacts and gaps, then build valid NN segments.

    Expected input is a beat series that has already been quality-screened
    (template SQI) and corrected (Lipponen-Tarvainen, ``src.artifacts``);
    ``corrected`` marks the intervals that correction altered.  A residual
    rolling-median check is kept as a safety net, complemented by the same
    factors applied to the *window* median (a run of half-rate intervals
    shifts the local median and escapes the rolling rule): flagged intervals
    are interpolated if isolated and cut as gaps if they form a long run.

    Args:
        ibis: Inter-beat intervals in milliseconds.
        beat_times: Beat timestamps (length = ``len(ibis) + 1``).
        config: Parsed configuration dictionary.
        reference_hr: Device HR for the optional ``hrv.device_hr_gate``.
        extra_invalid: Optional mask of intervals forced invalid.
        interpolation_method: Overrides ``hrv.interpolation_method``.
        corrected: Mask of intervals altered by artifact correction.
        interval_sqi: Template-matching SQI per interval.
        rhythm_hr: Standalone spectral HR (bpm) for the rhythm gate.
        corrected_beats: Per-beat correction mask (burden in beats).
        cadence_hz: Dominant accelerometer frequency when motion is
            periodic (``None`` disables the cadence-lock gate).

    Returns:
        Tuple of ``(segments, reasons)``.
    """
    processing = config.get("processing", {})
    hrv_cfg = config.get("hrv", {})

    ibi_range = tuple(processing.get("ibi_range", [300, 2000]))
    missed_factor = float(processing.get("missed_beat_factor", 1.5))
    extra_factor = float(processing.get("extra_beat_factor", 0.6))
    median_window = int(processing.get("median_window_beats", 11))
    gap_threshold = float(processing.get("gap_threshold_ms", 3000.0))
    min_duration = float(hrv_cfg.get("min_segment_duration_s", 30.0))
    min_beats = int(hrv_cfg.get("min_segment_beats", 20))
    max_interpolated = float(hrv_cfg.get("max_interpolated_fraction", 0.05))
    max_jitter = float(hrv_cfg.get("max_jitter_percent", 15.0))
    max_hr_deviation = float(hrv_cfg.get("max_hr_deviation_percent", 25.0))
    method = interpolation_method or hrv_cfg.get("interpolation_method", "linear")
    min_sqi = hrv_cfg.get("min_segment_sqi")
    ratio_range = tuple(hrv_cfg.get("rhythm_ratio_range", [0.67, 1.5]))
    use_device_gate = bool(hrv_cfg.get("device_hr_gate", False))

    empty_reasons = {"duration": 0, "interpolation": 0, "jitter": 0, "sqi": 0,
                     "rhythm": 0, "cadence": 0, "hr_deviation": 0, "total": 0}
    cadence_cfg = hrv_cfg.get("cadence_lock", {})
    if len(ibis) < min_beats or len(beat_times) != len(ibis) + 1:
        return [], empty_reasons

    invalid = detect_ibi_artifacts(
        ibis, ibi_range,
        missed_beat_factor=missed_factor,
        extra_beat_factor=extra_factor,
        median_window_beats=median_window,
    )
    in_range = ibis[(ibis >= ibi_range[0]) & (ibis <= ibi_range[1])]
    if len(in_range):
        window_median = float(np.median(in_range))
        invalid |= (ibis > missed_factor * window_median) | (ibis < extra_factor * window_median)
    if extra_invalid is not None:
        extra = np.asarray(extra_invalid, dtype=bool)
        if len(extra) == len(invalid):
            invalid = invalid | extra
    gap = detect_gaps(ibis, invalid, gap_threshold) | (ibis > ibi_range[1])

    return segment_nn_series(
        ibis, beat_times, invalid, gap,
        min_duration_s=min_duration,
        min_beats=min_beats,
        max_interpolated_fraction=max_interpolated,
        max_jitter_percent=max_jitter,
        reference_hr=reference_hr if use_device_gate else None,
        max_hr_deviation_percent=max_hr_deviation,
        interpolation_method=method,
        corrected=corrected,
        interval_sqi=interval_sqi,
        min_segment_sqi=float(min_sqi) if min_sqi is not None else None,
        rhythm_hr=rhythm_hr,
        rhythm_ratio_range=ratio_range,
        corrected_beats=corrected_beats,
        cadence_hz=cadence_hz,
        uncertain=uncertain,
        cadence_tolerance=float(cadence_cfg.get("tolerance", 0.08)),
        cadence_multiples=tuple(cadence_cfg.get("multiples", [0.5, 1.0, 2.0])),
    )


def segment_metrics(segment_nn: np.ndarray, config: dict,
                    synthetic: np.ndarray | None = None) -> dict[str, float]:
    """All per-segment HRV indices for one NN series."""
    hrv_cfg = config.get("hrv", {})
    metrics = compute_time_domain(segment_nn, synthetic)
    metrics.update(compute_poincare(segment_nn, synthetic))
    m = int(hrv_cfg.get("apen_m", 2))
    r = float(hrv_cfg.get("apen_r_factor", 0.2))
    metrics["apen"] = approximate_entropy(segment_nn, m=m, r_factor=r)
    metrics["sampen"] = sample_entropy(segment_nn, m=m, r_factor=r)
    metrics.update(compute_frequency_domain(
        segment_nn,
        resample_hz=float(hrv_cfg.get("resample_hz", 4.0)),
        min_hf_duration_s=float(hrv_cfg.get("min_hf_duration_s", 60.0)),
        min_lf_duration_s=float(hrv_cfg.get("min_lf_duration_s", 120.0)),
        nperseg_s=float(hrv_cfg.get("welch_window_s", 64.0)),
    ))
    return metrics


def aggregate_segment_metrics(
    per_segment: list[dict[str, float]], weights: list[float],
) -> dict[str, float]:
    """Duration-weighted aggregation; spectral ratios recomputed from powers.

    Ratios (LF/HF, normalised units) are not averaged across segments: they
    are recomputed from the duration-weighted LF and HF powers, so that the
    aggregate equals the ratio of aggregate powers.
    """
    out = {key: _duration_weighted_average([m.get(key, np.nan) for m in per_segment], weights)
           for key in SEGMENT_METRIC_KEYS}
    out.update(_spectral_ratios(out.get("lf_power", np.nan), out.get("hf_power", np.nan)))
    return out


def aggregate_hrv_metrics(
    segments: list[NNSegment],
    reasons: dict[str, int] | int,
    config: dict,
) -> dict[str, float]:
    """Compute duration-weighted HRV indices across NN segments.

    Args:
        segments: Valid NN segments from :func:`build_nn_segments`.
        reasons: Rejection counters from :func:`build_nn_segments`.
        config: Parsed configuration dictionary.

    Returns:
        Flat dictionary of HRV indices and QC counters.
    """
    counts = reasons if isinstance(reasons, dict) else {"total": int(reasons)}
    empty = {
        "n_segments": 0,
        "n_segments_excluded": counts.get("total", 0),
        "excluded_interpolation": counts.get("interpolation", 0),
        "excluded_jitter": counts.get("jitter", 0),
        "excluded_sqi": counts.get("sqi", 0),
        "excluded_rhythm": counts.get("rhythm", 0),
        "excluded_cadence": counts.get("cadence", 0),
        "excluded_hr_deviation": counts.get("hr_deviation", 0),
        "excluded_duration": counts.get("duration", 0),
        "n_beats": 0,
        "n_intervals": 0,
        "duration_s": 0.0,
        "pct_interpolated": np.nan,
        "segment_sqi": np.nan,
        **{key: np.nan for key in SEGMENT_METRIC_KEYS},
    }
    if not segments:
        return empty

    weights = [segment.duration_s for segment in segments]
    total_intervals = sum(len(segment.nn) for segment in segments)
    corrected_intervals = sum(segment.interpolated_fraction * len(segment.nn) for segment in segments)

    aggregated: dict[str, float] = dict(empty)
    aggregated.update({
        "n_segments": len(segments),
        "n_beats": sum(segment.n_beats for segment in segments),
        "n_intervals": total_intervals,
        "duration_s": float(sum(weights)),
        "pct_interpolated": float(100.0 * corrected_intervals / total_intervals)
        if total_intervals > 0 else np.nan,
        "segment_sqi": _duration_weighted_average([s.mean_sqi for s in segments], weights),
    })
    aggregated.update(aggregate_segment_metrics(
        [segment_metrics(segment.nn, config, segment.synthetic) for segment in segments], weights))
    return aggregated


def compute_hrv_metrics(
    ibis: np.ndarray,
    beat_times: pd.DatetimeIndex,
    config: dict,
    reference_hr: float | None = None,
) -> dict[str, float]:
    """Compute segment-aggregated HRV indices from raw inter-beat intervals.

    Convenience wrapper around :func:`build_nn_segments` and
    :func:`aggregate_hrv_metrics`.

    Args:
        ibis: Raw inter-beat intervals in milliseconds.
        beat_times: Beat timestamps (length = ``len(ibis) + 1``).
        config: Parsed configuration dictionary (``processing`` and ``hrv``
            sections).
        reference_hr: Optional independent heart-rate estimate (bpm) used for
            a physiological consistency cross-check.

    Returns:
        Flat dictionary of HRV indices.
    """
    segments, reasons = build_nn_segments(ibis, beat_times, config, reference_hr=reference_hr)
    return aggregate_hrv_metrics(segments, reasons, config)
