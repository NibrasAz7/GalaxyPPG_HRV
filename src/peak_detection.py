"""Multi-method systolic peak detection with ensemble consensus.

Implements four peak-detection strategies and an ensemble voting method:

1. **NeuroKit2** — Elgendi's two-stage moving-average algorithm (PPG) or
   NeuroKit's ECG R-peak detector (ECG).
2. **SciPy** — Adaptive-percentile thresholding with ``find_peaks``.
3. **Slope-sum (SSF)** — Derivative-rectified moving sum, used only for
   severe-motion PPG.
4. **Template matching** — Cross-correlation with a synthetic or
   clean-segment-derived waveform template, used only for severe motion.

The **ensemble** method clusters nearby peaks within a configurable
tolerance and selects the one with the highest weighted vote across methods.
A rhythm-constrained selector (``select_peaks_with_rhythm``) can then enforce
an expected cardiac interval, resolving missed and spurious beats.  Method weights
adapt to signal quality, motion level, and activity type.

Finally, ``select_optimal_peaks`` picks the best method's output based on
device- and context-specific priority ordering.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import correlate, find_peaks, welch

logger = logging.getLogger(__name__)


# Highest heart rate any detector may resolve.  Every detector's refractory
# distance is derived from these values so that no detector structurally
# enforces half-rate output at exercise heart rates (jogging reaches
# 150-190 bpm; the previous 120/150/160 bpm ceilings guaranteed missed beats).
MAX_HR_PPG_BPM = 200.0
MAX_HR_ECG_BPM = 220.0

# Fallback detector priority (overridden by ``detection.priority`` in the
# configuration, which is selected on the development participants only).
DEFAULT_PRIORITY = {
    "PolarH10": {"low": ["neurokit2", "scipy"], "moderate": ["neurokit2", "scipy"],
                 "severe": ["neurokit2", "scipy"]},
    "E4": {"low": ["neurokit2", "ensemble", "scipy"],
           "moderate": ["ensemble", "neurokit2", "scipy"],
           "severe": ["ensemble", "neurokit2", "scipy", "slope"]},
    "GalaxyWatch": {"low": ["ensemble", "neurokit2", "scipy"],
                    "moderate": ["ensemble", "neurokit2", "scipy"],
                    "severe": ["ensemble", "slope", "template", "neurokit2", "scipy"]},
}

# Fiducial authority used when a consensus cluster contains several members:
# the earliest method in this list supplies the beat position.
FIDUCIAL_ORDER = ("neurokit2", "scipy", "template", "slope")


def _refractory_samples(fs: float, device: str) -> int:
    max_hr = MAX_HR_ECG_BPM if device == "PolarH10" else MAX_HR_PPG_BPM
    return max(1, int(np.floor(60.0 / max_hr * fs)))


def relocalize_peaks(
    signal: np.ndarray,
    peaks: np.ndarray,
    fs: float,
    before_s: float = 0.05,
    after_s: float = 0.25,
) -> np.ndarray:
    """Move candidate beats onto the systolic maximum of *signal*.

    Detectors mark different fiducial points (the slope-sum function peaks
    on the upstroke, template correlation on the template alignment).
    Mixing them beat-to-beat adds timing jitter that inflates RMSSD/HF, so
    every candidate is re-localised to the local maximum of the filtered
    signal within ``[p - before_s, p + after_s]``.

    Args:
        signal: Filtered signal.
        peaks: Candidate indices.
        fs: Sampling frequency in Hz.
        before_s: Search window before the candidate (s).
        after_s: Search window after the candidate (s).

    Returns:
        Unique, sorted systolic-peak indices.
    """
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) == 0:
        return peaks
    lo = int(round(before_s * fs))
    hi = int(round(after_s * fs))
    out = np.empty(len(peaks), dtype=int)
    for i, p in enumerate(peaks):
        a = max(0, p - lo)
        b = min(len(signal), p + hi + 1)
        out[i] = a + int(np.argmax(signal[a:b])) if b > a else p
    return np.unique(out)


def detect_peaks_neurokit2(
    signal: np.ndarray,
    fs: float,
    device: str,
    motion_class: str | None = None,
) -> np.ndarray:
    """Detect peaks with NeuroKit2 (Elgendi for PPG, NeuroKit for ECG).

    NeuroKit2's internal artifact correction is always disabled: correction
    is applied exactly once, downstream, by the Lipponen-Tarvainen stage
    (``src.artifacts``), so that every corrected beat is counted in the
    correction burden.

    Args:
        signal: Filtered 1-D signal.
        fs: Sampling frequency in Hz.
        device: Device identifier.
        motion_class: Unused (kept for API compatibility).

    Returns:
        Array of peak indices.
    """
    import neurokit2 as nk

    try:
        if device == "PolarH10":
            _, info = nk.ecg_peaks(signal, sampling_rate=fs, method="neurokit")
            return np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        _, info = nk.ppg_peaks(signal, sampling_rate=fs, method="elgendi", correct_artifacts=False)
        return np.asarray(info.get("PPG_Peaks", []), dtype=int)
    except Exception as e:
        logger.warning("NeuroKit2 peak detection error: %s", e)
        return np.array([], dtype=int)


def detect_peaks_scipy(
    signal: np.ndarray,
    fs: float,
    device: str,
    motion_class: str,
    quality_metrics: dict,
    config: dict,
) -> np.ndarray:
    """Detect peaks using SciPy ``find_peaks`` with adaptive percentile thresholds.

    A rolling window computes local amplitude thresholds.  Prominence is
    raised for low SNR and motion; the refractory distance allows heart
    rates up to ``MAX_HR_PPG_BPM`` (PPG) or ``MAX_HR_ECG_BPM`` (ECG).

    Args:
        signal: Filtered 1-D signal.
        fs: Sampling frequency in Hz.
        device: Device identifier.
        motion_class: Motion level string.
        quality_metrics: Output of ``estimate_signal_quality``.
        config: Parsed configuration dictionary.

    Returns:
        Array of peak indices.
    """
    dev_cfg = config["devices"][device]
    snr_threshold = dev_cfg["snr_threshold"]

    try:
        win = int(8 * fs) if motion_class == "low" else int(5 * fs)
        if device == "PolarH10":
            percentile = 85
        elif motion_class == "low":
            percentile = 75
        elif motion_class == "moderate":
            percentile = 80
        else:
            percentile = 85

        threshold = np.zeros_like(signal)
        step = max(1, win // 2)
        for i in range(0, len(signal), step):
            start = max(0, i - win // 2)
            end = min(len(signal), i + win // 2)
            seg = signal[start:end]
            if len(seg) > 0:
                threshold[start:end] = np.percentile(seg, percentile)

        min_dist = _refractory_samples(fs, device)
        prominence = np.std(signal) * (0.4 if device == "PolarH10" else 0.15)
        if device != "PolarH10" and quality_metrics.get("snr_db", np.inf) < snr_threshold:
            prominence *= 1.5
        if motion_class in ("moderate", "severe"):
            prominence *= 1.3

        peaks, _ = find_peaks(signal, height=threshold, distance=min_dist, prominence=prominence)
        return peaks

    except Exception as e:
        logger.warning("SciPy peak detection error: %s", e)
        return np.array([], dtype=int)


def detect_peaks_slope_sum(signal: np.ndarray, fs: float) -> np.ndarray:
    """Detect beats with the slope-sum function (SSF), at the systolic peak.

    The rectified first derivative is summed over a 150 ms window (Zong et
    al., 2003); SSF maxima mark the systolic upstroke and are re-localised
    to the following systolic maximum so that the fiducial point matches
    the other detectors.

    Args:
        signal: Filtered 1-D signal.
        fs: Sampling frequency in Hz.

    Returns:
        Array of systolic-peak indices.
    """
    dx = np.diff(signal, prepend=signal[0])
    width = max(1, int(0.15 * fs))
    # Trailing (causal) window: the SSF maximum sits at the end of the upstroke.
    ssf = np.convolve(np.maximum(dx, 0), np.ones(width), "full")[: len(signal)]
    min_dist = _refractory_samples(fs, "PPG")
    peaks, _ = find_peaks(ssf, distance=min_dist, prominence=0.5 * np.std(ssf))
    return relocalize_peaks(signal, peaks, fs, before_s=0.05, after_s=0.15)


def _synthetic_template(fs: float) -> tuple[np.ndarray, int]:
    """Idealised PPG pulse (systolic + dicrotic) and its systolic index."""
    t = np.linspace(0, 1.0, int(fs * 1.0))
    systolic = np.exp(-t * 4.0) * np.sin(t * 2 * np.pi * 2)
    dicrotic = 0.3 * np.exp(-(t - 0.3) ** 2 / 0.01) * np.sin(t * 2 * np.pi * 8)
    template = systolic + dicrotic
    template = (template - template.mean()) / template.std()
    return template, int(np.argmax(template))


def _average_beat_template(
    signal: np.ndarray, fs: float, anchors: np.ndarray, clean_mask: np.ndarray | None,
) -> tuple[np.ndarray, int] | None:
    """Ensemble-average beat around clean anchor beats (systolic at known index)."""
    before = int(round(0.25 * fs))
    after = int(round(0.35 * fs))
    beats = []
    for p in np.asarray(anchors, dtype=int):
        if p - before < 0 or p + after >= len(signal):
            continue
        if clean_mask is not None and len(clean_mask) == len(signal) and not clean_mask[p]:
            continue
        beat = signal[p - before: p + after + 1]
        sd = beat.std()
        if sd > 0:
            beats.append((beat - beat.mean()) / sd)
    if len(beats) < 10:
        return None
    template = np.median(np.vstack(beats), axis=0)
    sd = template.std()
    if sd == 0:
        return None
    return (template - template.mean()) / sd, before


def template_matching(
    signal: np.ndarray,
    fs: float,
    clean_mask: np.ndarray | None,
    activity_type: str | None = None,
    template_threshold: int = 85,
    ibi_range: list | None = None,
    anchors: np.ndarray | None = None,
) -> np.ndarray:
    """Beat detection by normalised template correlation.

    The template is the median beat of the participant's own clean beats
    (``anchors`` inside ``clean_mask``) when at least ten are available,
    otherwise an idealised synthetic pulse.  Correlation maxima are shifted
    by the template's *systolic* index (not half its length, which offset
    beats by ~0.4 s) and re-localised to the systolic maximum.

    Args:
        signal: Filtered 1-D signal.
        fs: Sampling frequency in Hz.
        clean_mask: Boolean array of clean samples (may be ``None``).
        activity_type: Ignored (processing never uses activity labels).
        template_threshold: Percentile of |correlation| used as height.
        ibi_range: Unused (kept for API compatibility).
        anchors: Candidate beat indices used to build the template.

    Returns:
        Array of systolic-peak indices.
    """
    try:
        built = _average_beat_template(signal, fs, anchors, clean_mask) if anchors is not None else None
        template, systolic_index = built if built is not None else _synthetic_template(fs)
        sig_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)
        corr = correlate(sig_norm, template, mode="valid") / len(template)
        threshold = np.percentile(np.abs(corr), template_threshold)
        peaks, _ = find_peaks(corr, height=threshold, distance=_refractory_samples(fs, "PPG"))
        beats = peaks + systolic_index
        beats = beats[beats < len(signal)]
        return relocalize_peaks(signal, beats, fs, before_s=0.08, after_s=0.08)
    except Exception as e:
        logger.warning("Template matching error: %s", e)
        return np.array([], dtype=int)


def ensemble_peaks(
    signal: np.ndarray,
    fs: float,
    methods_results: dict,
    quality_metrics: dict | None = None,
    motion_metrics: dict | None = None,
    activity_type: str | None = None,
    tolerance_ms: float = 100.0,
    min_votes: int | None = None,
) -> np.ndarray:
    """Consensus beats: keep only beats confirmed by several detectors.

    Candidates from all detectors are clustered (members closer than
    ``tolerance_ms`` to the cluster start).  A cluster is retained only when
    it contains beats from at least ``min_votes`` *distinct* detectors
    (default ``max(2, ceil(k / 2))`` for ``k`` detectors that produced
    beats), so a false positive of a single detector can never enter the
    output.  The retained beat is the member of the most authoritative
    detector (``FIDUCIAL_ORDER``), keeping one fiducial definition.

    Args:
        signal: Filtered signal (unused; kept for API compatibility).
        fs: Sampling frequency in Hz.
        methods_results: Method name -> peak indices.
        quality_metrics: Unused.
        motion_metrics: Unused.
        activity_type: Unused.
        tolerance_ms: Clustering tolerance in milliseconds.
        min_votes: Required number of distinct agreeing detectors.

    Returns:
        Sorted consensus peak indices.
    """
    active = {m: np.asarray(p, dtype=int) for m, p in methods_results.items()
              if not m.endswith("_all") and m != "ensemble" and len(p) > 0}
    if not active:
        return np.array([], dtype=int)
    k = len(active)
    if min_votes is None:
        min_votes = max(2, int(np.ceil(k / 2)))
    if k < min_votes:
        # A single detector cannot form a consensus.
        return np.array([], dtype=int)

    members = sorted((int(p), m) for m, peaks in active.items() for p in peaks)
    tolerance = max(1, int(round(tolerance_ms / 1000.0 * fs)))
    rank = {m: (FIDUCIAL_ORDER.index(m) if m in FIDUCIAL_ORDER else len(FIDUCIAL_ORDER))
            for m in active}

    consensus: list[int] = []
    i = 0
    while i < len(members):
        j = i + 1
        while j < len(members) and members[j][0] - members[i][0] <= tolerance:
            j += 1
        cluster = members[i:j]
        if len({m for _, m in cluster}) >= min_votes:
            consensus.append(min(cluster, key=lambda pm: (rank[pm[1]], pm[0]))[0])
        i = j
    return np.unique(np.asarray(consensus, dtype=int))


def select_optimal_peaks(
    peak_results: dict,
    device: str,
    motion_class: str,
    quality_metrics: dict | None = None,
    activity_type: str | None = None,
    priority: dict | None = None,
) -> tuple[np.ndarray, str]:
    """Select one detector's output by device x motion priority.

    The priority table comes from ``detection.priority`` in the
    configuration (selected on development participants only) and falls
    back to ``DEFAULT_PRIORITY``.  Selection depends on the device and the
    accelerometer motion class only.

    Args:
        peak_results: Method name -> peak indices.
        device: Device identifier.
        motion_class: Motion level string.
        quality_metrics: Unused.
        activity_type: Unused.
        priority: Optional ``{device: {motion: [methods]}}`` table.

    Returns:
        Tuple of ``(peaks_array, method_name)``.
    """
    table = priority or DEFAULT_PRIORITY
    order = (table.get(device) or DEFAULT_PRIORITY.get(device, {})).get(
        motion_class, ["ensemble", "neurokit2", "scipy"]
    )
    min_peaks = 5 if motion_class in ("moderate", "severe") else 3
    for method in order:
        if method in peak_results and len(peak_results[method]) >= min_peaks:
            return np.asarray(peak_results[method], dtype=int), method
    for method, peaks in peak_results.items():
        if not method.endswith("_all") and len(peaks) >= 2:
            return np.asarray(peaks, dtype=int), method
    return np.array([], dtype=int), "none"


def refine_timing(
    timing_signal: np.ndarray,
    peaks: np.ndarray,
    fs: float,
    search_s: float = 0.12,
    fiducial: str = "peak",
    anchor: str = "systolic",
    upright: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-localise detected beats on a wide-band timing signal.

    Detection runs on a narrow-band signal (robust beat finding) whose
    smoothing distorts the pulse apex; each beat is therefore moved to the
    maximum of the minimally filtered *timing_signal* within ``+-search_s``,
    and the beat time is taken at the configured *fiducial* of that pulse
    (:func:`locate_fiducial`).  The output keeps one entry per input beat.

    Returns:
        Tuple ``(positions, integer_peaks)``: fractional sample positions and
        the integer maxima on *timing_signal*.
    """
    peaks = np.asarray(peaks, dtype=int)
    n = len(timing_signal)
    if len(peaks) == 0:
        return np.array([], dtype=float), peaks
    half = max(1, int(round(search_s * fs)))
    integer = np.empty(len(peaks), dtype=int)
    for i, p in enumerate(peaks):
        a, b = max(0, p - half), min(n, p + half + 1)
        integer[i] = a + int(np.argmax(timing_signal[a:b]))
    upright = timing_signal if upright is None else upright
    return locate_fiducial(upright, integer, fs, fiducial, anchor=anchor), integer


FIDUCIALS = ("peak", "onset", "max_slope", "tangent", "d2_max")


def locate_fiducial(
    signal: np.ndarray,
    peaks: np.ndarray,
    fs: float,
    kind: str = "peak",
    upsample: int = 10,
    onset_window: float = 0.45,
    anchor: str = "systolic",
) -> np.ndarray:
    """Locate a pulse fiducial for every detected beat.

    The signal must be oriented systolic-up.  With ``anchor="systolic"``
    each detected beat ``p`` is a systolic maximum and the upstroke is
    searched in ``[p - onset_window x IBI, p]``; with ``anchor="onset"``
    each beat is a pulse onset (minimum) and the systolic peak is searched in
    ``[p, p + onset_window x IBI]`` (IBI = smaller adjacent detected
    interval).  Fiducials:

    * ``peak``      - systolic maximum (parabolic interpolation);
    * ``onset``     - pulse onset: minimum preceding the upstroke (parabolic);
    * ``max_slope`` - maximum first derivative of the upstroke;
    * ``tangent``   - intersecting-tangents foot: tangent at the maximum slope
      intersected with the horizontal through the onset minimum (Chiu et
      al., 1991; Charlton et al., 2022);
    * ``d2_max``    - maximum second derivative before the maximum slope.

    Derivative-based points use a cubic spline evaluated at ``upsample``
    times the sampling rate.  One position per input peak is returned; a
    fiducial that cannot be located falls back to the systolic peak.

    Returns:
        Fractional sample positions.
    """
    from scipy.interpolate import CubicSpline

    peaks = np.asarray(peaks, dtype=int)
    n = len(signal)
    if kind not in FIDUCIALS:
        raise ValueError(f"Unknown fiducial {kind!r}")
    if anchor == "onset":
        onset_pos = refine_peaks_parabolic(-np.asarray(signal), peaks)
        if kind == "onset" or len(peaks) < 3:
            return onset_pos
    else:
        peak_pos = refine_peaks_parabolic(signal, peaks)
        if kind == "peak" or len(peaks) < 3:
            return peak_pos
    spline = CubicSpline(np.arange(n), signal)
    d1, d2 = spline.derivative(1), spline.derivative(2)
    ibi = np.diff(peaks)
    local = np.r_[ibi[0], np.minimum(ibi[:-1], ibi[1:]), ibi[-1]] if len(ibi) > 1 else np.full(len(peaks), ibi[0])
    out = (onset_pos if anchor == "onset" else peak_pos).copy()
    for i, q in enumerate(peaks):
        if anchor == "onset":
            b = min(n - 2, int(q + onset_window * local[i]))
            if b - q < 3 or q < 1:
                continue
            p = q + int(np.argmax(signal[q:b]))
            if kind == "peak":
                y0, y1, y2 = signal[p - 1], signal[p], signal[p + 1]
                den = y0 - 2 * y1 + y2
                out[i] = p + (float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if den != 0 else 0.0)
                continue
            a = q
        else:
            p = q
            a = max(1, int(p - onset_window * local[i]))
        if p - a < 3 or p >= n - 1:
            continue
        m = a + int(np.argmin(signal[a:p]))
        if 0 < m < n - 1:
            y0, y1, y2 = signal[m - 1], signal[m], signal[m + 1]
            den = y0 - 2 * y1 + y2
            onset = m + (float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if den != 0 else 0.0)
        else:
            onset = float(m)
        if kind == "onset":
            out[i] = onset
            continue
        if p - m < 2:
            continue
        grid = np.linspace(m, p, (p - m) * upsample + 1)
        slope = d1(grid)
        j = int(np.argmax(slope))
        if kind == "max_slope":
            out[i] = grid[j]
        elif kind == "tangent":
            if slope[j] > 0:
                out[i] = float(np.clip(grid[j] - (spline(grid[j]) - spline(onset)) / slope[j], m - 1, grid[j]))
        elif kind == "d2_max" and j > 0:
            out[i] = grid[: j + 1][int(np.argmax(d2(grid[: j + 1])))]
    return out


def refine_peaks_parabolic(signal: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Refine integer peak indices to sub-sample precision.

    Fits a parabola through each peak and its two neighbours and returns the
    vertex location.  Without this step, peak times are quantised to the
    sampling grid, which biases inter-beat intervals and inflates RMSSD: the
    quantisation floor is approximately 0.41/fs seconds per interval, i.e.
    ~16 ms at 25 Hz (Galaxy Watch 5) and ~6 ms at 64 Hz (Empatica E4).

    Args:
        signal: Filtered signal on which the peaks were detected.
        peaks: Integer peak sample indices.

    Returns:
        Float array of peak positions (in samples); each position is the
        integer index plus an offset in [-0.5, 0.5].
    """
    if len(peaks) == 0:
        return np.array([], dtype=float)

    positions = np.asarray(peaks, dtype=float).copy()
    interior = (peaks > 0) & (peaks < len(signal) - 1)
    if not np.any(interior):
        return positions

    indices = peaks[interior]
    y_prev = signal[indices - 1]
    y_peak = signal[indices]
    y_next = signal[indices + 1]

    curvature = y_prev - 2.0 * y_peak + y_next
    offsets = np.zeros(len(indices), dtype=float)
    valid = curvature != 0
    offsets[valid] = 0.5 * (y_prev[valid] - y_next[valid]) / curvature[valid]
    positions[interior] = indices + np.clip(offsets, -0.5, 0.5)

    return positions


def harmonic_sum_hr(
    signal: np.ndarray,
    fs: float,
    f_min: float = 0.6,
    f_max: float = 3.2,
    n_harmonics: int = 4,
    whiten: bool = True,
) -> float:
    """Estimate heart rate by harmonic summation of the power spectrum.

    The whitened power spectral density is summed at integer multiples of each
    candidate frequency, with weights 1/h.  Unlike a plain product (HPS), this
    does not collapse when the higher harmonics are absent, and whitening
    prevents the 1/f slope from dominating the choice.  The method reinforces
    the cardiac fundamental even when it is weaker than its second harmonic or
    partly masked by narrow-band motion artifacts.

    Args:
        signal: Filtered signal.
        fs: Sampling frequency in Hz.
        f_min: Lowest candidate frequency in Hz (0.6 Hz = 36 bpm).
        f_max: Highest candidate frequency in Hz (3.2 Hz = 192 bpm).
        n_harmonics: Number of harmonics summed.
        whiten: Divide the PSD by its local median before summation.

    Returns:
        Estimated heart rate in bpm, or ``NaN`` when no estimate is possible.
    """
    if len(signal) < int(4 * fs):
        return np.nan

    nperseg = int(min(len(signal), max(64, 12 * fs)))
    frequencies, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window="hann")
    if not np.any(psd > 0) or len(frequencies) < 8:
        return np.nan

    resolution = float(frequencies[1] - frequencies[0])
    if whiten:
        window_length = max(3, int(round(0.4 / resolution)) | 1)
        baseline = median_filter(psd, size=window_length, mode="nearest")
        psd = psd / (baseline + 1e-12)

    lowest = max(2, int(round(f_min / resolution)))
    highest = min(len(frequencies) - 1, int(round(f_max / resolution)))
    if highest <= lowest:
        return np.nan

    weights = 1.0 / np.arange(1, n_harmonics + 1)
    scores = np.zeros(highest - lowest + 1)
    for harmonic in range(1, n_harmonics + 1):
        indices = np.arange(lowest, highest + 1) * harmonic
        valid = indices < len(psd)
        contribution = np.zeros_like(scores)
        contribution[valid] = psd[indices[valid]]
        scores += weights[harmonic - 1] * contribution

    if not np.any(scores > 0):
        return np.nan
    peak = int(np.argmax(scores)) + lowest
    return float(frequencies[peak] * 60.0)


def estimate_rhythm_interval(
    signal: np.ndarray,
    fs: float,
    device_hr: float = np.nan,
    prior: str = "auto",
    plausible_range: tuple[float, float] = (35.0, 200.0),
) -> float:
    """Estimate the expected cardiac interval used for rhythm tracking.

    Args:
        signal: Filtered signal.
        fs: Sampling frequency in Hz.
        device_hr: Device-reported heart rate in bpm (may be ``NaN``).
        prior: ``"device"``, ``"hps"`` or ``"auto"`` (device when available and
            plausible, otherwise the harmonic product spectrum).
        plausible_range: Accepted HR range in bpm.

    Returns:
        Expected interval in milliseconds, or ``NaN`` when unavailable.
    """
    spectral_hr = harmonic_sum_hr(signal, fs)

    device_valid = np.isfinite(device_hr) and plausible_range[0] <= device_hr <= plausible_range[1]
    spectral_valid = np.isfinite(spectral_hr) and plausible_range[0] <= spectral_hr <= plausible_range[1]

    if prior == "device":
        estimate = device_hr if device_valid else np.nan
    elif prior == "hps":
        estimate = spectral_hr if spectral_valid else np.nan
    else:  # auto: prefer the device estimate, fall back to the spectrum
        estimate = device_hr if device_valid else (spectral_hr if spectral_valid else np.nan)

    return float(60000.0 / estimate) if np.isfinite(estimate) else np.nan


def select_peaks_with_rhythm(
    signal: np.ndarray,
    candidates: np.ndarray,
    fs: float,
    expected_interval_ms: float,
    tolerance: float = 0.30,
    penalty_weight: float = 1.0,
    max_interval_factor: float = 2.5,
    amplitude_weight: float = 1.0,
) -> np.ndarray:
    """Select a peak sequence that follows an expected cardiac rhythm.

    Candidate peaks are scored by amplitude and by how closely their spacing
    matches integer multiples of *expected_interval_ms*.  A dynamic-programming
    pass finds the highest-scoring chain, which structurally resolves both
    missed beats (spacing ≈ 2 × interval) and spurious detections (spacing far
    from any multiple).  The chain is then extended forwards and backwards so
    the full recording is covered.

    Args:
        signal: Filtered signal (peak amplitudes are read from it).
        candidates: Candidate peak indices (union of detectors).
        fs: Sampling frequency in Hz.
        expected_interval_ms: Expected cardiac interval in milliseconds.
        tolerance: Maximum relative deviation from the nearest multiple of the
            expected interval for an interval to be accepted.
        penalty_weight: Weight of the rhythm-deviation penalty.
        max_interval_factor: Longest accepted interval as a multiple of the
            expected interval.
        amplitude_weight: Weight of the amplitude term relative to the uniform
            coverage term (coverage dominates so that long rhythm-consistent
            chains are preferred over isolated tall peaks).

    Returns:
        Selected peak indices, sorted ascending.
    """
    candidates = np.unique(np.asarray(candidates, dtype=int))
    if len(candidates) < 3 or not np.isfinite(expected_interval_ms) or expected_interval_ms <= 0:
        return candidates

    interval = float(expected_interval_ms)
    times_ms = candidates / fs * 1000.0
    amplitudes = signal[candidates].astype(float)
    span = amplitudes.max() - amplitudes.min()
    normalised = (amplitudes - amplitudes.min()) / span if span > 0 else np.ones(len(candidates))
    # Coverage dominates so that a long, rhythm-consistent chain always beats a
    # short chain of tall but off-rhythm peaks.
    scores = 1.0 + amplitude_weight * normalised

    def _penalty(delta_ms: float) -> float:
        multiples = max(1, int(round(delta_ms / interval)))
        return abs(delta_ms - multiples * interval) / interval

    n = len(candidates)
    best = scores.copy()
    previous = np.full(n, -1)
    for i in range(n):
        for j in range(i):
            delta = times_ms[i] - times_ms[j]
            if delta < 0.4 * interval or delta > (max_interval_factor + 0.5) * interval:
                continue
            multiples = max(1, int(round(delta / interval)))
            deviation = abs(delta - multiples * interval) / interval
            if deviation > tolerance:
                continue
            candidate_score = best[j] + scores[i] - penalty_weight * deviation
            if candidate_score > best[i]:
                best[i] = candidate_score
                previous[i] = j

    end = int(np.argmax(best))
    chain: list[int] = []
    while end != -1:
        chain.append(end)
        end = previous[end]
    chain.sort()

    selected = set(chain)

    # Extend forwards: continue the rhythm after the last selected peak
    last_index = chain[-1]
    index = last_index + 1
    while index < n:
        delta = times_ms[index] - times_ms[last_index]
        multiples = max(1, int(round(delta / interval)))
        deviation = abs(delta - multiples * interval) / interval
        if deviation <= tolerance:
            selected.add(index)
            last_index = index
        index += 1

    # Extend backwards before the first selected peak
    first_index = chain[0]
    index = first_index - 1
    while index >= 0:
        delta = times_ms[first_index] - times_ms[index]
        multiples = max(1, int(round(delta / interval)))
        deviation = abs(delta - multiples * interval) / interval
        if deviation <= tolerance:
            selected.add(index)
            first_index = index
        index -= 1

    return candidates[sorted(selected)]


def spectral_rhythm_peaks(
    detection_results: dict,
    signal: np.ndarray,
    fs: float,
    device_hr: float = np.nan,
    tracking: dict | None = None,
    prior: str = "hps",
) -> np.ndarray:
    """Select a beat chain using a standalone spectral rhythm prior.

    Unlike the device-assisted tracker, the expected cardiac interval here is
    estimated from the harmonic content of the (motion-cleaned) PPG itself, so
    the method is independent of the device-reported HR and therefore
    non-circular.  The candidate union from the individual detectors is
    reduced to the subset that follows the spectral rhythm, which recovers the
    systematically missed-beat (half-rate) chains observed under locomotion.

    Args:
        detection_results: Method name → peak indices from the standard
            detectors (``ensemble`` and diagnostic ``*_all`` entries ignored).
        signal: The motion-cleaned, filtered signal used for the prior.
        fs: Sampling frequency in Hz.
        device_hr: Device-reported HR (bpm); unused by the ``hps`` prior but
            kept for API symmetry with ``estimate_rhythm_interval``.
        tracking: Rhythm-tracking parameters (tolerance, penalty_weight, ...).
        prior: Interval prior passed to ``estimate_rhythm_interval``.

    Returns:
        Selected peak indices, or an empty array when the prior or the
        candidate set is insufficient.
    """
    candidates = (
        np.unique(np.concatenate([
            np.asarray(peaks)
            for method, peaks in detection_results.items()
            if not method.endswith("_all") and method != "ensemble" and len(peaks) > 0
        ]))
        if detection_results else np.array([], dtype=int)
    )
    if len(candidates) < 5:
        return np.array([], dtype=int)

    interval = estimate_rhythm_interval(signal, fs, device_hr, prior)
    if not np.isfinite(interval) or interval <= 0:
        return np.array([], dtype=int)

    tracking = tracking or {}
    selected = select_peaks_with_rhythm(
        signal, candidates, fs, interval,
        tolerance=float(tracking.get("tolerance", 0.30)),
        penalty_weight=float(tracking.get("penalty_weight", 1.0)),
        max_interval_factor=float(tracking.get("max_interval_factor", 2.5)),
        amplitude_weight=float(tracking.get("amplitude_weight", 1.0)),
    )
    return selected if len(selected) >= 5 else np.array([], dtype=int)


def complete_rhythm_intervals(
    positions: np.ndarray,
    fs: float,
    expected_interval_ms: float,
    max_multiple: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert synthetic beats where the rhythm indicates missed ones.

    For every interval that is approximately an integer multiple ``k >= 2`` of
    the expected cardiac interval, ``k - 1`` evenly spaced beats are inserted.
    This removes the half-rate ambiguity of missed-beat patterns, at the cost
    of model-derived beats that are flagged so that downstream quality control
    can account for them.

    Args:
        positions: Peak positions in samples, ascending.
        fs: Sampling frequency in Hz.
        expected_interval_ms: Expected cardiac interval in milliseconds.
        max_multiple: Largest interval multiple that is repaired.

    Returns:
        Tuple of ``(completed_positions, synthetic_mask)``; the mask marks the
        inserted beats.
    """
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 2 or not np.isfinite(expected_interval_ms) or expected_interval_ms <= 0:
        return positions, np.zeros(len(positions), dtype=bool)

    expected_samples = expected_interval_ms / 1000.0 * fs
    completed: list[float] = [float(positions[0])]
    synthetic: list[bool] = [False]

    for position in positions[1:]:
        previous = completed[-1]
        delta = position - previous
        multiple = max(1, int(round(delta / expected_samples)))
        if 1 < multiple <= max_multiple:
            step = delta / multiple
            for _ in range(1, multiple):
                completed.append(completed[-1] + step)
                synthetic.append(True)
        completed.append(float(position))
        synthetic.append(False)

    return np.array(completed), np.array(synthetic, dtype=bool)


def multi_method_peak_detection(
    signal: np.ndarray,
    device: str,
    motion_class: str,
    clean_mask: np.ndarray,
    quality_metrics: dict,
    activity_type: str | None = None,
    config: dict | None = None,
    neurokit2_signal: np.ndarray | None = None,
) -> dict:
    """Run all applicable detectors and the consensus ensemble.

    * NeuroKit2 and SciPy always run.
    * PPG under severe motion additionally runs the slope-sum and template
      detectors, and beats falling in severe-motion samples are removed
      from NeuroKit2/SciPy (the unfiltered outputs are kept as ``*_all``).
    * The ensemble is a true consensus (see :func:`ensemble_peaks`).

    Args:
        signal: Filtered 1-D signal.
        device: Device identifier.
        motion_class: Motion level string.
        clean_mask: Boolean array of clean samples.
        quality_metrics: Signal quality dict.
        activity_type: Ignored (no activity-label branching).
        config: Parsed configuration dictionary.
        neurokit2_signal: Optional alternative input for NeuroKit2 only.

    Returns:
        Dict mapping method name -> peak indices.
    """
    if config is None:
        raise TypeError("multi_method_peak_detection requires a config")
    fs = config["devices"][device]["fs"]
    results: dict[str, np.ndarray] = {}
    severe_ppg = device != "PolarH10" and motion_class == "severe"
    has_mask = clean_mask is not None and len(clean_mask) == len(signal)

    def _masked(peaks: np.ndarray) -> np.ndarray:
        peaks = np.asarray(peaks, dtype=int)
        if not has_mask or len(peaks) == 0:
            return peaks
        peaks = peaks[peaks < len(clean_mask)]
        return peaks[clean_mask[peaks]]

    peaks_nk = detect_peaks_neurokit2(
        neurokit2_signal if neurokit2_signal is not None else signal, fs, device
    )
    if device != "PolarH10" and neurokit2_signal is not None and len(peaks_nk):
        peaks_nk = relocalize_peaks(signal, peaks_nk, fs, before_s=0.1, after_s=0.1)
    peaks_scipy = detect_peaks_scipy(signal, fs, device, motion_class, quality_metrics, config)

    if severe_ppg:
        results["neurokit2"], results["neurokit2_all"] = _masked(peaks_nk), peaks_nk
        results["scipy"], results["scipy_all"] = _masked(peaks_scipy), peaks_scipy
        try:
            results["slope"] = _masked(detect_peaks_slope_sum(signal, fs))
        except Exception:
            results["slope"] = np.array([], dtype=int)
        anchors = np.concatenate([results["neurokit2"], results["scipy"]])
        results["template"] = _masked(template_matching(
            signal, fs, clean_mask, template_threshold=90, anchors=anchors,
        ))
    else:
        results["neurokit2"] = peaks_nk
        results["scipy"] = peaks_scipy

    detection_cfg = config.get("detection", {})
    try:
        results["ensemble"] = ensemble_peaks(
            signal, fs, results,
            tolerance_ms=float(detection_cfg.get("ensemble_tolerance_ms", 100.0)),
            min_votes=detection_cfg.get("ensemble_min_votes"),
        )
    except Exception as e:
        logger.warning("Ensemble method error: %s", e)
        results["ensemble"] = np.array([], dtype=int)

    return results
