"""Multi-method systolic peak detection with ensemble consensus.

Implements four peak-detection strategies and an ensemble voting method:

1. **NeuroKit2** — Elgendi's two-stage moving-average algorithm (PPG) or
   NeuroKit's ECG R-peak detector (ECG).
2. **SciPy** — Adaptive-percentile thresholding with ``find_peaks``.
3. **Slope-sum (SSF)** — Derivative-rectified moving sum, used only for
   severe-motion PPG.
4. **Template matching** — Cross-correlation with a synthetic or
   clean-segment-derived waveform template, used only for severe motion.

The **ensemble** method clusters nearby peaks (within 60 ms) and selects
the one with the highest weighted vote across methods.  Method weights
adapt to signal quality, motion level, and activity type.

Finally, ``select_optimal_peaks`` picks the best method's output based on
device- and context-specific priority ordering.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from scipy.signal import correlate, find_peaks

logger = logging.getLogger(__name__)


def detect_peaks_neurokit2(
    signal: np.ndarray,
    fs: float,
    device: str,
    motion_class: str,
) -> np.ndarray:
    """Detect peaks using NeuroKit2's Elgendi (PPG) or NeuroKit (ECG) method.

    For PPG devices, artifact correction is enabled only under non-low
    motion.  For ECG, the standard NeuroKit R-peak detector is used.

    Args:
        signal: Filtered 1-D signal.
        fs: Sampling frequency in Hz.
        device: Device identifier.
        motion_class: ``"low"``, ``"moderate"``, or ``"severe"``.

    Returns:
        Array of peak indices.
    """
    import neurokit2 as nk

    try:
        if device == "GalaxyWatch":
            _, info = nk.ppg_peaks(
                signal, sampling_rate=fs, method="elgendi",
                correct_artifacts=(motion_class != "low"),
            )
            return info.get("PPG_Peaks", np.array([]))
        elif device == "PolarH10":
            _, info = nk.ecg_peaks(signal, sampling_rate=fs, method="neurokit")
            return info.get("ECG_R_Peaks", np.array([]))
        else:  # E4
            _, info = nk.ppg_peaks(signal, sampling_rate=fs, method="elgendi")
            return info.get("PPG_Peaks", np.array([]))
    except Exception as e:
        logger.warning("NeuroKit2 peak detection error: %s", e)
        return np.array([])


def detect_peaks_scipy(
    signal: np.ndarray,
    fs: float,
    device: str,
    motion_class: str,
    quality_metrics: dict,
    config: dict,
) -> np.ndarray:
    """Detect peaks using SciPy ``find_peaks`` with adaptive percentile thresholds.

    A rolling window computes local amplitude thresholds.  Minimum inter-peak
    distance and prominence are adjusted for device type, motion level, and
    signal quality (SNR).

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
        # Window size for local threshold computation
        win = int(8 * fs) if motion_class == "low" else int(5 * fs)

        # Percentile threshold adapts to device and motion
        if device == "PolarH10":
            percentile = 85
        elif motion_class == "low":
            percentile = 75
        elif motion_class == "moderate":
            percentile = 80
        else:
            percentile = 85

        # Build rolling threshold array
        threshold = np.zeros_like(signal)
        for i in range(0, len(signal), win // 2):
            start = max(0, i - win // 2)
            end = min(len(signal), i + win // 2)
            seg = signal[start:end]
            if len(seg) > 0:
                threshold[start:end] = np.percentile(seg, percentile)

        # Minimum inter-peak distance (max HR)
        if device == "PolarH10":
            min_dist = max(1, int(60 / 180 * fs))  # max 180 bpm
            prominence = np.std(signal) * 0.4
        else:
            min_dist = max(1, int(60 / 160 * fs))  # max 160 bpm
            prominence = np.std(signal) * 0.15

        # Increase prominence for low quality or motion
        if quality_metrics["snr_db"] < snr_threshold:
            prominence *= 1.5
        if motion_class in ("moderate", "severe"):
            prominence *= 1.3

        peaks, _ = find_peaks(signal, height=threshold, distance=min_dist, prominence=prominence)
        return peaks

    except Exception as e:
        logger.warning("SciPy peak detection error: %s", e)
        return np.array([])


def detect_peaks_slope_sum(signal: np.ndarray, fs: float) -> np.ndarray:
    """Detect peaks using the slope-sum function (SSF).

    Computes the first derivative, rectifies it (positive slopes only),
    and applies a 150 ms moving-sum convolution.  Peaks in the SSF
    correspond to systolic upstrokes.

    Used only for PPG under severe motion, where standard methods
    over-detect due to motion artifacts.

    Args:
        signal: Filtered 1-D signal.
        fs: Sampling frequency in Hz.

    Returns:
        Array of peak indices.
    """
    dx = np.diff(signal, prepend=signal[0])
    ssf = np.convolve(np.maximum(dx, 0), np.ones(int(0.15 * fs)), "same")

    min_dist = int(fs * 0.5)  # ~120 bpm max
    prominence = 0.5 * np.std(ssf)
    peaks, _ = find_peaks(ssf, distance=min_dist, prominence=prominence)
    return peaks


def _synthetic_template_matching(
    signal: np.ndarray,
    fs: float,
    activity_type: str,
    template_threshold: int,
    ibi_range: list,
) -> np.ndarray:
    """Template matching using a synthetic PPG waveform.

    Constructs an idealised PPG waveform (systolic + dicrotic component)
    and cross-correlates it with the normalised signal.  Correlation peaks
    above the given percentile are retained as beat locations.

    Args:
        signal: Filtered 1-D signal.
        fs: Sampling frequency in Hz.
        activity_type: Activity name (affects template duration and decay).
        template_threshold: Percentile for correlation threshold (e.g. 85).
        ibi_range: [min_ms, max_ms] for IBI validation.

    Returns:
        Array of peak indices.
    """
    # Activity-specific template shape
    if activity_type == "baseline":
        template_duration = 1.2
        decay = 3.0
    else:
        template_duration = 1.0
        decay = 4.0

    t = np.linspace(0, template_duration, int(fs * template_duration))
    systolic = np.exp(-t * decay) * np.sin(t * 2 * np.pi * 2)
    dicrotic = 0.3 * np.exp(-(t - 0.3) ** 2 / 0.01) * np.sin(t * 2 * np.pi * 8)
    template = systolic + dicrotic
    template = (template - np.mean(template)) / np.std(template)

    try:
        sig_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
        corr = correlate(sig_norm, template, mode="valid")
        corr = corr / (np.max(np.abs(corr)) + 1e-6)

        threshold = np.percentile(np.abs(corr), template_threshold)
        min_dist = int(60 / 150 * fs)  # max 150 bpm

        peaks, _ = find_peaks(corr, height=threshold, distance=min_dist, prominence=threshold * 0.3)

        # Adjust for template offset and validate IBI
        offset = len(template) // 2
        adjusted = peaks + offset
        valid = []
        for peak in adjusted:
            if offset <= peak < len(signal) - offset:
                if not valid:
                    valid.append(peak)
                else:
                    ibi_ms = (peak - valid[-1]) / fs * 1000
                    if ibi_range[0] <= ibi_ms <= ibi_range[1]:
                        valid.append(peak)
        return np.array(valid)

    except Exception as e:
        logger.warning("Synthetic template matching error: %s", e)
        return np.array([])


def _template_from_clean_segments(
    signal: np.ndarray,
    fs: float,
    clean_indices: np.ndarray,
    template_threshold: int,
    ibi_range: list,
) -> np.ndarray:
    """Template matching using a template extracted from clean signal segments.

    Finds the longest contiguous clean segment, uses it as the template,
    and cross-correlates with the full signal.

    Args:
        signal: Filtered 1-D signal.
        fs: Sampling frequency in Hz.
        clean_indices: Indices of clean (non-motion-contaminated) samples.
        template_threshold: Percentile for correlation threshold.
        ibi_range: [min_ms, max_ms] for IBI validation.

    Returns:
        Array of peak indices.
    """
    # Find longest contiguous clean segment
    segments = []
    current = [clean_indices[0]]
    for i in range(1, len(clean_indices)):
        if clean_indices[i] - clean_indices[i - 1] <= 2:
            current.append(clean_indices[i])
        else:
            segments.append(current)
            current = [clean_indices[i]]
    segments.append(current)

    longest = max(segments, key=len)
    start = longest[0]
    length = min(int(1.5 * fs), len(longest))
    template = signal[start : start + length]
    template = (template - np.mean(template)) / (np.std(template) + 1e-6)

    sig_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
    corr = correlate(sig_norm, template, mode="valid")
    corr = corr / (np.max(np.abs(corr)) + 1e-6)

    threshold = np.percentile(np.abs(corr), template_threshold)
    min_dist = int(60 / 150 * fs)

    peaks, _ = find_peaks(corr, height=threshold, distance=min_dist)

    offset = len(template) // 2
    adjusted = peaks + offset
    valid = []
    for peak in adjusted:
        if offset <= peak < len(signal) - offset:
            if not valid:
                valid.append(peak)
            else:
                ibi_ms = (peak - valid[-1]) / fs * 1000
                if ibi_range[0] <= ibi_ms <= ibi_range[1]:
                    valid.append(peak)
    return np.array(valid)


def template_matching(
    signal: np.ndarray,
    fs: float,
    clean_mask: np.ndarray,
    activity_type: str,
    template_threshold: int,
    ibi_range: list,
) -> np.ndarray:
    """Activity-aware template matching: clean-segment or synthetic fallback.

    If sufficient clean samples are available (≥ 1.5 s), a template is
    extracted from the cleanest segment.  Otherwise, a synthetic PPG
    waveform is used.

    Args:
        signal: Filtered 1-D signal.
        fs: Sampling frequency in Hz.
        clean_mask: Boolean array marking clean samples.
        activity_type: Activity name.
        template_threshold: Percentile for correlation threshold.
        ibi_range: [min_ms, max_ms] for IBI validation.

    Returns:
        Array of peak indices.
    """
    try:
        if clean_mask is not None and np.any(clean_mask):
            clean_idx = np.where(clean_mask)[0]
            if len(clean_idx) >= int(1.5 * fs):
                return _template_from_clean_segments(signal, fs, clean_idx, template_threshold, ibi_range)
        return _synthetic_template_matching(signal, fs, activity_type, template_threshold, ibi_range)
    except Exception as e:
        logger.warning("Template matching error: %s", e)
        return np.array([])


def ensemble_peaks(
    signal: np.ndarray,
    fs: float,
    methods_results: dict,
    quality_metrics: dict,
    motion_metrics: dict,
    activity_type: str,
) -> np.ndarray:
    """Combine peaks from multiple methods using weighted voting.

    Each method's peaks receive a weight based on:

    - Method identity (NeuroKit2 weighted highest for baseline, ensemble
      favoured for motion)
    - Signal quality (SNR > 20 dB → weight boost)
    - Clean ratio (< 0.3 → weight penalty)

    Nearby peaks (within 60 ms) are clustered, and the peak with the
    highest cumulative vote wins.  Ties are broken by local amplitude.

    Args:
        signal: Filtered 1-D signal (for amplitude tie-breaking).
        fs: Sampling frequency in Hz.
        methods_results: Dict mapping method name → peak indices array.
        quality_metrics: Signal quality dict (with ``snr_db``).
        motion_metrics: Motion metrics dict (with ``motion_level``, ``clean_ratio``).
        activity_type: Activity name.

    Returns:
        Array of ensemble-consensus peak indices.
    """
    if not methods_results:
        return np.array([])

    # Compute per-method weights
    weights: Dict[str, float] = {}
    motion_level = motion_metrics.get("motion_level", "low")
    snr = quality_metrics.get("snr_db", 0)
    clean_ratio = motion_metrics.get("clean_ratio", 1.0)

    for method, peaks in methods_results.items():
        if len(peaks) == 0:
            weights[method] = 0
            continue

        w = 1.0
        if method == "neurokit2":
            w = 1.4 if activity_type == "baseline" else 1.3
            if snr < 15:
                w *= 0.7
        elif method == "scipy":
            w = 1.2 if activity_type == "baseline" else 1.1
            if motion_level in ("moderate", "severe"):
                w *= 0.9
        elif method == "template":
            w = 0.8 if activity_type == "baseline" else 0.7
            if motion_level == "severe":
                w *= 1.2
        elif method == "slope":
            w = 1.1 if motion_level in ("moderate", "severe") else 0.8

        if snr > 20:
            w *= 1.1
        if clean_ratio < 0.3:
            w *= 0.7

        weights[method] = w

    # Accumulate votes per peak
    peak_votes: Dict[int, float] = {}
    for method, peaks in methods_results.items():
        w = weights.get(method, 0)
        if w > 0:
            for p in peaks:
                peak_votes[p] = peak_votes.get(p, 0) + w

    if not peak_votes:
        return np.array([])

    # Cluster nearby peaks (60 ms tolerance) and select highest-voted
    tolerance = int(0.06 * fs)
    sorted_peaks = sorted(peak_votes.keys())
    clustered: list[int] = []

    i = 0
    while i < len(sorted_peaks):
        cluster = [sorted_peaks[i]]
        votes = [peak_votes[sorted_peaks[i]]]

        j = i + 1
        while j < len(sorted_peaks) and sorted_peaks[j] - sorted_peaks[i] <= tolerance:
            cluster.append(sorted_peaks[j])
            votes.append(peak_votes[sorted_peaks[j]])
            j += 1

        # Select best peak in cluster
        best_idx = 0
        best_score = votes[0]
        for k in range(1, len(cluster)):
            if votes[k] > best_score or (votes[k] == best_score and signal[cluster[k]] > signal[cluster[best_idx]]):
                best_score = votes[k]
                best_idx = k
        clustered.append(cluster[best_idx])

        i = j

    return np.array(clustered)


def select_optimal_peaks(
    peak_results: dict,
    device: str,
    motion_class: str,
    quality_metrics: dict,
    activity_type: str,
) -> Tuple[np.ndarray, str]:
    """Select the best peak set based on device- and context-specific priority.

    Priority ordering considers:

    - **Baseline**: NeuroKit2 first for E4/PolarH10, ensemble for GalaxyWatch.
    - **Motion**: Ensemble first (consensus is more robust), then individual methods.
    - **Severe motion (GalaxyWatch)**: Slope and template methods get priority
      over NeuroKit2/SciPy which tend to over-detect under heavy artifacts.

    A minimum peak count (5 for motion, 3 for low motion) must be met.
    Falls back to any method with ≥ 2 peaks.

    Args:
        peak_results: Dict mapping method name → peak indices.
        device: Device identifier.
        motion_class: Motion level string.
        quality_metrics: Signal quality dict.
        activity_type: Activity name.

    Returns:
        Tuple of ``(peaks_array, method_name)``.
    """
    if activity_type == "baseline":
        if device == "PolarH10":
            priority = ["neurokit2", "scipy", "ensemble"]
        elif device == "E4":
            priority = ["neurokit2", "ensemble", "scipy"]
        else:
            priority = ["ensemble", "neurokit2", "scipy"]
    else:
        if device == "PolarH10":
            priority = ["neurokit2", "scipy", "ensemble"]
        elif device == "E4":
            priority = ["ensemble", "neurokit2", "scipy", "slope"]
        else:  # GalaxyWatch
            if motion_class == "severe":
                priority = ["ensemble", "slope", "template", "neurokit2", "scipy"]
            elif motion_class == "moderate":
                priority = ["ensemble", "slope", "neurokit2", "scipy", "template"]
            else:
                priority = ["ensemble", "neurokit2", "scipy"]

    min_peaks = 5 if motion_class in ("moderate", "severe") else 3

    for method in priority:
        if method in peak_results and len(peak_results[method]) >= min_peaks:
            return peak_results[method], method

    # Fallback: any method with at least 2 peaks
    for method, peaks in peak_results.items():
        if len(peaks) >= 2:
            return peaks, method

    return np.array([]), "none"


def multi_method_peak_detection(
    signal: np.ndarray,
    device: str,
    motion_class: str,
    clean_mask: np.ndarray,
    quality_metrics: dict,
    activity_type: str,
    config: dict,
) -> dict:
    """Run all applicable peak detection methods and return their results.

    Orchestrates the individual detectors, applies motion-mask filtering
    for severe motion, and runs the ensemble consensus.

    Args:
        signal: Filtered 1-D signal.
        device: Device identifier.
        motion_class: Motion level string.
        clean_mask: Boolean array of clean samples.
        quality_metrics: Signal quality dict.
        activity_type: Activity name.
        config: Parsed configuration dictionary.

    Returns:
        Dict mapping method name → peak indices array.  Always includes
        ``"ensemble"`` if at least one method produced peaks.
    """
    dev_cfg = config["devices"][device]
    fs = dev_cfg["fs"]
    ibi_range = config["processing"]["ibi_range"]
    results: Dict[str, np.ndarray] = {}

    # Determine conservative parameters for motion
    if motion_class in ("moderate", "severe"):
        template_threshold = 90
    else:
        template_threshold = 85

    # Method 1: NeuroKit2
    peaks_nk = detect_peaks_neurokit2(signal, fs, device, motion_class)
    if motion_class == "severe" and len(peaks_nk) > 0 and len(clean_mask) > 0:
        valid = peaks_nk[peaks_nk < len(clean_mask)]
        results["neurokit2"] = valid[clean_mask[valid]]
        results["neurokit2_all"] = peaks_nk
    else:
        results["neurokit2"] = peaks_nk

    # Method 2: SciPy adaptive percentile
    peaks_scipy = detect_peaks_scipy(signal, fs, device, motion_class, quality_metrics, config)
    if motion_class == "severe" and len(peaks_scipy) > 0 and len(clean_mask) > 0:
        valid = peaks_scipy[peaks_scipy < len(clean_mask)]
        results["scipy"] = valid[clean_mask[valid]]
        results["scipy_all"] = peaks_scipy
    else:
        results["scipy"] = peaks_scipy

    # Method 3: Slope-sum (PPG only, severe motion only)
    if device != "PolarH10" and motion_class == "severe":
        try:
            ssf_peaks = detect_peaks_slope_sum(signal, fs)
            if len(clean_mask) > 0:
                valid = ssf_peaks[ssf_peaks < len(clean_mask)]
                ssf_peaks = valid[clean_mask[valid]]
            results["slope"] = ssf_peaks
        except Exception:
            results["slope"] = np.array([])

    # Method 4: Template matching (severe motion only)
    if motion_class == "severe":
        results["template"] = template_matching(
            signal, fs, clean_mask, activity_type, template_threshold, ibi_range
        )

    # Method 5: Ensemble consensus
    motion_metrics = {
        "motion_level": motion_class,
        "clean_ratio": float(np.mean(clean_mask)) if len(clean_mask) > 0 else 1.0,
    }
    try:
        results["ensemble"] = ensemble_peaks(
            signal, fs, results, quality_metrics, motion_metrics, activity_type
        )
    except Exception as e:
        logger.warning("Ensemble method error: %s", e)
        results["ensemble"] = np.array([])

    return results
