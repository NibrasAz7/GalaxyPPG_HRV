"""Beat-level quality control: template-matching SQI and artifact correction.

Two standalone stages replace the previous rolling-median rule and the
device-HR consistency gate (which conditioned acceptance on agreement with
the vendor's own algorithm and was therefore circular):

1. **Template-matching signal quality index** (Orphanidou et al., 2015;
   Li & Clifford, 2012).  Every detected beat is correlated with the
   median beat of the recording; beats whose correlation falls below the
   threshold are removed, so that a corrupted pulse never contributes an
   interval.

2. **Lipponen-Tarvainen artifact classification and correction**
   (Lipponen & Tarvainen, 2019 - the algorithm implemented in Kubios HRV).
   Successive-difference (dRR) and median-deviation (mRR) series are scaled
   by time-varying thresholds (alpha x quartile deviation over 91 beats) and
   every beat is classified as normal, ectopic, missed, extra or long/short.
   Corrections operate on *beat times*, so they conserve time: a missed beat
   is split by inserting a beat, an extra beat is merged by deleting it, and
   ectopic / long-short beats are moved to the midpoint of their
   neighbours.  Unlike one-for-one interval replacement, the beat count and
   the cumulative time axis (used for spectral analysis) stay correct.

Every corrected interval is flagged and counted in the correction burden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LT_ALPHA = 5.2
LT_C1 = 0.13
LT_C2 = 0.17
LT_WINDOW = 91
LT_MEDIAN_WINDOW = 11


# ---------------------------------------------------------------------------
# Signal quality index
# ---------------------------------------------------------------------------

def beat_template_sqi(
    signal: np.ndarray,
    peaks: np.ndarray,
    fs: float,
    before_fraction: float = 0.4,
    after_fraction: float = 0.6,
) -> np.ndarray:
    """Per-beat template-matching SQI (Pearson r with the median beat).

    Each beat is the window ``[p - before*m, p + after*m]`` around its peak,
    where ``m`` is the median inter-beat interval in samples, so the window
    spans one cardiac cycle at any heart rate.

    Args:
        signal: Filtered signal.
        peaks: Integer beat indices.
        fs: Sampling frequency in Hz.
        before_fraction: Window start before the peak, in median IBIs.
        after_fraction: Window end after the peak, in median IBIs.

    Returns:
        Correlation per beat (NaN where the window does not fit).
    """
    peaks = np.asarray(peaks, dtype=int)
    sqi = np.full(len(peaks), np.nan)
    if len(peaks) < 5:
        return sqi
    m = float(np.median(np.diff(peaks)))
    if not np.isfinite(m) or m <= 0:
        return sqi
    before = max(1, int(round(before_fraction * m)))
    after = max(1, int(round(after_fraction * m)))
    inside = (peaks - before >= 0) & (peaks + after < len(signal))
    if inside.sum() < 5:
        return sqi
    idx = peaks[inside][:, None] + np.arange(-before, after + 1)[None, :]
    beats = signal[idx]
    beats = beats - beats.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(beats, axis=1)
    template = np.median(beats, axis=0)
    template -= template.mean()
    t_norm = np.linalg.norm(template)
    with np.errstate(invalid="ignore", divide="ignore"):
        sqi[inside] = beats @ template / (norms * t_norm)
    return sqi


# ---------------------------------------------------------------------------
# Lipponen-Tarvainen
# ---------------------------------------------------------------------------

def _rolling_quartile_deviation(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values)
    roll = series.rolling(window, center=True, min_periods=1)
    return ((roll.quantile(0.75) - roll.quantile(0.25)) / 2.0).to_numpy()


def classify_artifacts(rr: np.ndarray) -> dict[str, np.ndarray]:
    """Classify beats with the Lipponen-Tarvainen (2019) decision scheme.

    ``rr[i]`` is the interval ending at beat ``i + 1``.

    Args:
        rr: Inter-beat intervals in seconds.

    Returns:
        Dict of index arrays (into *rr*): ``ectopic``, ``missed``, ``extra``,
        ``longshort``.
    """
    rr = np.asarray(rr, dtype=float)
    n = len(rr)
    empty = {k: np.array([], dtype=int) for k in ("ectopic", "missed", "extra", "longshort")}
    if n < 4:
        return empty

    drr = np.diff(rr, prepend=np.nan)
    drr[0] = np.nanmean(drr[1:]) if n > 1 else 0.0
    th1 = LT_ALPHA * _rolling_quartile_deviation(np.abs(drr), LT_WINDOW)
    th1 = np.where(th1 > 0, th1, np.nan)
    drrs = drr / th1

    medrr = pd.Series(rr).rolling(LT_MEDIAN_WINDOW, center=True, min_periods=1).median().to_numpy()
    mrr = rr - medrr
    mrr = np.where(mrr < 0, 2.0 * mrr, mrr)
    th2 = LT_ALPHA * _rolling_quartile_deviation(np.abs(mrr), LT_WINDOW)
    th2 = np.where(th2 > 0, th2, np.nan)
    mrrs = mrr / th2

    drrs = np.nan_to_num(drrs)
    mrrs = np.nan_to_num(mrrs)
    padded = np.r_[0.0, 0.0, drrs, 0.0, 0.0]
    s12 = np.zeros(n)
    s22 = np.zeros(n)
    for i in range(n):
        d = i + 2
        if padded[d] > 0:
            s12[i] = max(padded[d - 1], padded[d + 1])
        elif padded[d] < 0:
            s12[i] = min(padded[d - 1], padded[d + 1])
        if padded[d] >= 0:
            s22[i] = min(padded[d + 1], padded[d + 2])
        else:
            s22[i] = max(padded[d + 1], padded[d + 2])

    found: dict[str, list[int]] = {k: [] for k in empty}
    i = 0
    while i < n:
        if abs(drrs[i]) <= 1 and abs(mrrs[i]) <= 3:
            i += 1
            continue
        eq1 = drrs[i] > 1 and s12[i] < (-LT_C1 * drrs[i] - LT_C2)
        eq2 = drrs[i] < -1 and s12[i] > (-LT_C1 * drrs[i] + LT_C2)
        if eq1 or eq2:
            found["ectopic"].append(i)
            i += 1
            continue
        candidates = [i]
        if i + 2 < n and abs(drrs[i + 1]) < abs(drrs[i + 2]):
            candidates.append(i + 1)
        for j in candidates:
            eq3 = drrs[j] > 1 and s22[j] < -1
            eq4 = abs(mrrs[j]) > 3
            eq5 = drrs[j] < -1 and s22[j] > 1
            if not (eq3 or eq4 or eq5):
                continue
            tol = th2[j] if np.isfinite(th2[j]) else 0.0
            eq6 = abs(rr[j] / 2.0 - medrr[j]) < tol
            eq7 = j + 1 < n and abs(rr[j] + rr[j + 1] - medrr[j]) < tol
            if (eq5 or rr[j] < medrr[j]) and eq7:
                found["extra"].append(j)
            elif (eq3 or rr[j] > medrr[j]) and eq6:
                found["missed"].append(j)
            else:
                found["longshort"].append(j)
        i += 1
    return {k: np.unique(np.asarray(v, dtype=int)) for k, v in found.items()}


@dataclass
class CorrectionResult:
    """Output of :func:`correct_beat_times`.

    Attributes:
        times: Corrected beat times (seconds, ascending).
        corrected: Boolean per output *interval*: ``True`` when the interval
            was created or altered by a correction.
        counts: Number of beats corrected per artifact class.
        altered_beats: Boolean per output *beat*: synthetic (interpolated
            position), or adjacent to a deleted beat - the unit of the
            correction burden.
        recovered_beats: Boolean per output *beat*: re-located onto a real
            pulse of the signal by the beat locator (not synthetic).
    """

    times: np.ndarray
    corrected: np.ndarray
    counts: dict[str, int] = field(default_factory=dict)
    altered_beats: np.ndarray | None = None
    recovered_beats: np.ndarray | None = None


def correct_beat_times(
    times_s: np.ndarray,
    max_iterations: int = 3,
    max_gap_s: float = 3.0,
    locator=None,
) -> CorrectionResult:
    """Time-conserving Lipponen-Tarvainen correction of a beat sequence.

    Intervals longer than *max_gap_s* are genuine gaps: they are never split
    and are left for the segmentation stage to cut at.

    With a *locator* (see :func:`make_beat_locator`), a beat to be inserted
    (missed) or moved (ectopic / long-short) is first searched for in the
    signal around its expected position; when a pulse matching the
    recording's beat template is found, its real time is used and the beat
    is marked *recovered*.  Only beats that cannot be recovered receive the
    interpolated (synthetic) midpoint and count towards the burden.

    Args:
        times_s: Beat times in seconds (ascending).
        max_iterations: Iterations of classification + correction (Kubios
            applies the procedure iteratively).
        max_gap_s: Longest interval considered for correction.
        locator: Optional ``f(target_s, lo_s, hi_s) -> float | None``.

    Returns:
        ``CorrectionResult``.
    """
    times = np.asarray(times_s, dtype=float).copy()
    altered = np.zeros(len(times), dtype=bool)  # per beat: synthetic position
    recovered = np.zeros(len(times), dtype=bool)  # per beat: re-located on the signal
    counts = {"ectopic": 0, "missed": 0, "extra": 0, "longshort": 0, "recovered": 0}

    def _locate(target: float, lo: float, hi: float) -> float | None:
        if locator is None or not hi > lo:
            return None
        found = locator(target, lo, hi)
        return float(found) if found is not None and lo <= found <= hi else None
    merged_at: list[float] = []

    for _ in range(max_iterations):
        if len(times) < 5:
            break
        rr = np.diff(times)
        found = classify_artifacts(rr)
        gap = rr > max_gap_s
        found = {k: v[~gap[v]] for k, v in found.items()}
        if not any(len(v) for v in found.values()):
            break

        moved = np.zeros(len(times), dtype=bool)
        relocated = np.zeros(len(times), dtype=bool)
        inserts: list[float] = []
        insert_real: list[bool] = []
        deletes: set[int] = set()

        for j in found["ectopic"].tolist() + found["longshort"].tolist():
            b = j + 1  # beat closing interval j
            if 0 < b < len(times) - 1 and b not in deletes:
                span = times[b + 1] - times[b - 1]
                target = 0.5 * (times[b - 1] + times[b + 1])
                real = _locate(target, times[b - 1] + 0.3 * span, times[b + 1] - 0.3 * span)
                if real is not None:
                    times[b] = real
                    relocated[b] = True
                else:
                    times[b] = target
                    moved[b] = True
        for j in found["missed"]:
            span = times[j + 1] - times[j]
            target = 0.5 * (times[j] + times[j + 1])
            real = _locate(target, times[j] + 0.3 * span, times[j + 1] - 0.3 * span)
            inserts.append(real if real is not None else target)
            insert_real.append(real is not None)
        for j in found["extra"]:
            b = j + 1
            if 0 < b < len(times) - 1:
                deletes.add(b)
                merged_at.append(times[b])

        counts["ectopic"] += len(found["ectopic"])
        counts["longshort"] += len(found["longshort"])
        counts["missed"] += len(found["missed"])
        counts["extra"] += len(deletes)
        counts["recovered"] += int(relocated.sum()) + int(sum(insert_real))

        keep = np.ones(len(times), dtype=bool)
        keep[list(deletes)] = False
        real_inserts = np.asarray(insert_real, dtype=bool)
        new_times = np.r_[times[keep], np.asarray(inserts)]
        new_altered = np.r_[((altered | moved) & ~relocated)[keep], ~real_inserts]
        new_recovered = np.r_[((recovered | relocated) & ~moved)[keep], real_inserts]
        order = np.argsort(new_times, kind="stable")
        times, altered, recovered = new_times[order], new_altered[order], new_recovered[order]

    corrected = np.zeros(max(len(times) - 1, 0), dtype=bool)
    altered_beats = altered.copy()
    if len(times) > 1:
        corrected |= altered[:-1] | altered[1:]
        for t in merged_at:
            k = np.searchsorted(times, t) - 1
            if 0 <= k < len(corrected):
                corrected[k] = True
                altered_beats[k] = True
    return CorrectionResult(times=times, corrected=corrected, counts=counts,
                            altered_beats=altered_beats, recovered_beats=recovered)


def make_beat_locator(
    signal: np.ndarray,
    sample_seconds: np.ndarray,
    t0_s: float,
    peaks: np.ndarray,
    min_correlation: float = 0.8,
    position_fn=None,
):
    """Build a locator that finds real pulses for the correction stage.

    The template is the median beat around *peaks* (one median cardiac cycle,
    as in :func:`beat_template_sqi`).  ``locate(target, lo, hi)`` examines
    every local maximum of *signal* between absolute times *lo* and *hi*,
    keeps those whose beat correlates with the template at least
    ``min_correlation``, and returns the sub-sample (parabolic) time of the
    best-correlated one (ties broken by closeness to *target*), or ``None``.

    Args:
        signal: Filtered signal the beats were detected on.
        sample_seconds: Fitted time of every sample relative to *t0_s*.
        t0_s: Absolute time (s) of sample 0.
        peaks: Accepted beat indices used to build the template.
        min_correlation: Acceptance threshold for a recovered pulse.
        position_fn: Optional ``f(index) -> fractional position`` that maps a
            recovered maximum to the pipeline's fiducial, so recovered beats
            are timed exactly like detected ones (default: parabolic peak).

    Returns:
        Callable, or ``None`` when no template can be built.
    """
    peaks = np.asarray(peaks, dtype=int)
    n = len(signal)
    if len(peaks) < 10 or n < 10:
        return None
    m = float(np.median(np.diff(peaks)))
    before, after = max(1, int(round(0.4 * m))), max(1, int(round(0.6 * m)))
    inside = peaks[(peaks - before >= 0) & (peaks + after < n)]
    if len(inside) < 10:
        return None
    offsets = np.arange(-before, after + 1)
    beats = signal[inside[:, None] + offsets[None, :]]
    beats = beats - beats.mean(axis=1, keepdims=True)
    template = np.median(beats, axis=0)
    template -= template.mean()
    t_norm = float(np.linalg.norm(template))
    if t_norm == 0:
        return None
    index = np.arange(n, dtype=float)

    def locate(target_s: float, lo_s: float, hi_s: float) -> float | None:
        lo = int(np.ceil(np.interp(lo_s - t0_s, sample_seconds, index)))
        hi = int(np.floor(np.interp(hi_s - t0_s, sample_seconds, index)))
        lo, hi = max(lo, before + 1), min(hi, n - after - 2)
        if hi <= lo:
            return None
        seg = signal[lo - 1: hi + 2]
        cand = np.where((seg[1:-1] >= seg[:-2]) & (seg[1:-1] > seg[2:]))[0] + lo
        best = None
        for p in cand:
            w = signal[p - before: p + after + 1]
            w = w - w.mean()
            norm = float(np.linalg.norm(w))
            if norm == 0:
                continue
            r = float(w @ template) / (norm * t_norm)
            if r < min_correlation:
                continue
            if position_fn is not None:
                pos = float(position_fn(int(p)))
            else:
                y0, y1, y2 = signal[p - 1], signal[p], signal[p + 1]
                den = y0 - 2 * y1 + y2
                pos = p + (float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if den != 0 else 0.0)
            t = t0_s + float(np.interp(pos, index, sample_seconds))
            key = (round(r, 3), -abs(t - target_s))
            if best is None or key > best[0]:
                best = (key, t)
        return None if best is None else best[1]

    return locate
