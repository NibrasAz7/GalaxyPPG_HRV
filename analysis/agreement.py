"""Method-comparison statistics for repeated-measures validation data.

Estimators follow the method-comparison literature rather than generic
hypothesis testing:

* **ICC(A,1)** - two-way random effects, absolute agreement, single
  measurement (McGraw & Wong, 1996, case 2A; Koo & Li, 2016), with a
  participant-bootstrap 95 % confidence interval.  Consistency ICC(3,1) ignores systematic
  bias and is not a measure of agreement.
* **Bland-Altman limits for repeated measurements** with a varying true
  value (Bland & Altman, 2007): the SD of differences combines between- and
  within-participant variance from a one-way ANOVA on the differences, so
  pooled activities from the same participant are not treated as
  independent.
* **Log-ratio Bland-Altman** for strictly positive, heteroscedastic HRV
  indices (RMSSD, HF power...): limits are computed on ln(test/reference)
  and back-transformed to ratios (Euser et al., 2008).
* **Two one-sided tests (TOST)** against a priori equivalence bounds
  (Lakens, 2017).
* **Participant-clustered bootstrap** confidence intervals for every
  summary: participants (not rows) are resampled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _finite_pairs(a, b) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _icc_a1_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ICC(A,1) of paired measurements along the last axis (vectorised over leading axes)."""
    n = a.shape[-1]
    k = 2
    row_mean = (a + b) / 2.0
    col_a, col_b = a.mean(axis=-1), b.mean(axis=-1)
    grand = (col_a + col_b) / 2.0
    ss_rows = k * ((row_mean - grand[..., None]) ** 2).sum(axis=-1)
    ss_cols = n * ((col_a - grand) ** 2 + (col_b - grand) ** 2)
    ss_total = ((a - grand[..., None]) ** 2 + (b - grand[..., None]) ** 2).sum(axis=-1)
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = (ss_total - ss_rows - ss_cols) / ((n - 1) * (k - 1))
    denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (msr - mse) / denominator, np.nan)


def icc_a1(test, reference, alpha: float = 0.05, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """ICC(A,1): two-way random effects, absolute agreement, single measurement.

    The point estimate follows McGraw & Wong (1996, case 2A; Koo & Li, 2016).
    The confidence interval is a percentile bootstrap over the rows (pass one
    row per participant so that the resampling unit is independent); the
    closed-form F-based interval is not used because it is unreliable when
    the systematic-difference variance is non-negligible.

    Args:
        test: Measurements of the device under test (one per participant).
        reference: Reference measurements (aligned with *test*).
        alpha: Significance level of the two-sided CI.
        n_boot: Bootstrap replicates.
        seed: RNG seed.

    Returns:
        Dict with ``icc``, ``ci_low``, ``ci_high`` and ``n``.
    """
    a, b = _finite_pairs(test, reference)
    n = len(a)
    out = {"icc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": n}
    if n < 3:
        return out
    point = float(_icc_a1_batch(a, b))
    out["icc"] = point
    if not np.isfinite(point) or n < 5:
        return out
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    samples = _icc_a1_batch(a[idx], b[idx])
    samples = samples[np.isfinite(samples)]
    if len(samples) >= n_boot // 2:
        out["ci_low"], out["ci_high"] = (float(q) for q in np.quantile(samples, [alpha / 2, 1 - alpha / 2]))
    return out


def bland_altman(test, reference, subjects=None) -> dict[str, float]:
    """Bland-Altman bias and 95 % limits of agreement.

    With *subjects* given (several observations per participant), the SD of
    differences follows Bland & Altman (2007) for a true value that varies
    within participants: ``SD^2 = s_between^2 + s_within^2`` estimated from a
    one-way ANOVA with unequal group sizes.

    Returns:
        Dict with ``bias``, ``sd``, ``loa_low``, ``loa_high``, ``n``,
        ``n_subjects``.
    """
    a, b = np.asarray(test, dtype=float), np.asarray(reference, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    n = len(d)
    out = {"bias": np.nan, "sd": np.nan, "loa_low": np.nan, "loa_high": np.nan,
           "n": n, "n_subjects": np.nan}
    if n < 3:
        return out
    bias = float(np.mean(d))
    if subjects is None:
        sd = float(np.std(d, ddof=1))
        out["n_subjects"] = n
    else:
        groups = np.asarray(subjects)[mask]
        frame = pd.DataFrame({"g": groups, "d": d})
        sizes = frame.groupby("g")["d"].size()
        m = len(sizes)
        out["n_subjects"] = m
        if m < 2 or n <= m:
            sd = float(np.std(d, ddof=1))
        else:
            means = frame.groupby("g")["d"].mean()
            ss_between = float((sizes * (means - bias) ** 2).sum())
            ss_within = float(((frame["d"] - frame["g"].map(means)) ** 2).sum())
            msb = ss_between / (m - 1)
            msw = ss_within / (n - m)
            divisor = (n ** 2 - (sizes ** 2).sum()) / ((m - 1) * n)
            var_between = max((msb - msw) / divisor, 0.0)
            sd = float(np.sqrt(var_between + msw))
    out.update({"bias": bias, "sd": sd, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd})
    return out


def log_ratio_bland_altman(test, reference, subjects=None) -> dict[str, float]:
    """Bland-Altman on ln(test/reference), back-transformed to ratios.

    Returns:
        Dict with ``ratio`` (geometric mean ratio), ``ratio_loa_low``,
        ``ratio_loa_high``, ``log_bias``, ``log_sd`` and ``n``.
    """
    a, b = np.asarray(test, dtype=float), np.asarray(reference, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    groups = None if subjects is None else np.asarray(subjects)[valid]
    result = bland_altman(np.log(a[valid]), np.log(b[valid]), groups)
    return {
        "ratio": float(np.exp(result["bias"])) if np.isfinite(result["bias"]) else np.nan,
        "ratio_loa_low": float(np.exp(result["loa_low"])) if np.isfinite(result["loa_low"]) else np.nan,
        "ratio_loa_high": float(np.exp(result["loa_high"])) if np.isfinite(result["loa_high"]) else np.nan,
        "log_bias": result["bias"],
        "log_sd": result["sd"],
        "n": result["n"],
    }


def tost_paired(differences, low: float, high: float, alpha: float = 0.05) -> dict[str, float]:
    """Two one-sided paired t-tests for equivalence within ``[low, high]``.

    Returns:
        Dict with ``p_tost`` (max of the two one-sided p-values), the
        ``(1 - 2 alpha)`` confidence interval of the mean difference
        (``ci90_low``/``ci90_high``) and ``equivalent`` (bool).
    """
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    out = {"p_tost": np.nan, "ci90_low": np.nan, "ci90_high": np.nan, "equivalent": False, "n": n}
    if n < 3:
        return out
    mean, se = float(np.mean(d)), float(np.std(d, ddof=1) / np.sqrt(n))
    if se == 0:
        inside = low < mean < high
        out.update({"p_tost": 0.0 if inside else 1.0, "ci90_low": mean, "ci90_high": mean,
                    "equivalent": inside})
        return out
    t_low = (mean - low) / se
    t_high = (mean - high) / se
    p_low = 1 - stats.t.cdf(t_low, n - 1)
    p_high = stats.t.cdf(t_high, n - 1)
    half = stats.t.ppf(1 - alpha, n - 1) * se
    p = float(max(p_low, p_high))
    out.update({"p_tost": p, "ci90_low": mean - half, "ci90_high": mean + half,
                "equivalent": bool(p < alpha)})
    return out


def cluster_bootstrap(frame: pd.DataFrame, statistic, cluster: str = "participant",
                      n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and percentile CI with participants as resampling units.

    Args:
        frame: Data with a *cluster* column.
        statistic: Callable ``frame -> float``.
        cluster: Column identifying the resampling unit.
        n_boot: Bootstrap replicates.
        seed: RNG seed (deterministic output).
        alpha: Two-sided level.

    Returns:
        Tuple ``(estimate, ci_low, ci_high)``.
    """
    estimate = float(statistic(frame)) if len(frame) else np.nan
    ids = frame[cluster].unique()
    if len(ids) < 3 or not np.isfinite(estimate):
        return estimate, np.nan, np.nan
    rng = np.random.default_rng(seed)
    codes = pd.factorize(frame[cluster])[0]
    rows_by_cluster = [np.flatnonzero(codes == k) for k in range(len(ids))]
    draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    values = np.empty(n_boot)
    for i, sample in enumerate(draws):
        values[i] = statistic(frame.iloc[np.concatenate([rows_by_cluster[k] for k in sample])])
    values = values[np.isfinite(values)]
    if len(values) < n_boot // 2:
        return estimate, np.nan, np.nan
    return estimate, float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def cluster_bootstrap_mean(frame: pd.DataFrame, column: str, cluster: str = "participant",
                           n_boot: int = 2000, seed: int = 0,
                           alpha: float = 0.05) -> tuple[float, float, float]:
    """Vectorised :func:`cluster_bootstrap` for the mean of one column.

    Each replicate's mean is the ratio of resampled per-cluster sums and
    counts, so all replicates are computed in one array operation.
    """
    sub = frame[[cluster, column]].dropna()
    if sub.empty:
        return np.nan, np.nan, np.nan
    estimate = float(sub[column].mean())
    grouped = sub.groupby(cluster)[column]
    sums, counts = grouped.sum().to_numpy(float), grouped.count().to_numpy(float)
    k = len(sums)
    if k < 3:
        return estimate, np.nan, np.nan
    draws = np.random.default_rng(seed).integers(0, k, size=(n_boot, k))
    values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return estimate, float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def friedman(frame: pd.DataFrame, value: str, within: str, subject: str = "participant") -> dict[str, float]:
    """Friedman test across *within* levels for complete subjects only.

    Returns:
        Dict with ``chi2``, ``p_value``, ``n_subjects``, ``k`` and Kendall's
        ``w`` (effect size).
    """
    wide = frame.pivot_table(index=subject, columns=within, values=value, observed=True).dropna()
    n, k = wide.shape
    if n < 3 or k < 3:
        return {"chi2": np.nan, "p_value": np.nan, "n_subjects": n, "k": k, "w": np.nan}
    chi2, p = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    return {"chi2": float(chi2), "p_value": float(p), "n_subjects": int(n), "k": int(k),
            "w": float(chi2 / (n * (k - 1)))}
