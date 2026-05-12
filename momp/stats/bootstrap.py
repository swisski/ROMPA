"""Bootstrap confidence intervals on multi-year metric aggregates.

The frontend aggregates progression-curve metrics (IOE, SPS, extent,
misplacement) across N years by reporting a per-day median + IQR. The
IQR is descriptive (year-to-year spread of the realised values) but not
inferential (uncertainty in the central tendency itself). For the
methods-paper-quality use case we also want a confidence interval on
the aggregator — "given these N years, how well-pinned is the median?"

This module implements a percentile-method nonparametric pair bootstrap
for that purpose. The unit of resampling is a *year*: the same N
resampled-year indices are applied to every per-day position
simultaneously, which preserves the year-to-day correlation structure
while still giving a valid CI for the marginal distribution of the
year-aggregated estimator at each day.

References
----------
Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the
    Bootstrap.* Chapman & Hall.
Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and
    Their Application.* Cambridge University Press, ch. 5.

Choices and caveats
-------------------
- Standard pair bootstrap, not block bootstrap. Years of monsoon data
  are not strictly independent (ENSO and decadal modes induce serial
  correlation), but for the typical N ≤ 30 here the bias from a naive
  bootstrap is small relative to the sampling variability, and the
  block-length tuning required to do better is fragile at this size.
  A block bootstrap is a follow-on if/when N grows or if a sensitivity
  study shows the naive form misses substantially.
- Percentile method, not BCa. Percentile is monotone-equivariant under
  monotone transforms (so CIs on log(IOE) and IOE agree visually) and
  has no acceleration/skew estimate to mis-specify. BCa is a defensible
  upgrade for skewed sampling distributions; left as a follow-on.
- Reproducibility via numpy.random.Generator. A caller-supplied seed
  pins the exact resample indices.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

DEFAULT_N_RESAMPLES = 1000
DEFAULT_CI_LEVEL = 0.95


def _ci_quantiles(ci_level: float) -> Tuple[float, float]:
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must lie in (0, 1); got {ci_level}")
    alpha = 1.0 - ci_level
    return alpha / 2.0, 1.0 - alpha / 2.0


def bootstrap_median_ci(
    arr: np.ndarray,
    *,
    axis: int = 0,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
    rng: np.random.Generator | int | None = None,
) -> dict:
    """Percentile-method bootstrap CI on the median along ``axis``.

    The same resampled indices along ``axis`` are used at every other
    position — i.e. each bootstrap replicate is a coherent resampling
    of the *years*, not an independent resampling per day. This matters:
    independent per-day resampling would shrink CIs by ignoring the
    fact that a year that is bad on day 150 also tends to be bad on
    day 160.

    Parameters
    ----------
    arr : ndarray
        Stacked replicate data, e.g. shape ``(n_years, n_days)`` for a
        progression curve or ``(n_years,)`` for a season scalar. NaN
        values along ``axis`` are dropped per-position from the median;
        positions where all years are NaN return NaN for both point and
        CI.
    axis : int
        Resampling axis (default 0).
    n_resamples : int
        Number of bootstrap replicates (default 1000). For percentile
        CIs the rule of thumb is ≥ 1000 for 95% CIs and ≥ 2000 for 99%.
    ci_level : float
        Confidence level in ``(0, 1)`` (default 0.95).
    rng : Generator | int | None
        ``numpy.random.Generator``, an integer seed, or None (uses the
        default generator, non-reproducible). Pass an int for tests.

    Returns
    -------
    dict with keys
        ``median``  : ndarray, point estimate (nanmedian along axis).
        ``ci_lo``, ``ci_hi`` : ndarray, percentile-method CI bounds.
        ``n_replicates`` : int, the number of resamples actually used.
        ``n_years`` : int, length of the resampling axis.
        ``ci_level`` : float, echoed back for the caller's records.
        ``method`` : str, ``"percentile-pair-bootstrap-median"``.

    Notes on degenerate cases
    -------------------------
    - ``n_years == 0``: returns NaNs for ``median``, ``ci_lo``, ``ci_hi``
      with the right shape.
    - ``n_years == 1``: the bootstrap distribution is a point mass at the
      single observed value; ``ci_lo == ci_hi == median``. Reported but
      flagged: the caller should not interpret this as a tight CI.
    - All-NaN slice along ``axis``: median and CI are NaN for that slice.
    """
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return _empty_result(a.shape, axis, n_resamples, ci_level)

    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(rng)

    a = np.moveaxis(a, axis, 0)
    n_years = a.shape[0]
    other_shape = a.shape[1:]

    point = _nanmedian_axis0(a)

    if n_years == 0:
        nan = np.full(other_shape, np.nan, dtype=float)
        return {
            "median": nan,
            "ci_lo": nan.copy(),
            "ci_hi": nan.copy(),
            "n_replicates": 0,
            "n_years": 0,
            "ci_level": float(ci_level),
            "method": "percentile-pair-bootstrap-median",
        }

    if n_years == 1:
        return {
            "median": point,
            "ci_lo": point.copy(),
            "ci_hi": point.copy(),
            "n_replicates": 0,
            "n_years": 1,
            "ci_level": float(ci_level),
            "method": "percentile-pair-bootstrap-median",
        }

    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1; got {n_resamples}")

    q_lo, q_hi = _ci_quantiles(ci_level)

    idx = gen.integers(0, n_years, size=(n_resamples, n_years))
    replicate_medians = np.empty((n_resamples, *other_shape), dtype=float)
    for b in range(n_resamples):
        replicate_medians[b] = _nanmedian_axis0(a[idx[b]])

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        ci_lo = np.nanquantile(replicate_medians, q_lo, axis=0)
        ci_hi = np.nanquantile(replicate_medians, q_hi, axis=0)

    return {
        "median": point,
        "ci_lo": np.asarray(ci_lo, dtype=float),
        "ci_hi": np.asarray(ci_hi, dtype=float),
        "n_replicates": int(n_resamples),
        "n_years": int(n_years),
        "ci_level": float(ci_level),
        "method": "percentile-pair-bootstrap-median",
    }


def _nanmedian_axis0(a: np.ndarray) -> np.ndarray:
    """nanmedian over axis 0; returns NaN for all-NaN slices without warning."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmedian(a, axis=0)


def _empty_result(shape, axis, n_resamples, ci_level):
    other = list(shape)
    if other:
        other.pop(axis if axis >= 0 else len(other) + axis)
    nan = np.full(tuple(other) if other else (), np.nan, dtype=float)
    return {
        "median": nan,
        "ci_lo": nan.copy(),
        "ci_hi": nan.copy(),
        "n_replicates": 0,
        "n_years": 0,
        "ci_level": float(ci_level),
        "method": "percentile-pair-bootstrap-median",
    }
