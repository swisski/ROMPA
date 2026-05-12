"""Neighborhood-based spatial verification for onset fields.

Implements the Fractions Skill Score (FSS) of

    Roberts & Lean 2008, "Scale-selective verification of rainfall
    accumulations from high-resolution forecasts of convective events,"
    Mon. Wea. Rev. 136, 78-97. doi:10.1175/2007MWR2123.1

Given two 2-D onset-date fields (forecast and observation), FSS is computed
by: (1) thresholding each field with ``fcst_doy <= threshold`` to produce a
binary "has onset by DOY" mask, with NaN/NaT treated as False; (2) computing
the fraction of "has onset" cells in every n-by-n neighborhood; (3) comparing
the forecast and observed fraction fields via

    FSS(tau, n) = 1 - MSE(F_f, F_o) / (mean(F_f**2) + mean(F_o**2))

where the MSE and means are taken over the whole domain. FSS = 1 is perfect,
FSS = 0 is no skill. The score is non-decreasing in the neighborhood size n.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter


def _as_binary_mask(field, threshold: float) -> np.ndarray:
    """Threshold a 2-D field to a {0, 1} float mask, NaN -> 0."""
    a = np.asarray(field, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"expected 2-D field, got shape {a.shape}")
    mask = (a <= threshold) & np.isfinite(a)
    return mask.astype(float)


def _fraction_field(binary: np.ndarray, neighborhood: int) -> np.ndarray:
    """Box-average binary over an ``neighborhood``-by-``neighborhood`` window.

    ``neighborhood`` must be a positive odd integer.

    Boundary convention: cells outside the domain are treated as 0
    (zero-padding via ``mode='constant', cval=0.0``). This matches
    pysteps and is the most common implementation choice, but it is
    *not* exactly what Roberts & Lean (2008) describe — they compute
    fractions only on fully-interior pixels, dividing by the window
    area regardless of how many cells fall inside the domain.
    Zero-padding biases boundary skill upward: a 1-cell shift at the
    domain edge scores higher than the same shift in the interior
    because the zero-padded fraction differences are smaller.

    For India onset, the monsoon front frequently enters and exits at
    the south and north of the domain — interpret edge FSS values
    accordingly.
    """
    if neighborhood <= 0 or neighborhood % 2 == 0:
        raise ValueError(f"neighborhood must be a positive odd integer, got {neighborhood}")
    if neighborhood == 1:
        return binary
    return uniform_filter(binary, size=neighborhood, mode="constant", cval=0.0)


def fss_single(
    forecast: np.ndarray,
    observed: np.ndarray,
    *,
    threshold: float,
    neighborhood: int,
) -> float:
    """Fractions Skill Score for one (threshold, neighborhood) pair.

    Returns NaN when the reference denominator is zero, i.e. both fields are
    identically False — the score is undefined there.
    """
    f_bin = _as_binary_mask(forecast, threshold)
    o_bin = _as_binary_mask(observed, threshold)
    if f_bin.shape != o_bin.shape:
        raise ValueError(
            f"forecast and observed shape mismatch: {f_bin.shape} vs {o_bin.shape}"
        )
    F_f = _fraction_field(f_bin, neighborhood)
    F_o = _fraction_field(o_bin, neighborhood)
    mse = float(np.mean((F_f - F_o) ** 2))
    denom = float(np.mean(F_f**2) + np.mean(F_o**2))
    if denom == 0.0:
        return float("nan")
    return 1.0 - mse / denom


def fss(
    forecast,
    observed,
    *,
    thresholds: Sequence[float],
    neighborhoods: Sequence[int],
    lat_coord: str = "lat",
    lon_coord: str = "lon",
) -> xr.DataArray:
    """Fractions Skill Score over a sweep of thresholds and neighborhood sizes.

    Parameters
    ----------
    forecast, observed : 2-D array-like or xr.DataArray
        Onset-date fields. NaN / NaT entries are treated as "no onset"
        (i.e. the threshold condition is False).
    thresholds : sequence of numeric
        DOY thresholds tau; the binary mask is ``field <= tau``.
    neighborhoods : sequence of positive odd int
        Square window sizes in grid cells.
    lat_coord, lon_coord : str
        Coordinate names used on xarray inputs; ignored for numpy arrays.

    Returns
    -------
    xr.DataArray
        Array with dims ``("threshold", "neighborhood")``.
    """
    f_arr = forecast.values if isinstance(forecast, xr.DataArray) else np.asarray(forecast)
    o_arr = observed.values if isinstance(observed, xr.DataArray) else np.asarray(observed)

    thr = np.asarray(thresholds, dtype=float)
    nbr = np.asarray(neighborhoods, dtype=int)

    out = np.full((thr.size, nbr.size), np.nan, dtype=float)
    for i, t in enumerate(thr):
        f_bin = _as_binary_mask(f_arr, float(t))
        o_bin = _as_binary_mask(o_arr, float(t))
        if f_bin.shape != o_bin.shape:
            raise ValueError(
                f"forecast and observed shape mismatch: {f_bin.shape} vs {o_bin.shape}"
            )
        for j, n in enumerate(nbr):
            F_f = _fraction_field(f_bin, int(n))
            F_o = _fraction_field(o_bin, int(n))
            mse = float(np.mean((F_f - F_o) ** 2))
            denom = float(np.mean(F_f**2) + np.mean(F_o**2))
            out[i, j] = 1.0 - mse / denom if denom > 0 else float("nan")

    return xr.DataArray(
        out,
        dims=("threshold", "neighborhood"),
        coords={"threshold": thr, "neighborhood": nbr},
        name="fss",
        attrs={
            "description": (
                "Fractions Skill Score. Roberts & Lean 2008. "
                "1 = perfect, 0 = no skill. Non-decreasing in neighborhood."
            )
        },
    )


def fss_multi_year(
    forecast_by_year: dict,
    observed_by_year: dict,
    *,
    thresholds: Sequence[float],
    neighborhoods: Sequence[int],
) -> xr.DataArray:
    """Per-year FSS across a shared set of thresholds and neighborhoods.

    ``forecast_by_year`` and ``observed_by_year`` are dicts keyed by year
    mapping to 2-D onset-date fields.

    Returns an xr.DataArray with dims ``("year", "threshold", "neighborhood")``.
    Missing years (keys in forecast but not observed, or vice versa) are
    skipped with a warning; present-in-both years are included.
    """
    shared_years = sorted(set(forecast_by_year) & set(observed_by_year))
    if not shared_years:
        raise ValueError("no overlapping years between forecast and observed dicts")

    thr = np.asarray(thresholds, dtype=float)
    nbr = np.asarray(neighborhoods, dtype=int)

    stack = np.full((len(shared_years), thr.size, nbr.size), np.nan, dtype=float)
    for k, yr in enumerate(shared_years):
        per_year = fss(
            forecast_by_year[yr],
            observed_by_year[yr],
            thresholds=thr,
            neighborhoods=nbr,
        )
        stack[k] = per_year.values

    return xr.DataArray(
        stack,
        dims=("year", "threshold", "neighborhood"),
        coords={"year": shared_years, "threshold": thr, "neighborhood": nbr},
        name="fss_multi_year",
        attrs={"description": "Per-year Fractions Skill Score (Roberts & Lean 2008)."},
    )


def base_rate(observed, threshold: float) -> float:
    """Fraction of finite cells in ``observed`` with onset DOY ≤ threshold.

    This is the climatological base rate ``p(τ)`` used to define the
    Roberts & Lean (2008) "useful skill" cutoff
    ``FSS_useful(τ) = 0.5 + 0.5·p(τ)``. Returns 0 when no cells qualify
    (either domain is all-NaN or the threshold is below every onset).
    """
    a = np.asarray(observed.values if hasattr(observed, "values") else observed,
                   dtype=float)
    finite = np.isfinite(a)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return 0.0
    n_on = int(((a <= float(threshold)) & finite).sum())
    return float(n_on) / float(n_finite)


def useful_skill_threshold(p: float) -> float:
    """Roberts & Lean (2008) useful-skill cutoff: ``0.5 + 0.5·p``.

    The intuition: a forecast that always says "no onset" everywhere
    achieves FSS = 0; a forecast distributed exactly like climatology
    achieves FSS = p; the halfway point ``0.5 + 0.5·p`` is the
    threshold above which the forecast is "doing something useful"
    relative to the trivial reference.
    """
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"base rate p must be in [0, 1]; got {p}")
    return 0.5 + 0.5 * p


def useful_scale(
    fss_values: np.ndarray,
    neighborhoods: Sequence[int],
    *,
    p: float,
) -> float:
    """Smallest neighborhood ``n`` at which ``FSS(τ, n)`` ≥ FSS_useful(τ).

    Given a 1-D FSS curve over neighborhoods (for one threshold τ) and
    the corresponding climatological base rate ``p(τ)``, return the
    smallest ``n`` whose FSS reaches the useful-skill cutoff. Linear
    interpolation between adjacent neighborhood sizes refines the
    crossing to a non-integer value (FSS is monotonic non-decreasing
    in ``n`` so the crossing is unique once it occurs).

    Returns
    -------
    float
        The interpolated useful scale in grid cells. ``NaN`` if the
        FSS curve never reaches the threshold within the tested range
        (i.e. the model has no useful skill at any tested scale).

    Raises
    ------
    ValueError
        If ``fss_values`` and ``neighborhoods`` lengths disagree, or
        ``neighborhoods`` is not strictly increasing.
    """
    f = np.asarray(fss_values, dtype=float)
    n = np.asarray(list(neighborhoods), dtype=float)
    if f.shape != n.shape or f.ndim != 1:
        raise ValueError(
            f"fss_values shape {f.shape} must match neighborhoods shape {n.shape}, both 1-D"
        )
    if n.size == 0:
        return float("nan")
    if np.any(np.diff(n) <= 0):
        raise ValueError("neighborhoods must be strictly increasing")

    threshold = useful_skill_threshold(p)
    # Smallest index where FSS first meets/exceeds the threshold.
    above = f >= threshold
    if not above.any():
        return float("nan")
    idx = int(np.argmax(above))   # first True
    if idx == 0:
        # Already useful at the smallest neighborhood — return that n.
        return float(n[0])
    f_lo, f_hi = float(f[idx - 1]), float(f[idx])
    n_lo, n_hi = float(n[idx - 1]), float(n[idx])
    if not np.isfinite(f_lo) or f_hi == f_lo:
        return float(n_hi)
    # Linear interpolation in (n, FSS) of the threshold crossing.
    frac = (threshold - f_lo) / (f_hi - f_lo)
    return float(n_lo + frac * (n_hi - n_lo))


def useful_scale_per_threshold(
    fss_matrix: np.ndarray,
    neighborhoods: Sequence[int],
    base_rates: Sequence[float],
) -> np.ndarray:
    """Vectorised ``useful_scale`` over a (threshold × neighborhood) matrix.

    ``fss_matrix`` has shape ``(n_thresholds, n_neighborhoods)``.
    ``base_rates`` has length ``n_thresholds``.
    Returns a 1-D array of length ``n_thresholds``; entries are the
    interpolated useful scales (or NaN if the curve never crosses).
    """
    f = np.asarray(fss_matrix, dtype=float)
    p_arr = np.asarray(list(base_rates), dtype=float)
    if f.ndim != 2 or f.shape[0] != p_arr.size:
        raise ValueError(
            f"fss_matrix shape {f.shape} inconsistent with base_rates length {p_arr.size}"
        )
    out = np.empty(f.shape[0], dtype=float)
    for i in range(f.shape[0]):
        out[i] = useful_scale(f[i], neighborhoods, p=float(p_arr[i]))
    return out
