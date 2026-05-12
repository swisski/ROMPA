"""Multi-year aggregation for the frontend metric endpoints.

Each helper takes a list of per-year metric dicts (the same shapes that
metrics.py emits) and returns a single aggregated dict. Aggregation rule
is per-metric:

- CRPS field          : per-cell mean across years (nanmean)
- FSS matrix          : per-(threshold, neighborhood) mean
- Displacement sweep  : per-threshold median + IQR
- Progression curves  : per-day median + IQR + 95% bootstrap CI on the
                         median, for ioe / extent / misp / sps. The IQR
                         is descriptive of year-to-year spread; the CI
                         is the inferential quantity (uncertainty in
                         the central tendency given N years). See
                         momp.stats.bootstrap.
- Isochrones          : NOT aggregated — one representative year only
- CORP                : pool raw (p, y) pairs across years, then decompose
                         once. Implemented in metrics.py via a separate
                         entry point (compute_corp_pooled).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from momp.stats.bootstrap import (
    DEFAULT_CI_LEVEL,
    DEFAULT_N_RESAMPLES,
    bootstrap_median_ci,
)

# Reproducible-by-default seed for the year-resampling bootstrap. Using a
# fixed seed at the aggregator means /api/metrics/progression is
# deterministic given the same input years and config — matters for the
# walkthrough where users want to reload and see identical bands.
PROGRESSION_BOOTSTRAP_SEED = 20260505


def _stack(lists, fill=np.nan) -> np.ndarray:
    """Stack a list of equal-length iterables into a 2-D array (years × N).
    None entries become ``fill``. Raises if rows differ in length — our
    aggregate callers always enforce a shared axis (global thresholds /
    global days), so a ragged input is a bug upstream, not a case to
    silently NaN-pad and misalign per-column quantiles."""
    rows = []
    for r in lists:
        rows.append([fill if v is None else float(v) for v in (r or [])])
    if not rows:
        return np.zeros((0, 0))
    widths = {len(r) for r in rows}
    if len(widths) > 1:
        raise ValueError(
            f"cannot stack per-year lists with differing lengths {sorted(widths)}; "
            f"callers must pass a common axis (thresholds/days) across years"
        )
    width = next(iter(widths))
    out = np.full((len(rows), width), fill, dtype=float)
    for i, row in enumerate(rows):
        out[i, :] = row
    return out


def _quantiles(arr: np.ndarray, q: float) -> list:
    if arr.size == 0:
        return []
    return [float(v) if np.isfinite(v) else None
            for v in np.nanquantile(arr, q, axis=0)]


def _median(arr: np.ndarray) -> list:
    if arr.size == 0:
        return []
    return [float(v) if np.isfinite(v) else None
            for v in np.nanmedian(arr, axis=0)]


def aggregate_crps(per_year: Sequence[dict]) -> dict:
    empty = {
        "field": None, "mean": None, "max": None, "n_finite": 0,
        "median": None, "q25": None, "q75": None,
        "fair": None, "n_members": None,
    }
    if not per_year:
        return {**empty, "n_years": 0}
    fields = [p.get("field") for p in per_year if p.get("field")]
    if not fields:
        return {**empty, "n_years": len(per_year)}
    lat, lon = fields[0]["lat"], fields[0]["lon"]
    Ny, Nx = len(lat), len(lon)
    stack = np.full((len(fields), Ny, Nx), np.nan, dtype=float)
    for i, f in enumerate(fields):
        for r in range(min(Ny, len(f["values"]))):
            row = f["values"][r] or []
            for c in range(min(Nx, len(row))):
                v = row[c]
                stack[i, r, c] = float(v) if v is not None else np.nan
    with np.errstate(invalid="ignore"):
        mean_field = np.nanmean(stack, axis=0)
    out_values = [
        [None if not np.isfinite(v) else float(v) for v in row]
        for row in mean_field
    ]
    finite = mean_field[np.isfinite(mean_field)]
    # Carry through the fair-CRPS flag + ensemble size from per-year dicts
    # (they all share the same (fair, n_members) for a given model) so the
    # multi-year CRPS schema matches the single-year schema.
    fair_vals = {p.get("fair") for p in per_year if p.get("fair") is not None}
    n_mem_vals = {p.get("n_members") for p in per_year if p.get("n_members") is not None}
    return {
        "field": {"lat": lat, "lon": lon, "values": out_values},
        "mean": float(finite.mean()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "n_finite": int(finite.size),
        "n_years": len(fields),
        "median": float(np.nanmedian(finite)) if finite.size else None,
        "q25": float(np.nanquantile(finite, 0.25)) if finite.size else None,
        "q75": float(np.nanquantile(finite, 0.75)) if finite.size else None,
        "fair": next(iter(fair_vals)) if len(fair_vals) == 1 else None,
        "n_members": next(iter(n_mem_vals)) if len(n_mem_vals) == 1 else None,
    }


def aggregate_fss(
    per_year: Sequence[dict],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int | None = PROGRESSION_BOOTSTRAP_SEED,
) -> dict:
    """Aggregate per-year FSS payloads with median + bootstrap CIs.

    The previous version used the across-year mean, which is sensitive
    to outlier years and inconsistent with the progression aggregator.
    This version reports:

    - per-(τ, n) median + percentile-method bootstrap CI on the median
    - per-τ useful_scale (median + bootstrap CI on the useful scale,
      computed per year then aggregated, NOT computed from the median
      curve — bootstrapping the useful scale captures year-to-year
      jiggle in the crossing position, which the median-curve approach
      averages away)

    Per-year payloads are expected to carry ``base_rate`` (one float per
    threshold). If a payload is missing it, the useful-scale block is
    set to None for that aggregate.
    """
    if not per_year:
        return {
            "thresholds": [], "neighborhoods": [], "fss": [],
            "fss_ci_lo": [], "fss_ci_hi": [], "useful_scale": None,
            "n_years": 0, "ci_level": float(ci_level),
            "n_resamples": int(n_resamples),
            "bootstrap_method": "percentile-pair-bootstrap-median",
        }
    thr = per_year[0]["thresholds"]
    nbr = per_year[0]["neighborhoods"]
    Nt, Nn = len(thr), len(nbr)
    stack = np.full((len(per_year), Nt, Nn), np.nan, dtype=float)
    for i, p in enumerate(per_year):
        rows = p.get("fss") or []
        for r in range(min(Nt, len(rows))):
            row = rows[r] or []
            for c in range(min(Nn, len(row))):
                v = row[c]
                stack[i, r, c] = float(v) if v is not None else np.nan
    with np.errstate(invalid="ignore"):
        median_fss = np.nanmedian(stack, axis=0)

    # Per-(τ, n) bootstrap CI on the median, with coherent year resampling
    # across the (threshold, neighborhood) grid.
    boot = bootstrap_median_ci(
        stack, axis=0, n_resamples=n_resamples,
        ci_level=ci_level, rng=seed,
    )

    def _to_2d_jsonable(arr2d):
        return [[None if not np.isfinite(v) else float(v) for v in row]
                for row in np.asarray(arr2d)]

    out = {
        "thresholds": thr,
        "neighborhoods": nbr,
        "fss": _to_2d_jsonable(median_fss),
        "fss_ci_lo": _to_2d_jsonable(boot["ci_lo"]),
        "fss_ci_hi": _to_2d_jsonable(boot["ci_hi"]),
        "n_years": len(per_year),
        "ci_level": float(ci_level),
        "n_resamples": int(n_resamples),
        "bootstrap_method": "percentile-pair-bootstrap-median",
    }
    out["useful_scale"] = _aggregate_useful_scales(
        per_year, thr, nbr, n_resamples=n_resamples,
        ci_level=ci_level, seed=seed,
    )
    return out


def _aggregate_useful_scales(
    per_year, thresholds, neighborhoods, *, n_resamples, ci_level, seed,
):
    """Compute median + bootstrap CI on the per-year useful scale, per τ."""
    from momp.metrics.neighborhood import useful_scale

    Nt = len(thresholds)
    Nn = len(neighborhoods)
    nbr_arr = np.asarray(neighborhoods, dtype=float)

    # Per-year × per-threshold useful-scale matrix; NaN for years where the
    # FSS curve never crossed the threshold for that τ.
    n_years = len(per_year)
    per_year_us = np.full((n_years, Nt), np.nan, dtype=float)
    base_rates_per_year = np.full((n_years, Nt), np.nan, dtype=float)
    any_base_rate = False
    for i, p in enumerate(per_year):
        rates = p.get("base_rate")
        if rates is None or len(rates) != Nt:
            continue
        any_base_rate = True
        rows = p.get("fss") or []
        for r in range(Nt):
            try:
                p_r = float(rates[r])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(p_r):
                continue
            base_rates_per_year[i, r] = p_r
            curve = rows[r] if r < len(rows) else None
            if curve is None or len(curve) != Nn:
                continue
            curve_arr = np.array([np.nan if v is None else float(v) for v in curve],
                                 dtype=float)
            per_year_us[i, r] = useful_scale(curve_arr, nbr_arr, p=p_r)

    if not any_base_rate:
        return None

    # Aggregate per-threshold across years.
    out = {"thresholds": list(thresholds), "per_threshold": []}
    for r in range(Nt):
        col = per_year_us[:, r]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            out["per_threshold"].append({
                "threshold": float(thresholds[r]),
                "useful_scale": None,
                "ci_lo": None,
                "ci_hi": None,
                "n_years": 0,
                "n_years_finite": 0,
                "n_years_no_skill": int(np.sum(~np.isfinite(col)
                                               & ~np.isnan(base_rates_per_year[:, r]))),
            })
            continue
        median_us = float(np.median(finite))
        if finite.size >= 2:
            boot = bootstrap_median_ci(
                finite, axis=0, n_resamples=n_resamples,
                ci_level=ci_level, rng=seed,
            )
            ci_lo = _scalar_or_none(boot["ci_lo"])
            ci_hi = _scalar_or_none(boot["ci_hi"])
        else:
            ci_lo = ci_hi = median_us
        out["per_threshold"].append({
            "threshold": float(thresholds[r]),
            "useful_scale": median_us,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_years": int(col.size),
            "n_years_finite": int(finite.size),
            # How many years had a defined base_rate but never crossed
            # the useful-skill threshold — i.e. "no useful skill at any
            # tested scale" years. Different from "missing data" years.
            "n_years_no_skill": int(np.sum(~np.isfinite(col)
                                           & np.isfinite(base_rates_per_year[:, r]))),
        })

    # Single headline number: useful scale at the median threshold (the
    # most populated cell of the FSS matrix in practice). Convenient for
    # the bench summary table; full per-threshold detail is above.
    headline_idx = Nt // 2
    headline = out["per_threshold"][headline_idx]
    out["headline"] = {
        "threshold": headline["threshold"],
        "useful_scale": headline["useful_scale"],
        "ci_lo": headline["ci_lo"],
        "ci_hi": headline["ci_hi"],
    }
    return out


def aggregate_displacement(per_year: Sequence[dict]) -> dict:
    if not per_year:
        return {"thresholds": [], "n_years": 0}
    thr = per_year[0]["thresholds"]
    keys = ("delta_lat_deg", "delta_lon_deg", "great_circle_km", "area_bias_fraction")
    out = {"thresholds": list(thr), "n_years": len(per_year)}
    for k in keys:
        stk = _stack([p.get(k, []) for p in per_year])
        out[k] = _median(stk)
        out[f"{k}_q25"] = _quantiles(stk, 0.25)
        out[f"{k}_q75"] = _quantiles(stk, 0.75)
    return out


def aggregate_progression(
    per_year: Sequence[dict],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int | None = PROGRESSION_BOOTSTRAP_SEED,
) -> dict:
    """Aggregate per-year progression metrics across years.

    Reports three uncertainty layers per per-day series and per
    season-integrated scalar:

    - point estimate: nanmedian across years
    - IQR (q25 / q75): year-to-year spread, descriptive
    - 95% bootstrap CI (ci_lo / ci_hi): uncertainty in the median
      estimator, percentile-method pair bootstrap on the year axis

    The bootstrap is deterministic given ``seed`` so refreshing the
    panel does not jiggle the bands. Set ``seed=None`` to opt out.
    """
    if not per_year:
        return {"days": [], "n_years": 0,
                "ci_level": float(ci_level), "n_resamples": 0}
    days = per_year[0]["days"]
    ks = ("ioe_km2", "extent_km2", "misplacement_km2", "sps_km2")
    out = {"days": list(days), "n_years": len(per_year),
           "ci_level": float(ci_level), "n_resamples": int(n_resamples),
           "bootstrap_method": "percentile-pair-bootstrap-median"}
    for k in ks:
        per_year_lists = [p.get(k) for p in per_year]
        # SPS may be None for det models; drop None lists
        per_year_lists = [lst for lst in per_year_lists if lst is not None]
        if not per_year_lists:
            out[k] = None
            out[f"{k}_q25"] = None
            out[f"{k}_q75"] = None
            out[f"{k}_ci_lo"] = None
            out[f"{k}_ci_hi"] = None
            continue
        stk = _stack(per_year_lists)
        out[k] = _median(stk)
        out[f"{k}_q25"] = _quantiles(stk, 0.25)
        out[f"{k}_q75"] = _quantiles(stk, 0.75)
        boot = bootstrap_median_ci(
            stk, axis=0, n_resamples=n_resamples,
            ci_level=ci_level, rng=seed,
        )
        out[f"{k}_ci_lo"] = _to_jsonable_list(boot["ci_lo"])
        out[f"{k}_ci_hi"] = _to_jsonable_list(boot["ci_hi"])

    # Season-integrated scalars: median + IQR + bootstrap CI. "n_years"
    # is the count of years that actually contributed a finite value
    # for the metric; the request count lives in out["n_years"] above.
    season = {"n_years_requested": len(per_year)}
    for k in ("ioe_km2_day", "extent_km2_day", "misplacement_km2_day", "sps_km2_day"):
        vals = [p["season"].get(k) for p in per_year if p.get("season")]
        vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            season[k] = None
            season[f"{k}_q25"] = None
            season[f"{k}_q75"] = None
            season[f"{k}_ci_lo"] = None
            season[f"{k}_ci_hi"] = None
            season[f"{k}_n_years"] = 0
        else:
            arr = np.asarray(vals, dtype=float)
            season[k] = float(np.median(arr))
            season[f"{k}_q25"] = float(np.quantile(arr, 0.25))
            season[f"{k}_q75"] = float(np.quantile(arr, 0.75))
            boot = bootstrap_median_ci(
                arr, axis=0, n_resamples=n_resamples,
                ci_level=ci_level, rng=seed,
            )
            season[f"{k}_ci_lo"] = _scalar_or_none(boot["ci_lo"])
            season[f"{k}_ci_hi"] = _scalar_or_none(boot["ci_hi"])
            season[f"{k}_n_years"] = int(arr.size)
    # Misplacement fraction: fraction of season-integrated IOE that is
    # misplacement (vs extent). 0 means pure size error (forecast region
    # is too big or too small but in the right place); 1 means pure
    # geographic-misplacement error (right area, wrong location). Per
    # year compute the ratio, then bootstrap the median across years.
    # Bootstrapping the ratio (rather than ratio of bootstraps) is
    # important — numerator and denominator are correlated within a
    # year and the wrong order would over-estimate the variance.
    misp_fracs = []
    for p in per_year:
        s = p.get("season") or {}
        ioe_y = s.get("ioe_km2_day")
        misp_y = s.get("misplacement_km2_day")
        if (ioe_y is None or misp_y is None
                or not np.isfinite(ioe_y) or not np.isfinite(misp_y)
                or ioe_y <= 0):
            continue
        misp_fracs.append(float(misp_y) / float(ioe_y))
    if misp_fracs:
        arr = np.asarray(misp_fracs, dtype=float)
        boot = bootstrap_median_ci(
            arr, axis=0, n_resamples=n_resamples,
            ci_level=ci_level, rng=seed,
        )
        season["misp_frac"] = float(np.median(arr))
        season["misp_frac_ci_lo"] = _scalar_or_none(boot["ci_lo"])
        season["misp_frac_ci_hi"] = _scalar_or_none(boot["ci_hi"])
        season["misp_frac_n_years"] = int(arr.size)
    else:
        season["misp_frac"] = None
        season["misp_frac_ci_lo"] = None
        season["misp_frac_ci_hi"] = None
        season["misp_frac_n_years"] = 0

    # Backcompat alias
    season["n_years"] = season["ioe_km2_day_n_years"]
    season["ci_level"] = float(ci_level)
    out["season"] = season

    # Peak-DOY diagnostic, with bootstrap CI on the peak DOY itself.
    # Bootstrapping the peak position is the right object — bootstrapping
    # the median curve and then taking its argmax loses the year-to-year
    # variability in *where* the peak sits, which is exactly what we
    # care about. Each replicate is a coherent year resample, peak found
    # per replicate, percentile CI on the resulting peak DOYs.
    out["peak"] = _aggregate_peaks(
        per_year, days, n_resamples=n_resamples,
        ci_level=ci_level, seed=seed,
    )

    return out


def _aggregate_peaks(per_year, days, *, n_resamples, ci_level, seed):
    """Compute median + bootstrap CI on the peak DOY for IOE and SPS."""
    from momp.metrics.progression import peak_doy

    days_arr = np.asarray(list(days), dtype=float)
    out = {"days": list(days)}

    for kind in ("ioe_km2", "sps_km2"):
        per_year_lists = [p.get(kind) for p in per_year]
        per_year_lists = [lst for lst in per_year_lists if lst is not None]
        if not per_year_lists:
            for k in ("doy", "value", "doy_ci_lo", "doy_ci_hi"):
                out[f"{kind.replace('_km2', '')}_{k}"] = None
            continue
        stk = _stack(per_year_lists)  # shape (n_years, n_days)

        # Per-year peak DOYs: argmax along day axis (with the same NaN
        # / non-positive guard as the helper). Aggregate as median +
        # bootstrap CI on those peak-DOY samples.
        per_year_peak_doys = np.array(
            [peak_doy(row, days_arr)[0] for row in stk], dtype=float,
        )
        per_year_peak_vals = np.array(
            [peak_doy(row, days_arr)[1] for row in stk], dtype=float,
        )

        finite_doy = per_year_peak_doys[np.isfinite(per_year_peak_doys)]
        if finite_doy.size == 0:
            for k in ("doy", "value", "doy_ci_lo", "doy_ci_hi"):
                out[f"{kind.replace('_km2', '')}_{k}"] = None
            continue
        med_doy = float(np.median(finite_doy))
        finite_vals = per_year_peak_vals[np.isfinite(per_year_peak_vals)]
        med_val = float(np.median(finite_vals)) if finite_vals.size else None

        if finite_doy.size >= 2:
            boot = bootstrap_median_ci(
                finite_doy, axis=0, n_resamples=n_resamples,
                ci_level=ci_level, rng=seed,
            )
            ci_lo = _scalar_or_none(boot["ci_lo"])
            ci_hi = _scalar_or_none(boot["ci_hi"])
        else:
            ci_lo = ci_hi = med_doy

        prefix = kind.replace("_km2", "")  # "ioe" / "sps"
        out[f"{prefix}_doy"] = med_doy
        out[f"{prefix}_value"] = med_val
        out[f"{prefix}_doy_ci_lo"] = ci_lo
        out[f"{prefix}_doy_ci_hi"] = ci_hi
        out[f"{prefix}_n_years"] = int(finite_doy.size)

    return out


def _to_jsonable_list(arr) -> list:
    """1-D ndarray -> list of float|None, NaN -> None."""
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(arr).ravel()]


def _scalar_or_none(v) -> float | None:
    fv = float(np.asarray(v).item())
    return None if not np.isfinite(fv) else fv


def expand_years(years_arg: str | None, single_year: int | None) -> list[int]:
    """Parse a CSV ``years=2019,2020,2021`` (or a range ``2019-2023``) or
    fall back to ``year=2023``. Raises ValueError with a clear message on
    malformed input; the caller maps that to HTTP 400."""
    if years_arg:
        out: list[int] = []
        for tok in (t.strip() for t in years_arg.split(",")):
            if not tok:
                continue
            if "-" in tok and not tok.startswith("-"):
                lo_s, hi_s = tok.split("-", 1)
                try:
                    lo, hi = int(lo_s), int(hi_s)
                except ValueError:
                    raise ValueError(
                        f"malformed year range {tok!r}: expected YYYY-YYYY integers"
                    )
                if hi < lo:
                    raise ValueError(
                        f"year range {tok!r} is inverted (hi < lo); "
                        f"use lo-hi with lo <= hi"
                    )
                out.extend(range(lo, hi + 1))
            else:
                try:
                    out.append(int(tok))
                except ValueError:
                    raise ValueError(f"malformed year token {tok!r}: expected integer")
        if not out:
            raise ValueError(f"no years parsed from {years_arg!r}")
        return sorted(set(out))
    if single_year is not None:
        return [int(single_year)]
    raise ValueError("must supply either ?year=YYYY or ?years=YYYY[,YYYY|YYYY-YYYY]")
