"""Thin wrappers that turn ROMP metric outputs into JSON-ready dicts."""
from __future__ import annotations

import numpy as np
import xarray as xr


# Default verification-window upper bound (DOY). Used as the sentinel
# placement for both CRPS censoring and the ensemble->deterministic
# projection so the same cell can't be classed as a real onset by one
# panel and "no onset" by another.
DEFAULT_SEASON_END = 220


def ensemble_deterministic(ens: xr.DataArray, *,
                           season_end: int = DEFAULT_SEASON_END) -> xr.DataArray:
    """Collapse an ensemble DOY field to a single deterministic DOY field
    via *sentinel-substituted median*:

    - Replace no-onset members (NaN) with a late sentinel (season_end+1).
    - Take the median across members.
    - If the resulting median lands on/above the sentinel (≥ 50% of
      members saw no onset within the window) mark that cell as no-onset
      again.

    Honest deterministic projection: a cell gets a finite onset DOY iff
    the *majority* of members agree on onset. ``ens.mean(skipna=True)``
    silently drops no-onset members, so a cell where 1 of 51 members
    fires early shows up with that one member's DOY as the "forecast"
    and the deterministic isochrone gets dragged toward early-firing
    outliers. Median-with-sentinel matches the treatment SPS gives
    no-onset members (probability 0).

    ``season_end`` controls the sentinel placement and the "majority
    no-onset" cutoff; it must match the value used elsewhere in the
    pipeline (CRPS sentinel, CORP threshold), or the same cell can be
    classed as a real onset by one panel and "no onset" by another."""
    sentinel = float(season_end) + 1.0
    vals = np.asarray(ens.values, dtype=float)
    vals = np.where(np.isfinite(vals), vals, sentinel)
    median = np.median(vals, axis=0)
    median = np.where(median >= sentinel - 0.5, np.nan, median)
    return xr.DataArray(
        median, dims=("lat", "lon"),
        coords={"lat": ens["lat"].values, "lon": ens["lon"].values},
        name="onset_doy",
        attrs={"summary": "ensemble-median onset with sentinel for no-onset members",
               "season_end": int(season_end)},
    )


def _as_list(a) -> list:
    arr = np.asarray(a, dtype=float)
    return [None if not np.isfinite(x) else float(x) for x in arr.ravel()]


def field_payload(da: xr.DataArray) -> dict:
    return {
        "lat": da["lat"].values.tolist(),
        "lon": da["lon"].values.tolist(),
        "values": [
            [None if not np.isfinite(v) else float(v) for v in row]
            for row in np.asarray(da.values, dtype=float)
        ],
    }


def compute_crps(ens: xr.DataArray, obs: xr.DataArray, *, season_end: int) -> dict:
    """Sentinel-augmented mixed-distribution CRPS.

    Uses the fair (Ferro 2014) finite-ensemble bias correction whenever
    the ensemble has ≥ 2 members; reduces to the raw Hersbach form for a
    single-member (deterministic) forecast where the fair correction is
    undefined.

    Cells where both obs and every ensemble member are missing (e.g.
    ocean cells culled by ``ROMP_LAND_MASK``) would otherwise map both
    sides to the same sentinel and yield a spurious CRPS = 0. We mask
    those cells to NaN in the output so the reported ``mean`` / ``n_finite``
    exclude them rather than treating them as perfect forecasts.
    """
    from momp.metrics.crps import censored_crps_field
    use_fair = "member" in ens.dims and int(ens.sizes["member"]) >= 2
    crps = censored_crps_field(ens, obs, season_end=season_end, fair=use_fair)
    # Identify cells where obs is missing AND every ensemble member is missing.
    # For those, CRPS collapses to 0 (both sides hit the sentinel); that 0 is
    # not a real skill signal, so null it out.
    obs_missing = ~np.isfinite(np.asarray(obs.values, dtype=float))
    ens_vals = np.asarray(ens.values, dtype=float)
    ens_all_missing = (~np.isfinite(ens_vals)).all(axis=0) if ens_vals.ndim == 3 \
        else ~np.isfinite(ens_vals)
    no_data = obs_missing & ens_all_missing
    crps_out = crps.where(~xr.DataArray(no_data, coords=crps.coords, dims=crps.dims))
    v = crps_out.values
    finite_mask = np.isfinite(v)
    return {
        "field": field_payload(crps_out),
        "mean": float(v[finite_mask].mean()) if finite_mask.any() else None,
        "max": float(v[finite_mask].max()) if finite_mask.any() else None,
        "n_finite": int(finite_mask.sum()),
        "fair": bool(use_fair),
        "n_members": int(ens.sizes["member"]) if "member" in ens.dims else 1,
    }


def compute_fss(fcst: xr.DataArray, obs: xr.DataArray, *,
                thresholds, neighborhoods) -> dict:
    from momp.metrics.neighborhood import (
        base_rate, fss, useful_scale_per_threshold,
    )
    out = fss(fcst, obs, thresholds=thresholds, neighborhoods=neighborhoods)
    v = np.asarray(out.values, dtype=float)
    # Per-threshold climatological base rate p(τ), needed for the
    # Roberts-Lean useful-skill threshold downstream. Computed from
    # obs (the authoritative reference), not the forecast.
    rates = [base_rate(obs, float(t)) for t in thresholds]
    us = useful_scale_per_threshold(v, list(neighborhoods), rates)
    return {
        "thresholds": list(thresholds),
        "neighborhoods": list(neighborhoods),
        "fss": [[None if not np.isfinite(x) else float(x) for x in row] for row in v],
        "base_rate": [float(r) for r in rates],
        "useful_scale": [None if not np.isfinite(x) else float(x) for x in us],
    }


def compute_displacement(fcst: xr.DataArray, obs: xr.DataArray, *, thresholds) -> dict:
    from momp.metrics.displacement import displacement_bias_sweep
    ds = displacement_bias_sweep(fcst, obs, thresholds=list(thresholds))
    return {
        "thresholds": list(thresholds),
        "delta_lat_deg": _as_list(ds["delta_lat_deg"].values),
        "delta_lon_deg": _as_list(ds["delta_lon_deg"].values),
        "great_circle_km": _as_list(ds["great_circle_km"].values),
        "area_bias_fraction": _as_list(ds["area_bias_fraction"].values),
    }


def compute_progression(fcst: xr.DataArray, ens: xr.DataArray | None,
                        obs: xr.DataArray, *, days) -> dict:
    from momp.metrics.progression import (
        integrated_onset_error, peak_doy, spatial_probability_score,
    )
    ioe = integrated_onset_error(fcst, obs, days=days)
    # Match the shape of aggregate_progression so single-year and multi-year
    # responses share a single schema: every season scalar has matching
    # _q25/_q75 keys (None in single-year).
    out = {
        "days": list(days),
        "ioe_km2": _as_list(ioe["ioe_km2"].values),
        "ioe_km2_q25": None, "ioe_km2_q75": None,
        "ioe_km2_ci_lo": None, "ioe_km2_ci_hi": None,
        "extent_km2": _as_list(ioe["extent_km2"].values),
        "extent_km2_q25": None, "extent_km2_q75": None,
        "extent_km2_ci_lo": None, "extent_km2_ci_hi": None,
        "misplacement_km2": _as_list(ioe["misplacement_km2"].values),
        "misplacement_km2_q25": None, "misplacement_km2_q75": None,
        "misplacement_km2_ci_lo": None, "misplacement_km2_ci_hi": None,
        "season": {
            "n_years": 1,
            "ioe_km2_day": float(ioe["ioe_season_km2_day"]),
            "ioe_km2_day_q25": None, "ioe_km2_day_q75": None,
            "ioe_km2_day_ci_lo": None, "ioe_km2_day_ci_hi": None,
            "extent_km2_day": float(ioe["extent_season_km2_day"]),
            "extent_km2_day_q25": None, "extent_km2_day_q75": None,
            "extent_km2_day_ci_lo": None, "extent_km2_day_ci_hi": None,
            "misplacement_km2_day": float(ioe["misplacement_season_km2_day"]),
            "misplacement_km2_day_q25": None, "misplacement_km2_day_q75": None,
            "misplacement_km2_day_ci_lo": None, "misplacement_km2_day_ci_hi": None,
        },
    }
    if ens is not None:
        sps = spatial_probability_score(ens, obs, days=days)
    else:
        # For deterministic forecasts, wrap the det field as a 1-member
        # ensemble and compute SPS anyway. By construction (see
        # test_sps_reduces_to_ioe_for_single_deterministic_member) this
        # gives SPS == IOE, keeping the response schema uniform across
        # det and ensemble models so the frontend doesn't need two shapes.
        det_ens = fcst.expand_dims({"member": [0]}).transpose("member", "lat", "lon")
        sps = spatial_probability_score(det_ens, obs, days=days)
    out["sps_km2"] = _as_list(sps["sps_km2"].values)
    out["sps_km2_q25"] = None
    out["sps_km2_q75"] = None
    out["sps_km2_ci_lo"] = None
    out["sps_km2_ci_hi"] = None
    out["season"]["sps_km2_day"] = float(sps["sps_season_km2_day"])
    out["season"]["sps_km2_day_q25"] = None
    out["season"]["sps_km2_day_q75"] = None
    out["season"]["sps_km2_day_ci_lo"] = None
    out["season"]["sps_km2_day_ci_hi"] = None

    # Misplacement fraction at the season level: well-defined for a
    # single year (no aggregation needed). 0 = pure extent error, 1 =
    # pure misplacement, NaN if IOE_season is zero (perfect forecast).
    ioe_s = out["season"]["ioe_km2_day"]
    misp_s = out["season"]["misplacement_km2_day"]
    out["season"]["misp_frac"] = (
        float(misp_s) / float(ioe_s)
        if ioe_s is not None and ioe_s > 0
        and misp_s is not None and np.isfinite(misp_s) and np.isfinite(ioe_s)
        else None
    )
    out["season"]["misp_frac_ci_lo"] = None
    out["season"]["misp_frac_ci_hi"] = None

    # Peak-DOY diagnostic: the DOY at which IOE / SPS is maximised.
    # Captures *when* the model's front is most wrong (lag bias is
    # invisible in the season-integrated headline). Single-year payload
    # carries no CI; aggregator fills CI keys for multi-year.
    ioe_peak_d, ioe_peak_v = peak_doy(out["ioe_km2"], list(days))
    sps_peak_d, sps_peak_v = peak_doy(out["sps_km2"], list(days))
    out["peak"] = {
        "ioe_doy":   _none_if_nan(ioe_peak_d),
        "ioe_value": _none_if_nan(ioe_peak_v),
        "ioe_doy_ci_lo": None, "ioe_doy_ci_hi": None,
        "sps_doy":   _none_if_nan(sps_peak_d),
        "sps_value": _none_if_nan(sps_peak_v),
        "sps_doy_ci_lo": None, "sps_doy_ci_hi": None,
    }
    return out


def _none_if_nan(x):
    fx = float(x)
    return None if not np.isfinite(fx) else fx


def compute_isochrones(fcst: xr.DataArray, obs: xr.DataArray, *, days) -> dict:
    from momp.graphics.isochrone import (
        extract_isochrone, isochrone_distance_sweep,
    )
    entries = []
    for d in days:
        f_segs = extract_isochrone(fcst, float(d))
        o_segs = extract_isochrone(obs, float(d))
        entries.append({
            "day": int(d),
            "forecast": [s.tolist() for s in f_segs],
            "observed": [s.tolist() for s in o_segs],
        })
    sweep = isochrone_distance_sweep(fcst, obs, days=list(days))
    return {
        "isochrones": entries,
        "days": list(days),
        "hausdorff_km": _as_list(sweep["hausdorff_km"].values),
        "frechet_km": _as_list(sweep["frechet_km"].values),
        "n_segments_fcst": [int(v) for v in sweep["n_segments_fcst"].values],
        "n_segments_obs": [int(v) for v in sweep["n_segments_obs"].values],
    }


def moran_i_2d(field) -> float:
    """Queen-4 (rook) spatial autocorrelation on a 2-D field, NaN-safe.

    Returns Moran's I in [−1, +1]; higher ⇒ stronger positive spatial
    autocorrelation. Used to derive an effective sample size
    ``n_eff = n · (1 − I) / (1 + I)`` (Clifford-Richardson 1989 /
    Cressie 1993 form; see ``effective_sample_size``) for pooled
    statistics where neighbouring grid cells are not independent.

    NaN cells are excluded from the mean, deviations, and from any
    neighbour pair. Returns NaN if the field has fewer than 4 finite
    cells, zero variance, or no valid neighbour pairs.
    """
    import numpy as np
    v = np.asarray(field, dtype=float)
    if v.ndim != 2:
        raise ValueError("moran_i_2d expects a 2-D array")
    finite = np.isfinite(v)
    if int(finite.sum()) < 4:
        return float("nan")

    mean = float(v[finite].mean())
    dev = np.where(finite, v - mean, 0.0)
    denom = float((dev[finite] ** 2).sum())
    if denom == 0:
        return float("nan")

    # Rook contiguity: each cell pairs with its (N,S,E,W) neighbours.
    # Vertical pair set (i, i+1 along rows) + horizontal pair set.
    cross = 0.0
    W = 0
    for a_idx, b_idx in (
        (np.s_[:-1, :], np.s_[1:, :]),   # vertical
        (np.s_[:, :-1], np.s_[:, 1:]),   # horizontal
    ):
        both = finite[a_idx] & finite[b_idx]
        cross += 2.0 * float((dev[a_idx] * dev[b_idx])[both].sum())
        W += 2 * int(both.sum())
    if W == 0:
        return float("nan")
    n = int(finite.sum())
    return (n / W) * cross / denom


def effective_sample_size(n: int, moran_i: float) -> float:
    """Effective-n correction for spatially autocorrelated data.

    Returns ``n`` unchanged for ``I ≤ 0``; otherwise
    ``n · (1 − I) / (1 + I)``, floored at 1.

    The formula is the Clifford-Richardson (1989) / Cressie (1993, §1.4)
    AR(1)-style variance-inflation approximation, applied with the
    Moran's-I estimate of spatial autocorrelation. It is *not* the more
    elaborate Dutilleul (1993) modified-t correction (which is a
    correlation-test-specific construction). For our use — reporting an
    honest effective-n alongside the raw cell count — the simpler
    variance-inflation form is the standard first-order correction."""
    if moran_i is None or not (moran_i == moran_i):  # NaN check
        return float(n)
    if moran_i <= 0:
        return float(n)
    return max(1.0, float(n) * (1.0 - moran_i) / (1.0 + moran_i))


def corp_inputs(ens: xr.DataArray | None, fcst: xr.DataArray,
                obs: xr.DataArray, *, tau: int, season_end: int):
    """Return raw (forecast probability, observed binary) arrays at threshold τ
    so multiple years can be pooled before the CORP decomposition."""
    if ens is not None:
        m = ens.values
        p = np.where(np.isnan(m) | (m > season_end), 0.0,
                     (m <= tau).astype(float)).mean(axis=0)
    else:
        f = fcst.values
        p = np.where(np.isnan(f) | (f > season_end), 0.0,
                     (f <= tau).astype(float))
    y = np.where(np.isfinite(obs.values) & (obs.values <= tau), 1.0, 0.0)
    return p.ravel(), y.ravel()


def compute_corp_pooled(p: np.ndarray, y: np.ndarray, *, tau: int) -> dict:
    """CORP decomposition on already-pooled (p, y) arrays."""
    from momp.graphics.corp_reliability import (
        corp_decompose_brier, _consolidate_curve,
    )
    decomp = corp_decompose_brier(p, y)
    f_rep, c_rep = _consolidate_curve(decomp.forecast_prob, decomp.calibrated_y)
    return {
        "tau": int(tau),
        "mean_score": float(decomp.mean_score),
        "mcb": float(decomp.mcb),
        "dsc": float(decomp.dsc),
        "unc": float(decomp.unc),
        "identity_residual": float(
            (decomp.mcb - decomp.dsc + decomp.unc) - decomp.mean_score
        ),
        "n": int(decomp.n),
        "curve": {
            "forecast_prob": [float(x) for x in f_rep],
            "calibrated_prob": [float(x) for x in c_rep],
        },
        "forecast_prob_histogram": _as_list(decomp.forecast_prob),
    }


def compute_corp(ens: xr.DataArray | None, fcst: xr.DataArray,
                 obs: xr.DataArray, *, tau: int, season_end: int) -> dict:
    """Build a probability of 'onset <= tau' and decompose its Brier score."""
    from momp.graphics.corp_reliability import (
        corp_decompose_brier, _consolidate_curve,
    )
    if ens is not None:
        m = ens.values
        p = np.where(np.isnan(m) | (m > season_end), 0.0,
                     (m <= tau).astype(float)).mean(axis=0)
    else:
        f = fcst.values
        p = np.where(np.isnan(f) | (f > season_end), 0.0,
                     (f <= tau).astype(float))
    y = np.where(np.isfinite(obs.values) & (obs.values <= tau), 1.0, 0.0)
    decomp = corp_decompose_brier(p.ravel(), y.ravel())
    f_rep, c_rep = _consolidate_curve(decomp.forecast_prob, decomp.calibrated_y)
    return {
        "tau": int(tau),
        "mean_score": float(decomp.mean_score),
        "mcb": float(decomp.mcb),
        "dsc": float(decomp.dsc),
        "unc": float(decomp.unc),
        "identity_residual": float(
            (decomp.mcb - decomp.dsc + decomp.unc) - decomp.mean_score
        ),
        "n": int(decomp.n),
        "curve": {
            "forecast_prob": [float(x) for x in f_rep],
            "calibrated_prob": [float(x) for x in c_rep],
        },
        "forecast_prob_histogram": _as_list(decomp.forecast_prob),
    }
