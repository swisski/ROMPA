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


def _downsample_2d(arr: np.ndarray, max_cells: int = 80) -> np.ndarray:
    """Stride-downsample a 2-D array so the longer axis fits in ``max_cells``.

    For payloads where the consumer is a JS heatmap, sending the full
    high-res grid wastes bandwidth and is invisible at typical viewport
    sizes. Returns the array unchanged if both axes already fit."""
    if arr.ndim != 2:
        return arr
    h, w = arr.shape
    step = max(1, max(h, w) // max_cells)
    return arr[::step, ::step] if step > 1 else arr


def field_2d_payload(values: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                     max_cells: int = 80) -> dict:
    """JSON-friendly 2-D field payload, stride-downsampled for transit."""
    vals = _downsample_2d(np.asarray(values, dtype=float), max_cells=max_cells)
    lat_d = lat[::max(1, max(values.shape) // max_cells)]
    lon_d = lon[::max(1, max(values.shape) // max_cells)]
    # The stride above can be off-by-one against the data; recompute properly.
    step = max(1, max(values.shape) // max_cells)
    lat_d = np.asarray(lat[::step], dtype=float)
    lon_d = np.asarray(lon[::step], dtype=float)
    return {
        "lat": lat_d.tolist(),
        "lon": lon_d.tolist(),
        "values": [
            [None if not np.isfinite(v) else float(v) for v in row]
            for row in vals
        ],
    }


def field_payload(da: xr.DataArray) -> dict:
    return {
        "lat": da["lat"].values.tolist(),
        "lon": da["lon"].values.tolist(),
        "values": [
            [None if not np.isfinite(v) else float(v) for v in row]
            for row in np.asarray(da.values, dtype=float)
        ],
    }


def upsampled_field_payload(da: xr.DataArray) -> dict:
    """Display-only bilinear upsample of a 2-D field, then serialize.

    Same Stage-1 smoothing the CRA panel uses: ~5× denser per axis on
    coarse AICE grids (4° AIFS = 8×9, 2° ensembles = 16×17). NaN cells
    (e.g. ocean under a land mask) are treated as a validity mask and
    re-applied after interpolation, so the no-data structure is
    preserved. Use this for fields the frontend renders as heatmaps
    or computes contours from — onset DOY, rainfall, etc. — where the
    visualization benefits from a smooth surface but the underlying
    metric scoring still happens on native cells.
    """
    vals = np.asarray(da.values, dtype=float)
    if vals.ndim != 2:
        return field_payload(da)
    lat = np.asarray(da["lat"].values, dtype=float)
    lon = np.asarray(da["lon"].values, dtype=float)
    mask = np.isfinite(vals)
    if not mask.any():
        return field_payload(da)
    lat_d, lon_d, dense = _upsample_for_display(vals, lat, lon, mask=mask)
    return {
        "lat": lat_d.tolist(),
        "lon": lon_d.tolist(),
        "values": [
            [None if not np.isfinite(v) else float(v) for v in row]
            for row in dense
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


def _align_forecast_to_obs_grid(fcst_vals: np.ndarray, fcst_lat: np.ndarray, fcst_lon: np.ndarray,
                                obs_lat: np.ndarray, obs_lon: np.ndarray) -> np.ndarray:
    """Nearest-neighbor regrid a 2-D forecast accumulation onto the obs grid."""
    if (fcst_lat.shape == obs_lat.shape and np.allclose(fcst_lat, obs_lat)
            and fcst_lon.shape == obs_lon.shape and np.allclose(fcst_lon, obs_lon)):
        return fcst_vals
    da = xr.DataArray(
        fcst_vals, dims=("lat", "lon"),
        coords={"lat": fcst_lat, "lon": fcst_lon},
    )
    target = xr.DataArray(
        np.zeros((obs_lat.size, obs_lon.size)),
        dims=("lat", "lon"), coords={"lat": obs_lat, "lon": obs_lon},
    )
    return np.asarray(da.interp_like(target, method="nearest").values, dtype=float)


def _upsample_for_display(values: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                          *, mask: np.ndarray | None = None,
                          target_cells: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinear-upsample a 2-D field for visualization only.

    At ``target_cells=16``, a 2° native grid (16×17 over India) passes
    the "already big enough" gate and is shipped as-is — Plotly's
    ``zsmooth='best'`` renderer-side bilinear is enough visual
    smoothness without inflating the payload. A 4° native grid (8×9,
    e.g. AIFS at the old resolution) still gets a 2× upsample so it's
    not a chunky pixel block.

    Coarse grids (4° AIFS = 8×9 cells over India, 2° ensembles = 16×17)
    render as chunky pixel blocks and produce stair-stepped contour
    lines. This helper interpolates the field onto a denser lat/lon
    mesh so heatmaps and contours are visually smooth — the CRA metric
    still scores on the native coarse cells, only the payload sent to
    Plotly is upsampled.

    If ``mask`` is given it is the country/display mask in the SAME
    native shape as ``values``. The field is interpolated with NaN
    cells filled to zero (so bilinear interp doesn't propagate NaN
    across edges and eat the country interior); the mask is upsampled
    separately via nearest-neighbor and reapplied at the end. Without
    this two-pass approach, every cell of the dense grid that touches
    an outside-country corner would go NaN and India would render
    nearly empty.

    No-op for grids already at or above ``target_cells`` per axis.
    Returns ``(lat_dense, lon_dense, values_dense)``."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    vals = np.asarray(values, dtype=float)
    ny, nx = vals.shape
    if ny >= target_cells and nx >= target_cells:
        return lat, lon, vals
    factor_y = max(1, int(round(target_cells / max(ny, 1))))
    factor_x = max(1, int(round(target_cells / max(nx, 1))))
    factor = min(factor_y, factor_x)
    if factor <= 1:
        return lat, lon, vals
    new_ny = (ny - 1) * factor + 1
    new_nx = (nx - 1) * factor + 1
    lat_d = np.linspace(lat[0], lat[-1], new_ny)
    lon_d = np.linspace(lon[0], lon[-1], new_nx)

    # Field: fill NaN to 0 so bilinear interp doesn't eat the interior.
    vals_filled = np.where(np.isnan(vals), 0.0, vals)
    da = xr.DataArray(
        vals_filled, dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )
    dense = np.asarray(da.interp(lat=lat_d, lon=lon_d, method="linear").values, dtype=float)

    # Mask: linearly interpolate (treats the binary mask as a fractional
    # "inside-ness") and threshold at 0.5 — gives a smooth country
    # boundary that follows the bilinear midpoints between native cells
    # rather than the native cell grid. Nearest-neighbor here would
    # leave the country edge stair-stepped at native resolution even
    # though the heatmap interior is smooth.
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=float)
        mda = xr.DataArray(
            mask_arr, dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
        )
        dense_mask = np.asarray(
            mda.interp(lat=lat_d, lon=lon_d, method="linear").values
        ) >= 0.5
        dense = np.where(dense_mask, dense, np.nan)
    return lat_d, lon_d, dense


def _country_display_mask(country: str | None, obs_lat: np.ndarray, obs_lon: np.ndarray):
    """Boolean (lat, lon) mask of cells *inside* ``country``, or None.

    Used by the CRA panel to blank cells outside the analysis region for
    visualization (the decomposition itself runs against the full grid).
    Errors fall back to ``None`` so a missing geopandas / regionmask
    install doesn't break the endpoint."""
    if not country:
        return None
    try:
        from momp.utils.land_mask import country_mask
        da = xr.DataArray(
            np.zeros((obs_lat.size, obs_lon.size)),
            dims=("lat", "lon"),
            coords={"lat": obs_lat, "lon": obs_lon},
        )
        return np.asarray(country_mask(da, country).values, dtype=bool)
    except Exception:
        return None


def compute_cra(fcst_vals: np.ndarray, fcst_lat: np.ndarray, fcst_lon: np.ndarray,
                obs_vals: np.ndarray, obs_lat: np.ndarray, obs_lon: np.ndarray,
                *, case: str, threshold: float, max_shift: float,
                subgrid_factor: int = 8,
                include_fields: bool = True,
                display_country: str | None = None,
                max_payload_cells: int = 240) -> dict:
    """Run CRA decomposition and serialize the result.

    The forecast is regridded onto the obs grid first; both fields then
    enter ``cra_decomposition`` as same-shape numpy arrays. Field payloads
    (obs / fcst / shifted) are stride-downsampled to ``max_payload_cells``
    per axis (default 240) so the response stays reasonable for
    high-resolution grids.

    ``display_country`` sets *both* the display mask (cells outside the
    country are NaN'd in the payload so the heatmap shows the boundary
    cleanly) and the ``verification_mask`` passed to
    ``cra_decomposition`` — the metric scores MSE only over cells inside
    the country, matching the demo's "India CRA objective". Without this
    the optimizer can shift the forecast off the analysis region and
    score artificially low because the score mask shrinks. Set
    ``display_country=None`` to score on the full grid.
    """
    from momp.metrics.cra import cra_decomposition

    fcst_on_obs = _align_forecast_to_obs_grid(
        fcst_vals, fcst_lat, fcst_lon, obs_lat, obs_lon,
    )
    obs_2d = np.asarray(obs_vals, dtype=float)
    fcst_2d = np.asarray(fcst_on_obs, dtype=float)

    # Same region mask is used for the verification objective AND the
    # display mask, so the maps and the scores tell a consistent story.
    region_mask = _country_display_mask(display_country, obs_lat, obs_lon)

    result, shifted, _cra_mask = cra_decomposition(
        case=case,
        obs=obs_2d,
        fcst=fcst_2d,
        threshold=float(threshold),
        max_shift=float(max_shift),
        verification_mask=region_mask,
        subgrid_factor=int(subgrid_factor),
    )

    payload = {
        "case": result.case,
        # Shift values in NATIVE CELL UNITS — floats now that
        # subgrid_factor > 1 enables fractional-cell shifts (0.125 cells
        # = 0.25° on a 2° native grid).
        "corrective_shift": {
            "dx": float(result.corrective_shift_dx),
            "dy": float(result.corrective_shift_dy),
        },
        "diagnosed_forecast_error": {
            "dx": float(result.diagnosed_forecast_error_dx),
            "dy": float(result.diagnosed_forecast_error_dy),
        },
        "subgrid_factor": int(subgrid_factor),
        "mse": {
            "total": _none_if_nan(result.mse_total),
            "shifted": _none_if_nan(result.mse_shifted),
            "displacement": _none_if_nan(result.mse_displacement),
            "volume": _none_if_nan(result.mse_volume),
            "pattern": _none_if_nan(result.mse_pattern),
        },
        "pct": {
            "displacement": _none_if_nan(result.pct_displacement),
            "volume": _none_if_nan(result.pct_volume),
            "pattern": _none_if_nan(result.pct_pattern),
        },
        "n_obs_objects": int(result.n_obs_objects),
        "n_fcst_objects": int(result.n_fcst_objects),
        "mean_obs": _none_if_nan(result.mean_obs),
        "mean_fcst_shifted": _none_if_nan(result.mean_fcst_shifted),
        "peak_obs": _none_if_nan(result.peak_obs),
        "peak_fcst_shifted": _none_if_nan(result.peak_fcst_shifted),
        "spatial_corr": {
            "original": _none_if_nan(result.spatial_corr_original),
            "shifted": _none_if_nan(result.spatial_corr_shifted),
        },
        "threshold": float(threshold),
        "max_shift": int(max_shift),
    }
    if include_fields:
        # For centroid math we want the masked field (rainfall outside
        # the country shouldn't pull the centroid). For display we want
        # the unmasked rainfall fed into the upsampler with the mask
        # re-applied AFTER upsampling — bilinear interp on a NaN-masked
        # field propagates NaN across the country interior on coarse
        # grids and renders India nearly empty.
        payload["display_country"] = display_country if region_mask is not None else None
        if region_mask is not None:
            obs_centroid_field     = np.where(region_mask, obs_2d, np.nan)
            fcst_centroid_field    = np.where(region_mask, fcst_2d, np.nan)
            shifted_centroid_field = np.where(region_mask, shifted, np.nan)
        else:
            obs_centroid_field     = obs_2d
            fcst_centroid_field    = fcst_2d
            shifted_centroid_field = shifted
        # Centroids on the *native* grid (pre-upsample) so lat/lon arrays
        # still match the field shape.
        payload["centroids"] = {
            "obs":            _masked_centroid(obs_centroid_field,     obs_lat, obs_lon),
            "fcst":           _masked_centroid(fcst_centroid_field,    obs_lat, obs_lon),
            "shifted":        _masked_centroid(shifted_centroid_field, obs_lat, obs_lon),
            "fcst_full":      _masked_centroid(fcst_2d,                obs_lat, obs_lon),
            "shifted_full":   _masked_centroid(shifted,                obs_lat, obs_lon),
        }
        # Display-only upsample for coarse AICE grids (4° AIFS = 8×9
        # cells, 2° ensembles = 16×17). Bilinear interpolation onto a
        # ~5× finer mesh so heatmap AND contour render smoothly; CRA's
        # actual scoring still happens on the native coarse cells, this
        # only inflates the bytes Plotly draws with. ``target_cells=40``
        # picks the smoothing factor to be visible but not deceptive.
        #
        # Two flavours of field payload:
        #   - ``fields``      : country-masked (used by the existing 3-panel
        #                        obs/fcst/shifted maps).
        #   - ``fields_full`` : NOT masked (used by the shift panel — the
        #                        forecast may sit over water before the
        #                        shift and only land on India after, so
        #                        cropping to India hides what's happening).
        disp_lat, disp_lon, obs_dense     = _upsample_for_display(obs_2d,  obs_lat, obs_lon, mask=region_mask)
        _,        _,        fcst_dense    = _upsample_for_display(fcst_2d, obs_lat, obs_lon, mask=region_mask)
        _,        _,        shifted_dense = _upsample_for_display(shifted, obs_lat, obs_lon, mask=region_mask)
        _,        _,        fcst_full_dense    = _upsample_for_display(fcst_2d, obs_lat, obs_lon)
        _,        _,        shifted_full_dense = _upsample_for_display(shifted, obs_lat, obs_lon)
        payload["fields"] = {
            "obs":     field_2d_payload(obs_dense,     disp_lat, disp_lon, max_cells=max_payload_cells),
            "fcst":    field_2d_payload(fcst_dense,    disp_lat, disp_lon, max_cells=max_payload_cells),
            "shifted": field_2d_payload(shifted_dense, disp_lat, disp_lon, max_cells=max_payload_cells),
        }
        payload["fields_full"] = {
            "fcst":    field_2d_payload(fcst_full_dense,    disp_lat, disp_lon, max_cells=max_payload_cells),
            "shifted": field_2d_payload(shifted_full_dense, disp_lat, disp_lon, max_cells=max_payload_cells),
        }
        # Country outline as raw polygon vertices, NOT as a mask
        # contour. The mask-contour approach traced the bilinearly-
        # upsampled isocontour of a binary mask, which produced
        # awkward shapes at coarse resolution (Northeast India
        # protrusion etc). Using the geographic polygon coordinates
        # gives a clean, recognizable country border at full
        # natural-earth resolution. Mainland only — islands are
        # dropped since they're irrelevant for the CRA shift story.
        if display_country:
            try:
                from momp.utils.land_mask import country_polygon_coords
                payload["country_outline"] = country_polygon_coords(display_country)
            except (ValueError, ImportError):
                pass
        # Useful for the frontend to compute the absolute shift in lat/lon
        # AND to display the native resolution of the source data so the
        # bilinear smoothing isn't read as real spatial detail.
        payload["grid_spacing"] = {
            "dlat": float(abs(np.median(np.diff(obs_lat)))) if obs_lat.size > 1 else None,
            "dlon": float(abs(np.median(np.diff(obs_lon)))) if obs_lon.size > 1 else None,
            "native_ny": int(obs_2d.shape[0]),
            "native_nx": int(obs_2d.shape[1]),
            "display_ny": int(disp_lat.size),
            "display_nx": int(disp_lon.size),
        }
    return payload


def _masked_centroid(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    """Return ``{lat, lon}`` of the cell-weighted centroid of a 2-D field.

    NaN cells and cells <= 0 are ignored. ``None`` if every contributing
    cell is masked out — happens when the display country mask covers no
    actual data (e.g. obs outside the model grid)."""
    vals = np.asarray(arr, dtype=float)
    finite = np.isfinite(vals) & (vals > 0)
    if not finite.any():
        return None
    weights = np.where(finite, vals, 0.0)
    total = weights.sum()
    if total <= 0:
        return None
    lat_c = float((weights.sum(axis=1) * lat).sum() / total)
    lon_c = float((weights.sum(axis=0) * lon).sum() / total)
    return {"lat": lat_c, "lon": lon_c}
