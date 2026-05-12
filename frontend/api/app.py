"""ROMP metrics frontend — FastAPI app.

Thin routing layer. Data discovery lives in ``catalog.py``, onset-DOY
construction in ``onset.py``, metric serialization in ``metrics.py``.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import xarray as xr
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from momp.utils.land_mask import country_mask

from . import aggregate as AGG
from . import metrics as M
from .catalog import load_catalog, model_by_key, obs_source, shared_years
from .onset import (
    OnsetParams, Region,
    available_inits, best_init_for, get_forecast_onset, get_obs_onset,
    onset_range,
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_catalog()  # warm the catalog cache
    yield


app = FastAPI(title="ROMP metrics API", version="0.4.1-cra-india-shift", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.exception_handler(FileNotFoundError)
async def _missing_file(request: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(KeyError)
async def _bad_key(request: Request, exc: KeyError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


def _parse_params(
    wet_init: float, wet_spell: int, wet_threshold: float,
    dry_spell: int, dry_threshold: float, dry_extent: int,
) -> OnsetParams:
    return OnsetParams(
        wet_init=wet_init, wet_spell=wet_spell, wet_threshold=wet_threshold,
        dry_spell=dry_spell, dry_threshold=dry_threshold, dry_extent=dry_extent,
    )


def _parse_region(
    lat_min: float | None, lat_max: float | None,
    lon_min: float | None, lon_max: float | None,
) -> Region:
    return Region(lat_min=lat_min, lat_max=lat_max,
                  lon_min=lon_min, lon_max=lon_max)


OnsetQuery = dict(
    wet_init=(1.0, "Wet-day rainfall threshold (mm)"),
    wet_spell=(3, "Min consecutive wet days"),
    wet_threshold=(20.0, "Cumulative wet-spell rainfall (mm)"),
    dry_spell=(7, "Dry-spell length (days)"),
    dry_threshold=(1.0, "Dry-day rainfall threshold (mm)"),
    dry_extent=(0, "Search extent after wet spell"),
)


def onset_deps(
    wet_init: float = Query(OnsetQuery["wet_init"][0]),
    wet_spell: int = Query(OnsetQuery["wet_spell"][0]),
    wet_threshold: float = Query(OnsetQuery["wet_threshold"][0]),
    dry_spell: int = Query(OnsetQuery["dry_spell"][0]),
    dry_threshold: float = Query(OnsetQuery["dry_threshold"][0]),
    dry_extent: int = Query(OnsetQuery["dry_extent"][0]),
) -> OnsetParams:
    return _parse_params(wet_init, wet_spell, wet_threshold,
                         dry_spell, dry_threshold, dry_extent)


def region_deps(
    lat_min: float | None = Query(None),
    lat_max: float | None = Query(None),
    lon_min: float | None = Query(None),
    lon_max: float | None = Query(None),
) -> Region:
    return _parse_region(lat_min, lat_max, lon_min, lon_max)


# ---------------------------------------------------------------------------
# catalog + config
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    import os
    return {
        "status": "ok",
        "version": app.version,
        "pid": os.getpid(),
        "data_root": os.environ.get("ROMP_DATA_ROOT", ""),
        "land_mask": os.environ.get("ROMP_LAND_MASK", "") or None,
    }


@app.post("/api/cache/clear")
def clear_cache():
    """Clear all in-process onset / catalog / land-mask caches WITHOUT
    restarting uvicorn. Useful after code changes when you'd otherwise
    keep seeing stale detection output. Returns the number of entries
    dropped from each cache for visibility."""
    from .onset import _obs_cache, _fcst_cache, _init_cache
    from . import rainfall as R
    dropped = {
        "obs_onset": len(_obs_cache),
        "fcst_onset": len(_fcst_cache),
        "init_list": len(_init_cache),
        "land_mask": len(_LAND_MASK_CACHE),
        "rainfall_accum": len(R._fcst_accum_cache) + len(R._obs_accum_cache),
    }
    _obs_cache.clear()
    _fcst_cache.clear()
    _init_cache.clear()
    _LAND_MASK_CACHE.clear()
    R.clear_caches()
    load_catalog.cache_clear()
    return {"ok": True, "cleared": dropped}


# Disable HTTP caching on static files and API responses to prevent the
# browser / any intermediate cache from re-serving a stale frontend after
# a code change.
@app.middleware("http")
async def _no_store(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.get("/api/catalog")
def catalog():
    import os
    cat = load_catalog()
    return {
        "root": cat["root"],
        "models": [m.to_json() for m in cat["models"]],
        "obs": cat["obs"].to_json() if cat["obs"] else None,
        "shared_years": list(shared_years()),
        "onset_defaults": {k: v[0] for k, v in OnsetQuery.items()},
        "onset_docs": {k: v[1] for k, v in OnsetQuery.items()},
        "land_mask": os.environ.get("ROMP_LAND_MASK", "") or None,
    }


@app.get("/api/inits")
def inits(model: str, year: int):
    m = model_by_key(model)
    times = available_inits(m, year)
    return {
        "model": m.key, "year": year, "n": len(times),
        "inits": [t.isoformat() for t in times],
    }


def _resolve_init(model_key: str, year: int, init: int | Literal["auto"] | None,
                  params: OnsetParams, obs_lo: float, obs_hi: float) -> int:
    """Resolve ``init`` param (int index, 'auto', or None→auto)."""
    if init is None or init == "auto":
        return best_init_for(model_by_key(model_key), obs_lo, obs_hi, params, year)
    try:
        return int(init)
    except (TypeError, ValueError):
        raise HTTPException(400, f"invalid init '{init}'")


def _align_fcst_to(fcst, obs):
    """Conform the forecast to the obs grid when they differ. Obs is the
    authoritative measurement; downsampling the model to the obs grid
    avoids inflating area-weighted metrics by N where N is the upsample
    ratio. For ensembles, we keep the member dim and interp per-member."""
    import numpy as np
    if (fcst.sizes.get("lat") == obs.sizes.get("lat") and
            fcst.sizes.get("lon") == obs.sizes.get("lon") and
            np.array_equal(fcst["lat"].values, obs["lat"].values) and
            np.array_equal(fcst["lon"].values, obs["lon"].values)):
        return fcst
    # interp_like preserves extra dims (including 'member'), using nearest.
    return fcst.interp_like(obs, method="nearest")


_LAND_MASK_CACHE: dict = {}


def _land_mask_for(da):
    """Return a (lat, lon) bool DataArray for the country named in
    ``ROMP_LAND_MASK`` (e.g. ``India``), or ``None`` if the env var is unset
    or the lookup fails. Cached per (region, grid) since the regionmask
    rasterisation is the dominant cost on repeat requests."""
    import os
    region = os.environ.get("ROMP_LAND_MASK", "").strip()
    if not region:
        return None
    lat_key = tuple(da["lat"].values.round(4).tolist())
    lon_key = tuple(da["lon"].values.round(4).tolist())
    key = (region, lat_key, lon_key)
    if key in _LAND_MASK_CACHE:
        return _LAND_MASK_CACHE[key]
    try:
        mask = country_mask(da, region)
    except Exception as exc:  # fall back to no mask on failure
        import sys
        print(f"[warn] land mask for {region!r} failed: {exc}", file=sys.stderr)
        mask = None
    _LAND_MASK_CACHE[key] = mask
    return mask


def _apply_land_mask(da):
    mask = _land_mask_for(da)
    if mask is None:
        return da
    return da.where(mask)


def _ensemble_deterministic(ens, season_end: int = 220):
    """Collapse an ensemble DOY field to a single deterministic DOY field
    for IOE / isochrones, using a *sentinel-substituted median*:

    - Replace no-onset members (NaN) with a late sentinel (season_end+1).
    - Take the median across members.
    - If the resulting median lands on/above the sentinel (⇒ ≥50% of
      members saw no onset), mark that cell as no-onset (NaN) again.

    This is the honest deterministic projection of a probabilistic forecast:
    a cell gets a finite onset DOY iff the *majority* of members agree on
    onset. Using ``ens.mean(skipna=True)`` silently excludes no-onset
    members, so a cell where only 1 of 51 members fires early shows up
    with that one member's DOY as the "forecast" — dramatically biasing
    isochrones toward early-firing outliers. Median-with-sentinel matches
    the treatment SPS gives no-onset members (probability 0)."""
    import numpy as np
    sentinel = float(season_end) + 1.0
    vals = np.asarray(ens.values, dtype=float)
    vals = np.where(np.isfinite(vals), vals, sentinel)
    median = np.median(vals, axis=0)
    # A cell whose median is the sentinel means ≥50% of members saw no
    # onset ⇒ treat as no onset.
    median = np.where(median >= sentinel - 0.5, np.nan, median)
    return xr.DataArray(
        median, dims=("lat", "lon"),
        coords={"lat": ens["lat"].values, "lon": ens["lon"].values},
        name="onset_doy",
        attrs={"summary": "ensemble-median onset with sentinel for no-onset members"},
    )


def _fields_for(model_key: str, year: int, init: int | str | None,
                params: OnsetParams, region: Region):
    obs = get_obs_onset(year, params)
    obs_lo, obs_hi = onset_range(obs)
    idx = _resolve_init(model_key, year, init, params, obs_lo, obs_hi)
    fcst = get_forecast_onset(model_key, year, idx, params)
    # Conform forecast -> obs (obs is the authoritative grid).
    fcst = _align_fcst_to(fcst, obs)
    # Optional land-only masking: cells outside the named country go NaN.
    obs = _apply_land_mask(obs)
    fcst = _apply_land_mask(fcst)
    if not region.is_empty():
        obs = region.crop(obs)
        fcst = region.crop(fcst)
    ens = fcst if "member" in fcst.dims else None
    det = _ensemble_deterministic(ens) if ens is not None else fcst
    return {"obs": obs, "fcst_det": det, "ens": ens, "init_idx": idx}


# ---------------------------------------------------------------------------
# state + metrics
# ---------------------------------------------------------------------------
@app.get("/api/state")
def state(
    model: str, year: int,
    init: str | None = None,
    params: OnsetParams = Depends(onset_deps),
    region: Region = Depends(region_deps),
):
    """Baseline onset fields + suggested isochrone days for a config."""
    bundle = _fields_for(model, year, init, params, region)
    obs_rng = onset_range(bundle["obs"])
    fcst_rng = onset_range(bundle["fcst_det"])
    ens_rng = onset_range(bundle["ens"]) if bundle["ens"] is not None else (None, None)

    lo = max(obs_rng[0], fcst_rng[0]) + 2
    hi = min(obs_rng[1], fcst_rng[1]) - 2
    import numpy as np
    iso_days = [int(round(x)) for x in np.linspace(lo, hi, 4)] if hi > lo else []
    iso_days = sorted(set(iso_days))

    return {
        "model": model, "year": year, "init_idx": bundle["init_idx"],
        "params": params.to_json(),
        "region": {"lat_min": region.lat_min, "lat_max": region.lat_max,
                   "lon_min": region.lon_min, "lon_max": region.lon_max},
        "is_ensemble": bundle["ens"] is not None,
        "n_members": int(bundle["ens"].sizes["member"]) if bundle["ens"] is not None else 1,
        "obs_onset": M.field_payload(bundle["obs"]),
        "fcst_onset": M.field_payload(bundle["fcst_det"]),
        "ens_mean_onset": (M.field_payload(bundle["ens"].mean("member", skipna=True))
                           if bundle["ens"] is not None else None),
        "obs_range": obs_rng,
        "fcst_range": fcst_rng,
        "ens_range": ens_rng,
        "iso_days": iso_days,
    }


def _multi_year_bundles(model_key, years, init, params, region):
    """Detect onset for each requested year. Skip years with no data;
    raise FileNotFoundError if no year succeeds."""
    out = []
    errors = []
    for y in years:
        try:
            out.append((int(y), _fields_for(model_key, int(y), init, params, region)))
        except FileNotFoundError as e:
            errors.append((int(y), str(e)))
            continue
    if not out:
        msg = f"no data for model={model_key!r} across years={list(years)!r}"
        if errors:
            msg += f" (first miss: {errors[0][1]})"
        raise FileNotFoundError(msg)
    return out


def _global_range(bundles, key):
    import numpy as np
    los, his = [], []
    for _, b in bundles:
        lo, hi = onset_range(b[key])
        if np.isfinite(lo) and np.isfinite(hi):
            los.append(lo); his.append(hi)
    if not los:
        return (float("nan"), float("nan"))
    return (min(los), max(his))


def _global_thresholds(bundles, n=4) -> list[int]:
    import numpy as np
    obs = _global_range(bundles, "obs")
    fcst = _global_range(bundles, "fcst_det")
    lo = max(obs[0], fcst[0]) + 2
    hi = min(obs[1], fcst[1]) - 2
    if hi <= lo:
        return []
    return sorted({int(round(x)) for x in np.linspace(lo, hi, n)})


def _global_progression_window(bundles) -> tuple[int, int]:
    obs = _global_range(bundles, "obs")
    fcst = _global_range(bundles, "fcst_det")
    lo = min(obs[0], fcst[0]) - 2
    hi = max(obs[1], fcst[1]) + 2
    return int(lo), int(hi)


def _resolve_years(years_arg: str | None, year: str | int | None) -> list[int]:
    """Accept ``year`` as int OR str so an empty string from a URL returns
    a clean 400 "malformed" alongside ``years=abc``, rather than the default
    Pydantic 422 that FastAPI would hand out for ``year: int``."""
    if year in ("", None):
        year_int: int | None = None
    elif isinstance(year, int):
        year_int = year
    else:
        try:
            year_int = int(year)
        except (ValueError, TypeError):
            raise HTTPException(400, f"malformed year {year!r}: expected integer")
    try:
        return AGG.expand_years(years_arg, year_int)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


def _years_meta(bundles, requested) -> dict:
    return {"requested": list(requested),
            "computed": [int(y) for y, _ in bundles]}


# ---------------------------------------------------------------------------
# metric routes (single-year passthrough, multi-year aggregation)
# ---------------------------------------------------------------------------

@app.get("/api/metrics/crps")
def crps(
    model: str,
    year: str | None = None, years: str | None = None,
    init: str | None = None, season_end: int = 220,
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    yrs = _resolve_years(years, year)
    bundles = _multi_year_bundles(model, yrs, init, params, region)
    per_year = []
    for _, b in bundles:
        ens = (b["ens"] if b["ens"] is not None
               else b["fcst_det"].expand_dims({"member": [0]}).transpose("member", "lat", "lon"))
        per_year.append(M.compute_crps(ens, b["obs"], season_end=season_end))
    if len(per_year) == 1:
        out = per_year[0]
        out["n_years"] = 1
        out["years"] = _years_meta(bundles, yrs)
        return out
    out = AGG.aggregate_crps(per_year)
    out["years"] = _years_meta(bundles, yrs)
    return out


@app.get("/api/metrics/fss")
def fss_route(
    model: str,
    year: str | None = None, years: str | None = None, init: str | None = None,
    thresholds: str | None = None,
    neighborhoods: str = "1,3,5,7,9",
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    yrs = _resolve_years(years, year)
    bundles = _multi_year_bundles(model, yrs, init, params, region)
    thr = ([int(t) for t in thresholds.split(",")] if thresholds
           else _global_thresholds(bundles, n=5))
    if not thr:
        raise HTTPException(
            422,
            "no valid FSS thresholds across the selected years — "
            "the forecast / obs onset windows do not overlap. "
            "Narrow the region, pick different years, or pass thresholds= explicitly."
        )
    nbr = [int(n) for n in neighborhoods.split(",")]
    per_year = [M.compute_fss(b["fcst_det"], b["obs"], thresholds=thr, neighborhoods=nbr)
                for _, b in bundles]

    # base rate per threshold across years (fraction of obs cells with onset <= tau)
    import numpy as np
    base_rates = []
    for t in thr:
        rates = []
        for _, b in bundles:
            v = b["obs"].values
            f = np.isfinite(v)
            if not f.any(): continue
            rates.append(float((v[f] <= t).mean()))
        base_rates.append(float(np.mean(rates)) if rates else None)

    if len(per_year) == 1:
        out = per_year[0]
        out["n_years"] = 1
    else:
        out = AGG.aggregate_fss(per_year)
    out["years"] = _years_meta(bundles, yrs)
    out["base_rate"] = base_rates
    # Roberts & Lean 2008 conventions:
    #   FSS_random  = base_rate (FSS of a forecast that always predicts f0)
    #   FSS_useful  = 0.5 + 0.5 * base_rate (Roberts & Lean's "useful" line)
    out["fss_no_skill"] = [br if br is not None else None for br in base_rates]
    out["fss_useful"]   = [0.5 + 0.5 * br if br is not None else None
                           for br in base_rates]
    return out


@app.get("/api/metrics/displacement")
def displacement(
    model: str,
    year: str | None = None, years: str | None = None, init: str | None = None,
    thresholds: str | None = None,
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    yrs = _resolve_years(years, year)
    bundles = _multi_year_bundles(model, yrs, init, params, region)
    thr = ([int(t) for t in thresholds.split(",")] if thresholds
           else _global_thresholds(bundles))
    if not thr:
        raise HTTPException(
            422,
            "no valid displacement thresholds across the selected years — "
            "the forecast / obs onset windows do not overlap."
        )
    per_year = [M.compute_displacement(b["fcst_det"], b["obs"], thresholds=thr)
                for _, b in bundles]
    if len(per_year) == 1:
        out = per_year[0]
        out["n_years"] = 1
    else:
        out = AGG.aggregate_displacement(per_year)
    out["years"] = _years_meta(bundles, yrs)
    return out


@app.get("/api/metrics/progression")
def progression(
    model: str,
    year: str | None = None, years: str | None = None, init: str | None = None,
    step: int = 3,
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    yrs = _resolve_years(years, year)
    bundles = _multi_year_bundles(model, yrs, init, params, region)
    lo, hi = _global_progression_window(bundles)
    days = list(range(int(lo), int(hi) + 1, max(1, step)))
    per_year = [M.compute_progression(b["fcst_det"], b["ens"], b["obs"], days=days)
                for _, b in bundles]
    if len(per_year) == 1:
        out = per_year[0]
        out["n_years"] = 1
    else:
        out = AGG.aggregate_progression(per_year)
    out["years"] = _years_meta(bundles, yrs)
    return out


@app.get("/api/metrics/isochrones")
def isochrones(
    model: str, year: int, init: str | None = None,
    days: str | None = None,
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    """Always single-year. The hero panel needs concrete contours; multi-year
    averaging would smear them. Use the iso_year sidebar control to pick which
    year's contours to render."""
    b = _fields_for(model, int(year), init, params, region)
    iso = ([int(d) for d in days.split(",")]
           if days else _global_thresholds([(year, b)]))
    out = M.compute_isochrones(b["fcst_det"], b["obs"], days=iso)
    out["year"] = int(year)
    return out


@app.get("/api/metrics/corp")
def corp(
    model: str,
    year: str | None = None, years: str | None = None, init: str | None = None,
    tau: int | None = None, season_end: int = 220,
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    """Multi-year CORP pools all (forecast probability, obs binary) pairs
    across years before the PAV decomposition — that's the statistically
    proper way to combine, not averaging the per-year decompositions."""
    import numpy as np
    yrs = _resolve_years(years, year)
    bundles = _multi_year_bundles(model, yrs, init, params, region)
    if tau is None:
        thr = _global_thresholds(bundles)
        tau = thr[len(thr) // 2] if thr else 170
    p_parts, y_parts = [], []
    moran_vals = []  # per-year Moran's I on the obs indicator for effective-n
    for _, b in bundles:
        p, y_ = M.corp_inputs(b["ens"], b["fcst_det"], b["obs"],
                              tau=tau, season_end=season_end)
        p_parts.append(p); y_parts.append(y_)
        # Moran's I on the 2D observed-by-τ indicator (finite cells only);
        # treat obs-NaN cells as NaN so they don't contribute as "= 0".
        obs_vals = np.asarray(b["obs"].values, dtype=float)
        y_field = np.where(np.isfinite(obs_vals), (obs_vals <= tau).astype(float), np.nan)
        mi = M.moran_i_2d(y_field)
        if mi == mi:  # not NaN
            moran_vals.append(mi)
    p_pool = np.concatenate(p_parts)
    y_pool = np.concatenate(y_parts)
    out = M.compute_corp_pooled(p_pool, y_pool, tau=tau)
    out["n_years"] = len(bundles)
    out["years"] = _years_meta(bundles, yrs)
    # Spatial-autocorrelation correction: report effective n alongside raw n.
    # Pooling independent-year CORP pairs is valid, but within a year the
    # grid cells are spatially correlated; raw n overstates independence.
    mean_mi = float(np.mean(moran_vals)) if moran_vals else float("nan")
    n_eff_per_year = M.effective_sample_size(
        int(out["n"] / max(1, len(bundles))), mean_mi
    )
    out["moran_i"] = None if mean_mi != mean_mi else mean_mi
    out["n_effective"] = int(round(n_eff_per_year * len(bundles)))
    return out


def _cra_one_year(
    model_key: str, year: int, init_idx: int,
    *, lead_start: int, lead_end: int, threshold: float, max_shift: int,
    include_fields: bool,
) -> dict:
    """Run CRA for one (model, year, init) and return the JSON-ready payload."""
    from . import rainfall as R
    fa = R.get_forecast_accum(model_key, int(year), int(init_idx),
                              int(lead_start), int(lead_end))
    valid_start, valid_end = R.valid_window(fa.init_time, lead_start, lead_end)
    oa = R.get_obs_accum(int(year), valid_start, valid_end)
    case = f"{model_key}_{int(year)}_init{fa.init_time:%Y%m%d}_lead{lead_start}-{lead_end}"
    payload = M.compute_cra(
        fa.rainfall, fa.lat, fa.lon,
        oa.rainfall, oa.lat, oa.lon,
        case=case, threshold=float(threshold), max_shift=int(max_shift),
        include_fields=include_fields,
    )
    payload["meta"] = {
        "model": model_key,
        "year": int(year),
        "init_idx": int(init_idx),
        "init_time": fa.init_time.strftime("%Y-%m-%d"),
        "valid_start": valid_start.strftime("%Y-%m-%d"),
        "valid_end": valid_end.strftime("%Y-%m-%d"),
        "lead_start": int(lead_start),
        "lead_end": int(lead_end),
        "n_members": fa.n_members,
    }
    return payload


def _cra_for_model(
    model_key: str, years: list[int],
    *, init_idx: int, lead_start: int, lead_end: int,
    threshold: float, max_shift: int,
) -> dict:
    """Single-year passthrough or multi-year aggregate for one model.

    Per-year failures (missing init index, year out of range for the
    forecast file) are recorded in ``warnings`` so the panel can still
    render for the years that succeeded — same robustness pattern as
    the displacement / progression aggregators.
    """
    if not years:
        raise HTTPException(422, "no years selected")
    per_year: list[dict] = []
    warnings: list[str] = []
    rep_fields: dict | None = None
    rep_meta: dict | None = None
    for i, yr in enumerate(years):
        is_last = (i == len(years) - 1)
        try:
            p = _cra_one_year(
                model_key, yr, init_idx,
                lead_start=lead_start, lead_end=lead_end,
                threshold=threshold, max_shift=max_shift,
                include_fields=is_last,
            )
        except (IndexError, ValueError, KeyError, FileNotFoundError) as exc:
            warnings.append(f"{yr}: {exc}")
            continue
        per_year.append(p)
        if "fields" in p:
            rep_fields = p["fields"]
            rep_meta = p["meta"]
    if not per_year:
        raise HTTPException(
            422,
            "CRA could not be computed for any selected year; "
            + ("; ".join(warnings) if warnings else "no diagnostic"),
        )
    if len(per_year) == 1 and len(years) == 1:
        out = per_year[0]
        out["n_years"] = 1
        if warnings:
            out["warnings"] = warnings
        return out
    out = AGG.aggregate_cra(per_year, representative_fields=rep_fields)
    if rep_meta is not None:
        out["representative_meta"] = rep_meta
    if warnings:
        out["warnings"] = warnings
    out["years_requested"] = list(years)
    out["years_computed"] = [int(p["meta"]["year"]) for p in per_year]
    return out


@app.get("/api/metrics/cra")
def cra(
    model: str,
    year: str | None = None,
    years: str | None = None,
    init: int = 0,
    lead_start: int = 1,
    lead_end: int = 15,
    threshold: float = 1.0,
    max_shift: int = 8,
):
    """Contiguous Rain Area MSE decomposition.

    Accepts a single ``year`` or a ``years`` range/CSV. Single-year mode
    returns the full per-cell payload (obs / forecast / shifted heatmaps,
    plus scalar decomposition). Multi-year mode returns median + IQR per
    scalar across the year axis and keeps one representative year's
    fields for visualization.
    """
    if max_shift < 0 or max_shift > 32:
        raise HTTPException(422, "max_shift must be in 0..32")
    if lead_end < lead_start or lead_start < 0:
        raise HTTPException(422, "lead window invalid: require 0 <= lead_start <= lead_end")
    yrs = _resolve_years(years, year)
    return _cra_for_model(
        model, yrs, init_idx=int(init),
        lead_start=int(lead_start), lead_end=int(lead_end),
        threshold=float(threshold), max_shift=int(max_shift),
    )


@app.get("/api/compare")
def compare(
    models: str,
    year: str | None = None, years: str | None = None, init: str | None = None,
    season_end: int = 220,
    cra_init: int = 0, cra_lead_start: int = 1, cra_lead_end: int = 15,
    cra_threshold: float = 1.0, cra_max_shift: int = 8,
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    """Cross-model summary table. Per-row season scalars are median-across-years
    (with q25/q75) when years > 1, raw when single year."""
    yrs = _resolve_years(years, year)
    keys = [k.strip() for k in models.split(",") if k.strip()]
    rows = []
    for mk in keys:
        try:
            bundles = _multi_year_bundles(mk, yrs, init, params, region)
        except FileNotFoundError as e:
            rows.append({"model": mk, "error": str(e)})
            continue
        lo, hi = _global_progression_window(bundles)
        days = list(range(int(lo), int(hi) + 1, 3))
        prog_per = [M.compute_progression(b["fcst_det"], b["ens"], b["obs"], days=days)
                    for _, b in bundles]
        prog = AGG.aggregate_progression(prog_per) if len(prog_per) > 1 else prog_per[0]
        # CRPS mean across years
        crps_per = []
        for _, b in bundles:
            ens_like = (b["ens"] if b["ens"] is not None
                        else b["fcst_det"].expand_dims({"member": [0]}).transpose("member", "lat", "lon"))
            crps_per.append(M.compute_crps(ens_like, b["obs"], season_end=season_end))
        crps_means = [c["mean"] for c in crps_per if c.get("mean") is not None]
        # CORP pooled across years
        thr = _global_thresholds(bundles)
        tau = thr[len(thr) // 2] if thr else 170
        import numpy as np
        p_parts, y_parts = [], []
        for _, b in bundles:
            p, y_ = M.corp_inputs(b["ens"], b["fcst_det"], b["obs"],
                                  tau=tau, season_end=season_end)
            p_parts.append(p); y_parts.append(y_)
        corp_out = M.compute_corp_pooled(np.concatenate(p_parts),
                                         np.concatenate(y_parts), tau=tau)

        # CRA: best-effort per-model summary so the comparison table
        # surfaces which models have displacement- vs pattern-dominant
        # rainfall errors. Failures (missing inits, no rainfall var, etc.)
        # collapse to a None block rather than fail the whole row.
        try:
            cra_out = _cra_for_model(
                mk, yrs, init_idx=int(cra_init),
                lead_start=int(cra_lead_start), lead_end=int(cra_lead_end),
                threshold=float(cra_threshold), max_shift=int(cra_max_shift),
            )
            pct = cra_out.get("pct") or {}
            shift = cra_out.get("corrective_shift") or {}
            if cra_out.get("n_years", 1) > 1:
                cra_summary = {
                    "pct_displacement": (pct.get("displacement") or {}).get("median"),
                    "pct_volume":       (pct.get("volume") or {}).get("median"),
                    "pct_pattern":      (pct.get("pattern") or {}).get("median"),
                    "shift_dx":         (shift.get("dx") or {}).get("median"),
                    "shift_dy":         (shift.get("dy") or {}).get("median"),
                    "n_years":          cra_out.get("n_years"),
                }
            else:
                cra_summary = {
                    "pct_displacement": pct.get("displacement"),
                    "pct_volume":       pct.get("volume"),
                    "pct_pattern":      pct.get("pattern"),
                    "shift_dx":         shift.get("dx"),
                    "shift_dy":         shift.get("dy"),
                    "n_years":          cra_out.get("n_years", 1),
                }
        except HTTPException as exc:
            cra_summary = {"error": exc.detail}
        except Exception as exc:  # data-shape mismatches, etc.
            cra_summary = {"error": str(exc)}

        model_info = model_by_key(mk)
        b0 = bundles[0][1]
        rows.append({
            "model": mk,
            "label": model_info.label,
            "is_ensemble": b0["ens"] is not None,
            "n_members": (int(b0["ens"].sizes["member"])
                          if b0["ens"] is not None else 1),
            "init_idx": b0["init_idx"],
            "n_years": len(bundles),
            "years": [int(y) for y, _ in bundles],
            # Pass the full progression block through so the frontend
            # has access to all uncertainty layers (q25/q75, ci_lo/ci_hi)
            # plus the peak-DOY diagnostic. Hand-picking specific keys
            # is brittle — earlier omission of ci_lo/ci_hi here meant
            # the chart's CI bands silently never rendered through the
            # /api/compare route.
            "progression": prog,
            "crps": {
                "mean": float(np.mean(crps_means)) if crps_means else None,
                "median": float(np.median(crps_means)) if crps_means else None,
                "n_years": len(crps_means),
            },
            "corp": {
                "tau": corp_out["tau"],
                "bs": corp_out["mean_score"],
                "mcb": corp_out["mcb"],
                "dsc": corp_out["dsc"],
                "unc": corp_out["unc"],
            },
            "cra": cra_summary,
        })
    return {"years": yrs, "n_years": len(yrs), "params": params.to_json(), "rows": rows}


def _suggest_thresholds(bundle) -> list[int]:
    import numpy as np
    obs_rng = onset_range(bundle["obs"])
    fcst_rng = onset_range(bundle["fcst_det"])
    lo = max(obs_rng[0], fcst_rng[0]) + 2
    hi = min(obs_rng[1], fcst_rng[1]) - 2
    if hi <= lo:
        return []
    return sorted({int(round(x)) for x in np.linspace(lo, hi, 4)})


def _progression_window(bundle) -> tuple[int, int]:
    obs_rng = onset_range(bundle["obs"])
    fcst_rng = onset_range(bundle["fcst_det"])
    lo = min(obs_rng[0], fcst_rng[0]) - 2
    hi = max(obs_rng[1], fcst_rng[1]) + 2
    return int(lo), int(hi)


# ---------------------------------------------------------------------------
# static frontend
# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        # Append ?v=<app.version> to the static script + stylesheet refs so
        # the browser can't reuse a cached app.js / app.css across a server
        # restart that changed the server code. StaticFiles ignores the
        # query string when resolving the path.
        v = app.version
        html = (STATIC_DIR / "index.html").read_text()
        html = html.replace('"/static/app.css"', f'"/static/app.css?v={v}"')
        html = html.replace('"/static/app.js"',  f'"/static/app.js?v={v}"')
        return HTMLResponse(content=html)
