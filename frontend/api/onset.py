"""Onset-DOY field construction with caching.

Wraps ROMP's production ``momp.stats.detect.detect_onset`` and exposes a
process-wide cache keyed by ``(source, year, init_idx, onset_params)`` so
the same configuration is detected once and reused across metric
endpoints."""
from __future__ import annotations

import threading
from dataclasses import dataclass, asdict
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr

from momp.stats.detect import detect_onset, detect_observed_onset

from .catalog import ModelInfo, ObsInfo, model_by_key, obs_source


@dataclass(frozen=True)
class OnsetParams:
    wet_init: float = 1.0
    wet_spell: int = 3
    wet_threshold: float = 20.0
    dry_spell: int = 7
    dry_threshold: float = 1.0
    dry_extent: int = 0

    def to_json(self) -> dict:
        return asdict(self)

    def detector_kwargs(self) -> dict:
        return dict(
            wet_init=self.wet_init,
            wet_spell=self.wet_spell,
            dry_spell=self.dry_spell,
            dry_threshold=self.dry_threshold,
            dry_extent=self.dry_extent,
        )


@dataclass(frozen=True)
class Region:
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None

    def is_empty(self) -> bool:
        return all(
            v is None for v in (self.lat_min, self.lat_max, self.lon_min, self.lon_max)
        )

    def crop(self, da: xr.DataArray, *, lat="lat", lon="lon") -> xr.DataArray:
        sel = {}
        if self.lat_min is not None and self.lat_max is not None:
            lats = da[lat].values
            mask = (lats >= self.lat_min) & (lats <= self.lat_max)
            sel[lat] = mask
        if self.lon_min is not None and self.lon_max is not None:
            lons = da[lon].values
            mask = (lons >= self.lon_min) & (lons <= self.lon_max)
            sel[lon] = mask
        out = da
        for k, m in sel.items():
            out = out.isel({k: np.where(m)[0]})
        return out


# Monsoon-onset search window (matches detect_observed_onset start_date).
# Forecast detection must respect the same window; otherwise a forecast
# can "detect onset" on pre-monsoon April rainfall while obs cannot —
# and the forecast / obs isochrones no longer track each other.
MONSOON_START_MONTH = 5
MONSOON_START_DAY = 1


def _monsoon_start_doy(year: int) -> int:
    return pd.Timestamp(year=int(year), month=MONSOON_START_MONTH,
                        day=MONSOON_START_DAY).dayofyear


def _first_onset_doy(rain: np.ndarray, start_doy: int, p: OnsetParams,
                     *, min_doy: int) -> float:
    """Return the DOY of the first qualifying wet spell in ``rain``, where
    ``start_doy`` is the calendar DOY of ``rain[0]``. Offsets whose calendar
    DOY is before ``min_doy`` are skipped so forecast detection uses the
    same search window as the observed detector (monsoon onset cannot
    occur before the configured start date, by convention)."""
    n = rain.size
    kw = p.detector_kwargs()
    skip_below = max(0, min_doy - start_doy)
    for offset in range(skip_below, n - p.wet_spell + 1):
        if detect_onset(day=offset + 1, forecast_series=rain,
                        thresh=p.wet_threshold, **kw):
            return float(start_doy + offset)
    return float("nan")


# --- module-level caches --------------------------------------------------

_cache_lock = threading.Lock()
_obs_cache: dict[tuple, xr.DataArray] = {}
_fcst_cache: dict[tuple, xr.DataArray] = {}
_init_cache: dict[tuple, list[pd.Timestamp]] = {}


def available_inits(model: ModelInfo, year: int) -> list[pd.Timestamp]:
    if year not in model.years:
        raise FileNotFoundError(
            f"model {model.key!r} has no data for year {year}; "
            f"available: {list(model.years)}"
        )
    key = (model.key, int(year))
    with _cache_lock:
        if key in _init_cache:
            return _init_cache[key]
    path = model.path / f"{year}.nc"
    with xr.open_dataset(path) as ds:
        times = [pd.Timestamp(t) for t in ds["time"].values]
    with _cache_lock:
        _init_cache[key] = times
    return times


def load_rainfall_forecast(model: ModelInfo, year: int) -> xr.DataArray:
    return xr.open_dataset(model.path / f"{year}.nc")[model.var_name]


def _detect_forecast(model: ModelInfo, year: int, init_idx: int,
                     params: OnsetParams) -> xr.DataArray:
    da = load_rainfall_forecast(model, year)
    sel = da.isel(time=init_idx)
    start_doy = pd.Timestamp(da["time"].values[init_idx]).dayofyear
    min_doy = _monsoon_start_doy(year)
    if "number" in sel.dims:
        arr = sel.transpose("number", "day", "lat", "lon").values
        M, _, Ny, Nx = arr.shape
        out = np.full((M, Ny, Nx), np.nan)
        for m in range(M):
            for i in range(Ny):
                for j in range(Nx):
                    out[m, i, j] = _first_onset_doy(
                        arr[m, :, i, j], start_doy, params, min_doy=min_doy,
                    )
        return xr.DataArray(
            out, dims=("member", "lat", "lon"),
            coords={
                "member": sel["number"].values,
                "lat": sel["lat"].values,
                "lon": sel["lon"].values,
            },
            name="onset_doy",
        )
    arr = sel.transpose("day", "lat", "lon").values
    out = np.full(arr.shape[1:], np.nan)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = _first_onset_doy(
                arr[:, i, j], start_doy, params, min_doy=min_doy,
            )
    return xr.DataArray(
        out, dims=("lat", "lon"),
        coords={"lat": sel["lat"].values, "lon": sel["lon"].values},
        name="onset_doy",
    )


def _detect_obs(obs: ObsInfo, year: int, params: OnsetParams) -> xr.DataArray:
    """Detect observed onset via ``momp.stats.detect.detect_observed_onset``.

    This is the same vectorized implementation that ``momp-run`` uses
    against obs data, so the frontend's obs onset DOYs match production
    ROMP output for the same onset-criteria values — previously this
    routine used a simpler May–Sep per-cell loop and could disagree on
    edge-case years.
    """
    with xr.open_dataset(obs.path / f"{year}.nc") as ds:
        rain = ds[obs.var_name]
        # detect_observed_onset expects a dim named "time".
        if "TIME" in rain.dims:
            rain = rain.rename({"TIME": "time"})
        rain = rain.load()

    onset_da = detect_observed_onset(
        rain_slice=rain,
        thresh_slice=params.wet_threshold,  # scalar threshold, broadcasts
        year=int(year),
        wet_init=params.wet_init,
        wet_spell=params.wet_spell,
        dry_spell=params.dry_spell,
        dry_threshold=params.dry_threshold,
        dry_extent=params.dry_extent,
        start_date=(int(year), 5, 1),    # May 1
        end_date=(int(year), 9, 30),     # Sep 30 + 47 days slack
        fallback_date=None,
        mok=None,
        extend_end_day=47,
    )

    # Convert datetime64 onset dates to DOY floats, NaT -> NaN.
    dt_flat = pd.to_datetime(onset_da.values.ravel())
    doy = np.asarray(
        [float(t.dayofyear) if t is not pd.NaT and not pd.isna(t) else np.nan
         for t in dt_flat],
        dtype=float,
    ).reshape(onset_da.shape)
    return xr.DataArray(
        doy,
        coords={"lat": onset_da["lat"].values, "lon": onset_da["lon"].values},
        dims=("lat", "lon"),
        name="onset_doy",
        attrs={"source": "momp.stats.detect.detect_observed_onset"},
    )


def get_obs_onset(year: int, params: OnsetParams) -> xr.DataArray:
    obs = obs_source()
    if year not in obs.years:
        raise KeyError(f"obs has no year {year}; available {obs.years}")
    key = (obs.key, int(year), params)
    with _cache_lock:
        cached = _obs_cache.get(key)
    if cached is not None:
        return cached
    da = _detect_obs(obs, year, params)
    with _cache_lock:
        _obs_cache[key] = da
    return da


def get_forecast_onset(model_key: str, year: int, init_idx: int,
                       params: OnsetParams) -> xr.DataArray:
    model = model_by_key(model_key)
    if year not in model.years:
        raise FileNotFoundError(
            f"model {model_key!r} has no data for year {year}; "
            f"available years: {list(model.years)}"
        )
    key = (model_key, int(year), int(init_idx), params)
    with _cache_lock:
        cached = _fcst_cache.get(key)
    if cached is not None:
        return cached
    da = _detect_forecast(model, year, init_idx, params)
    with _cache_lock:
        _fcst_cache[key] = da
    return da


def best_init_for(model: ModelInfo, obs_lo: float, obs_hi: float,
                  params: OnsetParams, year: int) -> int:
    """Pick the init whose detected onset range overlaps obs the most."""
    best_idx, best_overlap = 0, -1.0
    n = len(available_inits(model, year))
    for k in range(n):
        f = get_forecast_onset(model.key, year, k, params)
        v = f.values[np.isfinite(f.values)]
        if v.size == 0:
            continue
        lo, hi = float(v.min()), float(v.max())
        ov = max(0.0, min(hi, obs_hi) - max(lo, obs_lo))
        if ov > best_overlap:
            best_overlap, best_idx = ov, k
    return best_idx


def onset_range(da: xr.DataArray) -> tuple[float, float]:
    v = np.asarray(da.values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    return (float(finite.min()), float(finite.max()))
