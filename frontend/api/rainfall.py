"""Rainfall-accumulation loaders for the CRA endpoint.

CRA verification needs raw forecast and observed *rainfall* fields (not
onset DOY). This module loads each model's `{year}.nc` file, selects one
init time, sums over a lead-day window, and (for ensembles) collapses to
the ensemble mean. The matching observed accumulation is loaded over the
calendar window ``[init + lead_start, init + lead_end]`` so the two fields
are directly comparable.

Cached per (model, year, init_idx, lead_start, lead_end) and per
(year, valid_start, valid_end) to keep multi-model panels responsive.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .catalog import ModelInfo, ObsInfo, model_by_key, obs_source
from .onset import available_inits


@dataclass(frozen=True)
class ForecastAccum:
    rainfall: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    init_time: pd.Timestamp
    n_members: int


@dataclass(frozen=True)
class ObservedAccum:
    rainfall: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


_cache_lock = threading.Lock()
_fcst_accum_cache: dict[tuple, ForecastAccum] = {}
_obs_accum_cache: dict[tuple, ObservedAccum] = {}


def _normalize_obs_time(da: xr.DataArray) -> xr.DataArray:
    # IMD 2° obs spells coords as latitude/longitude (1901-2022) or lat/lon
    # (2023+), with TIME for the time dim throughout. CHIRPS-IMERG over
    # Ethiopia uses uppercase LATITUDE/LONGITUDE/TIME. Normalize all of
    # them to lowercase short names.
    renames = {}
    if "TIME" in da.dims: renames["TIME"] = "time"
    if "latitude" in da.dims: renames["latitude"] = "lat"
    if "longitude" in da.dims: renames["longitude"] = "lon"
    if "LATITUDE" in da.dims: renames["LATITUDE"] = "lat"
    if "LONGITUDE" in da.dims: renames["LONGITUDE"] = "lon"
    return da.rename(renames) if renames else da


def _load_forecast_accum(model: ModelInfo, year: int, init_idx: int,
                         lead_start: int, lead_end: int) -> ForecastAccum:
    inits = available_inits(model, year)
    if not (0 <= init_idx < len(inits)):
        raise IndexError(f"init_idx={init_idx} outside 0..{len(inits)-1} for {model.key} {year}")
    init_time = inits[init_idx]
    with xr.open_dataset(model.path / f"{year}.nc") as ds:
        da = ds[model.var_name].isel(time=init_idx).load()
    if "day" not in da.dims:
        raise ValueError(f"{model.key} {year} forecast has no 'day' dim; dims={da.dims}")
    n_day = da.sizes["day"]
    if lead_start < 0 or lead_end >= n_day or lead_end < lead_start:
        raise ValueError(
            f"lead window [{lead_start},{lead_end}] invalid for n_day={n_day}"
        )
    sliced = da.isel(day=slice(lead_start, lead_end + 1)).sum(dim="day", skipna=True)
    n_members = 1
    if "number" in sliced.dims:
        n_members = int(sliced.sizes["number"])
        sliced = sliced.mean(dim="number", skipna=True)
    return ForecastAccum(
        rainfall=np.asarray(sliced.values, dtype=float),
        lat=np.asarray(sliced["lat"].values, dtype=float),
        lon=np.asarray(sliced["lon"].values, dtype=float),
        init_time=init_time,
        n_members=n_members,
    )


def get_forecast_accum(model_key: str, year: int, init_idx: int,
                       lead_start: int, lead_end: int) -> ForecastAccum:
    model = model_by_key(model_key)
    key = (model_key, int(year), int(init_idx), int(lead_start), int(lead_end))
    with _cache_lock:
        cached = _fcst_accum_cache.get(key)
    if cached is not None:
        return cached
    out = _load_forecast_accum(model, int(year), int(init_idx), int(lead_start), int(lead_end))
    with _cache_lock:
        _fcst_accum_cache[key] = out
    return out


def _load_obs_accum(obs: ObsInfo, year: int,
                    valid_start: pd.Timestamp, valid_end: pd.Timestamp) -> ObservedAccum:
    with xr.open_dataset(obs.path / f"{year}.nc") as ds:
        da = _normalize_obs_time(ds[obs.var_name])
        windowed = da.sel(time=slice(valid_start, valid_end)).load()
    accum = windowed.sum(dim="time", skipna=True)
    return ObservedAccum(
        rainfall=np.asarray(accum.values, dtype=float),
        lat=np.asarray(accum["lat"].values, dtype=float),
        lon=np.asarray(accum["lon"].values, dtype=float),
        valid_start=valid_start,
        valid_end=valid_end,
    )


def get_obs_accum(year: int,
                  valid_start: pd.Timestamp, valid_end: pd.Timestamp) -> ObservedAccum:
    obs = obs_source()
    if year not in obs.years:
        raise KeyError(f"obs has no year {year}; available {obs.years}")
    key = (obs.key, int(year), valid_start.value, valid_end.value)
    with _cache_lock:
        cached = _obs_accum_cache.get(key)
    if cached is not None:
        return cached
    out = _load_obs_accum(obs, int(year), valid_start, valid_end)
    with _cache_lock:
        _obs_accum_cache[key] = out
    return out


def valid_window(init_time: pd.Timestamp, lead_start: int, lead_end: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Map a forecast lead window to the matching observed calendar window."""
    return (init_time + pd.Timedelta(days=int(lead_start)),
            init_time + pd.Timedelta(days=int(lead_end)))


def clear_caches() -> None:
    with _cache_lock:
        _fcst_accum_cache.clear()
        _obs_accum_cache.clear()
