"""Onset-region centroid displacement and area-bias diagnostics.

Given a 2-D forecast and observed onset-date field and a DOY threshold tau,
binary masks are formed by ``field <= tau`` (NaN treated as False). For each
mask we compute

  - the area-weighted centroid (lat_c, lon_c), weights = cos(lat) * dlat * dlon,
  - the total area in square km (same spherical weighting, earth radius R).

The forecast-minus-observed differences yield centroid displacement in
degrees, the corresponding great-circle distance in km, and the absolute
and fractional area biases.

These quantities answer "is the forecast onset region placed too far north /
east?" and "is it too big / too small?" directly and physically. They pair
with FSS (which diagnoses scale) as a direction diagnostic.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr

from momp.utils.spherical import EARTH_RADIUS_KM, _infer_spacing, cell_area_km2


def _extract_coords(field: xr.DataArray, lat_coord: str, lon_coord: str):
    if lat_coord not in field.coords or lon_coord not in field.coords:
        raise ValueError(
            f"field is missing lat/lon coords '{lat_coord}'/'{lon_coord}'; got {list(field.coords)}"
        )
    lat = field[lat_coord].values.astype(float)
    lon = field[lon_coord].values.astype(float)
    return lat, lon


def _cell_area_weights(lat: np.ndarray, lon: np.ndarray, *, area_weighted: bool) -> np.ndarray:
    """Return a 2-D weight array of shape (len(lat), len(lon)).

    If ``area_weighted`` is False, returns a uniform field of 1.0 (useful
    for analytical tests). Otherwise uses the standard spherical cell area
    in (degrees-lat * degrees-lon * cos(lat)) — a relative weight that is
    converted to km^2 downstream.
    """
    if area_weighted:
        w_lat = np.cos(np.deg2rad(lat))
        return np.outer(w_lat, np.ones_like(lon, dtype=float))
    return np.ones((lat.size, lon.size), dtype=float)


def _binary_mask(field: xr.DataArray, threshold: float) -> np.ndarray:
    vals = np.asarray(field.values, dtype=float)
    return ((vals <= threshold) & np.isfinite(vals)).astype(float)


def _area_km2(
    binary: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    earth_radius_km: float,
) -> float:
    return float(np.sum(binary * cell_area_km2(lat, lon, earth_radius_km)))


def _centroid(
    binary: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    area_weighted: bool,
) -> tuple[float, float]:
    weights = _cell_area_weights(lat, lon, area_weighted=area_weighted)
    wmask = binary * weights
    total = wmask.sum()
    if total == 0.0:
        return float("nan"), float("nan")
    lat_c = float((wmask.sum(axis=1) * lat).sum() / total)
    lon_c = float((wmask.sum(axis=0) * lon).sum() / total)
    return lat_c, lon_c


def _great_circle_km(
    lat_a: float, lon_a: float, lat_b: float, lon_b: float, radius_km: float
) -> float:
    if not all(np.isfinite(x) for x in (lat_a, lon_a, lat_b, lon_b)):
        return float("nan")
    la1, la2 = np.deg2rad(lat_a), np.deg2rad(lat_b)
    dlat = la2 - la1
    dlon = np.deg2rad(lon_b - lon_a)
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return float(2 * radius_km * np.arcsin(np.sqrt(a)))


def centroid_displacement(
    forecast: xr.DataArray,
    observed: xr.DataArray,
    *,
    threshold: float,
    lat_coord: str = "lat",
    lon_coord: str = "lon",
    earth_radius_km: float = EARTH_RADIUS_KM,
    area_weighted: bool = True,
) -> dict:
    """Return centroid displacement and area bias at a single threshold.

    Returns a dict with keys:
      centroid_fcst_lat, centroid_fcst_lon,
      centroid_obs_lat, centroid_obs_lon,
      delta_lat_deg, delta_lon_deg, great_circle_km,
      area_fcst_km2, area_obs_km2, area_bias_km2, area_bias_fraction
    """
    if not isinstance(forecast, xr.DataArray) or not isinstance(observed, xr.DataArray):
        raise TypeError("forecast and observed must be xarray DataArrays with lat/lon coords")

    lat_f, lon_f = _extract_coords(forecast, lat_coord, lon_coord)
    lat_o, lon_o = _extract_coords(observed, lat_coord, lon_coord)
    if not (np.array_equal(lat_f, lat_o) and np.array_equal(lon_f, lon_o)):
        raise ValueError("forecast and observed must share the same lat/lon grid")

    f_mask = _binary_mask(forecast, threshold)
    o_mask = _binary_mask(observed, threshold)

    f_lat_c, f_lon_c = _centroid(f_mask, lat_f, lon_f, area_weighted=area_weighted)
    o_lat_c, o_lon_c = _centroid(o_mask, lat_o, lon_o, area_weighted=area_weighted)

    f_area = _area_km2(f_mask, lat_f, lon_f, earth_radius_km)
    o_area = _area_km2(o_mask, lat_o, lon_o, earth_radius_km)

    delta_lat = f_lat_c - o_lat_c
    delta_lon = f_lon_c - o_lon_c
    gc_km = _great_circle_km(f_lat_c, f_lon_c, o_lat_c, o_lon_c, earth_radius_km)

    area_bias = f_area - o_area
    area_bias_frac = area_bias / o_area if o_area > 0 else float("nan")

    return {
        "threshold": float(threshold),
        "centroid_fcst_lat": f_lat_c,
        "centroid_fcst_lon": f_lon_c,
        "centroid_obs_lat": o_lat_c,
        "centroid_obs_lon": o_lon_c,
        "delta_lat_deg": delta_lat,
        "delta_lon_deg": delta_lon,
        "great_circle_km": gc_km,
        "area_fcst_km2": f_area,
        "area_obs_km2": o_area,
        "area_bias_km2": area_bias,
        "area_bias_fraction": area_bias_frac,
    }


def displacement_bias_sweep(
    forecast: xr.DataArray,
    observed: xr.DataArray,
    *,
    thresholds: Sequence[float],
    lat_coord: str = "lat",
    lon_coord: str = "lon",
    earth_radius_km: float = EARTH_RADIUS_KM,
    area_weighted: bool = True,
) -> xr.Dataset:
    """Sweep centroid displacement and area bias over a set of DOY thresholds.

    Returns an xr.Dataset indexed by ``threshold`` with variables:
    ``delta_lat_deg, delta_lon_deg, great_circle_km, area_fcst_km2,
    area_obs_km2, area_bias_km2, area_bias_fraction``.
    """
    results = [
        centroid_displacement(
            forecast,
            observed,
            threshold=float(t),
            lat_coord=lat_coord,
            lon_coord=lon_coord,
            earth_radius_km=earth_radius_km,
            area_weighted=area_weighted,
        )
        for t in thresholds
    ]
    keys = (
        "delta_lat_deg",
        "delta_lon_deg",
        "great_circle_km",
        "area_fcst_km2",
        "area_obs_km2",
        "area_bias_km2",
        "area_bias_fraction",
    )
    data = {k: ("threshold", np.array([r[k] for r in results], dtype=float)) for k in keys}
    return xr.Dataset(
        data,
        coords={"threshold": np.array(list(thresholds), dtype=float)},
        attrs={
            "description": (
                "Centroid displacement and area bias of onset-region masks "
                "(field <= threshold). Positive delta_lat / delta_lon mean the "
                "forecast onset region is displaced northward / eastward relative "
                "to observation. Positive area_bias means forecast region is larger."
            )
        },
    )
