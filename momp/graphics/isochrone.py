"""Isochrone extraction, overlay plots, and contour-distance summaries.

An isochrone of an onset-date field is the contour at a fixed DOY value — a
1-D curve on a 2-D map separating "monsoon arrived by this day" from "has
not arrived yet." Comparing forecast and observed isochrones at a sequence
of days gives a human-readable view of how well a model captures the advance
of the onset front.

We extract contours via matplotlib's marching-squares implementation (no
additional dependency) and compute Hausdorff / Fréchet distances between
forecast and observed contours via shapely. Distances are reported both in
raw degrees and in kilometres on the sphere using a local mid-latitude
conversion.
"""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import shapely
import shapely.geometry as sg
import xarray as xr

from momp.utils.spherical import KM_PER_DEG


def _extract_segments(
    field: xr.DataArray, day: float, *, lat_coord: str, lon_coord: str
) -> list[np.ndarray]:
    """Return list of (n, 2) arrays of (lon, lat) vertices on the iso-DOY contour.

    NaN cells (no observed onset) are replaced with a sentinel value well
    above any reasonable threshold before contouring; otherwise matplotlib
    treats NaN as a hole and traces a spurious contour ring around every
    isolated finite cell at every threshold below its value. The Goessling
    convention used elsewhere in ROMP (``_binary_by_day``) treats NaT as
    "never onset" — i.e. above any threshold — so we use the same here.
    """
    if lat_coord not in field.coords or lon_coord not in field.coords:
        raise ValueError(
            f"field missing lat/lon coords '{lat_coord}' / '{lon_coord}'"
        )
    lat = field[lat_coord].values.astype(float)
    lon = field[lon_coord].values.astype(float)
    Z = field.values.astype(float)

    NAN_SENTINEL = 1.0e6  # any value comfortably above the max real DOY
    Z = np.where(np.isfinite(Z), Z, NAN_SENTINEL)

    fig = plt.figure()
    try:
        ax = fig.add_subplot(111)
        cs = ax.contour(lon, lat, Z, levels=[float(day)])
        segments: list[np.ndarray] = []
        # matplotlib >= 3.8 exposes allsegs; older versions have the same attr.
        all_segs = getattr(cs, "allsegs", None)
        if all_segs is None:
            # matplotlib 3.10 path: cs.get_paths()
            for p in cs.get_paths():
                verts = p.vertices
                if verts.size:
                    segments.append(np.asarray(verts, dtype=float))
        else:
            for seg_list in all_segs:
                for seg in seg_list:
                    if seg.size:
                        segments.append(np.asarray(seg, dtype=float))
        return segments
    finally:
        plt.close(fig)


def _segments_to_shapely(segments: list[np.ndarray]):
    if not segments:
        return None
    lines = [sg.LineString(seg) for seg in segments if seg.shape[0] >= 2]
    if not lines:
        return None
    if len(lines) == 1:
        return lines[0]
    return sg.MultiLineString(lines)


def _project_segments_for_distance(
    segments: list[np.ndarray], mean_lat_deg: float
) -> list[np.ndarray]:
    """Scale longitude by cos(mean_lat) so that one unit of either axis is
    one meridional degree. shapely's hausdorff_distance treats coordinates
    as Cartesian, so a raw (lon, lat) line at mid-latitude over-counts E-W
    distance by 1/cos(lat). Pre-scaling lon yields a locally-isotropic
    metric — Cartesian distance × KM_PER_DEG ≈ great-circle km within the
    domain (equirectangular projection, accurate to ~1% over India-scale
    spans, degrades poleward of ~70°).
    """
    scale = float(np.cos(np.deg2rad(mean_lat_deg)))
    out: list[np.ndarray] = []
    for s in segments:
        if s.size == 0:
            continue
        proj = s.copy()
        proj[:, 0] = proj[:, 0] * scale  # column 0 is lon
        out.append(proj)
    return out


def extract_isochrone(
    field: xr.DataArray,
    day: float,
    *,
    lat_coord: str = "lat",
    lon_coord: str = "lon",
) -> list[np.ndarray]:
    """Extract all isochrone segments of the given DOY from a 2-D onset field."""
    return _extract_segments(field, day, lat_coord=lat_coord, lon_coord=lon_coord)


def isochrone_distance(
    forecast: xr.DataArray,
    observed: xr.DataArray,
    *,
    day: float,
    lat_coord: str = "lat",
    lon_coord: str = "lon",
) -> dict:
    """Hausdorff and Fréchet distances between forecast and observed isochrones.

    Distances are computed on a local equirectangular projection
    ``(lon · cos(mean_lat), lat)`` so that one Cartesian unit is one
    meridional degree in either direction; without that step shapely's
    Cartesian Hausdorff treats one degree of longitude and one degree of
    latitude as equal, over-counting E-W distance by ``1 / cos(lat)`` at
    mid-latitudes (~7% at India mid-lat, ~22% at India north).

    Returns a dict with the following keys:

    - ``hausdorff_deg``   : symmetric Hausdorff distance, in projected
                            meridional-degree-equivalents
    - ``hausdorff_km``    : ``hausdorff_deg * KM_PER_DEG``
    - ``frechet_deg``     : discrete Fréchet distance, same units as above
    - ``frechet_km``      : ditto in km
    - ``mean_lat_deg``    : mean latitude of the contour vertices, used
                            for the projection (raw degrees, not projected)
    - ``n_segments_fcst`` / ``n_segments_obs``

    If either isochrone is empty, numeric distances are NaN.

    Caveats: equirectangular accuracy degrades poleward of ~70° and over
    spans wider than ~30° of latitude. For India-scale isochrones the
    error is sub-percent.
    """
    f_segs = extract_isochrone(forecast, day, lat_coord=lat_coord, lon_coord=lon_coord)
    o_segs = extract_isochrone(observed, day, lat_coord=lat_coord, lon_coord=lon_coord)

    result = {
        "n_segments_fcst": len(f_segs),
        "n_segments_obs": len(o_segs),
        "hausdorff_deg": float("nan"),
        "hausdorff_km": float("nan"),
        "frechet_deg": float("nan"),
        "frechet_km": float("nan"),
        "mean_lat_deg": float("nan"),
    }
    if not f_segs or not o_segs:
        return result

    all_pts = np.concatenate(
        [np.asarray(s, dtype=float) for s in (f_segs + o_segs) if s.size]
    )
    if all_pts.size == 0:
        return result
    mean_lat = float(np.mean(all_pts[:, 1]))

    f_proj = _project_segments_for_distance(f_segs, mean_lat)
    o_proj = _project_segments_for_distance(o_segs, mean_lat)
    f_geom = _segments_to_shapely(f_proj)
    o_geom = _segments_to_shapely(o_proj)
    if f_geom is None or o_geom is None:
        result["mean_lat_deg"] = mean_lat
        return result

    haus = float(f_geom.hausdorff_distance(o_geom))
    try:
        frechet = float(shapely.frechet_distance(f_geom, o_geom))
    except (shapely.errors.GEOSException, AttributeError):
        frechet = float("nan")

    result.update(
        {
            "hausdorff_deg": haus,
            "frechet_deg": frechet,
            "hausdorff_km": haus * KM_PER_DEG,
            "frechet_km": frechet * KM_PER_DEG if np.isfinite(frechet) else float("nan"),
            "mean_lat_deg": mean_lat,
        }
    )
    return result


def isochrone_distance_sweep(
    forecast: xr.DataArray,
    observed: xr.DataArray,
    *,
    days: Sequence[float],
    lat_coord: str = "lat",
    lon_coord: str = "lon",
) -> xr.Dataset:
    """Sweep isochrone Hausdorff / Fréchet distances over a set of DOYs."""
    records = [
        isochrone_distance(
            forecast, observed, day=float(d), lat_coord=lat_coord, lon_coord=lon_coord
        )
        for d in days
    ]
    keys = ("hausdorff_deg", "hausdorff_km", "frechet_deg", "frechet_km", "mean_lat_deg")
    data = {k: ("day", np.array([r[k] for r in records], dtype=float)) for k in keys}
    data["n_segments_fcst"] = ("day", np.array([r["n_segments_fcst"] for r in records]))
    data["n_segments_obs"] = ("day", np.array([r["n_segments_obs"] for r in records]))
    return xr.Dataset(data, coords={"day": np.asarray(list(days), dtype=int)})


def isochrone_overlay(
    forecast: xr.DataArray,
    observed: xr.DataArray,
    *,
    days: Sequence[float],
    lat_coord: str = "lat",
    lon_coord: str = "lon",
    save_path: str | None = None,
    show: bool = False,
    title: str | None = None,
) -> plt.Figure:
    """Overlay forecast and observed isochrones at multiple DOYs on one figure.

    Forecast isochrones are drawn dashed, observed solid, coloured by DOY.
    Returns the Figure. If ``save_path`` is provided, the figure is written
    to disk (parent dir created as needed).
    """
    lat = forecast[lat_coord].values
    lon = forecast[lon_coord].values

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("viridis")
    days = list(days)
    norm = plt.Normalize(vmin=min(days), vmax=max(days)) if len(days) > 1 else None

    for d in days:
        color = cmap(norm(d)) if norm is not None else "tab:blue"
        cs_o = ax.contour(lon, lat, observed.values, levels=[float(d)], colors=[color],
                          linewidths=1.8, linestyles="solid")
        cs_f = ax.contour(lon, lat, forecast.values, levels=[float(d)], colors=[color],
                          linewidths=1.8, linestyles="dashed")
        # Label once per day near the bottom of the map.
        ax.clabel(cs_o, inline=True, fontsize=8, fmt=f"{int(d)}")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title)
    solid_handle = plt.Line2D([0], [0], color="k", linestyle="solid", label="Observed")
    dash_handle = plt.Line2D([0], [0], color="k", linestyle="dashed", label="Forecast")
    ax.legend(handles=[solid_handle, dash_handle], loc="lower right")
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig
