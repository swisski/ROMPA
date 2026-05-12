"""Progression-verification orchestration.

Composes IOE, SPS, and isochrone-distance diagnostics into a single output
package that a caller (test, notebook, or CLI) can invoke with pre-loaded
onset fields. No I/O is performed here unless an output directory is given.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import xarray as xr

from momp.graphics.isochrone import isochrone_distance_sweep, isochrone_overlay
from momp.metrics.progression import (
    integrated_onset_error,
    spatial_probability_score,
)


def progression_verification(
    forecast_onset: xr.DataArray,
    observed_onset: xr.DataArray,
    *,
    days: Sequence[int],
    isochrone_days: Sequence[int] | None = None,
    season_end: int,
    member_dim: str | None = "member",
    year_dim: str | None = None,
    lat_coord: str = "lat",
    lon_coord: str = "lon",
    output_dir: str | None = None,
    model_name: str = "model",
) -> xr.Dataset:
    """Run IOE + SPS (if ensemble) + isochrone distance sweep.

    Parameters
    ----------
    forecast_onset : xr.DataArray
        Onset DOY field. May include ``member_dim`` for ensembles and/or
        ``year_dim`` for a multi-year stack. Must include lat / lon.
    observed_onset : xr.DataArray
        Observed onset field on the same lat/lon grid. May include ``year_dim``.
    days : sequence of int
        DOYs at which to evaluate IOE / SPS.
    isochrone_days : sequence of int, optional
        DOYs at which to compute isochrone Hausdorff / Fréchet distances.
        Defaults to ``days``.
    season_end : int
        DOY upper bound of the onset window (used for metadata and to
        sanity-check inputs).
    member_dim : str or None
        Ensemble-member dim name on ``forecast_onset``. If absent from
        ``forecast_onset.dims``, the forecast is treated as deterministic.
    year_dim : str or None
        If given, IOE / SPS / isochrone metrics are computed per year and
        concatenated along this dim.
    output_dir : str, optional
        If given, writes a NetCDF of results and one overlay PNG per
        (year, model) to this directory.
    model_name : str
        Label used in filenames.

    Returns
    -------
    xr.Dataset
        A merged Dataset of IOE, SPS (if available), and isochrone
        distances. Dims include ``day``, possibly ``year``.
    """
    if isochrone_days is None:
        isochrone_days = list(days)

    has_member = member_dim is not None and member_dim in forecast_onset.dims
    has_year = year_dim is not None and year_dim in observed_onset.dims

    def _one_year(fcst_y: xr.DataArray, obs_y: xr.DataArray) -> xr.Dataset:
        # Build a deterministic forecast for IOE (use member mean if ensemble).
        if has_member:
            det_fcst = fcst_y.mean(dim=member_dim, skipna=True)
        else:
            det_fcst = fcst_y
        ds_ioe = integrated_onset_error(
            det_fcst, obs_y, days=days, lat_coord=lat_coord, lon_coord=lon_coord
        )
        parts = [ds_ioe]
        if has_member:
            ds_sps = spatial_probability_score(
                fcst_y,
                obs_y,
                days=days,
                member_dim=member_dim,
                lat_coord=lat_coord,
                lon_coord=lon_coord,
            )
            parts.append(ds_sps)
        ds_iso = isochrone_distance_sweep(
            det_fcst, obs_y, days=isochrone_days,
            lat_coord=lat_coord, lon_coord=lon_coord,
        )
        # Avoid day-coord collision when isochrone_days differ from days.
        ds_iso = ds_iso.rename({"day": "iso_day"}) if not np.array_equal(
            np.asarray(list(isochrone_days), dtype=int),
            np.asarray(list(days), dtype=int),
        ) else ds_iso
        parts.append(ds_iso)
        return xr.merge(parts, combine_attrs="drop_conflicts")

    if has_year:
        years = sorted(np.asarray(observed_onset[year_dim].values).tolist())
        per_year = []
        for yr in years:
            fcst_y = forecast_onset.sel({year_dim: yr})
            obs_y = observed_onset.sel({year_dim: yr})
            per_year.append(_one_year(fcst_y, obs_y))
        merged = xr.concat(per_year, dim=year_dim).assign_coords({year_dim: years})
    else:
        merged = _one_year(forecast_onset, observed_onset)

    merged.attrs.update(
        {
            "model": model_name,
            "season_end": int(season_end),
            "generator": "momp.app.progression_verification",
        }
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        nc_path = os.path.join(output_dir, f"progression_{model_name}.nc")
        merged.to_netcdf(nc_path)
        if has_year:
            for yr in years:
                fcst_y = forecast_onset.sel({year_dim: yr})
                if has_member:
                    fcst_y = fcst_y.mean(dim=member_dim, skipna=True)
                obs_y = observed_onset.sel({year_dim: yr})
                fig = isochrone_overlay(
                    fcst_y, obs_y,
                    days=isochrone_days,
                    lat_coord=lat_coord, lon_coord=lon_coord,
                    save_path=os.path.join(output_dir, f"isochrone_{model_name}_{yr}.png"),
                    show=False,
                    title=f"{model_name} — {yr} onset isochrones",
                )
                import matplotlib.pyplot as plt
                plt.close(fig)
        else:
            det = forecast_onset.mean(dim=member_dim, skipna=True) if has_member else forecast_onset
            fig = isochrone_overlay(
                det, observed_onset,
                days=isochrone_days,
                lat_coord=lat_coord, lon_coord=lon_coord,
                save_path=os.path.join(output_dir, f"isochrone_{model_name}.png"),
                show=False,
                title=f"{model_name} onset isochrones",
            )
            import matplotlib.pyplot as plt
            plt.close(fig)

    return merged
