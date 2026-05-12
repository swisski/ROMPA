"""CLI driver for progression verification (IOE + SPS + isochrone distances).

Example
-------
::

    momp-run-progression \\
        --forecast /path/to/forecast_onset.nc \\
        --observed /path/to/observed_onset.nc \\
        --forecast-var onset_doy \\
        --observed-var onset_doy \\
        --season-end 200 \\
        --days 120:200:5 \\
        --output-dir /path/to/output \\
        --model-name ECMWF-S2S
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

import numpy as np
import xarray as xr

from momp.app.progression_verification import progression_verification


logger = logging.getLogger("momp.progression")


def _parse_days_spec(spec: str) -> list[int]:
    """Parse a days spec such as ``120:200:5`` or ``130,140,150``."""
    spec = spec.strip()
    if "," in spec:
        return [int(s) for s in spec.split(",") if s.strip()]
    if ":" in spec:
        parts = [int(s) for s in spec.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(f"cannot parse days spec '{spec}'")
        if step <= 0:
            raise ValueError("days step must be positive")
        return list(range(start, stop + 1, step))
    return [int(spec)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="momp-run-progression",
        description="ROMP progression verification: IOE, SPS, isochrone distances.",
    )
    p.add_argument("--forecast", required=True, help="Path to forecast onset NetCDF")
    p.add_argument("--observed", required=True, help="Path to observed onset NetCDF")
    p.add_argument("--forecast-var", default="onset_doy",
                   help="Variable name for forecast onset field (default: onset_doy)")
    p.add_argument("--observed-var", default="onset_doy",
                   help="Variable name for observed onset field (default: onset_doy)")
    p.add_argument("--season-end", type=int, required=True,
                   help="DOY upper bound of the onset window")
    p.add_argument(
        "--days",
        default=None,
        help="DOYs for IOE/SPS. Format: 'start:stop[:step]' or 'a,b,c'. "
             "Default: 1-day step from 1 to season-end.",
    )
    p.add_argument(
        "--isochrone-days",
        default=None,
        help="DOYs for isochrone distance / overlay. Default: same as --days.",
    )
    p.add_argument("--member-dim", default="member",
                   help="Ensemble member dim (or '' for deterministic)")
    p.add_argument("--year-dim", default=None,
                   help="Year dim for multi-year inputs (if any)")
    p.add_argument("--lat-coord", default="lat")
    p.add_argument("--lon-coord", default="lon")
    p.add_argument("--output-dir", default=None,
                   help="If given, writes NetCDF + isochrone PNGs here")
    p.add_argument("--model-name", default="model",
                   help="Model label used in output filenames")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def _load_field(path: str, var: str) -> xr.DataArray:
    ds = xr.open_dataset(path)
    if var not in ds.variables:
        raise KeyError(f"variable '{var}' not found in {path}; available: {list(ds.variables)}")
    return ds[var]


def run_progression(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Loading forecast from %s (var=%s)", args.forecast, args.forecast_var)
    fcst = _load_field(args.forecast, args.forecast_var)
    logger.info("Loading observed from %s (var=%s)", args.observed, args.observed_var)
    obs = _load_field(args.observed, args.observed_var)

    days = (
        _parse_days_spec(args.days) if args.days
        else list(range(1, args.season_end + 1))
    )
    iso_days = _parse_days_spec(args.isochrone_days) if args.isochrone_days else list(days)

    logger.info(
        "Running progression verification: %d evaluation days, %d isochrone days",
        len(days), len(iso_days),
    )

    member_dim = args.member_dim if args.member_dim else None
    year_dim = args.year_dim if args.year_dim else None

    ds = progression_verification(
        fcst, obs,
        days=days,
        isochrone_days=iso_days,
        season_end=args.season_end,
        member_dim=member_dim,
        year_dim=year_dim,
        lat_coord=args.lat_coord,
        lon_coord=args.lon_coord,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )

    if "ioe_season_km2_day" in ds:
        ioe_val = ds["ioe_season_km2_day"]
        if ioe_val.ndim == 0:
            logger.info("Season-integrated IOE: %.3e km^2*day", float(ioe_val))
        else:
            logger.info("Season-integrated IOE (per-year):")
            for yr, val in zip(ioe_val[list(ioe_val.dims)[0]].values, ioe_val.values):
                logger.info("  year %s: %.3e km^2*day", yr, float(val))
    if "sps_season_km2_day" in ds:
        sps_val = ds["sps_season_km2_day"]
        if sps_val.ndim == 0:
            logger.info("Season-integrated SPS: %.3e km^2*day", float(sps_val))

    if args.output_dir:
        logger.info("Results written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(run_progression())
