"""Tests for the progression-verification app function and CLI driver."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr

from momp.app.progression_verification import progression_verification
from momp.driver_progression import _parse_days_spec, build_parser, run_progression


# ---------- app function ----------------------------------------------------


def _onset_ramp(lats, lons, slope=1.0, intercept=120.0):
    doy = slope * lats[:, None] + intercept + 0 * lons[None, :]
    return xr.DataArray(doy, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))


def test_progression_deterministic_returns_ioe_and_iso():
    lats = np.linspace(0, 20, 41)
    lons = np.linspace(0, 20, 41)
    obs = _onset_ramp(lats, lons)
    fcst = _onset_ramp(lats, lons, intercept=115.0)
    ds = progression_verification(
        fcst, obs,
        days=[130, 150, 170],
        isochrone_days=[130, 150, 170],
        season_end=200,
        member_dim=None,
    )
    assert "ioe_km2" in ds
    assert "hausdorff_km" in ds
    assert "sps_km2" not in ds  # deterministic -> no SPS
    assert ds.sizes["day"] == 3


def test_progression_ensemble_returns_sps():
    lats = np.linspace(0, 20, 21)
    lons = np.linspace(0, 20, 21)
    obs = _onset_ramp(lats, lons)
    # 3 members with different intercepts
    members = [
        _onset_ramp(lats, lons, intercept=117.0),
        _onset_ramp(lats, lons, intercept=120.0),
        _onset_ramp(lats, lons, intercept=123.0),
    ]
    ens = xr.concat(members, dim="member").assign_coords({"member": [0, 1, 2]})
    ds = progression_verification(
        ens, obs,
        days=[130, 150, 170],
        season_end=200,
        member_dim="member",
    )
    assert "ioe_km2" in ds
    assert "sps_km2" in ds
    assert np.all(ds["sps_km2"].values >= 0)


def test_progression_saves_outputs(tmp_path):
    lats = np.linspace(0, 20, 21)
    lons = np.linspace(0, 20, 21)
    obs = _onset_ramp(lats, lons)
    fcst = _onset_ramp(lats, lons, intercept=115.0)
    out = tmp_path / "prog_out"
    ds = progression_verification(
        fcst, obs,
        days=[130, 150, 170],
        season_end=200,
        member_dim=None,
        output_dir=str(out),
        model_name="TESTMDL",
    )
    nc = out / "progression_TESTMDL.nc"
    assert nc.exists() and nc.stat().st_size > 0
    png = out / "isochrone_TESTMDL.png"
    assert png.exists() and png.stat().st_size > 0


def test_progression_multi_year():
    lats = np.linspace(0, 20, 21)
    lons = np.linspace(0, 20, 21)
    years = [2001, 2002]
    f_stack = xr.concat(
        [_onset_ramp(lats, lons, intercept=115.0 + y) for y in (0, 1)],
        dim="year",
    ).assign_coords({"year": years})
    o_stack = xr.concat(
        [_onset_ramp(lats, lons, intercept=120.0 + y) for y in (0, 1)],
        dim="year",
    ).assign_coords({"year": years})
    ds = progression_verification(
        f_stack, o_stack,
        days=[130, 150, 170],
        season_end=200,
        member_dim=None,
        year_dim="year",
    )
    assert "year" in ds.dims
    assert ds.sizes["year"] == 2


# ---------- CLI argparse ----------------------------------------------------


def test_parse_days_spec_range():
    assert _parse_days_spec("120:125") == [120, 121, 122, 123, 124, 125]
    assert _parse_days_spec("120:130:5") == [120, 125, 130]


def test_parse_days_spec_list():
    assert _parse_days_spec("130,140,150") == [130, 140, 150]


def test_parse_days_spec_bad_input():
    with pytest.raises(ValueError):
        _parse_days_spec("1:2:3:4")
    with pytest.raises(ValueError):
        _parse_days_spec("1:10:0")


def test_cli_parser_accepts_minimum_args():
    parser = build_parser()
    ns = parser.parse_args([
        "--forecast", "f.nc",
        "--observed", "o.nc",
        "--season-end", "200",
    ])
    assert ns.forecast == "f.nc"
    assert ns.season_end == 200


def test_cli_end_to_end_with_tmp_files(tmp_path):
    lats = np.linspace(0, 20, 21)
    lons = np.linspace(0, 20, 21)
    obs = _onset_ramp(lats, lons)
    fcst = _onset_ramp(lats, lons, intercept=115.0)
    f_path = tmp_path / "fcst.nc"
    o_path = tmp_path / "obs.nc"
    xr.Dataset({"onset_doy": fcst}).to_netcdf(f_path)
    xr.Dataset({"onset_doy": obs}).to_netcdf(o_path)
    out = tmp_path / "out"

    rc = run_progression([
        "--forecast", str(f_path),
        "--observed", str(o_path),
        "--season-end", "200",
        "--days", "130:170:20",
        "--member-dim", "",
        "--output-dir", str(out),
        "--model-name", "CLIMDL",
    ])
    assert rc == 0
    assert (out / "progression_CLIMDL.nc").exists()
    assert (out / "isochrone_CLIMDL.png").exists()
