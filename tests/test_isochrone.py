"""Known-answer tests for momp.graphics.isochrone."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr

from momp.graphics.isochrone import (
    extract_isochrone,
    isochrone_distance,
    isochrone_distance_sweep,
    isochrone_overlay,
)
from momp.utils.spherical import KM_PER_DEG


def _ramp_field(lats: np.ndarray, lons: np.ndarray, slope=1.0, intercept=120.0):
    """Onset field with DOY = slope * lat + intercept (isochrones are horizontal lines)."""
    doy = slope * lats[:, None] + intercept + 0 * lons[None, :]
    return xr.DataArray(doy, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))


def test_identical_fields_zero_distance():
    lats = np.linspace(0, 30, 61)
    lons = np.linspace(0, 30, 61)
    f = _ramp_field(lats, lons)
    r = isochrone_distance(f, f, day=140.0)
    assert r["hausdorff_deg"] == pytest.approx(0.0, abs=1e-8)
    assert r["frechet_deg"] == pytest.approx(0.0, abs=1e-8) or np.isnan(r["frechet_deg"])
    assert r["n_segments_fcst"] >= 1
    assert r["n_segments_obs"] >= 1


def test_pure_northward_offset_recovered():
    # Observed: DOY = lat + 120. Forecast: DOY = (lat - 5) + 120 = lat + 115.
    # The DOY = 140 isochrone for obs is lat = 20; for fcst is lat = 25.
    lats = np.linspace(0, 30, 301)
    lons = np.linspace(0, 20, 201)
    obs = _ramp_field(lats, lons, slope=1.0, intercept=120.0)
    fcst = _ramp_field(lats, lons, slope=1.0, intercept=115.0)
    r = isochrone_distance(fcst, obs, day=140.0)
    assert r["hausdorff_deg"] == pytest.approx(5.0, abs=0.1)


def test_isochrone_missing_returns_nan():
    # DOY 300 lies outside the field range.
    lats = np.linspace(0, 30, 61)
    lons = np.linspace(0, 20, 41)
    obs = _ramp_field(lats, lons)
    fcst = _ramp_field(lats, lons)
    r = isochrone_distance(fcst, obs, day=300.0)
    assert np.isnan(r["hausdorff_deg"])
    assert r["n_segments_fcst"] == 0


def test_km_conversion_consistent_with_degrees():
    lats = np.linspace(0, 30, 61)
    lons = np.linspace(0, 20, 41)
    obs = _ramp_field(lats, lons, slope=1.0, intercept=120.0)
    fcst = _ramp_field(lats, lons, slope=1.0, intercept=115.0)
    r = isochrone_distance(fcst, obs, day=140.0)
    # KM_PER_DEG ≈ 111.195 so km should be ~5 deg * ~111 = ~556 km.
    assert r["hausdorff_km"] == pytest.approx(r["hausdorff_deg"] * KM_PER_DEG, rel=1e-6)


def test_pure_eastward_offset_uses_local_longitude_scale():
    # Two synthetic isochrones that are vertical lines (run N-S), one at
    # lon=80 and the other at lon=85, over an India-like latitude band.
    # Naïve (lon, lat) Cartesian Hausdorff would report 5 deg * KM_PER_DEG
    # ≈ 556 km; the equirectangular-projected distance correctly reports
    # 5 * cos(mean_lat) * KM_PER_DEG. At mean_lat≈21.5° this is ~517 km.
    lats = np.linspace(8.0, 35.0, 28)
    lons = np.linspace(68.0, 97.0, 30)
    obs_vals = np.broadcast_to(np.where(lons < 80.0, 145.0, 155.0),
                               (lats.size, lons.size)).astype(float)
    fcst_vals = np.broadcast_to(np.where(lons < 85.0, 145.0, 155.0),
                                (lats.size, lons.size)).astype(float)
    obs = xr.DataArray(obs_vals, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))
    fcst = xr.DataArray(fcst_vals, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))
    r = isochrone_distance(fcst, obs, day=150.0)
    mean_lat = r["mean_lat_deg"]
    expected_km = 5.0 * KM_PER_DEG * np.cos(np.deg2rad(mean_lat))
    assert r["hausdorff_km"] == pytest.approx(expected_km, rel=1e-3)
    # Reported deg is now in projected meridional-deg, not raw.
    assert r["hausdorff_deg"] == pytest.approx(5.0 * np.cos(np.deg2rad(mean_lat)), rel=1e-3)


def test_sweep_shape():
    lats = np.linspace(0, 30, 61)
    lons = np.linspace(0, 20, 41)
    obs = _ramp_field(lats, lons)
    fcst = _ramp_field(lats, lons, intercept=115.0)
    ds = isochrone_distance_sweep(fcst, obs, days=[130, 140, 150, 160])
    assert ds.sizes["day"] == 4
    assert "hausdorff_km" in ds


def test_extract_returns_vertex_arrays():
    lats = np.linspace(0, 30, 61)
    lons = np.linspace(0, 20, 41)
    field = _ramp_field(lats, lons)
    segs = extract_isochrone(field, day=140.0)
    assert len(segs) >= 1
    for s in segs:
        assert s.ndim == 2 and s.shape[1] == 2


def test_overlay_returns_figure_without_error():
    import matplotlib.pyplot as plt

    lats = np.linspace(0, 30, 61)
    lons = np.linspace(0, 20, 41)
    obs = _ramp_field(lats, lons)
    fcst = _ramp_field(lats, lons, intercept=115.0)
    fig = isochrone_overlay(fcst, obs, days=[130, 150, 170], show=False)
    assert fig is not None
    assert fig.axes
    plt.close(fig)


def test_overlay_can_save_to_tmp(tmp_path):
    import matplotlib.pyplot as plt

    lats = np.linspace(0, 30, 61)
    lons = np.linspace(0, 20, 41)
    obs = _ramp_field(lats, lons)
    fcst = _ramp_field(lats, lons, intercept=115.0)
    out = tmp_path / "iso.png"
    fig = isochrone_overlay(fcst, obs, days=[130, 150], save_path=str(out), show=False)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 0
