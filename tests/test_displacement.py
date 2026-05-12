"""Known-answer tests for momp.metrics.displacement."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from momp.metrics.displacement import (
    centroid_displacement,
    displacement_bias_sweep,
)


def _onset_field(values: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> xr.DataArray:
    return xr.DataArray(values, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))


def test_identical_fields_zero_displacement_zero_area_bias():
    lats = np.arange(0.0, 10.0)
    lons = np.arange(0.0, 10.0)
    vals = np.full((10, 10), np.nan)
    vals[3:7, :] = 150.0
    field = _onset_field(vals, lats, lons)
    r = centroid_displacement(field, field, threshold=180)
    assert r["delta_lat_deg"] == pytest.approx(0.0, abs=1e-12)
    assert r["delta_lon_deg"] == pytest.approx(0.0, abs=1e-12)
    assert r["area_bias_km2"] == pytest.approx(0.0, abs=1e-6)
    assert r["great_circle_km"] == pytest.approx(0.0, abs=1e-6)


def test_pure_northward_shift_recovered():
    # Unweighted centroid mode — analytical.
    lats = np.arange(0.0, 20.0)
    lons = np.arange(0.0, 20.0)
    fcst_vals = np.full((20, 20), np.nan)
    obs_vals = np.full((20, 20), np.nan)
    obs_vals[5:9, 5:15] = 150.0  # lat centroid at (5+6+7+8)/4 = 6.5
    fcst_vals[7:11, 5:15] = 150.0  # lat centroid at (7+8+9+10)/4 = 8.5
    fcst = _onset_field(fcst_vals, lats, lons)
    obs = _onset_field(obs_vals, lats, lons)
    r = centroid_displacement(fcst, obs, threshold=180, area_weighted=False)
    assert r["delta_lat_deg"] == pytest.approx(2.0, abs=1e-10)
    assert r["delta_lon_deg"] == pytest.approx(0.0, abs=1e-10)


def test_pure_eastward_shift_recovered():
    lats = np.arange(0.0, 20.0)
    lons = np.arange(0.0, 20.0)
    fcst_vals = np.full((20, 20), np.nan)
    obs_vals = np.full((20, 20), np.nan)
    obs_vals[5:15, 3:7] = 150.0
    fcst_vals[5:15, 6:10] = 150.0
    fcst = _onset_field(fcst_vals, lats, lons)
    obs = _onset_field(obs_vals, lats, lons)
    r = centroid_displacement(fcst, obs, threshold=180, area_weighted=False)
    assert r["delta_lat_deg"] == pytest.approx(0.0, abs=1e-10)
    assert r["delta_lon_deg"] == pytest.approx(3.0, abs=1e-10)


def test_area_bias_double_region():
    lats = np.arange(0.0, 20.0)
    lons = np.arange(0.0, 20.0)
    fcst_vals = np.full((20, 20), np.nan)
    obs_vals = np.full((20, 20), np.nan)
    obs_vals[5:10, 5:10] = 150.0
    fcst_vals[5:10, 5:15] = 150.0  # twice as wide
    fcst = _onset_field(fcst_vals, lats, lons)
    obs = _onset_field(obs_vals, lats, lons)
    r = centroid_displacement(fcst, obs, threshold=180)
    assert r["area_bias_fraction"] == pytest.approx(1.0, rel=1e-6)


def test_area_km2_reasonable_magnitude():
    # 1-degree cells centered near the equator should have ~12300 km^2 each.
    lats = np.arange(-2.0, 3.0)  # 5 cells straddling equator
    lons = np.arange(0.0, 5.0)
    vals = np.full((5, 5), 150.0)
    field = _onset_field(vals, lats, lons)
    r = centroid_displacement(field, field, threshold=180)
    # Total area should be ~25 * 12300 = 307500 km^2
    assert 2.5e5 < r["area_obs_km2"] < 3.5e5


def test_no_onset_in_obs_returns_nan_centroid_obs():
    lats = np.arange(0.0, 10.0)
    lons = np.arange(0.0, 10.0)
    fcst = _onset_field(np.full((10, 10), 150.0), lats, lons)
    obs = _onset_field(np.full((10, 10), np.nan), lats, lons)
    r = centroid_displacement(fcst, obs, threshold=180)
    assert np.isnan(r["centroid_obs_lat"])
    assert np.isnan(r["centroid_obs_lon"])
    assert np.isnan(r["delta_lat_deg"])
    assert r["area_obs_km2"] == 0.0
    assert np.isnan(r["area_bias_fraction"])


def test_nan_treated_as_no_onset():
    lats = np.arange(0.0, 10.0)
    lons = np.arange(0.0, 10.0)
    vals_f = np.full((10, 10), np.nan)
    vals_f[3:5, 3:5] = 150.0
    vals_o = np.full((10, 10), np.nan)
    vals_o[3:5, 3:5] = 150.0
    # Add NaNs outside the region in forecast; they should not count as onset.
    vals_f[7, 7] = np.nan
    fcst = _onset_field(vals_f, lats, lons)
    obs = _onset_field(vals_o, lats, lons)
    r = centroid_displacement(fcst, obs, threshold=180, area_weighted=False)
    assert r["delta_lat_deg"] == pytest.approx(0.0, abs=1e-10)
    assert r["delta_lon_deg"] == pytest.approx(0.0, abs=1e-10)


def test_grid_mismatch_raises():
    lats_a = np.arange(0.0, 10.0)
    lats_b = np.arange(0.0, 11.0)
    lons = np.arange(0.0, 10.0)
    fcst = _onset_field(np.full((10, 10), 150.0), lats_a, lons)
    obs = _onset_field(np.full((11, 10), 150.0), lats_b, lons)
    with pytest.raises(ValueError):
        centroid_displacement(fcst, obs, threshold=180)


def test_non_uniform_spacing_raises():
    lats = np.array([0.0, 1.0, 2.5, 5.0])
    lons = np.arange(0.0, 5.0)
    vals = np.full((4, 5), 150.0)
    field = _onset_field(vals, lats, lons)
    with pytest.raises(ValueError):
        centroid_displacement(field, field, threshold=180)


def test_sweep_returns_dataset():
    lats = np.arange(0.0, 10.0)
    lons = np.arange(0.0, 10.0)
    vals_f = np.full((10, 10), np.nan)
    vals_f[3:7, 3:7] = np.tile(np.arange(140, 160, 5)[:4], (4, 1))
    vals_o = vals_f.copy()
    fcst = _onset_field(vals_f, lats, lons)
    obs = _onset_field(vals_o, lats, lons)
    ds = displacement_bias_sweep(fcst, obs, thresholds=[145, 155, 165])
    assert "delta_lat_deg" in ds
    assert ds.sizes["threshold"] == 3
    assert float(ds["delta_lat_deg"].sel(threshold=155)) == pytest.approx(0.0, abs=1e-12)


def test_great_circle_distance_matches_haversine():
    # Construct a forecast with centroid exactly 5 deg N of obs at the same lon.
    lats = np.arange(0.0, 40.0)
    lons = np.arange(0.0, 20.0)
    vals_o = np.full((40, 20), np.nan)
    vals_f = np.full((40, 20), np.nan)
    vals_o[10:12, 8:12] = 150.0  # lat centroid ~10.5
    vals_f[15:17, 8:12] = 150.0  # lat centroid ~15.5
    fcst = _onset_field(vals_f, lats, lons)
    obs = _onset_field(vals_o, lats, lons)
    r = centroid_displacement(fcst, obs, threshold=180, area_weighted=False)
    assert r["delta_lat_deg"] == pytest.approx(5.0, abs=1e-10)
    # Haversine at constant lon: arc length = R * delta_lat_in_radians.
    expected = 6371.0088 * np.deg2rad(5.0)
    assert r["great_circle_km"] == pytest.approx(expected, rel=1e-5)
