"""Known-answer tests for momp.metrics.progression (IOE + SPS)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from momp.metrics.progression import (
    integrated_onset_error,
    spatial_probability_score,
)


def _field(values: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> xr.DataArray:
    return xr.DataArray(values, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))


# ---------- IOE ------------------------------------------------------------


def test_ioe_zero_for_identical_fields():
    lats = np.arange(0.0, 10.0)
    lons = np.arange(0.0, 10.0)
    vals = np.full((10, 10), 150.0)
    f = _field(vals, lats, lons)
    ds = integrated_onset_error(f, f, days=[120, 140, 150, 160, 180])
    assert float(ds["ioe_season_km2_day"]) == pytest.approx(0.0, abs=1e-6)
    assert np.all(ds["ioe_km2"].values == 0.0)


def test_ioe_equals_extent_when_uniform_fields_differ_in_date():
    # Forecast onset everywhere on DOY 120; obs everywhere on DOY 150.
    lats = np.arange(0.0, 10.0)
    lons = np.arange(0.0, 10.0)
    fvals = np.full((10, 10), 120.0)
    ovals = np.full((10, 10), 150.0)
    f = _field(fvals, lats, lons)
    o = _field(ovals, lats, lons)
    ds = integrated_onset_error(f, o, days=[120, 135, 150])

    at_120 = ds["ioe_km2"].sel(day=120)
    extent_120 = ds["extent_km2"].sel(day=120)
    # At d=120 fcst is all on, obs all off -> SD = full domain = extent.
    assert float(at_120) == pytest.approx(float(extent_120), rel=1e-12)
    # Misplacement zero because fcst and obs are spatially uniform.
    assert float(ds["misplacement_km2"].sel(day=120)) == pytest.approx(0.0, abs=1e-8)
    # At d=150 both fully on.
    assert float(ds["ioe_km2"].sel(day=150)) == pytest.approx(0.0, abs=1e-6)


def test_ioe_shifted_blocks_have_pure_misplacement():
    # Equal-area blocks, different locations -> extent = 0, misplacement = IOE.
    lats = np.arange(0.0, 20.0)
    lons = np.arange(0.0, 20.0)
    fvals = np.full((20, 20), np.nan)
    ovals = np.full((20, 20), np.nan)
    ovals[5:10, 5:10] = 150.0
    fvals[5:10, 10:15] = 150.0  # same-size block shifted east by 5
    f = _field(fvals, lats, lons)
    o = _field(ovals, lats, lons)
    ds = integrated_onset_error(f, o, days=[180])
    ioe = float(ds["ioe_km2"].sel(day=180))
    extent = float(ds["extent_km2"].sel(day=180))
    misp = float(ds["misplacement_km2"].sel(day=180))
    assert extent == pytest.approx(0.0, abs=1e-8)
    assert misp == pytest.approx(ioe, rel=1e-12)
    assert ioe > 0


def test_ioe_ge_extent_pointwise():
    rng = np.random.default_rng(11)
    lats = np.arange(0.0, 15.0)
    lons = np.arange(0.0, 15.0)
    fvals = rng.uniform(120, 200, size=(15, 15))
    ovals = rng.uniform(120, 200, size=(15, 15))
    # Sprinkle NaNs.
    fvals[rng.random((15, 15)) < 0.2] = np.nan
    ovals[rng.random((15, 15)) < 0.2] = np.nan
    f = _field(fvals, lats, lons)
    o = _field(ovals, lats, lons)
    ds = integrated_onset_error(f, o, days=range(120, 201, 5))
    assert np.all(ds["ioe_km2"].values >= ds["extent_km2"].values - 1e-6)


def test_ioe_symmetric_in_forecast_and_obs():
    rng = np.random.default_rng(13)
    lats = np.arange(0.0, 10.0)
    lons = np.arange(0.0, 10.0)
    fvals = rng.uniform(120, 200, size=(10, 10))
    ovals = rng.uniform(120, 200, size=(10, 10))
    f = _field(fvals, lats, lons)
    o = _field(ovals, lats, lons)
    a = integrated_onset_error(f, o, days=[130, 150, 170, 190])
    b = integrated_onset_error(o, f, days=[130, 150, 170, 190])
    assert np.allclose(a["ioe_km2"].values, b["ioe_km2"].values, rtol=1e-12)


def test_ioe_nat_obs_is_never_on():
    lats = np.arange(0.0, 5.0)
    lons = np.arange(0.0, 5.0)
    fvals = np.full((5, 5), 140.0)
    ovals = np.full((5, 5), np.nan)  # never onset in observation
    f = _field(fvals, lats, lons)
    o = _field(ovals, lats, lons)
    ds = integrated_onset_error(f, o, days=[140, 200])
    # fcst says on at both days; obs says off at both; SD = full domain.
    assert float(ds["ioe_km2"].sel(day=140)) == pytest.approx(
        float(ds["ioe_km2"].sel(day=200)), rel=1e-12
    )


def test_ioe_advancing_front_analytical_value():
    # Linear front in latitude: onset = a + b*lat. Forecast leads obs by 10 days.
    # At each d the "onset-by-d" mask is an exact lat-strip, so IOE has a
    # closed form in terms of row areas; this is the §7 Milestone-2 check.
    from momp.utils.spherical import cell_area_km2

    lats = np.arange(0.0, 11.0)
    lons = np.arange(0.0, 10.0)
    a_f, a_o, b = 100.0, 110.0, 1.0
    base = np.broadcast_to(lats[:, None], (lats.size, lons.size)).astype(float)
    fvals = a_f + b * base
    ovals = a_o + b * base
    f = _field(fvals, lats, lons)
    o = _field(ovals, lats, lons)

    days = [100, 105, 110, 115, 120]
    ds = integrated_onset_error(f, o, days=days)

    A = cell_area_km2(lats, lons, 6371.0088)
    for d in days:
        fmask = (fvals <= d).astype(float)
        omask = (ovals <= d).astype(float)
        expected = float(np.sum(np.abs(fmask - omask) * A))
        assert float(ds["ioe_km2"].sel(day=d)) == pytest.approx(expected, rel=1e-12, abs=1e-9)

    # Anchor 1: at d=100 only forecast row 0 is on, obs entirely off -> IOE = area(row 0).
    row0_area = float(A[0, :].sum())
    assert float(ds["ioe_km2"].sel(day=100)) == pytest.approx(row0_area, rel=1e-12)

    # Anchor 2: at d=110 fcst fully on, obs has only row 0 on -> IOE = area(rows 1..10).
    rows_rest_area = float(A[1:, :].sum())
    assert float(ds["ioe_km2"].sel(day=110)) == pytest.approx(rows_rest_area, rel=1e-12)

    from scipy.integrate import trapezoid
    expected_season = float(trapezoid(ds["ioe_km2"].values, np.asarray(days, dtype=float)))
    assert float(ds["ioe_season_km2_day"]) == pytest.approx(expected_season, rel=1e-12)


def test_ioe_season_integral_matches_trapezoid():
    lats = np.arange(0.0, 5.0)
    lons = np.arange(0.0, 5.0)
    fvals = np.full((5, 5), 120.0)
    ovals = np.full((5, 5), 150.0)
    f = _field(fvals, lats, lons)
    o = _field(ovals, lats, lons)
    days = [120, 130, 140, 150]
    ds = integrated_onset_error(f, o, days=days)
    from scipy.integrate import trapezoid

    expected = float(trapezoid(ds["ioe_km2"].values, np.asarray(days, dtype=float)))
    assert float(ds["ioe_season_km2_day"]) == pytest.approx(expected, rel=1e-12)


# ---------- SPS ------------------------------------------------------------


def test_sps_reduces_to_ioe_for_single_deterministic_member():
    lats = np.arange(0.0, 10.0)
    lons = np.arange(0.0, 10.0)
    fvals = np.full((10, 10), np.nan)
    ovals = np.full((10, 10), np.nan)
    fvals[3:7, 3:7] = 140.0
    ovals[5:9, 5:9] = 150.0
    f = _field(fvals, lats, lons)
    o = _field(ovals, lats, lons)

    ens = xr.DataArray(
        fvals[None, ...],
        dims=("member", "lat", "lon"),
        coords={"lat": lats, "lon": lons, "member": [0]},
    )
    days = [140, 150, 160]
    ds_sps = spatial_probability_score(ens, o, days=days)
    ds_ioe = integrated_onset_error(f, o, days=days)
    assert np.allclose(ds_sps["sps_km2"].values, ds_ioe["ioe_km2"].values, rtol=1e-12)


def test_sps_nonnegative_and_matches_analytical_uniform_probability():
    # 2-member ensemble where one predicts onset everywhere at d=140, the other
    # predicts never. So P(onset<=d) = 0.5 for d>=140 until none, else 0.
    lats = np.arange(0.0, 4.0)
    lons = np.arange(0.0, 4.0)
    fvals = np.stack([np.full((4, 4), 140.0), np.full((4, 4), np.nan)])
    ens = xr.DataArray(
        fvals,
        dims=("member", "lat", "lon"),
        coords={"lat": lats, "lon": lons, "member": [0, 1]},
    )
    obs = _field(np.full((4, 4), np.nan), lats, lons)
    ds = spatial_probability_score(ens, obs, days=[140, 200])
    # At d=140: P=0.5 everywhere, O=0 -> per-cell Brier = 0.25
    # SPS = 0.25 * total_area
    from momp.utils.spherical import cell_area_km2

    total_area = float(cell_area_km2(lats, lons, 6371.0088).sum())
    assert float(ds["sps_km2"].sel(day=140)) == pytest.approx(0.25 * total_area, rel=1e-12)


def test_sps_missing_member_dim_raises():
    lats = np.arange(0.0, 4.0)
    lons = np.arange(0.0, 4.0)
    ens = _field(np.full((4, 4), 140.0), lats, lons)
    obs = _field(np.full((4, 4), 140.0), lats, lons)
    with pytest.raises(ValueError):
        spatial_probability_score(ens, obs, days=[140])


# ---------- peak_doy --------------------------------------------------------


def test_peak_doy_bell_shape_returns_apex():
    from momp.metrics.progression import peak_doy
    days = [120, 130, 140, 150, 160, 170]
    curve = [0.0, 1.0, 3.0, 5.0, 2.0, 0.0]  # apex at d=150
    d, v = peak_doy(curve, days)
    assert d == 150.0
    assert v == 5.0


def test_peak_doy_monotone_rising_returns_last_day():
    from momp.metrics.progression import peak_doy
    days = [120, 130, 140, 150]
    curve = [0.0, 1.0, 2.0, 3.0]
    d, v = peak_doy(curve, days)
    assert d == 150.0
    assert v == 3.0


def test_peak_doy_tie_returns_earliest():
    # Convention: earliest argmax wins. Useful when a curve plateaus.
    from momp.metrics.progression import peak_doy
    days = [120, 130, 140, 150]
    curve = [1.0, 5.0, 5.0, 5.0]
    d, _ = peak_doy(curve, days)
    assert d == 130.0


def test_peak_doy_all_zero_curve_returns_nan():
    from momp.metrics.progression import peak_doy
    days = [120, 130, 140]
    curve = [0.0, 0.0, 0.0]
    d, v = peak_doy(curve, days)
    assert np.isnan(d)
    assert np.isnan(v)


def test_peak_doy_all_nan_returns_nan():
    from momp.metrics.progression import peak_doy
    days = [120, 130, 140]
    curve = [np.nan, np.nan, np.nan]
    d, v = peak_doy(curve, days)
    assert np.isnan(d)
    assert np.isnan(v)


def test_peak_doy_skips_nan_to_finite_max():
    from momp.metrics.progression import peak_doy
    days = [120, 130, 140, 150]
    curve = [np.nan, 1.0, 5.0, np.nan]
    d, v = peak_doy(curve, days)
    assert d == 140.0
    assert v == 5.0


def test_peak_doy_empty_returns_nan():
    from momp.metrics.progression import peak_doy
    d, v = peak_doy([], [])
    assert np.isnan(d)
    assert np.isnan(v)


def test_peak_doy_mismatched_lengths_returns_nan():
    from momp.metrics.progression import peak_doy
    d, v = peak_doy([1.0, 2.0], [120, 130, 140])
    assert np.isnan(d)
    assert np.isnan(v)


def test_peak_doy_ioe_advancing_front_lag_diagnostic():
    # Two synthetic "models" with the same season-integrated IOE but
    # one peaks earlier than the other — the headline diagnostic is
    # the peak-DOY shift, which the season integral is blind to.
    from momp.metrics.progression import (
        integrated_onset_error, peak_doy,
    )
    lats = np.arange(0.0, 11.0)
    lons = np.arange(0.0, 10.0)
    base = np.broadcast_to(lats[:, None], (lats.size, lons.size)).astype(float)
    obs_vals = 110.0 + base                      # obs front: 110 -> 120
    early_vals = 100.0 + base                    # early model: leads by 10
    late_vals  = 120.0 + base                    # late model: lags by 10
    obs = _field(obs_vals, lats, lons)
    early = _field(early_vals, lats, lons)
    late  = _field(late_vals,  lats, lons)
    days = list(range(95, 135))

    ds_e = integrated_onset_error(early, obs, days=days)
    ds_l = integrated_onset_error(late,  obs, days=days)
    de, _ = peak_doy(ds_e["ioe_km2"].values, ds_e["day"].values)
    dl, _ = peak_doy(ds_l["ioe_km2"].values, ds_l["day"].values)
    # The late model's IOE peaks later than the early model's. By
    # symmetry of the linear-front construction the season-integrated
    # IOE is identical for both, so without peak-DOY this lag bias
    # would be invisible.
    assert dl > de
    assert float(ds_e["ioe_season_km2_day"]) == pytest.approx(
        float(ds_l["ioe_season_km2_day"]), rel=1e-12,
    )
