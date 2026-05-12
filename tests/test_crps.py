"""Known-answer tests for momp.metrics.crps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.metrics.crps import (
    CensoredCRPSDecomposition,
    censored_crps,
    censored_crps_decomposition,
    censored_crps_field,
    censored_crps_skill_score,
    crps_ensemble,
)


RNG = np.random.default_rng(20260417)


# ---------- base ensemble CRPS ---------------------------------------------


def test_crps_deterministic_ensemble_reduces_to_absolute_error():
    ens = np.full(20, 150.0)
    val = crps_ensemble(ens, 155.0)
    assert float(val) == pytest.approx(5.0, abs=1e-12)


def test_crps_nonnegative_random():
    for _ in range(10):
        m = RNG.integers(5, 40)
        ens = RNG.uniform(100, 200, size=m)
        y = float(RNG.uniform(100, 200))
        val = crps_ensemble(ens, y)
        assert float(val) >= -1e-12


def test_fair_crps_below_raw_for_finite_ensemble():
    """Raw ensemble CRPS is known to OVERestimate the true CRPS of the
    underlying distribution; Ferro 2014 fair CRPS corrects the bias
    downward. For a non-degenerate ensemble (spread > 0) the fair score
    is therefore strictly less than the raw score."""
    m = 11
    ens = RNG.uniform(100, 200, size=m)
    y = 150.0
    raw = float(crps_ensemble(ens, y, fair=False))
    fair = float(crps_ensemble(ens, y, fair=True))
    assert fair <= raw + 1e-12
    assert raw - fair > 0      # strictly positive on a non-degenerate sample


def test_fair_crps_converges_to_raw_as_m_grows():
    """The bias term (1/m² vs 1/m(m-1)) vanishes as m → ∞."""
    samples_from = RNG.normal(150, 10, size=2000)
    y = 150.0
    diffs = []
    for m in (5, 20, 200, 1500):
        ens = samples_from[:m]
        raw = float(crps_ensemble(ens, y, fair=False))
        fair = float(crps_ensemble(ens, y, fair=True))
        diffs.append(abs(fair - raw))
    assert diffs[0] > diffs[1] > diffs[2] > diffs[3]
    assert diffs[-1] < 1e-2


def test_fair_crps_requires_at_least_two_members():
    with pytest.raises(ValueError):
        crps_ensemble(np.array([150.0]), 155.0, fair=True)


def test_crps_closed_form_matches_pairwise_formula():
    # Brute-force CRPS via the definitional Hersbach formula, cross-check.
    m = 7
    ens = RNG.uniform(100, 200, size=m)
    y = float(RNG.uniform(100, 200))
    mae = np.mean(np.abs(ens - y))
    pairwise = np.mean(np.abs(ens[:, None] - ens[None, :]))
    expected = mae - 0.5 * pairwise
    got = float(crps_ensemble(ens, y))
    assert got == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_crps_vectorized_over_grid():
    # Ensemble shape (m, lat, lon), obs shape (lat, lon).
    m, lat, lon = 12, 6, 7
    ens = RNG.uniform(100, 200, size=(m, lat, lon))
    y = RNG.uniform(100, 200, size=(lat, lon))
    out = crps_ensemble(ens, y)
    assert out.shape == (lat, lon)
    # Cross-check a random cell with the 1-D routine.
    i, j = 3, 2
    expected = float(crps_ensemble(ens[:, i, j], y[i, j]))
    assert float(out[i, j]) == pytest.approx(expected, rel=1e-12)


def test_crps_rejects_nan_input():
    with pytest.raises(ValueError):
        crps_ensemble(np.array([1.0, np.nan, 2.0]), 1.5)
    with pytest.raises(ValueError):
        crps_ensemble(np.array([1.0, 2.0]), np.nan)


# ---------- censored CRPS ---------------------------------------------------


def test_censored_reduces_to_standard_when_all_members_and_obs_have_onset():
    season_end = 200
    ens = np.array([140.0, 150.0, 160.0, 170.0])
    y = 155.0
    c = float(censored_crps(ens, y, season_end=season_end))
    ref = float(crps_ensemble(ens, y))
    assert c == pytest.approx(ref, rel=1e-12)


def test_censored_zero_when_all_no_onset():
    season_end = 200
    ens = np.array([np.nan, np.nan, np.nan])
    c = float(censored_crps(ens, np.nan, season_end=season_end))
    assert c == pytest.approx(0.0, abs=1e-12)


def test_censored_all_no_onset_members_but_observed_onset():
    season_end = 200
    ens = np.array([np.nan, np.nan, np.nan])  # sentinel = 201
    y = 150.0
    c = float(censored_crps(ens, y, season_end=season_end))
    # All ensemble members at sentinel 201, obs at 150 -> CRPS = |201 - 150| = 51
    assert c == pytest.approx(51.0, rel=1e-12)


def test_censored_all_onset_members_but_no_onset_observed():
    season_end = 200
    ens = np.array([120.0, 130.0, 140.0])
    y = np.nan
    c = float(censored_crps(ens, y, season_end=season_end))
    # Deterministic... no: spread across members. obs at sentinel 201.
    ref = float(crps_ensemble(ens, 201.0))
    assert c == pytest.approx(ref, rel=1e-12)


def test_censored_integer_input_accepted():
    season_end = 200
    ens = np.array([140, 150, 160])
    c = float(censored_crps(ens, 155, season_end=season_end))
    assert c > 0


def test_censored_datetime64_input_converted_to_doy():
    season_end = 200
    # 2024 is leap year; DOY of 2024-05-19 = 140, 2024-05-29 = 150.
    ens = np.array(
        [
            np.datetime64("2024-05-19"),
            np.datetime64("2024-05-29"),
            np.datetime64("NaT"),  # no-onset member
        ]
    )
    y = np.datetime64("2024-06-03")  # DOY 155
    c = float(censored_crps(ens, y, season_end=season_end))
    # Sanity: should be finite and positive, bounded by sentinel distance.
    assert 0 < c < 100


def test_skill_score_perfect_and_zero():
    # Perfect: forecast CRPS = 0, reference > 0 -> SS = 1
    ss = float(censored_crps_skill_score(np.array(0.0), np.array(10.0)))
    assert ss == pytest.approx(1.0, abs=1e-12)
    # Equal to reference: SS = 0
    ss = float(censored_crps_skill_score(np.array(10.0), np.array(10.0)))
    assert ss == pytest.approx(0.0, abs=1e-12)
    # Reference zero: NaN
    ss = float(censored_crps_skill_score(np.array(5.0), np.array(0.0)))
    assert np.isnan(ss)


# ---------- diagnostic decomposition ---------------------------------------


def test_decomposition_brier_matches_onset_probability_gap():
    season_end = 200
    ens = np.array([120.0, 130.0, np.nan, np.nan])  # 2/4 no-onset -> pi = 0.5
    y = 150.0  # onset observed -> indicator 0
    d = censored_crps_decomposition(ens, y, season_end=season_end)
    assert isinstance(d, CensoredCRPSDecomposition)
    assert d.n_onset_members == 2
    assert d.n_no_onset_members == 2
    # Brier = (pi - 1_no_onset)^2 = (0.5 - 0)^2 = 0.25
    assert d.brier_atom == pytest.approx(0.25, abs=1e-12)
    # Continuous CRPS on onset members ([120, 130]) vs y=150
    ref = float(crps_ensemble(np.array([120.0, 130.0]), 150.0))
    assert d.crps_continuous == pytest.approx(ref, rel=1e-12)


def test_decomposition_zero_continuous_when_obs_no_onset():
    season_end = 200
    ens = np.array([120.0, 130.0, np.nan, np.nan])
    y = np.nan
    d = censored_crps_decomposition(ens, y, season_end=season_end)
    assert d.crps_continuous == 0.0
    # Brier = (pi - 1)^2 = (0.5 - 1)^2 = 0.25
    assert d.brier_atom == pytest.approx(0.25, abs=1e-12)


# ---------- xarray field wrapper ------------------------------------------


def test_field_wrapper_shapes_and_values():
    season_end = 200
    m, lat, lon = 6, 4, 5
    rng = np.random.default_rng(42)
    ens_vals = rng.uniform(100, 200, size=(m, lat, lon))
    # Inject NaNs
    ens_vals.ravel()[::7] = np.nan
    obs_vals = rng.uniform(100, 200, size=(lat, lon))
    obs_vals.ravel()[::9] = np.nan

    ensemble = xr.DataArray(
        ens_vals,
        dims=("member", "lat", "lon"),
        coords={"lat": np.arange(lat), "lon": np.arange(lon)},
    )
    observed = xr.DataArray(
        obs_vals,
        dims=("lat", "lon"),
        coords={"lat": np.arange(lat), "lon": np.arange(lon)},
    )
    da = censored_crps_field(ensemble, observed, season_end=season_end)
    assert da.dims == ("lat", "lon")
    assert da.shape == (lat, lon)
    assert np.all(np.isfinite(da.values))
    assert np.all(da.values >= 0)
    # Spot-check a cell with the scalar routine
    i, j = 2, 3
    expected = float(
        censored_crps(ens_vals[:, i, j], obs_vals[i, j], season_end=season_end)
    )
    assert float(da.isel(lat=i, lon=j)) == pytest.approx(expected, rel=1e-12)


def test_field_missing_member_dim_raises():
    ens = xr.DataArray(np.zeros((3, 4)), dims=("lat", "lon"))
    obs = xr.DataArray(np.zeros((3, 4)), dims=("lat", "lon"))
    with pytest.raises(ValueError):
        censored_crps_field(ens, obs, season_end=200)
