"""Known-answer tests for momp.metrics.neighborhood.fss.

FSS = 1 - MSE(F_f, F_o) / (mean F_f^2 + mean F_o^2).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from momp.metrics.neighborhood import fss, fss_multi_year, fss_single


RNG = np.random.default_rng(20260417)


def _random_onset_field(shape=(40, 50), fraction_onset=0.5):
    """Random DOY field with given onset fraction; NaN elsewhere."""
    doy = RNG.integers(120, 200, size=shape).astype(float)
    mask = RNG.uniform(0, 1, size=shape) < fraction_onset
    doy[~mask] = np.nan
    return doy


def test_perfect_forecast_equals_one():
    f = _random_onset_field()
    o = f.copy()
    val = fss_single(f, o, threshold=180, neighborhood=1)
    assert val == pytest.approx(1.0, abs=1e-12)
    val3 = fss_single(f, o, threshold=180, neighborhood=5)
    assert val3 == pytest.approx(1.0, abs=1e-12)


def test_forecast_says_yes_obs_says_no_is_zero():
    # Forecast has onsets everywhere by tau; obs has none.
    shape = (20, 30)
    f = np.full(shape, 150.0)
    o = np.full(shape, np.nan)
    val = fss_single(f, o, threshold=180, neighborhood=1)
    assert val == pytest.approx(0.0, abs=1e-12)


def test_both_empty_returns_nan():
    shape = (20, 30)
    f = np.full(shape, np.nan)
    o = np.full(shape, np.nan)
    val = fss_single(f, o, threshold=180, neighborhood=1)
    assert np.isnan(val)


def test_monotone_in_neighborhood_for_shifted_field():
    # Forecast is obs shifted by 2 cells. For n >= 2*shift+1 the neighborhoods
    # overlap fully and FSS should climb toward 1.
    shape = (40, 40)
    obs = np.full(shape, np.nan)
    obs[10:30, 10:30] = 150.0
    shift = 2
    fcst = np.full(shape, np.nan)
    fcst[10 + shift : 30 + shift, 10 + shift : 30 + shift] = 150.0

    # With constant zero-padding, FSS reaches exactly 1 only when the window
    # covers the entire domain from every cell, i.e. n >= 2*L - 1 where L is
    # the domain side. For L=40 that means n=79.
    neighborhoods = (1, 3, 5, 7, 9, 15, 25, 39, 79)
    scores = [
        fss_single(fcst, obs, threshold=180, neighborhood=n) for n in neighborhoods
    ]
    # Roberts-Lean monotone-in-neighborhood property (up to floating tolerance).
    for a, b in zip(scores, scores[1:]):
        assert b >= a - 1e-10
    # Raw displacement should cost skill at small n relative to large n.
    assert scores[0] < scores[-1] - 1e-6
    # Asymptotic: a neighborhood as large as the domain drives FSS to 1.
    assert scores[-1] == pytest.approx(1.0, abs=1e-6)


def test_fss_is_symmetric_in_forecast_and_obs():
    f = _random_onset_field()
    o = _random_onset_field()
    a = fss_single(f, o, threshold=170, neighborhood=3)
    b = fss_single(o, f, threshold=170, neighborhood=3)
    assert a == pytest.approx(b, abs=1e-12)


def test_sweep_shape_and_coords():
    f = _random_onset_field()
    o = _random_onset_field()
    thresholds = [140, 160, 180]
    neighborhoods = [1, 3, 5]
    da = fss(f, o, thresholds=thresholds, neighborhoods=neighborhoods)
    assert da.dims == ("threshold", "neighborhood")
    assert da.shape == (3, 3)
    assert list(da.threshold.values) == thresholds
    assert list(da.neighborhood.values) == neighborhoods
    # FSS lies in [-inf, 1]; for well-posed cases in [0, 1].
    assert (da.values <= 1.0 + 1e-12).all()


def test_even_neighborhood_raises():
    f = _random_onset_field((10, 10))
    o = _random_onset_field((10, 10))
    with pytest.raises(ValueError):
        fss_single(f, o, threshold=180, neighborhood=4)


def test_shape_mismatch_raises():
    f = _random_onset_field((10, 10))
    o = _random_onset_field((10, 12))
    with pytest.raises(ValueError):
        fss_single(f, o, threshold=180, neighborhood=1)


def test_nan_treated_as_no_onset():
    shape = (8, 8)
    f = np.full(shape, np.nan)
    o = np.full(shape, 150.0)
    # Forecast says no onset anywhere; obs says onset everywhere by tau=180.
    # Under NaN-as-False, f's fraction is 0 and o's fraction is 1.
    val = fss_single(f, o, threshold=180, neighborhood=1)
    assert val == pytest.approx(0.0, abs=1e-12)


def test_xarray_input_accepted():
    shape = (20, 20)
    arr_f = _random_onset_field(shape)
    arr_o = _random_onset_field(shape)
    f = xr.DataArray(arr_f, dims=("lat", "lon"))
    o = xr.DataArray(arr_o, dims=("lat", "lon"))
    da = fss(f, o, thresholds=[160, 180], neighborhoods=[1, 3])
    assert da.shape == (2, 2)


def test_multi_year_basic():
    fby = {y: _random_onset_field((20, 20)) for y in (2001, 2002, 2003)}
    oby = {y: _random_onset_field((20, 20)) for y in (2001, 2002, 2003)}
    da = fss_multi_year(fby, oby, thresholds=[160, 180], neighborhoods=[1, 3])
    assert da.dims == ("year", "threshold", "neighborhood")
    assert list(da.year.values) == [2001, 2002, 2003]


# ---------- base_rate / useful_skill_threshold / useful_scale -------------


def test_base_rate_simple():
    from momp.metrics.neighborhood import base_rate
    obs = xr.DataArray(
        np.array([[140.0, 150.0, 160.0, np.nan]]),
        dims=("lat", "lon"),
    )
    # Threshold τ=155: cells {140, 150} on, {160} off, {NaN} skipped.
    # base_rate = 2 finite-on / 3 finite = 0.6667
    assert base_rate(obs, 155.0) == pytest.approx(2.0 / 3.0, abs=1e-12)
    # τ very high → all finite cells on → 1.0
    assert base_rate(obs, 1000.0) == pytest.approx(1.0)
    # τ very low → none on → 0.0
    assert base_rate(obs, 0.0) == 0.0


def test_base_rate_all_nan_returns_zero():
    from momp.metrics.neighborhood import base_rate
    obs = xr.DataArray(
        np.full((3, 3), np.nan), dims=("lat", "lon")
    )
    assert base_rate(obs, 150.0) == 0.0


def test_useful_skill_threshold_formula():
    from momp.metrics.neighborhood import useful_skill_threshold
    assert useful_skill_threshold(0.0) == pytest.approx(0.50)
    assert useful_skill_threshold(0.5) == pytest.approx(0.75)
    assert useful_skill_threshold(1.0) == pytest.approx(1.00)


def test_useful_skill_threshold_rejects_out_of_range():
    from momp.metrics.neighborhood import useful_skill_threshold
    with pytest.raises(ValueError):
        useful_skill_threshold(-0.1)
    with pytest.raises(ValueError):
        useful_skill_threshold(1.5)


def test_useful_scale_clean_crossing():
    # FSS rises from 0 → 1; threshold 0.75 (p=0.5).
    from momp.metrics.neighborhood import useful_scale
    fss_curve = np.array([0.20, 0.50, 0.85, 0.95])
    nbrs = [1, 3, 5, 7]
    n_star = useful_scale(fss_curve, nbrs, p=0.5)
    # Crossing between idx 1 (FSS=0.50, n=3) and idx 2 (FSS=0.85, n=5).
    # frac = (0.75 - 0.50) / (0.85 - 0.50) = 0.7142...
    # n* = 3 + 0.7142 * (5 - 3) = 4.4286
    assert n_star == pytest.approx(3 + (0.75 - 0.50) / (0.85 - 0.50) * 2, abs=1e-12)


def test_useful_scale_already_at_smallest_neighborhood():
    from momp.metrics.neighborhood import useful_scale
    # FSS already exceeds threshold at n=1.
    fss_curve = np.array([0.85, 0.90, 0.95])
    nbrs = [1, 3, 5]
    assert useful_scale(fss_curve, nbrs, p=0.5) == 1.0


def test_useful_scale_never_crosses_returns_nan():
    from momp.metrics.neighborhood import useful_scale
    fss_curve = np.array([0.10, 0.20, 0.30])
    nbrs = [1, 3, 5]
    assert np.isnan(useful_scale(fss_curve, nbrs, p=0.5))


def test_useful_scale_exact_match_at_boundary():
    from momp.metrics.neighborhood import useful_scale
    # FSS exactly equals threshold at n=5 — useful scale is 5.
    fss_curve = np.array([0.20, 0.50, 0.75])
    nbrs = [1, 3, 5]
    assert useful_scale(fss_curve, nbrs, p=0.5) == 5.0


def test_useful_scale_higher_p_means_higher_threshold():
    # Same FSS curve; higher p → higher useful-skill cutoff → larger n*.
    from momp.metrics.neighborhood import useful_scale
    fss_curve = np.array([0.40, 0.60, 0.80, 0.95])
    nbrs = [1, 3, 5, 7]
    n_low  = useful_scale(fss_curve, nbrs, p=0.2)   # threshold 0.60
    n_high = useful_scale(fss_curve, nbrs, p=0.8)   # threshold 0.90
    assert n_high > n_low


def test_useful_scale_rejects_non_increasing_neighborhoods():
    from momp.metrics.neighborhood import useful_scale
    with pytest.raises(ValueError):
        useful_scale(np.array([0.5, 0.6]), [3, 3], p=0.5)
    with pytest.raises(ValueError):
        useful_scale(np.array([0.5, 0.6]), [5, 3], p=0.5)


def test_useful_scale_per_threshold_matches_loop():
    from momp.metrics.neighborhood import useful_scale, useful_scale_per_threshold
    matrix = np.array([
        [0.10, 0.30, 0.60, 0.90],   # τ_0
        [0.20, 0.55, 0.85, 0.99],   # τ_1
        [0.50, 0.95, 0.99, 1.00],   # τ_2  (already useful at small n)
    ])
    nbrs = [1, 3, 5, 7]
    p = [0.30, 0.40, 0.50]
    out = useful_scale_per_threshold(matrix, nbrs, p)
    expected = np.array([useful_scale(matrix[i], nbrs, p=p[i]) for i in range(3)])
    np.testing.assert_allclose(out, expected, equal_nan=True)
