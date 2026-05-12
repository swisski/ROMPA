"""Known-answer tests for momp.stats.bootstrap.

The function under test is a percentile-method pair bootstrap on the
median: given a stacked array of per-year values, resample years with
replacement B times, take the median per resample, and report the
percentile-method CI on the resulting bootstrap distribution.
"""
from __future__ import annotations

import numpy as np
import pytest

from momp.stats.bootstrap import bootstrap_median_ci


# ---- shape, schema, degenerate cases -------------------------------


def test_returns_documented_keys():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = bootstrap_median_ci(arr, n_resamples=200, rng=0)
    for k in ("median", "ci_lo", "ci_hi", "n_replicates", "n_years",
              "ci_level", "method"):
        assert k in out, k


def test_1d_returns_scalars():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    out = bootstrap_median_ci(arr, n_resamples=500, rng=0)
    assert out["median"].shape == ()
    assert out["ci_lo"].shape == ()
    assert out["ci_hi"].shape == ()
    assert float(out["median"]) == pytest.approx(4.0)


def test_2d_collapses_resampling_axis():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(20, 5))
    out = bootstrap_median_ci(arr, axis=0, n_resamples=500, rng=1)
    assert out["median"].shape == (5,)
    assert out["ci_lo"].shape == (5,)
    assert out["ci_hi"].shape == (5,)


def test_axis_kwarg():
    # axis=1 resamples along the second axis. Make each row constant in
    # that axis so every resample yields the same median for that row,
    # collapsing the CI to the point estimate.
    arr = np.empty((3, 50))
    arr[0, :] = 1.0
    arr[1, :] = 2.0
    arr[2, :] = 3.0
    out = bootstrap_median_ci(arr, axis=1, n_resamples=200, rng=0)
    assert out["median"].shape == (3,)
    np.testing.assert_allclose(out["median"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(out["ci_lo"], out["median"])
    np.testing.assert_allclose(out["ci_hi"], out["median"])


def test_n_years_zero():
    arr = np.zeros((0,))
    out = bootstrap_median_ci(arr, n_resamples=100, rng=0)
    assert np.isnan(out["median"])
    assert np.isnan(out["ci_lo"])
    assert np.isnan(out["ci_hi"])
    assert out["n_years"] == 0


def test_n_years_one_is_degenerate_point_mass():
    # With a single replicate, every bootstrap resample picks the same
    # value, so CI collapses to the point estimate. This is a defined
    # behavior the callers rely on.
    arr = np.array([42.0])
    out = bootstrap_median_ci(arr, n_resamples=500, rng=0)
    assert float(out["median"]) == 42.0
    assert float(out["ci_lo"]) == 42.0
    assert float(out["ci_hi"]) == 42.0
    assert out["n_replicates"] == 0  # signal that no resampling happened


# ---- correctness properties ----------------------------------------


def test_median_matches_nanmedian():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(25, 4))
    out = bootstrap_median_ci(arr, n_resamples=500, rng=1)
    np.testing.assert_allclose(out["median"], np.nanmedian(arr, axis=0))


def test_ci_contains_median():
    rng = np.random.default_rng(2)
    arr = rng.normal(size=(30, 6))
    out = bootstrap_median_ci(arr, n_resamples=1000, rng=3)
    assert np.all(out["ci_lo"] <= out["median"] + 1e-12)
    assert np.all(out["median"] <= out["ci_hi"] + 1e-12)


def test_ci_subset_of_data_range():
    rng = np.random.default_rng(4)
    arr = rng.normal(size=(40, 3))
    out = bootstrap_median_ci(arr, n_resamples=1000, rng=5)
    lo = np.nanmin(arr, axis=0)
    hi = np.nanmax(arr, axis=0)
    # Bootstrap medians are drawn from the empirical distribution, so each
    # replicate median lies between the smallest and largest observed values.
    assert np.all(out["ci_lo"] >= lo - 1e-12)
    assert np.all(out["ci_hi"] <= hi + 1e-12)


def test_constant_data_zero_width_ci():
    arr = np.full((20, 4), 7.5)
    out = bootstrap_median_ci(arr, n_resamples=500, rng=0)
    np.testing.assert_allclose(out["median"], 7.5)
    np.testing.assert_allclose(out["ci_lo"], 7.5)
    np.testing.assert_allclose(out["ci_hi"], 7.5)


def test_wider_data_wider_ci():
    rng = np.random.default_rng(7)
    n = 40
    tight = rng.normal(scale=0.1, size=(n,))
    wide = rng.normal(scale=10.0, size=(n,))
    out_t = bootstrap_median_ci(tight, n_resamples=1000, rng=8)
    out_w = bootstrap_median_ci(wide, n_resamples=1000, rng=8)
    width_t = float(out_t["ci_hi"] - out_t["ci_lo"])
    width_w = float(out_w["ci_hi"] - out_w["ci_lo"])
    assert width_w > width_t


# ---- reproducibility -----------------------------------------------


def test_same_seed_same_result():
    arr = np.linspace(0, 1, 50).reshape(25, 2)
    a = bootstrap_median_ci(arr, n_resamples=300, rng=42)
    b = bootstrap_median_ci(arr, n_resamples=300, rng=42)
    np.testing.assert_array_equal(a["ci_lo"], b["ci_lo"])
    np.testing.assert_array_equal(a["ci_hi"], b["ci_hi"])


def test_different_seed_different_result():
    # With non-degenerate data and a moderate sample, two different seeds
    # almost always produce different CI bounds. Probabilistic but very safe.
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(30, 4))
    a = bootstrap_median_ci(arr, n_resamples=300, rng=1)
    b = bootstrap_median_ci(arr, n_resamples=300, rng=2)
    assert not np.array_equal(a["ci_lo"], b["ci_lo"])


def test_generator_object_accepted():
    arr = np.arange(20.0)
    gen = np.random.default_rng(99)
    out = bootstrap_median_ci(arr, n_resamples=200, rng=gen)
    assert np.isfinite(out["median"])


# ---- coherent year resampling --------------------------------------


def test_coherent_year_resampling_preserves_perfect_correlation():
    # Two columns that are perfectly correlated within a year. If the
    # bootstrap resampled each column independently, the CI on column-1 -
    # column-0 would be much wider than zero. With coherent year
    # resampling, every replicate's two columns are the *same* rows, so
    # the difference is exactly the within-row difference (here zero).
    rng = np.random.default_rng(0)
    col = rng.normal(size=(40,))
    arr = np.stack([col, col], axis=1)  # shape (40, 2), perfectly correlated
    out = bootstrap_median_ci(arr, n_resamples=500, rng=1)
    # Column medians of every replicate are identical -> CIs are identical.
    np.testing.assert_allclose(out["ci_lo"][0], out["ci_lo"][1])
    np.testing.assert_allclose(out["ci_hi"][0], out["ci_hi"][1])


# ---- NaN handling --------------------------------------------------


def test_partial_nan_column_uses_finite_values_only():
    arr = np.array(
        [[1.0, np.nan],
         [2.0, 5.0],
         [3.0, 6.0],
         [4.0, 7.0],
         [5.0, 8.0]]
    )
    out = bootstrap_median_ci(arr, n_resamples=500, rng=0)
    # Column 0 median is 3.0; column 1 nanmedian is median(5,6,7,8)=6.5.
    assert float(out["median"][0]) == pytest.approx(3.0)
    assert float(out["median"][1]) == pytest.approx(6.5)
    assert np.all(np.isfinite([out["ci_lo"][0], out["ci_hi"][0]]))


def test_all_nan_column_is_nan():
    arr = np.array(
        [[1.0, np.nan],
         [2.0, np.nan],
         [3.0, np.nan]]
    )
    out = bootstrap_median_ci(arr, n_resamples=200, rng=0)
    assert np.isfinite(out["median"][0])
    assert np.isnan(out["median"][1])
    assert np.isnan(out["ci_lo"][1])
    assert np.isnan(out["ci_hi"][1])


# ---- argument validation -------------------------------------------


def test_invalid_ci_level_rejected():
    with pytest.raises(ValueError):
        bootstrap_median_ci(np.zeros(10), ci_level=0.0, rng=0)
    with pytest.raises(ValueError):
        bootstrap_median_ci(np.zeros(10), ci_level=1.0, rng=0)


def test_invalid_n_resamples_rejected():
    with pytest.raises(ValueError):
        bootstrap_median_ci(np.zeros(10), n_resamples=0, rng=0)


# ---- ci_level scaling ----------------------------------------------


def test_higher_ci_level_wider_band():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(50,))
    out_50 = bootstrap_median_ci(arr, n_resamples=2000, rng=1, ci_level=0.50)
    out_95 = bootstrap_median_ci(arr, n_resamples=2000, rng=1, ci_level=0.95)
    width_50 = float(out_50["ci_hi"] - out_50["ci_lo"])
    width_95 = float(out_95["ci_hi"] - out_95["ci_lo"])
    assert width_95 > width_50
