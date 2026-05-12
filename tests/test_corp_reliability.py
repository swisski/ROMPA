"""Known-answer tests for momp.graphics.corp_reliability.

All tests use the Brier score. The CORP identity is

    S̄  =  MCB  -  DSC  +  UNC

For each synthetic forecast we know one or more of {MCB, DSC} in closed form.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np
import pandas as pd
import pytest

from momp.graphics.corp_reliability import (
    CORPDecomposition,
    corp_decompose_brier,
    corp_reliability_diagram,
)


RNG = np.random.default_rng(20260417)


def _identity_holds(d: CORPDecomposition, atol: float = 1e-10) -> bool:
    return abs(d.mean_score - (d.mcb - d.dsc + d.unc)) < atol


def test_identity_random():
    n = 5000
    f = RNG.uniform(0, 1, size=n)
    y = (RNG.uniform(0, 1, size=n) < f).astype(int)
    d = corp_decompose_brier(f, y)
    assert _identity_holds(d)
    # For a perfectly sampled forecast, MCB should be small and DSC > 0.
    assert d.mcb < 0.01
    assert d.dsc > 0.05


def test_perfect_forecast_zero_mcb_dsc_equals_unc():
    # y = 1 iff latent > 0; deterministic perfect forecast f = y.
    n = 1000
    y = RNG.integers(0, 2, size=n).astype(int)
    f = y.astype(float)
    d = corp_decompose_brier(f, y)
    assert _identity_holds(d)
    assert d.mean_score == pytest.approx(0.0, abs=1e-12)
    assert d.mcb == pytest.approx(0.0, abs=1e-12)
    # For a perfect forecast DSC equals UNC and MCB equals 0.
    assert d.dsc == pytest.approx(d.unc, abs=1e-12)


def test_climatological_forecast_zero_mcb_zero_dsc():
    # Constant forecast at the base rate: calibrated; zero discrimination.
    n = 2000
    y = (RNG.uniform(0, 1, size=n) < 0.3).astype(int)
    f = np.full(n, y.mean())
    d = corp_decompose_brier(f, y)
    assert _identity_holds(d)
    assert d.mcb == pytest.approx(0.0, abs=1e-12)
    assert d.dsc == pytest.approx(0.0, abs=1e-12)
    assert d.mean_score == pytest.approx(d.unc, abs=1e-12)


def test_constant_forecast_off_base_rate_mcb_matches_brier_gap():
    # Forecast is a constant c different from the base rate; PAV collapses to
    # the base rate so the calibrated score equals UNC. Then MCB = (c - ȳ)^2
    # and DSC = 0.
    n = 4000
    y = (RNG.uniform(0, 1, size=n) < 0.4).astype(int)
    c = 0.7
    f = np.full(n, c)
    d = corp_decompose_brier(f, y)
    y_bar = y.mean()
    assert _identity_holds(d)
    assert d.dsc == pytest.approx(0.0, abs=1e-12)
    assert d.mcb == pytest.approx((c - y_bar) ** 2, rel=1e-6, abs=1e-8)


def test_anti_correlated_forecast_has_large_mcb():
    # f = 1 - y. PAV regresses y on f; since the map is monotone decreasing,
    # isotonic regression collapses ĉ to a constant (the base rate). So
    # DSC = 0 and MCB = mean_score - UNC > 0 with mean_score large.
    n = 3000
    y = RNG.integers(0, 2, size=n).astype(int)
    f = 1.0 - y.astype(float)
    d = corp_decompose_brier(f, y)
    assert _identity_holds(d)
    assert d.dsc == pytest.approx(0.0, abs=1e-10)
    assert d.mcb > 0.1
    assert d.mean_score > d.unc


def test_nonbinary_observed_raises():
    with pytest.raises(ValueError):
        corp_decompose_brier([0.1, 0.2], [0.5, 0.5])


def test_out_of_range_forecast_raises():
    with pytest.raises(ValueError):
        corp_decompose_brier([1.5, 0.2], [1, 0])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        corp_decompose_brier([0.1, 0.2, 0.3], [0, 1])


def test_all_nan_raises():
    with pytest.raises(ValueError):
        corp_decompose_brier([np.nan, np.nan], [0, 1])


def test_drops_nan_pairs():
    f = np.array([0.1, np.nan, 0.9, 0.5])
    y = np.array([0, 1, 1, np.nan])
    d = corp_decompose_brier(f, y)
    assert d.n == 2


def test_monotonicity_of_calibration_curve():
    n = 2000
    f = RNG.uniform(0, 1, size=n)
    y = (RNG.uniform(0, 1, size=n) < f**2).astype(int)
    d = corp_decompose_brier(f, y)
    diffs = np.diff(d.calibrated_y)
    assert np.all(diffs >= -1e-12)


def test_diagram_returns_shape_compatible():
    df = pd.DataFrame(
        {
            "predicted_prob": RNG.uniform(0, 1, size=400),
            "observed_onset": RNG.integers(0, 2, size=400),
            "bin_label": ["Days 1-5"] * 400,
        }
    )
    fig, ax, results_df, decomp = corp_reliability_diagram(
        df,
        model="TESTMODEL",
        verification_window=(120, 180),
        day_bins=[(1, 5)],
        save_fig=False,
        dir_fig="/tmp",
        extract_bins=False,
        show_plot=False,
    )
    import matplotlib.pyplot as plt

    assert fig is not None
    assert "forecast_prob" in results_df.columns
    assert "calibrated_prob" in results_df.columns
    assert isinstance(decomp, CORPDecomposition)
    assert _identity_holds(decomp)
    plt.close(fig)
