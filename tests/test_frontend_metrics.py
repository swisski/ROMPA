"""Tests for frontend metric wrappers — specifically the interactions
added in the five-point audit pass (fair CRPS default, all-NaN cell
handling, shape consistency across single-year vs multi-year)."""
from __future__ import annotations

import numpy as np
import xarray as xr
import pytest

from frontend.api.metrics import (
    compute_crps, compute_progression, corp_inputs, compute_corp_pooled,
)
from frontend.api.aggregate import (
    aggregate_crps, aggregate_progression, expand_years, _stack,
)


def _field(values, lats, lons):
    return xr.DataArray(values, coords={"lat": lats, "lon": lons},
                        dims=("lat", "lon"))


def _ens(values, lats, lons, n_members=3):
    stacked = np.broadcast_to(values, (n_members,) + values.shape).copy()
    return xr.DataArray(
        stacked, dims=("member", "lat", "lon"),
        coords={"member": np.arange(n_members), "lat": lats, "lon": lons},
    )


# ---------------------------------------------------------------------------
# compute_crps: land-mask / all-NaN-cell regression (audit bug A-2)
# ---------------------------------------------------------------------------

def test_compute_crps_nulls_all_nan_cells_instead_of_counting_them_as_zero():
    """A cell where obs is NaN AND every ensemble member is NaN would map
    both sides to the same sentinel and yield CRPS = 0. Those synthetic
    zeros must NOT enter the reported mean or n_finite."""
    lats = np.arange(0.0, 4.0)
    lons = np.arange(0.0, 5.0)
    obs_vals = np.array([
        [150.0, 151.0, 152.0, 153.0, 154.0],
        [155.0, 156.0, np.nan, np.nan, np.nan],   # ocean / outside-mask
        [160.0, 161.0, np.nan, np.nan, np.nan],
        [165.0, 166.0, np.nan, np.nan, np.nan],
    ])
    ens_vals = np.broadcast_to(obs_vals + 5.0, (4,) + obs_vals.shape).copy()
    ens_vals[:, 1:, 2:] = np.nan                   # same missing mask on ens

    obs = _field(obs_vals, lats, lons)
    ens = xr.DataArray(
        ens_vals, dims=("member", "lat", "lon"),
        coords={"member": np.arange(4), "lat": lats, "lon": lons},
    )
    out = compute_crps(ens, obs, season_end=220)

    # Only the 11 "land" cells should be counted; the 9 NaN cells must
    # appear as null in the field payload, not 0.
    n_expected_land = int(np.isfinite(obs_vals).sum())
    assert out["n_finite"] == n_expected_land
    flat = [
        v for row in out["field"]["values"] for v in row
    ]
    # Every masked cell renders as None (JSON null), not 0.
    assert flat[1 * 5 + 2] is None
    assert flat[2 * 5 + 3] is None
    # Reported mean only averages real cells.
    real_cells = np.array(
        [v for v in flat if v is not None], dtype=float
    )
    assert out["mean"] == pytest.approx(float(real_cells.mean()), rel=1e-12)


def test_compute_crps_uses_fair_for_ensembles_and_raw_for_det():
    lats = np.array([0.0, 1.0]); lons = np.array([0.0, 1.0])
    obs = _field(np.full((2, 2), 150.0), lats, lons)
    ens = _ens(np.full((2, 2), 148.0), lats, lons, n_members=11)
    out = compute_crps(ens, obs, season_end=220)
    assert out["fair"] is True and out["n_members"] == 11

    det = ens.isel(member=[0])  # single-member
    out_det = compute_crps(det, obs, season_end=220)
    assert out_det["fair"] is False and out_det["n_members"] == 1


# ---------------------------------------------------------------------------
# compute_progression single-year vs aggregate_progression shape parity
# (audit bug B-2)
# ---------------------------------------------------------------------------

def test_single_and_multi_year_progression_share_season_schema():
    lats = np.arange(0.0, 5.0); lons = np.arange(0.0, 5.0)
    obs = _field(np.full((5, 5), 150.0), lats, lons)
    fcst = _field(np.full((5, 5), 155.0), lats, lons)
    single = compute_progression(fcst, None, obs, days=[140, 150, 160])
    multi = aggregate_progression([single, single, single])

    # Every uncertainty-band key present in multi must also be present in
    # single (q25 / q75 / ci_lo / ci_hi). The single-year payload uses
    # None placeholders; what matters is shape parity for the frontend.
    multi_season_keys = set(multi["season"].keys())
    single_season_keys = set(single["season"].keys())
    suffixes = ("_q25", "_q75", "_ci_lo", "_ci_hi")
    band_keys = {k for k in multi_season_keys if k.endswith(suffixes)}
    missing = band_keys - single_season_keys
    assert not missing, f"single-year season is missing {missing}"


# ---------------------------------------------------------------------------
# Bootstrap CIs on the multi-year progression aggregate
# ---------------------------------------------------------------------------

def _make_progression_payload(*, ioe_curve, season_ioe, days=(140, 150, 160),
                              season_extent=None, season_misp=None):
    """Build a stub per-year progression payload with the schema shape that
    aggregate_progression expects.

    By default the payload represents pure misplacement (extent_season=0,
    misp_season=ioe_season). Override season_extent / season_misp to test
    other decompositions; the per-day curves do not need to match the
    season scalars for the season-aggregator path.
    """
    days = list(days)
    n = len(days)
    if season_extent is None:
        season_extent = 0.0
    if season_misp is None:
        season_misp = float(season_ioe)
    return {
        "days": days,
        "ioe_km2":          list(ioe_curve),
        "extent_km2":       [0.0] * n,
        "misplacement_km2": list(ioe_curve),
        "sps_km2":          list(ioe_curve),
        "season": {
            "ioe_km2_day":          float(season_ioe),
            "extent_km2_day":       float(season_extent),
            "misplacement_km2_day": float(season_misp),
            "sps_km2_day":          float(season_ioe),
        },
    }


def test_aggregate_progression_emits_ci_keys():
    payloads = [
        _make_progression_payload(ioe_curve=[1.0, 2.0, 3.0], season_ioe=2.0),
        _make_progression_payload(ioe_curve=[1.5, 2.5, 3.5], season_ioe=2.5),
        _make_progression_payload(ioe_curve=[2.0, 3.0, 4.0], season_ioe=3.0),
        _make_progression_payload(ioe_curve=[2.5, 3.5, 4.5], season_ioe=3.5),
        _make_progression_payload(ioe_curve=[3.0, 4.0, 5.0], season_ioe=4.0),
    ]
    out = aggregate_progression(payloads, n_resamples=200)

    # Curve-level CIs.
    for k in ("ioe_km2", "extent_km2", "misplacement_km2", "sps_km2"):
        assert f"{k}_ci_lo" in out, k
        assert f"{k}_ci_hi" in out, k
        # ci_lo <= median <= ci_hi at every position where finite.
        for med, lo, hi in zip(out[k], out[f"{k}_ci_lo"], out[f"{k}_ci_hi"]):
            if med is None:
                continue
            assert lo is None or lo <= med + 1e-9
            assert hi is None or med <= hi + 1e-9

    # Season-level CIs.
    season = out["season"]
    for k in ("ioe_km2_day", "extent_km2_day",
              "misplacement_km2_day", "sps_km2_day"):
        med = season[k]
        lo = season[f"{k}_ci_lo"]
        hi = season[f"{k}_ci_hi"]
        if med is None:
            continue
        assert lo is None or lo <= med + 1e-9
        assert hi is None or med <= hi + 1e-9

    assert out["bootstrap_method"] == "percentile-pair-bootstrap-median"
    assert out["ci_level"] == 0.95
    assert out["n_resamples"] == 200


def test_aggregate_progression_single_year_ci_collapses_to_point():
    payload = _make_progression_payload(
        ioe_curve=[1.0, 2.0, 3.0], season_ioe=2.0,
    )
    out = aggregate_progression([payload], n_resamples=100)
    # With one year, the bootstrap distribution is a point mass; the
    # IQR collapses (q25 == q75 == median) and so does the CI.
    for med, lo, hi in zip(out["ioe_km2"], out["ioe_km2_ci_lo"], out["ioe_km2_ci_hi"]):
        assert lo == med
        assert hi == med
    season = out["season"]
    assert season["ioe_km2_day_ci_lo"] == season["ioe_km2_day"]
    assert season["ioe_km2_day_ci_hi"] == season["ioe_km2_day"]


def test_aggregate_progression_seed_is_deterministic():
    payloads = [
        _make_progression_payload(ioe_curve=[1.0, 2.0], season_ioe=1.0,
                                  days=(140, 150)),
        _make_progression_payload(ioe_curve=[2.0, 4.0], season_ioe=3.0,
                                  days=(140, 150)),
        _make_progression_payload(ioe_curve=[3.0, 6.0], season_ioe=5.0,
                                  days=(140, 150)),
        _make_progression_payload(ioe_curve=[4.0, 8.0], season_ioe=7.0,
                                  days=(140, 150)),
    ]
    a = aggregate_progression(payloads, n_resamples=300, seed=12345)
    b = aggregate_progression(payloads, n_resamples=300, seed=12345)
    assert a["ioe_km2_ci_lo"] == b["ioe_km2_ci_lo"]
    assert a["ioe_km2_ci_hi"] == b["ioe_km2_ci_hi"]
    assert a["season"]["ioe_km2_day_ci_lo"] == b["season"]["ioe_km2_day_ci_lo"]


def test_aggregate_progression_emits_peak_block():
    payloads = [
        _make_progression_payload(ioe_curve=[1.0, 5.0, 2.0], season_ioe=8.0),
        _make_progression_payload(ioe_curve=[1.0, 5.0, 2.0], season_ioe=8.0),
        _make_progression_payload(ioe_curve=[1.0, 5.0, 2.0], season_ioe=8.0),
    ]
    out = aggregate_progression(payloads, n_resamples=200)
    peak = out.get("peak")
    assert peak is not None
    # Every per-year curve peaks at day 150 -> per-year peak DOYs are
    # all 150, median is 150, CI collapses to 150.
    assert peak["ioe_doy"] == 150.0
    assert peak["ioe_doy_ci_lo"] == 150.0
    assert peak["ioe_doy_ci_hi"] == 150.0
    assert peak["ioe_value"] == 5.0


def test_aggregate_progression_peak_ci_brackets_median_when_jittered():
    # Years where the peak day jitters between 140 and 150 — the median
    # peak should be in {140, 150} and the CI should bracket it.
    payloads = []
    for d in (140, 140, 140, 150, 150, 150, 150):
        if d == 140:
            curve = [5.0, 1.0, 0.5]   # peak at 140
        else:
            curve = [0.5, 5.0, 1.0]   # peak at 150
        payloads.append(_make_progression_payload(
            ioe_curve=curve, season_ioe=6.0, days=(140, 150, 160),
        ))
    out = aggregate_progression(payloads, n_resamples=500)
    peak = out["peak"]
    assert peak["ioe_doy"] in (140.0, 150.0)
    assert peak["ioe_doy_ci_lo"] <= peak["ioe_doy"] <= peak["ioe_doy_ci_hi"]
    # The CI must lie within the observed peak-DOY range.
    assert peak["ioe_doy_ci_lo"] >= 140.0
    assert peak["ioe_doy_ci_hi"] <= 150.0


def test_aggregate_progression_misp_frac_pure_extent():
    # Every year has IOE = 100 and extent = 100, misp = 0 -> pure extent.
    payloads = [
        _make_progression_payload(
            ioe_curve=[100.0], season_ioe=100.0, days=(150,),
            season_extent=100.0, season_misp=0.0,
        ) for _ in range(5)
    ]
    out = aggregate_progression(payloads, n_resamples=200)
    assert out["season"]["misp_frac"] == 0.0
    assert out["season"]["misp_frac_ci_lo"] == 0.0
    assert out["season"]["misp_frac_ci_hi"] == 0.0


def test_aggregate_progression_misp_frac_pure_misplacement():
    # Every year has IOE = 100 and misp = 100, extent = 0 -> pure misplacement.
    payloads = [
        _make_progression_payload(
            ioe_curve=[100.0], season_ioe=100.0, days=(150,),
            season_extent=0.0, season_misp=100.0,
        ) for _ in range(5)
    ]
    out = aggregate_progression(payloads, n_resamples=200)
    assert out["season"]["misp_frac"] == 1.0
    assert out["season"]["misp_frac_ci_lo"] == 1.0
    assert out["season"]["misp_frac_ci_hi"] == 1.0


def test_aggregate_progression_misp_frac_brackets_median():
    # Years with a mix of decomposition shapes; the bootstrap CI should
    # bracket the realised median of misp/IOE ratios.
    payloads = [
        _make_progression_payload(season_ioe=100.0, ioe_curve=[100.0],
                                  days=(150,), season_extent=70.0,
                                  season_misp=30.0),
        _make_progression_payload(season_ioe=100.0, ioe_curve=[100.0],
                                  days=(150,), season_extent=50.0,
                                  season_misp=50.0),
        _make_progression_payload(season_ioe=100.0, ioe_curve=[100.0],
                                  days=(150,), season_extent=20.0,
                                  season_misp=80.0),
        _make_progression_payload(season_ioe=100.0, ioe_curve=[100.0],
                                  days=(150,), season_extent=40.0,
                                  season_misp=60.0),
        _make_progression_payload(season_ioe=100.0, ioe_curve=[100.0],
                                  days=(150,), season_extent=60.0,
                                  season_misp=40.0),
    ]
    out = aggregate_progression(payloads, n_resamples=500)
    s = out["season"]
    # Realised ratios: [0.30, 0.50, 0.80, 0.60, 0.40]; median = 0.50.
    assert s["misp_frac"] == pytest.approx(0.50, abs=1e-9)
    assert s["misp_frac_ci_lo"] <= s["misp_frac"] + 1e-12
    assert s["misp_frac"] <= s["misp_frac_ci_hi"] + 1e-12
    # CI should lie within the observed-ratio range [0.3, 0.8].
    assert 0.30 - 1e-12 <= s["misp_frac_ci_lo"]
    assert s["misp_frac_ci_hi"] <= 0.80 + 1e-12


def test_aggregate_progression_misp_frac_skips_zero_ioe_years():
    # Years with IOE = 0 (perfect forecast) cannot contribute a finite
    # ratio; they must be dropped before aggregation, not treated as 0/0.
    payloads = [
        _make_progression_payload(season_ioe=0.0, ioe_curve=[0.0],
                                  days=(150,), season_extent=0.0,
                                  season_misp=0.0),
        _make_progression_payload(season_ioe=100.0, ioe_curve=[100.0],
                                  days=(150,), season_extent=20.0,
                                  season_misp=80.0),
        _make_progression_payload(season_ioe=100.0, ioe_curve=[100.0],
                                  days=(150,), season_extent=40.0,
                                  season_misp=60.0),
    ]
    out = aggregate_progression(payloads, n_resamples=200)
    s = out["season"]
    # Only 2 finite-ratio years contributed; ratios = [0.80, 0.60], median 0.70.
    assert s["misp_frac_n_years"] == 2
    assert s["misp_frac"] == pytest.approx(0.70, abs=1e-9)


def test_aggregate_progression_iqr_and_ci_are_distinct_concepts():
    # With many years from a wide-spread distribution, the IQR (year-to-year
    # spread) should be wider than the CI (uncertainty in the median).
    rng = np.random.default_rng(0)
    payloads = []
    for _ in range(40):
        v = float(rng.normal(100.0, 25.0))
        payloads.append(_make_progression_payload(
            ioe_curve=[v, v, v], season_ioe=v,
        ))
    out = aggregate_progression(payloads, n_resamples=1000)
    season = out["season"]
    iqr = season["ioe_km2_day_q75"] - season["ioe_km2_day_q25"]
    ci_width = season["ioe_km2_day_ci_hi"] - season["ioe_km2_day_ci_lo"]
    # 95% CI on the median of N=40 should be much narrower than the IQR
    # by roughly a factor of sqrt(N) (asymptotic median-CI scaling).
    assert ci_width < iqr


# ---------------------------------------------------------------------------
# aggregate_crps: empty-fields fallback shape (audit bug B-1)
# ---------------------------------------------------------------------------

def test_aggregate_crps_empty_input_schema():
    out_empty = aggregate_crps([])
    for k in ("field", "mean", "max", "n_finite", "n_years",
              "median", "q25", "q75"):
        assert k in out_empty, f"missing key {k!r}"

    # Years requested but all yielded no field -> same schema.
    stub = {"mean": None, "max": None, "n_finite": 0}  # no "field"
    out_nofields = aggregate_crps([stub, stub])
    for k in ("field", "mean", "max", "n_finite", "n_years",
              "median", "q25", "q75"):
        assert k in out_nofields, f"missing key {k!r}"
    assert out_nofields["n_years"] == 2


# ---------------------------------------------------------------------------
# expand_years: error handling (audit bug B-3)
# ---------------------------------------------------------------------------

def test_expand_years_inverted_range_raises():
    with pytest.raises(ValueError, match="inverted"):
        expand_years("2020-2019", None)


def test_expand_years_bad_token_raises_clean():
    with pytest.raises(ValueError, match="malformed"):
        expand_years("abc", None)


def test_expand_years_bad_range_bound_raises_clean():
    with pytest.raises(ValueError, match="malformed"):
        expand_years("2019-abc", None)


def test_expand_years_accepts_mix_of_csv_and_range():
    yrs = expand_years("2019,2021-2023", None)
    assert yrs == [2019, 2021, 2022, 2023]


# ---------------------------------------------------------------------------
# _stack: ragged-input guard (audit bug B-6)
# ---------------------------------------------------------------------------

def test_stack_raises_on_ragged_input():
    with pytest.raises(ValueError, match="differing lengths"):
        _stack([[1.0, 2.0, 3.0], [4.0, 5.0]])


# ---------------------------------------------------------------------------
# corp_inputs / compute_corp_pooled: pooled CORP identity
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Land mask disambiguation (audit bug A-1)
# ---------------------------------------------------------------------------

def test_land_mask_exact_match_wins_over_substring():
    """'Niger' is a prefix of 'Nigeria' in the Natural Earth country list.
    The fix must pick exact Niger, not silently select Nigeria via substring."""
    pytest.importorskip("regionmask")
    import os
    from frontend.api import app as A

    da = xr.DataArray(
        np.zeros((3, 3)),
        coords={"lat": [12., 14., 16.], "lon": [7., 8., 9.]},  # Niger domain
        dims=("lat", "lon"),
    )

    os.environ["ROMP_LAND_MASK"] = "Niger"
    A._LAND_MASK_CACHE.clear()
    mask = A._land_mask_for(da)
    assert mask is not None, "exact 'Niger' should match"
    # The 3x3 above covers southern Niger; at least some cells should be land
    # AND all should be Niger-country (not Nigeria which is further south).
    assert mask.any(), "Niger mask should include at least one of those cells"


def test_land_mask_ambiguous_substring_rejected():
    """'Korea' is not an exact country name and matches both North and South
    Korea. The fix must reject ambiguous inputs rather than silently pick one."""
    pytest.importorskip("regionmask")
    import os
    from frontend.api import app as A

    da = xr.DataArray(
        np.zeros((3, 3)),
        coords={"lat": [36., 37., 38.], "lon": [126., 127., 128.]},
        dims=("lat", "lon"),
    )

    os.environ["ROMP_LAND_MASK"] = "Korea"
    A._LAND_MASK_CACHE.clear()
    mask = A._land_mask_for(da)
    # Ambiguous input -> ValueError is caught, warning printed, mask = None.
    assert mask is None, "ambiguous 'Korea' should be rejected, not silently picked"


# ---------------------------------------------------------------------------
# Moran's I + effective sample size (audit follow-up)
# ---------------------------------------------------------------------------

from frontend.api.metrics import moran_i_2d, effective_sample_size


def test_moran_i_random_field_near_zero():
    rng = np.random.default_rng(42)
    field = rng.standard_normal((20, 20))
    i = moran_i_2d(field)
    assert -0.2 < i < 0.2, f"random field Moran I should be near 0, got {i}"


def test_moran_i_gradient_field_strongly_positive():
    """A smooth north-south gradient is strongly spatially autocorrelated."""
    lat = np.arange(0.0, 20.0)
    field = np.broadcast_to(lat[:, None], (20, 20)).astype(float)
    i = moran_i_2d(field)
    assert i > 0.8, f"gradient field Moran I should be > 0.8, got {i}"


def test_moran_i_checkerboard_strongly_negative():
    ii, jj = np.indices((10, 10))
    field = ((ii + jj) % 2).astype(float)   # +1, 0 alternating
    i = moran_i_2d(field)
    assert i < -0.8, f"checkerboard Moran I should be < -0.8, got {i}"


def test_moran_i_nan_safety():
    field = np.full((6, 6), np.nan)
    field[0, 0] = 1.0
    field[0, 1] = 2.0
    field[1, 0] = 3.0
    field[1, 1] = 4.0
    # 4 finite cells, but contiguous — should return a real I.
    i = moran_i_2d(field)
    assert not np.isnan(i)
    # Too few finite cells -> NaN
    tiny = np.full((3, 3), np.nan)
    tiny[0, 0] = 1.0
    assert np.isnan(moran_i_2d(tiny))


def test_effective_sample_size_shrinks_with_positive_autocorrelation():
    assert effective_sample_size(100, 0.0) == 100.0
    assert effective_sample_size(100, -0.2) == 100.0   # negative I clamped to no-shrink
    eff = effective_sample_size(100, 0.5)
    assert eff == pytest.approx(100 * 0.5 / 1.5, rel=1e-9)
    assert effective_sample_size(100, 0.99) >= 1.0     # floored at 1


def test_pooled_corp_identity_holds():
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([0.0, 1.0, 2.0])
    rng = np.random.default_rng(7)
    obs_vals = rng.uniform(140, 170, size=(3, 3))
    ens_vals = obs_vals[None, :, :] + rng.normal(0, 5, size=(10, 3, 3))
    obs = _field(obs_vals, lats, lons)
    ens = xr.DataArray(
        ens_vals, dims=("member", "lat", "lon"),
        coords={"member": np.arange(10), "lat": lats, "lon": lons},
    )
    p, y = corp_inputs(ens, None, obs, tau=155, season_end=200)
    out = compute_corp_pooled(p, y, tau=155)
    assert abs(out["identity_residual"]) < 1e-9
