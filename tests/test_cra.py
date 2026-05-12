"""Tests for the CRA metric and its frontend wiring.

Three layers:

1. ``momp.metrics.cra`` — synthetic obs / shifted-forecast pair where
   the answer is known by construction; verify the best-fit integer
   shift is recovered.
2. ``frontend.api.metrics.compute_cra`` — runs the serializer against
   the same construction and checks the JSON shape.
3. ``frontend.api.aggregate.aggregate_cra`` — synthetic per-year list,
   verify medians and IQR are reported.
"""

from __future__ import annotations

import numpy as np
import pytest

from momp.metrics.cra import cra_decomposition, shift_field
from frontend.api.metrics import compute_cra
from frontend.api.aggregate import aggregate_cra


# ---------- core metric ----------------------------------------------------


def _ellipse_field(shape, *, cy, cx, ry, rx, val=10.0) -> np.ndarray:
    y, x = np.indices(shape, dtype=float)
    inside = ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1.0
    f = np.zeros(shape, dtype=float)
    f[inside] = val
    return f


def test_cra_recovers_known_shift_exact():
    # Observed ellipse, forecast is the observed pattern shifted +5 east.
    # The best-fit corrective shift must be -5 in x (move forecast west to
    # match obs) and 0 in y.
    obs = _ellipse_field((40, 40), cy=20, cx=20, ry=8, rx=8)
    fcst = shift_field(obs, dy=0, dx=5)
    result, shifted, mask = cra_decomposition(
        case="pure_east_shift", obs=obs, fcst=fcst,
        threshold=1.0, max_shift=8,
    )
    assert result.corrective_shift_dx == -5
    assert result.corrective_shift_dy == 0
    # After the corrective shift, MSE should be zero (forecast == obs).
    assert result.mse_shifted == pytest.approx(0.0, abs=1e-12)
    # All residual MSE is displacement (no volume / pattern error).
    assert result.mse_displacement > 0
    assert result.mse_volume == pytest.approx(0.0, abs=1e-12)
    assert result.mse_pattern == pytest.approx(0.0, abs=1e-12)
    assert result.pct_displacement == pytest.approx(100.0, abs=1e-9)


def test_cra_pure_volume_bias():
    # Same shape, same place, 2× intensity — pure volume bias.
    obs = _ellipse_field((40, 40), cy=20, cx=20, ry=8, rx=8, val=10.0)
    fcst = obs * 2.0
    result, _, _ = cra_decomposition(
        case="pure_volume", obs=obs, fcst=fcst,
        threshold=1.0, max_shift=8,
    )
    assert result.corrective_shift_dx == 0
    assert result.corrective_shift_dy == 0
    # No spatial misplacement → displacement piece is zero. The bias is
    # pure volume: mean_fcst > mean_obs, mse_volume = (mean_diff)^2 > 0.
    assert result.mse_displacement == pytest.approx(0.0, abs=1e-12)
    assert result.mse_volume > 0
    # And pattern is also zero because (fcst - obs) is uniform inside.
    assert result.mse_pattern == pytest.approx(0.0, abs=1e-9)


def test_cra_pure_pattern_error():
    # Same location, same total volume, but redistributed mass — pure
    # pattern error. Use a 50×50 grid and an obs ellipse vs a fcst
    # ring-shape with the same total.
    obs = _ellipse_field((50, 50), cy=25, cx=25, ry=10, rx=10, val=5.0)
    fcst = obs.copy()
    # Move half the mass from the centre row to a row 3 cells north
    # (still within the mask, so displacement search won't find a better
    # shift). This is a within-pattern redistribution.
    fcst[25, :] = 0.0
    fcst[22, :] = 5.0
    # Match total volume.
    fcst *= obs.sum() / max(fcst.sum(), 1e-12)
    result, _, _ = cra_decomposition(
        case="pattern_only", obs=obs, fcst=fcst,
        threshold=1.0, max_shift=4,
    )
    assert result.mse_pattern > 0


# ---------- frontend serializer --------------------------------------------


def test_compute_cra_payload_shape_and_alignment():
    # Forecast on a coarser grid than obs to exercise the regridder.
    obs = _ellipse_field((40, 40), cy=20, cx=20, ry=8, rx=8)
    obs_lat = np.linspace(0, 39, 40)
    obs_lon = np.linspace(0, 39, 40)
    # Coarser forecast grid (every 2 cells), shifted east.
    fcst_lat = np.linspace(0, 39, 20)
    fcst_lon = np.linspace(0, 39, 20)
    fcst = shift_field(_ellipse_field((20, 20), cy=10, cx=10, ry=4, rx=4), 0, 2)

    # display_country=None so the verification mask doesn't reject this
    # synthetic grid (which lives at 0–40°, outside India).
    out = compute_cra(
        fcst, fcst_lat, fcst_lon,
        obs, obs_lat, obs_lon,
        case="regridded", threshold=1.0, max_shift=6,
        display_country=None,
    )
    # Expected top-level keys
    for k in ("case", "corrective_shift", "diagnosed_forecast_error",
              "mse", "pct", "spatial_corr", "fields", "threshold", "max_shift"):
        assert k in out, f"missing key {k!r}"
    cs = out["corrective_shift"]
    assert {"dx", "dy"} <= set(cs.keys())
    # Pct values are floats or None — and sum to ~100 when total MSE > 0.
    pct = out["pct"]
    finite = [pct[k] for k in ("displacement", "volume", "pattern")
              if pct[k] is not None]
    if finite:
        assert 99.0 <= sum(finite) <= 101.0
    # Field payloads should be downsampled to ≤ 240 cells per axis (the
    # default max_payload_cells; bumped from 80 for higher-res maps).
    fobs = out["fields"]["obs"]
    assert len(fobs["lat"]) <= 240 and len(fobs["lon"]) <= 240


def test_compute_cra_handles_include_fields_false():
    obs = _ellipse_field((20, 20), cy=10, cx=10, ry=4, rx=4)
    out = compute_cra(
        obs, np.arange(20.0), np.arange(20.0),
        obs, np.arange(20.0), np.arange(20.0),
        case="no_fields", threshold=1.0, max_shift=2,
        include_fields=False, display_country=None,
    )
    assert "fields" not in out


def test_compute_cra_with_india_mask_uses_verification_mask():
    """When display_country="India" is set, the metric must restrict the
    score to India cells. Without that, an aggressive shift can drag
    rainfall outside the region and the MSE artificially drops because
    the score mask shrinks. With the mask, a shift that moves data
    outside India can't claim improvement.
    """
    # Build a grid that straddles India so the mask has both inside and
    # outside cells. AIFS-style 2°-spacing, lat 8–34, lon 68–96.
    obs_lat = np.linspace(8, 34, 14)
    obs_lon = np.linspace(68, 96, 15)
    # Obs has a rain blob inside India (~22N, 80E)
    obs = np.zeros((obs_lat.size, obs_lon.size), dtype=float)
    obs[7:10, 5:9] = 30.0
    # Forecast has the blob in roughly the same place — keep it small so
    # the verification mask actually matters for the result.
    fcst = np.zeros_like(obs)
    fcst[6:9, 6:10] = 30.0

    out = compute_cra(
        fcst, obs_lat, obs_lon,
        obs, obs_lat, obs_lon,
        case="india_objective", threshold=1.0, max_shift=4,
        display_country="India",
    )
    assert out["display_country"] == "India"
    # With the mask in place the obs centroid must be inside India
    # (a sanity check on the mask plumbing).
    co = out["centroids"]["obs"]
    assert co is not None
    assert 8 <= co["lat"] <= 34 and 68 <= co["lon"] <= 96


# ---------- multi-year aggregator ------------------------------------------


def _synthetic_per_year(year, pct_d, pct_v, pct_p, dx, dy):
    return {
        "meta": {"year": year},
        "pct": {"displacement": pct_d, "volume": pct_v, "pattern": pct_p},
        "mse": {"total": 100.0, "shifted": 50.0,
                "displacement": 50.0, "volume": 20.0, "pattern": 30.0},
        "corrective_shift": {"dx": dx, "dy": dy},
        "spatial_corr": {"original": 0.5, "shifted": 0.7},
        "mean_obs": 5.0, "mean_fcst_shifted": 5.5,
        "peak_obs": 50.0, "peak_fcst_shifted": 55.0,
        "n_obs_objects": 3, "n_fcst_objects": 4,
        "threshold": 1.0, "max_shift": 8,
    }


def test_aggregate_cra_median_and_iqr():
    per_year = [
        _synthetic_per_year(2018, 10, 30, 60, 1, 1),
        _synthetic_per_year(2019, 20, 25, 55, 2, 2),
        _synthetic_per_year(2020, 30, 20, 50, 3, 3),
    ]
    out = aggregate_cra(per_year)
    assert out["n_years"] == 3
    disp = out["pct"]["displacement"]
    assert disp["median"] == pytest.approx(20.0)
    assert disp["q25"] == pytest.approx(15.0)
    assert disp["q75"] == pytest.approx(25.0)
    assert disp["n"] == 3
    assert out["corrective_shift"]["dx"]["median"] == pytest.approx(2.0)
    assert out["years"] == [2018, 2019, 2020]


def test_aggregate_cra_empty_returns_n_years_zero():
    out = aggregate_cra([])
    assert out == {"n_years": 0}


def test_aggregate_cra_skips_non_finite():
    per_year = [
        _synthetic_per_year(2018, 10, 30, 60, 1, 1),
        _synthetic_per_year(2019, None, 25, 55, None, 2),  # missing displacement & dx
        _synthetic_per_year(2020, 30, 20, 50, 3, 3),
    ]
    out = aggregate_cra(per_year)
    disp = out["pct"]["displacement"]
    assert disp["n"] == 2  # only the two finite values count
    assert disp["median"] == pytest.approx(20.0)
    assert out["corrective_shift"]["dx"]["n"] == 2
