"""End-to-end integration: build onset-DOY fields from raw AIFS / NGCM / IMD
demo rainfall using ROMP's production `detect_onset`, then exercise every
new Milestone-1 / Milestone-2 metric. Verifies the metric implementations
work on real production-detector output, not just synthetic known-answer
cases.

Marked ``integration`` and skipped automatically when the demo data files
are absent. Run with:

    pytest tests/test_integration_real_pipeline.py -m integration -v
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.stats.detect import detect_onset

DEMO = Path("/home/alex/classwork/DSICLINIC/ROMPA/demo/data")
AIFS = DEMO / "aifs" / "2015.nc"
NGCM = DEMO / "ngcm" / "2015.nc"
OBS = DEMO / "obs" / "2015.nc"

ONSET_KW = dict(wet_init=1.0, wet_spell=3, dry_spell=7, dry_threshold=1.0, dry_extent=0)
WET_THRESH = 20.0

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (AIFS.exists() and NGCM.exists() and OBS.exists()),
        reason="demo data missing (AIFS / NGCM / IMD 2015 NetCDFs)",
    ),
]


def _first_onset_doy(rain_series: np.ndarray, start_doy: int) -> float:
    n = rain_series.size
    for offset in range(n - ONSET_KW["wet_spell"] + 1):
        if detect_onset(
            day=offset + 1,
            forecast_series=rain_series,
            thresh=WET_THRESH,
            **ONSET_KW,
        ):
            return float(start_doy + offset)
    return float("nan")


def _obs_onset_field() -> xr.DataArray:
    rain = xr.open_dataset(OBS)["RAINFALL"]
    times = pd.to_datetime(rain["TIME"].values)
    sub = rain.isel(TIME=np.where((times.month >= 5) & (times.month <= 9))[0])
    start_doy = int(pd.Timestamp(sub["TIME"].values[0]).dayofyear)
    vals = sub.values
    out = np.full(vals.shape[1:], np.nan)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = _first_onset_doy(vals[:, i, j], start_doy)
    return xr.DataArray(
        out, coords={"lat": sub["lat"].values, "lon": sub["lon"].values},
        dims=("lat", "lon"), name="onset_doy",
    )


def _fcst_onset_field(fcst_da: xr.DataArray, init_select: int) -> xr.DataArray:
    has_member = "number" in fcst_da.dims
    sel = fcst_da.isel(time=init_select)
    start_doy = pd.Timestamp(fcst_da["time"].values[init_select]).dayofyear
    if has_member:
        arr = sel.transpose("number", "day", "lat", "lon").values
        M, _, Ny, Nx = arr.shape
        out = np.full((M, Ny, Nx), np.nan)
        for m in range(M):
            for i in range(Ny):
                for j in range(Nx):
                    out[m, i, j] = _first_onset_doy(arr[m, :, i, j], start_doy)
        return xr.DataArray(
            out, dims=("member", "lat", "lon"),
            coords={"member": sel["number"].values,
                    "lat": sel["lat"].values, "lon": sel["lon"].values},
            name="onset_doy",
        )
    arr = sel.transpose("day", "lat", "lon").values
    out = np.full(arr.shape[1:], np.nan)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = _first_onset_doy(arr[:, i, j], start_doy)
    return xr.DataArray(
        out, dims=("lat", "lon"),
        coords={"lat": sel["lat"].values, "lon": sel["lon"].values},
        name="onset_doy",
    )


def _pick_overlapping_init(fcst_da: xr.DataArray, obs_lo: float, obs_hi: float):
    best_idx, best_overlap, best_range = 0, -1.0, (np.nan, np.nan)
    for k in range(fcst_da.sizes["time"]):
        f = _fcst_onset_field(fcst_da, init_select=k)
        v = f.values[np.isfinite(f.values)]
        if v.size == 0:
            continue
        lo, hi = float(v.min()), float(v.max())
        ov = max(0.0, min(hi, obs_hi) - max(lo, obs_lo))
        if ov > best_overlap:
            best_overlap, best_idx, best_range = ov, k, (lo, hi)
    return best_idx, best_range, best_overlap


@pytest.fixture(scope="module")
def real_fields():
    obs = _obs_onset_field()
    obs_v = obs.values[np.isfinite(obs.values)]
    assert obs_v.size > 0, "production detector found no obs onset"
    obs_lo, obs_hi = float(obs_v.min()), float(obs_v.max())

    aifs_da = xr.open_dataset(AIFS)["tp"]
    ngcm_da = xr.open_dataset(NGCM)["tp"]
    aifs_init, aifs_range, _ = _pick_overlapping_init(aifs_da, obs_lo, obs_hi)
    ngcm_init, ngcm_range, _ = _pick_overlapping_init(ngcm_da, obs_lo, obs_hi)

    aifs = _fcst_onset_field(aifs_da, init_select=aifs_init)
    ens = _fcst_onset_field(ngcm_da, init_select=ngcm_init)

    lo = max(obs_lo, aifs_range[0], ngcm_range[0]) + 2
    hi = min(obs_hi, aifs_range[1], ngcm_range[1]) - 2
    iso_days = sorted({int(round(x)) for x in np.linspace(lo, hi, 3)})

    return {"obs": obs, "aifs": aifs, "ens": ens, "iso_days": iso_days}


def test_censored_crps_on_real_ensemble(real_fields):
    from momp.metrics.crps import censored_crps_field
    crps = censored_crps_field(real_fields["ens"], real_fields["obs"], season_end=220)
    v = crps.values
    assert np.isfinite(v).any()
    assert np.nanmin(v) >= -1e-9


def test_fss_on_real_deterministic(real_fields):
    from momp.metrics.neighborhood import fss
    fss_da = fss(real_fields["aifs"], real_fields["obs"],
                 thresholds=real_fields["iso_days"], neighborhoods=[1, 3, 5])
    v = fss_da.values
    assert np.isfinite(v).any()
    assert np.all(np.isnan(v) | ((v >= -1e-9) & (v <= 1.0 + 1e-9)))


def test_displacement_on_real_deterministic(real_fields):
    from momp.metrics.displacement import displacement_bias_sweep
    disp = displacement_bias_sweep(real_fields["aifs"], real_fields["obs"],
                                   thresholds=real_fields["iso_days"])
    assert np.isfinite(float(disp["great_circle_km"].mean()))
    assert "area_bias_fraction" in disp


def test_ioe_pointwise_inequality_on_real_data(real_fields):
    from momp.metrics.progression import integrated_onset_error
    days = list(range(real_fields["iso_days"][0] - 5, real_fields["iso_days"][-1] + 6, 3))
    ds = integrated_onset_error(real_fields["aifs"], real_fields["obs"], days=days)
    assert np.all(ds["ioe_km2"].values + 1e-6 >= ds["extent_km2"].values)
    assert float(ds["ioe_season_km2_day"]) > 0


def test_sps_reduces_to_ioe_on_real_aifs(real_fields):
    from momp.metrics.progression import (
        spatial_probability_score, integrated_onset_error,
    )
    days = list(range(real_fields["iso_days"][0] - 5, real_fields["iso_days"][-1] + 6, 3))
    det_ens = real_fields["aifs"].expand_dims({"member": [0]}).transpose("member", "lat", "lon")
    sps = spatial_probability_score(det_ens, real_fields["obs"], days=days)
    ioe = integrated_onset_error(real_fields["aifs"], real_fields["obs"], days=days)
    assert np.allclose(sps["sps_km2"].values, ioe["ioe_km2"].values, rtol=1e-10, atol=1e-6)


def test_sps_ensemble_nonnegative_on_real_ngcm(real_fields):
    from momp.metrics.progression import spatial_probability_score
    days = list(range(real_fields["iso_days"][0] - 5, real_fields["iso_days"][-1] + 6, 3))
    sps = spatial_probability_score(real_fields["ens"], real_fields["obs"], days=days)
    assert np.all(sps["sps_km2"].values >= -1e-6)
    assert float(sps["sps_season_km2_day"]) > 0


def test_isochrone_distances_finite_at_overlapping_thresholds(real_fields):
    from momp.graphics.isochrone import isochrone_distance_sweep
    ds = isochrone_distance_sweep(real_fields["aifs"], real_fields["obs"],
                                  days=real_fields["iso_days"])
    assert np.isfinite(ds["hausdorff_km"].values).any()
    assert (ds["n_segments_fcst"].values > 0).any()
    assert (ds["n_segments_obs"].values > 0).any()


def test_corp_decomposition_identity_on_real_data(real_fields):
    from momp.graphics.corp_reliability import corp_decompose_brier
    tau = real_fields["iso_days"][len(real_fields["iso_days"]) // 2]
    m = real_fields["ens"].values
    p = np.where(np.isnan(m) | (m > 250), 0.0, (m <= tau).astype(float)).mean(axis=0)
    obs = real_fields["obs"].values
    y = np.where(np.isfinite(obs) & (obs <= tau), 1.0, 0.0)
    decomp = corp_decompose_brier(p.ravel(), y.ravel())
    residual = abs((decomp.mcb - decomp.dsc + decomp.unc) - decomp.mean_score)
    assert residual < 1e-10
    assert decomp.mean_score >= 0


def test_progression_orchestration_emits_all_metrics(real_fields):
    from momp.app.progression_verification import progression_verification
    days = list(range(real_fields["iso_days"][0] - 3, real_fields["iso_days"][-1] + 4, 3))
    ds = progression_verification(
        forecast_onset=real_fields["ens"],
        observed_onset=real_fields["obs"],
        days=days,
        isochrone_days=real_fields["iso_days"],
        season_end=220,
        member_dim="member",
    )
    for var in ("ioe_km2", "sps_km2", "hausdorff_km"):
        assert var in ds.data_vars, f"orchestration missing {var}"
