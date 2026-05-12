"""Lookup-safety tests for ``momp.utils.land_mask.country_mask``."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from momp.utils.land_mask import country_mask


@pytest.fixture
def india_grid():
    lat = np.linspace(8.0, 35.0, 28)
    lon = np.linspace(68.0, 97.0, 30)
    return xr.DataArray(
        np.zeros((lat.size, lon.size)),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
    )


def test_exact_match_returns_bool_grid(india_grid):
    m = country_mask(india_grid, "India")
    assert m.dims == ("lat", "lon")
    assert m.dtype == bool
    assert int(m.sum()) > 0


def test_lookup_is_case_insensitive(india_grid):
    assert int(country_mask(india_grid, "india").sum()) == int(country_mask(india_grid, "India").sum())


def test_exact_match_wins_over_substring():
    # Niger is a substring of Nigeria. Without exact-first, "Niger" could
    # silently resolve to Nigeria. Use a grid that covers both countries
    # so the masks aren't trivially all-False.
    lat = np.linspace(4.0, 24.0, 21)
    lon = np.linspace(0.0, 16.0, 17)
    da = xr.DataArray(
        np.zeros((lat.size, lon.size)),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
    )
    a = country_mask(da, "Niger")
    b = country_mask(da, "Nigeria")
    assert int(a.sum()) > 0 and int(b.sum()) > 0
    assert not np.array_equal(a.values, b.values)


def test_ambiguous_substring_raises(india_grid):
    with pytest.raises(ValueError, match="ambiguous"):
        country_mask(india_grid, "Korea")


def test_missing_country_raises(india_grid):
    with pytest.raises(ValueError, match="not found"):
        country_mask(india_grid, "NotARealCountry")


def test_xarray_broadcast_works_with_member_dim(india_grid):
    # The mask is (lat, lon); xarray should broadcast it across an
    # additional 'member' dim when used with .where().
    members = xr.DataArray(
        np.ones((5, india_grid.sizes["lat"], india_grid.sizes["lon"])),
        coords={
            "member": np.arange(5),
            "lat": india_grid["lat"],
            "lon": india_grid["lon"],
        },
        dims=("member", "lat", "lon"),
    )
    masked = members.where(country_mask(india_grid, "India"))
    assert masked.dims == ("member", "lat", "lon")
    # NaN outside India for every member, 1.0 inside.
    inside = int((masked == 1.0).sum())
    outside = int(np.isnan(masked.values).sum())
    assert inside > 0 and outside > 0
