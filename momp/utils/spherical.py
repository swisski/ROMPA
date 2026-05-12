"""Shared spherical-geometry constants and helpers.

Centralises the Earth radius, the degree-to-km conversion derived from it,
and the per-cell area formula for a regular lat/lon grid. The displacement,
progression, and isochrone modules previously declared these independently.
"""

from __future__ import annotations

import math

import numpy as np


EARTH_RADIUS_KM = 6371.0088
KM_PER_DEG = math.pi * EARTH_RADIUS_KM / 180.0


def _infer_spacing(coord: np.ndarray) -> float:
    if coord.size < 2:
        raise ValueError("need at least 2 values to infer spacing")
    diffs = np.diff(coord)
    if not np.allclose(diffs, diffs[0], rtol=1e-5, atol=1e-8):
        raise ValueError("coordinate is not uniformly spaced")
    return float(abs(diffs[0]))


def cell_area_km2(
    lat: np.ndarray,
    lon: np.ndarray,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """Per-cell area in km^2 on a regular lat/lon grid, shape ``(lat, lon)``.

    Uses the standard ``R^2 * deg2rad(dlat) * deg2rad(dlon) * cos(lat)``
    formula. Requires uniformly-spaced ``lat`` and ``lon`` arrays.
    """
    dlat = _infer_spacing(lat)
    dlon = _infer_spacing(lon)
    cos_lat = np.cos(np.deg2rad(lat))
    per_lat = (earth_radius_km ** 2) * np.deg2rad(dlat) * np.deg2rad(dlon) * cos_lat
    return np.broadcast_to(per_lat[:, None], (lat.size, lon.size)).copy()
