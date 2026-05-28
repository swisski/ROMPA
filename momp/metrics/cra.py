"""Contiguous Rain Area (CRA) verification helpers.

The functions here implement a compact, reusable subset of the CRA ideas used
in the demo scripts: best-fit displacement (integer-cell or sub-cell via
``subgrid_factor``), a fixed CRA score mask (held constant across all
candidate shifts to make the MSE decomposition algebraically exact), and
the original MSE-style decomposition into displacement, volume, and
pattern terms.

Identity: ``mse_total == mse_displacement + mse_volume + mse_pattern``
holds exactly (within float tolerance), enforced by an assertion in
``cra_decomposition``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class CraResult:
    case: str
    imposed_forecast_dx: float
    imposed_forecast_dy: float
    # Corrective-shift values are in NATIVE CELL UNITS. Float because
    # ``subgrid_factor > 1`` allows fractional-cell shifts (e.g. 0.125 cells
    # = 0.25° on a 2° native grid). Display: multiply by grid spacing
    # (dlat / dlon in degrees) to get the geographic shift.
    corrective_shift_dx: float
    corrective_shift_dy: float
    diagnosed_forecast_error_dx: float
    diagnosed_forecast_error_dy: float
    n_obs_objects: int
    n_fcst_objects: int
    mse_total: float
    mse_shifted: float
    mse_displacement: float
    mse_volume: float
    mse_pattern: float
    pct_displacement: float
    pct_volume: float
    pct_pattern: float
    mean_obs: float
    mean_fcst_shifted: float
    peak_obs: float
    peak_fcst_shifted: float
    spatial_corr_original: float
    spatial_corr_shifted: float


def shift_field(field: np.ndarray, dy: float, dx: float, fill: float = 0.0) -> np.ndarray:
    """Translate a 2-D field by ``(dy, dx)`` cells without wraparound.

    Integer shifts use a pure-NumPy slice path (fast, exact). Non-integer
    shifts bilinearly resample via ``scipy.ndimage.shift(..., order=1)``
    so sub-grid shifts (e.g. 0.125 cells = 0.25° on a 2° native grid)
    are well-defined.
    """
    dy_f, dx_f = float(dy), float(dx)
    is_int = (dy_f == round(dy_f)) and (dx_f == round(dx_f))

    if is_int:
        dy_i, dx_i = int(round(dy_f)), int(round(dx_f))
        shifted = np.full_like(field, fill, dtype=float)
        ny, nx = field.shape

        src_y0 = max(0, -dy_i)
        src_y1 = min(ny, ny - dy_i)
        src_x0 = max(0, -dx_i)
        src_x1 = min(nx, nx - dx_i)

        dst_y0 = max(0, dy_i)
        dst_y1 = min(ny, ny + dy_i)
        dst_x0 = max(0, dx_i)
        dst_x1 = min(nx, nx + dx_i)

        if src_y0 < src_y1 and src_x0 < src_x1:
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = field[src_y0:src_y1, src_x0:src_x1]
        return shifted

    # Fractional shift: bilinear resample. NaN cells become ``fill`` —
    # the scipy.ndimage.shift call requires finite inputs, so fill first.
    src = np.where(np.isfinite(field), field, fill).astype(float)
    return ndimage.shift(src, (dy_f, dx_f), order=1, mode="constant", cval=float(fill))


def finite_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Spatial correlation over a mask; returns NaN for constant fields."""
    aa = a[mask].ravel()
    bb = b[mask].ravel()
    valid = np.isfinite(aa) & np.isfinite(bb)

    if valid.sum() < 2:
        return np.nan

    aa = aa[valid]
    bb = bb[valid]
    if np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
        return np.nan

    return float(np.corrcoef(aa, bb)[0, 1])


def masked_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean squared error over a boolean mask."""
    valid_mask = mask & np.isfinite(a) & np.isfinite(b)
    if not np.any(valid_mask):
        return np.nan

    diff = a[valid_mask] - b[valid_mask]
    return float(np.nanmean(diff**2))


def object_count(field: np.ndarray, threshold: float) -> int:
    """Count contiguous objects above a threshold."""
    _, count = ndimage.label(np.asarray(field) >= threshold)
    return int(count)


def best_shift_by_mse(
    obs: np.ndarray,
    fcst: np.ndarray,
    *,
    threshold: float,
    max_shift: float,
    verification_mask: np.ndarray | None = None,
    dy_values=None,
    dx_values=None,
    subgrid_factor: int = 1,
):
    """
    Find the forecast translation ``(dy, dx)`` that minimizes MSE over a
    FIXED CRA score mask.

    The score mask is the analysis region — ``(obs_mask | fcst_mask)``
    optionally intersected with ``verification_mask`` — computed ONCE at
    zero shift and held constant across all candidate shifts. Holding
    the mask fixed eliminates a denominator-shopping pathology where
    the optimizer could lower the masked mean MSE just by enlarging the
    union with zero-residual cells (instead of by actually aligning
    rain). This deviates from the literal Ebert & Gallus 2009 relaxed
    mask only in that the mask doesn't grow with each candidate; the
    paper's rationale for the union (matching nearby non-overlapping
    features) is preserved because the union is still built from
    obs+fcst at zero shift.

    ``subgrid_factor=N`` searches at ``1/N``-cell resolution via
    bilinear resampling. ``N=1`` is the original integer-cell sweep
    (fastest, exact). ``N=8`` on a 2°-native grid → 0.25° shift
    granularity, which matches typical IMD obs resolution.

    Returns ``(dy, dx, shifted, score_mask, mse)`` where ``dy``/``dx``
    are floats in native cell units (so ``dy = -0.5`` means half a
    native cell south). Multiply by grid spacing for the geographic
    shift.
    """
    obs = np.asarray(obs, dtype=float)
    fcst = np.asarray(fcst, dtype=float)

    if obs.shape != fcst.shape:
        raise ValueError(f"obs and fcst must have the same shape, got {obs.shape} and {fcst.shape}")
    if verification_mask is not None and verification_mask.shape != obs.shape:
        raise ValueError(
            f"verification_mask must have shape {obs.shape}, got {verification_mask.shape}"
        )
    subgrid_factor = max(1, int(subgrid_factor))

    obs_mask = obs >= threshold
    fcst_mask = fcst >= threshold

    # FIXED score mask, computed once. Constant across all candidates →
    # no denominator-shopping.
    score_mask = obs_mask | fcst_mask
    if verification_mask is not None:
        score_mask = score_mask & verification_mask

    # Candidate shifts. With ``subgrid_factor=N``, step size is 1/N cell.
    if dy_values is None:
        step = 1.0 / subgrid_factor
        dy_values = np.arange(-max_shift, max_shift + step * 0.5, step)
    if dx_values is None:
        step = 1.0 / subgrid_factor
        dx_values = np.arange(-max_shift, max_shift + step * 0.5, step)

    # Tie-break by minimum-magnitude shift: when two candidates score
    # equal MSE (common in pure-volume-bias or constant-interior cases
    # where many shifts hit the same denominator/numerator), prefer the
    # one closer to (0, 0). Avoids the "first-encountered" lottery that
    # would otherwise pick the maximum negative shift just because the
    # search iterates from -max_shift up.
    best = None  # (mse, magnitude, dy, dx, shifted, score_mask)
    for dy in dy_values:
        for dx in dx_values:
            shifted = shift_field(fcst, float(dy), float(dx))
            mse = masked_mse(shifted, obs, score_mask)

            if np.isnan(mse):
                continue
            mag = abs(float(dy)) + abs(float(dx))
            if best is None:
                best = (mse, mag, float(dy), float(dx), shifted, score_mask)
                continue
            tol = max(1e-12, 1e-12 * abs(best[0]))
            if mse < best[0] - tol:
                best = (mse, mag, float(dy), float(dx), shifted, score_mask)
            elif mse < best[0] + tol and mag < best[1]:
                best = (mse, mag, float(dy), float(dx), shifted, score_mask)

    if best is None:
        raise RuntimeError("No valid shift found")

    # Return in the legacy order: (dy, dx, shifted, score_mask, mse).
    return (best[2], best[3], best[4], best[5], best[0])


def cra_decomposition(
    case: str,
    obs: np.ndarray,
    fcst: np.ndarray,
    *,
    imposed_forecast_dx: float = np.nan,
    imposed_forecast_dy: float = np.nan,
    threshold: float = 1.0,
    max_shift: float = 80,
    verification_mask: np.ndarray | None = None,
    dy_values=None,
    dx_values=None,
    subgrid_factor: int = 1,
) -> tuple[CraResult, np.ndarray, np.ndarray]:
    """Compute a CRA-style MSE decomposition for one forecast-observation pair.

    All MSE computations — total, shifted, displacement, volume, pattern —
    are evaluated on a SINGLE fixed score mask (obs_mask ∪ fcst_mask,
    optionally intersected with verification_mask). Holding the mask
    constant across mse_total and mse_shifted is what makes the
    algebraic identity ``mse_total = mse_displacement + mse_volume +
    mse_pattern`` hold exactly; the per-candidate union mask the
    previous implementation used could leak the identity.

    Pass ``subgrid_factor > 1`` to search at sub-native-cell resolution
    via bilinear resampling. Default 1 (integer-cell shifts).
    """
    obs = np.asarray(obs, dtype=float)
    fcst = np.asarray(fcst, dtype=float)
    dy, dx, shifted, score_mask, mse_shifted = best_shift_by_mse(
        obs,
        fcst,
        threshold=threshold,
        max_shift=max_shift,
        verification_mask=verification_mask,
        dy_values=dy_values,
        dx_values=dx_values,
        subgrid_factor=subgrid_factor,
    )

    # SAME score mask for mse_total — what made the identity break in
    # the old implementation was scoring mse_total against a smaller
    # union mask than mse_shifted.
    valid_mask = score_mask & np.isfinite(obs) & np.isfinite(shifted)
    mse_total = masked_mse(fcst, obs, score_mask)
    mean_obs = float(np.nanmean(obs[valid_mask])) if valid_mask.any() else float("nan")
    mean_fcst_shifted = float(np.nanmean(shifted[valid_mask])) if valid_mask.any() else float("nan")

    # Decomposition identity (exact algebra with the shared mask):
    #   mse_total == mse_displacement + mse_volume + mse_pattern.
    # mse_shifted minimizes MSE over the search window so it must be
    # ≤ mse_total (zero-shift is in the window). mse_volume is the
    # bias-squared piece. mse_pattern is what's left after volume.
    # Negative values within float tolerance are a numerical hiccup,
    # not a metric pathology — clip to zero AFTER asserting.
    mse_displacement = mse_total - mse_shifted
    mse_volume = (mean_fcst_shifted - mean_obs) ** 2
    mse_pattern = mse_shifted - mse_volume

    # Tolerance: a few ULPs above zero. If the assertion fires the
    # decomposition is genuinely broken — investigate, don't silently
    # clip with max(.,0) as the old code did.
    tol = max(1e-9, 1e-9 * abs(mse_total))
    assert mse_displacement >= -tol, f"mse_displacement negative ({mse_displacement}); investigate"
    assert mse_pattern >= -tol, f"mse_pattern negative ({mse_pattern}); investigate"
    mse_displacement = max(mse_displacement, 0.0)
    mse_pattern = max(mse_pattern, 0.0)

    if mse_total > 0:
        pct_displacement = 100.0 * mse_displacement / mse_total
        pct_volume = 100.0 * mse_volume / mse_total
        pct_pattern = 100.0 * mse_pattern / mse_total
    else:
        pct_displacement = pct_volume = pct_pattern = np.nan

    original_corr = finite_corr(fcst, obs, valid_mask)
    shifted_corr = finite_corr(shifted, obs, valid_mask)

    result = CraResult(
        case=case,
        imposed_forecast_dx=imposed_forecast_dx,
        imposed_forecast_dy=imposed_forecast_dy,
        corrective_shift_dx=float(dx),
        corrective_shift_dy=float(dy),
        diagnosed_forecast_error_dx=float(-dx),
        diagnosed_forecast_error_dy=float(-dy),
        n_obs_objects=object_count(obs, threshold),
        n_fcst_objects=object_count(fcst, threshold),
        mse_total=mse_total,
        mse_shifted=mse_shifted,
        mse_displacement=mse_displacement,
        mse_volume=mse_volume,
        mse_pattern=mse_pattern,
        pct_displacement=pct_displacement,
        pct_volume=pct_volume,
        pct_pattern=pct_pattern,
        mean_obs=mean_obs,
        mean_fcst_shifted=mean_fcst_shifted,
        peak_obs=float(np.nanmax(obs[valid_mask])) if valid_mask.any() else float("nan"),
        peak_fcst_shifted=float(np.nanmax(shifted[valid_mask])) if valid_mask.any() else float("nan"),
        spatial_corr_original=original_corr,
        spatial_corr_shifted=shifted_corr,
    )

    return result, shifted, score_mask
