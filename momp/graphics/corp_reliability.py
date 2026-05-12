"""CORP reliability diagrams and MCB-DSC-UNC decomposition.

CORP = Consistent, Optimal w.r.t. score, Reproducible, PAV-based.
Dimitriadis, Gneiting, Jordan, 2021. PNAS 118 (8) e2016191118.

For a proper scoring rule S, every mean score admits the decomposition

    S̄  =  MCB  -  DSC  +  UNC

- MCB (miscalibration): penalty for deviation from calibration.
- DSC (discrimination): reward for separating outcome regimes.
- UNC (uncertainty): score of the marginal (climatological) forecast.

The calibrated reference forecast is the PAV (isotonic) regression of the
binary outcome y on the raw forecast probability f. MCB and DSC are then
evaluated against that calibrated reference.

This module currently implements the Brier-score decomposition (binary
outcomes, probability forecasts), which is the relevant case for ROMP's
reliability-diagram use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

from momp.io.dict import extract_pd_bins
from momp.utils.printing import tuple_to_str


@dataclass(frozen=True)
class CORPDecomposition:
    """Result of the Brier-score CORP decomposition."""

    mean_score: float  # S̄  = mean((f - y)^2)
    mcb: float         # miscalibration
    dsc: float         # discrimination
    unc: float         # uncertainty
    n: int             # number of (f, y) pairs
    bin_edges_f: np.ndarray       # calibration curve x-coords (unique f after PAV)
    calibrated_y: np.ndarray      # calibration curve y-coords (PAV-regressed y)
    forecast_prob: np.ndarray     # raw forecast probabilities (sorted)
    observed: np.ndarray          # observed binary outcomes aligned with forecast_prob

    @property
    def score_check(self) -> float:
        """Identity check: should equal mean_score up to floating point."""
        return self.mcb - self.dsc + self.unc


def corp_decompose_brier(forecast_prob, observed) -> CORPDecomposition:
    """Compute the Brier-score CORP decomposition.

    Parameters
    ----------
    forecast_prob : array-like of float in [0, 1]
        Raw forecast probabilities f_i.
    observed : array-like of {0, 1}
        Binary observed outcomes y_i.

    Returns
    -------
    CORPDecomposition
    """
    f = np.asarray(forecast_prob, dtype=float)
    y = np.asarray(observed, dtype=float)

    if f.shape != y.shape:
        raise ValueError(f"shape mismatch: forecast_prob {f.shape} vs observed {y.shape}")

    mask = np.isfinite(f) & np.isfinite(y)
    f, y = f[mask], y[mask]
    n = f.size
    if n == 0:
        raise ValueError("no finite (forecast, observed) pairs")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("observed must be binary {0, 1}")
    if np.any((f < 0) | (f > 1)):
        raise ValueError("forecast_prob must lie in [0, 1]")

    order = np.argsort(f, kind="mergesort")
    f_s = f[order]
    y_s = y[order]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    c_s = iso.fit_transform(f_s, y_s)

    y_bar = y_s.mean()
    mean_score = float(np.mean((f_s - y_s) ** 2))
    reference_score = float(np.mean((c_s - y_s) ** 2))
    unc = float(y_bar * (1.0 - y_bar))

    mcb = mean_score - reference_score
    dsc = unc - reference_score

    return CORPDecomposition(
        mean_score=mean_score,
        mcb=mcb,
        dsc=dsc,
        unc=unc,
        n=int(n),
        bin_edges_f=f_s,
        calibrated_y=c_s,
        forecast_prob=f_s,
        observed=y_s,
    )


def _consolidate_curve(f_s: np.ndarray, c_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the PAV step function to one (f, c) pair per level set."""
    if f_s.size == 0:
        return f_s, c_s
    uniq_c, first_idx = np.unique(c_s, return_index=True)
    order = np.argsort(first_idx)
    uniq_c = uniq_c[order]
    first_idx = first_idx[order]
    last_idx = np.r_[first_idx[1:] - 1, f_s.size - 1]
    f_rep = 0.5 * (f_s[first_idx] + f_s[last_idx])
    return f_rep, uniq_c


def corp_reliability_diagram(
    combined_forecast_obs: pd.DataFrame,
    *,
    model: str,
    verification_window,
    day_bins,
    save_fig: bool,
    dir_fig: str,
    extract_bins: bool = True,
    show_plot: bool = True,
    forecast_col: str = "predicted_prob",
    observed_col: str = "observed_onset",
    title_suffix: str = "",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame, CORPDecomposition]:
    """Plot a CORP reliability diagram from forecast-observation pairs.

    Drop-in-compatible signature with ``momp.graphics.reliability.plot_reliability_diagram``:
    accepts the same kwargs and returns ``(fig, ax, results_df)`` plus the
    CORP decomposition object as a fourth element. Callers that only want
    the first three can unpack with ``fig, ax, df, _ = ...``.
    """
    if extract_bins:
        combined_forecast_obs = extract_pd_bins(combined_forecast_obs, day_bins)

    f = combined_forecast_obs[forecast_col].to_numpy()
    y = combined_forecast_obs[observed_col].to_numpy()

    decomp = corp_decompose_brier(f, y)

    f_rep, c_rep = _consolidate_curve(decomp.forecast_prob, decomp.calibrated_y)

    print("\nCORP reliability decomposition (Brier score):")
    print(f"  N pairs             : {decomp.n}")
    print(f"  Mean score (Brier)  : {decomp.mean_score:.6f}")
    print(f"  MCB (miscalibration): {decomp.mcb:.6f}")
    print(f"  DSC (discrimination): {decomp.dsc:.6f}")
    print(f"  UNC (uncertainty)   : {decomp.unc:.6f}")
    print(f"  Identity residual   : {decomp.score_check - decomp.mean_score:+.2e}")

    results_df = pd.DataFrame(
        {
            "forecast_prob": f_rep,
            "calibrated_prob": c_rep,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.plot(f_rep, c_rep, marker="o", color="tab:blue", linewidth=2, markersize=6,
            label="CORP calibration curve")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect reliability")

    ax2 = ax.twinx()
    ax2.set_yscale("log")
    hist_bins = np.linspace(0, 1, 21)
    counts, _ = np.histogram(decomp.forecast_prob, bins=hist_bins)
    centers = 0.5 * (hist_bins[:-1] + hist_bins[1:])
    total = counts.sum()
    freq = counts / total if total > 0 else counts.astype(float)
    positive = freq[freq > 0]
    if positive.size:
        ax2.bar(centers, freq, width=0.045, alpha=0.3, color="gray")
        ax2.set_ylim(positive.min() * 0.5, freq.max() * 2)
    ax2.set_ylabel("Forecast frequency", fontsize=12)

    ax.set_xlabel("Forecast probability f", fontsize=12)
    ax.set_ylabel("Calibrated probability ĉ = PAV(y | f)", fontsize=12)
    title = (
        f"CORP reliability — {model}\n"
        f"BS={decomp.mean_score:.3f}  MCB={decomp.mcb:.3f}  "
        f"DSC={decomp.dsc:.3f}  UNC={decomp.unc:.3f}"
    )
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")

    if save_fig:
        os.makedirs(dir_fig, exist_ok=True)
        window_str = tuple_to_str(verification_window)
        fig_fn = os.path.join(dir_fig, f"corp_reliability_{model}_{window_str}.png")
        fig.savefig(fig_fn, dpi=600, bbox_inches="tight")
        print(f"Figure saved to: {fig_fn}")

    plt.tight_layout()
    if show_plot:
        plt.show()

    return fig, ax, results_df, decomp
