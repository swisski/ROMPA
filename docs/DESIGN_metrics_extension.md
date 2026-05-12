# ROMP metrics extension — design doc

**Status:** draft
**Scope:** two-milestone metric extension for the ROMP PyPI package
**Audience:** ROMP developers and methods-paper reviewers

## 1. Motivation

ROMP currently ships only point-wise verification: MAE, FAR, Miss Rate, Brier Score, Fair BS, RPS, Fair RPS, AUC, binned reliability diagrams, and BSS/RPSS against climatology. This leaves two gaps that a serious onset-verification package must fill:

1. **Mixed-distribution probabilistic gap.** Onset outcomes are not purely continuous — "no onset this year" is a positive-probability atom, not missing data. The current metrics either mask NaT or silently mishandle it. A proper scoring rule for the mixed discrete–continuous onset distribution is the right foundation.
2. **Spatial progression gap.** Monsoon onset is a moving front; models fail systematically in the *advance* of onset, not only in the DOY at any given cell (Chevuturi et al. 2021, SEAS5). ROMP has no metric that sees the front as a moving object. No published Python package does.

We close both gaps in two shippable milestones. Milestone 1 deepens the probabilistic core; Milestone 2 adds a new scientific axis.

## 2. Non-goals

Metrics explicitly deferred, with reasons:

- **Full CRA (Ebert & McBride 2000).** Forced fit on DOY fields — volume decomposition has no physical meaning; thresholding is arbitrary; no published precedent on onset fields.
- **SAL on raw DOY.** Thresholding a DOY field produces meaningless objects. A SAL-on-P(onset≤τ) variant is defensible but largely redundant once IOE/SPS ship.
- **Energy Score / Variogram Score.** Sample-size-starved at ~20–30 annual onset maps. Bootstrap CIs will swallow the signal.
- **MODE / MODE-TD.** MET install is a heavyweight C++ dependency incompatible with a lightweight PyPI package.
- **Wavelet / band-depth / log score.** Diminishing return relative to existing BS/RPS and the additions planned here.

## 3. Milestone 1 — Probabilistic depth

Strengthens the existing core. Ships as ROMP 0.1.0.

### 3.1 Censored/mixed CRPS

**What:** CRPS for the mixed distribution `F(y) = π · δ_{no onset} + (1 − π) · G(y)` where `π` is ensemble probability of no onset and `G` is the conditional DOY distribution.

**Why:** Currently, years or grid cells with "no onset" either drop out of CRPS via masking or get scored by an ad-hoc fill value. Neither is a proper scoring rule. The mixed-CRPS formulation rewards both the onset probability and the DOY distribution jointly, is a proper score by construction, and is the mathematically correct object for onset verification.

**Implementation note on terminology.** The construction used here is a
**sentinel-augmented ensemble CRPS** for the mixed atom-plus-continuous
distribution: no-onset outcomes are mapped to a sentinel value beyond the
verification window and the Hersbach (2000) ensemble CRPS is then evaluated
in the augmented sample space. By Gneiting & Raftery (2007) this is a
proper score for the mixed distribution. It is *related to but not
identical with* the analytical censored-Gaussian CRPS of Hemri et al.
(2014, GRL), which derives a closed form for a Gaussian forecast
censored at a known threshold — a distinct object. We should not claim
equivalence with Hemri 2014. The fair (Ferro 2014) finite-ensemble bias
correction is also applied by default for ensembles with m ≥ 2.

**References:**
- Hersbach 2000, *Wea. Forecasting* — ensemble CRPS closed form.
- Gneiting & Raftery 2007, *JASA* — proper scoring rules tutorial.
- Ferro 2014, *QJRMS* — fair scores for ensemble forecasts.
- Leutbecher 2019, *QJRMS* — ensemble size and scoring-rule bias.
- Hemri et al. 2014, *GRL* — analytical censored-Gaussian CRPS (related,
  different construction).

**Module:** `momp/metrics/crps.py`

**Public API (draft):**
```python
def censored_crps(
    ensemble_onset: xr.DataArray,  # (member, lat, lon) with NaT for no-onset
    obs_onset: xr.DataArray,        # (lat, lon) with NaT for no-onset
    *,
    season_end: int,                # DOY upper bound for "no onset"
) -> xr.DataArray:                  # (lat, lon) CRPS per cell
    ...

def censored_crps_skill_score(
    crps_forecast, crps_reference
) -> xr.DataArray:
    ...
```

**Implementation notes:**
- Sentinel augmentation: ``no_onset → season_end + 1``, then Hersbach
  ensemble CRPS over the augmented sample space. The Brier-on-atom vs
  continuous-CRPS decomposition is exposed by ``censored_crps_decomposition``
  as a diagnostic (note: the two pieces do *not* exactly sum to the mixed
  CRPS; they are different proper-score constructions).
- Fair (Ferro 2014) finite-ensemble bias correction: enabled by default
  through ``crps_ensemble(..., fair=True)`` / ``censored_crps(..., fair=True)``
  whenever ``m ≥ 2``. Single-member forecasts fall back to the raw form
  (for which fair is undefined).
- Per-lead-bin output via the existing ``momp/stats/bins.py`` machinery.

**Tests:**
- Degenerate ensemble → reduces to absolute error where onset occurs, to Brier term where it does not.
- All-NaT forecast vs all-NaT obs → zero CRPS.
- Mixed case validated against a synthetic distribution with known analytical CRPS.

**LOC estimate:** ~300 LOC implementation, ~250 LOC tests.

### 3.2 CORP reliability diagrams

**What:** Replace binned reliability plots with isotonic-regression (PAV) based CORP diagrams yielding the MCB–DSC–UNC decomposition.

**Why:** Binned reliability is sensitive to bin count and endpoint effects. CORP is tuning-free, consistent, and produces a proper-score decomposition that generalizes Brier's REL–RES–UNC to any proper score. Already standard in ML-weather eval papers.

**Reference:** Dimitriadis, Gneiting, Jordan 2021, *PNAS*.

**Module:** `momp/graphics/reliability.py` upgrade (preserve existing function; add `corp_reliability_diagram`).

**Implementation:** wrap `scores.isoreg_cdf` (from the `scores` Xarray package) or implement PAV directly via `sklearn.isotonic.IsotonicRegression`.

**LOC estimate:** ~150 LOC + ~80 LOC tests.

### 3.3 FSS — Fractions Skill Score

**What:** Neighborhood-based spatial skill on thresholded onset-window masks.

**Why:** Cheap, canonical, addresses the "pattern right but shifted" diagnostic gap at multiple scales. Users expect it in any modern verification package.

**Reference:** Roberts & Lean 2008, *MWR*.

**Module:** `momp/metrics/neighborhood.py`

**Public API (draft):**
```python
def fss(
    fcst_onset: xr.DataArray,
    obs_onset: xr.DataArray,
    *,
    thresholds: Sequence[int],          # DOY thresholds defining "onset by"
    neighborhoods: Sequence[int],       # radii in grid cells
) -> xr.DataArray:                      # (threshold, neighborhood)
    ...
```

**LOC estimate:** ~80 LOC + ~100 LOC tests.

### 3.4 Centroid displacement + area bias

**What:** Per-quantile-threshold onset-region centroid shift (Δlat, Δlon) and area bias.

**Why:** Directly answers "is onset too far north?" which is the question monsoon modelers actually ask. Cheap, interpretable, pairs with FSS.

**Module:** fold into `momp/metrics/error.py`.

**Implementation:** `scipy.ndimage.label` + `scipy.ndimage.center_of_mass` on binary onset-by-DOY masks, NaN-safe.

**LOC estimate:** ~60 LOC.

### 3.5 Milestone 1 deliverable

`pip install romp==0.1.0` gives users:
- `momp.metrics.crps.censored_crps` and skill-score variant
- `momp.metrics.neighborhood.fss`
- `momp.graphics.reliability.corp_reliability_diagram`
- centroid/area-bias fields in existing spatial outputs
- all wired into the existing CLI driver and lead-bin machinery
- updated example notebook in `docs/`

## 4. Milestone 2 — Progression verification

New scientific axis. Ships as ROMP 0.2.0 and forms the basis of a methods paper.

### 4.1 Core idea

Treat onset as an *advancing front* rather than a static DOY map. For any calendar day `d` in the onset window, the binary field `onset_by_d(x) = (onset_DOY(x) ≤ d)` defines a contour separating "monsoon has arrived" from "has not." Verification then scores the forecast and observed contours — directly, at each `d`, and integrated over the season. This mirrors sea-ice edge verification (Goessling 2016, 2018), where an analogous problem has a mature solution.

### 4.2 Integrated Onset Error (IOE)

**What:** Per day `d`, the area of symmetric difference between forecast and observed "onset by `d`" masks. Decomposes into absolute extent error (area bias) and misplacement error (the geographically informative piece). Season-integrated IOE = ∫ IOE(d) d(d).

**Reference:** Goessling et al. 2016, *GRL* — IIEE for sea-ice edge, adapted.

**Module:** `momp/metrics/progression.py`

**Public API (draft):**
```python
def integrated_onset_error(
    fcst_onset: xr.DataArray,       # (lat, lon) DOY
    obs_onset: xr.DataArray,        # (lat, lon) DOY
    *,
    days: Sequence[int],            # DOYs at which to evaluate
    area_weights: xr.DataArray | None = None,  # lat-weighted area
) -> xr.Dataset:                    # per-day IOE, extent, misplacement + season-integrated
    ...
```

**Implementation notes:**
- `mask_d = (onset_DOY ≤ d)` with NaT treated as never (False).
- Cell-area weights from `cos(lat)` if not supplied.
- Decomposition: `extent = |area_F − area_O|`; `misplacement = IOE − extent`.
- Season integral via trapezoid rule over `days`.

**LOC estimate:** ~350 LOC + ~250 LOC tests.

### 4.3 Spatial Probability Score (SPS)

**What:** Ensemble extension of IOE. For each day `d`, SPS(d) = integrated Brier score over cells of `P_fcst(onset ≤ d)` vs `1_obs(onset ≤ d)`. Reduces to IOE for a deterministic member.

**Reference:** Goessling 2018, *QJRMS* — SPS for sea-ice contours.

**Module:** same as IOE.

**LOC estimate:** ~200 LOC + ~150 LOC tests.

### 4.4 Isochrone extraction and overlay plots

**What:** For selected DOYs (e.g., climatological 1 June, 15 June, 1 July isochrones), extract contours from forecast and observation and plot as overlays. Compute modified-Hausdorff and discrete Fréchet distance per isochrone.

**Why:** This is the hero figure. Users and reviewers see at a glance whether the model gets the advance right. The Fréchet/Hausdorff numbers give a single summary per isochrone.

**Module:** `momp/graphics/isochrone.py`

**New dependencies:** `scikit-image` (for `measure.find_contours`), `shapely` (for Fréchet/Hausdorff on line geometries). Both mainstream and conda-forge available. Add to `pyproject.toml`.

**LOC estimate:** ~250 LOC + ~150 LOC tests.

### 4.5 CLI integration

**What:** New driver `momp-run-progression` parallel to existing `momp-run`. Drives the progression metrics across the configured ensemble, years, and lead-time bins.

**Module:** `momp/app/progression_verification.py` + entry point in `pyproject.toml`.

**LOC estimate:** ~150 LOC.

### 4.6 Milestone 2 deliverable

`pip install romp==0.2.0` additionally gives users:
- `momp.metrics.progression.integrated_onset_error`
- `momp.metrics.progression.spatial_probability_score`
- `momp.graphics.isochrone.isochrone_overlay`, `...frechet_distance_per_isochrone`
- `momp-run-progression` CLI
- example notebook for progression verification

## 5. Explicit scope guardrails

To prevent creep:
- **No MET dependency** under any circumstance. Keep the PyPI install lightweight.
- **No metric shipped without a tested handling of the NaT/no-onset case.** Silent masking is a bug, not a feature.
- **No metric whose per-year sampling uncertainty hasn't been examined in the tests.** With ~20–30 years, users need to know when a number is noise.
- **Every new metric integrates with the existing lead-bin machinery.** Lead-time binning is ROMP's differentiator; cross-cutting by default.

## 6. Order of work

1. **Milestone 1 first, in the order 3.2 → 3.3 → 3.4 → 3.1.** CORP is drop-in; FSS and centroid are cheap wins; censored CRPS is the hardest and benefits from having the cheap additions in place to cross-validate against.
2. **Milestone 2 second**, in the order 4.2 → 4.3 → 4.4 → 4.5. IOE is the foundation; SPS extends it to ensembles; isochrones produce the figure; CLI wires it all together.

Each item has its own test file and docs update before moving on. No batched merges.

## 7. Success criteria

Milestone 1 is done when:
- All four metrics ship with tests that include a synthetic "known-answer" case.
- The example notebook shows MCB–DSC–UNC decomposition on a real model.
- `romp==0.1.0` is tagged and published to TestPyPI.

Milestone 2 is done when:
- IOE on a synthetic advancing-front case gives the analytically expected season-integrated value.
- SPS on a deterministic ensemble reduces numerically to IOE.
- An isochrone-overlay figure for at least one real forecast/observation pair is generated from the example notebook.
- `romp==0.2.0` is tagged and published to TestPyPI.

## 8. Paper angle

Milestone 2 is the novel scientific capability and carries the methods paper: *"Verifying monsoon onset as an advancing front: IOE and SPS for seasonal forecast benchmarking."* Milestone 1 carries the JOSS/GMD package paper documenting ROMP as a whole. The two are independently reviewable.

## 9. Open questions

- **Fair censored CRPS.** *Resolved in the frontend implementation.* The
  Ferro 2014 correction transfers cleanly to the sentinel-augmented
  mixed CRPS: apply the 1/(m(m−1)) denominator on the spread term in
  the augmented sample space. Enabled by default for m ≥ 2 ensembles.
- **Region-masking conventions.** *Resolved in the frontend.* The
  ``ROMP_LAND_MASK`` environment variable accepts a country name
  (e.g. ``India``); when set, both the obs and forecast onset-DOY
  fields are masked via ``regionmask.defined_regions.natural_earth_v5_0_0``
  before every metric is computed. Unset by default.
- **Obs/forecast grid alignment.** *Resolved in the frontend.* When
  obs and model grids differ (e.g. 4° IMD vs 2° ensembles), the
  forecast is nearest-neighbour conformed to the obs grid rather than
  the other way around — obs is the authoritative measurement and
  upsampling obs would inflate area-weighted metrics by the upsample
  ratio.
- **Choice of `days` grid for IOE.** Daily is natural but expensive.
  5-day stepping may suffice; still needs empirical check.
- **Isochrone selection convention.** Climatological onset quantiles?
  Fixed calendar dates? Region-specific defaults? Current frontend
  heuristic: 4 equal-spaced DOYs in the shared obs–fcst onset range.
