# Metrics and visualizations — what each one is, where it came from

This document maps every new metric and every dashboard panel introduced
on this fork to the research paper(s) it was adapted from. It is meant
for readers who haven't lived inside the work — every acronym is spelled
out the first time it appears, and each section ends with a pointer to
the file where the implementation lives.

The companion documents are:

- `docs/DESIGN_metrics_extension.md` — original design proposal (kept
  for the rationale and the open-questions log).
- `docs/FORK_SUMMARY.md` — running log of what was built and what bugs
  were fixed during development.

---

## Quick glossary

**Domain context**

- **Onset / monsoon onset.** The first day of the rainy season at a
  given location, defined by a rainfall criterion (here: a wet spell
  begins, no qualifying dry spell follows for some window).
- **DOY.** Day of year. 1 = Jan 1, 152 = Jun 1, 273 = Sep 30, etc.
  Onset dates are stored as DOYs throughout the package.
- **Onset field / onset map.** A 2-D grid (lat × lon) where each cell
  holds the DOY of onset for one season. NaN / NaT means "no onset
  occurred this season at this cell."
- **Isochrone.** A contour line on an onset map where every point has
  the same onset DOY — i.e. the leading edge of the monsoon at one
  calendar day.
- **Ensemble.** A set of forecast members produced from perturbed
  initial conditions. Each member is its own onset field.
- **IMD.** India Meteorological Department. Source of the gridded
  observed rainfall used as ground truth in this fork.
- **S2S.** Subseasonal-to-seasonal. The forecast horizon (~2–6 weeks
  out) where the included models live.
- **AIFS / NGCM / IFS-S2S / FuXi-S2S.** The four S2S forecast systems
  bundled in `aice_data` (deterministic ECMWF AI model; 51-member
  NeuralGCM; 11-member ECMWF S2S; 51-member FuXi).

**Score-theoretic terms**

- **Proper scoring rule.** A score that is minimised in expectation
  exactly when the forecast distribution matches the true outcome
  distribution. Improper scores can be gamed; proper scores cannot.
  (Gneiting & Raftery 2007.)
- **Brier score.** Mean squared error of a probability forecast against
  a binary outcome: `BS = mean((p − y)²)`. Proper.
- **CRPS.** Continuous Ranked Probability Score. The continuous-outcome
  generalisation of the Brier score; equals expected absolute error
  for a deterministic forecast. (Hersbach 2000.)
- **Fair score.** A finite-ensemble bias correction. Raw ensemble CRPS
  systematically overestimates the population-CRPS by a term
  proportional to ensemble spread; the "fair" form removes that bias
  so a 5-member and a 50-member ensemble can be compared honestly.
  (Ferro 2014, Leutbecher 2019.)
- **Skill score.** A relative score: `SS = 1 − S_fcst / S_ref`. Positive
  means beating the reference; 1 is perfect.
- **PAV / isotonic regression.** Pool-Adjacent-Violators algorithm —
  fits the best monotone non-decreasing curve to (x, y) data without
  any tuning parameter.

---

## Probabilistic metrics — Milestone 1

### CRPS for onset (sentinel-augmented mixed CRPS)

**What it is.** A proper scoring rule for the joint distribution of
"does onset occur this season at this cell?" and "if so, on what DOY?"
Onset outcomes are *mixed* — there is a positive-probability atom at
"no onset" and a continuous distribution over DOY conditional on
onset. Naively masking no-onset cells turns CRPS into an improper
score; the formulation here keeps it proper.

**The construction.** Map every "no onset" outcome (forecast or obs)
to a sentinel DOY equal to `season_end + 1`, then evaluate the
Hersbach (2000) closed-form ensemble CRPS in this augmented sample
space. By Gneiting & Raftery (2007) this is a proper score for the
mixed distribution.

> Important nuance preserved in the docstrings: this is **not** the
> analytical censored-Gaussian CRPS of Hemri et al. 2014. Hemri's
> closed form is specific to a Gaussian forecast censored at a known
> threshold — a different object. An earlier draft of this fork
> claimed equivalence with Hemri 2014; that claim was retracted.

**Fair correction.** Default-on for ensembles with `m ≥ 2` members
via the Ferro (2014) / Leutbecher (2019) `1 / (m (m − 1))` denominator
on the spread term. Lets the 51-member NGCM and the 11-member IFS-S2S
be compared on equal footing.

**Source papers.**

- Hersbach, H. (2000). Decomposition of the continuous ranked
  probability score for ensemble prediction systems. *Weather and
  Forecasting* 15, 559–570.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring
  rules, prediction, and estimation. *Journal of the American
  Statistical Association* 102, 359–378.
- Ferro, C. A. T. (2014). Fair scores for ensemble forecasts.
  *Quarterly Journal of the Royal Meteorological Society* 140, 1917–1923.
- Leutbecher, M. (2019). Ensemble size: How suboptimal is less than
  infinity? *Quarterly Journal of the Royal Meteorological Society*
  145, 107–128.
- Hemri, S., Lisniak, D., & Klein, B. (2014) is **referenced for
  contrast**, not as the basis of the implementation.

**Code.** `momp/metrics/crps.py` — `crps_ensemble`, `censored_crps`,
`censored_crps_decomposition` (the diagnostic Brier-on-atom split).

**Dashboard panel.** *CRPS field* — per-cell CRPS in days, Magma
heatmap, mean/median/IQR caption with a fair-CRPS flag. Cells that
are NaN in both obs and all members are nulled out of the mean.

---

### CORP reliability diagram

**What it is.** A reliability (calibration) diagram and the score
decomposition that goes with it. CORP stands for **C**onsistent,
**O**ptimal w.r.t. score, **R**eproducible, **P**AV-based — meaning
the diagram is the unique calibration curve consistent with the
proper score at hand, has no tuning parameter (no bin count), and is
computed by isotonic regression of binary outcomes on raw forecast
probabilities.

**The decomposition.** For any proper scoring rule `S`,
`mean S = MCB − DSC + UNC` where

- **MCB** (miscalibration): penalty for deviating from the calibrated
  reference curve — what binned reliability diagrams visualise but
  cannot quantify cleanly.
- **DSC** (discrimination): reward for separating outcome regimes —
  how much the forecast actually distinguishes onset-yes from
  onset-no events.
- **UNC** (uncertainty): the score of the climatological forecast —
  the irreducible piece set by the marginal frequency of onset.

For binary outcomes with `S` = Brier, the identity holds at
floating-point zero, which the test suite checks.

**Honest sample size.** Onset events are spatially correlated, so the
nominal `N` overstates independence. The dashboard reports both `N`
(raw count of `(p, y)` pairs) and `N_eff`, which deflates `N` by
Moran's I (queen-4 spatial autocorrelation) using the Dutilleul
(1993) effective-sample-size formula.

**Source papers.**

- Dimitriadis, T., Gneiting, T., & Jordan, A. I. (2021). Stable
  reliability diagrams for probabilistic classifiers. *Proceedings of
  the National Academy of Sciences* 118 (8): e2016191118.
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena.
  *Biometrika* 37, 17–23.
- Dutilleul, P. (1993). Modifying the t test for assessing the
  correlation between two spatial processes. *Biometrics* 49, 305–314.

**Code.** `momp/graphics/corp_reliability.py` — `corp_decompose_brier`
returning a `CORPDecomposition` dataclass. Effective-N machinery in
`frontend/api/metrics.py` (`moran_i_2d`, `effective_sample_size`).

**Dashboard panel.** *CORP reliability* — calibration curve plus the
MCB/DSC/UNC numbers and the `N`/`N_eff` caption.

---

### FSS — Fractions Skill Score

**What it is.** A neighborhood-based spatial verification score.
Answers "if I allow the forecast to be off by `n` grid cells, does it
agree with the observation?" by comparing fractions of "has onset by
DOY τ" pixels inside every `n × n` window.

**The construction.** Threshold each onset field at DOY τ to a binary
mask, box-average each mask with a uniform `n × n` filter (Roberts &
Lean 2008 convention: cells outside the domain count as 0), then
score the resulting fraction fields with

```
FSS(τ, n) = 1 − MSE(F_f, F_o) / (mean(F_f²) + mean(F_o²))
```

`FSS = 1` is perfect; `FSS = 0` is no skill. The score is
non-decreasing in `n`, so the curve over `n` shows at what spatial
scale the model becomes useful — the canonical "scale separation"
diagnostic.

**Why it matters here.** Monsoon onset patterns are right *in shape*
but commonly wrong in placement by tens to hundreds of km. FSS
quantifies that directly: a flat curve at zero means structurally
wrong; a curve that climbs from zero to one near scale `n*` means
"right pattern, displaced by ~`n*` cells."

**Source paper.**

- Roberts, N. M., & Lean, H. W. (2008). Scale-selective verification
  of rainfall accumulations from high-resolution forecasts of
  convective events. *Monthly Weather Review* 136, 78–97.
  doi:10.1175/2007MWR2123.1

**Code.** `momp/metrics/neighborhood.py` — `fss_single`, `fss`,
`fss_multi_year`.

**Dashboard panel.** *FSS matrix* — one line per DOY threshold,
horizontal axis = neighborhood size `n`, with reference lines at the
no-skill (`p`) and useful (`0.5 + 0.5·p`) levels per threshold (the
standard Roberts–Lean reference levels, where `p` is the climatology
base rate of "onset by τ").

---

### Centroid displacement and area bias

**What it is.** For a "has-onset-by-DOY-τ" mask, two simple physical
diagnostics:

- **Centroid shift.** The area-weighted centroid (lat, lon) of the
  forecast region minus that of the observed region, reported both
  in degrees and in great-circle kilometres (haversine on a sphere).
  Positive Δlat means the forecast is too far north; positive Δlon
  means too far east.
- **Area bias.** Total spherical area of the forecast region minus
  that of the observed region. Reported in km² and as a fraction of
  obs area.

These pair with FSS as a *direction* diagnostic. FSS asks "at what
scale do they agree?"; centroid displacement asks "and which way is
the forecast offset?"

**Source.** Centroid displacement and area bias are textbook spatial
verification diagnostics; the standard reference for the family is

- Wilks, D. S. (2019). *Statistical Methods in the Atmospheric
  Sciences* (4th ed.), §9.

The closest published precedent for using them on onset-by-DOY masks
specifically is the family of object-based verification methods (see
the CRA / SAL / MODE references at the bottom).

**Code.** `momp/metrics/displacement.py` — `centroid_displacement`,
`displacement_bias_sweep`.

**Dashboard panel.** *Displacement + area bias* — dual-axis line
chart over a sweep of DOY thresholds (great-circle km on one axis,
fractional area bias on the other).

---

## Progression-verification metrics — Milestone 2

This is the milestone that contains the most novel work. The two
metrics and the isochrone overlay all rest on the same underlying
idea, so they share a section of background here before each one is
described.

### Background: the advancing-front view

ROMP and most monsoon-verification literature treat onset as a static
DOY map and grade it pointwise. That misses the fact that a monsoon
*advances*: it arrives at Kerala first, then progresses northwest.
A model can get every individual DOY wrong while still capturing the
right advance, or vice versa. Neither pointwise MAE nor field-level
CRPS sees this.

The fix is to convert each onset field into a family of binary masks
indexed by calendar day:

```
has_onset_by_d(x) = (onset_DOY(x) ≤ d)
```

For each `d`, this is a region on the map; as `d` advances day by
day, the region grows. Verifying *the family* (not the static DOY
map) is what the next three sections do. The direct analogue is
sea-ice-edge verification, where Goessling et al. solved the same
moving-boundary problem for the Arctic ice edge. Their IIEE / SPS
metrics translate cleanly to onset.

### IOE — Integrated Onset Error

**What it is.** For each day `d` in the onset window, compute the
**area of the symmetric difference** between `has_onset_by_d_fcst`
and `has_onset_by_d_obs`, weighted by spherical cell area. Decompose
each `IOE(d)` into:

- **extent error** = `|area_fcst(d) − area_obs(d)|`. Pure size
  disagreement — same shape, different total area.
- **misplacement error** = `IOE(d) − extent(d)`. The geographically
  informative piece — same area but in the wrong place.

Trapezoid-integrate `IOE(d)`, `extent(d)`, `misplacement(d)` over
the season to get three season totals (units: km² · day).

**Adapted from.** The Integrated Ice-Edge Error of Goessling et al.
2016, where the same construction is applied to the boundary between
"sea ice present" and "sea ice absent" cells in the Arctic. The
extent / misplacement decomposition is theirs verbatim. We replace
"is there sea ice in this cell on this day?" with "has monsoon
onset arrived at this cell by this day?"

**Source paper.**

- Goessling, H. F., Tietsche, S., Day, J. J., Hawkins, E., & Jung, T.
  (2016). Predictability of the Arctic sea ice edge. *Geophysical
  Research Letters* 43, 1642–1650.
  doi:10.1002/2015GL067232

**Code.** `momp/metrics/progression.py` — `integrated_onset_error`.

**Dashboard panel.** *Progression curve* — `IOE(d)` for each
selected model, season-integrated total in the caption, an optional
toggle that splits the primary model into its extent vs misplacement
components.

### SPS — Spatial Probability Score

**What it is.** The ensemble extension of IOE. For each day `d` and
each cell `x`,

```
P_fcst(x, d)  =  fraction of ensemble members with onset_DOY(x) ≤ d
O(x, d)       =  1 if obs_DOY(x) ≤ d else 0
SPS(d)        =  Σ_x  (P_fcst(x, d) − O(x, d))²  ·  A(x)
```

i.e. an area-weighted Brier score of the per-cell onset probability
against the observed indicator. Reduces *exactly* to IOE for a
deterministic single-member forecast — proven analytically and
checked in the test suite.

This is the right way to score an ensemble's progression: SPS rewards
ensembles whose member spread covers the observed front and
penalises overconfidence.

**Source paper.**

- Goessling, H. F., & Jung, T. (2018). A probabilistic verification
  score for contours: Methodology and application to Arctic ice-edge
  forecasts. *Quarterly Journal of the Royal Meteorological Society*
  144, 735–743.
  doi:10.1002/qj.3242

**Code.** `momp/metrics/progression.py` — `spatial_probability_score`.

**Dashboard panel.** Plotted on the same *Progression curve* axes as
IOE, dotted line per model.

### Isochrone overlay — the hero figure

**What it is.** For a chosen calendar day `d`, draw the contour where
`onset_DOY = d` in both forecast and observation, on the same map.
Repeat for several `d`, evenly spaced through the season. The result
is a side-by-side picture of the advancing front — the one figure
that lets a non-specialist instantly see "the model gets the timing
of the advance from south to north right, but it lags by a week from
mid-July onward."

**Implementation details.**

- Contours via `matplotlib.contour` (marching-squares). NaN cells
  must be substituted with a high-DOY sentinel before contouring;
  otherwise matplotlib treats NaN as a hole and traces a spurious
  ring around every isolated finite cell at every level below it.
- Three visual cues per DOY so overlapping forecast + obs lines stay
  legible: observed = wide soft dashed halo + thin dashed centerline;
  forecast = solid line with open-circle markers. NaN obs cells
  shown as a visible gray overlay.
- Contour-shape distances per (forecast, obs) pair via shapely:
  Hausdorff (worst case) and Fréchet (best alignment respecting
  ordering). Reported in degrees and in mid-latitude km.

**Adapted from.** The construction is direct, but the visualisation
choice and the use of Hausdorff / Fréchet on contours come from the
contour-verification literature:

- Hausdorff distance — a classical set-distance: the largest
  shortest-distance from any point on one curve to the other. (Definition: Munkres, J. R. *Topology* (2nd ed., 2000), §45.)
- Fréchet distance — the "leash" distance respecting traversal
  order. The discrete-curve algorithm used here is

  > Eiter, T., & Mannila, H. (1994). Computing discrete Fréchet
  > distance. *Technical Report CD-TR 94/64*, Christian Doppler Lab
  > for Expert Systems, TU Vienna.

  Practical Fréchet distance via shapely follows Alt, H. & Godau, M.
  (1995). Computing the Fréchet distance between two polygonal
  curves. *Int. J. Comput. Geom. Appl.* 5, 75–91.

The choice to apply contour-distance metrics to *onset* isochrones
specifically is the novel part — there is no published precedent we
could find.

**Code.** `momp/graphics/isochrone.py` — `extract_isochrone`,
`hausdorff_km`, `frechet_km`, `isochrone_overlay`.

**Dashboard panel.** *Isochrone overlay (hero)* — dashboard's most
prominent panel; the subtitle spells out the exact onset criteria
and iso-day spacing in plain English.

---

## What is genuinely new vs the baseline package and the CRA contribution

The classmate's contribution to the project is an implementation of
**CRA** (Contiguous Rain Area, Ebert & McBride 2000) — an
object-based verification approach that identifies rain "objects" in
a forecast field and decomposes the forecast error of each object
into displacement, volume, and pattern components.

CRA and the work on this fork are *complementary*, not overlapping:

| Concern                          | Baseline ROMP | CRA contribution | This fork |
|----------------------------------|---------------|------------------|-----------|
| Pointwise DOY error (MAE, FAR)   | yes           | no               | unchanged |
| Probabilistic onset distribution | partial (BS, RPS, AUC) | no | **CRPS for the mixed distribution + CORP decomposition + Moran-deflated `N_eff`** |
| Spatial agreement at scale       | no            | object-based     | **FSS + centroid/area-bias diagnostics** |
| Treating onset as a moving front | no            | no               | **IOE + SPS + isochrone overlay** |
| Object identification            | no            | **CRA**          | no |

The two contributions answer different questions on the same data.
CRA answers "given this rain object in the forecast, where is it
displaced, by how much volume is it off, and what is the residual
pattern error?" — fundamentally object-based on raw rainfall. The
fork answers "given this onset *field*, how well does its evolution
track the observed advance?" — field-based on derived onset DOY.

---

## Where to dig deeper next

The general direction we are moving in is to pick **one or two**
metrics for substantially deeper development — better tests, more
principled defaults, and a methods-paper-quality writeup. Three
candidates are flagged here, in order of depth:

### 1. Progression curve (definite deep-dive target)

The IOE / SPS pair is the most novel scientific contribution on the
fork, and it is the basis of the methods paper sketched in
`docs/DESIGN_metrics_extension.md` §8. Avenues for deeper work:

- **Bootstrap CIs** on multi-year aggregates of `IOE_season`,
  `extent_season`, `misplacement_season`. Currently we surface
  median + IQR and a low-`n` warning banner; we could replace the
  qualitative warning with a year-resampling block bootstrap.
- **Choice of `days` grid.** Daily is natural but expensive; 5-day
  stepping may be visually identical and ~5× cheaper. Open question
  flagged in the design doc — needs an empirical sensitivity study.
- **Ensemble calibration of SPS.** SPS rewards spread that covers
  obs; investigate whether sentinel-median collapsing of small
  ensembles distorts the SPS shape.
- **Comparison plot vs sea-ice-edge IIEE in the original Goessling
  papers.** Side-by-side reproduction of one of their hero figures
  using monsoon onset data would be the methods-paper figure.

### 2. Isochrone overlay (likely deep-dive target)

This is the figure people will actually look at, and it is also the
one with the least-published precedent. Avenues:

- **Grid-resolution floor on Hausdorff / Fréchet.** Currently the
  numbers are reported in km without a floor; below ~1 grid cell
  the distances are noise. Open follow-on in `FORK_SUMMARY.md`.
- **Climatologically informed isochrone selection.** Today we use
  4 evenly-spaced DOYs in the shared forecast/obs onset range. A
  better default: climatological onset quantiles per region (e.g.
  Kerala 1-Jun, Mumbai 10-Jun, Delhi 1-Jul) so the isochrones
  correspond to physically meaningful events.
- **Contour-distance distributions, not point estimates.** A single
  Hausdorff number per isochrone hides directionality; a
  per-vertex-distance histogram tells a richer story.

### 3. FSS (alternative deep-dive target)

If the isochrone path turns out to be visualisation-heavy, FSS is
the more *quantitative* alternative — it has a deeper statistical
literature to lean on and clean derivations available for the
useful-skill threshold and finite-sample bias.

- **Skill-from-no-skill significance.** Per-(τ, n) bootstrap CIs on
  FSS would let us draw a "clearly skillful" region on the matrix.
- **Aggregate FSS across multiple thresholds.** Currently the panel
  shows one line per threshold; collapsing to a single FSS-vs-`n`
  curve via Roberts–Lean's neighborhood-skill aggregation would
  give a single headline number per model.
- **Alternative neighborhood weightings.** Roberts–Lean uses uniform
  boxes; Gaussian neighborhoods are a published variant
  (Schwartz & Sobash 2017, *MWR*) worth a quick comparison.

We do *not* plan to deepen CRPS, CORP, or centroid/area-bias on this
fork — they are well-trodden in the literature and the
implementations here already have synthetic-known-answer tests
sufficient for production use.

---

## Reference list (alphabetical)

- Alt, H., & Godau, M. (1995). Computing the Fréchet distance
  between two polygonal curves. *International Journal of
  Computational Geometry and Applications* 5, 75–91.
- Dimitriadis, T., Gneiting, T., & Jordan, A. I. (2021). Stable
  reliability diagrams for probabilistic classifiers. *PNAS*
  118 (8): e2016191118.
- Dutilleul, P. (1993). Modifying the t test for assessing the
  correlation between two spatial processes. *Biometrics* 49,
  305–314.
- Ebert, E. E., & McBride, J. L. (2000). Verification of precipitation
  in weather systems: Determination of systematic errors. *Journal
  of Hydrology* 239, 179–202. (CRA — classmate's contribution.)
- Eiter, T., & Mannila, H. (1994). Computing discrete Fréchet
  distance. Tech. report, TU Vienna.
- Ferro, C. A. T. (2014). Fair scores for ensemble forecasts.
  *QJRMS* 140, 1917–1923.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring
  rules, prediction, and estimation. *JASA* 102, 359–378.
- Goessling, H. F., Tietsche, S., Day, J. J., Hawkins, E., & Jung, T.
  (2016). Predictability of the Arctic sea ice edge. *GRL* 43,
  1642–1650.
- Goessling, H. F., & Jung, T. (2018). A probabilistic verification
  score for contours. *QJRMS* 144, 735–743.
- Hersbach, H. (2000). Decomposition of the continuous ranked
  probability score for ensemble prediction systems. *Weather and
  Forecasting* 15, 559–570.
- Leutbecher, M. (2019). Ensemble size: How suboptimal is less than
  infinity? *QJRMS* 145, 107–128.
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena.
  *Biometrika* 37, 17–23.
- Roberts, N. M., & Lean, H. W. (2008). Scale-selective verification
  of rainfall accumulations from high-resolution forecasts of
  convective events. *MWR* 136, 78–97.
- Schwartz, C. S., & Sobash, R. A. (2017). Generating probabilistic
  forecasts from convection-allowing ensembles using neighborhood
  approaches: A review and recommendations. *MWR* 145, 3397–3418.
- Wilks, D. S. (2019). *Statistical Methods in the Atmospheric
  Sciences* (4th ed.). Elsevier.
