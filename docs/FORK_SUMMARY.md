# ROMP fork — summary of work

A summary of all contributions made on this fork of ROMP, beyond the
baseline package. Covers both the metric additions to the `momp`
package and the interactive verification dashboard built on top.

## Branch layout

```
frontend (HEAD)  — multi-model verification dashboard + polish
main             — new verification metrics added to the ROMP package
origin/main      — baseline ROMP fork
```

25 commits total on the fork, local, nothing pushed. **107 tests passing.**

---

## On `main` — package-level metric additions

### Milestone 1: probabilistic depth

Four new metrics in the `momp` package.

**`momp/metrics/crps.py`** — sentinel-augmented mixed-distribution
CRPS. Maps "no onset" outcomes to `season_end + 1` and runs the
Hersbach ensemble CRPS in the augmented sample space. Proper score for
the atom-plus-continuous onset distribution (Gneiting & Raftery 2007).
Includes the Ferro 2014 fair finite-ensemble bias correction,
default-on for ensembles with ≥ 2 members.

**`momp/metrics/neighborhood.py`** — Fractions Skill Score
(Roberts & Lean 2008). Box-averages binary onset-by-DOY masks over
n-by-n neighborhoods. NaN-safe.

**`momp/metrics/displacement.py`** — centroid shift (haversine km) and
area bias between forecast and observed onset regions at a set of DOY
thresholds.

**`momp/graphics/corp_reliability.py`** — CORP reliability via
isotonic (PAV) regression. Yields the MCB–DSC–UNC decomposition of the
Brier score with the proper-score identity residual at floating-point
zero.

### Milestone 2: progression verification

**`momp/metrics/progression.py`** — Integrated Onset Error (IOE,
adapted from Goessling 2016 IIEE) and Spatial Probability Score
(SPS, Goessling 2018). IOE is the area of symmetric difference
between forecast and observed "onset-by-`d`" masks, area-weighted on
the sphere. SPS generalizes to ensembles via per-cell Brier of
`P(onset ≤ d)`. SPS reduces exactly to IOE for a 1-member
deterministic ensemble.

**`momp/graphics/isochrone.py`** — extracts isochrone contours from a
2-D DOY field via matplotlib marching squares, with a NaN-sentinel
substitution to prevent spurious rings around no-data holes. Hausdorff
+ Fréchet distances between forecast/observed contour pairs via
shapely.

**`momp/app/progression_verification.py`** + **`momp/driver_progression.py`** —
orchestration and the `momp-run-progression` CLI entry point.

### Tests and documentation on `main`

- 78 unit / known-answer synthetic tests. Degenerate ensemble → CRPS
  reduces to MAE; advancing-front → analytically integrated IOE
  matches; SPS on 1-member → exactly equals IOE; CORP identity residual
  exactly 0.
- 9 integration tests using the production `detect_onset` on raw
  AIFS/NGCM/IMD demo rainfall.
- `docs/DESIGN_metrics_extension.md` — two-milestone design doc with
  references, open questions, guardrails.
- `docs/example_milestone1_probabilistic.ipynb` and
  `docs/example_milestone2_progression.ipynb` — executed notebooks
  demonstrating the §7 success artifacts (CORP decomposition and
  isochrone overlay hero figure).

---

## On `frontend` — multi-model verification dashboard

### Backend layering (`frontend/api/`)

**`catalog.py`** — data-source discovery. Scans `ROMP_DATA_ROOT`
(default `demo/data`, overridable for `aice_data`), reports available
models, years, ensemble sizes.

**`onset.py`** — onset-field construction with process-wide caching
keyed on `(model, year, init, params)`. Observed onset goes through
the production `momp.stats.detect.detect_observed_onset` so obs DOYs
match what `momp-run` would produce. Forecast onset uses the same
criterion with an enforced May-1 lower-bound DOY so pre-monsoon April
rain in a forecast can't fire spurious onsets obs couldn't see.

**`metrics.py`** — JSON-ready metric serializers. Includes the
sentinel-median deterministic projection — cells get a forecast
onset only when ≥ 50% of ensemble members agree, treating no-onset
members as very-late sentinels — used for IOE and isochrones. Also
holds `moran_i_2d` (queen-4 spatial autocorrelation) and
`effective_sample_size` (Dutilleul 1993) for CORP's honest `N_eff`
reporting.

**`aggregate.py`** — multi-year aggregation with correct per-metric
semantics: per-cell mean CRPS, pooled `(p, y)` pairs for CORP (not
averaging per-year decompositions — statistically the right combine),
median + IQR per-day for IOE / SPS / displacement, per-(τ, n) mean
FSS.

**`app.py`** — FastAPI routing. 13 endpoints including `/api/health`,
`/api/catalog`, `/api/inits`, `/api/state`, one per metric,
`/api/compare`, `/api/cache/clear`. Obs-to-model grid alignment via
downsampling the forecast onto the obs grid. Optional
`ROMP_LAND_MASK=India` via `regionmask` with exact-match-first then
unambiguous substring.

### Frontend (`frontend/static/`)

- **`index.html`** — two-column layout: sticky sidebar controls,
  scrollable main column.
- **`app.css`** — observatory aesthetic: Fraunces serif headers,
  IBM Plex Sans/Mono body, deep monsoon-storm palette, amber + sky
  accents, grain overlay, per-card loading shimmers.
- **`app.js`** — ~1150 LOC vanilla JS driving state, multi-model
  color-stable chip selection, per-panel independent fetching with
  error isolation, shared Plotly theme.

### What the UI does

**Sidebar controls** (all re-run detection from raw rainfall on apply):

- Year range with quick buttons (single / last 5 / last 10 / all).
- Multi-select model chips — leftmost active = primary (drives hero
  + per-model panels).
- Separate iso-year picker for the hero panel.
- Init picker for the iso year.
- Six onset-criteria inputs (`wet_init`, `wet_spell`, `wet_threshold`,
  `dry_spell`, `dry_threshold`, `dry_extent`) with an explanatory
  paragraph.
- Optional lat/lon region bbox.
- Pulsing apply button when params are dirty.

**Panels:**

- **Cross-model summary table** at top — per-row `median [q25–q75]`
  scalars, horizontal scroll inside the card.
- **Low-n banner** — shown when the selected range is < 20 years,
  clarifying that IQR is descriptive, not statistical.
- **Isochrone overlay (hero)** — three visual cues per DOY so
  overlapping lines stay legible. Observed rendered as a wide soft
  dashed halo (bottom) plus a thin dashed centerline; forecast as a
  crisp solid line with open-circle markers (top). NaN obs cells
  shown as a visible gray overlay. DOY labels staggered to prevent
  collisions. Subtitle spells out the exact onset criteria and
  iso-day spacing.
- **Progression curve** — IOE (solid + markers), SPS (dotted) per
  model, stable color per model. Multi-year: median line + IQR band.
  Optional extent / misplacement decomposition toggle for the primary
  model. Interp footer explains the hump-then-decay shape and what a
  rising-to-end curve means.
- **CORP reliability** — calibration curve with MCB / DSC / UNC
  breakdown. `N` and `N_eff` (with Moran's I) in the caption. Interp
  footer explains the decomposition identity.
- **CRPS field** — per-cell forecast error in days (Magma heatmap).
  Mean / median / IQR caption, fair-CRPS flag. Cells where both obs
  and all members were NaN are explicitly nulled out of the mean.
- **Displacement + area bias** — dual-axis line chart (great-circle
  km + area-bias %).
- **FSS matrix** — one line per DOY threshold, reference lines at the
  no-skill (`p`) and useful (`0.5 + 0.5·p`) levels per threshold.

### Scripts and tooling

- **`frontend/run.sh`** — auto-detects a Python interpreter with
  `uvicorn + momp`, auto-picks `.data_aice` as `ROMP_DATA_ROOT` if
  present.
- **`frontend/link_aice_data.sh`** — builds the `.data_aice/` flat
  symlink tree over the nested `aice_data/` sibling repo (four models
  × up to 37 shared years).
- **`frontend/validate.py`** — boots uvicorn on a free port, hits
  every endpoint, checks shapes + identities + param flow-through.
- **`frontend/check_fresh.sh`** — flags any cached model/year whose
  earliest forecast DOY < 121 (detects stale pre-fix caches).

---

## Bugs found and fixed during development

Three parallel audit agents plus multiple iterative walkthroughs
uncovered and fixed the following.

### Numerically consequential

- Obs-to-model upsampling inflated area-weighted metrics by the
  upsample ratio. Fixed to downsample model onto the obs grid.
- Ocean cells masked by `ROMP_LAND_MASK` scored CRPS = 0 via
  sentinel-vs-sentinel and inflated the mean. Fixed to null them.
- Forecast detection had no lower-bound DOY, so pre-monsoon April
  rain fired false onsets obs couldn't see. Fixed to enforce May 1.
- `fcst_det = ens.mean(skipna=True)` silently excluded no-onset
  members — one outlier firing early could define the forecast.
  Fixed to sentinel-substituted median (majority-agreement required).
- CORP `n` overstated independence. Added Moran's I → effective-`n`
  (Dutilleul 1993).
- Frontend obs detection used a hard-coded May–Sep window. Fixed to
  call production `detect_observed_onset` with `extend_end_day=47`.
- Land-mask substring match could silently pick the wrong country
  (`Niger → Nigeria`, `Korea → South Korea`). Fixed to
  exact-match-first with ambiguous rejection.
- Fair CRPS now default for `m ≥ 2` so small ensembles aren't
  penalized relative to larger ones.

### Shape / robustness

- Isochrone contour extraction treated NaN as holes, producing
  spurious rings. Fixed with a sentinel substitution.
- Multi-year CRPS response was dropping `fair` / `n_members`. Fixed.
- Single-year vs multi-year `season` dict schemas diverged. Unified.
- Empty `year=` or inverted year ranges returned 422 Pydantic errors.
  Fixed to clean 400s.
- Displacement / FSS error paths were swallowing errors to console
  with stale plots left up. Fixed to purge + in-plot error messages.
- The isochrone overlay's fully-coincident lines were visually
  indistinguishable. Fixed with a three-cue rendering (wider soft halo
  + thin dashed centerline + solid-with-open-circle markers).
- Explicit `/api/cache/clear` endpoint + static-URL `?v=<app.version>`
  cache-busting to avoid confusion across code-change cycles.

### Terminology

- Retracted an incorrect "equivalent to Hemri 2014" claim in
  docstrings and the design doc. Correct framing is
  "sentinel-augmented mixed-distribution CRPS" citing Hersbach 2000,
  Gneiting & Raftery 2007, Ferro 2014, Leutbecher 2019.

---

## Interpretability pass

Every panel got a plain-language interp footer explaining what to
look for and how to read failure modes. The sidebar onset-criteria
section got an explanatory paragraph. Eyebrow labels were rewritten
from cryptic citations (`goessling`, `roberts & lean`) to descriptive
phrases (`area of disagreement through the season`, `spatial skill
vs neighborhood size`). The bench summary got horizontal scroll, the
progression legend got more vertical room, and all items from the
usability walkthrough were addressed.

---

## What's not done (explicit follow-ons)

- Onset start / end months are still hard-coded (`May 1 – Sep 30`
  plus `47-day extend`) — not yet query parameters.
- No bootstrap confidence intervals on metric aggregates; the low-n
  banner handles this qualitatively.
- Hausdorff / Fréchet don't report the grid-resolution floor.
- Nothing pushed to origin; nothing tagged for release.

---

## Current state

- API version: `0.2.6-interp-pass`
- 107 / 107 tests green
- 25 commits on `frontend` ahead of `main`
- Running against `aice_data`: four models (AIFS deterministic
  2019–2024, NGCM51 51-member 1965–1978 + 2019–2024, IFS-S2S 11-member
  2004–2023, FuXi-S2S 51-member 2002–2021), optional India land mask
- Branch is in a clean, shippable state for demos and internal review;
  not yet prepared for an open-source release (no `CHANGELOG`, no
  versioning beyond the API tag, etc.)

The work has moved from "adding metrics to the package" to
"adding metrics + an interactive verification environment for
comparing competing S2S models against IMD ground truth" —
substantially beyond the original two-milestone design doc.
