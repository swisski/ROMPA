# ROMP fork — summary of work

A complete record of contributions on this fork of ROMP beyond the
baseline package. Covers (a) the verification metrics added to the
`momp` package, (b) the interactive verification dashboard built on
top, and (c) the Ethiopia region support layered on at the end of
development.

This document is the canonical narrative. Skim it first when picking
up the work, before running `git log` or grepping the codebase.

## Branch layout

```
main (HEAD)  — all fork work, single branch
origin/main  — bosup/ROMP baseline (fetched as a remote)
```

Everything has been merged into `main`. The classmate's CRA feature
branch (`giomhern/feature/cra`) was merged via PR #1 and the
`frontend` branch was merged at commit `2a3287e`.

- API version: `0.6.2-shift-float-display`
- Package version: 0.0.1 (see `pyproject.toml`; fork-only artifact)
- Test count: 175 (incl. integration suite, skipped without demo data)

## Where to start reading

1. **`README.md`** at the repo root — short pitch + install + run.
2. **`docs/CHANGELOG.md`** — the per-release ledger.
3. **`docs/DESIGN_metrics_extension.md`** — original two-milestone
   design doc for the metrics work.
4. **`docs/METRICS_AND_PAPERS.md`** — every metric ↔ its source
   paper, with explicit comparison vs the classmate's CRA.
5. **This file** — survey of everything end-to-end.

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
substitution to prevent spurious rings around no-data holes.
Hausdorff + Fréchet distances between forecast/observed contour pairs
via shapely.

**`momp/app/progression_verification.py`** + **`momp/driver_progression.py`** —
orchestration and the `momp-run-progression` CLI entry point.

### Classmate's contribution: CRA

**`momp/metrics/cra.py`** — Contiguous Rain Area, Ebert & McBride
2000. Object-based decomposition of raw-rainfall forecast MSE into
displacement / volume / pattern slices. Originally merged from
`giomhern/feature/cra` (PR #1). The classmate's metric is
object-based on raw rainfall; the metrics above are field-based on
derived onset DOY. They are intentionally complementary.

**CRA improvements made on this fork on top of the merged base:**

- **Fixed score mask.** The optimizer's per-candidate union mask
  (`obs ∪ fcst ∪ shifted`) was letting it lower MSE by enlarging the
  union with zero-residual cells instead of by actually aligning rain.
  Score mask is now computed once at zero shift and held constant
  across all candidate shifts. Regression test added.
- **Decomposition identity now exact.** Holding the mask constant
  across `mse_total` and `mse_shifted` makes
  `mse_total == mse_displacement + mse_volume + mse_pattern` hold
  exactly (within float tolerance). An assertion enforces it; the old
  silent `max(., 0)` clipping is gone.
- **Sub-cell shifts via `subgrid_factor`.** `1/N`-cell search via
  bilinear resampling (`scipy.ndimage.shift`). `N=1` keeps the
  original integer-cell sweep; default `N=8` gives 0.25° granularity
  on a 2° native grid.
- **Tie-break by minimum-magnitude shift.** Pure-volume-bias or
  constant-interior cases produce many ties — the old code picked
  whichever shift was hit first by the iteration order (the maximum
  negative shift, in practice). Ties now resolve to the smaller-
  magnitude shift.
- **India verification mask piped through `cra_decomposition`** so
  the optimizer can't game MSE by translating rain off India.
- **Pooled per-slice percentages in multi-year aggregation.** The
  median of per-year percentages doesn't sum to 100% across slices;
  the pooled `sum_slice_mse / sum_total_mse` does. Per-year spread is
  still reported as q25/q75 around the pooled value.

### Tests and documentation on `main`

- 175 unit / known-answer synthetic tests + 12 integration tests.
  Examples: degenerate ensemble → CRPS reduces to MAE;
  advancing-front → analytically integrated IOE matches; SPS on
  1-member → exactly equals IOE; CORP identity residual exactly 0;
  CRA pure-volume-bias inside a verification mask returns `(0, 0)`
  shift; CRA `subgrid_factor=4` recovers a 0.5-cell synthetic shift.
- `docs/DESIGN_metrics_extension.md` — two-milestone design doc with
  references, open questions, guardrails.
- `docs/METRICS_AND_PAPERS.md` — metric ↔ paper map, acronym
  glossary, explicit CRA-vs-fork-metrics comparison.
- `docs/example_milestone1_probabilistic.ipynb` and
  `docs/example_milestone2_progression.ipynb` — executed notebooks
  demonstrating the §7 success artifacts (CORP decomposition and
  isochrone overlay hero figure).

---

## Interactive verification dashboard (`frontend/`)

### Backend layering (`frontend/api/`)

**`catalog.py`** — data-source discovery. Scans `ROMP_DATA_ROOT`
(default `demo/data`, overridable for `aice_data` / Ethiopia),
reports available models, years, ensemble sizes. The obs label is
read from the NetCDF `title` attribute (CHIRPS) or
`ROMP_OBS_LABEL`, falling back to "observation" — so the same code
serves IMD over India and CHIRPS-IMERG over Ethiopia without
hard-coded strings.

**`onset.py`** — onset-field construction with process-wide caching
keyed on `(model, year, init, params)`. Observed onset goes through
the production `momp.stats.detect.detect_observed_onset` so obs DOYs
match what `momp-run` would produce. Forecast onset uses the same
criterion with an enforced May-1 lower-bound DOY so pre-monsoon April
rain in a forecast can't fire spurious onsets obs couldn't see.
Post-Sep search extension is configurable via
`ROMP_OBS_END_EXTEND_DAYS` (default 47, India-tuned) — set to 0 for
Ethiopia where post-Sep rainfall is a different rain system (Deyr)
and shouldn't be claimed as a late Kiremt onset.

**`metrics.py`** — JSON-ready metric serializers. Includes the
sentinel-median deterministic projection — cells get a forecast onset
only when ≥ 50% of ensemble members agree, treating no-onset
members as very-late sentinels — used for IOE and isochrones. Also
holds `moran_i_2d` (queen-4 spatial autocorrelation) and
`effective_sample_size` (Dutilleul 1993) for CORP's honest `N_eff`
reporting. `upsampled_field_payload` bilinearly densifies onset DOY
fields for display so isochrone contours look smooth on coarse AICE
grids (4° AIFS = 8×9, 2° ensembles = 16×17) — scoring metrics still
run on native cells.

**`aggregate.py`** — multi-year aggregation with correct per-metric
semantics: per-cell mean CRPS, pooled `(p, y)` pairs for CORP (not
averaging per-year decompositions — statistically the right combine),
median + IQR per-day for IOE / SPS / displacement, per-(τ, n) mean
FSS, pooled per-slice percentages for CRA.

**`app.py`** — FastAPI routing. 14 endpoints including `/api/health`,
`/api/catalog`, `/api/inits`, `/api/state`, one per metric,
`/api/compare`, `/api/cra`, `/api/cache/clear`. Obs-to-model grid
alignment via downsampling the forecast onto the obs grid. Optional
`ROMP_LAND_MASK=India` (or Ethiopia, etc.) via `regionmask` with
exact-match-first then unambiguous substring. The progression
endpoint clips the d-axis upper bound to the intersection of obs and
forecast onset DOY ranges — so the IOE/SPS curve doesn't plateau on
the asymmetric-NaN floor where the forecast detector ran out of lead
window but the obs detector kept resolving.

**`rainfall.py`** — raw-rainfall accumulator used by CRA. Caches the
sum over `[lead_start, lead_end]` for a given (model, year, init)
plus the observed accumulator over the matching valid window.
Normalizes any of `TIME`/`time`, `latitude`/`lat`/`LATITUDE`,
`longitude`/`lon`/`LONGITUDE` to lowercase short names so it works
across IMD (1901-2022 vs 2023+) and CHIRPS-IMERG file conventions.

### Frontend (`frontend/static/`)

- **`index.html`** — two-column layout: sticky sidebar controls,
  scrollable main column.
- **`app.css`** — observatory aesthetic: Fraunces serif headers,
  IBM Plex Sans/Mono body, deep monsoon-storm palette, amber + sky
  accents, grain overlay, per-card loading shimmers.
- **`app.js`** — ~1700 LOC vanilla JS driving state, multi-model
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
  dashed halo + thin dashed centerline; forecast as a crisp solid
  line with open-circle markers. NaN obs cells shown as a visible
  gray overlay. DOY labels staggered to prevent collisions.
- **Progression curve** — IOE (solid + markers), SPS (dotted) per
  model, stable color per model. Multi-year: median line + IQR band.
  Optional extent / misplacement decomposition toggle for the primary
  model.
- **CORP reliability** — calibration curve with MCB / DSC / UNC
  breakdown. `N` and `N_eff` (with Moran's I) in the caption.
- **CRPS field** — per-cell forecast error in days (Magma heatmap).
  Mean / median / IQR caption, fair-CRPS flag.
- **Displacement + area bias** — dual-axis line chart (great-circle
  km + area-bias %).
- **FSS matrix** — one line per DOY threshold, reference lines at the
  no-skill (`p`) and useful (`0.5 + 0.5·p`) levels per threshold.
- **CRA panel** — % MSE explained per slice (horizontal stacked bar
  per model), 3-panel obs/fcst/shifted rainfall maps with centroids
  and the corrective-shift arrow, plus a *shift diagnostic*
  before/after panel showing the full unmasked forecast field with
  the country outline overlaid so the viewer can see exactly what
  the translation does — including cells that move from water to
  land or vice versa.

### Scripts and tooling

- **`frontend/run.sh`** — auto-detects a Python interpreter with
  `uvicorn + momp`, auto-picks `.data_ethiopia/` (preferred when
  present) then `.data_aice/` as `ROMP_DATA_ROOT`. For Ethiopia,
  also defaults `ROMP_LAND_MASK=Ethiopia` and
  `ROMP_OBS_END_EXTEND_DAYS=0`.
- **`frontend/link_aice_data.sh`** — builds `.data_aice/` over the
  nested `aice_data/` sibling repo (four models). India runs at 2°
  throughout (obs + ensembles) — no obs↔model regridding inflation.
- **`frontend/link_ethiopia_data.sh`** — builds `.data_ethiopia/`
  over the in-repo CHIRPS-IMERG / AIFS / GenCast drops at 0.25°.
- **`frontend/validate.py`** — boots uvicorn on a free port, hits
  every endpoint, checks shapes + identities + param flow-through.
- **`frontend/check_fresh.sh`** — flags any cached model/year whose
  earliest forecast DOY < 121 (detects stale pre-fix caches).

---

## Region support

The fork supports India (original target) and Ethiopia (added late in
development) via the same pipeline. All region-specific behavior is
controlled by environment variables — no code changes needed to add a
new region as long as it's a `regionmask`/`natural_earth` country
name.

| Env var                       | Default   | India     | Ethiopia    |
| ----------------------------- | --------- | --------- | ----------- |
| `ROMP_DATA_ROOT`              | demo/data | .data_aice| .data_ethiopia |
| `ROMP_LAND_MASK`              | (off)     | India     | Ethiopia    |
| `ROMP_OBS_END_EXTEND_DAYS`    | 47        | 47        | 0           |
| `ROMP_OBS_LABEL`              | (NetCDF title or "observation") | "IMD" | "CHIRPS Version 3.0" |
| `ROMP_SHAPEFILE_DIR`          | sibling dirs | aice_data | (natural-earth fallback) |

The Ethiopia adapter required exactly three changes:

1. **Coord-name normalization** — CHIRPS uses uppercase
   `LATITUDE`/`LONGITUDE`/`TIME`. `_normalize_obs_time` and
   `_detect_obs` now normalize any of these to lowercase short names.
2. **Obs end-extension off** — Ethiopia's Deyr rains are a different
   system from the Kiremt monsoon ROMP detects, so post-Sep cells
   must stay NaN (not be claimed as very-late Kiremt). Configurable
   via `ROMP_OBS_END_EXTEND_DAYS=0`.
3. **Obs label decoupled from "IMD"** — `catalog.py` reads from the
   NetCDF `title` attr or `ROMP_OBS_LABEL`, falling back to
   "observation". Cosmetic but breaks the India-only assumption in
   the UI header.

---

## Bugs found and fixed during development

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
- CRA variable-mask denominator-shopping: the per-candidate
  `obs|fcst|shifted` mask let the optimizer lower MSE by inflating
  the denominator with zero-residual cells. Fixed to a fixed score
  mask + minimum-magnitude tie-break.
- CRA aggregated per-slice percentages were medians of per-year
  percentages — which don't sum to 100% across slices. Fixed to
  pooled `sum_slice_mse / sum_total_mse`.
- IOE/SPS progression curve plateaued on the asymmetric-NaN floor
  past the forecast's lead-window horizon. Fixed by clipping the
  d-axis upper bound to `min(obs_max_doy, fcst_max_doy)`.

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
  indistinguishable. Fixed with three-cue rendering (halo + dashed
  centerline + solid-with-markers).
- Explicit `/api/cache/clear` endpoint + static-URL `?v=<app.version>`
  cache-busting to avoid confusion across code-change cycles.
- CRA frontend was passing the wrong field names (`modelKey`/
  `modelLabel`) to the bars renderer after a refactor changed them
  to `key`/`label`. Fixed.

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
phrases. The bench summary got horizontal scroll, the progression
legend got more vertical room, the CRA panel gained a whole second
diagnostic panel that *shows* what the corrective shift physically
does to the forecast field rather than just printing the (dx, dy)
numbers.

---

## What's not done (explicit follow-ons)

- Onset start / end months are still hard-coded (`May 1 – Sep 30`
  plus the configurable extend) — not yet query parameters.
- No bootstrap confidence intervals on metric aggregates; the low-n
  banner handles this qualitatively.
- Hausdorff / Fréchet don't report the grid-resolution floor.
- Ethiopia obs is currently CHIRPS-IMERG only; multi-source
  comparison (e.g. CHIRPS vs ERA5-Land) would require adding a
  second `obs` slot to the catalog.
- The `monsoon-bench` interpreter is still in `run.sh`'s fallback
  search list (historical artifact from another sibling project) —
  harmless but worth pruning.

---

## Reproducing the dashboard

1. Clone this repo.
2. Either:
   - Drop `aice_data/` next to it (India), then
     `./frontend/link_aice_data.sh`; or
   - Drop `CHIRPS_IMERG/`, `aifs/0p25/`, `gencast/0p25/` inside the
     repo (Ethiopia), then `./frontend/link_ethiopia_data.sh`.
3. `pip install -e .[frontend]`
4. `./frontend/run.sh` and open `http://127.0.0.1:8000`.

`run.sh` picks whichever data tree exists and configures the land
mask + obs-end-extend defaults to match. Override either by setting
the env vars listed in the region-support table above.
