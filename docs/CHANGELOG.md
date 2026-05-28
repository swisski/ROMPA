# Changelog

All notable changes on the ROMPA fork beyond the baseline `bosup/ROMP`
package. Loose semver: PATCH bumps for the package; the frontend API
keeps its own version string in `frontend/api/app.py`.

The history is grouped by milestone, not by individual commit — for
per-commit detail use `git log`.

## [0.1.0-rompa-handoff] — 2026-05-28

Tagged for handoff to the upstream ROMP author. Everything below has
been merged into `main` and the fork is in a shippable state.

### Added — verification metrics in `momp/`

- **CRPS** (`momp/metrics/crps.py`) — sentinel-augmented
  mixed-distribution CRPS for the atom-plus-continuous onset
  distribution. Hersbach 2000 ensemble form, Ferro 2014 fair
  finite-ensemble correction default-on for `m ≥ 2`. Defends against
  CRPS = 0 sentinel-vs-sentinel pathology when both obs and all
  ensemble members report no onset.
- **FSS** (`momp/metrics/neighborhood.py`) — Fractions Skill Score
  (Roberts & Lean 2008). Box-average over `n × n` neighborhoods of
  binary onset-by-DOY masks. NaN-safe.
- **Displacement / area bias** (`momp/metrics/displacement.py`) —
  Haversine centroid km + area-bias % at a set of DOY thresholds.
- **CORP reliability** (`momp/graphics/corp_reliability.py`) —
  isotonic-regression calibration with MCB / DSC / UNC decomposition
  of the Brier score; proper-score identity residual at floating-point
  zero. Reports `N_eff` via Moran's I (Dutilleul 1993).
- **IOE + SPS** (`momp/metrics/progression.py`) — Integrated Onset
  Error (Goessling 2016 IIEE-analog) and Spatial Probability Score
  (Goessling & Jung 2018). SPS reduces exactly to IOE for a 1-member
  ensemble.
- **Isochrone geometry** (`momp/graphics/isochrone.py`) — contour
  extraction with NaN-sentinel substitution + Hausdorff + Fréchet
  distances via shapely.
- **CRA** (`momp/metrics/cra.py`) — Contiguous Rain Area decomposition
  per Ebert & McBride 2000. Initially contributed by Gio Hernandez
  (PR #1); polished on the fork as listed below.

### Added — interactive dashboard (`frontend/`)

- FastAPI backend with 14 endpoints (`/api/catalog`, `/api/state`,
  one per metric, `/api/compare`, `/api/cache/clear`, ...).
- Vanilla-JS + Plotly UI (~1700 LOC) with two-column sticky-sidebar
  layout, multi-model color-stable chip selection, per-panel error
  isolation, per-panel interpretation footers.
- 8 verification panels: cross-model summary table, isochrone hero
  overlay (three-cue rendering), progression curve (IOE + SPS),
  CORP reliability, CRPS field, displacement / area bias, FSS
  matrix, CRA bars + maps + shift-vector diagnostic.
- Multi-year aggregation with correct per-metric semantics: per-cell
  mean CRPS, pooled `(p, y)` pairs for CORP, median + IQR per-day
  for IOE / SPS / displacement, per-(τ, n) mean FSS, pooled
  per-slice percentages for CRA.

### Added — region support

- **India** — original target. 2° common grid (16×17 over India)
  end-to-end via `frontend/link_aice_data.sh`. Four models:
  AIFS deterministic (2019–2024), NGCM51 (51-member, 1965–1978 +
  2019–2024), IFS-S2S (11-member, 2004–2023), FuXi-S2S (51-member,
  2002–2021). Optional `ROMP_LAND_MASK=India`.
- **Ethiopia** — added late in development. 0.25° CHIRPS-IMERG obs
  + AIFS + GenCast forecast at the same grid. Linked via
  `frontend/link_ethiopia_data.sh`. Required:
  - Coord-name normalization (`LATITUDE`/`LONGITUDE`/`TIME` →
    `lat`/`lon`/`time`) in `frontend/api/{onset,rainfall}.py`.
  - `ROMP_OBS_END_EXTEND_DAYS=0` so Ethiopia's post-Sep Deyr
    rainfall isn't falsely claimed as a late Kiremt onset.
  - `ROMP_OBS_LABEL` (or NetCDF `title` attr) drives the obs label
    in the UI instead of a hard-coded "IMD".

### Fixed — numerical correctness

- Obs-to-model upsampling was inflating area-weighted metrics by
  the upsample ratio. Now downsamples the model onto the obs grid.
- Forecast onset detection had no May 1 lower bound, firing on
  pre-monsoon April rain. Now matches the observed-detector window.
- `ens.mean(skipna=True)` silently dropped no-onset members. Now
  sentinel-substituted median requires ≥ 50% agreement.
- Land-mask substring lookup could pick the wrong country
  (`Niger → Nigeria`, `Korea → South Korea`). Now exact-match-first
  with ambiguous rejection.
- CORP `n` overstated independence. Now reports `N_eff` via
  Moran's I.
- CRPS = 0 for ocean cells under a land mask inflated the mean.
  Now nulled.
- **CRA variable-mask denominator-shopping**: the optimizer's
  per-candidate `obs|fcst|shifted` mask let it lower MSE by
  inflating the denominator with zero-residual cells instead of
  by actually aligning rain. Fixed with a score mask computed once
  at zero shift and held constant across all candidate shifts.
  Regression test added.
- **CRA decomposition identity** now exact (within float
  tolerance). Old `max(., 0)` clipping in `cra_decomposition` is
  gone; an assertion now enforces
  `mse_total == mse_displacement + mse_volume + mse_pattern`.
- **CRA aggregated percentages** — old code took medians of per-year
  percentages, which don't sum to 100% across slices. Now pools
  `sum_slice_mse / sum_total_mse`; per-year q25/q75 still reported.
- **CRA ties** broke first-encountered, biasing toward maximum
  negative shifts. Now minimum-magnitude wins.
- **IOE/SPS curve plateau** past the forecast's lead horizon — the
  d-axis upper bound is now clipped to
  `min(obs_max_doy, fcst_max_doy)`, evaluating only within the DOY
  range both detectors could resolve.

### Fixed — robustness

- Isochrone contour extraction treated NaN as holes (spurious rings).
  Sentinel substitution.
- Multi-year CRPS response dropped `fair` / `n_members`. Restored.
- Single-year vs multi-year `season` dict schemas diverged. Unified.
- Empty `year=` or inverted year ranges returned 422 Pydantic
  errors. Now 400s.
- Error paths in displacement / FSS panels swallowed errors
  silently with stale plots. Now purge + in-plot error messages.
- Coincident isochrone lines were visually indistinguishable. Now
  three-cue rendering (halo + dashed centerline + solid-with-markers).
- `/api/cache/clear` + static-URL `?v=<app.version>` cache-busting.
- Hardcoded personal paths in `frontend/check_fresh.sh`,
  `frontend/validate.py`, and `tests/test_integration_real_pipeline.py`
  replaced with `sys.executable` / repo-relative / env-var lookups.
- Hardcoded shapefile path in `momp/utils/land_mask.py` replaced
  with relative-to-cwd search + `ROMP_SHAPEFILE_DIR` env override.

### Added — sub-cell CRA shift

- `momp.metrics.cra.shift_field` now supports float `(dy, dx)` via
  `scipy.ndimage.shift(order=1)`.
- `cra_decomposition(subgrid_factor=N)` searches at `1/N`-cell
  granularity. `N=1` keeps the original integer-cell sweep
  (default for the package API). `N=4–8` via the frontend.

### Added — CRA shift-vector diagnostic in the UI

A second panel below the existing CRA bars + 3-panel maps shows
the corrective shift as exactly what it is: a translation of the
forecast field. Two side-by-side panels (before / after) on the
*full* unmasked forecast (water + land) with the country outline
overlaid, plus a large amber arrow showing the (dy, dx) shift.
Makes obvious which cells moved from water to land under the
shift — the part the India-cropped 3-panel hides.

### Added — bundled-shapefile polygon outline helper

- `momp.utils.land_mask.country_polygon_coords(region)` returns the
  simplified mainland exterior as `{"lat": [...], "lon": [...]}`,
  preferring a bundled `aice_data/ind_map_shpfile/india_shapefile.shp`
  (when present, via `ROMP_SHAPEFILE_DIR` or sibling-dir search) and
  falling back to `regionmask`'s natural-earth polygon for arbitrary
  countries. Used to draw a clean coastline on heatmap panels
  instead of bilinear-mask isocontours.

### Terminology

- Retracted an incorrect "equivalent to Hemri 2014" claim in CRPS
  docstrings and the design doc. Correct citations:
  Hersbach 2000, Gneiting & Raftery 2007, Ferro 2014, Leutbecher
  2019.

### Documentation

- `docs/FORK_SUMMARY.md` — canonical narrative of every change on
  the fork.
- `docs/CHANGELOG.md` (this file).
- `docs/DESIGN_metrics_extension.md` — two-milestone design doc.
- `docs/METRICS_AND_PAPERS.md` — metric ↔ paper map, acronym
  glossary, and an explicit CRA-vs-fork-metrics comparison.
- `docs/example_milestone1_probabilistic.ipynb`,
  `docs/example_milestone2_progression.ipynb`,
  `docs/example_progression_analysis.ipynb`,
  `docs/example_realdata_cross_model.ipynb` — executed example
  notebooks.

### Frontend API version history

- `0.6.2-shift-float-display` (HEAD) — float-aware shift labels;
  CRA defaults updated to `max_shift=4`, `subgrid_factor=2`; CRA
  decomposition assertion; bundled-shapefile country outline on
  CRA panels; IOE/SPS d-axis intersection-clipped; obs label from
  NetCDF / env.
- `0.4.2-cra-objective` — India verification mask piped through
  `cra_decomposition`; higher-res CRA maps; corrective-shift arrow.
- `0.4.x` — CRA wired into the frontend (PR #1 merged).
- `0.2.6-interp-pass` — interpretability pass on every panel.
- `0.2.x` — three-cue isochrone overlay; isochrone shading.
- `0.1.x` — Milestone-2 metrics (IOE / SPS / isochrones) merged.

## [0.0.1] — 2026-04-17

Baseline ROMP per `bosup/ROMP`. Not changed in this fork.
