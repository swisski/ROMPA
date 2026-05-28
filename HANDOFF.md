# ROMPA — handoff doc

A short note from the maintainers of the fork
([Alex Baumgartner](mailto:alex57baumgartner@gmail.com) and
[Gio Hernandez](https://github.com/giomhern)) to Bo Dong.

Bo — this is what we built on top of `bosup/ROMP` during the
UChicago DSI Capstone. The fork lives at
**https://github.com/swisski/ROMPA**. We did not open an upstream
PR because the diff is large and structurally divergent; we wanted
to give you the chance to look at the additions and decide which
pieces (if any) are worth pulling in piecemeal vs adopting the fork
wholesale.

> **Status: work in progress.** This is a snapshot at the end of the
> capstone term, not a finished release. The two tracks we'd point at
> as closest to publication-ready are **CRA** (algorithmic identity
> nailed down, sub-cell shifts, regression test, shift-vector
> diagnostic in the UI) and the **progression-curve metrics
> (IOE + SPS)** plus their **isochrone overlay** (design doc, executed
> notebooks, intersection-clipped d-axis, three-cue rendering). The
> other four metrics — CRPS, FSS, displacement, CORP — are functional
> and tested, but had less iteration. Treat them as v0.x. The
> dashboard is a daily-driver for our own analysis, but it's still
> rough around the edges in places we call out below.

This document tells you what's there, how to try it in five minutes,
and what we'd flag if we were reviewing the diff.

## What we added

Two tracks (metrics + dashboard). Polish level inside the metrics
track is uneven on purpose — we deepened the two we picked as
capstone deep-dives (CRA + progression curve) and left the rest at
v0.x.

1. **Seven new verification metrics in `momp/`**.
   - **Most polished:** progression curve (IOE + SPS, with isochrone
     geometry) — `momp/metrics/progression.py`, `momp/graphics/isochrone.py`.
     This is the one with the methods-doc-style write-up
     (`docs/DESIGN_metrics_extension.md` §8), executed notebooks,
     d-axis intersection clipping, three-cue rendering, and the
     longest test list.
   - **Also polished:** CRA (Ebert & McBride 2000) — `momp/metrics/cra.py`.
     Initially contributed by Gio Hernandez via PR #1, then
     reworked on the fork: fixed score mask (decomposition identity
     now exact + asserted), sub-cell shift search, regression test
     for the variable-mask denominator bug, UI shift-vector
     diagnostic.
   - **Functional but earlier-stage:** CRPS (`momp/metrics/crps.py`),
     FSS (`momp/metrics/neighborhood.py`), displacement / area bias
     (`momp/metrics/displacement.py`), CORP reliability
     (`momp/graphics/corp_reliability.py`). All tested and wired
     into the dashboard, but they didn't get the same number of
     iteration rounds as CRA and progression. Treat them as the
     first cut.

   Each metric tracks back to a specific paper; the map is in
   `docs/METRICS_AND_PAPERS.md`.

2. **An interactive verification dashboard** (`frontend/`) — FastAPI +
   vanilla-JS + Plotly — that lets you compare any subset of S2S
   models across any year range against IMD/CHIRPS observations, on
   any of the metrics, with multi-year aggregation and per-panel
   interpretation footers.

We also added region support for **India** and **Ethiopia** via a
small set of env vars. Adding a third country should be one
shapefile drop + one symlink-tree script away.

The baseline ROMP CLI (`momp-run`, `params/config.in`,
`params/region_def.py`) is untouched and continues to work.

## Five-minute try

```bash
git clone https://github.com/swisski/ROMPA.git
cd ROMPA
python -m venv .venv && source .venv/bin/activate
pip install -e .[frontend]
./frontend/run.sh
```

Open `http://127.0.0.1:8000`. With no data tree linked the dashboard
runs against the small `demo/data/` bundle (2015 AIFS + IMD) — enough
to see every panel render.

To run against the four-model India bundle from the AICE project,
drop the `aice_data/` sibling repo next to this one and run
`./frontend/link_aice_data.sh` once. For the 0.25° Ethiopia bundle,
drop `CHIRPS_IMERG/`, `aifs/0p25/`, `gencast/0p25/` into the repo
root and run `./frontend/link_ethiopia_data.sh`.

`run.sh` auto-picks whichever data tree exists and sets the matching
land mask / observation defaults.

## Where to look

| File | What it is |
| --- | --- |
| `README.md` | Pitch + install + run. |
| `docs/FORK_SUMMARY.md` | Canonical narrative. Read this first. |
| `docs/CHANGELOG.md` | Per-release ledger. |
| `docs/DESIGN_metrics_extension.md` | Two-milestone design doc that drove the metric additions. |
| `docs/METRICS_AND_PAPERS.md` | Each metric ↔ its source paper, with an explicit CRA-vs-fork-metrics comparison table. |
| `docs/example_*.ipynb` | Four executed example notebooks (probabilistic, progression, real-data cross-model). |
| `momp/metrics/`, `momp/graphics/` | Where the new metrics live. |
| `momp/utils/land_mask.py` | Extended with bundled-shapefile lookup + polygon outline helper. |
| `frontend/` | The dashboard. `README.md` inside has its own quick-start. |
| `tests/` | 175 unit / known-answer tests + 12 integration tests (skipped when demo data is absent). All green at HEAD. |

## What we'd flag in review

In rough priority order.

**Worth a careful look:**

- The **CRA decomposition identity** fix and the regression test
  for it (`tests/test_cra.py::test_cra_pure_volume_bias_with_verification_mask`,
  `momp/metrics/cra.py::cra_decomposition`). The original code
  silently clipped negative mse_displacement / mse_pattern with
  `max(., 0)` and used a variable per-candidate score mask that
  let the optimizer game MSE by inflating the denominator. The fix
  makes the identity hold exactly and asserts it.
- The **CORP `N_eff` via Moran's I** (Dutilleul 1993) in
  `frontend/api/metrics.py`. The reliability decomposition was
  reporting `N` = cell count, overstating independence on the
  spatially autocorrelated onset field.
- The **forecast-onset May 1 lower bound** and the **sentinel-
  median ensemble projection** in `frontend/api/onset.py`. Both
  fix numerical pathologies that would inflate skill scores on
  the original detector.
- The **IOE/SPS d-axis intersection clip** in
  `frontend/api/app.py::_global_progression_window`. Without it
  the curve plateaus on the asymmetric-NaN floor past the
  forecast's lead horizon — a permanent visual offset that has
  nothing to do with forecast skill.

**Probably uncontroversial:**

- The metric additions themselves (`crps.py`, `neighborhood.py`,
  `displacement.py`, `progression.py`, `corp_reliability.py`,
  `isochrone.py`, `cra.py`). Each is self-contained, tested with
  known-answer cases, and accompanied by the design doc.
- The `momp.utils.land_mask` extensions (`country_polygon_coords`,
  `country_mask` exact-match-first lookup).
- The Ethiopia region adapter (env-var driven, no code branches).

**Stuff we'd punt to a follow-up:**

- Onset start/end months still hard-coded (`May 1 – Sep 30` plus
  the configurable extend). Should become query parameters.
- No bootstrap CIs on metric aggregates — the low-n banner handles
  this qualitatively.
- Hausdorff / Fréchet don't report the grid-resolution floor.
- The `monsoon-bench` interpreter is still in `frontend/run.sh`'s
  fallback search list (historical artifact from another sibling
  project) — harmless but worth pruning.
- Frontend API versioning is ad-hoc strings, not semver.

## Adoption paths

Three ways to take what's here:

1. **Cherry-pick metrics.** Each `momp/metrics/*.py` file is
   self-contained — you can pull just CRPS, or just CRA, into your
   own `main` without taking the dashboard. The test files mirror
   the structure (`tests/test_crps.py`, `tests/test_cra.py`, ...).
2. **Adopt the fork as your next minor release.** Squash-merge or
   rebase as you prefer. We're happy to rebase ourselves on a
   target branch you specify.
3. **Leave it as a sibling project.** The fork can live at
   `swisski/ROMPA` indefinitely; we'll keep maintaining it and
   point downstream users here for the dashboard while you
   continue developing baseline ROMP.

We have no preference — whichever fits your timeline.

## Open questions for you

A couple of things we'd be glad to hear your read on:

- The CRPS sentinel substitution (`season_end + 1`). We picked the
  smallest sentinel that strictly orders all "no onset" outcomes
  past the latest possible real onset. Is there an internal
  convention at HCWF you'd prefer?
- The land-mask substring lookup. We changed it from "first
  substring match" to "exact-match-first, ambiguous-substring-
  rejected" because the old behavior could pick Nigeria for
  `Niger`. If you have downstream callers that rely on the old
  behavior we should know.
- Whether you want us to open an upstream PR for the
  `region_polygon_coords` / `country_mask` additions to
  `momp/utils/land_mask.py` as a small first piece — those are
  low-controversy and would unblock pulling more of the dashboard
  later if you want to.

## Contact

- Alex Baumgartner — `alex57baumgartner@gmail.com`
  (lead on metric additions, dashboard, region support)
- Gio Hernandez — [giomhern on GitHub](https://github.com/giomhern)
  (CRA contribution)

Either of us is happy to do a walkthrough by video call if that's
useful. Thanks for ROMP — building on top of it was a pleasure.
