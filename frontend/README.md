# ROMP onset-metrics frontend

Zero-build Plotly UI over a small FastAPI backend for exploring monsoon
onset as an advancing front. See the repository root README for what
ROMP is; this directory is just the interactive viewer for the new M1+M2
metric stack.

Scope vs. the `ROMP_frontend_POC` sibling: the POC is broader and wraps
the original skill-score pipeline with a React/Vite frontend. This
frontend is narrower and newer — it focuses on the M1 probabilistic
depth metrics (CRPS, FSS, displacement, CORP) and the M2 progression /
isochrone narrative, across multiple models side by side.

## Install

```bash
uv venv .venv
uv pip install -e '.[frontend]'
```

This creates `.venv/` in the repo root and installs momp editable plus
fastapi, uvicorn, and scikit-learn. No node, no bundler, no build step.
The static page is served directly by FastAPI.

## Run

```bash
./frontend/run.sh
```

Then open <http://127.0.0.1:8000>.

No venv activation required — `run.sh` auto-detects a Python
interpreter that has both `uvicorn` and `momp` installed. It tries, in
order: `$PY` (if set), `.venv/bin/python`, `../monsoon-bench/.venv/bin/python`,
then `python3` / `python` on PATH. If none of them satisfies the import
check you get a clear install hint instead of a traceback.

Override the interpreter with `PY=/path/to/python ./frontend/run.sh`.

## Pointing at richer data

By default the app reads from `demo/data/` in the repo. Override with:

```bash
export ROMP_DATA_ROOT=/path/to/some/model_tree
```

Restrict every metric to land cells of a named country by setting
`ROMP_LAND_MASK` (Natural Earth 1:10m country name or substring). The
default is `India`, since IMD obs is the only ground truth bundled with
the demo and forecast skill is undefined outside its coverage:

```bash
export ROMP_LAND_MASK=India   # default; override or set empty to disable
export ROMP_LAND_MASK=        # leave all cells in play
```

When set, obs and forecast onset-DOY fields are masked to land-only
before every metric is computed. The catalog surfaces the current value
so the UI can show a `Mask: India` indicator in the masthead.

The expected layout is:

```
$ROMP_DATA_ROOT/
├── obs/
│   ├── 2015.nc
│   ├── 2016.nc
│   └── ...
├── aifs/
│   ├── 2015.nc
│   └── ...
├── ngcm/
│   └── ...
└── <other-model>/
    └── YYYY.nc
```

One NetCDF per `(model, year)`. Variable names inside each file are
auto-detected — the catalog uses the first data variable, so any sane
rainfall field works without renaming. Models whose NetCDFs carry a
`number` dim are treated as ensembles and surface their member count in
`/api/catalog`.

## UI tour

Sidebar controls:

- **Year** — restricted to years present for both obs and at least one model.
- **Model chips** — multi-select. Leftmost selected chip is the *primary*
  model; other panels that only support one model use this one.
- **Init picker** — shows initialization dates available for
  `(primary model, year)`; `auto` picks the earliest valid init.
- **Onset criteria** — six inputs matching `OnsetParams` (rain
  threshold, accumulation window, dry-spell tolerance, etc.).
- **Region bbox** — optional `lat_min/lat_max/lon_min/lon_max` clip.

Panels:

- **Cross-model summary table** — one row per selected model with the
  headline scalars (CRPS mean, FSS at default scale, centroid shift,
  CORP MCB/DSC/UNC).
- **Hero isochrone overlay** — observed vs. forecast contours of the
  onset front for selected iso-days, primary model.
- **Progression curves** — IOE / extent / misplacement / SPS through the
  season, overlaid for every selected model.
- **CORP reliability** — primary-model calibration curve plus the
  MCB / DSC / UNC bar.
- **CRPS heatmap** — per-cell censored CRPS for the primary model.
- **Centroid displacement + area bias** — primary-model sweep across
  thresholds.
- **FSS matrix** — primary-model FSS over the threshold × neighborhood
  grid.

## API reference

All metric endpoints accept the onset-criteria query params (matching
`OnsetParams`) and an optional region bbox; defaults come from
`/api/catalog`.

| Endpoint | Purpose | Key params |
| --- | --- | --- |
| `GET /api/health` | Liveness ping. | — |
| `GET /api/catalog` | Discoverable models, years, obs source, shared years, onset-param defaults + docs. | — |
| `GET /api/inits` | Init dates available for a `(model, year)`. | `model`, `year` |
| `GET /api/state` | Onset fields (obs, member mean, members) plus suggested iso-days. | `model`, `year`, `init=auto\|INT` |
| `GET /api/metrics/crps` | Per-cell censored CRPS for one model. | `model`, `year`, `init` |
| `GET /api/metrics/fss` | FSS sweep over thresholds × neighborhoods. | `thresholds`, `neighborhoods` |
| `GET /api/metrics/displacement` | Centroid shift + area bias across thresholds. | `thresholds` |
| `GET /api/metrics/progression` | IOE / extent / misplacement / SPS through the season. | `step` |
| `GET /api/metrics/isochrones` | Observed + forecast isochrone polylines, plus Hausdorff / Fréchet. | `days` |
| `GET /api/metrics/corp` | CORP MCB / DSC / UNC decomposition + reliability curve. | `tau` |
| `GET /api/compare` | Side-by-side scalar summary for several models. | `models=a,b,c`, `year` |

Backend layering:

- `frontend/api/catalog.py` — data discovery; reads `demo/data/` or
  `ROMP_DATA_ROOT`.
- `frontend/api/onset.py` — wraps `momp.stats.detect.detect_onset` with
  a process-wide cache keyed by `(model, year, init, params)` and a
  parameterized `OnsetParams` + optional `Region` bbox.
- `frontend/api/metrics.py` — turns each metric output into a
  JSON-friendly dict.
- `frontend/api/app.py` — FastAPI routes using
  `Depends(onset_deps)` + `Depends(region_deps)`.

## Layout

```
frontend/
├── api/
│   ├── app.py
│   ├── catalog.py
│   ├── onset.py
│   └── metrics.py
├── static/
│   ├── index.html
│   ├── app.css
│   └── app.js
├── run.sh
├── validate.py   (optional backend smoke test)
└── README.md
```

## Troubleshooting

The first request for a given `(model, year, init, onset-params)` tuple
runs the full onset detection and can take several seconds. Subsequent
requests for the same tuple hit an in-process cache and return
immediately. The cache lives in the FastAPI worker, so it clears on
process restart — re-running `./frontend/run.sh` warms from cold again.
