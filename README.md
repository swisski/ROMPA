# ROMPA — ROMP with extra verification metrics and an interactive
dashboard

**ROMPA** is a fork of [bosup/ROMP](https://github.com/bosup/ROMP) —
the Rainy Season Onset Metrics Package — adding:

- **Six new verification metrics** layered on top of ROMP's baseline
  binned / spatial skill scores: sentinel-augmented CRPS, FSS,
  displacement / area bias, CORP reliability decomposition, the
  Integrated Onset Error (IOE) and Spatial Probability Score (SPS)
  for monsoon-front progression, and isochrone geometry
  (Hausdorff + Fréchet).
- **CRA (Contiguous Rain Area, Ebert & McBride 2000)** as a 7th
  metric — object-based MSE decomposition of raw rainfall into
  displacement / volume / pattern. Contributed by Gio Hernandez
  ([giomhern](https://github.com/giomhern)) via PR #1 and improved
  here with a fixed score mask (exact decomposition identity),
  sub-cell shift search, and an in-UI shift-vector diagnostic.
- **An interactive verification dashboard** (`frontend/`) — FastAPI
  + vanilla-JS + Plotly — for browsing every metric across multiple
  S2S models, multiple years, with year-range aggregation, multi-
  model overlay, and per-panel interpretation footers.
- **Region support for India and Ethiopia** out of the box, with the
  pipeline structured so adding another country is mostly a question
  of pointing `ROMP_DATA_ROOT` at a new symlink tree and setting
  `ROMP_LAND_MASK`.

The baseline ROMP package (`momp/`) — onset detection,
configuration plumbing, the binned-skill-score driver — is **left
intact and continues to work**. The fork only adds.

For the per-commit / per-feature narrative, including everything
fixed in audit passes, see **[docs/FORK_SUMMARY.md](docs/FORK_SUMMARY.md)**.
For the metric ↔ paper map and acronym glossary, see
**[docs/METRICS_AND_PAPERS.md](docs/METRICS_AND_PAPERS.md)**. For
release-level notes, see **[docs/CHANGELOG.md](docs/CHANGELOG.md)**.

---

## What's new at a glance

| Track                 | Where                                    | Status |
| --------------------- | ---------------------------------------- | ------ |
| CRPS (mixed dist'n)   | `momp/metrics/crps.py`                   | ✓      |
| FSS                   | `momp/metrics/neighborhood.py`           | ✓      |
| Centroid / area bias  | `momp/metrics/displacement.py`           | ✓      |
| CORP reliability      | `momp/graphics/corp_reliability.py`      | ✓      |
| IOE + SPS             | `momp/metrics/progression.py`            | ✓      |
| Isochrone geometry    | `momp/graphics/isochrone.py`             | ✓      |
| CRA decomposition     | `momp/metrics/cra.py`                    | ✓      |
| Dashboard backend     | `frontend/api/*.py`                      | ✓      |
| Dashboard frontend    | `frontend/static/*`                      | ✓      |
| India support         | env: `ROMP_LAND_MASK=India`              | ✓      |
| Ethiopia support      | env: `ROMP_LAND_MASK=Ethiopia`           | ✓      |

**Test count:** 175 unit / known-answer tests (12 of which are
integration tests, auto-skipped when demo data isn't present). All
green at HEAD.

---

## Quick start — dashboard

The fastest path to seeing the fork's contribution in action:

```bash
git clone https://github.com/swisski/ROMPA.git
cd ROMPA
python -m venv .venv && source .venv/bin/activate
pip install -e .[frontend]
./frontend/run.sh
```

With no data tree linked, `frontend/run.sh` falls back to the small
`demo/data/` bundled with the repo (2015 AIFS + IMD). Open
`http://127.0.0.1:8000`.

To run against the AICE four-model India bundle (AIFS deterministic,
NGCM51, IFS-S2S, FuXi-S2S) drop the `aice_data/` sibling repo next
to this one and run `./frontend/link_aice_data.sh`. For the Ethiopia
0.25° bundle (CHIRPS-IMERG + AIFS + GenCast), drop the three data
folders inside this repo and run `./frontend/link_ethiopia_data.sh`.
`run.sh` auto-picks whichever symlink tree exists.

See [`docs/FORK_SUMMARY.md`](docs/FORK_SUMMARY.md) §Region support
for the env-var table.

---

## Quick start — package only (CLI workflow)

The baseline ROMP CLI is unchanged:

```bash
pip install -e .
momp-run                  # original binned-skill-score workflow
momp-run-progression      # NEW — IOE + SPS + isochrones via Milestone-2 driver
```

Both read `params/config.in` and `params/region_def.py` the same way.

---

## Installation details

Same as upstream — Python 3.10–3.13. Heavy scientific deps
(`xarray`, `cartopy`, `geopandas`, `netcdf4`, `regionmask`) are
easier under conda; the `frontend` extra adds `fastapi`, `uvicorn`,
`scikit-learn`. The package is editable-installable with
`pip install -e .[dev,frontend]`.

```bash
# pip
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev,frontend]

# conda
conda create -n momp "python>=3.10"
conda activate momp
pip install -e .[dev,frontend]
```

Verify:

```bash
python -c "import momp; print(momp.__file__)"
pytest -q                                  # 175 tests, ~30s
pytest -m integration -v                   # +12 integration tests if demo data present
```

---

## Package organization (high level)

The original ROMP layout is preserved. New additions:

```
momp/
  metrics/
    crps.py             [new] sentinel-augmented mixed-distribution CRPS
    neighborhood.py     [new] Fractions Skill Score
    displacement.py     [new] centroid km + area-bias
    progression.py      [new] IOE + SPS
    cra.py              [new] CRA decomposition (Gio Hernandez + fork polish)
  graphics/
    corp_reliability.py [new] CORP / MCB-DSC-UNC
    isochrone.py        [new] contour extraction + Hausdorff/Fréchet
  app/
    progression_verification.py  [new] driver for IOE/SPS/iso
  utils/
    land_mask.py        [extended] regionmask country masks + bundled-shapefile
                                   override + polygon-coords helper for outlines

frontend/                [new] dashboard, see docs/FORK_SUMMARY.md
  api/                       FastAPI backend (14 endpoints)
  static/                    vanilla JS + Plotly UI

docs/
  FORK_SUMMARY.md       [new] canonical narrative — start here
  CHANGELOG.md          [new] per-release ledger
  DESIGN_metrics_extension.md   [new] two-milestone design doc
  METRICS_AND_PAPERS.md [new] metric ↔ paper map + acronym glossary
  example_milestone1_probabilistic.ipynb  [new] executed demo
  example_milestone2_progression.ipynb    [new] executed demo
  example_realdata_cross_model.ipynb      [new] real-data cross-model verification
  example_progression_analysis.ipynb      [new] IOE/SPS deep-dive
```

---

## Versioning

Semantic versioning per upstream. Package version in
`pyproject.toml` remains `0.0.1` (fork-only artifact, the original
authors hold version-bump authority). The frontend API is tagged
separately at `0.6.2-shift-float-display`.

---

## License

MIT, per upstream.

---

## Citation

If you use this fork in research, please cite both:

> Dong, B. et al. *ROMP: Rainy Season Onset Metrics Package.*
> UChicago HCWF Authors, 2026.

> Baumgartner, A. and Hernandez, G. *ROMPA: ROMP fork with
> additional verification metrics and an interactive dashboard.*
> UChicago DSI Capstone, 2026.

---

## Contact

- Upstream (ROMP): Bo Dong (`bodong@uchicago.edu`)
- Fork (ROMPA): Alex Baumgartner (`alex57baumgartner@gmail.com`)
- CRA contribution: Gio Hernandez ([giomhern](https://github.com/giomhern))
