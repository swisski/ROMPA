"""Data-source catalog.

Discovers models and years available under a configured data root. By
default this is ``ROMPA/demo/data`` so the frontend works out-of-the-box
with the package's shipped sample; set the ``ROMP_DATA_ROOT`` environment
variable to point at a richer tree (e.g. ``aice_data`` or ``monsoon-bench``
sibling repos).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import xarray as xr

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_DATA_ROOT = REPO_ROOT / "demo" / "data"
DATA_ROOT = Path(os.environ.get("ROMP_DATA_ROOT", str(DEFAULT_DATA_ROOT)))

OBS_KEY = "obs"
YEAR_PATTERN = re.compile(r"^(\d{4})\.nc$")


@dataclass(frozen=True)
class ModelInfo:
    key: str
    label: str
    path: Path
    years: tuple[int, ...]
    is_ensemble: bool
    n_members: int
    var_name: str

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "years": list(self.years),
            "is_ensemble": self.is_ensemble,
            "n_members": self.n_members,
        }


@dataclass(frozen=True)
class ObsInfo:
    key: str
    label: str
    path: Path
    years: tuple[int, ...]
    var_name: str

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "years": list(self.years),
            "var": self.var_name,
        }


# Canonical model labels; unknown keys fall back to a title-cased path name.
_LABELS = {
    "aifs": "AIFS (deterministic)",
    "ngcm": "NGCM (51-member)",
    "ngcm51": "NGCM51",
    "ifs": "IFS (ensemble)",
    "ifs-s2s": "IFS-S2S",
    "fuxi": "FuXi",
    "fuxi_s2s": "FuXi-S2S",
    "gencast52": "GenCast-52",
    "graphcast37": "GraphCast-37",
}


def _years_for_dir(d: Path) -> tuple[int, ...]:
    ys: list[int] = []
    for p in d.glob("*.nc"):
        m = YEAR_PATTERN.match(p.name)
        if m:
            ys.append(int(m.group(1)))
    return tuple(sorted(set(ys)))


def _ensemble_info(sample_nc: Path) -> tuple[bool, int, str]:
    try:
        with xr.open_dataset(sample_nc) as ds:
            var = next(iter(ds.data_vars))
            da = ds[var]
            is_ens = "number" in da.dims
            n = int(da.sizes["number"]) if is_ens else 1
            return is_ens, n, var
    except Exception:
        return False, 1, "tp"


def _label_for(key: str) -> str:
    return _LABELS.get(key.lower(), key.replace("_", " ").upper())


@lru_cache(maxsize=1)
def load_catalog(root: Path = DATA_ROOT) -> dict:
    """Scan ``root`` once; return a dict with ``models`` and ``obs`` lists."""
    models: list[ModelInfo] = []
    obs_info: ObsInfo | None = None

    if not root.exists():
        return {"root": str(root), "models": [], "obs": None}

    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        years = _years_for_dir(sub)
        if not years:
            continue
        if sub.name.lower() == OBS_KEY:
            with xr.open_dataset(sub / f"{years[0]}.nc") as ds:
                var = next(iter(ds.data_vars))
            obs_info = ObsInfo(key="imd", label="IMD / observation",
                               path=sub, years=years, var_name=var)
            continue
        sample = sub / f"{years[0]}.nc"
        is_ens, n_members, var = _ensemble_info(sample)
        models.append(ModelInfo(
            key=sub.name, label=_label_for(sub.name), path=sub, years=years,
            is_ensemble=is_ens, n_members=n_members, var_name=var,
        ))
    return {"root": str(root), "models": models, "obs": obs_info}


def model_by_key(key: str) -> ModelInfo:
    cat = load_catalog()
    for m in cat["models"]:
        if m.key == key:
            return m
    raise KeyError(f"model '{key}' not in catalog (have: {[m.key for m in cat['models']]})")


def obs_source() -> ObsInfo:
    cat = load_catalog()
    if cat["obs"] is None:
        raise RuntimeError(f"no observation directory under {cat['root']}")
    return cat["obs"]


def models_for_year(year: int) -> Iterable[ModelInfo]:
    for m in load_catalog()["models"]:
        if year in m.years:
            yield m


def shared_years() -> tuple[int, ...]:
    """Years that both the observation set and at least one model cover."""
    cat = load_catalog()
    if cat["obs"] is None or not cat["models"]:
        return ()
    model_years: set[int] = set()
    for m in cat["models"]:
        model_years.update(m.years)
    return tuple(sorted(set(cat["obs"].years) & model_years))
