"""End-to-end smoke test for the ROMP metrics FastAPI backend.

Boots ``frontend.api.app:app`` via uvicorn on a free port, then hits every
public endpoint and validates response shape. Prints a summary table and
exits 0 on success, 1 on any failure.

Usage:
    python frontend/validate.py
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/home/alex/classwork/DSICLINIC/monsoon-bench/.venv/bin/python"
HOST = "127.0.0.1"
BOOT_TIMEOUT = 60.0
REQ_TIMEOUT = 120.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def wait_for_health(base: str, timeout: float) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base}/api/health", timeout=2.0)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


class Results:
    def __init__(self) -> None:
        self.rows: list[tuple[str, int, float, float, str, bool, str]] = []

    def record(self, label: str, resp: requests.Response, ms: float,
               ok: bool, note: str) -> None:
        size_kb = len(resp.content) / 1024.0
        try:
            body = resp.json()
            if isinstance(body, dict):
                hint = ",".join(list(body.keys())[:4])
            else:
                hint = type(body).__name__
        except Exception:
            hint = "non-json"
        self.rows.append((label, resp.status_code, ms, size_kb, hint, ok, note))

    def all_ok(self) -> bool:
        return all(r[5] for r in self.rows)

    def print_table(self) -> None:
        header = ("ENDPOINT", "STATUS", "MS", "KB", "KEYS", "OK", "NOTE")
        widths = [42, 6, 7, 8, 38, 3, 40]
        def fmt(cols):
            return "  ".join(str(c)[:w].ljust(w) for c, w in zip(cols, widths))
        print(fmt(header))
        print("-" * (sum(widths) + 2 * (len(widths) - 1)))
        for lbl, st, ms, kb, hint, ok, note in self.rows:
            mark = "OK" if ok else "FAIL"
            print(fmt((lbl, st, f"{ms:.0f}", f"{kb:.1f}", hint, mark, note)))


def call(session: requests.Session, base: str, path: str,
         params: dict | None = None) -> tuple[requests.Response, float]:
    t0 = time.time()
    r = session.get(f"{base}{path}", params=params, timeout=REQ_TIMEOUT)
    return r, (time.time() - t0) * 1000.0


def check(cond: bool, msg: str) -> tuple[bool, str]:
    return (True, "") if cond else (False, msg)


# ---------------------------------------------------------------------------
# shape validators (return (ok, note))
# ---------------------------------------------------------------------------
def v_health(j: dict) -> tuple[bool, str]:
    return check("status" in j and "version" in j, "missing status/version")


def v_catalog(j: dict) -> tuple[bool, str]:
    if "models" not in j or "shared_years" not in j:
        return False, "no models/shared_years"
    if len(j["models"]) < 1:
        return False, "no models in catalog"
    if len(j["shared_years"]) < 1:
        return False, "no shared_years"
    for field in ("root", "obs", "onset_defaults", "onset_docs"):
        if field not in j:
            return False, f"missing {field}"
    return True, f"{len(j['models'])}m/{len(j['shared_years'])}y"


def v_inits(j: dict) -> tuple[bool, str]:
    if not all(k in j for k in ("model", "year", "n", "inits")):
        return False, "missing keys"
    if j["n"] != len(j["inits"]):
        return False, "n != len(inits)"
    return True, f"n={j['n']}"


def v_state(j: dict) -> tuple[bool, str]:
    if "obs_onset" not in j or "values" not in j["obs_onset"]:
        return False, "no obs_onset.values"
    vals = j["obs_onset"]["values"]
    if not isinstance(vals, list) or not vals or not isinstance(vals[0], list):
        return False, "obs_onset.values not 2D"
    if not isinstance(j.get("is_ensemble"), bool):
        return False, "is_ensemble not bool"
    iso = j.get("iso_days", [])
    note = f"iso_days={len(iso)}"
    if len(iso) == 0:
        note += " (warn: empty)"
    return True, note


def v_crps(j: dict) -> tuple[bool, str]:
    if "field" not in j or "values" not in j["field"]:
        return False, "no field.values"
    vals = j["field"]["values"]
    if not isinstance(vals, list) or not vals or not isinstance(vals[0], list):
        return False, "field.values not 2D"
    if j.get("mean") is not None and not isinstance(j["mean"], (int, float)):
        return False, "mean wrong type"
    return True, f"mean={j.get('mean')}"


def v_fss(j: dict) -> tuple[bool, str]:
    thr, nbr, fss = j.get("thresholds"), j.get("neighborhoods"), j.get("fss")
    if not isinstance(fss, list):
        return False, "fss not list"
    if len(fss) != len(thr):
        return False, f"fss rows {len(fss)} != thr {len(thr)}"
    for row in fss:
        if len(row) != len(nbr):
            return False, f"fss cols {len(row)} != nbr {len(nbr)}"
    return True, f"{len(thr)}x{len(nbr)}"


def v_displacement(j: dict) -> tuple[bool, str]:
    for k in ("thresholds", "delta_lat_deg", "delta_lon_deg",
              "great_circle_km", "area_bias_fraction"):
        if k not in j:
            return False, f"missing {k}"
    n = len(j["thresholds"])
    for k in ("delta_lat_deg", "delta_lon_deg", "great_circle_km"):
        if len(j[k]) != n:
            return False, f"{k} len {len(j[k])} != {n}"
    return True, f"n={n}"


def v_progression(j: dict) -> tuple[bool, str]:
    if "days" not in j or "ioe_km2" not in j:
        return False, "missing days/ioe_km2"
    if len(j["days"]) != len(j["ioe_km2"]):
        return False, f"days {len(j['days'])} != ioe_km2 {len(j['ioe_km2'])}"
    return True, f"n_days={len(j['days'])}"


def v_isochrones(j: dict) -> tuple[bool, str]:
    for k in ("isochrones", "days", "hausdorff_km", "frechet_km"):
        if k not in j:
            return False, f"missing {k}"
    n = len(j["days"])
    if not (len(j["isochrones"]) == n == len(j["hausdorff_km"])):
        return False, "length mismatch"
    return True, f"n={n}"


def v_corp(j: dict) -> tuple[bool, str]:
    for k in ("mcb", "dsc", "unc", "mean_score", "tau"):
        if k not in j:
            return False, f"missing {k}"
    resid = abs(j["mcb"] - j["dsc"] + j["unc"] - j["mean_score"])
    if resid >= 1e-9:
        return False, f"identity residual {resid:.2e} >= 1e-9"
    return True, f"tau={j['tau']}, resid={resid:.1e}"


def v_compare(expected_rows: int):
    def _v(j: dict) -> tuple[bool, str]:
        if "rows" not in j:
            return False, "no rows"
        if len(j["rows"]) != expected_rows:
            return False, f"rows {len(j['rows'])} != {expected_rows}"
        return True, f"rows={len(j['rows'])}"
    return _v


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    port = free_port()
    base = f"http://{HOST}:{port}"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    print(f"[boot] launching uvicorn on {base} ...")
    proc = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "frontend.api.app:app",
         "--host", HOST, "--port", str(port), "--log-level", "warning"],
        env=env, cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    results = Results()
    try:
        if not wait_for_health(base, BOOT_TIMEOUT):
            print("[boot] FAILED: /api/health never came up", file=sys.stderr)
            try:
                err = proc.stderr.read().decode("utf-8", errors="replace")
                print(err, file=sys.stderr)
            except Exception:
                pass
            return 1
        print("[boot] healthy")

        s = requests.Session()

        # 1. health
        r, ms = call(s, base, "/api/health")
        ok, note = v_health(r.json()) if r.status_code == 200 else (False, f"HTTP {r.status_code}")
        results.record("/api/health", r, ms, ok, note)

        # 2. catalog
        r, ms = call(s, base, "/api/catalog")
        cat = r.json() if r.status_code == 200 else {}
        ok, note = v_catalog(cat) if r.status_code == 200 else (False, f"HTTP {r.status_code}")
        results.record("/api/catalog", r, ms, ok, note)
        if not ok:
            results.print_table()
            print("[fatal] cannot proceed without catalog")
            return 1

        models = cat["models"]
        shared_years = cat["shared_years"]
        year = shared_years[0]
        model_key = models[0]["key"]
        print(f"[pick] model={model_key} year={year} "
              f"(n_models={len(models)}, n_years={len(shared_years)})")

        # 3. inits
        r, ms = call(s, base, "/api/inits",
                     {"model": model_key, "year": year})
        ok, note = v_inits(r.json()) if r.status_code == 200 else (False, f"HTTP {r.status_code}")
        results.record("/api/inits", r, ms, ok, note)

        # 4. state
        r, ms = call(s, base, "/api/state",
                     {"model": model_key, "year": year})
        ok, note = v_state(r.json()) if r.status_code == 200 else (False, f"HTTP {r.status_code}")
        results.record("/api/state", r, ms, ok, note)

        # 5. crps
        r, ms = call(s, base, "/api/metrics/crps",
                     {"model": model_key, "year": year})
        ok, note = v_crps(r.json()) if r.status_code == 200 else (False, f"HTTP {r.status_code}")
        results.record("/api/metrics/crps", r, ms, ok, note)

        # 6. fss
        r, ms = call(s, base, "/api/metrics/fss",
                     {"model": model_key, "year": year,
                      "thresholds": "160,170,180", "neighborhoods": "1,3,5"})
        ok, note = v_fss(r.json()) if r.status_code == 200 else (False, f"HTTP {r.status_code}")
        results.record("/api/metrics/fss", r, ms, ok, note)

        # 7. displacement
        r, ms = call(s, base, "/api/metrics/displacement",
                     {"model": model_key, "year": year,
                      "thresholds": "160,170,180"})
        ok, note = v_displacement(r.json()) if r.status_code == 200 else (False, f"HTTP {r.status_code}")
        results.record("/api/metrics/displacement", r, ms, ok, note)

        # 8. progression
        r, ms = call(s, base, "/api/metrics/progression",
                     {"model": model_key, "year": year, "step": 5})
        ok, note = v_progression(r.json()) if r.status_code == 200 else (False, f"HTTP {r.status_code}")
        results.record("/api/metrics/progression", r, ms, ok, note)

        # 9. isochrones
        r, ms = call(s, base, "/api/metrics/isochrones",
                     {"model": model_key, "year": year,
                      "days": "160,175,190"})
        ok, note = v_isochrones(r.json()) if r.status_code == 200 else (False, f"HTTP {r.status_code}")
        results.record("/api/metrics/isochrones", r, ms, ok, note)

        # 10. corp
        r, ms = call(s, base, "/api/metrics/corp",
                     {"model": model_key, "year": year, "tau": 170})
        ok, note = v_corp(r.json()) if r.status_code == 200 else (False, f"HTTP {r.status_code}")
        results.record("/api/metrics/corp", r, ms, ok, note)

        # 11. compare
        keys_csv = ",".join(m["key"] for m in models[:2])
        expected = min(2, len(models))
        r, ms = call(s, base, "/api/compare",
                     {"models": keys_csv, "year": year})
        ok, note = v_compare(expected)(r.json()) if r.status_code == 200 else (False, f"HTTP {r.status_code}")
        results.record("/api/compare", r, ms, ok, note)

        # 12. onset params pass-through (wet_spell=3 vs 5)
        r1, ms1 = call(s, base, "/api/state",
                       {"model": model_key, "year": year, "wet_spell": 3})
        r2, ms2 = call(s, base, "/api/state",
                       {"model": model_key, "year": year, "wet_spell": 5})
        if r1.status_code == 200 and r2.status_code == 200:
            j1, j2 = r1.json(), r2.json()
            # server just needs to accept both; compare a couple of fields
            diff = (j1.get("init_idx") != j2.get("init_idx")
                    or j1.get("obs_range") != j2.get("obs_range")
                    or j1.get("fcst_range") != j2.get("fcst_range"))
            note = ("fields differ" if diff else "fields stable (accepted)")
            results.record("/api/state?wet_spell=3", r1, ms1, True, note)
            results.record("/api/state?wet_spell=5", r2, ms2, True, note)
        else:
            results.record("/api/state?wet_spell=3", r1, ms1,
                           r1.status_code == 200, f"HTTP {r1.status_code}")
            results.record("/api/state?wet_spell=5", r2, ms2,
                           r2.status_code == 200, f"HTTP {r2.status_code}")

        print()
        results.print_table()
        print()
        if results.all_ok():
            print("PASS: all endpoints OK")
            return 0
        fails = [r[0] for r in results.rows if not r[5]]
        print(f"FAIL: {len(fails)} endpoint(s) failed: {', '.join(fails)}")
        return 1

    finally:
        print("[teardown] killing uvicorn")
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        except Exception as e:
            print(f"[teardown] error: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
