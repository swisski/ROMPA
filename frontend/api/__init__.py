"""ROMP frontend API package.

Force matplotlib into the headless Agg backend BEFORE any submodule
imports pyplot. Running the default Tk backend from a uvicorn worker
thread raises ``RuntimeError: main thread is not in main loop`` and
``Tcl_AsyncDelete: async handler deleted by the wrong thread`` because
Tk's cleanup requires the main thread.
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
