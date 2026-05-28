#!/usr/bin/env bash
# Build .data_ethiopia/ from the in-repo Ethiopia drop:
#
#   .data_ethiopia/
#     ├── obs/YYYY.nc      -> CHIRPS_IMERG/YYYY.nc  (CHIRPS-IMERG ground truth)
#     ├── aifs/YYYY.nc     -> aifs/0p25/YYYY.nc     (AIFS deterministic, 0.25°)
#     └── gencast/YYYY.nc  -> gencast/0p25/YYYY.nc  (GenCast 32-member, 0.25°)
#
# All at 0.25° on the same 49x61 lat/lon grid (lat 3-15, lon 33-48). Run
# once, then `./frontend/run.sh` auto-picks this tree (see run.sh).
# Sources are optional individually — directories that don't exist are
# skipped (so dropping in additional models later just means re-running).
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
OBS_SRC="${ROMP_ETH_OBS:-$REPO_ROOT/CHIRPS_IMERG}"
AIFS_SRC="${ROMP_ETH_AIFS:-$REPO_ROOT/aifs/0p25}"
GENCAST_SRC="${ROMP_ETH_GENCAST:-$REPO_ROOT/gencast/0p25}"
LINK="$REPO_ROOT/.data_ethiopia"

if [ ! -d "$OBS_SRC" ]; then
    echo "Ethiopia obs source missing: $OBS_SRC" >&2
    exit 1
fi

rm -rf "$LINK"
mkdir -p "$LINK/obs"

link_dir() {
    local src="$1" dst="$2" count=0
    if [ ! -d "$src" ]; then
        echo "  $dst: SKIPPED (source missing: $src)"
        return 0
    fi
    mkdir -p "$LINK/$dst"
    for f in "$src"/*.nc; do
        [ -e "$f" ] || continue
        local b
        b=$(basename "$f")
        case "$b" in *:Zone.Identifier) continue;; esac
        ln -sf "$f" "$LINK/$dst/$b"
        count=$((count + 1))
    done
    echo "  $dst: $count files"
}

link_dir "$OBS_SRC"     obs
link_dir "$AIFS_SRC"    aifs
link_dir "$GENCAST_SRC" gencast

echo
echo "Done. Start the frontend with:"
echo "    ROMP_DATA_ROOT=$LINK ROMP_LAND_MASK=Ethiopia ./frontend/run.sh"
echo "(run.sh picks .data_ethiopia automatically if it exists.)"
