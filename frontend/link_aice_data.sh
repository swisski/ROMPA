#!/usr/bin/env bash
# Build the flat symlink tree ROMP expects at .data_aice/ from the nested
# layout in the sibling aice_data/ repo.
#
#   .data_aice/
#     ├── obs/YYYY.nc      -> aice_data/imd_rainfall_data/4p0/YYYY.nc     (8x9 lat/lon)
#     ├── aifs/YYYY.nc     -> aice_data/model_forecast_data/aifs/daily_0z/tp_4p0/YYYY.nc
#     ├── ngcm51/YYYY.nc   -> aice_data/model_forecast_data/ngcm51/anomaly/tp_2p0/YYYY.nc
#     ├── ifs_s2s/YYYY.nc  -> aice_data/model_forecast_data/IFS-S2S/tp_2p0/YYYY.nc
#     └── fuxi_s2s/YYYY.nc -> aice_data/model_forecast_data/fuxi_s2s/tp_2p0/YYYY.nc
#
# Obs is at 4p0 (8x9). Ensembles are at 2p0 (16x17) — the backend nearest-
# neighbor interpolates obs onto each model's grid at request time.
#
# Run once, then `./frontend/run.sh` auto-picks this tree (see run.sh).
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
AICE="${ROMP_AICE_ROOT:-$REPO_ROOT/../aice_data}"
LINK="$REPO_ROOT/.data_aice"

if [ ! -d "$AICE" ]; then
    echo "aice_data tree not found at $AICE" >&2
    echo "Set ROMP_AICE_ROOT to override." >&2
    exit 1
fi

rm -rf "$LINK"
mkdir -p "$LINK/obs" "$LINK/aifs" "$LINK/ngcm51" "$LINK/ifs_s2s" "$LINK/fuxi_s2s"

link_dir() {
    local src="$1" dst="$2" count=0
    [ -d "$src" ] || { echo "  skipping $dst (source missing: $src)"; return 0; }
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

link_dir "$AICE/imd_rainfall_data/4p0"                             obs
link_dir "$AICE/model_forecast_data/aifs/daily_0z/tp_4p0"          aifs
link_dir "$AICE/model_forecast_data/ngcm51/anomaly/tp_2p0"         ngcm51
link_dir "$AICE/model_forecast_data/IFS-S2S/tp_2p0"                ifs_s2s
link_dir "$AICE/model_forecast_data/fuxi_s2s/tp_2p0"               fuxi_s2s

echo
echo "Done. Start the frontend with:"
echo "    ROMP_DATA_ROOT=$LINK ./frontend/run.sh"
echo "(run.sh picks .data_aice automatically if it exists.)"
