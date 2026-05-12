#!/usr/bin/env bash
# Probe the running ROMP frontend backend and confirm it's serving the
# min-DOY fix (no forecast onset before May 1 = DOY 121).
set -e
PORT="${ROMP_FRONTEND_PORT:-8000}"
HOST="${ROMP_FRONTEND_HOST:-127.0.0.1}"
BASE="http://${HOST}:${PORT}"
PY=/home/alex/classwork/DSICLINIC/monsoon-bench/.venv/bin/python

if ! curl -s -m 2 "$BASE/api/health" > /dev/null; then
    echo "no server reachable at $BASE"
    exit 1
fi

echo "=== /api/health ==="
curl -s "$BASE/api/health"
echo

echo
echo "=== earliest forecast DOY for each cached (model, year) combo ==="
echo "(should all be >= 121; anything lower means stale cache or old code)"
for model in aifs ngcm51 ifs_s2s fuxi_s2s; do
    for year in 2021 2022 2023 2024; do
        out=$(curl -s -w "|%{http_code}" "$BASE/api/state?model=${model}&year=${year}" 2>/dev/null)
        code=$(echo "$out" | awk -F'|' '{print $NF}')
        body=$(echo "$out" | sed 's/|[0-9]*$//')
        if [ "$code" != "200" ]; then continue; fi
        min_doy=$(echo "$body" | "$PY" -c "
import json, sys
d = json.load(sys.stdin)
flat = [v for row in d['fcst_onset']['values'] for v in row if v is not None]
print(f\"{min(flat):.1f}\" if flat else 'n/a')
" 2>/dev/null)
        flag=""
        if [ -n "$min_doy" ] && [ "$min_doy" != "n/a" ]; then
            awk "BEGIN { exit !($min_doy < 121) }" && flag=" ← STALE (< 121)"
        fi
        printf "  %-10s %-5s  earliest fcst DOY = %s%s\n" "$model" "$year" "$min_doy" "$flag"
    done
done
