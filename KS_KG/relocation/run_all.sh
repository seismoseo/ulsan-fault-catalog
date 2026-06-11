#!/usr/bin/env bash
# Reproducible BATCH relocation of every 5-25 Hz multiplet family (reuse scheme) + aggregation.
#
# 1. batch_relocate.py  — loops all 117 families (largest-first), running the validated per-family reuse
#    pipeline (make_catalog -> scaffold -> stage -> gather -> HypoInverse+ph2dt+dt.ct+dt.cc+HypoDD ->
#    bootstrap -> tidy table). Robust (failures logged + skipped, never aborts) and RESUMABLE (already-
#    relocated families are skipped), writing batch_manifest.csv. Any extra args pass straight through
#    (e.g. ./run_all.sh --no-bootstrap, --families 738,1218, --limit 5, --redo).
# 2. aggregate_results.py — master_metrics.csv + master_map_relocated.png + top-N fault-frame thumbnails.
#
# Prereqs: Fortran hyp1.40/ph2dt/hypoDD on PATH; the pq-gpu env (obspy, torch+EQNet, pygmt). Override the
# interpreter with  PQ_PY=/path/to/python ./run_all.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PQ_PY:-/home/msseo/miniforge3/envs/pq-gpu/bin/python}

"$PY" "$HERE/batch_relocate.py" "$@"
"$PY" "$HERE/aggregate_results.py"
"$PY" "$HERE/build_batch_summary_nb.py"
( cd "$HERE" && "$PY" -m jupyter nbconvert --to notebook --execute --inplace batch_summary.ipynb )
echo "Done — see $HERE/{master_metrics.csv, master_map_relocated.png, batch_manifest.csv, batch_summary.ipynb}"
