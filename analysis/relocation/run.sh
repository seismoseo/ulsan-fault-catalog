#!/usr/bin/env bash
# Reproducible precise relocation of the largest Ulsan multiplet (family 738, the Nov-2016 cluster)
# via the PocketQuake Fortran HypoInverse + HypoDD pipeline at the kim2011 velocity model.
#
# Two runs are compared, BOTH through the same pipeline on the SAME staged Ulsan waveforms — they
# differ ONLY in the picks:
#   (1) f738_reuse  — reuse Ulsan's EXISTING PhaseNet+ picks  (skip picking; preserved SAC a/t0 + picks CSV)
#   (2) f738_fresh  — PocketQuake RE-PICKS PhaseNet+ on the identical waveforms
#
# No re-invention: every scientific step is PocketQuake's own pipeline; the local scripts only prepare
# the catalog + symlink the waveforms/picks into a PocketQuake stp_sac cluster. Re-runnable end to end.
#
# Prereqs: Fortran binaries hyp1.40 / ph2dt / hypoDD on PATH; a Python env with the eq-cycle pipeline
# deps (obspy, torch+EQNet for PhaseNet+) — override with  PQ_PY=/path/to/python ./run.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PQ=/home/msseo/works/15.PocketQuake
PIPE=$PQ/external/korea-cluster-relocation
PY=${PQ_PY:-/home/msseo/miniforge3/envs/pq-gpu/bin/python}
export PYTHONPATH=$PQ:$PIPE
rp(){ ( cd "$PIPE" && "$PY" -m pipeline.cli.run_pipeline "$@" ); }

# 0. catalog + member list (reproduces family 738 from the cached 5-25 Hz cc + single-linkage CC>=0.9)
"$PY" "$HERE/make_catalog.py"
EPI=$(awk '{print $2}' "$HERE/family738/scaffold_args.txt")
RB=$(awk  '{print $4}' "$HERE/family738/scaffold_args.txt")

# ---------- (1) REUSE existing Ulsan PhaseNet+ picks ----------
"$PY" "$HERE/scaffold_offline.py" f738_reuse --catalog "$HERE/family738/catalog_kma.csv" \
      --epicenter "$EPI" --region-bounds "$RB"
"$PY" "$HERE/stage.py" f738_reuse --reuse-picks                                   # raw SAC symlinks + picks CSV
rp --cluster f738_reuse --stage-from stations --through waveforms                 # stations + gather raw->100km (PRESERVES a/t0); no picking
rp --cluster f738_reuse --stage-from hypoinverse --through dtcc --arc-velmodel kim2011   # locate + dt.ct + xcorr + dt.cc on the reused picks

# ---------- (2) FRESH PhaseNet+ re-pick on the SAME waveforms ----------
"$PY" "$HERE/scaffold_offline.py" f738_fresh --catalog "$HERE/family738/catalog_kma.csv" \
      --epicenter "$EPI" --region-bounds "$RB"
"$PY" "$HERE/stage.py" f738_fresh                                                 # raw SAC symlinks only
rp --cluster f738_fresh --stage-from stations --through dtcc --arc-velmodel kim2011       # gather + RE-PICK + locate + reloc

# ---------- bootstrap 95% relative-location errors (Fortran hypoDD, n=1000; cached) ----------
# n=1000 replicas x 2 clusters = 2000 full HypoDD inversions; parallelise across BOOT_CORES
# (default 48; the pipeline default cfg.num_cores is much smaller, hence override here).
BOOT_CORES=${BOOT_CORES:-48}
for slug in f738_reuse f738_fresh; do
  ( cd "$PIPE" && "$PY" -c "import sys; sys.path.insert(0,'.'); from pipeline import config; from pipeline.core import hypodd; hypodd.bootstrap_relocation(config.load_cluster('$slug'), branch='dtcc', n=1000, seed=0, cores=$BOOT_CORES)" )
done

# ---------- save tidy result tables (reloc + bootstrap errors) into family738/ ----------
"$PY" "$HERE/save_results.py"

# ---------- compare (absolute vs reuse-dt.cc vs fresh-dt.cc; collapse, sections, bootstrap) ----------
"$PY" "$HERE/build_compare_nb.py"
( cd "$HERE" && "$PY" -m jupyter nbconvert --to notebook --execute --inplace compare_relocations.ipynb )

# ---------- dedicated reuse summary (PocketQuake-style §1 locations, no FM) + PyGMT before/after map ----------
"$PY" "$HERE/build_summary_nb.py"
( cd "$HERE" && "$PY" -m jupyter nbconvert --to notebook --execute --inplace summary_reuse.ipynb )
echo "Done — open $HERE/compare_relocations.ipynb and $HERE/summary_reuse.ipynb"
