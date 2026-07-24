#!/bin/bash
# Auto-resume dt.cc driver for a flaky shared box (OOM-resilient).
# - Relaunches the xcorr->dtcc pipeline with XCORR_RESUME=1 each time it dies (resume skips the
#   pairs whose per-pair dt.cc files already exist, so a kill costs <= one chunk).
# - Waits for memory headroom (free > MIN_FREE_GB) before each launch, so it never starts into a
#   memory-starved box (avoids immediate collateral OOM).
# - Single process (--cores 1): smallest footprint, no redundant pre-interp re-scan, OOM-safest.
# - Stops when 02.dt.cc/hypoDD.reloc exists (dtcc done) or after MAX_TRIES / 3 no-progress rounds.
set -u
source /home/msseo/miniforge3/etc/profile.d/conda.sh; conda activate pq-gpu
export PYTHONPATH=/home/msseo/works/15.PocketQuake:/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation:/home/msseo/works/02.Ulsan_Fault_detection/src/uflib
export XCORR_RESUME=1 XCORR_TRACE_CACHE=2000 XCORR_PAIR_CHUNK=50000
PIPE=/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation
RD=$PIPE/pipeline/runs/uf_subregion_reuse
RELOC=$RD/2.HypoDD/02.dt.cc/hypoDD.reloc
PDIR=$RD/2.HypoDD/02.dt.cc/dt.cc_P
MIN_FREE_GB=40; MAX_TRIES=100
prev=-1; stuck=0
cd $PIPE
for t in $(seq 1 $MAX_TRIES); do
  if [ -s "$RELOC" ]; then echo "$(date '+%m-%d %H:%M') DONE: hypoDD.reloc exists ($(wc -l < "$RELOC") events)"; break; fi
  done=$(ls "$PDIR" 2>/dev/null | wc -l)
  if [ "$done" -le "$prev" ] && [ "$t" -gt 1 ]; then stuck=$((stuck+1)); else stuck=0; fi
  if [ "$stuck" -ge 3 ]; then echo "$(date '+%m-%d %H:%M') STOP: no progress 3 rounds (done=$done) — needs attention"; break; fi
  prev=$done
  # wait for memory headroom before launching
  waited=0
  while [ "$(free -g | awk 'NR==2{print $7}')" -lt "$MIN_FREE_GB" ]; do
    [ "$waited" -eq 0 ] && echo "$(date '+%m-%d %H:%M') waiting for memory headroom (free<${MIN_FREE_GB}GB)..."
    waited=1; sleep 120
  done
  echo "$(date '+%m-%d %H:%M') try $t: $done pairs done; launching (resume, --cores 1)"
  python -m pipeline.cli.run_pipeline --cluster uf_subregion_reuse \
      --stage-from xcorr --through dtcc --arc-velmodel kim2011 --cores 1
  echo "$(date '+%m-%d %H:%M') try $t exited (code $?)"
  sleep 30
done
echo "$(date '+%m-%d %H:%M') auto-resume wrapper finished"
