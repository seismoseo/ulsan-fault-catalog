#!/bin/bash
# Unattended 2021-09 dense benchmark (v4: morning-prioritised).
# PhaseNet+ is ~5x slower per station-day than original and CPU-bound with poor scaling -> it cannot finish by
# morning. So: run PhaseNet-original at NORMAL priority (finishes ~5 h, associated by morning) and PhaseNet+ as
# 3 day-shards at nice 19 (progress on spare cores WITHOUT starving original). Original is associated the moment
# it finishes; PhaseNet+ association + the native secondary follow whenever they complete (afternoon).
set -u
cd /home/msseo/works/02.Ulsan_Fault_detection/Gyeongju_catalog/detection_test
export CUDA_VISIBLE_DEVICES=0
L=logs; mkdir -p $L
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
assoc(){ # picker suffix
  local pk=$1 sfx=${2:-}
  python3 lib/run_association.py --picker $pk ${sfx:+--suffix $sfx} --assoc pyocto --month 2021-09 >> $L/assoc_2021_09.log 2>&1
  timeout 10800 python3 lib/run_association.py --picker $pk ${sfx:+--suffix $sfx} --assoc harpa --month 2021-09 >> $L/assoc_2021_09.log 2>&1 \
      || say "  HARPA $pk$sfx timed out/failed (PyOcto stands)"
}

say "STAGE 0: pre-decimate NS -> 100 Hz (skips existing)"
python3 lib/predecimate_ns.py --month 2021-09 --workers 40 > $L/predecimate_2021_09.log 2>&1
say "STAGE 0 done"

# 1 · original (normal prio) + PhaseNet+ x3 shards (nice 19), all concurrent
say "STAGE 1: original (normal) + PhaseNet+ x3 shards (nice 19), predecimated"
rm -rf picks/pnplus_raw_2021_09; mkdir -p picks/pnplus_raw_2021_09
python3 lib/run_seisbench_picker.py --model original --month 2021-09 --predecimated > $L/pick_original_2021_09.log 2>&1 &
PO=$!
nice -n 19 conda run -n eqnet python3 lib/run_pnplus_month.py --month 2021-09 --predecimated --doy-start 244 --doy-end 253 > $L/pnplus_s1.log 2>&1 &
nice -n 19 conda run -n eqnet python3 lib/run_pnplus_month.py --month 2021-09 --predecimated --doy-start 254 --doy-end 263 > $L/pnplus_s2.log 2>&1 &
nice -n 19 conda run -n eqnet python3 lib/run_pnplus_month.py --month 2021-09 --predecimated --doy-start 264 --doy-end 273 > $L/pnplus_s3.log 2>&1 &

# 2a · associate original AS SOON AS it finishes (do not wait for PhaseNet+) -> MORNING deliverable
wait $PO
say "STAGE 2a: original picking done ($(grep -c 'picks   \[' $L/pick_original_2021_09.log)/241 sta); associating"
assoc original
say "STAGE 2a done. ORIGINAL PRIMARY COMPLETE."

# 2b · PhaseNet+: wait for the 3 shards, assemble canonical parquet, associate (afternoon)
wait
say "STAGE 2b: PhaseNet+ shards done ($(ls picks/pnplus_raw_2021_09/ | wc -l)/30 day-csvs); assembling + associating"
conda run -n eqnet python3 lib/run_pnplus_month.py --month 2021-09 --predecimated > $L/pick_pnplus_2021_09.log 2>&1
assoc phasenet_plus
for s in 1 2; do
  timeout 10800 python3 lib/run_association.py --picker phasenet_plus --assoc harpa --seed $s --month 2021-09 >> $L/assoc_2021_09.log 2>&1 || say "  HARPA seed$s timed out"
done
say "STAGE 2b done. PHASENET_PLUS COMPLETE."

# 3 · native-200 secondary (decimate-vs-native)
say "STAGE 3: native-200 original"
python3 lib/run_seisbench_picker.py --model original --month 2021-09 --native > $L/pick_original_2021_09_native.log 2>&1
assoc original _native
say "STAGE 3 done"
say "PIPELINE DONE"
