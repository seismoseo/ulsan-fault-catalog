#!/bin/bash
# Confirming full-month run of the SETTLED, OPTIMIZED config, clean + fully timed:
#   PhaseNet+ (GPU-move fix, predecimated, 3 day-shards) -> assemble -> PyOcto (4/2/2, daily-chunked).
# Produces the definitive optimized month timing + a clean 2021-09 PN+ catalog. Fresh caches.
set -u
cd /home/msseo/works/02.Ulsan_Fault_detection/Gyeongju_catalog/detection_test
export CUDA_VISIBLE_DEVICES=0
L=logs; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
say "CONFIRM RUN start (PhaseNet+ GPU-fixed -> PyOcto, 2021-09, clean)"

rm -rf picks/pnplus_raw_2021_09; mkdir -p picks/pnplus_raw_2021_09
rm -f picks/picks_phasenet_plus_2021_09.parquet catalogs/catalog_phasenet_plus_2021_09_pyocto.* catalogs/assign_phasenet_plus_2021_09_pyocto.*

# --- PhaseNet+ detection, 3 day-shards ---
P0=$(date +%s)
say "STAGE 1: PhaseNet+ detection (3 shards, predecimated, GPU-move fix)"
conda run -n eqnet python3 lib/run_pnplus_month.py --month 2021-09 --predecimated --doy-start 244 --doy-end 253 > $L/confirm_s1.log 2>&1 &
conda run -n eqnet python3 lib/run_pnplus_month.py --month 2021-09 --predecimated --doy-start 254 --doy-end 263 > $L/confirm_s2.log 2>&1 &
conda run -n eqnet python3 lib/run_pnplus_month.py --month 2021-09 --predecimated --doy-start 264 --doy-end 273 > $L/confirm_s3.log 2>&1 &
wait
PN_PICK=$(date +%s)
say "STAGE 1 done: PhaseNet+ detection $(( (PN_PICK-P0)/60 )) min ($(ls picks/pnplus_raw_2021_09/|wc -l)/30 days)"

# --- assemble canonical parquet (base env has pyarrow) ---
python3 - <<'PY' > $L/confirm_assemble.log 2>&1
import glob, pandas as pd
rows=[]
for f in sorted(glob.glob("picks/pnplus_raw_2021_09/picks_2021.*.csv")):
    r=pd.read_csv(f)
    for _,p in r.iterrows():
        pa=str(p["station_id"]).strip().strip(",").split(".")
        rows.append(dict(net=pa[0],sta=pa[1],phase=p["phase_type"],time=str(p["phase_time"]),prob=float(p["phase_score"])))
d=pd.DataFrame(rows); d["picker"]="phasenet_plus"; d.to_parquet("picks/picks_phasenet_plus_2021_09.parquet",index=False)
print(f"{len(d)} picks, {d.sta.nunique()} stations")
PY
say "assembled: $(cat $L/confirm_assemble.log)"

# --- PyOcto association, 4/2/2, daily-chunked ---
A0=$(date +%s)
say "STAGE 2: PyOcto association (4/2/2, daily-chunked)"
python3 -u lib/associate_daily.py --picker phasenet_plus --month 2021-09 > $L/confirm_assoc.log 2>&1
A_END=$(date +%s)
say "STAGE 2 done: PyOcto $(( (A_END-A0)/60 )) min ($(tail -1 $L/confirm_assoc.log))"

say "================ CONFIRM TOTALS (2021-09 dense month, clean, 1 GPU) ================"
say "  PhaseNet+ detection (3-shard) : $(( (PN_PICK-P0)/60 )) min"
say "  PyOcto association (4/2/2)     : $(( (A_END-A0)/60 )) min"
say "  TOTAL PN+ -> PyOcto           : $(( (A_END-P0)/60 )) min   (x12 ~= 1 year)"
say "CONFIRM_DONE"
