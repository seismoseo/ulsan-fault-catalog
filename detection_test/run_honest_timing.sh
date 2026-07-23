#!/bin/bash
# HONEST end-to-end runtime of the SETTLED config on one dense month (2021-09), run clean (no contention),
# fresh (caches cleared), fully timed:  PhaseNet+ (predecimated, 3 day-shards) -> assemble -> PyOcto (4/2/2 daily).
# Purpose is the wall-clock number, not the catalog. Reports per-stage minutes so a 1-year figure = x12.
set -u
cd /home/msseo/works/02.Ulsan_Fault_detection/Gyeongju_catalog/detection_test
export CUDA_VISIBLE_DEVICES=0
L=logs; mkdir -p $L
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
T0=$(date +%s)

# --- PhaseNet+ detection (fresh: clear the raw-pick cache so nothing is skipped) ---
rm -rf picks/pnplus_raw_2021_09; mkdir -p picks/pnplus_raw_2021_09
say "PN+ DETECTION START (3 shards, predecimated, 241 stations x 30 days, clean)"
P0=$(date +%s)
conda run -n eqnet python3 lib/run_pnplus_month.py --month 2021-09 --predecimated --doy-start 244 --doy-end 253 > $L/honest_pnplus_s1.log 2>&1 &
conda run -n eqnet python3 lib/run_pnplus_month.py --month 2021-09 --predecimated --doy-start 254 --doy-end 263 > $L/honest_pnplus_s2.log 2>&1 &
conda run -n eqnet python3 lib/run_pnplus_month.py --month 2021-09 --predecimated --doy-start 264 --doy-end 273 > $L/honest_pnplus_s3.log 2>&1 &
wait
PN_PICK=$(date +%s)
say "PN+ shards done in $(( (PN_PICK-P0)/60 )) min; assembling parquet (base env)"
# assemble canonical parquet from the 3 shards' raw csvs (base env has pyarrow; eqnet does not)
python3 - <<'PY' > $L/honest_pnplus_assemble.log 2>&1
import glob, pandas as pd
rows=[]
for f in sorted(glob.glob("picks/pnplus_raw_2021_09/picks_2021.*.csv")):
    r=pd.read_csv(f)
    for _,p in r.iterrows():
        pa=str(p["station_id"]).strip().strip(",").split(".")
        rows.append(dict(net=pa[0],sta=pa[1],phase=p["phase_type"],time=str(p["phase_time"]),prob=float(p["phase_score"])))
d=pd.DataFrame(rows); d["picker"]="phasenet_plus"; d.to_parquet("picks/picks_phasenet_plus_2021_09.parquet",index=False)
print(len(d),"picks",d.sta.nunique(),"stations")
PY
PN_END=$(date +%s)
say "PN+ DETECTION+ASSEMBLE TOTAL: $(( (PN_END-P0)/60 )) min ($(cat $L/honest_pnplus_assemble.log))"

# --- PyOcto association (4/2/2 fixed gate, daily-chunked) ---
say "PYOCTO ASSOCIATION START (4/2/2, daily-chunked)"
A0=$(date +%s)
python3 -u lib/associate_daily.py --picker phasenet_plus --month 2021-09 > $L/honest_assoc_pnplus.log 2>&1
A_END=$(date +%s)
say "PYOCTO ASSOCIATION: $(( (A_END-A0)/60 )) min ($(tail -1 $L/honest_assoc_pnplus.log))"

# --- summary ---
say "================ HONEST RUNTIME (one dense month, 2021-09, clean, 1 GPU) ================"
say "  PhaseNet+ detection+assemble : $(( (PN_END-P0)/60 )) min"
say "  PyOcto association (4/2/2)    : $(( (A_END-A0)/60 )) min"
say "  TOTAL PN+ -> PyOcto          : $(( (A_END-T0)/60 )) min  ->  ~1 year = x12"
say "HONEST_RUN_DONE"
