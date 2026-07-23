#!/bin/bash
# Full 2016 year: PhaseNet+ (GPU-fixed) -> PyOcto (kim2011, 4/2/2 daily), ALL available stations.
# 2016 = KS/KG all year + the GJ temporary aftershock array Sep-Dec (deployed on the 09-12 M5.8 mainshock day;
# mixed 100/200 Hz + occasional 1000 Hz -> decimated on the fly). No NS (didn't exist yet).
# min-coverage = 0 => NO coverage filter: EVERY station with any local data is used (all gappy/partial stations
# included; station set = in-region + operating-epoch + HH/EL/HG channel + has data). No ad-hoc cutoff.
set -u
cd /home/msseo/works/02.Ulsan_Fault_detection/Gyeongju_catalog/detection_test
export CUDA_VISIBLE_DEVICES=0
L=logs; MINCOV=0; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
say "2016 FULL YEAR START: PhaseNet+ -> PyOcto, min-coverage $MINCOV"
Y0=$(date +%s)
for MM in 01 02 03 04 05 06 07 08 09 10 11 12; do
  M=2016-$MM; M0=$(date +%s)
  python3 lib/build_stations.py --month $M > $L/2016_${MM}_stations.log 2>&1
  nst=$(python3 -c "import pandas as pd; S=pd.read_csv('cache/stations_2016_${MM}.csv'); u=S[(S.coverage>0)&(S.coverage>=$MINCOV)]; print(f'{len(u)} ({dict(u.net.value_counts())})')" 2>/dev/null)
  say "=== $M : $nst stations (cov>=$MINCOV) ==="
  # detection (eqnet; per-day raw csvs; final parquet write fails w/o pyarrow -> we assemble in base env)
  rm -rf picks/pnplus_raw_2016_${MM}; mkdir -p picks/pnplus_raw_2016_${MM}
  conda run -n eqnet python3 lib/run_pnplus_month.py --month $M --min-coverage $MINCOV > $L/2016_${MM}_pnplus.log 2>&1 || true
  # assemble picks parquet (base env)
  python3 - "$MM" > $L/2016_${MM}_assemble.log 2>&1 <<'PY'
import sys,glob,pandas as pd
mm=sys.argv[1]; rows=[]
for f in sorted(glob.glob(f"picks/pnplus_raw_2016_{mm}/picks_2016.*.csv")):
    r=pd.read_csv(f)
    for _,p in r.iterrows():
        pa=str(p["station_id"]).strip().strip(",").split(".")
        rows.append(dict(net=pa[0],sta=pa[1],phase=p["phase_type"],time=str(p["phase_time"]),prob=float(p["phase_score"])))
d=pd.DataFrame(rows,columns=["net","sta","phase","time","prob"]); d["picker"]="phasenet_plus"
d.to_parquet(f"picks/picks_phasenet_plus_2016_{mm}.parquet",index=False); print(len(d),"picks")
PY
  npk=$(cat $L/2016_${MM}_assemble.log)
  # association (PyOcto 4/2/2, daily-chunked)
  python3 -u lib/associate_daily.py --picker phasenet_plus --month $M --min-coverage $MINCOV > $L/2016_${MM}_assoc.log 2>&1 || true
  nev=$(tail -1 $L/2016_${MM}_assoc.log 2>/dev/null | grep -oE '[0-9]+ events' | head -1)
  say "    detection $npk | association $nev | month wall $(( ($(date +%s)-M0)/60 )) min"
done
say "2016 FULL YEAR DONE in $(( ($(date +%s)-Y0)/60 )) min"
