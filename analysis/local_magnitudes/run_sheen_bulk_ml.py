"""Sheen 2018 bulk ML pass — parallel branch alongside the Heo v3/v4 results.

Uses the same catalog + event roots as `02.Compute_ML_all_events.ipynb`, but swaps
`attenuation_fn=ml_sheen2018` (broader-network Korea formula, 100-km reference,
3-component) and lets `ml_pipeline._auto_pre_filt` resolve the broader bandpass
(0.5–30 Hz, vs Heo's 2–20 Hz).

Outputs (in this directory):
  catalog_phasenet_plus_2010_2024_blastclean_with_ml_sheen.csv     — event-level
  catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_sheen.csv — per-station

Run: nohup python run_sheen_bulk_ml.py > run_sheen_bulk_ml.log 2>&1 &
Expected wall: 30–40 min at workers=4 (same compute path as the Heo run).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_pipeline as mlp
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
HYP = "/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv"

ATT_FN = mlp.ml_sheen2018
RESTRICT_Z = False  # Sheen uses all 3 components per the paper's coefficients

CAT_IN = f"{HYP}/catalog_phasenet_plus_2010_2024_blastclean.csv"
CAT_OUT = f"{HERE}/catalog_phasenet_plus_2010_2024_blastclean_with_ml_sheen.csv"
PER_ST_OUT = f"{HERE}/catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_sheen.csv"
EVENT_ROOTS = (
    f"{HYP}/event_waveforms_blastclean",
    f"{HYP}/event_waveforms_ulsanfault",
)
MASTER_XML = "/home/msseo/works/02.Ulsan_Fault_detection/data/metadata/responses/master"
FETCHED_XML = "/home/msseo/works/02.Ulsan_Fault_detection/data/metadata/responses/fetched"

inv = mlp.load_combined_inventory(MASTER_XML, FETCHED_XML)
cat = pd.read_csv(CAT_IN)
print(f"[sheen] catalog: {len(cat):,} events", flush=True)
print(f"[sheen] attenuation: {ATT_FN.__name__}, restrict_to_z={RESTRICT_Z}", flush=True)
print(f"[sheen] pre_filt auto: {mlp._auto_pre_filt(ATT_FN)}", flush=True)

t0 = time.time()
aug = mlp.export_ml_catalog(
    cat, EVENT_ROOTS, inv,
    attenuation_fn=ATT_FN, restrict_to_z=RESTRICT_Z,
    workers=4, skip_existing=False,
    out_path=CAT_OUT,
    per_station_csv_path=PER_ST_OUT,
    progress=True,
)
dt = time.time() - t0
print(f"\n[sheen] done in {dt / 60:.1f} min → {CAT_OUT}", flush=True)
print(f"[sheen] per-station CSV → {PER_ST_OUT}", flush=True)

ok = aug[aug.mag_status == "ok"]
print(f"[sheen] events with ok ML: {len(ok):,} / {len(aug):,}", flush=True)
print(f"[sheen] ML range: {ok.magnitude.min():.2f} … {ok.magnitude.max():.2f}", flush=True)
print(f"[sheen] median ML: {ok.magnitude.median():.3f}", flush=True)
