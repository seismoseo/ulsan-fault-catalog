"""Foundational Heo-2024 ML recompute under the LOCKED scheme (2026-06-25):
  * HYPOCENTRAL distance (sqrt(epicentral^2 + depth^2)) for BOTH the Sheen S-window and the
    Heo Eq.3 attenuation term  (was epicentral -> near-station ML biased low).
  * Sheen (2018) [dist/4, dist/2] S-window; peak/peak SNR gate snr_pp >= 2.0.
  * P reference = PhaseNet+ pick, else THEORETICAL kim2011 P (TauP) -> missed-P stations with a
    clear S still contribute (use_taup_for_missing_p=True; the ml_pipeline default).
Writes event-level + per-station catalogs (Heo only; Sheen kept for later by user request).
Run: ML_WORKERS=8 python run_heo_recompute.py
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_pipeline as mlp
import pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
HYP  = "/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv"
CAT_IN = f"{HYP}/catalog_phasenet_plus_2010_2024_blastclean.csv"
EVENT_ROOTS = (f"{HYP}/event_waveforms_blastclean", f"{HYP}/event_waveforms_ulsanfault")
CAT_OUT = f"{HERE}/catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo.csv"
PER_OUT = f"{HERE}/catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo.csv"
WORKERS = int(os.environ.get("ML_WORKERS", "8"))

# force a clean full recompute (old CSVs are epicentral -> must not be resumed)
for p in (CAT_OUT, PER_OUT):
    if os.path.exists(p):
        os.rename(p, p + ".epicentral_bak")
        print(f"backed up stale {os.path.basename(p)} -> .epicentral_bak")

cat = pd.read_csv(CAT_IN)
print(f"catalog: {len(cat):,} events ({cat.time.iloc[0]} ... {cat.time.iloc[-1]})")
inv = mlp.load_combined_inventory(f"{HERE}/responses/master", f"{HERE}/responses/fetched")
print(f"inventory: {sum(len(n) for n in inv)} stations | workers={WORKERS}")
print("scheme: Heo2024 Z-only, HYPOCENTRAL dist, snr_pp>=2.0, TauP theoretical-P for missing picks")

t0 = time.time()
aug = mlp.export_ml_catalog(
    cat, EVENT_ROOTS, inv,
    attenuation_fn=mlp.ml_heo2024, restrict_to_z=True, require_pick=True,
    workers=WORKERS, skip_existing=True, out_path=CAT_OUT,
    per_station_csv_path=PER_OUT, progress=True,
)
dt = time.time() - t0
ok = aug[aug.mag_status == "ok"]
print(f"\nDONE: {len(aug):,} rows, {aug.magnitude.notna().sum():,} with ML, "
      f"median ML {ok.magnitude.median():.3f}, median n_used {ok.n_used.median():.0f}, wall {dt/60:.1f} min")
print(f"  -> {CAT_OUT}")
print(f"  -> {PER_OUT}")
# Gyeongju benchmarks
for t, kma in {"2016-09-12 11:32:54": 5.8, "2016-09-12 10:44:32": 5.4, "2016-09-19 11:33:58": 4.5}.items():
    hit = ok[ok.time.str.startswith(t)]
    if len(hit):
        r = hit.iloc[0]
        print(f"  bench {t}: KMA {kma} -> {r.magnitude:.2f} (n={int(r.n_used)}, resid {r.magnitude-kma:+.2f})")
