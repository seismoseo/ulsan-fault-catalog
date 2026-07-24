"""Canonical bulk-ML recompute: local magnitudes STRICTLY per the published scales,
with the require_pick detectability gate (the Buan fix), and NO station corrections.

  * Heo et al. 2024  — Southeastern-Korea (GHBSN) scale, calibrated on the VERTICAL
                       component, hypocentral R, 17 km reference  -> Z-only.
  * Sheen et al. 2018 — South-Korea scale, all THREE components (geom-mean horizontals
                       + Z), epicentral R, 100 km reference        -> 3-component.

Both: require_pick=True (only stations whose phase was detected contribute, so the pre-P
noise window is real and far unpicked stations cannot leak in), snr_threshold=3.0, S=0.

Outputs (event-level + per-station) into THIS directory, read directly by the summary
notebooks:
  catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo.csv     (+ _per_station_ml_heo.csv)
  catalog_phasenet_plus_2010_2024_blastclean_with_ml_sheen.csv   (+ _per_station_ml_sheen.csv)
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_pipeline as mlp
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
HYP  = "/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv"

CAT_IN = f"{HYP}/catalog_phasenet_plus_2010_2024_blastclean.csv"
EVENT_ROOTS = (
    f"{HYP}/event_waveforms_blastclean",
    f"{HYP}/event_waveforms_ulsanfault",
)
MASTER_XML  = "/home/msseo/works/02.Ulsan_Fault_detection/data/metadata/responses/master"
FETCHED_XML = "/home/msseo/works/02.Ulsan_Fault_detection/data/metadata/responses/fetched"

cat = pd.read_csv(CAT_IN)
print(f"catalog: {len(cat):,} events ({cat.time.iloc[0]} ... {cat.time.iloc[-1]})")
inv = mlp.load_combined_inventory(MASTER_XML, FETCHED_XML)
print(f"inventory: {sum(len(n) for n in inv)} stations across {len(inv)} network(s)")

WORKERS = int(os.environ.get("ML_WORKERS", "6"))

# attenuation, Z-only?, output suffix — STRICT per each paper's calibration
for att_fn, restrict_z, suffix in [
    (mlp.ml_heo2024,   True,  "_heo"),     # Heo 2024 — VERTICAL only (paper calibration)
    (mlp.ml_sheen2018, False, "_sheen"),   # Sheen 2018 — all 3 components (paper coefficients)
]:
    cat_out = f"{HERE}/catalog_phasenet_plus_2010_2024_blastclean_with_ml{suffix}.csv"
    per_out = f"{HERE}/catalog_phasenet_plus_2010_2024_blastclean_per_station_ml{suffix}.csv"
    t0 = time.time()
    print(f"\n=== bulk-ML  {att_fn.__name__}  z_only={restrict_z}  require_pick=True  workers={WORKERS} ===")
    aug = mlp.export_ml_catalog(
        cat, EVENT_ROOTS, inv,
        attenuation_fn=att_fn, restrict_to_z=restrict_z, require_pick=True,
        workers=WORKERS, skip_existing=True, out_path=cat_out,
        per_station_csv_path=per_out, progress=True,
    )
    dt = time.time() - t0
    print(f"  {att_fn.__name__}: {len(aug):,} rows, {aug.magnitude.notna().sum():,} with ML, "
          f"median ML {aug.magnitude.median():.3f}, wall {dt/60:.1f} min")
    print(f"  -> {cat_out}")
