"""Rebuild the derived Heo catalogs from the freshly recomputed foundational catalogs
(LOCKED scheme: hypocentral dist + snr_pp>=2.0 + TauP theoretical-P). Two stages:

  qc    : location-coherence QC (qc_location_coherence.score_catalog, ML-independent) on the new
          `_with_ml_heo.csv` -> `_with_ml_heo_scored.csv`, `_with_ml_heo_clean.csv`,
          `_with_ml_heo_coherence_audit.csv`. (clean = catalog minus coherence_fail edge-mislocations.)
  homog : after nb09 has rebuilt `catalog_ml_heo_station_homogenised.csv`, build
          `_with_ml_heo_homogenised_clean.csv` = the clean catalog with magnitude <- ml_homogenised
          (station-homogenised ML), joined on event time. (Verified recipe: corr==1.0 with the prior file.)

Run:  python rebuild_derived_heo.py qc      # after run_heo_recompute.py
      python rebuild_derived_heo.py homog   # after executing 09.Station_corrections_ML.ipynb
"""
import os, sys
import pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Restructured 2026-07: qc_location_coherence -> analysis/hypoinv; stead detection/station tables moved.
REPO = "/home/msseo/works/02.Ulsan_Fault_detection"
sys.path.insert(0, f"{REPO}/analysis/hypoinv")
BASE = f"{HERE}/catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo"


def stage_qc():
    import qc_location_coherence as qc
    cat = pd.read_csv(f"{BASE}.csv")
    print(f"scoring {len(cat):,} events for location coherence ...")
    scored = qc.score_catalog(cat, f"{REPO}/outputs/detection_location",
                              f"{REPO}/data/metadata/stations/ks_kg",
                              stations_prefix="stations_")
    qc._report(scored)
    scored.to_csv(f"{BASE}_scored.csv", index=False)
    clean = scored[~scored.coherence_fail].drop(columns=["coherence_fail"])
    clean.to_csv(f"{BASE}_clean.csv", index=False)
    scored[scored.coherence_fail].to_csv(f"{BASE}_coherence_audit.csv", index=False)
    print(f"  scored -> {BASE}_scored.csv  ({len(scored):,})")
    print(f"  clean  -> {BASE}_clean.csv   ({len(clean):,})   [removed {int(scored.coherence_fail.sum())}]")


def stage_homog():
    clean = pd.read_csv(f"{BASE}_clean.csv")
    sh = pd.read_csv(f"{HERE}/catalog_ml_heo_station_homogenised.csv")
    clean["t"] = pd.to_datetime(clean["time"]).astype("int64")
    sh["t"]    = pd.to_datetime(sh["event_time"]).astype("int64")
    m = clean.merge(sh[["t", "ml_homogenised"]], on="t", how="left").dropna(subset=["ml_homogenised"])
    m["magnitude"] = m["ml_homogenised"]                    # station-homogenised ML replaces raw
    m = m.drop(columns=["t", "ml_homogenised"])
    out = f"{BASE}_homogenised_clean.csv"
    m.to_csv(out, index=False)
    print(f"  homogenised_clean -> {out}  ({len(m):,})  [magnitude = station-homogenised ML]")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "qc"
    {"qc": stage_qc, "homog": stage_homog}[stage]()
