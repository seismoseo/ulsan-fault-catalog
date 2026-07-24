#!/usr/bin/env python
"""GJ-inclusive offline scaffold for a UF whole-box cluster (year-general; --slug/--catalog/--station-table).

Mirrors analysis/relocation/scaffold_offline.py, but the station roster comes from OUR per-year station cache
(KS + KG + GJ — the whole point is the GJ temporary array in 2016) instead of the KS/KG-only station_table,
and networks=("KS","KG","GJ") so GJ survives into the pipeline's station master. The caller (run_picker_reloc.py)
passes the year-correct --slug, --catalog, and --station-table; defaults resolve to the 2016 paths for
back-compat. (Filename kept as scaffold_2016.py; it is year-general.) Run with PYTHONPATH = PocketQuake:pipeline.
"""
import os, shutil, sys, argparse
import pandas as pd
from pocketquake.scaffold import ClusterSpec, register_cluster, write_cluster_module

ROOT = "/home/msseo/works/02.Ulsan_Fault_detection/detection_test/reloc_2016_uf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="uf_2016")
    ap.add_argument("--catalog", default=os.path.join(ROOT, "catalog_kma.csv"))
    ap.add_argument("--station-table", default=os.path.join(ROOT, "station_table", "stations_2016.csv"),
                    help="year-general station table (Network,Code,Latitude,Longitude,Elevation)")
    a = ap.parse_args()
    SLUG = a.slug
    epi = tuple(float(x) for x in "35.7539,129.3804".split(","))
    rb = tuple(float(x) for x in "35.551,35.949,129.200,129.599".split(","))
    spec = ClusterSpec(name=SLUG, region=SLUG,
                       catalog_csv=a.catalog,
                       epicenter=epi, region_bounds=rb,
                       networks=("KS", "KG", "GJ"),          # <-- GJ included
                       wf_backend="stp", loc_backend="hypoinverse", reloc_backend="hypodd")
    src = spec.src_root
    for sub in ("event_catalog", "station_table", "stp_download"):
        os.makedirs(os.path.join(src, sub), exist_ok=True)
    shutil.copyfile(spec.catalog_csv, os.path.join(src, "event_catalog", "event_catalog.csv"))
    # station roster from OUR cache (Network,Code,Latitude,Longitude,Elevation), split per network
    allsta = pd.read_csv(a.station_table)
    cols = ["Network", "Code", "Latitude", "Longitude", "Elevation"]
    for net in spec.networks:
        allsta[allsta.Network == net][cols].reset_index(drop=True).to_csv(
            os.path.join(src, "station_table", f"{net}_station.csv"), index=False)
    mod = write_cluster_module(spec); register_cluster(spec)
    print(f"scaffolded {SLUG} -> {src}")
    print(f"  roster {len(allsta)} (" + " + ".join(f"{int((allsta.Network==n).sum())} {n}" for n in spec.networks)
          + f"); module {mod}")


if __name__ == "__main__":
    main()
