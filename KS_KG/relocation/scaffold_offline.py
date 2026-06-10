#!/usr/bin/env python
"""Offline scaffold of a PocketQuake **stp_sac** cluster for an Ulsan multiplet — fully reproducible,
no STP network call. Mirrors `pocketquake.scaffold.scaffold_all` but writes the per-network station
tables from the **Ulsan** station_table CSVs (the authoritative roster for these 2016-era events)
instead of fetching the historical roster from STP. Reuses PocketQuake's `write_cluster_module` +
`register_cluster` verbatim (so the generated cluster module + config registration are byte-identical
to a normal scaffold).

Usage (PYTHONPATH must include the PocketQuake root AND the eq-cycle pipeline repo):
  python scaffold_offline.py <slug> --catalog family738/catalog_kma.csv \
        --epicenter LAT,LON --region-bounds A,B,C,D
"""
import argparse
import glob
import os
import shutil

import pandas as pd

from pocketquake.scaffold import ClusterSpec, register_cluster, write_cluster_module

ULSAN_STA = "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/station_table"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--epicenter", required=True, help="LAT,LON")
    ap.add_argument("--region-bounds", required=True, help="lat0,lat1,lon0,lon1")
    a = ap.parse_args()

    lat, lon = (float(x) for x in a.epicenter.split(","))
    rb = tuple(float(x) for x in a.region_bounds.split(","))
    spec = ClusterSpec(name=a.slug, region=a.slug, catalog_csv=os.path.abspath(a.catalog),
                       epicenter=(lat, lon), region_bounds=rb, networks=("KS", "KG"),
                       wf_backend="stp", loc_backend="hypoinverse", reloc_backend="hypodd")
    src = spec.src_root

    # 1. dirs + 2. catalog (same as scaffold(), minus the STP station fetch)
    for sub in ("event_catalog", "station_table", "stp_download"):
        os.makedirs(os.path.join(src, sub), exist_ok=True)
    shutil.copyfile(spec.catalog_csv, os.path.join(src, "event_catalog", "event_catalog.csv"))

    # 3. per-network station tables from the Ulsan roster (union over all years), eq-cycle columns
    frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ULSAN_STA, "stations_*.csv")))]
    allsta = pd.concat(frames, ignore_index=True).drop_duplicates(["Network", "Code"])
    cols = ["Network", "Code", "Latitude", "Longitude", "Elevation"]
    for net in spec.networks:
        sub = allsta[allsta["Network"] == net][cols].reset_index(drop=True)
        sub.to_csv(os.path.join(src, "station_table", f"{net}_station.csv"), index=False)

    # 4. write the cluster module + register (reused verbatim from PocketQuake)
    mod = write_cluster_module(spec)
    register_cluster(spec)
    print(f"scaffolded {a.slug} -> {src}")
    print(f"  station roster: {len(allsta)} ({(allsta.Network=='KS').sum()} KS + "
          f"{(allsta.Network=='KG').sum()} KG); module {mod}")


if __name__ == "__main__":
    main()
