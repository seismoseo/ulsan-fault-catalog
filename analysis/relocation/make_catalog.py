#!/usr/bin/env python
"""Build a PocketQuake KMA-format catalog for ONE waveform-similarity multiplet (default: the largest
family, the Nov-2016 cluster), so it can be relocated with the PocketQuake HypoInverse+HypoDD pipeline.

Reproducible end to end:
  - reproduce the 5-25 Hz single-linkage (CC>=0.9) families at KG.HDB HHZ (the cached cc matrix),
  - take the chosen family's member event-ids,
  - join the blast-clean catalog (+ ML) for each event's UTC origin + lat/lon/depth/magnitude,
  - convert UTC -> KST (PocketQuake catalogs are KST; it subtracts 9 h to recover the UTC origin, whose
    YYYYMMDDHHMMSS equals the Ulsan event-dir name),
  - write  <outdir>/catalog_kma.csv  (Year,Month,Day,Hour,Minute,Second,Latitude,Longitude,Magnitude,Depth),
           <outdir>/members.txt      (the event-id list, one per line),
           <outdir>/scaffold_args.txt (the --epicenter / --region-bounds for pocketquake.sh).

Usage:  python make_catalog.py [--family largest|<id>] [--outdir family738]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HYPO = "/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv"
ML_CSV = ("/home/msseo/works/02.Ulsan_Fault_detection/analysis/local_magnitudes/"
          "catalog_phasenet_plus_2010_2024_blastclean_with_ml.csv")
KST_OFFSET_H = 9.0
STATION, COMP, WIN, PRIMARY, MAXLAG = "KG.HDB", "HHZ", (-0.5, 7.5), (5, 25), 0.2
from uflib import uf_waveform_similarity as wf   # noqa: E402


def family_members(family="largest", band=(5, 25)):
    """Reproduce the single-linkage CC>=0.9 clustering at `band` and return (family_id, [event_id...])."""
    res = wf.make_bands(wf.list_events(station=STATION, comp=COMP), station=STATION, comp=COMP,
                        bands=[band, (1, 10), (2, 8), (4, 12)], win=WIN, cache_dir=wf.CACHE_DIR,
                        verbose=False)
    kept = res["kept"]
    meta = wf.load_event_meta(kept)
    tag = (f"{STATION}_{COMP}_w{WIN[0]}_{WIN[1]}_b{band[0]}-{band[1]}_lag{MAXLAG}_n{len(kept)}"
           .replace(".", "p"))
    cc = np.load(os.path.join(wf.CACHE_DIR, f"cc_{tag}.npy"))
    labels, _, _ = wf.ward_clusters(cc, threshold=1 - 0.9, method="single")
    rep = wf.repeater_table(meta, labels, cc, min_size=3)
    fam = int(rep.iloc[0]["cluster"]) if family == "largest" else int(family)
    m = meta.assign(fam=labels)
    members = sorted(m[m["fam"] == fam]["event"].tolist())
    return fam, members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="largest", help="'largest' (default) or a numeric family id")
    ap.add_argument("--outdir", default="family738", help="output subdir (created under this script's dir)")
    ap.add_argument("--band", default="5-25", help="clustering band, e.g. 5-25 (default) or 5-15")
    args = ap.parse_args()

    band = tuple(int(x) for x in args.band.split("-"))
    fam, members = family_members(args.family, band)
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # catalog with ML (UTC time, lat, lon, depth, magnitude)
    cat = pd.read_csv(ML_CSV)
    cat["time"] = pd.to_datetime(cat["time"], utc=True)
    cat["evid"] = cat["time"].dt.strftime("%Y%m%d%H%M%S")
    cat = cat.drop_duplicates("evid", keep="first").set_index("evid")
    miss = [e for e in members if e not in cat.index]
    if miss:
        print(f"WARNING: {len(miss)} members not in catalog: {miss[:5]}", file=sys.stderr)
    members = [e for e in members if e in cat.index]

    rows = []
    for e in members:
        r = cat.loc[e]
        kst = r["time"] + pd.Timedelta(hours=KST_OFFSET_H)          # PocketQuake catalog is KST
        mag = r["magnitude"] if "magnitude" in cat.columns and pd.notna(r.get("magnitude")) else 1.0
        rows.append(dict(Year=kst.year, Month=kst.month, Day=kst.day, Hour=kst.hour, Minute=kst.minute,
                         Second=int(kst.second), Latitude=round(float(r["lat"]), 5),
                         Longitude=round(float(r["lon"]), 5), Magnitude=round(float(mag), 2),
                         Depth=round(float(r["depth"]), 2)))
    df = pd.DataFrame(rows, columns=["Year", "Month", "Day", "Hour", "Minute", "Second",
                                     "Latitude", "Longitude", "Magnitude", "Depth"])
    df.to_csv(os.path.join(outdir, "catalog_kma.csv"), index=False)
    with open(os.path.join(outdir, "members.txt"), "w") as fh:
        fh.write("\n".join(members) + "\n")

    # epicenter (centroid) + region bounds (+0.05 deg pad) for pocketquake.sh
    clat, clon = df["Latitude"].mean(), df["Longitude"].mean()
    pad = 0.05
    bounds = (df["Latitude"].min() - pad, df["Latitude"].max() + pad,
              df["Longitude"].min() - pad, df["Longitude"].max() + pad)
    epi = f"{clat:.4f},{clon:.4f}"
    rb = f"{bounds[0]:.3f},{bounds[1]:.3f},{bounds[2]:.3f},{bounds[3]:.3f}"
    with open(os.path.join(outdir, "scaffold_args.txt"), "w") as fh:
        fh.write(f"--epicenter {epi} --region-bounds {rb}\n")

    print(f"family {fam}: {len(members)} events, {members[0]} .. {members[-1]}")
    print(f"  catalog_kma.csv  + members.txt  -> {outdir}")
    print(f"  --epicenter {epi}  --region-bounds {rb}")


if __name__ == "__main__":
    main()
