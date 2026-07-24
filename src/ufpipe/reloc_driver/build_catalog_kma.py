#!/usr/bin/env python
"""Build catalog_kma.csv + members.txt + members_event_idx.csv + scaffold_args.txt for the 2016 UF whole-box
relocation, PER PICKER. Only events whose event_idx SAC dir exists (>=1 .sac) become members; member i <->
catalog_kma row i <-> event_sac/<event_idx>. catalog_kma time is KST (utc + 9h). phasenet_plus -> reloc_2016_uf/;
any other picker -> reloc_2016_uf_<picker>/.

  python build_catalog_kma.py [--picker phasenet_plus|original|stead|eqt]
"""
import os, glob, argparse, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import year_paths as YP

HERE = YP.HERE
UF = (129.25, 129.55, 35.60, 35.90)
KST_OFFSET_H = 9.0
EPI = "35.7539,129.3804"; RB = "35.551,35.949,129.200,129.599"


def has_wf(SAC, idx):
    d = os.path.join(SAC, str(int(idx)))
    return os.path.isdir(d) and any(f.endswith(".sac") for f in os.listdir(d))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--picker", default="phasenet_plus")
    YP.add_year_arg(ap)
    a = ap.parse_args()
    ROOT = YP.root_dir(a.year, a.picker); SAC = os.path.join(ROOT, "event_sac")
    EV = pd.read_csv(YP.pyocto_year(a.year, a.picker))
    EV["time"] = pd.to_datetime(EV.time, format="ISO8601", utc=True)
    box = EV[(EV.lon >= UF[0]) & (EV.lon <= UF[1]) & (EV.lat >= UF[2]) & (EV.lat <= UF[3])].copy()
    box = box.sort_values("time").reset_index(drop=True)
    box["has_wf"] = box["idx"].map(lambda i: has_wf(SAC, i))
    mem = box[box.has_wf].reset_index(drop=True)                       # member i <-> catalog row i
    print(f"[{a.picker}] UF box {len(box)} events; {len(mem)} with SAC -> members")

    rows = []
    for _, r in mem.iterrows():
        kst = r["time"] + pd.Timedelta(hours=KST_OFFSET_H)             # PocketQuake catalog convention = KST
        rows.append(dict(Year=kst.year, Month=kst.month, Day=kst.day, Hour=kst.hour,
                         Minute=kst.minute, Second=int(kst.second),
                         Latitude=round(r.lat, 4), Longitude=round(r.lon, 4),
                         Magnitude=0.0, Depth=round(r.depth, 2)))
    cat = pd.DataFrame(rows, columns=["Year", "Month", "Day", "Hour", "Minute", "Second",
                                      "Latitude", "Longitude", "Magnitude", "Depth"])
    cat.to_csv(os.path.join(ROOT, "catalog_kma.csv"), index=False)
    mem[["idx"]].astype(int).to_csv(os.path.join(ROOT, "members.txt"), index=False, header=False)
    me = mem.rename(columns={"idx": "event_idx"})[["event_idx", "time", "lat", "lon", "depth"]].copy()
    me["magnitude"] = 0.0
    me.to_csv(os.path.join(ROOT, "members_event_idx.csv"), index=False)
    with open(os.path.join(ROOT, "scaffold_args.txt"), "w") as f:
        f.write(f"--epicenter {EPI} --region-bounds {RB}\n")
    print(f"  -> catalog_kma.csv ({len(cat)}) + members.txt + members_event_idx.csv (in {ROOT})")


if __name__ == "__main__":
    main()
