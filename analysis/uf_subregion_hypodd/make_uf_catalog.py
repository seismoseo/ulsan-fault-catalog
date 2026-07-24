#!/usr/bin/env python
"""Build the PocketQuake KMA-format catalog + members for the WHOLE Ulsan-Fault subregion as a SINGLE
cluster — every UF-box event in the clean Heo catalog that has an event_idx-keyed waveform directory.

CLEAN-CUT KEYING (no timestamp matching): the clean catalog carries the frozen `event_idx` (master row
id); the waveform store `event_waveforms_ufidx/<event_idx>/` is keyed by the same id (see
build_uf_event_idx.py). `members.txt` lists `event_idx` and `catalog_kma.csv` rows are written in the
identical order, so member i <-> catalog row i <-> waveform dir <event_idx>. The old strftime
second-truncated `evid` (which silently dropped events whose origin re-solved across a second boundary)
is GONE.

Usage:  python make_uf_catalog.py
Writes uf_subregion/{catalog_kma.csv, members.txt, scaffold_args.txt}.
"""
import os
import pandas as pd

# Restructured 2026-07: KS_KG/local_magnitudes -> analysis/local_magnitudes; KS_KG/HypoInv -> data/hypoinv.
REPO  = "/home/msseo/works/02.Ulsan_Fault_detection"
CLEAN = (f"{REPO}/analysis/local_magnitudes/"
         "catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo_clean.csv")
STORE = f"{REPO}/data/hypoinv/event_waveforms_ufidx"  # event_idx-keyed waveform store
UF_BOX = (129.25, 129.55, 35.60, 35.90)              # lon0, lon1, lat0, lat1
KST_OFFSET_H = 9.0
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "uf_subregion")


def has_waveforms(idx):
    d = os.path.join(STORE, str(idx))
    return os.path.isdir(d) and any(f.endswith(".sac") for f in os.listdir(d))


def main():
    os.makedirs(OUT, exist_ok=True)
    cat = pd.read_csv(CLEAN)
    assert "event_idx" in cat.columns, "clean catalog must carry event_idx (run the freeze/thread step)"
    cat["time"] = pd.to_datetime(cat["time"], utc=True)
    cat["event_idx"] = cat["event_idx"].astype(int)
    in_box = ((cat.lon >= UF_BOX[0]) & (cat.lon <= UF_BOX[1]) &
              (cat.lat >= UF_BOX[2]) & (cat.lat <= UF_BOX[3]))
    box = cat[in_box].drop_duplicates("event_idx").sort_values("event_idx").reset_index(drop=True)
    box["has_wf"] = box["event_idx"].map(has_waveforms)
    members_df = box[box.has_wf].reset_index(drop=True)        # member i <-> catalog row i
    n_box = len(box)
    print(f"UF box: {n_box} clean-catalog events; {len(members_df)} with event_idx waveforms "
          f"({n_box - len(members_df)} without SACs, skipped)")

    rows = []
    for _, r in members_df.iterrows():                         # SAME order as members.txt
        kst = r["time"] + pd.Timedelta(hours=KST_OFFSET_H)     # PocketQuake catalog is KST
        mag = r["magnitude"] if pd.notna(r.get("magnitude")) else 1.0
        rows.append(dict(Year=kst.year, Month=kst.month, Day=kst.day, Hour=kst.hour,
                         Minute=kst.minute, Second=int(kst.second),
                         Latitude=round(float(r["lat"]), 5), Longitude=round(float(r["lon"]), 5),
                         Magnitude=round(float(mag), 2), Depth=round(float(r["depth"]), 2)))
    df = pd.DataFrame(rows, columns=["Year", "Month", "Day", "Hour", "Minute", "Second",
                                     "Latitude", "Longitude", "Magnitude", "Depth"])
    df.to_csv(os.path.join(OUT, "catalog_kma.csv"), index=False)
    with open(os.path.join(OUT, "members.txt"), "w") as fh:
        fh.write("\n".join(str(i) for i in members_df["event_idx"]) + "\n")
    # explicit member->event_idx map (members.txt already IS this, but keep a labelled CSV for downstream
    # reloc parsing: hypoDD id = 200000 + position -> event_idx = members[position])
    members_df[["event_idx", "time", "lat", "lon", "depth", "magnitude"]].to_csv(
        os.path.join(OUT, "members_event_idx.csv"), index=False)

    clat, clon = df["Latitude"].mean(), df["Longitude"].mean()
    pad = 0.05
    rb = (f"{df['Latitude'].min()-pad:.3f},{df['Latitude'].max()+pad:.3f},"
          f"{df['Longitude'].min()-pad:.3f},{df['Longitude'].max()+pad:.3f}")
    with open(os.path.join(OUT, "scaffold_args.txt"), "w") as fh:
        fh.write(f"--epicenter {clat:.4f},{clon:.4f} --region-bounds {rb}\n")
    print(f"  -> {OUT}/catalog_kma.csv  members.txt  members_event_idx.csv  scaffold_args.txt")
    print(f"  --epicenter {clat:.4f},{clon:.4f} --region-bounds {rb}")


if __name__ == "__main__":
    main()
