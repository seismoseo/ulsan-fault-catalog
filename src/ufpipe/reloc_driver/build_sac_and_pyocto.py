#!/usr/bin/env python
# DEPRECATED (2026-07): superseded by src/ufpipe/reloc_inputs.py, which builds this store from ufpipe's OWN
# per-year detection+association (no detection_test/lib per-month feeder). ufpipe.relocate calls that and runs
# run_picker_reloc.py --skip-build. This script is used only when --skip-build is absent. See lib/DEPRECATED.md.
"""Build the 2016 UF-subregion event-idx-keyed SAC store (whole-box relocation input), PER PICKER.

Reuses data/hypoinv/event_sac_export.py unchanged (multi-archive handled via the symlinked merged_archive),
feeding <picker> -> PyOcto picks through the pyocto_root path. Steps:
  1. stations_2016.csv  (Network,Code,Latitude,Longitude,Elevation) from our station cache (picker-independent).
  2. year PyOcto files with GLOBAL event_idx: pyocto_kim2011_2016.csv (events) +
     pyocto_assignment_kim2011_2016.csv (the <picker> assign parquets, concatenated, offset).
  3. UF-box catalog_df keyed event_id = str(global event_idx)  (=> SAC dirs named by event_idx).
  4. export_catalog -> event_sac/<event_idx>/<event_idx>.<NET>.<STA>.<CHAN>.sac + picks in SAC a/t0.

  python build_sac_and_pyocto.py [--picker phasenet_plus|original|stead|eqt]
The picker is the ONLY variable: phasenet_plus -> reloc_2016_uf/ (the original PN+ run, unchanged);
any other picker -> reloc_2016_uf_<picker>/, sharing the picker-independent station table + merged_archive.
"""
import os, sys, glob, argparse
import pandas as pd, numpy as np
from obspy import UTCDateTime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import year_paths as YP
from uflib import event_sac_export as ese

HERE = YP.HERE
UF = (129.25, 129.55, 35.60, 35.90)
VELMODEL = YP.VELMODEL
# SAC store keeps each station's NATIVE sampling rate (None => no 100 Hz resample/guard in event_sac_export);
# durable full-band data product. dt.cc xcorr interpolates to a common interp_hz at correlation time.
SAC_TARGET_HZ = None


def build_stations(ROOT, year, picker):
    S = []
    for mm in range(1, 13):
        f = YP.station_cache(mm, year)
        if not os.path.exists(f):
            print(f"    (skip: no station cache for {year}-{mm:02d})"); continue
        c = pd.read_csv(f); S.append(c[c.coverage > 0])
    assert S, f"no station caches found for {year} (run lib/build_stations.py --month {year}-MM)"
    S = pd.concat(S).drop_duplicates("sta")
    out = pd.DataFrame({"Network": S.net, "Code": S.sta, "Latitude": S.lat,
                        "Longitude": S.lon, "Elevation": S.elev})
    dst = YP.station_table(year, picker)
    out.to_csv(dst, index=False)
    print(f"  {os.path.basename(dst)}: {len(out)} stations")


def build_pyocto(ROOT, picker, year):
    ev_parts, asg_parts, off = [], [], 0
    for mm in range(1, 13):
        cf, af = YP.catalog_pyocto(picker, mm, year), YP.assign_pyocto(picker, mm, year)
        if not (os.path.exists(cf) and os.path.exists(af)):
            print(f"    (skip: no association for {picker} {year}-{mm:02d})"); continue
        cat = pd.read_csv(cf)
        asg = pd.read_parquet(af)
        assert "event_idx" in cat.columns, f"{mm}: catalog missing event_idx (rerun associate_daily)"
        cat = cat.copy(); asg = asg.copy()
        cat["idx"] = cat["event_idx"] + off
        asg["event_idx"] = asg["event_idx"] + off
        off = int(max(cat["idx"].max(), asg["event_idx"].max())) + 1
        ev_parts.append(cat[["idx", "time", "lat", "lon", "depth"]])
        asg_parts.append(asg[["event_idx", "pick_idx", "residual", "station", "phase", "time"]])
    assert ev_parts, f"no monthly associations found for {picker} {year} (run lib/associate_daily.py)"
    EV = pd.concat(ev_parts, ignore_index=True)
    ASG = pd.concat(asg_parts, ignore_index=True)
    EV.to_csv(YP.pyocto_year(year, picker), index=False)
    ASG.to_csv(YP.pyocto_assignment_year(year, picker), index=False)
    print(f"  pyocto events {len(EV)} + assignment {len(ASG)} (global event_idx 0..{off-1})")
    return EV


def build_uf_catalog(EV):
    EV = EV.copy(); EV["time"] = pd.to_datetime(EV.time, format="ISO8601", utc=True)
    box = EV[(EV.lon >= UF[0]) & (EV.lon <= UF[1]) & (EV.lat >= UF[2]) & (EV.lat <= UF[3])].reset_index(drop=True)
    box["event_id"] = box["idx"].astype(int).astype(str)                 # SAC dir = str(event_idx)
    box["origin_utc"] = box["time"].apply(lambda t: UTCDateTime(t.to_pydatetime()))
    print(f"  UF-box catalog: {len(box)} events")
    return box[["event_id", "origin_utc", "time", "lat", "lon", "depth", "idx"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picker", default="phasenet_plus")
    YP.add_year_arg(ap)
    a = ap.parse_args()
    ROOT = YP.root_dir(a.year, a.picker)
    os.makedirs(os.path.join(ROOT, "station_table"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "pyocto"), exist_ok=True)
    # share the picker-independent merged_archive (symlink store of GJ/KG/KS continuous archives)
    ma = os.path.join(ROOT, "merged_archive")
    if not os.path.lexists(ma):
        os.symlink(os.path.join(YP.shared_dir(a.year), "merged_archive"), ma)
    print(f"=== {a.year} picker {a.picker} -> {ROOT} ===")
    print("[1] stations"); build_stations(ROOT, a.year, a.picker)
    print("[2] pyocto year files"); EV = build_pyocto(ROOT, a.picker, a.year)
    print("[3] UF catalog"); cat = build_uf_catalog(EV)
    print("[4] SAC extraction (merged archive, PyOcto picks -> a/t0) ...")
    summ = ese.export_catalog(
        cat, picks_root="", continuous_root=ma,
        station_table_root=os.path.join(ROOT, "station_table"),
        out_root=os.path.join(ROOT, "event_sac"),
        pyocto_root=os.path.join(ROOT, "pyocto"), pyocto_velmodel=VELMODEL,
        target_hz=SAC_TARGET_HZ, skip_existing=True, progress=True)
    summ.to_csv(f"{ROOT}/event_sac_export_summary.csv", index=False)
    ok = (summ.status == "ok").sum() if "status" in summ else len(summ)
    print(f"DONE: {ok}/{len(cat)} events with SAC -> {ROOT}/event_sac/")


if __name__ == "__main__":
    main()
