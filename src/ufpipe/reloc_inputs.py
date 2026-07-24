"""Build the relocation input store (event-idx-keyed native-rate SAC + PyOcto tables) from ufpipe's OWN
per-year detection + association outputs — the feeder for the `relocate` stage.

This replaces the old `detection_test/reloc_2016_uf/build_sac_and_pyocto.py`, which stitched together
`detection_test/lib`'s per-MONTH catalogs. ufpipe now produces everything the relocation needs directly:

  * per-year PyOcto events + assignment (global `event_idx`) — `config.pyocto_events/assign(model, year)`;
  * a per-year multi-network station table — `stations.build_year_table` (KS/KG/GJ/NS).

So the feeder is just: normalise those into the reloc layout, symlink a network-spanning `merged_archive`,
UF-box filter, and call `uflib.event_sac_export.export_catalog` (which reads the ufpipe assignment schema
natively: it splits `station` -> Network/Code and maps epoch `time` -> peak_time).

Output layout (unchanged, so the PocketQuake driver + stage.py consume it as before):
    <out_root>/
      pyocto/pyocto_kim2011_<year>.csv         event table (idx,time,lat,lon,depth,...)
      pyocto/pyocto_assignment_kim2011_<year>.csv
      station_table/stations_<year>.csv        Network,Code,Latitude,Longitude,Elevation
      merged_archive/<CODE> -> <net-dir>/<CODE> symlinks spanning KS_KG/ GJ/ NS/
      event_sac/<event_idx>/<event_idx>.<NET>.<STA>.<CHAN>.sac  (+ picks in SAC a/t0)
"""
import os
import sys
import argparse

import pandas as pd
from obspy import UTCDateTime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import stations as _stations
from uflib import event_sac_export as ese

# UF subregion box (lon0, lon1, lat0, lat1) — same as the validated reloc.
UF_BOX = (129.25, 129.55, 35.60, 35.90)
VELMODEL = config.ASSOC_VELMODEL          # "kim2011" — matches the pyocto_<vm>_<year> filename
SAC_TARGET_HZ = None                      # keep each station's NATIVE rate (durable full-band product;
#                                            dt.cc xcorr interpolates to a common interp_hz at match time)


def _net_dir(net):
    return {"KS": config.KS_KG_DIR, "KG": config.KS_KG_DIR,
            "GJ": config.GJ_DIR, "NS": config.NS_DIR}.get(net)


def build_station_table(model, year, out_root):
    """Write station_table/stations_<year>.csv from ufpipe's multi-network year table."""
    S = _stations.build_year_table(year)
    out = pd.DataFrame({"Network": S.net, "Code": S.sta, "Latitude": S.lat,
                        "Longitude": S.lon, "Elevation": S.elev}).drop_duplicates("Code")
    dst = os.path.join(out_root, "station_table", f"stations_{year}.csv")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    out.to_csv(dst, index=False)
    print(f"  stations_{year}.csv: {len(out)} stations ({dict(S.net.value_counts())})")
    return out


def build_pyocto(model, year, out_root):
    """Copy ufpipe's per-year PyOcto events + assignment into the reloc pyocto/ dir, in the
    filenames event_sac_export expects (pyocto_<vm>_<year>.csv). ufpipe already keys events by a
    GLOBAL event_idx, so no month-concatenation/re-offset is needed (that was the old lib path)."""
    ev = pd.read_csv(config.pyocto_events(model, year))
    asg = pd.read_csv(config.pyocto_assign(model, year))
    pyd = os.path.join(out_root, "pyocto")
    os.makedirs(pyd, exist_ok=True)
    ev.to_csv(os.path.join(pyd, f"pyocto_{VELMODEL}_{year}.csv"), index=False)
    asg.to_csv(os.path.join(pyd, f"pyocto_assignment_{VELMODEL}_{year}.csv"), index=False)
    print(f"  pyocto events {len(ev)} + assignment {len(asg)} (global event_idx)")
    return ev


def build_merged_archive(model, year, out_root):
    """Symlink a network-spanning continuous archive: merged_archive/<CODE> -> <net-dir>/<CODE>.
    event_sac_export globs <continuous_root>/<CODE>/<CHA>.D/<NET>.<CODE>..<CHA>.D.<year>.<ddd>, so one
    flat <CODE> level spanning KS_KG/ GJ/ NS/ is what it needs. Rebuilt each run (idempotent)."""
    ma = os.path.join(out_root, "merged_archive")
    # Legacy reloc dirs had merged_archive as a SYMLINK (sometimes dangling after the restructure);
    # os.makedirs would crash on a dangling link. Replace any symlink with a real dir.
    if os.path.islink(ma):
        os.unlink(ma)
    os.makedirs(ma, exist_ok=True)
    S = _stations.build_year_table(year)
    n = 0
    for _, r in S.iterrows():
        src = os.path.join(_net_dir(r.net), r.sta)
        if not os.path.isdir(src):
            continue
        link = os.path.join(ma, r.sta)
        if os.path.lexists(link):
            if os.path.realpath(link) == os.path.realpath(src):
                n += 1
                continue
            os.unlink(link)
        os.symlink(src, link)
        n += 1
    print(f"  merged_archive: {n} station symlinks -> KS_KG/ GJ/ NS/")
    return ma


def build_uf_catalog(ev):
    """UF-box filter -> catalog keyed event_id = str(global event_idx) (SAC dir names)."""
    ev = ev.copy()
    ev["time"] = pd.to_datetime(ev.time, format="ISO8601", utc=True)
    # ufpipe events use latitude/longitude; accept lat/lon too for robustness
    lat = ev["latitude"] if "latitude" in ev.columns else ev["lat"]
    lon = ev["longitude"] if "longitude" in ev.columns else ev["lon"]
    box = ev[(lon >= UF_BOX[0]) & (lon <= UF_BOX[1]) & (lat >= UF_BOX[2]) & (lat <= UF_BOX[3])].copy()
    box = box.reset_index(drop=True)
    box["lat"] = (box["latitude"] if "latitude" in box.columns else box["lat"])
    box["lon"] = (box["longitude"] if "longitude" in box.columns else box["lon"])
    box["event_id"] = box["idx"].astype(int).astype(str)
    box["origin_utc"] = box["time"].apply(lambda t: UTCDateTime(t.to_pydatetime()))
    print(f"  UF-box catalog: {len(box)} events")
    return box[["event_id", "origin_utc", "time", "lat", "lon", "depth", "idx"]]


def build_reloc_inputs(model, year, out_root):
    """Build the full reloc input store for (model, year) under out_root, from ufpipe's own outputs."""
    print(f"=== reloc inputs: {model} {year} -> {out_root} ===")
    print("[1] station table"); build_station_table(model, year, out_root)
    print("[2] pyocto year files"); ev = build_pyocto(model, year, out_root)
    print("[3] merged archive"); ma = build_merged_archive(model, year, out_root)
    print("[4] UF catalog"); cat = build_uf_catalog(ev)
    print("[5] SAC extraction (merged archive, PyOcto picks -> a/t0) ...")
    summ = ese.export_catalog(
        cat, picks_root="", continuous_root=ma,
        station_table_root=os.path.join(out_root, "station_table"),
        out_root=os.path.join(out_root, "event_sac"),
        pyocto_root=os.path.join(out_root, "pyocto"), pyocto_velmodel=VELMODEL,
        target_hz=SAC_TARGET_HZ, skip_existing=True, progress=True)
    summ.to_csv(os.path.join(out_root, "event_sac_export_summary.csv"), index=False)
    ok = int((summ.status == "ok").sum()) if "status" in summ else len(summ)
    print(f"DONE: {ok}/{len(cat)} events with SAC -> {out_root}/event_sac/")
    return out_root


def main():
    ap = argparse.ArgumentParser(description="Build relocation inputs from ufpipe's own detection+association.")
    ap.add_argument("--model", default="phasenet_plus")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out-root", required=True, help="reloc working dir (e.g. detection_test/reloc_<year>_uf[_<model>])")
    a = ap.parse_args()
    build_reloc_inputs(a.model, a.year, a.out_root)


if __name__ == "__main__":
    main()
