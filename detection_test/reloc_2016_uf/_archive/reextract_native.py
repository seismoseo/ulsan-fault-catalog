#!/usr/bin/env python
"""Parallel re-extraction of the 2016 UF event-SAC store at NATIVE sampling rate.

Reuses the already-built station_table/pyocto/catalog from build_sac_and_pyocto.py (unchanged) and re-runs
event_sac_export.export_catalog with target_hz=None (keep native rate: KS/KG 100, NS 200, GJ mixed 100/200/1000).
This restores the >100 Hz GJ picks that the old 100 Hz guard silently dropped (2845/3867 events affected) AND
preserves the full band as a durable data product. Parallel over catalog chunks — each worker writes disjoint
<event_id> dirs, so it is process-safe. dt.cc interpolates to a common interp_hz at correlation time; HypoInverse
uses only the rate-independent a/t0 pick times.

    python reextract_native.py --workers 12       # wipe + rebuild event_sac/ native, in parallel
"""
import os, sys, glob, argparse, time
import pandas as pd
from multiprocessing import Pool
from obspy import UTCDateTime
sys.path.insert(0, "/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv")
import event_sac_export as ese

ROOT = "/home/msseo/works/02.Ulsan_Fault_detection/detection_test/reloc_2016_uf"
UF = (129.25, 129.55, 35.60, 35.90)
VELMODEL = "kim2011"
SAC_TARGET_HZ = None          # native rate (see build_sac_and_pyocto.py for the rationale)


def build_uf_catalog():
    EV = pd.read_csv(f"{ROOT}/pyocto/pyocto_{VELMODEL}_2016.csv")
    EV["time"] = pd.to_datetime(EV.time, format="ISO8601", utc=True)
    box = EV[(EV.lon >= UF[0]) & (EV.lon <= UF[1]) & (EV.lat >= UF[2]) & (EV.lat <= UF[3])].reset_index(drop=True)
    box["event_id"] = box["idx"].astype(int).astype(str)
    box["origin_utc"] = box["time"].apply(lambda t: UTCDateTime(t.to_pydatetime()))
    return box[["event_id", "origin_utc", "time", "lat", "lon", "depth", "idx"]]


def _run_chunk(chunk):
    return ese.export_catalog(
        chunk, picks_root="", continuous_root=f"{ROOT}/merged_archive",
        station_table_root=f"{ROOT}/station_table", out_root=f"{ROOT}/event_sac",
        pyocto_root=f"{ROOT}/pyocto", pyocto_velmodel=VELMODEL,
        target_hz=SAC_TARGET_HZ, skip_existing=True, progress=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cat = build_uf_catalog()
    if a.limit:
        cat = cat.head(a.limit)
    n = len(cat)
    chunks = [cat.iloc[i::a.workers].reset_index(drop=True) for i in range(a.workers)]  # round-robin -> balanced days
    print(f"re-extracting {n} events NATIVE rate, {a.workers} workers -> {ROOT}/event_sac", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        parts = pool.map(_run_chunk, chunks)
    summ = pd.concat(parts, ignore_index=True)
    summ.to_csv(f"{ROOT}/event_sac_export_summary.csv", index=False)
    vc = summ.status.value_counts().to_dict()
    print(f"DONE {n} events in {time.time()-t0:.0f}s. status: {vc}", flush=True)


if __name__ == "__main__":
    main()
