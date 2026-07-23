#!/usr/bin/env python
"""Pre-decimate the 200 Hz stations of a month to 100 Hz, ONCE, in parallel.

    python predecimate_ns.py --month 2021-09 --workers 40

Reads cache/stations_<tag>.csv, and for every usable station whose native rate exceeds 100 Hz (the NS/GJ
200 Hz arrays), anti-alias decimates each day's 3 components to 100 Hz and writes them to a mirror archive
    NS_100hz/<STA>/<BAND>?.D/<same filename>
keeping the identical filename convention so the pickers can read it unchanged. Stations already at 100 Hz
(KS/KG HH/EL/HG) are skipped — the pickers read those from their original archive.

This is the #1 full-run accelerator: it turns the per-run 3.36 s/channel decimation into a one-time cost and
halves bytes read on every subsequent picking pass. STEIM2 int32 output keeps files compact.
"""
import os, sys, glob, argparse, time
import numpy as np, pandas as pd, obspy
from multiprocessing import Pool

ROOT = "/home/msseo/works/02.Ulsan_Fault_detection"
HERE = f"{ROOT}/detection_test"
OUT_ROOT = f"{ROOT}/NS_100hz"          # mirror archive for pre-decimated 200 Hz stations
import gj_config as GJC                     # single disclosed source of parameters
TARGET_FS = GJC.TARGET_FS


def decimate_station(args):
    archive, sta, band, Y, doys = args
    n_written = 0
    for doy in doys:
        for f in sorted(glob.glob(os.path.join(archive, sta, f"{band}?.D", f"*.{Y}.{doy:03d}"))):
            ch = os.path.basename(os.path.dirname(f))            # e.g. HHZ.D
            outdir = os.path.join(OUT_ROOT, sta, ch)
            outpath = os.path.join(outdir, os.path.basename(f))
            if os.path.exists(outpath):
                n_written += 1; continue
            try:
                st = obspy.read(f, format="MSEED")
                st.merge(fill_value=0)
                for tr in st:
                    fs = tr.stats.sampling_rate
                    if fs > TARGET_FS:
                        fac = int(round(fs / TARGET_FS))
                        tr.filter("lowpass", freq=GJC.ANTIALIAS_FRAC * TARGET_FS, corners=GJC.ANTIALIAS_CORNERS, zerophase=True)
                        tr.decimate(fac, no_filter=True)
                    tr.data = np.round(tr.data).astype(np.int32)  # compact STEIM2
                os.makedirs(outdir, exist_ok=True)
                st.write(outpath, format="MSEED", encoding="STEIM2", reclen=4096)
                n_written += 1
            except Exception as e:
                print(f"    ERR {sta} {os.path.basename(f)}: {e}", flush=True)
    return sta, n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2021-09")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="only first N stations (speed test)")
    a = ap.parse_args()
    Y, MO = int(a.month[:4]), int(a.month[5:7]); tag = f"{Y}_{MO:02d}"
    T0 = pd.Timestamp(f"{a.month}-01"); T1 = T0 + pd.offsets.MonthEnd(0)
    doys = list(range(T0.dayofyear, T1.dayofyear + 1))

    S = pd.read_csv(os.path.join(HERE, "cache", f"stations_{tag}.csv"))
    S = S[S.coverage >= 0.8].copy()
    # only stations whose native rate is >100 Hz -> probe one file per station
    def native_fs(r):
        fs = sorted(glob.glob(os.path.join(r.archive, r.sta, f"{r.band}?.D", f"*.{Y}.*")))
        if not fs: return 0
        try: return obspy.read(fs[0], headonly=True)[0].stats.sampling_rate
        except Exception: return 0
    S["native_fs"] = [native_fs(r) for _, r in S.iterrows()]
    HR = S[S.native_fs > TARGET_FS].reset_index(drop=True)
    if a.limit: HR = HR.head(a.limit)
    print(f"[{tag}] {len(HR)}/{len(S)} stations are >100 Hz -> pre-decimating to {OUT_ROOT} "
          f"({len(doys)} days, {a.workers} workers)", flush=True)

    jobs = [(r.archive, r.sta, r.band, Y, doys) for _, r in HR.iterrows()]
    t0 = time.time(); done = 0
    with Pool(a.workers) as pool:
        for sta, n in pool.imap_unordered(decimate_station, jobs):
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} stations  [{time.time()-t0:.0f}s]", flush=True)
    print(f"[{tag}] pre-decimation done: {len(jobs)} stations in {time.time()-t0:.0f}s -> {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
