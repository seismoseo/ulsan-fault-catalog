#!/usr/bin/env python
"""Normalise the GJ temporary-array 2016 archive to a uniform 100 Hz mirror for the SAC store.

The GJ archive is MIXED-RATE across days WITHIN a station (a given station is 100 Hz on some days,
200 or 1000 Hz on others). Detection saw them at 100 Hz because EQNet's dataset resamples every trace to
100 Hz internally; but event_sac_export.read_continuous_for_event asserts sampling_rate==100 Hz and rejects
anything else, silently dropping every >100 Hz GJ pick (2845/3867 UF events affected -> zero-phase events).

This builds GJ_100hz/<STA>/<CHA>.D/<same filename> where every file is 100 Hz, using the SAME anti-alias
decimation as predecimate_ns.py / the detection pipeline (gj_config: lowpass 0.4*100 Hz, 4 corners, zerophase,
integer decimate). Files already at 100 Hz are symlinked through unchanged (no needless rewrite / duplication).
Per-file rate handling => robust to the within-station rate changes. merged_archive/<STA> is then repointed
here so the SAC extractor reads uniform 100 Hz.

    python decimate_gj.py --workers 40            # build mirror for all 29 GJ stations, 2016
"""
import os, sys, glob, argparse, time
import numpy as np, obspy, pandas as pd
from multiprocessing import Pool

HERE = "/home/msseo/works/02.Ulsan_Fault_detection/detection_test"
sys.path.insert(0, os.path.join(HERE, "lib"))
import gj_config as GJC
ROOT = "/home/msseo/works/02.Ulsan_Fault_detection"
GJ_ARCH = os.path.join(ROOT, "GJ")
OUT_ROOT = os.path.join(ROOT, "GJ_100hz")       # uniform-100 Hz mirror
TARGET_FS = GJC.TARGET_FS
YEAR = 2016


def norm_one(f):
    """Normalise a single day-file to a uniform 100 Hz in the mirror.

    GJ day-files can carry DIFFERENT sampling rates WITHIN one file (a rate change mid-day), so we anti-alias
    decimate EACH trace to 100 Hz FIRST (per-trace factor), and only THEN merge — merging before decimation
    fails with "differing sampling rates". No symlink shortcut: a file whose first record is 100 Hz may still
    contain 200/1000 Hz records later, which the SAC extractor's 100 Hz guard would reject. Always rewrite."""
    sta = f.split("/")[-3]                                    # GJ/<STA>/<CHA>.D/<file>
    ch = os.path.basename(os.path.dirname(f))                 # e.g. HHZ.D
    outdir = os.path.join(OUT_ROOT, sta, ch)
    outpath = os.path.join(outdir, os.path.basename(f))
    if os.path.lexists(outpath):
        return 0
    os.makedirs(outdir, exist_ok=True)
    try:
        st = obspy.read(f, format="MSEED")
        for tr in st:                                         # decimate per-trace BEFORE merge
            fs = tr.stats.sampling_rate
            if fs > TARGET_FS + 0.1:
                fac = int(round(fs / TARGET_FS))
                tr.filter("lowpass", freq=GJC.ANTIALIAS_FRAC * TARGET_FS,
                          corners=GJC.ANTIALIAS_CORNERS, zerophase=True)
                tr.decimate(fac, no_filter=True)
            tr.data = np.round(tr.data).astype(np.int32)
        st.merge(fill_value=0)                                # now all 100 Hz -> merges cleanly
        st.write(outpath, format="MSEED", encoding="STEIM2", reclen=4096)
        return 1
    except Exception as e:
        print(f"  ERR {sta}/{os.path.basename(f)}: {e}", flush=True); return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=40)
    a = ap.parse_args()
    st = pd.read_csv(os.path.join(HERE, "reloc_2016_uf", "station_table", "stations_2016.csv"))
    gj = sorted(st[st.Network == "GJ"].Code.tolist())
    files = []
    for s in gj:
        files += sorted(glob.glob(os.path.join(GJ_ARCH, s, "*.D", f"*.{YEAR}.*")))
    print(f"GJ stations {len(gj)}; day-files to normalise: {len(files)} -> {OUT_ROOT} ({a.workers} workers)", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(norm_one, files, chunksize=8), 1):
            res.append(r)
            if i % 500 == 0:
                print(f"  {i}/{len(files)}  ({time.time()-t0:.0f}s)", flush=True)
    sym = sum(1 for r in res if r == 1)
    err = sum(1 for r in res if r == -1)
    print(f"DONE: {len(files)} files, {err} errors, {time.time()-t0:.0f}s -> {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
