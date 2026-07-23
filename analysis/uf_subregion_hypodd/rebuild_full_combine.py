#!/usr/bin/env python
"""Rebuild dt.cc_P_0.7 / dt.cc_S_0.7 / dt.cc_0.7_combined over ALL event pairs.

WHY: the pipeline's xcorr combine (core/xcorr.py) reuses the `pairs` variable AFTER the XCORR_RESUME
step has reduced it to only the not-yet-computed pairs, so on a resumed run `_filter_combine` writes
ONLY the new pairs' CC links into dt.cc_0.7_combined — silently dropping every reused pair. The per-pair
files (dt.cc_{P,S}_<ts1>_<ts2>) for ALL pairs are intact on disk; this rebuilds the combined over the
FULL combinations(events,2), exactly replicating _filter_combine (header lines + cc in [0.7,1.0] from
col 23:34), parallelised across cores. Order within the combined is irrelevant to HypoDD.
"""
import glob
import os
from itertools import combinations
from multiprocessing import Pool

B = "/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/uf_subregion_reuse/2.HypoDD/02.dt.cc"
WF = "/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/uf_subregion_reuse/waveforms_100km"
THR = 0.7
NPROC = 24

events = sorted(os.path.basename(d) for d in glob.glob(os.path.join(WF, "20*")))
pairs = list(combinations(events, 2))
# CURRENT cuspid = 200000 + position in the sorted event list (matches write_phs / this run's event.dat).
# REUSED per-pair files carry STALE header cuspids (the previous run's positions, shifted by the 106 added
# events), so we DISCARD each file's header and rewrite it with the CURRENT cuspids from the filename
# timestamps. The data lines (station dt weight phase) carry no cuspid and stay verbatim.
ts2cusp = {t: 200000 + i for i, t in enumerate(events)}
print(f"events {len(events)}  full pairs {len(pairs):,}", flush=True)


def _filter_chunk(args):
    idx, phase, lo, hi = args
    pd = os.path.join(B, f"dt.cc_{phase}")
    out = os.path.join(B, f".tmp_{phase}_{idx:04d}")
    nh = no = 0
    with open(out, "w") as o:
        for e1, e2 in pairs[lo:hi]:
            p = os.path.join(pd, f"dt.cc_{phase}_{e1}_{e2}")
            if not os.path.exists(p):
                continue
            c1, c2 = ts2cusp[e1], ts2cusp[e2]          # CURRENT ids from the (stable) filename timestamps
            wrote_hdr = False
            for line in open(p):
                if line.startswith("#"):
                    continue                            # drop the file's stale cuspid header
                try:
                    cc = float(line[23:34].replace(" ", ""))
                    if THR <= cc <= 1.0:
                        if not wrote_hdr:
                            o.write(f"#  {c1}  {c2}   0.0\n"); nh += 1; wrote_hdr = True
                        o.write(line); no += 1
                except ValueError:
                    pass
    return idx, phase, nh, no


def run_phase(phase):
    sz = (len(pairs) + NPROC - 1) // NPROC
    tasks = [(i, phase, i * sz, min((i + 1) * sz, len(pairs))) for i in range(NPROC)]
    with Pool(NPROC) as pool:
        res = sorted(pool.map(_filter_chunk, tasks))
    out = os.path.join(B, f"dt.cc_{phase}_0.7")
    with open(out, "w") as o:
        for idx, _, _, _ in res:
            t = os.path.join(B, f".tmp_{phase}_{idx:04d}")
            o.write(open(t).read()); os.remove(t)
    nh = sum(r[2] for r in res); no = sum(r[3] for r in res)
    print(f"  dt.cc_{phase}_0.7: {nh:,} pair-headers, {no:,} cc>=0.7 observations", flush=True)
    return nh, no


print("filtering P ...", flush=True); ph, po = run_phase("P")
print("filtering S ...", flush=True); sh, so = run_phase("S")
combined = os.path.join(B, "dt.cc_0.7_combined")
with open(combined, "w") as o:
    for f in ("dt.cc_P_0.7", "dt.cc_S_0.7"):
        o.write(open(os.path.join(B, f)).read())
print(f"\nwrote {combined}", flush=True)
print(f"TOTAL used links: P {po:,} + S {so:,} = {po+so:,} cc>=0.7 observations "
      f"over {ph:,} P-pairs / {sh:,} S-pairs (vs the broken 301,920-pair combine)", flush=True)
