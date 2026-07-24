#!/usr/bin/env python
"""Build the CORRECT uf_cluster-QC'd catalog for the 2016 UF-box relocation.

FIX: the HypoInverse .sum row i corresponds to members.txt[i] (== catalog_kma row i), verified by
corr(HypoInv num, pyocto npick)=0.999. An earlier cuspid->timestamp mapping was wrong (a 5 s time-match
falsely validated it inside the dense aftershock sequence), producing an all-pre-GJ 596-set with NO GJ picks.
Correct QC set = 190 pre-GJ + 406 post-GJ = 596, of which 67% carry GJ picks (GJ is the dominant network,
5346 picks) -> the genuine GJ-array-enhanced relocation input.

Writes catalog_kma_qc.csv, members_qc.txt, members_event_idx_qc.csv (over-writing the buggy versions)."""
import os, sys, argparse
import pandas as pd, numpy as np
from uflib import uf_cluster as uf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import year_paths as YP
HERE = YP.HERE
RUNS = YP.RUNS


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--picker", default="phasenet_plus")
    YP.add_year_arg(ap)
    a = ap.parse_args()
    ROOT, slug = YP.root_dir(a.year, a.picker), YP.slug(a.year, a.picker)
    SUM = f"{RUNS}/{slug}/1.HypoInv/kim2011/{slug}.sum"
    cat = pd.read_csv(f"{ROOT}/catalog_kma.csv")                    # row i <-> members[i]
    mem = pd.read_csv(f"{ROOT}/members.txt", header=None)[0].tolist()
    me = pd.read_csv(f"{ROOT}/members_event_idx.csv")
    sm = uf.read_sum(SUM).reset_index(drop=True)
    QC = uf.QC
    if len(sm) == len(cat):                                       # nothing dropped -> exact row-order (sm row i <-> members[i])
        smm = sm; matched = np.arange(len(cat))
        print(f"[{a.picker}] {len(sm)} located = {len(cat)} members (no drops) -> exact row-order")
    else:                                                         # HypoInverse dropped events -> .sum is a time-ordered
        # SUBSEQUENCE of members; monotonic two-pointer align on origin time (tz offset auto-detected).
        cat_t = pd.to_datetime(dict(year=cat.Year, month=cat.Month, day=cat.Day, hour=cat.Hour,
                                    minute=cat.Minute, second=cat.Second.clip(0, 59))).to_numpy()
        sm_t = pd.to_datetime(sm["time"]).to_numpy()
        def med_nn(o):
            s = sm_t - np.timedelta64(int(o), "s"); k = np.clip(np.searchsorted(cat_t, s), 0, len(cat_t) - 1)
            d = np.abs((cat_t[k] - s)).astype("timedelta64[s]").astype(float)
            km = np.clip(k - 1, 0, len(cat_t) - 1); dm = np.abs((cat_t[km] - s)).astype("timedelta64[s]").astype(float)
            return np.median(np.minimum(d, dm))
        off = min([0, 32400, -32400], key=med_nn)
        sm_t = sm_t - np.timedelta64(int(off), "s")
        sm_row = np.full(len(cat), -1, int); j = 0; TOL = np.timedelta64(10, "s")
        for i in range(len(cat)):
            while j < len(sm) and sm_t[j] < cat_t[i] - TOL: j += 1
            if j < len(sm) and abs(sm_t[j] - cat_t[i]) <= TOL: sm_row[i] = j; j += 1
        matched = np.where(sm_row >= 0)[0]; smm = sm.iloc[sm_row[matched]]
        print(f"[{a.picker}] aligned {len(matched)}/{len(cat)} catalog events "
              f"({len(sm)} located, {len(cat)-len(sm)} dropped by HypoInverse; tz off {off/3600:+.0f}h)")
    qmask = ((smm.erh.values < QC["erh"]) & (smm.erz.values < QC["erz"]) & (smm.gap.values < QC["gap"])
             & (smm.num.values > QC["num"]) & (smm.rms.values < QC["rms"]))
    rows = matched[qmask]                                         # catalog/members row indices passing QC
    qc_eidx = [int(mem[i]) for i in rows]
    cat.iloc[rows].reset_index(drop=True).to_csv(f"{ROOT}/catalog_kma_qc.csv", index=False)
    pd.Series(qc_eidx).to_csv(f"{ROOT}/members_qc.txt", index=False, header=False)
    keep = set(qc_eidx)
    me[me.event_idx.isin(keep)].to_csv(f"{ROOT}/members_event_idx_qc.csv", index=False)

    t = pd.to_datetime(me.set_index("event_idx").loc[qc_eidx, "time"].values, utc=True)
    if a.year == 2016:                                             # Gyeongju mainshock 2016-09-12 (KST)
        pre = int((t < pd.Timestamp("2016-09-12", tz="UTC")).sum())
        print(f"QC catalog: {len(qc_eidx)} events ({pre} pre-GJ + {len(qc_eidx)-pre} post-GJ)")
    else:
        print(f"QC catalog: {len(qc_eidx)} events ({a.year})")
    print(f"  -> catalog_kma_qc.csv, members_qc.txt, members_event_idx_qc.csv")


if __name__ == "__main__":
    main()
