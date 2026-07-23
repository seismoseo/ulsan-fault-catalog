#!/usr/bin/env python
"""Stage the canonical event_idx-keyed UF waveform store into a scaffolded PocketQuake **stp_sac** cluster.

The korea-cluster-relocation pipeline is internally TIMESTAMP-keyed: `waveforms.load_catalog` derives
`event_id = strftime((catalog KST) - 9h)` and the gather/write_phs look for `stp_sac/<timestamp>/`. So we
bridge our canonical `event_idx` store to the pipeline's timestamp id HERE, drift-free by construction:
for each member (event_idx) we compute the SAME timestamp the pipeline will (`floor(UTC origin)`) from the
SINGLE current catalog (members_event_idx.csv) and stage that member's SACs under `stp_sac/<timestamp>/`,
renaming the `<event_idx>.` prefix to `<timestamp>.`. No cross-version matching: catalog, staged dir and
SAC origin all come from one current source.

    event_waveforms_ufidx/<event_idx>/<event_idx>.<NET>.<STA>.<CHAN>.sac
 -> <stp_sac_root>/<timestamp>/<HH|HG|EL>/<timestamp>.<NET>.<STA>.<CHAN>.sac

Picks (probability weights) are converted to picks/<timestamp>_picks.csv.

Usage: python stage.py <slug> --reuse-picks --members members.txt [--catalog members_event_idx.csv]
"""
import argparse
import glob
import os
import sys

import pandas as pd

from pipeline import config

STORE = "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/HypoInv/event_waveforms_ufidx"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cluster")
    ap.add_argument("--reuse-picks", action="store_true")
    ap.add_argument("--members", default=os.path.join(HERE, "family738", "members.txt"))
    ap.add_argument("--wf-root", default=STORE, help="event_idx-keyed waveform store")
    ap.add_argument("--catalog", default=None,
                    help="members_event_idx.csv (event_idx,time,...); default: alongside --members")
    a = ap.parse_args()
    wf_root = a.wf_root
    cat_path = a.catalog or os.path.join(os.path.dirname(a.members), "members_event_idx.csv")

    cfg = config.load_cluster(a.cluster)
    stp_root = cfg.stp_sac_root
    members = [ln.strip() for ln in open(a.members) if ln.strip()]
    # event_idx -> pipeline timestamp (floor UTC second == strftime((KST,int sec) - 9h))
    mc = pd.read_csv(cat_path)
    ts_of = {str(int(r.event_idx)): pd.to_datetime(r.time, utc=True).floor("s").strftime("%Y%m%d%H%M%S")
             for r in mc.itertuples()}
    seen_ts = {}

    n_wf = 0
    for eid in members:
        ts = ts_of.get(eid)
        if ts is None:
            print(f"  NO catalog time for member {eid}", file=sys.stderr); continue
        if ts in seen_ts and seen_ts[ts] != eid:
            print(f"  DOUBLET same-second {ts}: members {seen_ts[ts]} & {eid} -> pipeline keeps one", file=sys.stderr)
        seen_ts.setdefault(ts, eid)
        src = os.path.join(wf_root, eid)
        if not os.path.isdir(src):
            print(f"  MISSING waveforms: {eid}", file=sys.stderr); continue
        for f in glob.glob(os.path.join(src, f"{eid}.*.sac")):
            parts = os.path.basename(f).split(".")            # {event_idx}.{NET}.{STA}.{CHAN}.sac
            if len(parts) < 5:
                continue
            chan = parts[3]; sensor = chan[:2]
            if sensor not in ("HH", "HG", "EL"):
                continue
            d = os.path.join(stp_root, ts, sensor); os.makedirs(d, exist_ok=True)
            dst = os.path.join(d, ".".join([ts] + parts[1:]))  # rename prefix event_idx -> timestamp
            if not os.path.lexists(dst):
                os.symlink(os.path.realpath(f), dst); n_wf += 1
    print(f"staged {n_wf} SAC symlinks -> {stp_root}  ({len(seen_ts)} timestamp dirs)")

    if a.reuse_picks:
        pdir = config.picks_dir(cfg); os.makedirs(pdir, exist_ok=True)
        n_pk = 0
        for eid in members:
            ts = ts_of.get(eid)
            pc = os.path.join(wf_root, eid, f"{eid}_picks.csv")
            if ts is None or not os.path.exists(pc):
                continue
            df = pd.read_csv(pc)
            out = pd.DataFrame({
                "Event_ID": ts, "Network": df["Network"], "Station": df["Code"],
                "Phase": df["phase"], "Time": df["peak_time"], "Probability": df["probability"],
            })
            out.to_csv(config.picks_csv(cfg, ts), index=False); n_pk += 1
        print(f"converted {n_pk} pick CSVs -> {pdir}")


if __name__ == "__main__":
    main()
