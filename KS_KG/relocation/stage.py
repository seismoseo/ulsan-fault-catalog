#!/usr/bin/env python
"""Stage the EXISTING Ulsan KS/KG event waveforms (and, for the 'reuse' run, the existing PhaseNet+
picks) into a scaffolded PocketQuake **stp_sac** cluster — so the PocketQuake pipeline relocates them
with no re-download and no custom relocation code.

The Ulsan SAC are already in STP naming (`{eid}.{NET}.{STA}.{CHAN}.sac`); the stp_sac layout just nests
them in HH/HG/EL subdirs (STP_GLOB = `{sensor}/*{sensor}{comp}*.sac`). So we **symlink**:
    event_waveforms_ulsanfault/{eid}/{eid}.{NET}.{STA}.{CHAN}.sac
 -> <stp_sac_root>/{eid}/{HH|HG|EL}/{eid}.{NET}.{STA}.{CHAN}.sac

For --reuse-picks we also convert each Ulsan `{eid}_picks.csv` (Network,Code,phase,peak_time,…,
probability) into PocketQuake's `picks/{eid}_picks.csv` (Event_ID,Network,Station,Phase,Time,
Probability) so HypoInverse/ph2dt read the existing picks and the picking stage can be skipped.

Usage:  PYTHONPATH=<eq-cycle pipeline repo> python stage.py <cluster_slug> [--reuse-picks]
"""
import argparse
import glob
import os
import sys

import pandas as pd

from pipeline import config

ULSAN_WF = "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/HypoInv/event_waveforms_ulsanfault"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cluster")
    ap.add_argument("--reuse-picks", action="store_true",
                    help="also convert the existing Ulsan picks into picks/{eid}_picks.csv")
    ap.add_argument("--members", default=os.path.join(HERE, "family738", "members.txt"))
    a = ap.parse_args()

    cfg = config.load_cluster(a.cluster)
    stp_root = cfg.stp_sac_root
    members = [ln.strip() for ln in open(a.members) if ln.strip()]

    n_wf = 0
    for eid in members:
        src = os.path.join(ULSAN_WF, eid)
        if not os.path.isdir(src):
            print(f"  MISSING waveforms: {eid}", file=sys.stderr); continue
        for f in glob.glob(os.path.join(src, f"{eid}.*.sac")):
            chan = os.path.basename(f).split(".")[3]          # {eid}.{NET}.{STA}.{CHAN}.sac
            sensor = chan[:2]
            if sensor not in ("HH", "HG", "EL"):
                continue
            d = os.path.join(stp_root, eid, sensor)
            os.makedirs(d, exist_ok=True)
            dst = os.path.join(d, os.path.basename(f))
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(f), dst); n_wf += 1
    print(f"staged {n_wf} SAC symlinks -> {stp_root}")

    if a.reuse_picks:
        pdir = config.picks_dir(cfg); os.makedirs(pdir, exist_ok=True)
        n_pk = 0
        for eid in members:
            pc = os.path.join(ULSAN_WF, eid, f"{eid}_picks.csv")
            if not os.path.exists(pc):
                print(f"  MISSING picks: {eid}", file=sys.stderr); continue
            df = pd.read_csv(pc)
            out = pd.DataFrame({
                "Event_ID": eid, "Network": df["Network"], "Station": df["Code"],
                "Phase": df["phase"], "Time": df["peak_time"], "Probability": df["probability"],
            })
            out.to_csv(config.picks_csv(cfg, eid), index=False); n_pk += 1
        print(f"converted {n_pk} pick CSVs -> {pdir}  (run with --stage-from hypoinverse)")


if __name__ == "__main__":
    main()
