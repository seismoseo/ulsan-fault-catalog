#!/usr/bin/env python
"""merge_ns_nested.py — merge the accidentally-nested NS/NS/ SDS tree up into NS/.

USAGE (run from 02.Ulsan_Fault_detection/):
    python merge_ns_nested.py            # DRY RUN: full report, nothing moved
    python merge_ns_nested.py --apply    # perform the merge

WHAT IT DOES
  For every station directory NS/NS/<STA>/:
    * if NS/<STA> does not exist          -> move the whole station directory up (os.rename, instant)
    * else, for each channel dir <CH>.D/  -> move it whole if absent, otherwise move file-by-file
  COLLISION POLICY: a file that already exists at the destination is NEVER overwritten.
    - identical size  -> counted as "duplicate (identical size)"; source file is LEFT IN PLACE
    - different size  -> counted as "CONFLICT";                   source file is LEFT IN PLACE
    Both lists are written to merge_ns_nested_report.txt for review. Nothing is deleted.
  Emptied directories are removed afterwards; NS/NS itself is removed only if fully empty.

Everything happens within one filesystem (NS/NS is inside NS), so moves are renames — no data is copied.
"""
import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DST  = os.path.join(ROOT, "NS")
SRC  = os.path.join(DST, "NS")
APPLY = "--apply" in sys.argv

moved_dirs = moved_files = dup_same = conflicts = 0
dup_list, conf_list = [], []

def log(msg): print(msg)

if not os.path.isdir(SRC):
    sys.exit(f"nothing to do: {SRC} does not exist")

log(f"{'APPLY' if APPLY else 'DRY RUN'}: merging {SRC} -> {DST}")
for sta in sorted(os.listdir(SRC)):
    s_sta = os.path.join(SRC, sta); d_sta = os.path.join(DST, sta)
    if not os.path.isdir(s_sta):
        log(f"[skip non-dir] {s_sta}"); continue
    if not os.path.isdir(d_sta):                       # station absent above -> move whole dir
        log(f"[move station] {sta}")
        if APPLY: os.rename(s_sta, d_sta)
        moved_dirs += 1
        continue
    for ch in sorted(os.listdir(s_sta)):               # station exists -> merge channel dirs
        s_ch = os.path.join(s_sta, ch); d_ch = os.path.join(d_sta, ch)
        if not os.path.isdir(s_ch): continue
        if not os.path.isdir(d_ch):
            log(f"[move channel] {sta}/{ch}")
            if APPLY: os.rename(s_ch, d_ch)
            moved_dirs += 1
            continue
        for f in sorted(os.listdir(s_ch)):             # channel exists -> merge files
            s_f = os.path.join(s_ch, f); d_f = os.path.join(d_ch, f)
            if not os.path.exists(d_f):
                if APPLY: os.rename(s_f, d_f)
                moved_files += 1
            elif os.path.getsize(s_f) == os.path.getsize(d_f):
                dup_same += 1; dup_list.append(f)      # left in place, listed for review
            else:
                conflicts += 1
                conf_list.append(f"{f}  src={os.path.getsize(s_f)}B dst={os.path.getsize(d_f)}B")

if APPLY:                                              # clean up emptied dirs (only if empty)
    for dirpath, dirnames, filenames in os.walk(SRC, topdown=False):
        if not os.listdir(dirpath): os.rmdir(dirpath)

rep = os.path.join(ROOT, "merge_ns_nested_report.txt")
with open(rep, "w") as fh:
    fh.write(f"duplicates (identical size, left in source): {dup_same}\n")
    fh.writelines(f"  {x}\n" for x in dup_list)
    fh.write(f"\nCONFLICTS (different size, left in source): {conflicts}\n")
    fh.writelines(f"  {x}\n" for x in conf_list)

log("-" * 70)
log(f"station/channel dirs moved : {moved_dirs}{'' if APPLY else ' (would be)'}")
log(f"individual files moved     : {moved_files}{'' if APPLY else ' (would be)'}")
log(f"duplicates (same size)     : {dup_same}   -> left in {SRC}, listed in {os.path.basename(rep)}")
log(f"CONFLICTS (size differs)   : {conflicts}   -> left in {SRC}, listed in {os.path.basename(rep)}")
if APPLY and not os.path.isdir(SRC):
    log(f"{SRC} fully merged and removed.")
elif APPLY:
    log(f"NOTE: {SRC} still contains the duplicate/conflict files above — review the report, then decide.")
else:
    log("DRY RUN complete — nothing was moved. Re-run with --apply to perform the merge.")
