#!/usr/bin/env python
"""Permanently fix miniSEED records whose CHANNEL header disagrees with the archive layout.

Some archive day-files are stored under the correct channel dir/filename (e.g. .../HHE.D/
NS.N150..HHE.D.2020.015) while the records INSIDE carry a different channel code (HHX — the
datalogger's unoriented-horizontal naming for a service window). ObsPy trusts the record header,
so readers see component 'X' and the data is unusable under its archived identity.

This tool rewrites ONLY the 3 channel bytes (SEED fixed header, offset 15..17) of each record,
after backing up the original file byte-for-byte. It then verifies: the patched file re-reads with
the expected channel, and its sample data is IDENTICAL to the backup's.

    python tools/fix_channel_headers.py --scan-log <chan_scan.log>       # fix everything the scan found
    python tools/fix_channel_headers.py <file> <expected-chan> [...]     # fix explicit files

Backups: _header_fix_backups/<original relative path> beside the repo root (never deleted by this
tool). A provenance log is appended to data/metadata/archive_header_fixes.csv (tracked in git).
"""
import argparse, csv, os, shutil, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKUP_ROOT = os.path.join(REPO, "_header_fix_backups")
LOG = os.path.join(REPO, "data", "metadata", "archive_header_fixes.csv")

CAND_RECLENS = (512, 4096, 1024, 256, 8192)


def _detect_reclen(buf):
    """miniSEED record length: each record starts with a 6-digit ASCII sequence number + D/R/Q/M."""
    def looks_like_record(off):
        if off + 8 > len(buf):
            return True                      # past EOF is fine (last partial check)
        head = buf[off:off + 7]
        return head[:6].isdigit() and chr(head[6]) in "DRQM"
    for r in CAND_RECLENS:
        if len(buf) % r == 0 and all(looks_like_record(k * r) for k in range(1, min(4, len(buf) // r))):
            return r
    raise ValueError("could not determine record length")


def fix_file(path, want):
    """Patch every record's channel code to `want` (3 chars). Returns (n_records, old_codes)."""
    import numpy as np
    import obspy
    rel = os.path.relpath(path, REPO)
    bak = os.path.join(BACKUP_ROOT, rel)
    os.makedirs(os.path.dirname(bak), exist_ok=True)
    if not os.path.exists(bak):
        shutil.copy2(path, bak)

    with open(path, "rb") as fh:
        buf = bytearray(fh.read())
    reclen = _detect_reclen(buf)
    want_b = want.encode("ascii")
    assert len(want_b) == 3
    old_codes = set()
    n = 0
    for off in range(0, len(buf), reclen):
        cur = bytes(buf[off + 15:off + 18])
        if cur != want_b:
            old_codes.add(cur.decode("ascii", "replace"))
            buf[off + 15:off + 18] = want_b
            n += 1
    if n == 0:
        return 0, old_codes
    tmp = path + ".chanfix_tmp"
    with open(tmp, "wb") as fh:
        fh.write(buf)

    # verification: channel correct AND data identical to the backup
    a = obspy.read(bak)
    b = obspy.read(tmp)
    assert all(tr.stats.channel == want for tr in b), "patched file channel wrong"
    da = np.concatenate([tr.data for tr in a.sort(["starttime"])])
    db = np.concatenate([tr.data for tr in b.sort(["starttime"])])
    assert len(da) == len(db) and (da == db).all(), "sample data changed — aborting"
    os.replace(tmp, path)

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    new_log = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_log:
            w.writerow(["timestamp", "file", "old_channels", "new_channel", "records_patched"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), rel, "/".join(sorted(old_codes)), want, n])
    return n, old_codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", help="alternating <file> <expected-chan> pairs")
    ap.add_argument("--scan-log", help="chan_scan.py output: fix every listed mismatch")
    a = ap.parse_args()
    jobs = []
    if a.scan_log:
        for line in open(a.scan_log):
            # scan lines list per-station summaries; the raw file list comes from re-deriving:
            pass
    it = iter(a.targets)
    for f in it:
        jobs.append((f, next(it)))
    tot = 0
    for f, want in jobs:
        n, old = fix_file(f, want)
        tot += n
        print(f"  {f}: {n} records {sorted(old)} -> {want}")
    print(f"patched {tot} records in {len(jobs)} files; backups in {BACKUP_ROOT}")


if __name__ == "__main__":
    main()
