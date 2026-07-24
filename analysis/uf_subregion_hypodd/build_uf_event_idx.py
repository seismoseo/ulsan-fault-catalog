#!/usr/bin/env python
"""Build the canonical, event_idx-keyed UF event-waveform store — the clean-cut fix that makes the
fragile timestamp matching obsolete.

For every UF-box event in the clean Heo catalog (which now carries the frozen `event_idx` = master row
id), we populate:

    <STORE>/<event_idx>/<event_idx>.<NET>.<STA>.<CHAN>.sac     (symlinks, renamed prefix)
    <STORE>/<event_idx>/<event_idx>_picks.csv

Source resolution is EXACT, never fuzzy:
  1. Replicate event_sac_export.load_catalog's id rule on the *master* (sort by time + same-second
     suffix b/c/…) so each event_idx maps to the exact `event_id` the original export used.
  2. If that `event_id` dir exists under event_waveforms_ulsanfault OR event_waveforms_blastclean
     (content-correct SACs already cut from the continuous archive) -> SYMLINK with event_idx-renamed
     filenames. (handles same-second doublets correctly.)
  3. Otherwise (origin time re-solved across versions, or never exported) -> RE-EXPORT from the
     continuous archive keyed by event_idx (event_sac_export.export_event with event_id=str(idx)),
     so the lost events come back exactly, with no timestamp guessing.

Writes an audit CSV (every event: event_idx, time, source=symlink/reexport/FAILED, n_sac).
Run:  python build_uf_event_idx.py
"""
import os, sys, glob, shutil
import numpy as np, pandas as pd

# Restructured 2026-07: KS_KG/HypoInv -> data/hypoinv; KS_KG/local_magnitudes -> analysis/local_magnitudes;
# continuous station dirs live at KS_KG/ root; the phasenet_plus pyocto/station_table live under outputs/models/.
REPO  = "/home/msseo/works/02.Ulsan_Fault_detection"
HYPO  = f"{REPO}/data/hypoinv"
MASTER= f"{HYPO}/catalog_phasenet_plus_2010_2024_blastclean.csv"
CLEAN = f"{REPO}/analysis/local_magnitudes/catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo_clean.csv"
WF_ROOTS = (f"{HYPO}/event_waveforms_ulsanfault", f"{HYPO}/event_waveforms_blastclean")
STORE = f"{HYPO}/event_waveforms_ufidx"
UF_BOX = (129.25, 129.55, 35.60, 35.90)
# re-export inputs (the phasenet_plus whole-year run this catalog was built from)
CONT  = f"{REPO}/KS_KG"
PYO   = f"{REPO}/outputs/models/phasenet_plus/pyocto"
STA   = f"{REPO}/outputs/models/phasenet_plus/station_table"
PICKS = f"{REPO}/outputs/picks"
HERE  = os.path.dirname(os.path.abspath(__file__))


def master_event_ids():
    """event_idx -> exact event_id the SAC exporter used (same-second suffix rule), from the master."""
    from uflib import event_sac_export as ese
    m = pd.read_csv(MASTER)
    assert "event_idx" in m.columns, "master must carry frozen event_idx (run the freeze step first)"
    m["time"] = pd.to_datetime(m["time"], utc=True)
    m = m.sort_values("time").reset_index(drop=True)
    base = m["time"].apply(ese.event_id_from_time)
    k = base.groupby(base).cumcount()
    m["event_id"] = [b if kk == 0 else f"{b}{chr(ord('a')+kk)}" for b, kk in zip(base, k)]
    return dict(zip(m["event_idx"].astype(int), m["event_id"])), dict(zip(m["event_idx"].astype(int), m["time"]))


def find_src(event_id):
    for root in WF_ROOTS:
        d = os.path.join(root, event_id)
        if os.path.isdir(d) and any(f.endswith(".sac") for f in os.listdir(d)):
            return d
    return None


def symlink_event(idx, src):
    dst = os.path.join(STORE, str(idx)); os.makedirs(dst, exist_ok=True)
    n = 0
    for f in glob.glob(os.path.join(src, "*.sac")):
        parts = os.path.basename(f).split(".")          # {eid}.{NET}.{STA}.{CHAN}.sac
        if len(parts) < 5:
            continue
        new = ".".join([str(idx)] + parts[1:])           # {event_idx}.{NET}.{STA}.{CHAN}.sac
        link = os.path.join(dst, new)
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(f), link); n += 1
    # picks: copy as {event_idx}_picks.csv (content has no eid; stage.py sets Event_ID from members)
    pk = glob.glob(os.path.join(src, "*_picks.csv"))
    if pk:
        shutil.copyfile(pk[0], os.path.join(dst, f"{idx}_picks.csv"))
    return n


def main():
    os.makedirs(STORE, exist_ok=True)
    eid_of, time_of = master_event_ids()
    clean = pd.read_csv(CLEAN)
    box = clean[(clean.lon >= UF_BOX[0]) & (clean.lon <= UF_BOX[1]) &
                (clean.lat >= UF_BOX[2]) & (clean.lat <= UF_BOX[3])].copy()
    box["event_idx"] = box["event_idx"].astype(int)
    print(f"UF-box clean events: {len(box)}")

    reexport = []
    audit = []
    n_sym = 0
    for _, r in box.iterrows():
        idx = int(r.event_idx); eid = eid_of.get(idx)
        src = find_src(eid) if eid else None
        if src is not None:
            n = symlink_event(idx, src)
            audit.append((idx, str(r.time), "symlink", n)); n_sym += 1
        else:
            reexport.append(r)
            audit.append((idx, str(r.time), "reexport_pending", 0))
    print(f"  symlinked (exact existing SACs): {n_sym}")
    print(f"  need re-export from continuous : {len(reexport)}")

    # ---- re-export the non-exact (drifted / never-exported) events, keyed by event_idx ----
    if reexport:
        from uflib import event_sac_export as ese
        from obspy import UTCDateTime
        done = {}
        years = sorted({pd.to_datetime(r.time, utc=True).year for r in reexport})
        sta_by_year = {y: ese.load_stations_for_year(STA, y) for y in years}
        for r in reexport:
            idx = int(r.event_idx); o = UTCDateTime(pd.to_datetime(r.time, utc=True).to_pydatetime())
            row = pd.Series(dict(event_id=str(idx), origin_utc=o,
                                 lat=float(r.lat), lon=float(r.lon), depth=float(r.depth)))
            try:
                summ = ese.export_event(row, PICKS, CONT, sta_by_year[o.year], STORE,
                                        pyocto_root=PYO, pyocto_velmodel="kim1983")
                done[idx] = int(summ.get("n_sac", 0))
            except Exception as exc:
                done[idx] = -1; print(f"    reexport FAILED idx {idx}: {type(exc).__name__}: {exc}")
        # patch audit
        audit = [(i, t, ("reexport" if done.get(i, 0) > 0 else "FAILED") if s == "reexport_pending" else s,
                  done.get(i, n) if s == "reexport_pending" else n) for (i, t, s, n) in audit]

    A = pd.DataFrame(audit, columns=["event_idx", "time", "source", "n_sac"]).sort_values("event_idx")
    A.to_csv(os.path.join(HERE, "uf_event_idx_audit.csv"), index=False)
    ok = A[A.n_sac > 0]
    print(f"\nSTORE {STORE}")
    print(f"  events with SACs: {len(ok)} / {len(A)}  (symlink {int((A.source=='symlink').sum())}, "
          f"reexport {int((A.source=='reexport').sum())}, FAILED {int((A.source=='FAILED').sum())})")
    print(f"  audit -> {HERE}/uf_event_idx_audit.csv")
    # sanity: the M3.99 (2014-09-23) must be present
    big = A[A.time.str.startswith("2014-09-23")]
    print(f"  2014-09-23 events in store: {len(big)} -> {big[['event_idx','source','n_sac']].to_dict('records')}")


if __name__ == "__main__":
    main()
