"""Multi-network instrument responses (KS / KG / GJ / NS) — one merged ObsPy Inventory.

The companion of ``stations.py``: that module is the single source of truth for *where* a station
is, this one for *how it responds*. Local magnitude removes the instrument response and simulates a
Wood-Anderson seismograph, so every network that contributes picks must contribute a response too —
otherwise its stations are silently dropped from ML and the magnitude is computed from a biased
subset of the network.

Sources (disclosed):
  * KS + KG : ``config.RESP_MASTER_DIR``  StationXML (``KS_KG_metadata_1.0.2.xml``), plus
              ``config.RESP_FETCHED_DIR/extracted`` SEED RESP for the 7 KS stations the master
              file omits (fetched from NECIS — see responses/README.md).
  * GJ      : ``config.RESP_GJ_DIR``      SEED RESP, 30 stations x 3 channels (MARA server).
  * NS      : ``config.RESP_NS_DIR``      SEED RESP, 200 stations x 3 channels, produced by
              ``obspy-dataless2resp NS-HH.dataless`` (dataless from Dabeen Heo, 2020-06-03).
  * NS N201-N220 : ``config.RESP_NS_DERIVED_DIR`` — **DERIVED, not authoritative**. The MARA
              dataless predates that deployment block. All 200 authoritative NS RESP files are
              byte-identical apart from the station code and the epoch start date, so these are
              clones carrying each station's real first-day-on-disk as the start date. See
              ``write_derived_ns`` for the assumption this rests on and how to re-generate.

Everything is read by ``obspy.read_inventory`` regardless of format, and merged master-first so an
authoritative entry always wins over a fill or a clone for the same (network, station, channel).
"""
import os
import sys
import glob
import shutil
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import stations as _stations

# Stations whose response is cloned rather than measured (see write_derived_ns).
DERIVED_NS_PREFIX = "N2"
DERIVED_MARKER = "#\t\tDERIVED by ufpipe.responses.write_derived_ns"


def _resp_dirs(networks=None, include_derived=True):
    """Ordered (network-scope, directory) pairs to merge, authoritative first."""
    networks = tuple(networks) if networks else config.DETECT_NETWORKS
    out = []
    if {"KS", "KG"} & set(networks):
        out.append(("KS/KG", os.path.join(config.RESP_FETCHED_DIR, "extracted")))
    if "GJ" in networks:
        out.append(("GJ", config.RESP_GJ_DIR))
    if "NS" in networks:
        out.append(("NS", config.RESP_NS_DIR))
        if include_derived:
            out.append(("NS(derived)", config.RESP_NS_DERIVED_DIR))
    return out


def load_inventory(networks=None, include_derived=True, verbose=False):
    """Merged ``obspy.Inventory`` covering `networks` (default config.DETECT_NETWORKS).

    The KS/KG master StationXML is loaded first, then the SEED RESP directories in the order given
    by ``_resp_dirs``. ObsPy keeps every appended entry, and ``Inventory.get_response`` returns the
    FIRST match, so master-first means an authoritative response is never shadowed by a fill."""
    import obspy
    networks = tuple(networks) if networks else config.DETECT_NETWORKS
    inv = obspy.Inventory()
    if {"KS", "KG"} & set(networks):
        pat = os.path.join(config.RESP_MASTER_DIR, "*.xml")
        if glob.glob(pat):
            inv += obspy.read_inventory(pat)
        else:
            warnings.warn(f"no master StationXML under {config.RESP_MASTER_DIR} — KS/KG responses "
                          f"will be missing (it is gitignored; see responses/README.md)", RuntimeWarning)
    for label, d in _resp_dirs(networks, include_derived):
        if not os.path.isdir(d):
            continue
        n = 0
        for f in sorted(glob.glob(os.path.join(d, "RESP.*"))):
            try:
                inv += obspy.read_inventory(f)
                n += 1
            except Exception as exc:                       # noqa: BLE001
                warnings.warn(f"failed to load {f}: {exc}", RuntimeWarning)
        if verbose:
            print(f"  {label:<12s} {n:4d} RESP files from {os.path.relpath(d, config.RESPONSE_DIR)}/")
    return inv


def coverage_report(year, inv=None, networks=None):
    """Per-station response coverage for `year`'s station table.

    Returns a DataFrame ``net, sta, covered, n_channels`` — the check to run before trusting an ML
    catalogue, since a missing response removes a station silently rather than loudly."""
    S = _stations.build_year_table(year, networks=networks)
    inv = inv if inv is not None else load_inventory(networks=networks)
    # Each appended SEED RESP file becomes its OWN Station object (one per channel), so the
    # channel count has to be accumulated across duplicate (net, sta) entries, not overwritten.
    have = {}
    for n in inv:
        for st in n:
            have[(n.code, st.code)] = have.get((n.code, st.code), 0) + len(st)
    rows = [dict(net=r.net, sta=r.sta, covered=(r.net, r.sta) in have,
                 n_channels=have.get((r.net, r.sta), 0)) for _, r in S.iterrows()]
    return pd.DataFrame(rows)


def _first_day_on_disk(sta, archive=None):
    """(year, jday) of the earliest day this NS station has on local disk, or None."""
    archive = archive or config.NS_DIR
    d = os.path.join(archive, sta)
    if not os.path.isdir(d):
        return None
    best = None
    for ch in sorted(os.listdir(d)):
        if not ch.endswith(".D"):
            continue
        for f in os.listdir(os.path.join(d, ch)):
            p = f.split(".")
            if len(p) >= 7 and p[-2].isdigit() and p[-1].isdigit():
                cand = (int(p[-2]), int(p[-1]))
                best = cand if best is None or cand < best else best
        if best:
            break
    return best


def write_derived_ns(out_dir=None, template_sta=None, dry_run=False):
    """Generate SEED RESP for the NS stations the MARA dataless does not cover (N201-N220).

    THE ASSUMPTION, stated plainly: every one of the 200 authoritative NS RESP files is
    byte-identical apart from the station code and the epoch start date, i.e. the whole dense array
    was deployed with one instrument+digitizer configuration. The uncovered block (N201-N220, first
    data 2022.237) postdates the 2020-06-03 dataless, and is assumed to continue that configuration.
    Corroborating but NOT conclusive: those stations record at the same 200 Hz, with 3 components,
    at count amplitudes in the same range as the covered ones. There is no manufacturer record here
    to confirm it. If the array was re-equipped in 2022 these magnitudes carry that error, so treat
    ML from N2xx stations as provisional and drop this directory the moment real metadata arrives.

    Each clone carries the station's OWN first-day-on-disk as the epoch start date (not the
    template's), so ``get_response`` resolves for every real trace and for nothing earlier.
    Idempotent: rewrites the directory from scratch each call."""
    out_dir = out_dir or config.RESP_NS_DERIVED_DIR
    have = {os.path.basename(f).split(".")[2]
            for f in glob.glob(os.path.join(config.RESP_NS_DIR, "RESP.NS.*"))}
    need = set()
    for y in range(2010, 2031):
        S = _stations.build_year_table(y, networks=("NS",))
        need |= set(S.sta)
    missing = sorted(need - have)
    if not missing:
        print("  no uncovered NS stations — nothing to derive")
        return []

    template_sta = template_sta or sorted(have)[0]
    written = []
    for sta in missing:
        day = _first_day_on_disk(sta)
        if day is None:
            warnings.warn(f"{sta}: no data on disk, skipping derived response", RuntimeWarning)
            continue
        start = f"{day[0]},{day[1]:03d}"
        for cha in ("HHE", "HHN", "HHZ"):
            src = os.path.join(config.RESP_NS_DIR, f"RESP.NS.{template_sta}..{cha}")
            if not os.path.exists(src):
                continue
            txt = open(src, errors="ignore").read().replace(template_sta, sta)
            out = []
            for line in txt.splitlines():
                if line.startswith("B052F22"):
                    line = f"B052F22     Start date:  {start}"
                out.append(line)
            body = (f"{DERIVED_MARKER} from RESP.NS.{template_sta}..{cha}\n"
                    f"#\t\tstart date = {sta} first day on disk; NOT authoritative metadata\n"
                    + "\n".join(out) + "\n")
            dst = os.path.join(out_dir, f"RESP.NS.{sta}..{cha}")
            if not dry_run:
                os.makedirs(out_dir, exist_ok=True)
                with open(dst, "w") as fh:
                    fh.write(body)
            written.append(dst)
    print(f"  derived {len(written)} RESP files for {len(missing)} NS stations "
          f"({', '.join(missing[:4])}{' ...' if len(missing) > 4 else ''}) from template {template_sta}")
    return written


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build / check the multi-network response inventory.")
    ap.add_argument("--derive-ns", action="store_true", help="regenerate ns_derived/ for uncovered NS stations")
    ap.add_argument("--coverage", type=int, metavar="YEAR", help="report response coverage for a year")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.derive_ns:
        write_derived_ns(dry_run=a.dry_run)
    if a.coverage:
        inv = load_inventory(verbose=True)
        R = coverage_report(a.coverage, inv=inv)
        print(f"\n  response coverage {a.coverage}: {int(R.covered.sum())}/{len(R)} stations")
        for net, g in R.groupby("net"):
            miss = sorted(g[~g.covered].sta)
            print(f"    {net}: {int(g.covered.sum()):3d}/{len(g):3d}" + (f"   MISSING {miss}" if miss else ""))


if __name__ == "__main__":
    main()
