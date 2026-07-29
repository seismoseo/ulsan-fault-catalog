#!/usr/bin/env python
"""GJ-inclusive offline scaffold for a UF whole-box cluster (year-general; --slug/--catalog/--station-table).

Mirrors analysis/relocation/scaffold_offline.py, but the station roster comes from OUR per-year station cache
(KS + KG + GJ — the whole point is the GJ temporary array in 2016) instead of the KS/KG-only station_table,
and networks=("KS","KG","GJ") so GJ survives into the pipeline's station master. The caller (run_picker_reloc.py)
passes the year-correct --slug, --catalog, and --station-table; defaults resolve to the 2016 paths for
back-compat. (Filename kept as scaffold_2016.py; it is year-general.) Run with PYTHONPATH = PocketQuake:pipeline.
"""
import os, shutil, sys, argparse
import pandas as pd
from pocketquake.scaffold import ClusterSpec, register_cluster, write_cluster_module

ROOT = "/home/msseo/works/02.Ulsan_Fault_detection/detection_test/reloc_2016_uf"


_XCORR_KEYS = ("interp_hz", "fmin", "fmax", "cc_threshold", "pre", "post", "margin")
_XCORR_MARK = "# ufpipe xcorr overrides (appended by scaffold --xcorr; do not edit by hand)"


def parse_xcorr(s):
    """Parse 'interp_hz=1000,fmin=5,fmax=20,cc_threshold=0.7,pre=0.5,post=0.5,margin=0.5' into a dict."""
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        k = k.strip()
        if k not in _XCORR_KEYS:
            raise SystemExit(f"--xcorr: unknown key {k!r} (valid: {_XCORR_KEYS})")
        out[k] = int(float(v)) if k == "interp_hz" else float(v)
    return out


def apply_xcorr_overrides(module_path, xc):
    """Append a CONFIG=replace(...) xcorr-override block to the generated cluster module (idempotent:
    any previous override block is stripped first). fmin/fmax map onto the engine's bandpass tuple."""
    src = open(module_path).read()
    if _XCORR_MARK in src:                                  # strip a previous block (marker to EOF)
        src = src[:src.index(_XCORR_MARK)].rstrip() + "\n"
    upd = {k: v for k, v in xc.items() if k in ("interp_hz", "cc_threshold", "pre", "post", "margin")}
    if "fmin" in xc or "fmax" in xc:
        upd["bandpass"] = (xc.get("fmin", 5.0), xc.get("fmax", 20.0))
    kv = ", ".join(f"{k}={v!r}" for k, v in upd.items())
    src += f"\n{_XCORR_MARK}\nCONFIG = replace(CONFIG, xcorr=dict(CONFIG.xcorr, {kv}))\n"
    open(module_path, "w").write(src)
    print(f"  xcorr overrides -> {upd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="uf_2016")
    ap.add_argument("--catalog", default=os.path.join(ROOT, "catalog_kma.csv"))
    ap.add_argument("--station-table", default=os.path.join(ROOT, "station_table", "stations_2016.csv"),
                    help="year-general station table (Network,Code,Latitude,Longitude,Elevation)")
    ap.add_argument("--xcorr", default=None,
                    help="dt.cc cross-correlation overrides, e.g. "
                         "'interp_hz=1000,fmin=5,fmax=20,cc_threshold=0.7,pre=0.5,post=0.5,margin=0.5' "
                         "(defaults = the engine's validated values)")
    a = ap.parse_args()
    SLUG = a.slug
    epi = tuple(float(x) for x in "35.7539,129.3804".split(","))
    rb = tuple(float(x) for x in "35.551,35.949,129.200,129.599".split(","))
    spec = ClusterSpec(name=SLUG, region=SLUG,
                       catalog_csv=a.catalog,
                       epicenter=epi, region_bounds=rb,
                       networks=("KS", "KG", "GJ"),          # <-- GJ included
                       # LSQR, never SVD: our recompiled hypoDD (MAXDATA=15M / MAXEVE=6500) sizes
                       # the SVD workspace by its huge static arrays, so an SVD solve grinds for
                       # HOURS even on a ~90-event cluster (2012: 4.7 h in the engine's internal
                       # dtcc baseline). 2016 only escaped via the MAXDATA0-overflow LSQR fallback.
                       # The driver's adaptive kim2011/ISTART=2 LSQR run is the published result;
                       # this makes the engine's internal baseline take the same fast path.
                       dtct_isolv=2,
                       wf_backend="stp", loc_backend="hypoinverse", reloc_backend="hypodd")
    src = spec.src_root
    for sub in ("event_catalog", "station_table", "stp_download"):
        os.makedirs(os.path.join(src, sub), exist_ok=True)
    shutil.copyfile(spec.catalog_csv, os.path.join(src, "event_catalog", "event_catalog.csv"))
    # station roster from OUR cache (Network,Code,Latitude,Longitude,Elevation), split per network
    allsta = pd.read_csv(a.station_table)
    cols = ["Network", "Code", "Latitude", "Longitude", "Elevation"]
    for net in spec.networks:
        allsta[allsta.Network == net][cols].reset_index(drop=True).to_csv(
            os.path.join(src, "station_table", f"{net}_station.csv"), index=False)
    mod = write_cluster_module(spec); register_cluster(spec)
    if a.xcorr:
        apply_xcorr_overrides(mod, parse_xcorr(a.xcorr))
    print(f"scaffolded {SLUG} -> {src}")
    print(f"  roster {len(allsta)} (" + " + ".join(f"{int((allsta.Network==n).sum())} {n}" for n in spec.networks)
          + f"); module {mod}")


if __name__ == "__main__":
    main()
