#!/usr/bin/env python
"""Aggregate the batch relocation (run after batch_relocate.py):
  (i)   master_metrics.csv      — one row per family: counts, before/after spreads, collapse ratio,
                                  bootstrap 95% medians, mean_cc, recurrence, centroid, status.
  (ii)  master_map_relocated.png — ONE combined PyGMT map of all dt.cc-relocated families on the UF
                                  subregion (fault traces + coastline), each family a distinct colour;
                                  absolute-only families shown as faint grey context.
  (iii) top-N fault-frame thumbnails — pygmt_reloc_map.make_map for the N largest relocated families.

Reuses the per-family run outputs + batch_relocate's enumeration/paths, save_results' spread convention,
pygmt_reloc_map's fault/subregion helpers, and uf_waveform_similarity's colours.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import batch_relocate as B                              # noqa: E402  (paths, family_table, reloc/sum helpers)
import pygmt_reloc_map as PM                            # noqa: E402  (_ufc, _plot_faults, make_map)
sys.path.insert(0, B.PQ); sys.path.insert(0, B.PIPE)
from pipeline.core import sumio                          # noqa: E402
from obspy.geodetics.base import gps2dist_azimuth        # noqa: E402
from uflib import uf_waveform_similarity as wf                      # noqa: E402

MASTER = os.path.join(HERE, "master_metrics.csv")        # reassigned per band in main()
MASTER_MAP = os.path.join(HERE, "master_map_relocated.png")
FAILURES = os.path.join(HERE, "failures.csv")
RELOCATED = ("done", "done_cached")


def _spread(df):
    """RMS horizontal spread (m) about the centroid + depth std (m). Matches build_compare_nb/spread."""
    if df is None or len(df) < 2:
        return (np.nan, np.nan)
    clat, clon = df.lat.mean(), df.lon.mean()
    h = np.array([gps2dist_azimuth(clat, clon, la, lo)[0] for la, lo in zip(df.lat, df.lon)])
    return float(np.sqrt((h ** 2).mean())), float(df.depth.std() * 1000.0)


def _read_sum(slug):
    p = B.sum_path(slug)
    try:
        return sumio.read_sum(p) if os.path.exists(p) else None
    except Exception:
        return None


def _read_reloc(slug):
    try:
        return sumio.read_reloc(B.reloc_path(slug)) if B.n_reloc(slug) > 0 else None
    except Exception:
        return None


def _boot_medians(slug):
    p = os.path.join(B.RUNS, slug, "2.HypoDD", "02.dt.cc", "bootstrap_errors.csv")
    if not os.path.exists(p):
        return (np.nan, np.nan)
    try:
        bb = pd.read_csv(p, comment="#")
        return float(np.nanmedian(np.hypot(bb.ex95, bb.ey95))), float(np.nanmedian(bb.ez95))
    except Exception:
        return (np.nan, np.nan)


def _r(x):
    return int(round(x)) if (x is not None and np.isfinite(x)) else np.nan


def master_table(rep, manifest):
    rows = []
    for r in rep.itertuples():
        fid = int(r.cluster); slug = B.slug_for(fid)
        mrow = manifest.get(fid, {})
        A, D = _read_sum(slug), _read_reloc(slug)
        ah, az = _spread(A); dh, dz = _spread(D)
        bh, bz = _boot_medians(slug)
        rows.append(dict(
            id=fid, n=int(r.n), n_relocated=(len(D) if D is not None else 0),
            status=mrow.get("status", "not_run"),
            stage_failed=mrow.get("stage_failed", ""),
            note=(mrow.get("error_msg", "") or "")[:140],
            mean_cc=round(float(r.mean_cc), 3),
            lat_c=round(float(r.lat_c), 5), lon_c=round(float(r.lon_c), 5),
            depth_med_km=round(float(r.depth_med), 2),
            abs_spread_horiz_m=_r(ah), abs_spread_depth_m=_r(az),
            dtcc_spread_horiz_m=_r(dh), dtcc_spread_depth_m=_r(dz),
            collapse_ratio=(round(dh / ah, 3) if (np.isfinite(ah) and np.isfinite(dh) and ah > 0) else np.nan),
            boot_horiz95_m=_r(bh), boot_depth95_m=_r(bz),
            t_first=str(r.t_first), t_last=str(r.t_last), span_days=round(float(r.span_days), 1),
            recur_med_days=(round(float(r.recur_med_days), 1) if pd.notna(r.recur_med_days) else np.nan),
            daytime_frac=round(float(r.daytime_frac), 2), rayleigh_p=round(float(r.rayleigh_p), 3)))
    df = pd.DataFrame(rows)
    df.to_csv(MASTER, index=False)
    return df


def master_map(rep, manifest):
    import pygmt
    ufc = PM._ufc()
    ids = [int(r.cluster) for r in rep.itertuples()
           if manifest.get(int(r.cluster), {}).get("status") in RELOCATED]
    cols = wf.cluster_colors(ids)                       # dict id -> rgba (size order)
    sub = ufc.SUBREGION; pad = 0.02
    region = [sub[0] - pad, sub[1] + pad, sub[2] - pad, sub[3] + pad]
    fig = pygmt.Figure()
    pygmt.config(FORMAT_GEO_MAP="ddd.xx", MAP_FRAME_TYPE="plain",
                 FONT_TITLE="15p,Helvetica-Bold", FONT_ANNOT_PRIMARY="9p")
    bandstr = f"{B.BAND[0]}-{B.BAND[1]}"
    fig.basemap(region=region, projection="M20c",
                frame=[f"WSne+tUlsan multiplet families - dt.cc relocated ({bandstr} Hz, {len(ids)} families)",
                       "xa0.1f0.05", "ya0.1f0.05"])
    fig.coast(land="245", water="220/233/245", shorelines="0.4p,gray60")
    PM._plot_faults(fig, ufc, pen="0.8p,black")
    for r in rep.itertuples():                          # absolute-only families = faint grey context
        fid = int(r.cluster)
        if manifest.get(fid, {}).get("status") == "absolute_only":
            A = _read_sum(B.slug_for(fid))
            if A is not None:
                fig.plot(x=A.lon, y=A.lat, style="c0.05c", fill="gray75")
    for fid in ids:                                     # relocated families, coloured by family
        D = _read_reloc(B.slug_for(fid))
        if D is not None:
            fig.plot(x=D.lon, y=D.lat, style="c0.09c", fill=wf._gmt_rgb(cols[fid]), pen="0.2p,black")
    bl, ba = ufc._subregion_box(sub)
    fig.plot(x=bl, y=ba, pen="1.2p,blue")
    fig.savefig(MASTER_MAP, dpi=250)
    print(f"wrote {MASTER_MAP}  ({len(ids)} relocated families)")


def thumbnails(rep, manifest, topn):
    ids = [int(r.cluster) for r in rep.itertuples()
           if manifest.get(int(r.cluster), {}).get("status") in RELOCATED][:topn]
    for fid in ids:
        try:
            PM.make_map(B.slug_for(fid), outdir=B.outdir_for(fid))
            print(f"  thumbnail family{fid}{B.BT}")
        except Exception as e:                          # noqa: BLE001
            print(f"  thumbnail family{fid} skipped: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topn", type=int, default=10, help="number of largest relocated families to thumbnail")
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--no-thumbs", action="store_true")
    ap.add_argument("--band", default="5-25", help="clustering band, e.g. 5-25 (default) or 5-15")
    a = ap.parse_args()

    global MASTER, MASTER_MAP, FAILURES                  # band-tag every output; configure the driver too
    B.BAND = tuple(int(x) for x in a.band.split("-")); B.BT = B._bt(a.band)
    B.MANIFEST = os.path.join(HERE, f"batch_manifest{B.BT}.csv")
    MASTER = os.path.join(HERE, f"master_metrics{B.BT}.csv")
    MASTER_MAP = os.path.join(HERE, f"master_map_relocated{B.BT}.png")
    FAILURES = os.path.join(HERE, f"failures{B.BT}.csv")

    rep = B.family_table()
    manifest = B.load_manifest()
    df = master_table(rep, manifest)
    vc = df.status.value_counts().to_dict()
    print(f"{os.path.basename(MASTER)}: {len(df)} families | status {vc}")
    rel = df[df.status.isin(RELOCATED)]
    if len(rel):
        print(f"  relocated: {len(rel)} | median collapse_ratio {rel.collapse_ratio.median():.2f} | "
              f"median dt.cc horiz spread {rel.dtcc_spread_horiz_m.median():.0f} m | "
              f"median boot 95% horiz {rel.boot_horiz95_m.median():.0f} m")
    # explicit tracking of the families that did NOT fully relocate (absolute-only / failed / not-run)
    bad = df[~df.status.isin(RELOCATED)]
    if len(bad):
        bad[["id", "n", "status", "stage_failed", "note"]].to_csv(FAILURES, index=False)
        print(f"\n  {len(bad)} family/ies NOT fully relocated (-> {os.path.basename(FAILURES)}):")
        print("   " + bad[["id", "n", "status", "stage_failed"]].to_string(index=False).replace("\n", "\n   "))
    if not a.no_map:
        master_map(rep, manifest)
    if not a.no_thumbs:
        thumbnails(rep, manifest, a.topn)


if __name__ == "__main__":
    main()
