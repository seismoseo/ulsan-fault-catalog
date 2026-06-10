#!/usr/bin/env python
"""Save clean, readable relocation result TABLES (CSV) for the family-738 runs into `family738/`,
merging PocketQuake's raw 24-column `hypoDD.reloc` with the bootstrap 95% errors and the event-ids —
so results are inspectable without parsing the raw HypoDD output.

Writes, under family738/:
  - reloc_f738_reuse.csv / reloc_f738_fresh.csv   per-run relocated catalog
        event_id, time_utc, lat, lon, depth_km, x_m, y_m, z_m, ex95_m, ey95_m, ez95_m, nccp, nccs
  - reloc_compare.csv                              (1) reuse vs (2) fresh per-event offsets
"""
import os
import sys

import numpy as np
import pandas as pd

PQ = "/home/msseo/works/15.PocketQuake"
PIPE = os.path.join(PQ, "external", "korea-cluster-relocation")
sys.path.insert(0, PQ); sys.path.insert(0, PIPE)
from pipeline import config                       # noqa: E402
from pipeline.core import sumio                   # noqa: E402
from obspy.geodetics.base import gps2dist_azimuth  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "family738")
RUNS = os.path.join(PIPE, "pipeline", "runs")


def load(slug):
    rl = sumio.read_reloc(os.path.join(RUNS, slug, "2.HypoDD", "02.dt.cc", "hypoDD.reloc"))
    rl["event_id"] = rl["time"].apply(lambda t: t.strftime("%Y%m%d%H%M%S"))
    rl["time_utc"] = rl["time"].apply(str)
    bpath = os.path.join(RUNS, slug, "2.HypoDD", "02.dt.cc", "bootstrap_errors.csv")
    if os.path.exists(bpath):                      # merge bootstrap 95% half-widths (m) if present
        bb = pd.read_csv(bpath, comment="#")       # first line is a '# bootstrap_errors ...' header
        bb[["ex95", "ey95", "ez95"]] = bb[["ex95", "ey95", "ez95"]].round(1)
        rl = rl.merge(bb[["id", "ex95", "ey95", "ez95"]], on="id", how="left")
    else:
        print(f"  (no bootstrap_errors.csv for {slug} yet — run the bootstrap first)")
    cols = (["event_id", "time_utc", "lat", "lon", "depth", "x", "y", "z"]
            + [c for c in ("ex95", "ey95", "ez95") if c in rl.columns]
            + ["nccp", "nccs", "nctp", "ncts"])
    out = rl[[c for c in cols if c in rl.columns]].copy()
    out.to_csv(os.path.join(OUT, f"reloc_{slug}.csv"), index=False)
    return rl


def main():
    os.makedirs(OUT, exist_ok=True)
    r1, r2 = load("f738_reuse"), load("f738_fresh")
    m = r1.set_index("id").join(r2.set_index("id"), lsuffix="_reuse", rsuffix="_fresh", how="inner")
    m["horiz_offset_m"] = np.round([gps2dist_azimuth(a, b, c, e)[0]
                          for a, b, c, e in zip(m.lat_reuse, m.lon_reuse, m.lat_fresh, m.lon_fresh)]).astype(int)
    m["depth_offset_m"] = ((m.depth_reuse - m.depth_fresh).abs() * 1000.0).round().astype(int)
    cmp = (m.reset_index()[["event_id_reuse", "lat_reuse", "lon_reuse", "depth_reuse",
                            "lat_fresh", "lon_fresh", "depth_fresh", "horiz_offset_m", "depth_offset_m"]]
           .rename(columns={"event_id_reuse": "event_id"}))
    cmp.to_csv(os.path.join(OUT, "reloc_compare.csv"), index=False)
    print(f"wrote reloc_f738_reuse.csv, reloc_f738_fresh.csv, reloc_compare.csv -> {OUT}")
    print(f"  (1)-vs-(2): horiz offset median {cmp.horiz_offset_m.median():.0f} m, "
          f"depth offset median {cmp.depth_offset_m.median():.0f} m  ({len(cmp)} events)")


if __name__ == "__main__":
    main()
