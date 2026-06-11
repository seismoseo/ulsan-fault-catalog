#!/usr/bin/env python
"""PyGMT subregion map of family 738 (reuse run): the **exact event locations before vs after** the
dt.cc relocation, on a shared extent + shared depth colour scale so the collapse is visible directly.

Per the project convention, spatial maps use **PyGMT** (not matplotlib/cartopy). Two panels at the same
region: (left) absolute HypoInverse(kim2011); (right) dt.cc HypoDD. Circles colour = depth, size =
KMA local magnitude. Saves family738/pygmt_reloc_<slug>.png and returns the path.

Usage:  python pygmt_reloc_map.py [slug]      (default slug f738_reuse)
"""
import os
import sys

import numpy as np
import pygmt

PQ = "/home/msseo/works/15.PocketQuake"
PIPE = os.path.join(PQ, "external", "korea-cluster-relocation")
sys.path.insert(0, PQ); sys.path.insert(0, PIPE)
from pipeline.core import sumio                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "family738")
RUNS = os.path.join(PIPE, "pipeline", "runs")


def _load(slug):
    A = sumio.read_sum(os.path.join(RUNS, slug, "1.HypoInv", "kim2011", f"{slug}.sum"))
    D = sumio.read_reloc(os.path.join(RUNS, slug, "2.HypoDD", "02.dt.cc", "hypoDD.reloc"))
    mag = D.set_index("id")["mag"]                     # reloc carries the KMA magnitude; map onto abs by id
    A = A.assign(mag=A["id"].map(mag).fillna(mag.median()))
    return A, D


def _square(clat, clon, half_km):
    """Square-in-distance [W, E, S, N] box of half-size `half_km` about (clat, clon)."""
    dlat = half_km / 111.32
    dlon = half_km / (111.32 * np.cos(np.radians(clat)))
    return [clon - dlon, clon + dlon, clat - dlat, clat + dlat]


def _ufc():
    """Lazy import of the Ulsan-fault helper (SUBREGION extent + fault traces + plot_faults)."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "HypoInv"))
    import uf_cluster as ufc
    return ufc


def _plot_faults(fig, ufc, pen="0.8p,black"):
    """Overlay the UF fault traces as ONE NaN-separated polyline (no legend label — unlike
    `ufc.plot_faults`, which sets label='Fault trace' and leaves an empty legend box on each panel)."""
    xs, ys = [], []
    for s in ufc.read_fault_segments(ufc.FAULT_TRACE):
        xs.extend(list(s[:, 1]) + [np.nan]); ys.extend(list(s[:, 0]) + [np.nan])
    fig.plot(x=np.array(xs), y=np.array(ys), pen=pen)


def make_map(slug="f738_reuse"):
    A, D = _load(slug)
    os.makedirs(OUT, exist_ok=True)
    ufc = _ufc()

    # zoom extent: square-in-distance about the cloud MEAN (robust to the one southern outlier),
    # sized to include every event, so the collapse is obvious on a common frame
    lat = np.r_[A.lat, D.lat]; lon = np.r_[A.lon, D.lon]
    clat, clon = float(lat.mean()), float(lon.mean())
    rad_km = max(np.hypot((lat - clat) * 111.32, (lon - clon) * 111.32 * np.cos(np.radians(clat))))
    zoom = _square(clat, clon, 1.12 * rad_km)
    zbx = [zoom[0], zoom[1], zoom[1], zoom[0], zoom[0]]; zby = [zoom[2], zoom[2], zoom[3], zoom[3], zoom[2]]
    # regional extent: the whole UF subregion, also square (so all three panels share one height)
    sub = ufc.SUBREGION
    rclat, rclon = (sub[2] + sub[3]) / 2, (sub[0] + sub[1]) / 2
    rhalf = 1.05 * max((sub[3] - sub[2]) / 2 * 111.32, (sub[1] - sub[0]) / 2 * 111.32 * np.cos(np.radians(rclat)))
    reg = _square(rclat, rclon, rhalf)
    dmin, dmax = float(min(A.depth.min(), D.depth.min())), float(max(A.depth.max(), D.depth.max()))

    def sizes(mag):                                    # circle diameter (cm) from local magnitude
        return 0.11 * (np.asarray(mag) + 1.6)

    PANEL, MX = 5.6, 1.3                               # cm: square panel width + inter-panel gap
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE="11p,Helvetica-Bold", FONT_HEADING="13p,Helvetica-Bold",
                 FONT_ANNOT_PRIMARY="7p", FONT_LABEL="9p", MAP_TITLE_OFFSET="5p",
                 MAP_FRAME_TYPE="plain", FORMAT_GEO_MAP="ddd.xxF")
    pygmt.makecpt(cmap="viridis", series=[dmin, dmax], reverse=True)
    with fig.subplot(nrows=1, ncols=3, figsize=(f"{3 * PANEL + 2 * MX}c", f"{PANEL}c"),
                     margins=[f"{MX}c", "0.5c"], autolabel="(a)+o0.1c",
                     title=f"Family 738 ({slug.replace('f738_', '')}) - dt.cc relocation in the Ulsan Fault subregion"):
        # (a) regional framework — whole subregion, coastline + fault traces, zoom box + cluster star
        with fig.set_panel(0):
            fig.basemap(region=reg, projection=f"M{PANEL}c",
                        frame=["WSne+tRegional: UF subregion and faults", "xa0.1f0.05", "ya0.1f0.05"])
            fig.coast(land="245", water="220/233/245", shorelines="0.4p,gray60")
            _plot_faults(fig, ufc)
            fig.plot(x=zbx, y=zby, pen="1.2p,red")                          # zoom-area box
            fig.plot(x=[float(D.lon.mean())], y=[float(D.lat.mean())], style="a0.45c", fill="red", pen="0.5p,black")
            fig.basemap(map_scale="jBL+w10k+o0.3c/0.3c")
        # (b) before, (c) after — shared zoom extent + depth scale; faults overlaid for the same framework
        for j, (df, sides, ttl) in enumerate([(A, "wSne", "Before - HypoInverse (kim2011)"),
                                              (D, "wSnE", "After - dt.cc HypoDD")], start=1):
            with fig.set_panel(j):
                fig.basemap(region=zoom, projection=f"M{PANEL}c",
                            frame=[f"{sides}+t{ttl}", "xa0.02f0.01", "ya0.02f0.01"])
                _plot_faults(fig, ufc)
                fig.plot(x=df.lon, y=df.lat, size=sizes(df.mag), fill=df.depth, cmap=True,
                         style="cc", pen="0.3p,black", transparency=15)
                fig.basemap(map_scale="jBL+w0.5k+o0.3c/0.3c")              # plain scale bar, no box
    fig.colorbar(position="JBC+w7c/0.3c+h+o0c/1.0c", frame=["xaf+lDepth (km) - panels (b), (c)"], cmap=True)
    out = os.path.join(OUT, f"pygmt_reloc_{slug}.png")
    fig.savefig(out, dpi=300)
    print(f"wrote {out}  | zoom {np.round(zoom, 3)}  regional {np.round(reg, 3)}  depth [{dmin:.2f},{dmax:.2f}] km")
    return out


if __name__ == "__main__":
    make_map(sys.argv[1] if len(sys.argv) > 1 else "f738_reuse")
