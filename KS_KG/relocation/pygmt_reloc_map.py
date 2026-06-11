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


def make_map(slug="f738_reuse"):
    A, D = _load(slug)
    os.makedirs(OUT, exist_ok=True)
    # shared, square-in-distance extent centred on the cloud MEAN (robust to the one southern outlier),
    # sized to include every event, so the collapse is obvious on a common frame
    lat = np.r_[A.lat, D.lat]; lon = np.r_[A.lon, D.lon]
    clat, clon = float(lat.mean()), float(lon.mean())
    rad_km = max(np.hypot((lat - clat) * 111.32, (lon - clon) * 111.32 * np.cos(np.radians(clat))))
    half_km = 1.12 * rad_km
    dlat = half_km / 111.32; dlon = half_km / (111.32 * np.cos(np.radians(clat)))
    region = [clon - dlon, clon + dlon, clat - dlat, clat + dlat]
    dmin, dmax = float(min(A.depth.min(), D.depth.min())), float(max(A.depth.max(), D.depth.max()))

    def sizes(mag):                                    # circle diameter (cm) from magnitude
        return 0.11 * (np.asarray(mag) + 1.6)

    fig = pygmt.Figure()
    pygmt.config(FONT_HEADING="14p,Helvetica-Bold", FONT_ANNOT_PRIMARY="7p", FONT_LABEL="9p",
                 MAP_TITLE_OFFSET="6p", MAP_FRAME_TYPE="plain")
    pygmt.makecpt(cmap="viridis", series=[dmin, dmax], reverse=True)
    panels = [(A, f"Before - absolute HypoInverse (kim2011), N={len(A)}"),
              (D, f"After - dt.cc HypoDD relocation, N={len(D)}")]
    with fig.subplot(nrows=1, ncols=2, figsize=("16c", "8c"), margins="0.7c", autolabel="(a)",
                     title=f"Family 738 ({slug.replace('f738_', '')}) - event locations before vs after dt.cc relocation"):
        for j, (df, lab) in enumerate(panels):
            with fig.set_panel(j):
                fig.basemap(region=region, projection="M8c", frame=["WSne", "xa0.005f0.001", "ya0.005f0.001"])
                fig.plot(x=df.lon, y=df.lat, size=sizes(df.mag), fill=df.depth, cmap=True,
                         style="cc", pen="0.3p,black", transparency=15)
                fig.text(position="TC", text=lab, font="8.5p,Helvetica-Bold", offset="0c/-0.3c", no_clip=True)
                fig.basemap(map_scale="jBL+w0.5k+o0.3c/0.3c+f", box="+gwhite@30+p0.3p")
    fig.colorbar(position="JBC+w8c/0.35c+h+o0c/1.0c", frame=["xaf+lDepth (km)"], cmap=True)
    out = os.path.join(OUT, f"pygmt_reloc_{slug}.png")
    fig.savefig(out, dpi=300)
    print(f"wrote {out}  | region {region}  depth [{dmin:.2f},{dmax:.2f}] km")
    return out


if __name__ == "__main__":
    make_map(sys.argv[1] if len(sys.argv) > 1 else "f738_reuse")
