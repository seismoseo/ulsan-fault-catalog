#!/usr/bin/env python
"""Generate plot_heo2024_catalog.ipynb — the published Heo (2024) GHBSN catalog on the SAME UF
subregion extent and in the SAME PyGMT style as plot_dtct_result.ipynb (full-res coast + UF faults,
depth-coloured circles, blue subregion box, plain 5 km scale bar lifted above the frame). Single panel
(one catalog), for visual comparison against our HypoInverse/HypoDD relocation of the same zone."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md("""# Heo (2024) GHBSN catalog — Ulsan-Fault subregion

The **published Heo (2024) GHBSN catalog** plotted on the same extent and in the same style as the
HypoDD dt.ct snapshot (`plot_dtct_result.ipynb`), for a like-for-like visual comparison against our
PhaseNet+ → HypoInverse → HypoDD relocation of the same zone.""")

co("""import os, sys
import numpy as np, pandas as pd
PQ   = "/home/msseo/works/15.PocketQuake"
HYPO = "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/HypoInv"
for p in (PQ, PQ + "/external/korea-cluster-relocation", HYPO):
    if p not in sys.path: sys.path.insert(0, p)
import uf_cluster as ufc

CAT = "/home/msseo/works/07.SeismoStats/catalog/Heo2024_GHBSN_catalog.csv"
sub = ufc.SUBREGION; pad = 0.02
region = [sub[0]-pad, sub[1]+pad, sub[2]-pad, sub[3]+pad]

c = pd.read_csv(CAT)
c = c.rename(columns={"Latitude": "lat", "Longitude": "lon", "Depth": "depth", "Magnitude": "mag"})
inreg = ((c.lon >= region[0]) & (c.lon <= region[1]) & (c.lat >= region[2]) & (c.lat <= region[3]))
inbox = ((c.lon >= sub[0]) & (c.lon <= sub[1]) & (c.lat >= sub[2]) & (c.lat <= sub[3]))
C = c[inreg].reset_index(drop=True)
print(f"Heo 2024 GHBSN: {len(c)} events total | {int(inbox.sum())} in UF box | {len(C)} in map extent")
print(f"depth (UF box): {c[inbox].depth.min():.1f}-{c[inbox].depth.max():.1f} km, "
      f"mean {c[inbox].depth.mean():.1f} km")""")

md("""## Map view — Heo 2024 GHBSN catalog (same extent + style as the dt.ct map)""")
co("""import pygmt
# match the dt.ct map's depth colour range so the two are directly comparable (Heo has a few
# 40+ km outliers that would otherwise wash out the scale); fall back to Heo percentiles
try:
    from pipeline.core import sumio
    _D = sumio.read_reloc("/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/"
                          "pipeline/runs/uf_subregion_reuse/2.HypoDD/01.dt.ct/hypoDD.reloc")
    dmin, dmax = float(_D.depth.min()), float(_D.depth.max())
except Exception:
    dmin, dmax = float(np.percentile(C.depth, 1)), float(np.percentile(C.depth, 99))
print(f"depth colour range: {dmin:.1f}-{dmax:.1f} km (matched to dt.ct map)")

# fault traces as ONE no-label polyline (avoids the empty "Fault trace" legend box)
_fx, _fy = [], []
for _s in ufc.read_fault_segments():
    _fx.extend(list(_s[:, 1]) + [np.nan]); _fy.extend(list(_s[:, 0]) + [np.nan])
_fx, _fy = np.array(_fx), np.array(_fy)

fig = pygmt.Figure()
pygmt.config(FORMAT_GEO_MAP="ddd.xx", MAP_FRAME_TYPE="plain",
             FONT_TITLE="13p,Helvetica-Bold", FONT_ANNOT_PRIMARY="8p")
pygmt.makecpt(cmap="viridis", series=[dmin, dmax], reverse=True)
fig.basemap(region=region, projection="M14c",
            frame=["WSne+tHeo 2024 GHBSN catalog", "xa0.1f0.05", "ya0.1f0.05"])
fig.coast(region=region, projection="M14c", land="245",
          water="220/233/245", shorelines="0.4p,gray60", resolution="f")
fig.plot(x=_fx, y=_fy, pen="0.8p,black")                       # faults, no legend label
fig.plot(x=C.lon, y=C.lat, style="c0.06c", fill=C.depth, cmap=True,
         pen="0.15p,black", transparency=20)
bl, ba = ufc._subregion_box(sub); fig.plot(x=bl, y=ba, pen="1.0p,blue")
fig.basemap(map_scale="jTR+w5k+o0.2c/-0.7c")                   # plain 5 km bar, lifted above frame
fig.colorbar(position="JBC+w9c/0.35c+h+o0c/0.9c", frame=["xaf+lDepth (km)"], cmap=True)
fig.show(width=1200)""")

nb.cells = C
out = "/home/msseo/works/02.Ulsan_Fault_detection/analysis/uf_subregion_hypodd/plot_heo2024_catalog.ipynb"
nbf.write(nb, out); print("wrote", out, len(C), "cells")
