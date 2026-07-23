#!/usr/bin/env python
"""Generate plot_dtct_result.ipynb — a snapshot of the whole-subregion HypoDD **dt.ct** relocation
(catalog differential times) vs the absolute HypoInverse(kim2011) starting locations, before the
all-pairs dt.cc finishes. Map view (PyGMT, full-res coast + UF faults), depth sections, and the
absolute->dt.ct collapse statistics. Re-point RELOC to 02.dt.cc/hypoDD.reloc to re-use for dt.cc."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md("""# Ulsan-Fault whole-subregion HypoDD — dt.ct relocation snapshot

The catalog-differential-time (**dt.ct**) HypoDD relocation of the whole UF box as one cluster, against
the absolute **HypoInverse (kim2011)** starting locations. This is the dt.ct stage only (the all-pairs
**dt.cc** cross-correlation run is still in progress) — so expect modest tightening here; dt.cc is what
collapses the repeating families. Events absent from `hypoDD.reloc` are dt.ct singletons (no qualifying
catalog-distance link) and keep their absolute location.""")

co("""import os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib as mpl, matplotlib.font_manager as fm

PQ   = "/home/msseo/works/15.PocketQuake"
PIPE = os.path.join(PQ, "external", "korea-cluster-relocation")
HYPO = "/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv"
for p in (PQ, PIPE, HYPO):
    if p not in sys.path: sys.path.insert(0, p)
from pipeline.core import sumio
import uf_cluster as ufc

# Helvetica for plot text (graceful fallback)
_avail = {f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica", "Arial", "Nimbus Sans", "TeX Gyre Heros", "DejaVu Sans"):
    if _f in _avail:
        mpl.rcParams["font.family"] = _f; break
mpl.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})

RUN  = os.path.join(PIPE, "pipeline", "runs", "uf_subregion_reuse")
SUM  = os.path.join(RUN, "1.HypoInv", "kim2011", "uf_subregion_reuse.sum")
RELOC = os.path.join(RUN, "2.HypoDD", "01.dt.ct", "hypoDD.reloc")   # dt.ct stage
print("sum  :", os.path.exists(SUM), SUM)
print("reloc:", os.path.exists(RELOC), RELOC)""")

md("""## 1 · Load absolute + dt.ct relocations, match by event id""")
co("""A = sumio.read_sum(SUM)                 # absolute HypoInverse (kim2011)
D = sumio.read_reloc(RELOC)            # dt.ct HypoDD
A = A.set_index("id"); D = D.set_index("id")
common = A.index.intersection(D.index)
M = pd.DataFrame({
    "lon0": A.lon, "lat0": A.lat, "dep0": A.depth,
    "lon1": D.lon, "lat1": D.lat, "dep1": D.depth,
    "nctp": D.nctp, "ncts": D.ncts, "cid": D.cid,
}).loc[common]
print(f"absolute: {len(A)} events | dt.ct relocated: {len(D)} | matched: {len(M)}")
print(f"dt.ct singletons (not relocated): {len(A) - len(D)}")
print(f"HypoDD clusters (cid): {M.cid.nunique()}  "
      f"(largest {int(M.cid.value_counts().iloc[0])} events)")""")

md("""## 2 · Map view — absolute vs dt.ct (PyGMT, full-resolution coast + UF faults)""")
co("""import pygmt
sub = ufc.SUBREGION; pad = 0.02
region = [sub[0]-pad, sub[1]+pad, sub[2]-pad, sub[3]+pad]
dmin, dmax = float(np.nanmin([M.dep0.min(), M.dep1.min()])), float(np.nanmax([M.dep0.max(), M.dep1.max()]))

# fault traces as ONE no-label polyline (avoids the empty "Fault trace" legend box ufc.plot_faults adds)
_fx, _fy = [], []
for _s in ufc.read_fault_segments():
    _fx.extend(list(_s[:, 1]) + [np.nan]); _fy.extend(list(_s[:, 0]) + [np.nan])
_fx, _fy = np.array(_fx), np.array(_fy)

fig = pygmt.Figure()
pygmt.config(FORMAT_GEO_MAP="ddd.xx", MAP_FRAME_TYPE="plain",
             FONT_TITLE="13p,Helvetica-Bold", FONT_ANNOT_PRIMARY="8p")
pygmt.makecpt(cmap="viridis", series=[dmin, dmax], reverse=True)
with fig.subplot(nrows=1, ncols=2, figsize=("24c", "12c"), margins=["1.0c", "0.5c"],
                 sharey="l"):
    for j, (lo, la, ttl) in enumerate([(M.lon0, M.lat0, "Absolute - HypoInverse (kim2011)"),
                                       (M.lon1, M.lat1, "dt.ct - HypoDD")]):
        with fig.set_panel(j):
            sides = "WSne" if j == 0 else "wSnE"
            fig.basemap(region=region, projection="M11.5c",
                        frame=[f"{sides}+t{ttl}", "xa0.1f0.05", "ya0.1f0.05"])
            fig.coast(region=region, projection="M11.5c", land="245",
                      water="220/233/245", shorelines="0.4p,gray60", resolution="f")
            fig.plot(x=_fx, y=_fy, pen="0.8p,black")          # faults, no legend label
            fig.plot(x=lo, y=la, style="c0.06c", fill=(M.dep0 if j == 0 else M.dep1),
                     cmap=True, pen="0.15p,black", transparency=20)
            bl, ba = ufc._subregion_box(sub); fig.plot(x=bl, y=ba, pen="1.0p,blue")
            fig.basemap(map_scale="jTR+w5k+o0.2c/-0.7c")      # plain 5 km bar, lifted above frame (clears blue box)
            fig.text(text=f"({'ab'[j]})", position="TL", offset="0.1c/1.0c",
                     no_clip=True, font="15p,Helvetica")        # label raised further above the frame (not bold)
fig.colorbar(position="JBC+w9c/0.35c+h+o0c/0.9c", frame=["xaf+lDepth (km)"], cmap=True)
fig.show(width=2000)""")

md("""## 3 · Depth sections — longitude / latitude vs depth (before grey, after coloured by depth)""")
co("""fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), dpi=120)
for a, (x0, x1, lab) in zip(ax, [(M.lon0, M.lon1, "Longitude (°E)"),
                                 (M.lat0, M.lat1, "Latitude (°N)")]):
    a.scatter(x0, M.dep0, s=6, c="0.7", edgecolor="none", label="Absolute (HypoInverse)", zorder=1)
    sc = a.scatter(x1, M.dep1, s=8, c=M.dep1, cmap="viridis_r", edgecolor="k",
                   linewidth=0.15, label="dt.ct (HypoDD)", zorder=2)
    a.set(xlabel=lab, ylabel="Depth (km)"); a.invert_yaxis(); a.legend(fontsize=8, loc="lower right")
ax[0].set_title("Longitude–depth section"); ax[1].set_title("Latitude–depth section")
fig.tight_layout(); plt.show()""")

md("""## 4 · How far did events move, and did the cloud tighten?

Per-event horizontal + vertical shift from the absolute to the dt.ct location, and the depth-distribution
change. dt.ct uses *catalog-distance* links, so it removes coarse scatter but does not collapse repeaters
the way dt.cc will.""")
co("""from obspy.geodetics.base import gps2dist_azimuth
dh = np.array([gps2dist_azimuth(la0, lo0, la1, lo1)[0]
               for lo0, la0, lo1, la1 in zip(M.lon0, M.lat0, M.lon1, M.lat1)])   # m
dz = (M.dep1 - M.dep0).to_numpy() * 1000.0                                       # m

def _rms_spread(lon, lat, dep):
    clat, clon = lat.mean(), lon.mean()
    h = np.array([gps2dist_azimuth(clat, clon, a, o)[0] for a, o in zip(lat, lon)])
    return np.sqrt((h**2).mean()), dep.std() * 1000.0
sh0, sz0 = _rms_spread(M.lon0, M.lat0, M.dep0)
sh1, sz1 = _rms_spread(M.lon1, M.lat1, M.dep1)

fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=120)
ax[0].hist(dh, bins=40, color="steelblue", edgecolor="k", linewidth=0.3)
ax[0].axvline(np.median(dh), color="k", ls="--", lw=1, label=f"median {np.median(dh):.0f} m")
ax[0].set(xlabel="Horizontal shift |absolute → dt.ct| (m)", ylabel="Events",
          title="Epicentral shift"); ax[0].legend(fontsize=8)
ax[1].hist(dz, bins=40, color="indianred", edgecolor="k", linewidth=0.3)
ax[1].axvline(np.median(dz), color="k", ls="--", lw=1, label=f"median {np.median(dz):+.0f} m")
ax[1].set(xlabel="Depth shift (m, + = deeper)", ylabel="Events", title="Depth shift"); ax[1].legend(fontsize=8)
ax[2].hist(M.dep0, bins=30, color="0.7", label="Absolute", alpha=0.8)
ax[2].hist(M.dep1, bins=30, histtype="step", color="tab:green", lw=1.6, label="dt.ct")
ax[2].set(xlabel="Depth (km)", ylabel="Events", title="Depth distribution"); ax[2].legend(fontsize=8)
fig.tight_layout(); plt.show()

print(f"RMS horizontal spread about centroid:  absolute {sh0:,.0f} m  ->  dt.ct {sh1:,.0f} m")
print(f"Depth std:                              absolute {sz0:,.0f} m  ->  dt.ct {sz1:,.0f} m")
print(f"Median per-event shift:  horizontal {np.median(dh):,.0f} m,  vertical {np.median(dz):+,.0f} m")""")

nb.cells = C
out = "/home/msseo/works/02.Ulsan_Fault_detection/analysis/uf_subregion_hypodd/plot_dtct_result.ipynb"
nbf.write(nb, out); print("wrote", out, len(C), "cells")
