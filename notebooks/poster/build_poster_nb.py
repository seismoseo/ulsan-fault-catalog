#!/usr/bin/env python
"""Generate notebooks/poster/00.Poster_figures.ipynb — poster panels from the ESTABLISHED catalogs.

Data sources (all already computed; nothing here launches heavy processing):
  * OLD full-era catalog (KS/KG, 2010-2024): data/hypoinv/kim2011/UF<year>_filtered.sum
  * OLD whole-box dt.cc relocation (2,722 events): runs/uf_subregion_reuse/.../hypoDD.reloc
  * Homogenized ML: analysis/local_magnitudes/catalog_ml_heo_const.csv
  * NEW-pipeline highlight (2010-2016 complete): outputs/reloc/reloc_2016_uf/results/hypoDD.reloc.dtcc

Panels (each saved as PDF + PNG at poster resolution into notebooks/poster/figs/):
  P1 overview map        — dt.cc-relocated seismicity, stations, UF box (PyGMT)
  P2 space-time          — latitude vs time, absolute + relocated, Gyeongju marked
  P3 magnitude-time      — ML vs time + cumulative count, largest events labeled
  P4 Gyeongju response   — weekly rate 2015-2018 + pre/post 6-month maps (PyGMT)
  P5 new-pipeline 2016   — absolute vs dt.cc for the rebuilt catalog (the methods highlight)

    python notebooks/poster/build_poster_nb.py        # (re)writes the unexecuted notebook

Kernel: base (pygmt). Light on CPU/disk — safe to run alongside detections.
"""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "00.Poster_figures.ipynb")
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Poster figures — Ulsan Fault persistent seismicity

Panels for the poster (print Aug 18) built from the **established** catalogs: the full-era KS/KG
catalog + whole-box dt.cc relocation + homogenized ML, with the rebuilt-pipeline 2016 relocation as
the methods highlight. Every figure saves PDF + PNG into `figs/`.

**Kernel: base.** Safe to run while detections are ongoing (catalog-level plots only).""")

co(r'''# ============================ PARAMETERS ============================
REPO      = "/home/msseo/works/02.Ulsan_Fault_detection"
RUNS      = "/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs"

YEARS     = range(2010, 2025)
UF_BOX    = (129.25, 129.55, 35.60, 35.90)        # lon0, lon1, lat0, lat1 (selection box)
MAP_BOX   = (129.20, 129.58, 35.58, 35.92)        # display box (slightly wider — reloc moves events)
GJ_EQ     = dict(time="2016-09-12T11:32:54", lat=35.77, lon=129.19, mw=5.5)   # Gyeongju mainshock (UTC)
GJ_WINDOW_M = 6                                    # months of elevated response to mark
LARGEST   = [("2014-02-09", 3.5, "2014 ML 3.5"), ("2023-05-22", 4.0, "2023 ML 4.0")]  # label anchors (edit dates if needed)

OLD_SUM   = REPO + "/data/hypoinv/kim2011/UF{year}_filtered.sum"
OLD_RELOC = RUNS + "/uf_subregion_reuse/2.HypoDD/02.dt.cc/hypoDD.reloc"
ML_CSV    = REPO + "/analysis/local_magnitudes/catalog_ml_heo_const.csv"
NEW_2016  = REPO + "/outputs/reloc/reloc_2016_uf/results/hypoDD.reloc.dtcc"
NEW_2016_SUM = REPO + "/outputs/models/phasenet_plus/HypoInv/kim2011/UF2016.sum"
FIGS      = REPO + "/notebooks/poster/figs"

import os, sys, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import pygmt
sys.path.insert(0, REPO + "/src")
from uflib import uf_cluster as uf

try:
    fm.findfont("Helvetica", fallback_to_default=False)
    plt.rcParams["font.family"] = "Helvetica"
except Exception:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 600, "legend.framealpha": 1.0,
                     "legend.facecolor": "white", "axes.titlesize": 11})
os.makedirs(FIGS, exist_ok=True)

def save(fig, name, pygmt_fig=False):
    """Save PDF + PNG at poster resolution."""
    for ext in ("pdf", "png"):
        p = os.path.join(FIGS, f"{name}.{ext}")
        (fig.savefig(p) if pygmt_fig else fig.savefig(p, bbox_inches="tight"))
    print(f"  saved figs/{name}.pdf + .png")''')

md(r"""## Load the established catalogs""")

co(r'''# OLD full-era absolute catalog (QC'd), UF box
frames = []
for y in YEARS:
    p = OLD_SUM.format(year=y)
    if not os.path.exists(p):
        print(f"  ! missing {p}")
        continue
    d = uf.read_sum(p).dropna(subset=["time"])
    d["year"] = y
    frames.append(d)
CAT = pd.concat(frames, ignore_index=True)
CAT = CAT[(CAT.lon.between(UF_BOX[0], UF_BOX[1])) & (CAT.lat.between(UF_BOX[2], UF_BOX[3]))].copy()
CAT = CAT.sort_values("time").reset_index(drop=True)

# OLD whole-box dt.cc relocation
RC = ["id","lat","lon","depth","x","y","z","ex","ey","ez","yr","mo","dy","hr","mi","sc",
      "mag","nccp","nccs","nctp","ncts","rcc","rct","cid"]
REL = pd.read_csv(OLD_RELOC, sep=r"\s+", names=RC)
REL["time"] = pd.to_datetime(dict(year=REL.yr, month=REL.mo, day=REL.dy,
                                  hour=REL.hr, minute=REL.mi, second=REL.sc.astype(float).clip(0, 59.999)))

# ML (homogenized, constant network)
ML = pd.read_csv(ML_CSV, parse_dates=["event_time"])

GJ_T = pd.Timestamp(GJ_EQ["time"])
print(f"absolute (QC, in box) {len(CAT):,} | dt.cc relocated {len(REL):,} | ML rows {len(ML):,}")''')

md(r"""## P1 — overview map (dt.cc-relocated seismicity)""")

co(r'''fig = pygmt.Figure()
fig.basemap(region=list(MAP_BOX), projection="M12c", frame=["af", "WSen"])
fig.coast(shorelines="0.4p,gray30", land="gray98", water="azure1")
# relocated events colored by depth
pygmt.makecpt(cmap="viridis", series=[0, 20])
fig.plot(x=REL.lon, y=REL.lat, style="c0.09c", fill=REL.depth, cmap=True, pen="0.1p,gray20")
fig.colorbar(frame='af+l"Depth (km)"', position="JMR+o0.4c/0c+w6c")
# UF selection box + Gyeongju epicenter (outside, to the west)
lon0, lon1, la0, la1 = UF_BOX
fig.plot(x=[lon0, lon1, lon1, lon0, lon0], y=[la0, la0, la1, la1, la0], pen="0.8p,gray40,2_2")
fig.plot(x=[GJ_EQ["lon"]], y=[GJ_EQ["lat"]], style="a0.5c", fill="red3", pen="0.5p,black")
fig.text(x=GJ_EQ["lon"], y=GJ_EQ["lat"] - 0.02, text=f"2016 Mw {GJ_EQ['mw']}", font="9p,Helvetica,red3")
fig.basemap(map_scale="jBL+w10k+o0.5c/0.5c")
save(fig, "P1_overview_map", pygmt_fig=True)
fig.show(width=800)''')

md(r"""## P2 — space-time evolution""")

co(r'''fig, ax = plt.subplots(figsize=(10, 4.2))
ax.scatter(CAT.time, CAT.lat, s=4, c="0.75", label=f"absolute (QC, n={len(CAT):,})", rasterized=True)
ax.scatter(REL.time, REL.lat, s=6, c=REL.depth, cmap="viridis", vmin=0, vmax=20,
           label=f"dt.cc relocated (n={len(REL):,})", rasterized=True)
ax.axvline(GJ_T, color="red3", lw=1.2)
ax.text(GJ_T, ax.get_ylim()[1] if ax.get_ylim()[1] else UF_BOX[3], "  Gyeongju Mw 5.5",
        color="red3", fontsize=9, va="top")
ax.set_ylim(UF_BOX[2], UF_BOX[3])
ax.set_ylabel("Latitude (deg)")
ax.set_title("Space-time evolution of Ulsan Fault seismicity (2010-2024)")
leg = ax.legend(loc="upper left", markerscale=2); leg.set_zorder(10)
cb = fig.colorbar(ax.collections[1], ax=ax, pad=0.01); cb.set_label("Depth (km)")
save(fig, "P2_space_time")
plt.show()''')

md(r"""## P3 — magnitude-time + cumulative count""")

co(r'''fig, ax = plt.subplots(figsize=(10, 4.2))
m = ML.dropna(subset=["ml_const"])
ax.scatter(m.event_time, m.ml_const, s=6, c="steelblue", alpha=0.6, rasterized=True,
           label=f"ML (constant network, n={len(m):,})")
ax.axvline(GJ_T, color="red3", lw=1.2)
for dstr, mag, lab in LARGEST:
    ax.annotate(lab, (pd.Timestamp(dstr), mag), textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set_ylabel("ML"); ax.set_title("Magnitudes and cumulative seismicity")
ax2 = ax.twinx()
ax2.plot(CAT.time, np.arange(1, len(CAT) + 1), color="0.2", lw=1.4)
ax2.set_ylabel("Cumulative event count", color="0.2")
leg = ax.legend(loc="upper left"); leg.set_zorder(10)
save(fig, "P3_mag_time")
plt.show()''')

md(r"""## P4 — Gyeongju triggering response""")

co(r'''# weekly event rate in the UF box, 2015-2018
w = CAT.set_index("time").loc["2015":"2018"].resample("7D").size()
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.step(w.index, w.values, where="mid", color="0.25", lw=1.2)
ax.axvline(GJ_T, color="red3", lw=1.2, label="Gyeongju Mw 5.5 (outside the box, ~10 km W)")
ax.axvspan(GJ_T, GJ_T + pd.DateOffset(months=GJ_WINDOW_M), color="red3", alpha=0.10,
           label=f"elevated response ({GJ_WINDOW_M} months)")
ax.set_ylabel("Events / week (UF box)")
ax.set_title("Response of Ulsan Fault seismicity to the 2016 Gyeongju earthquake")
leg = ax.legend(loc="upper right"); leg.set_zorder(10)
save(fig, "P4_gyeongju_response")
plt.show()

# quantify: mean weekly rate before/after (printed, cite on the poster)
pre = CAT[(CAT.time >= GJ_T - pd.DateOffset(months=12)) & (CAT.time < GJ_T)]
post = CAT[(CAT.time >= GJ_T) & (CAT.time < GJ_T + pd.DateOffset(months=GJ_WINDOW_M))]
after = CAT[(CAT.time >= GJ_T + pd.DateOffset(months=GJ_WINDOW_M))
            & (CAT.time < GJ_T + pd.DateOffset(months=GJ_WINDOW_M + 12))]
for name, d, months in (("pre (12 mo)", pre, 12), (f"response ({GJ_WINDOW_M} mo)", post, GJ_WINDOW_M),
                        ("after (12 mo)", after, 12)):
    print(f"  {name:<16} {len(d):>5} events  = {len(d)/(months*4.345):.1f} /week")''')

co(r'''# pre/post 6-month maps side by side (PyGMT)
fig = pygmt.Figure()
for i, (label, dsub) in enumerate((
        (f"{GJ_WINDOW_M} months BEFORE", CAT[(CAT.time >= GJ_T - pd.DateOffset(months=GJ_WINDOW_M)) & (CAT.time < GJ_T)]),
        (f"{GJ_WINDOW_M} months AFTER",  CAT[(CAT.time >= GJ_T) & (CAT.time < GJ_T + pd.DateOffset(months=GJ_WINDOW_M))]))):
    if i:
        fig.shift_origin(xshift="9.5c")
    fig.basemap(region=list(MAP_BOX), projection="M8.5c",
                frame=["af", ("WSen" if i == 0 else "wSen") + f'+t"{label} (n={len(dsub)})"'])
    fig.coast(shorelines="0.4p,gray30", land="gray98", water="azure1")
    fig.plot(x=dsub.lon, y=dsub.lat, style="c0.08c", fill="firebrick", pen="0.1p,gray20")
    fig.plot(x=[GJ_EQ["lon"]], y=[GJ_EQ["lat"]], style="a0.45c", fill="gold", pen="0.5p,black")
    fig.basemap(map_scale="jBL+w10k+o0.4c/0.4c")
save(fig, "P4b_prepost_maps", pygmt_fig=True)
fig.show(width=900)''')

md(r"""## P5 — rebuilt-pipeline highlight: 2016 absolute vs dt.cc (methods panel)""")

co(r'''if os.path.exists(NEW_2016):
    NR = pd.read_csv(NEW_2016, sep=r"\s+", names=RC)
    qc16 = uf.apply_qc(uf.read_sum(NEW_2016_SUM).dropna(subset=["time"]))
    qc16 = qc16[(qc16.lon.between(UF_BOX[0], UF_BOX[1])) & (qc16.lat.between(UF_BOX[2], UF_BOX[3]))]
    fig = pygmt.Figure()
    fig.basemap(region=list(MAP_BOX), projection="M8.5c", frame=["af", 'WSen+t"2016 absolute (QC)"'])
    fig.coast(shorelines="0.4p,gray30", land="gray98", water="azure1")
    fig.plot(x=qc16.lon, y=qc16.lat, style="c0.08c", fill="gray50", pen="0.1p,gray30")
    fig.basemap(map_scale="jBL+w10k+o0.4c/0.4c")
    fig.shift_origin(xshift="9.5c")
    fig.basemap(region=list(MAP_BOX), projection="M8.5c", frame=["af", 'wSen+t"2016 HypoDD dt.cc (rebuilt)"'])
    fig.coast(shorelines="0.4p,gray30", land="gray98", water="azure1")
    fig.plot(x=NR.lon, y=NR.lat, style="c0.08c", fill="#d62728", pen="0.1p,black")
    fig.basemap(map_scale="jBL+w10k+o0.4c/0.4c")
    save(fig, "P5_new2016_abs_vs_dtcc", pygmt_fig=True)
    fig.show(width=900)
else:
    print("new 2016 reloc not found — skip P5")''')

md(r"""## Summary — computed numbers for the poster text""")

co(r'''rows = []
rows.append(("absolute QC events in UF box, 2010-2024", f"{len(CAT):,}"))
rows.append(("dt.cc-relocated events (old whole-box)", f"{len(REL):,}"))
rows.append(("dt.cc with >=1 cc link", f"{int(((REL.nccp+REL.nccs)>0).sum()):,} ({100*((REL.nccp+REL.nccs)>0).mean():.0f}%)"))
pre_r = len(pre)/(12*4.345); post_r = len(post)/(GJ_WINDOW_M*4.345); aft_r = len(after)/(12*4.345)
rows.append(("weekly rate pre / response / after", f"{pre_r:.1f} / {post_r:.1f} / {aft_r:.1f}"))
rows.append(("rate amplification during response", f"x{post_r/max(pre_r,1e-9):.1f}"))
if os.path.exists(NEW_2016):
    rows.append(("rebuilt 2016: located / QC / dt.cc", f"13,123 / 9,440 / {len(NR):,}"))
S = pd.DataFrame(rows, columns=["quantity", "value"])
print(S.to_string(index=False))
S.to_csv(os.path.join(FIGS, "poster_numbers.csv"), index=False)
print("\n-> figs/poster_numbers.csv (cite these on the poster)")''')

nb = nbf.v4.new_notebook(cells=cells,
                         metadata={"kernelspec": {"name": "python3", "display_name": "Python 3 (base)",
                                                  "language": "python"}})
with open(NB, "w") as f:
    nbf.write(nb, f)
print(f"wrote {NB} ({len(cells)} cells, unexecuted)")
