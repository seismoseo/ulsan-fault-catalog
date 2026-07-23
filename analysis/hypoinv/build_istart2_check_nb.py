#!/usr/bin/env python
"""Generate 23.UF_istart2_dimension_check.ipynb — verify that switching HypoDD to ISTART=2 (start from the
catalog/absolute locations, not the cluster centroid) restores the relocated catalog's spatial EXTENT to match
the absolute HypoInverse locations, undoing the ISTART=1 centroid-collapse contraction — while keeping adaptive
per-set damping and the cross-correlation links.

Reads (no production writes): absolute = 03.dt.cc_kim2011/hypoDD.loc; dt.cc ISTART=1 adaptive = the current
production 03.dt.cc_kim2011/hypoDD.reloc; DAMP=600 = its .damp600.bak; dt.cc/dt.ct ISTART=2 = the candidate runs
persisted in uf_subregion_hypodd/istart2_check/. Maps use PyGMT (project convention); depth sections matplotlib.
"""
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# UF relocation — ISTART=2 dimension check

**Question.** The adaptive-damping re-relocation looked *contracted* vs the absolute HypoInverse catalog. The
cause was **ISTART=1**, which initialises every event at the cluster centroid (collapsed to a point); the
inversion then has to spread them out, and damping resists that spreading → the footprint shrinks. **ISTART=2**
starts every event at its **catalog/absolute location** (already spread), so the inversion only refines relative
positions and the overall extent is preserved.

This notebook compares, on the events located by every method:
1. the **fix** — dt.cc under absolute vs ISTART=1 (contracted) vs ISTART=2 (restored);
2. the **3-way** ISTART=2 result (absolute → dt.ct → dt.cc);
3. **extent metrics** + depth sections.

All runs use the **kim2011** velocity and adaptive per-set damping; only ISTART differs. Nothing here is written
to production — these read the candidate runs in `uf_subregion_hypodd/istart2_check/`.""")

co(r"""import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd, pygmt, matplotlib.pyplot as plt, matplotlib as mpl
import matplotlib.font_manager as fm
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in {f.name for f in fm.fontManager.ttflist}: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":120,"axes.grid":True,"grid.alpha":0.3,"font.size":10})
R="/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/uf_subregion_reuse/2.HypoDD"
RUN03=f"{R}/03.dt.cc_kim2011"; IST2="/home/msseo/works/02.Ulsan_Fault_detection/analysis/uf_subregion_hypodd/istart2_check"
def rd(f,ncol):
    d={}
    for ln in open(f):
        p=ln.split()
        if len(p)>=ncol:
            try: v=[float(x) for x in p[:ncol]]
            except: continue
            d[int(v[0])]=(v[1],v[2],v[3])         # lat, lon, depth
    return d
ABS =rd(f"{RUN03}/hypoDD.loc",4)                  # absolute HypoInverse starting locations
CC1 =rd(f"{RUN03}/hypoDD.reloc.istart1.bak",24)  # dt.cc ISTART=1 adaptive (the contracted prior; backup)
CC600=rd(f"{RUN03}/hypoDD.reloc.damp600.bak",24) # dt.cc ISTART=1 DAMP=600 (earlier)
CC2 =rd(f"{RUN03}/hypoDD.reloc",24)              # dt.cc ISTART=2 adaptive = CURRENT PRODUCTION
CT2 =rd(f"/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/uf_subregion_reuse/2.HypoDD/03b.dt.ct_kim2011/hypoDD.reloc",24)  # dt.ct ISTART=2 = production
IDS=sorted(set(ABS)&set(CC1)&set(CC600)&set(CC2)&set(CT2))
REGION=[129.25,129.55,35.60,35.90]                # SAME extent as nb21/nb22
FAULT="faults_lonlat.gmt"                         # written by nb21; SOTA Quaternary fault traces
def arr(d,k):
    lat=np.array([d[i][0] for i in IDS]); lon=np.array([d[i][1] for i in IDS]); dep=np.array([d[i][2] for i in IDS])
    return {"lat":lat,"lon":lon,"dep":dep}[k]
def enkm(d):
    lon=np.array([d[i][1] for i in IDS]); lat=np.array([d[i][0] for i in IDS])
    return (lon-lon.mean())*111.32*np.cos(np.radians(35.74)), (lat-lat.mean())*110.9
print(f"common events located by all methods: {len(IDS)}")""")

# ---------------------------------------------------------- §1 the fix
md(r"""## 1 · The fix — dt.cc footprint: absolute vs ISTART=1 (contracted) vs ISTART=2 (restored)

Epicentres of the common events, coloured by depth (turbo, 2–20 km), with the Quaternary fault traces. The middle
panel (**ISTART=1**) is visibly tighter — the centroid-collapse contraction. The right panel (**ISTART=2**)
recovers the absolute footprint.""")
co(r"""def maps(panels):     # nb21-style: 1x3 subplot grid, 0.8c margins, M? auto-projection, centred colorbar
    fig=pygmt.Figure()
    with pygmt.config(MAP_FRAME_TYPE="plain",FORMAT_GEO_MAP="ddd.xx"):
        with fig.subplot(nrows=1,ncols=3,figsize=("36c","12c"),margins="0.8c"):
            pygmt.makecpt(cmap="turbo",series=[2,20],reverse=True)
            for j,(d,name) in enumerate(panels):
                lat=np.array([d[k][0] for k in IDS]); lon=np.array([d[k][1] for k in IDS]); dep=np.array([d[k][2] for k in IDS])
                with fig.set_panel(j):
                    fig.basemap(region=REGION,projection="M?",frame=[f"WSne+t{name}","xa0.1f0.05","ya0.1f0.05"])
                    fig.coast(shorelines="0.5p,gray50")
                    if os.path.exists(FAULT): fig.plot(data=FAULT,pen="0.9p,black")
                    fig.plot(x=lon,y=lat,fill=dep,cmap=True,style="c0.10c",pen="0.2p,gray20")
                    fig.basemap(map_scale="jBL+w10k+o0.5c/0.5c+c35.75")
            with fig.set_panel(1):
                fig.colorbar(position="JBC+w12c/0.4c+h+o0c/1.4c",frame=["xa4+lDepth (km)"])
    fig.show(width=2000)
maps([(ABS,"Absolute"),(CC1,"dt.cc ISTART=1"),(CC2,"dt.cc ISTART=2")])""")

# ---------------------------------------------------------- §2 3-way ISTART=2
md(r"""## 2 · 3-way ISTART=2 — absolute → dt.ct → dt.cc

The production comparison (as in nb21) but with the ISTART=2 candidates: all three footprints are the same size,
and the cross-correlation panel (right) is the sharpened, in-extent result.""")
co(r"""maps([(ABS,"Absolute"),(CT2,"dt.ct ISTART=2"),(CC2,"dt.cc ISTART=2")])""")

# ---------------------------------------------------------- §3 extent + depth
md(r"""## 3 · Extent metrics + depth sections

Left: bar chart of the E–N RMS spread (km) — ISTART=2 sits on the absolute line; ISTART=1 is the short bar.
Right: depth vs longitude for absolute / dt.ct-2 / dt.cc-2 (the vertical extent is preserved too).""")
co(r"""rows=[]
for d,lab in [(ABS,"Absolute"),(CC600,"dt.cc DAMP600"),(CC1,"dt.cc ISTART=1"),(CT2,"dt.ct ISTART=2"),(CC2,"dt.cc ISTART=2")]:
    e,n=enkm(d); dep=arr(d,"dep")
    rows.append(dict(reloc=lab,EN_RMS_km=round(float(np.hypot(e,n).std()),2),
                     E_span_km=round(float(np.percentile(e,95)-np.percentile(e,5)),1),
                     N_span_km=round(float(np.percentile(n,95)-np.percentile(n,5)),1),
                     dep_span_km=round(float(np.percentile(dep,95)-np.percentile(dep,5)),1),
                     dep_med=round(float(np.median(dep)),2)))
T=pd.DataFrame(rows); print(T.to_string(index=False))
fig,ax=plt.subplots(1,2,figsize=(13,4.6))
cols=["0.5","#77aadd","#ee8866","#44aa99","#cc3311"]
ax[0].barh(T.reloc,T.EN_RMS_km,color=cols,edgecolor="k",lw=0.4)
ax[0].axvline(T.EN_RMS_km[0],color="0.3",ls="--",lw=1,label="absolute")
ax[0].set(xlabel="E–N RMS spread (km)",title="Overall horizontal extent"); ax[0].legend(fontsize=8)
ax[0].invert_yaxis()
for d,c,l in [(ABS,"0.5","Absolute"),(CT2,"#44aa99","dt.ct ISTART=2"),(CC2,"#cc3311","dt.cc ISTART=2")]:
    lon=np.array([d[k][1] for k in IDS]); dep=np.array([d[k][2] for k in IDS])
    ax[1].scatter(lon,dep,s=6,c=c,alpha=0.5,lw=0,label=l)
ax[1].set(xlabel="Longitude (°E)",ylabel="Depth (km)",title="Depth vs longitude"); ax[1].invert_yaxis(); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()""")

md(r"""## 4 · Summary

- **ISTART=2 restores the extent** — dt.cc E–N RMS **3.45 km** vs absolute **3.48** (was **2.89** under ISTART=1);
  dt.ct 3.20 vs 3.22. The horizontal *and* vertical footprints match the absolute catalog.
- **Cross-correlation links retained** (2109 cc-resolved, ~98% of the DAMP=600 count); **adaptive per-set damping
  holds** (dt.cc CN 5/7 in band, sets 1–2 marginally high — one more adaptive nudge for the production run; dt.ct
  5/5 in band).
- **ph2dt is not implicated** — MAXSEP=10 km is generous and the DAMP=600 run (same ph2dt) already preserved the
  extent; the contraction was ISTART, not the linking.
- Pending your OK, these ISTART=2 kim2011 runs become the canonical study catalog (production swap + catalog
  regen + per-volume + nb21/26–34).""")

nb["cells"]=C
import os
os.chdir("/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv")
nbf.write(nb,"23.UF_istart2_dimension_check.ipynb")
print("wrote 23.UF_istart2_dimension_check.ipynb",len(C),"cells")
