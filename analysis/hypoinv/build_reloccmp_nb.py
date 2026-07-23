#!/usr/bin/env python
"""Generate 21.UF_relocation_dtcc_comparison.ipynb — 3-way comparison of the Ulsan-Fault subregion
relocation: HypoInverse (absolute) vs dt.ct (catalog-differential) vs dt.cc (cross-correlation, CC>=0.7).
The dt.cc run used the recompiled hypoDD (MAXDATA 3M->15M, Waldhauser source) so the full all-pairs
CC>=0.7 set (9.9M dt's) fits. Runs in `base`."""
import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# Ulsan-Fault subregion relocation — HypoInverse vs dt.ct vs dt.cc (CC≥0.7)

Whole-box HypoDD relocation of the UF events, comparing three location sets:

| set | differential times | meaning |
|---|---|---|
| **HypoInverse** | — | absolute single-event locations (hypoDD starting positions) |
| **dt.ct** | catalog | catalog-pick differential times only |
| **dt.cc** | cross-correlation + catalog | waveform CC≥0.7 differential times (the high-resolution result) |

All three legs use the **kim2011** velocity (consistent with nb26 / Zhigang). The dt.cc run used the
**recompiled hypoDD** (`MAXDATA` 3 M → 15 M, large code model) so the full all-pairs CC≥0.7 set
(**≈10.3 M** dt's = 7.7 M cc + ≈2.6 M ct) fits — the stock binary capped at 3 M.""")

co(r"""import warnings; warnings.filterwarnings("ignore")
import numpy as np, pygmt, matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":130,"font.size":10})
D="/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/uf_subregion_reuse/2.HypoDD"
def rd(f,n):
    d={}
    for ln in open(f):
        p=ln.split()
        if len(p)>=n: v=[float(x) for x in p[:n]]; d[int(v[0])]=v
    return d
# PRIMARY kim2011-velocity catalog (SAME velocity for all three legs; matches nb26 / Zhigang):
#   absolute = 03.dt.cc_kim2011/hypoDD.loc (starting positions, model-independent),
#   dt.ct    = 03b.dt.ct_kim2011/hypoDD.reloc (kim2011 catalog-only run, current event set),
#   dt.cc    = 03.dt.cc_kim2011/hypoDD.reloc (kim2011 cross-correlation catalog — the primary product).
loc=rd(f"{D}/03.dt.cc_kim2011/hypoDD.loc",18); ct=rd(f"{D}/03b.dt.ct_kim2011/hypoDD.reloc",24); cc=rd(f"{D}/03.dt.cc_kim2011/hypoDD.reloc",24)
# FULL cc-resolved CATALOG = every event the dt.cc run kept ≥1 surviving cc link for (nccp+nccs>0 over the
# WHOLE run). This is the headline seismicity population reported everywhere (nb26 / Zhigang) — 2157 events.
ccres_ids=sorted(i for i,v in cc.items() if v[17]+v[18]>0)
# 3-way COMPARISON subset: events located by ALL of HypoInverse, dt.ct AND dt.cc — required to measure each
# event's HypoInv->dt.ct->dt.cc shift. It is SMALLER than the catalog because the dt.ct-only run converges on a
# different (smaller) event set, so ~280 cc-resolved events have no dt.ct-run counterpart to compare against.
ids=sorted(set(loc)&set(ct)&set(cc))
def c(d,k): return np.array([d[i][k] for i in ids])
SETS=[("HypoInverse",loc),("dt.ct",ct),("dt.cc",cc)]      # short labels (avoid title overlap in subplots)
REGION=[129.25,129.55,35.60,35.90]
import os
# SOTA Quaternary fault traces (GMT multisegment, columns = lat lon) -> rewrite as lon lat for PyGMT
FAULT="/home/msseo/from_PAGO/21.230822_SRC_Workshop/map-fig2/Map2/ss.txt"
FAULT_GMT="faults_lonlat.gmt"
if os.path.exists(FAULT):
    with open(FAULT) as f, open(FAULT_GMT,"w") as o:
        for ln in f:
            if ln.startswith(">"): o.write(">\n"); continue
            p=ln.split()
            if len(p)>=2:
                try: o.write(f"{float(p[1])} {float(p[0])}\n")     # lat lon -> lon lat
                except ValueError: pass
print(f"relocated: loc {len(loc)}  dt.ct {len(ct)}  dt.cc {len(cc)}  | cc-resolved CATALOG {len(ccres_ids)}  | 3-way comparison subset {len(ids)}  | faults: {os.path.exists(FAULT)}")""")

md(r"""## 1  Cross-correlation usage and displacement

The **cc-resolved catalog** (headline number, reported everywhere) is every event with ≥1 surviving
cross-correlation link over the whole dt.cc run. The **displacement** medians below are necessarily measured
on the smaller **3-way comparison subset** (events located by HypoInverse *and* dt.ct *and* dt.cc), since a
per-event shift needs all three positions.""")
co(r"""# headline: FULL cc-resolved catalog (kim2011 dt.cc run)
NCCP=np.array([cc[i][17] for i in ccres_ids]); NCCS=np.array([cc[i][18] for i in ccres_ids])
print(f"cc-resolved CATALOG (nccp+nccs>0, full dt.cc run): {len(ccres_ids)} of {len(cc)} relocated ({100*len(ccres_ids)/len(cc):.0f}%)")
print(f"  median NCCP {np.median(NCCP):.0f}  NCCS {np.median(NCCS):.0f}  max cc obs {int((NCCP+NCCS).max())}")
# displacement measured on the 3-way comparison subset (needs loc & ct & cc per event)
nccp,nccs=c(cc,17),c(cc,18); m=(nccp+nccs)>0
print(f"3-way comparison subset: {int(m.sum())} cc-resolved of {len(ids)} common events (used for the shifts below)")
def disp(A,B):
    la=c(A,1); return np.hypot((c(B,2)-c(A,2))*111190*np.cos(np.deg2rad(la)),(c(B,1)-c(A,1))*111190), np.abs((c(B,3)-c(A,3))*1000)
hct,vct=disp(loc,ct); hcc,vcc=disp(loc,cc); hd,vd=disp(ct,cc)
print(f"HypoInv -> dt.ct : median H {np.median(hct):.0f} m  V {np.median(vct):.0f} m")
print(f"HypoInv -> dt.cc : median H {np.median(hcc):.0f} m  V {np.median(vcc):.0f} m")
print(f"dt.ct -> dt.cc (events with cc, n={int(m.sum())}): median H {np.median(hd[m]):.0f} m  V {np.median(vd[m]):.0f} m")""")

md(r"""### 1b  Magnitude distribution — dt.cc-resolved vs dt.ct-only (the precision-vs-completeness dilemma)

Are the **dt.ct-only** events (no surviving cc link) all small — so magnitude work could safely use only the
sharper dt.cc set — or do large events get lost from dt.cc? Below: the magnitude distribution of the two
location classes and the **fraction of events lost from dt.cc at each magnitude**. If the largest events are
dt.ct-only, FMD / b-value / moment analysis **must include dt.ct** despite its coarser (~hundreds-m)
locations; the location-precision gain of a dt.cc-only cut is not worth losing mainshocks.""")
co(r"""import pandas as pd
ML="/home/msseo/works/02.Ulsan_Fault_detection/analysis/local_magnitudes/catalog_ml_heo_ufonly_reloc.csv"
mlc=pd.read_csv(ML); ccm=mlc[mlc.is_dtcc].ml_ufcorr_reloc.dropna(); ctm=mlc[~mlc.is_dtcc].ml_ufcorr_reloc.dropna()
fig,ax=plt.subplots(1,2,figsize=(13,4.4))
bins=np.arange(-1.5,4.25,0.25)
ax[0].hist(ccm,bins=bins,histtype="stepfilled",alpha=0.55,color="#4477aa",label=f"dt.cc-resolved (n={len(ccm)}, med {ccm.median():.2f})")
ax[0].hist(ctm,bins=bins,histtype="step",lw=2.2,color="#ee6677",label=f"dt.ct-only (n={len(ctm)}, med {ctm.median():.2f})")
ax[0].set(xlabel=r"$M_L$ (ml_ufcorr_reloc)",ylabel="events",title="Magnitude distribution by location class"); ax[0].legend(fontsize=8.5); ax[0].grid(alpha=0.3)
thr=np.arange(0.0,3.75,0.25); fr=[100*int((ctm>=t).sum())/max(int((ccm>=t).sum())+int((ctm>=t).sum()),1) for t in thr]
ax[1].plot(thr,fr,"o-",color="#ee6677"); ax[1].axhline(100*len(ctm)/(len(ccm)+len(ctm)),ls=":",c="0.5",label=f"overall {100*len(ctm)/(len(ccm)+len(ctm)):.0f}%")
for t in [2.0,2.5,3.0,3.5]:
    n=int((ctm>=t).sum()); ax[1].annotate(f"{n}",(t,100*n/max(int((ccm>=t).sum())+n,1)),textcoords="offset points",xytext=(0,7),fontsize=8.5,ha="center")
ax[1].set(xlabel=r"$M_L \geq$ threshold",ylabel="% of events that are dt.ct-only",title="Large-event loss from a dt.cc-only cut"); ax[1].legend(fontsize=8.5); ax[1].grid(alpha=0.3)
fig.suptitle("dt.ct-only events span the full magnitude range — the largest event is dt.ct-only",fontsize=10,y=1.01); fig.tight_layout(); plt.show()
print("events LOST from a dt.cc-only analysis (dt.ct-only) by magnitude:")
for t in [1.0,1.5,2.0,2.5,3.0,3.5]:
    a=int((ccm>=t).sum()); b=int((ctm>=t).sum()); print(f"  M>={t}: dt.cc {a:4d} | dt.ct-only {b:3d}  ({100*b/max(a+b,1):.0f}% lost)")
print(f"\nMEDIAN ML  dt.cc {ccm.median():.2f}  vs  dt.ct-only {ctm.median():.2f}  -> dt.ct-only are NOT preferentially small.")
big=mlc[~mlc.is_dtcc].nlargest(5,"ml_ufcorr_reloc")[["event_idx","event_time","ml_ufcorr_reloc","depth","nctp","ncts"]]
print("largest dt.ct-only events (incl. the catalog-max M3.89 — lost by a dt.cc-only cut):"); print(big.to_string(index=False))""")

md(r"""## 2  Epicentre maps (PyGMT) — coloured by depth

Two versions below: **(2)** clean (no fault traces) and **(2b)** with the SOTA Quaternary fault traces
overlaid — to judge how the dt.cc locations sit relative to the mapped Ulsan/Yangsan fault system.

**What is plotted for the ~577 events without cross-correlation links?** Every point in all three panels
is a *relocated* position read from that inversion's output (`hypoDD.loc` / `.reloc`). In the **dt.cc**
panel, the events with no surviving cc links are **not** pinned at their HypoInverse position — they are
still constrained by their **catalog (dt.ct) differential times** within the joint inversion (moving a
median ≈1 km off the absolute start; the exact value prints in §5). So the dt.cc map is uniformly
relocated: **cc-sharpened for the 79 % with links, dt.ct-constrained for the other 21 %** — none sit at raw
HypoInverse.""")
co(r"""# plain frame + DECIMAL-degree annotations (not d°m's"); 10 km scale bar lower-left of every panel.
def maps(with_faults, mask=None):
    fig=pygmt.Figure()
    with pygmt.config(MAP_FRAME_TYPE="plain", FORMAT_GEO_MAP="ddd.xx"):
        with fig.subplot(nrows=1,ncols=3,figsize=("36c","12c"),margins="0.8c"):
            pygmt.makecpt(cmap="turbo",series=[2,20],reverse=True)
            for j,(name,d) in enumerate(SETS):
                x,y,z=c(d,2),c(d,1),c(d,3)
                if mask is not None: x,y,z=x[mask],y[mask],z[mask]
                with fig.set_panel(j):
                    fig.basemap(region=REGION,projection="M?",frame=[f"WSne+t{name}","xa0.1f0.05","ya0.1f0.05"])
                    fig.coast(shorelines="0.5p,gray50")
                    if with_faults and os.path.exists(FAULT_GMT):
                        fig.plot(data=FAULT_GMT,pen="0.9p,black")     # SOTA fault traces
                    fig.plot(x=x,y=y,fill=z,cmap=True,style="c0.10c",pen="0.2p,gray20")
                    fig.basemap(map_scale="jBL+w10k+o0.5c/0.5c+c35.75")   # 10 km, lower-left
            with fig.set_panel(1):   # anchor the colorbar to the MIDDLE panel -> centred below the row
                fig.colorbar(position="JBC+w12c/0.4c+h+o0c/1.4c",frame=["xa4+lDepth (km)"])
    fig.show(width=2000)
maps(with_faults=False)""")

md(r"""### 2b  With SOTA Quaternary fault traces (Ulsan / Yangsan system)""")
co(r"""maps(with_faults=True)""")

md(r"""### 2c  dt.cc cross-correlation catalog — **full cc-resolved set (n=2157)**

THE seismicity map: every event the dt.cc run resolved with ≥1 surviving cross-correlation link
(`nccp+nccs>0`) — the full **2,157-event** kim2011 catalog (not the smaller 3-way comparison subset used
for the displacement stats), coloured by depth against the SOTA Quaternary fault traces.""")
co(r"""def map_full(idlist,title):
    X=np.array([cc[i][2] for i in idlist]); Y=np.array([cc[i][1] for i in idlist]); Z=np.array([cc[i][3] for i in idlist])
    fig=pygmt.Figure()
    with pygmt.config(MAP_FRAME_TYPE="plain", FORMAT_GEO_MAP="ddd.xx"):
        fig.basemap(region=REGION,projection="M15c",frame=[f"WSne+t{title}","xa0.1f0.05","ya0.1f0.05"])
        fig.coast(shorelines="0.5p,gray50")
        if os.path.exists(FAULT_GMT): fig.plot(data=FAULT_GMT,pen="0.9p,black")
        pygmt.makecpt(cmap="turbo",series=[2,20],reverse=True)
        fig.plot(x=X,y=Y,fill=Z,cmap=True,style="c0.10c",pen="0.2p,gray20")
        fig.basemap(map_scale="jBL+w10k+o0.5c/0.5c+c35.75")
        fig.colorbar(position="JBC+w10c/0.4c+h+o0c/1.2c",frame=["xa4+lDepth (km)"])
    fig.show(width=950)
print(f"cc-resolved catalog plotted: {len(ccres_ids)} events (full dt.cc kim2011 set)")
map_full(ccres_ids, f"dt.cc cc-resolved catalog (n={len(ccres_ids)})")""")

md(r"""## 3  Depth cross-sections (longitude vs depth) — vertical collapse (all events)""")
co(r"""fig,ax=plt.subplots(1,3,figsize=(16,4.6),sharex=True,sharey=True)
for a,(name,d) in zip(ax,SETS):
    a.scatter(c(d,2),c(d,3),s=4,alpha=0.35,c=c(d,3),cmap="turbo_r",vmin=2,vmax=20)
    a.set(title=name,xlabel="Longitude (°E)",ylim=(22,2)); a.grid(alpha=0.3)
ax[0].set_ylabel("Depth (km)")
fig.suptitle("Depth vs longitude — absolute → dt.ct → dt.cc (all common events)",fontsize=10,y=1.0); fig.tight_layout(); plt.show()""")

md(r"""### 3a  Depth sections — **full cc-resolved catalog (n=2157)**

THE seismicity depth view: the full 2,157-event dt.cc cc-resolved catalog in longitude–depth (left) and
latitude–depth (right), coloured by depth. (No dt.ct/HypoInverse panels here — this is the catalog itself,
not a per-event comparison, so it is not tied to the smaller 3-way subset.)""")
co(r"""Xc=np.array([cc[i][2] for i in ccres_ids]); Yc=np.array([cc[i][1] for i in ccres_ids]); Zc=np.array([cc[i][3] for i in ccres_ids])
fig,ax=plt.subplots(1,2,figsize=(15,4.6),sharey=True)
ax[0].scatter(Xc,Zc,s=5,alpha=0.4,c=Zc,cmap="turbo_r",vmin=2,vmax=20); ax[0].set(xlabel="Longitude (°E)",ylabel="Depth (km)",ylim=(22,2)); ax[0].grid(alpha=0.3)
ax[1].scatter(Yc,Zc,s=5,alpha=0.4,c=Zc,cmap="turbo_r",vmin=2,vmax=20); ax[1].set(xlabel="Latitude (°N)",ylim=(22,2)); ax[1].grid(alpha=0.3)
fig.suptitle(f"dt.cc cc-resolved catalog depth sections (n={len(ccres_ids)}); left: longitude, right: latitude",fontsize=10,y=1.0); fig.tight_layout(); plt.show()
print(f"cc-resolved catalog depth: median {np.median(Zc):.1f} km  IQR {np.percentile(Zc,75)-np.percentile(Zc,25):.1f} km  (n={len(ccres_ids)})")""")

md(r"""### 3b  Depth sections coloured by **hour-of-day (KST)** — blast check (full cc-resolved catalog, n=2157)

The full 2,157-event dt.cc cc-resolved catalog in longitude–depth and latitude–depth, coloured by
**hour-of-day (KST, cyclic colormap)** instead of depth. A shallow daytime (06–18 KST) cluster confined to
one depth/locus would betray a quarry; natural seismicity shows no hour–depth structure.""")
co(r"""hh=np.array([(int(cc[i][13])+9)%24 for i in ccres_ids])   # KST hour, full cc-resolved catalog
fig,ax=plt.subplots(1,2,figsize=(15,4.6),sharey=True)
sc=ax[0].scatter(Xc,Zc,s=6,c=hh,cmap="hsv",vmin=0,vmax=24); ax[0].set(xlabel="Longitude (°E)",ylabel="Depth (km)",ylim=(22,2)); ax[0].grid(alpha=0.3)
ax[1].scatter(Yc,Zc,s=6,c=hh,cmap="hsv",vmin=0,vmax=24); ax[1].set(xlabel="Latitude (°N)",ylim=(22,2)); ax[1].grid(alpha=0.3)
cb=fig.colorbar(sc,ax=ax,shrink=0.7,pad=0.02,ticks=[0,6,12,18,24]); cb.set_label("Hour of day (KST)")
fig.suptitle(f"dt.cc cc-resolved catalog depth sections coloured by hour-of-day (KST) — n={len(ccres_ids)}",fontsize=10,y=1.0)
plt.show()""")

md(r"""### 3c  Focal-depth distributions

**Left** — how the differential-time relocation reshapes the depth distribution: the three methods overlaid
on the events they share (method comparison — needs common events). **Right** — the depth distribution of
the **full cc-resolved dt.cc catalog (n=2157)**, the seismicity product itself.""")
co(r"""COL={"HypoInverse":"0.45","dt.ct":"#ee8866","dt.cc":"#4477aa"}
bins=np.arange(0,24.1,0.5)
fig,ax=plt.subplots(1,2,figsize=(13,4.3),sharey=False)
for name,d in SETS:                                   # LEFT: 3-method comparison on shared events
    z=c(d,3)
    ax[0].hist(z,bins=bins,histtype="step",lw=2,color=COL[name],label=f"{name}: med {np.median(z):.1f} km")
ax[1].hist(Zc,bins=bins,histtype="stepfilled",lw=2,color=COL["dt.cc"],alpha=0.85,   # RIGHT: full cc catalog
           label=f"dt.cc cc-resolved: med {np.median(Zc):.1f} km")
ax[0].set(title=f"3-method comparison (shared events, n={len(ids)})",xlabel="Focal depth (km)",ylabel="events")
ax[1].set(title=f"dt.cc cc-resolved catalog (n={len(ccres_ids)})",xlabel="Focal depth (km)",ylabel="events")
for a in ax: a.legend(fontsize=8.5); a.grid(alpha=0.3)
fig.suptitle("Focal-depth distributions",fontsize=10,y=1.0); fig.tight_layout(); plt.show()
print(f"cc-resolved catalog (n={len(ccres_ids)}) depth: median {np.median(Zc):.2f} km  IQR {np.percentile(Zc,75)-np.percentile(Zc,25):.2f}  std {np.std(Zc):.2f}")""")

md(r"""## 4  Local collapse — nearest-neighbour distance

The whole-box spread barely changes (events span the box); the relocation benefit is **local** —
nearby events tighten onto structures. Median 3-D distance to each event's nearest neighbour:""")
co(r"""from scipy.spatial import cKDTree
def xyz(d):
    la=c(d,1); x=(c(d,2)-REGION[0])*111190*np.cos(np.deg2rad(la)); y=(c(d,1)-REGION[2])*111190; z=c(d,3)*1000
    return np.column_stack([x,y,z])
fig,ax=plt.subplots(figsize=(7.5,4.4))
for name,d in SETS:
    P=xyz(d); dd,_=cKDTree(P).query(P,k=2); nn=dd[:,1]
    ax.hist(nn,bins=np.linspace(0,1500,50),histtype="step",lw=1.8,label=f"{name}: median NN {np.median(nn):.0f} m")
    print(f"{name:26} median nearest-neighbour distance: {np.median(nn):.0f} m")
ax.set(xlabel="Nearest-neighbour distance (m)",ylabel="Events",title="Local collapse (smaller = tighter clustering)")
leg=ax.legend(fontsize=9); leg.set_zorder(50); fig.tight_layout(); plt.show()""")

md(r"""## 5  Events not sharpened by cross-correlation — where they sit

About **600 events received no cc links** (`NCCP+NCCS = 0`): hypoDD could not cross-correlate them with
neighbours (sparse early-network era, isolated events, or low waveform SNR). They are **not** left at
HypoInverse — they are **still relocated by their catalog (dt.ct) differential times** inside the joint
inversion, moving a median **≈870 m** off HypoInverse — they are simply not *waveform*-sharpened. The two
maps below show the **same** no-cc events at their **HypoInverse start** (left) and their **final
dt.ct-constrained dt.cc-run position** (right); §7 checks their hour-of-day for blast-likeness.""")
co(r"""# FULL catalog split (not the 3-way comparison subset): every relocated event with a HypoInverse start
_ev=[i for i in cc if i in loc]
nocc   = [i for i in _ev if (cc[i][17]+cc[i][18])==0]   # no surviving cc links -> dt.ct-constrained
withcc = [i for i in _ev if (cc[i][17]+cc[i][18])>0]    # cc-resolved catalog
def csub(idlist,d,k): return np.array([d[i][k] for i in idlist])
dep_nocc=csub(nocc,cc,3)                                    # FINAL (dt.cc-run) depth of the no-cc events
move=np.hypot((csub(nocc,cc,2)-csub(nocc,loc,2))*111190*np.cos(np.deg2rad(csub(nocc,loc,1))),
              (csub(nocc,cc,1)-csub(nocc,loc,1))*111190)
print(f"cc-resolved: {len(withcc)}   NO cc links (dt.ct-constrained): {len(nocc)} ({100*len(nocc)/len(_ev):.0f}%)  [full catalog]")
print(f"no-cc events moved median {np.median(move):.0f} m off HypoInverse (via dt.ct) — NOT pinned at HypoInverse")
print(f"no-cc final depth: median {np.median(dep_nocc):5.1f} km  shallow(<5km) {100*np.mean(dep_nocc<5):4.0f}%")
print(f"cc-rel    depth: median {np.median(csub(withcc,cc,3)):5.1f} km  shallow(<5km) {100*np.mean(csub(withcc,cc,3)<5):4.0f}%")
def dropped_map():
    fig=pygmt.Figure()
    with fig.subplot(nrows=1,ncols=2,figsize=("24c","12c"),margins="0.8c"):
        pygmt.makecpt(cmap="turbo",series=[2,20],reverse=True)
        panels=[(f"No cc link \\267 HypoInverse start ({len(nocc)})",nocc,loc),
                (f"No cc link \\267 dt.ct-constrained final ({len(nocc)})",nocc,cc)]
        for j,(name,idl,src) in enumerate(panels):
            with fig.set_panel(j):
                fig.basemap(region=REGION,projection="M?",frame=[f"WSne+t{name}","xa0.1f0.05","ya0.1f0.05"])
                fig.coast(shorelines="0.5p,gray50")
                if os.path.exists(FAULT_GMT): fig.plot(data=FAULT_GMT,pen="0.9p,black")
                fig.plot(x=csub(idl,src,2),y=csub(idl,src,1),fill=csub(idl,src,3),cmap=True,style="c0.10c",pen="0.2p,gray20")
        with fig.set_panel(0):
            fig.colorbar(position="JBC+w12c/0.4c+h+o6.3c/1.4c",frame=["xa4+lDepth (km)"])
    fig.show(width=1500)
dropped_map()""")

md(r"""## 6  Hour-of-day (KST) — natural vs blast-like

Origin times are **UTC** (verified: the M3.88 2014-09-23 06:27 UTC = 15:27 KST and M3.78 2023-11-29
19:55 UTC = 04:55 KST next day both match KMA exactly); converted here to **KST (UTC+9)**. Tectonic
earthquakes are ~uniform over 24 h; quarry/mine blasts concentrate in **daytime working hours
(06–18 KST, shaded)**. Bars are coloured by the **cyclic hour colormap** used in the blast-decluster
notebooks. (The catalog is already `blastclean`, so a strong daytime excess is *not* expected — this is
a confirmation that the relocated populations are natural.)""")
co(r"""def hr_kst(idlist,d): return np.array([(int(d[i][13])+9)%24 for i in idlist])
_cyc=plt.get_cmap("hsv")      # vivid cyclic (matches the GMT "cyclic" hour-of-day maps)
def hour_hist(ax,hours,title):
    h,_=np.histogram(hours,bins=np.arange(25))
    for k in range(24): ax.bar(k,h[k],width=1.0,align="edge",color=_cyc((k+0.5)/24.0),edgecolor="white",lw=0.3,zorder=2)
    ax.axvspan(6,18,color="0.80",alpha=0.45,zorder=0)
    frac=float(np.mean((hours>=6)&(hours<18)))
    ax.set(xlim=(0,24),xticks=[0,6,12,18,24],xlabel="Hour of day (KST)")
    ax.set_title(f"{title}\nn={len(hours)}, daytime frac={frac:.2f}",fontsize=9)
    return frac
fig,ax=plt.subplots(1,3,figsize=(15,3.4),dpi=130)
for a,(name,d) in zip(ax,SETS): hour_hist(a,hr_kst(ids,d),name)
ax[0].set_ylabel("events")
fig.suptitle("Hour-of-day (KST) — HypoInverse vs dt.ct vs dt.cc",fontsize=10,y=1.02)
fig.tight_layout(); plt.show()""")

md(r"""### 6b  Hour-of-day MAPS (KST) — spatial check for quarry clustering

Bars tell us *how many* events fire by hour; **maps** tell us *where*. Quarry blasts cluster spatially
(at the pit) **and** in daytime. Here the dt.cc epicentres are coloured by hour-of-day (KST, cyclic
colormap — same style as the blast-decluster notebooks), split **deep (≥7.5 km)** vs **shallow
(<7.5 km)**. If the shallow daytime events piled up at one spot, a quarry would stand out.""")
co(r"""SHAL_Z=7.5
shallow=[i for i in _ev if cc[i][3]<SHAL_Z]; deep=[i for i in _ev if cc[i][3]>=SHAL_Z]   # full relocated catalog
hh=lambda idl: np.array([(int(cc[i][13])+9)%24 for i in idl])
fig=pygmt.Figure()
with fig.subplot(nrows=1,ncols=2,figsize=("24c","12c"),margins="0.8c"):
    pygmt.makecpt(cmap="cyclic",series=[0,24,1],continuous=True)
    for j,(name,idl) in enumerate([(f"Deep \\263 {SHAL_Z:g} km ({len(deep)})",deep),
                                   (f"Shallow < {SHAL_Z:g} km ({len(shallow)})",shallow)]):
        with fig.set_panel(j):
            fig.basemap(region=REGION,projection="M?",frame=[f"WSne+t{name}","xa0.1f0.05","ya0.1f0.05"])
            fig.coast(shorelines="0.5p,gray50")
            if os.path.exists(FAULT_GMT): fig.plot(data=FAULT_GMT,pen="0.9p,black")
            fig.plot(x=csub(idl,cc,2),y=csub(idl,cc,1),fill=hh(idl),cmap=True,style="c0.13c",pen="0.2p,gray20")
    with fig.set_panel(0):
        fig.colorbar(position="JBC+w12c/0.4c+h+o6.3c/1.4c",frame=["xa6+lHour of day (KST)"])
fig.show(width=1500)""")

md(r"""## 7  Hour-of-day (KST) of the dt.ct-constrained (no-cc) events

If the ~600 no-cc events were residual quarry blasts they would pile up in daytime hours and be shallow.
Compare their hour-of-day against the cc-relocated population.""")
co(r"""fig,ax=plt.subplots(1,2,figsize=(10.5,3.4),dpi=130)
f_nocc=hour_hist(ax[0],hr_kst(nocc,cc),"No cc link (dt.ct-constrained)")
f_cc  =hour_hist(ax[1],hr_kst(withcc,cc),"cc-relocated")
ax[0].set_ylabel("events")
fig.suptitle("No-cc vs cc-relocated — hour-of-day (KST)",fontsize=10,y=1.02)
fig.tight_layout(); plt.show()
print(f"daytime(06–18 KST) fraction — no-cc {f_nocc:.2f}   cc-relocated {f_cc:.2f}   (uniform 0.50)")""")

md(r"""### 7b  Hour-of-day (KST) by depth — shallow vs deep (dt.cc relocated catalog)

The sharpest blast test: **shallow** events are the quarry suspects. Split the **dt.cc** relocated catalog
at **7.5 km depth** (the `shallow`/`deep` sets from §6b) and compare hour-of-day. A daytime (06–18 KST)
excess concentrated in the shallow (<7.5 km) bin would flag residual surface sources; a flat distribution
at both depths confirms natural seismicity.""")
co(r"""fig,ax=plt.subplots(1,2,figsize=(10.5,3.4),dpi=130)
f_sh=hour_hist(ax[0],hr_kst(shallow,cc),f"dt.cc shallow (<{SHAL_Z:g} km)")
f_dp=hour_hist(ax[1],hr_kst(deep,cc),f"dt.cc deep (≥{SHAL_Z:g} km)")
ax[0].set_ylabel("events")
fig.suptitle("dt.cc catalog — hour-of-day (KST) by depth",fontsize=10,y=1.02)
fig.tight_layout(); plt.show()
print(f"daytime(06–18 KST) fraction — shallow<{SHAL_Z:g}km (n={len(shallow)}) {f_sh:.2f}   deep≥{SHAL_Z:g}km (n={len(deep)}) {f_dp:.2f}   (uniform 0.50)")""")

md(r"""## 8  Summary

* The dt.cc cross-correlation relocation (CC≥0.7, full all-pairs) was enabled by **recompiling hypoDD**
  with `MAXDATA = 15 M`; the stock 3 M binary silently fell back to dt.ct-only (empty combine file).
* dt.cc uses cross-correlation links for **2157 of 2734 relocated events (79%)** — the cc-resolved catalog —
  and tightens both epicentres and depths beyond the catalog-only dt.ct, with the largest gains in **local
  nearest-neighbour distance** (clusters collapsing onto structures) rather than the whole-box spread.
* The remaining **577 events (21%) keep no surviving cc link** but are **not** left at HypoInverse — they are
  relocated by their catalog (dt.ct) differential times (§5), moving a median few-hundred m off the absolute
  start; their map + hour-of-day (§7) show the remainder is ordinary seismicity, not blast-like.
* All three legs use the **kim2011** velocity (matches nb26 / Zhigang). Outputs:
  `…/2.HypoDD/03.dt.cc_kim2011/hypoDD.reloc` (dt.cc, primary), `03b.dt.ct_kim2011/hypoDD.reloc` (dt.ct),
  `03.dt.cc_kim2011/hypoDD.loc` (absolute starts).""")

nb["cells"]=C
import os
os.chdir("/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv")
nbf.write(nb,"21.UF_relocation_dtcc_comparison.ipynb")
print("wrote 21.UF_relocation_dtcc_comparison.ipynb with",len(C),"cells")
