#!/usr/bin/env python
"""Generate 10.Picker_comparison_2016.ipynb — controlled 4-picker comparison on 2016 UF-box.
Consistent P=S=0.2 threshold, identical downstream (PyOcto 4/2/2, kim2011, uf_cluster QC, ISTART=2 adaptive
dt.ct/dt.cc). The picker is the ONLY variable. Funnel per picker: picks(P/S) -> associated events -> UF-box
events -> QC-passed -> dt.ct-relocated -> dt.cc-relocated -> cc-resolved; + location quality; + 4 dt.cc maps."""
import nbformat as nbf
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# ML-picker comparison on 2016 Ulsan-Fault — PN+ vs PhaseNet-original vs PhaseNet-STEAD vs EQT-STEAD

Four ML pickers, one pipeline. Every stage after picking is identical — PyOcto association (gate 4/2/2,
kim2011), HypoInverse (kim2011), `uf_cluster` QC (`erh<5, erz<5, gap<270, num>5, rms<1.0`), and HypoDD
dt.ct/dt.cc (kim2011, ISTART=2, adaptive LSQR damping CND 40-80, interp_hz=1000). **Consistent P=S=0.2
threshold on all four** (single `gj_config.PICK_PROB`). The picker is the only variable, so any difference in
the funnel below is attributable to the picker.

| key | picker | weights |
|---|---|---|
| **pn+** | PhaseNet+ (EQNet) | your Buan-tuned build |
| **original** | PhaseNet-original | NCEDC (Zhu & Beroza 2019) |
| **stead** | PhaseNet-STEAD | STEAD-retrained |
| **eqt** | EQTransformer-STEAD | Mousavi 2020 (STEAD) |""")

co(r"""import os, glob, numpy as np, pandas as pd, pyarrow.parquet as pq, pygmt
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm, sys
_av={f.name for f in fm.fontManager.ttflist}
for _f in ["Helvetica","Arial","Nimbus Sans","Liberation Sans"]:
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"font.size":10,"figure.dpi":130,"axes.grid":True,"grid.alpha":0.3,
                     "legend.framealpha":1.0,"legend.facecolor":"white"})
from uflib import uf_cluster as uf
DT="/home/msseo/works/02.Ulsan_Fault_detection/detection_test"
RUNS="/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs"
REGION=[129.25,129.55,35.60,35.90]; FAULT="/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv/faults_lonlat.gmt"
COAST="/home/msseo/works/02.Ulsan_Fault_detection/analysis/reloc_analysis/coastline_lonlat.gmt"
PICKERS=[("phasenet_plus","PN+"),("original","PN-original"),("stead","PN-STEAD"),("eqt","EQT-STEAD")]

def paths(p):
    if p=="phasenet_plus": return f"{DT}/reloc_2016_uf","uf_2016","uf_2016_qc"
    return f"{DT}/reloc_2016_uf_{p}",f"uf_2016_{p}",f"uf_2016_{p}_qc"
def rd_reloc(f):
    if not os.path.exists(f): return None
    r=[l.split() for l in open(f) if l.split()]
    return pd.DataFrame([x[:24] for x in r],columns=["id","lat","lon","depth","x","y","z","ex","ey","ez",
        "yr","mo","dy","hr","mi","sc","mag","nccp","nccs","nctp","ncts","rcc","rct","cid"][:len(r[0])]).apply(pd.to_numeric,errors="coerce")

def funnel(p):
    root,full,qc=paths(p)
    nP=nS=0
    for m in range(1,13):
        f=f"{DT}/picks/picks_{p}_2016_{m:02d}.parquet"
        if os.path.exists(f):
            ph=pq.read_table(f,columns=["phase"]).column("phase").to_pandas().str.upper()
            nP+=int((ph=="P").sum()); nS+=int((ph=="S").sum())
    ev=sum(len(pd.read_csv(f)) for f in sorted(glob.glob(f"{DT}/catalogs/catalog_{p}_2016_*_pyocto.csv")))
    ufbox=len(open(f"{root}/members.txt").read().split()) if os.path.exists(f"{root}/members.txt") else np.nan
    qcn=len(open(f"{root}/members_qc.txt").read().split()) if os.path.exists(f"{root}/members_qc.txt") else np.nan
    cc=rd_reloc(f"{RUNS}/{qc}/2.HypoDD/02.dt.cc/hypoDD.reloc")
    ct=rd_reloc(f"{RUNS}/{qc}/2.HypoDD/01b.dtct_qc/hypoDD.reloc")
    n_cc=len(cc) if cc is not None else np.nan
    n_ct=len(ct) if ct is not None else np.nan
    n_ccres=int(((cc.nccp+cc.nccs)>0).sum()) if cc is not None else np.nan
    return dict(picks_P=nP,picks_S=nS,picks=nP+nS,events=ev,ufbox=ufbox,QC=qcn,
                dtct=n_ct,dtcc=n_cc,ccres=n_ccres)

F=pd.DataFrame({lab:funnel(p) for p,lab in PICKERS}).T
F.index.name="picker"
print(F.to_string())""")

md(r"""## 1 · Summary of picks, associations, and locations — whole region vs UF-subregion

Two scopes side by side. **Whole region** = everything the year-2016 pipeline produced (all picks, all
PyOcto-associated events over the full search region, all HypoInverse locations). **UF-subregion** = the
Ulsan-Fault box (129.25–129.55 °E, 35.60–35.90 °N) — the members that enter the relocation. The whole-region
numbers show raw picker productivity; the UF numbers show what actually feeds the fault study.""")

co(r"""def region_summary(p):
    root,full,qc=paths(p)
    # picks (P/S) over the whole year
    nP=nS=0
    for m in range(1,13):
        f=f"{DT}/picks/picks_{p}_2016_{m:02d}.parquet"
        if os.path.exists(f):
            ph=pq.read_table(f,columns=["phase"]).column("phase").to_pandas().str.upper()
            nP+=int((ph=="P").sum()); nS+=int((ph=="S").sum())
    # all associated events over the whole search region (pyocto year file, unfiltered)
    yf=f"{root}/pyocto/pyocto_kim2011_2016.csv"
    reg_ev=len(pd.read_csv(yf)) if os.path.exists(yf) else \
           sum(len(pd.read_csv(f)) for f in sorted(glob.glob(f"{DT}/catalogs/catalog_{p}_2016_*_pyocto.csv")))
    # all HypoInverse-located events (whole region .sum has only UF members staged, so region-located == UF-box
    # located; we report the UF located count as the located scope)
    smf=f"{RUNS}/{full}/1.HypoInv/kim2011/{full}.sum"
    uf_located=len(uf.read_sum(smf)) if os.path.exists(smf) else np.nan
    uf_box=len(open(f"{root}/members.txt").read().split()) if os.path.exists(f"{root}/members.txt") else np.nan
    return dict(picks_P=nP,picks_S=nS,picks=nP+nS,region_events=reg_ev,
                ufbox_events=uf_box,uf_located=uf_located)

R=pd.DataFrame({lab:region_summary(p) for p,lab in PICKERS}).T
R.index.name="picker"
R["S/P"]=(R.picks_S/R.picks_P).round(2)
R["uf_frac_%"]=(100*R.ufbox_events/R.region_events).round(2)
print("WHOLE REGION (2016, full search area)")
print(R[["picks_P","picks_S","picks","S/P","region_events"]].to_string())
print("\nUF-SUBREGION (129.25-129.55E, 35.60-35.90N)")
print(R[["ufbox_events","uf_located","uf_frac_%"]].to_string())
print("\n  region_events = all PyOcto-associated events (whole region)")
print("  ufbox_events  = associated events inside the UF box (relocation members)")
print("  uf_located    = of those, HypoInverse-located (kim2011)")
print("  uf_frac_%     = ufbox_events / region_events")""")

md(r"""## 2 · The funnel — picks → events → UF-box → QC → dt.ct → dt.cc → cc-resolved

Each row is one picker; each column a stage of the pipeline. The picker sets the *input* (picks); the identical
downstream then filters. The interesting question is not who picks most, but who yields the most **cc-resolved,
well-located** events (last column).""")

co(r"""fig,ax=plt.subplots(figsize=(11,4.6))
stages=["picks","events","ufbox","QC","dtct","dtcc","ccres"]
labels=["Picks","Associated\nevents","UF-box","QC-\npassed","dt.ct\nreloc","dt.cc\nreloc","dt.cc-\nonly"]
cols={"PN+":"#4477aa","PN-original":"#ee6677","PN-STEAD":"#228833","EQT-STEAD":"#aa7733"}
x=np.arange(len(stages)); w=0.2
for i,(p,lab) in enumerate(PICKERS):
    ax.bar(x+(i-1.5)*w,[F.loc[lab,s] for s in stages],w,label=lab,color=cols[lab])
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("count (log)")
ax.legend(ncol=1,loc="upper right")
fig.tight_layout(); plt.show()
# yield ratios: QC-passed per 1000 picks (efficiency), cc-resolved / QC (relocatability)
print("\nEfficiency metrics:")
for p,lab in PICKERS:
    r=F.loc[lab]
    print(f"  {lab:12}  QC/1k-picks {1000*r.QC/r.picks:6.2f}   cc-res/QC {r.ccres/r.QC*100 if r.QC else 0:5.0f}%   "
          f"S/P {r.picks_S/r.picks_P:.2f}   picks/event {r.picks/r.events:.0f}")""")

md(r"""## 3 · Location quality — QC pass-rate and HypoInverse errors

A permissive picker that floods false events will have a LOW QC pass-rate (few of its UF-box events are
well-located) and larger location errors. This is where pick *quality* (not quantity) shows.""")

co(r"""qual=[]
for p,lab in PICKERS:
    root,full,qc=paths(p)
    sm=uf.read_sum(f"{RUNS}/{full}/1.HypoInv/kim2011/{full}.sum") if os.path.exists(f"{RUNS}/{full}/1.HypoInv/kim2011/{full}.sum") else None
    if sm is None: continue
    q=sm[(sm.erh<5)&(sm.erz<5)&(sm.gap<270)&(sm.num>5)&(sm.rms<1.0)]
    cc=rd_reloc(f"{RUNS}/{qc}/2.HypoDD/02.dt.cc/hypoDD.reloc")
    qual.append(dict(picker=lab, ufbox=len(sm), QC=len(q), qc_pct=100*len(q)/len(sm),
                     med_gap=sm.gap.median(), med_erh=sm.erh.median(), med_erz=sm.erz.median(),
                     med_num=sm.num.median(),
                     dtcc_relerr_m=(np.median(np.hypot(cc.ex,cc.ey)) if cc is not None else np.nan)))  # ex/ey already in m
Q=pd.DataFrame(qual).set_index("picker")
print(Q.round(2).to_string())
fig,ax=plt.subplots(1,3,figsize=(14,4))
labs=list(Q.index); c=[cols[l] for l in labs]
ax[0].bar(labs,Q.qc_pct,color=c); ax[0].set_title("QC pass-rate (% of UF-box events well-located)"); ax[0].set_ylabel("%")
ax[1].bar(labs,Q.med_gap,color=c); ax[1].set_title("Median azimuthal gap (°)"); ax[1].axhline(270,ls=":",c="0.5")
ax[2].bar(labs,Q.med_erh,color=c); ax[2].set_title("Median ERH (km)"); ax[2].axhline(5,ls=":",c="0.5")
for a in ax: a.tick_params(axis="x",rotation=20)
fig.tight_layout(); plt.show()""")

md(r"""## 4 - Spatial comparison: the dt.cc catalog from each picker (2x2)

Same region, fault trace, coastline. Depth-coloured, equal markers. **First grid:** all dt.cc-relocated events per
picker. **Second grid:** the **cc-resolved** subset only (>=1 surviving cross-correlation link), the highest-
precision events. Do the pickers resolve the same fault structure, or does a noisier picker smear it?""")

co(r"""def four_maps(resolved, title):
    fig = pygmt.Figure()
    with pygmt.config(MAP_FRAME_TYPE="plain", FORMAT_GEO_MAP="ddd.xx"):
        pygmt.makecpt(cmap="turbo", series=[2, 20], reverse=True)
        with fig.subplot(nrows=2, ncols=2, figsize=("16c", "19c"), margins=["0.7c", "1.2c"],
                         frame=["WSne", "xa0.1f0.05", "ya0.1f0.05"], title=title):
            for j, (p, lab) in enumerate(PICKERS):
                root, full, qc = paths(p)
                cc = rd_reloc(f"{RUNS}/{qc}/2.HypoDD/02.dt.cc/hypoDD.reloc")
                if resolved and cc is not None: cc = cc[(cc.nccp + cc.nccs) > 0]
                n = len(cc) if cc is not None else 0
                with fig.set_panel(j):
                    fig.basemap(region=REGION, projection="M?", frame=[f"+t{lab}  n={n}"])
                    if os.path.exists(COAST): fig.plot(data=COAST, pen="0.5p,60")
                    if os.path.exists(FAULT): fig.plot(data=FAULT, pen="0.8p,black")
                    if cc is not None and n:
                        fig.plot(x=cc.lon, y=cc.lat, style="c0.11c", fill=cc.depth, cmap=True, pen="0.2p,gray30")
                    fig.basemap(map_scale="jTL+w10k+o0.3c/0.4c+c35.75")   # top-left, clear of axis labels
        fig.colorbar(frame=["x+lDepth (km)"], position="JBC+o0c/1.3c+w10c/0.4c+h")
    fig.show(width=760)

four_maps(False, "All dt.cc-relocated events")""")

co(r"""four_maps(True, "dt.cc-resolved events only (>=1 cc link)")""")

md(r"""## 5 · Temporal evolution — cumulative dt.cc-only events through 2016

Cumulative count of **dt.cc-only** events versus time, one staircase per picker. "dt.cc-only" = the events with
at least one surviving cross-correlation differential-time link (`nccp+nccs > 0`) in the dt.ct+dt.cc HypoDD run —
the highest-precision subset, the same events drawn in the §4 second map grid (**not** all dt.ct+dt.cc-relocated
events). Origin time comes from the `hypoDD.reloc` yr/mo/dy/hr/mi/sc fields (UTC). The 2016-09-12 Gyeongju
mainshock dominates every curve: a flat pre-September background then a near-vertical aftershock jump. The
comparison shows whether the pickers track the same temporal history and how much of each catalog is the
September sequence.""")

co(r"""def reloc_times(p):
    root,full,qc=paths(p)
    cc=rd_reloc(f"{RUNS}/{qc}/2.HypoDD/02.dt.cc/hypoDD.reloc")
    if cc is None or not len(cc): return None
    cc=cc[(cc.nccp+cc.nccs)>0]                        # dt.cc-only: >=1 surviving cross-correlation link
    if not len(cc): return None
    t=pd.to_datetime(dict(year=cc.yr.astype(int),month=cc.mo.astype(int),day=cc.dy.astype(int),
                          hour=cc.hr.astype(int),minute=cc.mi.astype(int),
                          second=cc.sc.clip(0,59).astype(int)),errors="coerce",utc=True)
    return t.dropna().sort_values().reset_index(drop=True)

fig,ax=plt.subplots(figsize=(11,5))
GJ=pd.Timestamp("2016-09-12 11:32:54",tz="UTC")   # Gyeongju M5.5 mainshock (UTC)
for p,lab in PICKERS:
    t=reloc_times(p)
    if t is None: continue
    ax.step(t, np.arange(1,len(t)+1), where="post", lw=2.0, color=cols[lab], label=f"{lab} (n={len(t)})")
ax.axvline(GJ, ls="--", lw=1.2, color="0.4")
ax.text(GJ-pd.Timedelta(days=2), ax.get_ylim()[1]*0.97, "Gyeongju M5.5", rotation=90,
        va="top", ha="right", fontsize=9, color="0.3")
ax.set_xlim(pd.Timestamp("2016-01-01",tz="UTC"), pd.Timestamp("2017-01-01",tz="UTC"))
ax.set_ylabel("Cumulative dt.cc-relocated events", fontsize=13); ax.set_xlabel("Time", fontsize=13)
ax.legend(loc="upper left"); fig.tight_layout(); plt.show()

# pre- vs post-Gyeongju split of each dt.cc catalog
print("dt.cc-relocated events, pre- vs post-Gyeongju (2016-09-12):")
for p,lab in PICKERS:
    t=reloc_times(p)
    if t is None: continue
    pre=int((t<GJ).sum()); post=len(t)-pre
    print(f"  {lab:12}  total {len(t):4d}   pre-GJ {pre:4d}   post-GJ {post:4d}  ({100*post/len(t):.0f}% aftershocks)")""")

md(r"""## 6 · Used stations — before vs after the Gyeongju temporary deployment

Side-by-side maps of the stations actually used in 2016, split by the Gyeongju mainshock. **Before** = union of
stations with data in the pre-GJ months (Jan–Aug 2016); **after** = union of stations with data in the GJ months
(Sep–Dec 2016), when KIGAM/KMA deployed the dense temporary **GJ** array. Stations are coloured by network:
**KS = KMA** (national permanent), **KG = KIGAM** (permanent), **GJ = temporary (Gyeongju aftershock array)**.
The station geometry is the reason the post-GJ relocations (and the September dt.cc sequence) are so much
better-resolved — the azimuthal gap collapses once the temporary array is in.""")

co(r"""# used stations: coverage>0 in the pre-GJ months (Jan-Aug) vs GJ months (Sep-Dec), union over months
def used_union(months):
    S=pd.concat([pd.read_csv(f"{DT}/cache/stations_2016_{m:02d}.csv") for m in months])
    S=S[S.coverage>0].drop_duplicates("sta").reset_index(drop=True)
    return S[["net","sta","lat","lon"]]
before=used_union(range(1,9)); after=used_union(range(9,13))
NETC={"KS":"#4477aa","KG":"#228833","GJ":"#ee6677"}      # KMA / KIGAM / temporary(GJ)
NETL={"KS":"KS (KMA)","KG":"KG (KIGAM)","GJ":"GJ (temporary)"}
print(f"BEFORE (Jan-Aug): {len(before)} stations  {before.net.value_counts().to_dict()}")
print(f"AFTER  (Sep-Dec): {len(after)} stations  {after.net.value_counts().to_dict()}")

fig=pygmt.Figure()
with pygmt.config(MAP_FRAME_TYPE="plain", FORMAT_GEO_MAP="ddd.xx", FONT_TITLE="12p"):
    with fig.subplot(nrows=1, ncols=2, figsize=("17c","9c"), margins=["1.0c","0.6c"],
                     frame=["WSne","xa0.1f0.05","ya0.1f0.05"]):
        for j,(St,ttl) in enumerate([(before,f"Before GJ (Jan-Aug)   n={len(before)}"),
                                     (after, f"After GJ (Sep-Dec)   n={len(after)}")]):
            with fig.set_panel(j):
                fig.basemap(region=REGION, projection="M?", frame=[f"+t{ttl}"])
                if os.path.exists(COAST): fig.plot(data=COAST, pen="0.5p,60")
                if os.path.exists(FAULT): fig.plot(data=FAULT, pen="0.8p,black")
                for net in ["KS","KG","GJ"]:
                    d=St[St.net==net]
                    if len(d):
                        lab=NETL[net] if j==1 else None      # legend once (right panel)
                        fig.plot(x=d.lon, y=d.lat, style="t0.32c", fill=NETC[net], pen="0.4p,black",
                                 label=lab)
                fig.basemap(map_scale="jBL+w10k+o0.4c/0.4c+c35.75")
                if j==1: fig.legend(position="jTR+o0.2c", box="+gwhite+p0.5p,black+c0.15c")
fig.show(width=900)""")

md(r"""The two panels above are cropped to the UF fault box, so the wider KS/KMA and KG/KIGAM permanent stations
(which anchor the azimuthal coverage) fall off the edge. The version below uses a region padded to enclose **all**
used stations, showing the full network geometry that actually locates the events. The dashed rectangle marks the
UF box.""")

co(r"""# same station sets, but a region wide enough to show EVERY used station
FULL=[128.50, 129.65, 35.15, 36.80]     # pad of ~0.15 deg around the 46-station extent (128.6-129.5, 35.25-36.70)
UFR=[129.25,129.55,35.60,35.90]         # the tight UF box, drawn as a reference rectangle
fig=pygmt.Figure()
with pygmt.config(MAP_FRAME_TYPE="plain", FORMAT_GEO_MAP="ddd.xx", FONT_TITLE="12p"):
    with fig.subplot(nrows=1, ncols=2, figsize=("17c","11c"), margins=["0.3c","0.6c"],
                     frame=["WSne","xa0.5f0.25","ya0.5f0.25"]):
        for j,(St,ttl) in enumerate([(before,f"Before GJ (Jan-Aug)   n={len(before)}"),
                                     (after, f"After GJ (Sep-Dec)   n={len(after)}")]):
            with fig.set_panel(j):
                fig.basemap(region=FULL, projection="M?", frame=[f"+t{ttl}"])
                fig.coast(shorelines="0.4p,60", resolution="h", area_thresh=50)
                if os.path.exists(FAULT): fig.plot(data=FAULT, pen="0.8p,black")
                fig.plot(x=[UFR[0],UFR[1],UFR[1],UFR[0],UFR[0]],
                         y=[UFR[2],UFR[2],UFR[3],UFR[3],UFR[2]], pen="0.9p,gray30,--")  # UF box
                for net in ["KS","KG","GJ"]:
                    d=St[St.net==net]
                    if len(d):
                        fig.plot(x=d.lon, y=d.lat, style="t0.30c", fill=NETC[net], pen="0.4p,black")  # no legend
                fig.basemap(map_scale="jBL+w20k+o0.5c/0.5c+c36.0")
fig.show(width=900)""")

md(r"""## 7 · Summary""")
co(r"""bar="="*112
print(bar); print("2016 ULSAN-FAULT 4-PICKER COMPARISON (consistent 0.2 threshold)".center(112)); print(bar)
print(F[["picks","events","ufbox","QC","dtct","dtcc","ccres"]].to_string())
best_ev=F.events.idxmax(); best_cc=F.ccres.idxmax(); best_qc=(F.QC/F.picks).idxmax()
print(f'''
TAKE-HOMES
  - Most associated events: {best_ev}.  Most cc-resolved (well-located) events: {best_cc}.
  - Highest QC efficiency (QC per pick): {best_qc}.
  - Raw pick yield does NOT predict cc-resolved yield: the pickers reorder between the first and last columns.
  - STEAD-retrained models (PN-STEAD, EQT-STEAD) are far more conservative than NCEDC/EQNet (PN-original, PN+)
    at a fixed 0.2 threshold -> a fixed probability threshold is not equivalent across pickers.
  - S/P pick ratio is a picker fingerprint (PN-STEAD S-heavy, PN-original P-heavy); the association gate
    (>=2 P AND >=2 S) rebalances it.
NEXT: threshold-matched comparison (tune each picker's threshold to equal event yield), and ML-magnitude of the
cc-resolved catalogs.''')
print(bar)""")

nb=nbf.v4.new_notebook(); nb["cells"]=C
nb["metadata"]={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}
OUT="/home/msseo/works/02.Ulsan_Fault_detection/detection_test/reloc_2016_uf/10.Picker_comparison_2016.ipynb"
nbf.write(nb,OUT); print("wrote",OUT)
