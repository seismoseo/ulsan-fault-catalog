#!/usr/bin/env python
"""Generate 35.UF_cluster_svd_top8count.ipynb — per-volume LSQR-CND relocation + global-bootstrap uncertainties for
the 8 largest clusters BY MEMBER COUNT (the nb32 set; NND labels 0-7 under descending-size labeling). Companion to
nb33 (top-2 by magnitude) and nb34 (ranks 3-10 by magnitude), ranked here by member count instead. Four of the
eight families (0,1,6,7) coincide with the by-magnitude set but every volume is relocated independently in its own
svd_volumes_top8count/ tree so this notebook is self-contained. Same methodology and house style as nb33/nb34.

Reported solution = LSQR-CND seeded on the whole-box LSQR locations (light damping, CND 40-80). Absolute centroid
depth is a DD near-null-space direction — undamped SVD wanders down it, LSQR-CND holds the physical whole-box
centroid and gives the same relative structure. Bootstrap = global resampling, LSQR-CND, n=200; ez95 = RELATIVE
precision (absolute centroid ~km, uncaptured).

Reads ONLY cached runner outputs (uf_subregion_hypodd/svd_volumes_top8count/c{0..7}/): hypoDD.reloc (LSQR-CND),
hypoDD.reloc.svd (SVD diagnostic), bootstrap/*, analysis/*, volume_events.csv, run_meta.json. Base env.

Sections: (0) method + master summary table; (1) absolute-depth null space (SVD vs LSQR-CND vs whole-box), all 8;
(2) per-volume E/N/depth sections w/ 95% bootstrap bars; (3) uncertainty distributions; (4) KG.HDB Z/N/E waveform
gathers + CC matrices (coherence of each volume, gated on data); (5) summary."""
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# Per-volume relocation of the 8 largest clusters by member count — companion to nb33 / nb34

The same per-volume analysis nb33 (top-2 by magnitude) and nb34 (ranks 3–10 by magnitude) applied to the **8
richest NND families by member count** — the nb32 set (Cluster 0–7 under descending-size labeling). Ranking by
**member count** rather than mainshock magnitude surfaces the most-developed sequences on the fault regardless of
their largest event: four families (0, 1, 6, 7) coincide with the by-magnitude set, the other four (the count-rich
but moderate-magnitude 2, 3, 4, 5) are new here. The current NND labels are listed by the load cell below and
tracked via `volumes.txt`, since they change when the catalog is re-relocated. For each, the full population — the
1 km³ context cube **unioned with every NND family member** (so no grouped event, including a cc-starved mainshock,
is ever omitted) — is re-relocated stand-alone.

**Reported solver — LSQR-CND seeded on the whole-box LSQR locations** (see nb33 §1 for the full argument). The
differential data pins *relative* positions tightly but barely constrains the *absolute centroid depth* — a
near-null-space direction. Undamped SVD slides the whole cloud down it (several volumes drift ~2–4 km in absolute depth), so SVD
is kept only as the §1 diagnostic; light-damped LSQR (CND 40–80) holds the physical whole-box centroid and gives
the same relative structure. **Bootstrap = global resampling, LSQR-CND, `n = 200`;** `ez95` is **relative
precision**, not the ~1–2 km absolute-depth ambiguity (§1).

**Caveat — some of these volumes are small** (a few have only 4–5 events): with few differential times the
internal structure and the absolute depth are weakly constrained (large `ez95`, low CND). The master table flags
`n`, drops, CND, and median `ez95` per volume so the well- and poorly-constrained volumes are distinguishable.
House style follows nb33 / `analysis/relocation`: hollow circles coloured by origin time, grey 95% bootstrap bars,
depth positive-down.""")

# ------------------------------------------------------------------ §0 setup + load
co(r"""import warnings; warnings.filterwarnings("ignore")
import os, json, glob
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.font_manager as fm
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams["axes.unicode_minus"]=False
mpl.rcParams.update({"figure.dpi":130,"axes.grid":True,"grid.alpha":0.3,"font.size":10,
                     "legend.framealpha":1,"legend.edgecolor":"black","legend.facecolor":"white"})
BASE="/home/msseo/works/02.Ulsan_Fault_detection/analysis/uf_subregion_hypodd/svd_volumes_top8count"
WF100=("/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/"
       "uf_subregion_reuse/waveforms_100km")
MEIDX="/home/msseo/works/02.Ulsan_Fault_detection/analysis/uf_subregion_hypodd/uf_subregion/members_event_idx.csv"
RC=["id","lat","lon","depth","x","y","z","ex","ey","ez","yr","mo","dy","hr","mi","sc","mag",
    "nccp","nccs","nctp","ncts","rcc","rct","cid"]
# volumes ordered by mainshock magnitude (descending) — read the CURRENT set from the runner's volumes.txt
# (NND family labels change when the catalog is re-relocated; same physical clusters, new c<k> labels)
_vtxt="/home/msseo/works/02.Ulsan_Fault_detection/analysis/uf_subregion_hypodd/svd_volumes_top8count/volumes.txt"
VOLS=[ln.split()[0] for ln in open(_vtxt)] if os.path.exists(_vtxt) else ["c11","c4","c40","c41","c1","c105","c8","c95"]
CCBAND=(5,20)   # HypoDD dt.cc cross-correlation bandpass (Hz, 4 corners, zerophase)
def mag_size(m,smin=25,smax=1500): return np.clip(5*np.exp(2*np.asarray(m,float)),smin,smax)
def time_colors(t):
    tn=mdates.date2num(pd.to_datetime(t).dt.to_pydatetime()); nrm=plt.Normalize(tn.min(),tn.max())
    return plt.cm.coolwarm(nrm(tn)), nrm, tn
def time_colorbar(fig,ax,nrm):
    sm=plt.cm.ScalarMappable(norm=nrm,cmap="coolwarm"); sm.set_array([])
    cb=fig.colorbar(sm,ax=ax,fraction=0.03,pad=0.02); tk=np.linspace(nrm.vmin,nrm.vmax,5)
    cb.set_ticks(tk); cb.set_ticklabels([mdates.num2date(x).strftime("%Y-%m") for x in tk]); cb.set_label("Origin time")
    return cb
_mei=pd.read_csv(MEIDX); _mei["ts"]=pd.to_datetime(_mei.time,utc=True,format="ISO8601").dt.floor("s").dt.strftime("%Y%m%d%H%M%S")
E2TS=dict(zip(_mei.event_idx.astype(int),_mei.ts))

D={}; TITLE={}
for v in VOLS:
    d=f"{BASE}/{v}"
    ve=pd.read_csv(f"{d}/volume_events.csv"); ve["time"]=pd.to_datetime(ve.time,utc=True,format="ISO8601")
    rel=pd.read_csv(f"{d}/hypoDD.reloc",sep=r"\s+",header=None,names=RC)
    be=pd.read_csv(f"{d}/bootstrap/bootstrap_errors.csv",comment="#")
    eq=pd.read_csv(f"{d}/analysis/event_quality.csv")
    z=np.load(f"{d}/bootstrap/bootstrap_samples.npz")["data"]
    S=pd.DataFrame(z,columns=["ri","id","x","y","zz"]); S["id"]=S.id.astype(int)
    meta=json.load(open(f"{d}/run_meta.json")); summ=json.load(open(f"{d}/analysis/summary.json"))
    bmeta=json.load(open(f"{d}/bootstrap/bootstrap_meta.json"))
    svd=pd.read_csv(f"{d}/hypoDD.reloc.svd",sep=r"\s+",header=None,names=RC) if os.path.exists(f"{d}/hypoDD.reloc.svd") else None
    m=rel.merge(ve[["id","event_idx","cat","nnd_cluster","magU","time","suspect"]],on="id",how="left")
    m=m.merge(be[["id","n_boot","ex95","ey95","ez95"]],on="id",how="left")
    m=m.merge(eq[["id","wb_shift_m","in_volume"]],on="id",how="left")
    m["ts"]=m.event_idx.map(lambda e: E2TS.get(int(e)) if pd.notna(e) else None)
    D[v]=dict(ve=ve,rel=m,svd=svd,S=S,meta=meta,summ=summ,bmeta=bmeta)
    TITLE[v]=f"{v} — {meta.get('famsize','?')} members, M{meta['mainshock_mag']:.2f} ({str(meta['mainshock_time'])[:10]})"
    print(f"{v}: M{meta['mainshock_mag']:.2f} | reloc {meta['n_relocated']}/{meta['n_in_cube']} drop {meta['n_dropped']} | "
          f"{meta['primary_solver']} CND {meta['primary_cnd']} depth {np.median(rel.depth):.2f}km "
          f"(SVD {np.median(svd.depth):.2f}km) | ez95 med {be.ez95.median():.0f}m nbmin {int(be.n_boot.min())} | shape {summ['fit_main']['shape']}")""")

md(r"""## 0 · Master summary table

One row per volume: mainshock, in-cube / relocated / dropped counts, reported LSQR-CND damping + condition number,
median absolute depth (and the SVD-wander value for contrast), median relative-precision `ez95`, shape descriptor,
and the whole-box→LSQR-CND shift count. Read the small/weak volumes (large `ez95`, low CND, few events) with care.""")
co(r"""rows=[]
for v in VOLS:
    m=D[v]["rel"]; meta=D[v]["meta"]; summ=D[v]["summ"]; fm_=summ["fit_main"]; bm=D[v]["bmeta"]
    ok=m[m.ez95.notna()]
    rows.append(dict(vol=v,M=round(meta["mainshock_mag"],2),n_cube=meta["n_in_cube"],n_reloc=meta["n_relocated"],
        drop=meta["n_dropped"],damp=meta.get("primary_damp"),cnd=meta.get("primary_cnd"),
        depth_km=round(float(np.nanmedian(m.depth)),2),svd_depth_km=round(float(np.nanmedian(D[v]["svd"].depth)),2),
        rel95_E=round(float(ok.ex95.median()),1),rel95_N=round(float(ok.ey95.median()),1),rel95_Z=round(float(ok.ez95.median()),1),
        shape=fm_["shape"],L_km=f"{fm_['L1']}x{fm_['L2']}x{fm_['L3']}",
        boot_fail=round(bm["failed_frac"]*100,1),nb_min=int(m.n_boot.min()),big_shift=len(summ["big_shift_ids"])))
SUM=pd.DataFrame(rows)
print(SUM.to_string(index=False))
print("\ndepth_km = reported LSQR-CND absolute depth; svd_depth_km = where undamped SVD wandered (null-space).")
print("rel95_* = median RELATIVE-precision 95% half-width (m); absolute centroid depth is a separate ~km null space.")""")

# ------------------------------------------------------------------ §1 absolute depth null space
md(r"""## 1 · Absolute-depth null space (why LSQR-CND, not SVD) — all 8 volumes

Per-event absolute depth from three solutions: **whole-box LSQR** (physical anchor), reported **LSQR-CND**
(light damping, whole-box seed), and **undamped SVD**. Where the SVD (red dashed) sits far from the whole-box /
LSQR-CND (grey / green) the cloud has slid down the poorly-constrained centroid-depth direction — most extreme in
**c4** (SVD ~7 km vs physical ~11 km). LSQR-CND holds the physical depth in every case while preserving the
internal spread.""")
co(r"""fig,axs=plt.subplots(2,4,figsize=(17,8))
for ax,v in zip(axs.ravel(),VOLS):
    m=D[v]["rel"]; ve=D[v]["ve"].set_index("id"); sv=D[v]["svd"].set_index("id")
    ax.set_facecolor("#FAFAFA"); ax.grid(True,ls=":",alpha=0.6)
    wbd=ve.depth.reindex(m.id.values).values; cnd=m.depth.values; svd=sv.depth.reindex(m.id.values).values
    lo=np.nanmin([wbd,cnd,svd])-0.1; hi=np.nanmax([wbd,cnd,svd])+0.1; bins=np.linspace(lo,hi,24)
    ax.hist(wbd,bins=bins,color="0.6",alpha=0.7,label=f"whole-box ({np.nanmedian(wbd):.2f})")
    ax.hist(cnd,bins=bins,histtype="step",lw=2.2,color="tab:green",label=f"LSQR-CND ({np.nanmedian(cnd):.2f})")
    ax.hist(svd,bins=bins,histtype="step",lw=2.2,color="tab:red",ls="--",label=f"SVD ({np.nanmedian(svd):.2f})")
    ax.set(title=TITLE[v],xlabel="Absolute depth (km)"); ax.legend(fontsize=6.5,loc="upper left")
fig.suptitle("Absolute depth is a DD null space: undamped SVD wanders (seed-dependent), LSQR-CND holds the physical whole-box centroid",y=1.005)
fig.tight_layout(); plt.show()
for v in VOLS:
    wbd=np.nanmedian(D[v]["ve"].set_index("id").depth); sv=np.nanmedian(D[v]["svd"].depth); cn=np.nanmedian(D[v]["rel"].depth)
    print(f"{v}: whole-box {wbd:.2f} km | LSQR-CND {cn:.2f} km | SVD {sv:.2f} km  -> SVD wander {abs(sv-wbd)*1000:.0f} m")""")

# ------------------------------------------------------------------ §2 sections
md(r"""## 2 · East / North / depth sections with 95% bootstrap uncertainties — per volume

Map view (East–North) + two depth sections per volume, km relative to the cluster centroid (East/North) and
absolute depth (vertical), equal aspect. Hollow circles sized by ML, **coloured by origin time**; grey **95%
bootstrap relative-precision bars**; ★ = mainshock; events that relocated **out of the 1 km volume** drawn as red
× at the frame edge (excluded from the shape descriptor only). The printed shape descriptor (L1×L2×L3, gate:
planar / linear / blob) says whether the internal structure is resolved above the bootstrap precision.""")
co(r"""for v in VOLS:
    m=D[v]["rel"]; inv=m.in_volume.fillna(True).values
    cE=np.median(m.x.values[inv]); cN=np.median(m.y.values[inv])
    E=(m.x.values-cE)/1000; N=(m.y.values-cN)/1000; Z=m.depth.values
    exk,eyk,ezk=(m.ex95/1000).values,(m.ey95/1000).values,(m.ez95/1000).values
    cols,nrm,_=time_colors(m.time); sz=mag_size(m.magU)
    star=m.magU.idxmax(); zc=float(np.median(Z[inv])); half=max(np.ptp(E[inv]),np.ptp(N[inv]),np.ptp(Z[inv]))*0.75+0.05
    fig,axs=plt.subplots(1,3,figsize=(16.5,5.2))
    P=[(E,N,exk,eyk,"East offset (km)","North offset (km)",0.0,0.0,False),
       (E,Z,exk,ezk,"East offset (km)","Depth (km)",0.0,zc,True),
       (N,Z,eyk,ezk,"North offset (km)","Depth (km)",0.0,zc,True)]
    for ax,(X,Y,XE,YE,xl,yl,xc,yc,flip) in zip(axs,P):
        ax.set_facecolor("#FAFAFA"); ax.grid(True,ls=":",alpha=0.7)
        ax.errorbar(X[inv],Y[inv],xerr=XE[inv],yerr=YE[inv],fmt="none",ecolor="0.55",elinewidth=0.6,capsize=1.5,zorder=2)
        ax.scatter(X[inv],Y[inv],s=sz[inv],facecolors="none",edgecolors=cols[inv],linewidth=1.8,zorder=4)
        if (~inv).any(): ax.scatter(np.clip(X[~inv],xc-half,xc+half),np.clip(Y[~inv],yc-half,yc+half),
                                    s=40,marker="x",c="red",lw=1.3,zorder=5,label=f"relocated out ({int((~inv).sum())})")
        ax.scatter(X[star],Y[star],s=320,marker="*",color="red",edgecolor="k",lw=0.6,zorder=6)
        ax.set_xlim(xc-half,xc+half); ax.set_ylim((yc+half,yc-half) if flip else (yc-half,yc+half))
        ax.set(xlabel=xl,ylabel=yl); ax.set_aspect("equal")
        if (~inv).any(): ax.legend(loc="upper left",fontsize=8)
    time_colorbar(fig,axs,nrm)
    fm_=D[v]["summ"]["fit_main"]
    fig.suptitle(f"{TITLE[v]} — East/North/depth, 95% global-bootstrap bars (LSQR-CND, n={D[v]['bmeta']['n']}); "
                 f"shape {fm_['shape']}, widths {fm_['L1']}×{fm_['L2']}×{fm_['L3']} km",y=1.02)
    plt.show()
    med=np.nanmedian(np.c_[exk[inv],eyk[inv],ezk[inv]]*1000,axis=0); L3m=fm_["L3"]*1000
    print(f"{v}: {int(inv.sum())} in-volume | median rel-precision E {med[0]:.0f}/N {med[1]:.0f}/depth {med[2]:.0f} m | "
          f"shortest axis {L3m:.0f} m = {L3m/max(med.min(),1):.0f}x precision -> "
          f"{'RESOLVED 3-D shape' if L3m>3*med.min() else 'near location error (blob)'}")""")

# ------------------------------------------------------------------ §3 uncertainty
md(r"""## 3 · Uncertainty — relative-precision 95% half-widths (E / N / depth)

Distribution of the per-event 95% bootstrap half-widths across the 8 volumes (global resampling, LSQR-CND). These
are **relative precision** (event-to-event); the absolute centroid depth carries a separate ~1–2 km null-space
uncertainty (§1). Small volumes (few links) show the widest half-widths.""")
co(r"""fig,axs=plt.subplots(1,3,figsize=(15,4.6))
for k,(bc,lab) in enumerate([("ex95","East"),("ey95","North"),("ez95","Depth")]):
    ax=axs[k]
    allv=np.concatenate([D[v]["rel"][bc].dropna().values for v in VOLS])
    hi=np.nanpercentile(allv,95)*1.15+1; bins=np.linspace(0,hi,28)
    for v in VOLS:
        x=D[v]["rel"][bc].dropna().values
        ax.hist(x,bins=bins,histtype="step",lw=1.4,label=f"{v} ({np.median(x):.0f})")
    ax.set(xlabel=f"{lab} 95% bootstrap half-width (m)",ylabel="events",title=lab)
    if k==2: ax.legend(fontsize=7,ncol=2)
fig.suptitle("Per-event relative-precision 95% half-widths across the 8 volumes (global LSQR-CND bootstrap, n=200)",y=1.02)
fig.tight_layout(); plt.show()
print("Volume median relative-precision (E/N/depth, m) and the well- vs weakly-constrained split:")
for v in VOLS:
    m=D[v]["rel"]; zz=m.ez95.median()
    print(f"  {v}: E {m.ex95.median():.0f} / N {m.ey95.median():.0f} / depth {zz:.0f} m  ({'well' if zz<30 else 'moderate' if zz<80 else 'WEAK'} constrained)")""")

# ------------------------------------------------------------------ §4 waveform gathers + CC matrices
md(r"""## 4 · KG.HDB waveform gathers + cross-correlation matrices — volume coherence

The definitive test of whether a 1 km³ volume is a coherent repeating source: waveform similarity at **KG.HDB**
(the common close station), **5–20 Hz** (the HypoDD dt.cc band), P-aligned, events **ordered by origin time**. For
each volume with enough KG.HDB records: the HHZ gather (left; blue dashed = P, blue bars = S) and the HHZ/HHN/HHE
CC matrices (seismic colormap). Bright blocks = groups of near-identical waveforms = co-located repeaters. This
is where the persistent-volume question is answered — one of these is a known long-lived repeater site.""")
co(r"""from obspy import read
def load_hdb(ts,comp,band=CCBAND,win=(-0.5,3.0)):
    p=f"{WF100}/{ts}/{ts}.KG.HDB.HH{comp}.sac"
    if not os.path.exists(p): return None
    tr=read(p)[0]; tr.detrend("demean"); tr.detrend("linear"); tr.taper(0.05)
    tr.filter("bandpass",freqmin=band[0],freqmax=band[1],corners=4,zerophase=True)
    sac=tr.stats.sac; a=sac.get("a",-12345.0)
    if a==-12345.0: return None
    pt=tr.stats.starttime+(a-sac.get("b",0.0)); seg=tr.slice(pt+win[0],pt+win[1])
    if seg.stats.npts<10: return None
    d=seg.data.astype(float); d=d-d.mean(); mx=np.abs(d).max()
    t0=sac.get("t0",-12345.0); s_off=(t0-a) if t0!=-12345.0 else None
    return (np.linspace(win[0],win[1],len(d)), d/mx if mx>0 else d, s_off)
def load_arr(ts,comp,band=CCBAND,win=(-0.2,2.5),sr=100):
    p=f"{WF100}/{ts}/{ts}.KG.HDB.HH{comp}.sac"
    if not os.path.exists(p): return None
    tr=read(p)[0]; tr.detrend("demean"); tr.detrend("linear"); tr.taper(0.05)
    tr.filter("bandpass",freqmin=band[0],freqmax=band[1],corners=4,zerophase=True)
    sac=tr.stats.sac; a=sac.get("a",-12345.0)
    if a==-12345.0: return None
    pt=tr.stats.starttime+(a-sac.get("b",0.0)); seg=tr.slice(pt+win[0],pt+win[1]); L=int(round((win[1]-win[0])*sr))
    d=seg.data.astype(float)[:L]
    if len(d)<L: d=np.pad(d,(0,L-len(d)))
    d=d-d.mean(); nn=np.linalg.norm(d); return d/nn if nn>0 else None
def cc_matrix(arrs,maxlag=20):
    n=len(arrs); M=np.full((n,n),np.nan); idx=[i for i,a in enumerate(arrs) if a is not None]
    if len(idx)<2: return M,idx
    W=np.array([arrs[i] for i in idx]); G=W@W.T
    for lag in range(1,maxlag+1):
        Wp=np.zeros_like(W); Wp[:,:-lag]=W[:,lag:]; G=np.maximum(G,Wp@W.T)
        Wm=np.zeros_like(W); Wm[:,lag:]=W[:,:-lag]; G=np.maximum(G,Wm@W.T)
    for a,ia in enumerate(idx):
        for b,ib in enumerate(idx): M[ia,ib]=G[a,b]
    return M,idx
for v in VOLS:
    m=D[v]["rel"].sort_values("time").reset_index(drop=True); ts=m.ts.tolist(); n=len(m)
    fig=plt.figure(figsize=(17,max(4.2,0.16*n+1)))
    gs=fig.add_gridspec(1,4,width_ratios=[1.1,1,1,1])
    axg=fig.add_subplot(gs[0,0]); axg.set_facecolor("#FAFAFA"); nP=0
    for i,r in m.iterrows():
        wf=load_hdb(r["ts"],"Z") if isinstance(r["ts"],str) else None
        if wf is None: continue
        t,d,s_off=wf; nP+=1; axg.plot(t,-i+0.45*d,color="0.15",lw=0.5)
        if s_off is not None and -0.5<s_off<3.0: axg.plot([s_off,s_off],[-i-0.32,-i+0.32],color="tab:blue",lw=0.9,zorder=3)
    axg.axvline(0,color="tab:blue",ls="--",lw=0.8); axg.set(xlabel="Time from P (s)",title=f"HHZ gather (P={nP}/{n})",xlim=(-0.5,3.0))
    axg.set_yticks(-np.arange(n)); axg.set_yticklabels([f"{str(r.time)[:10]} M{r.magU:.1f}" for _,r in m.iterrows()],fontsize=5); axg.set_ylim(-n,1)
    cmap=plt.cm.seismic.copy(); cmap.set_bad("0.85"); medtxt=[]
    for j,comp in enumerate("ZNE"):
        ax=fig.add_subplot(gs[0,1+j]); arrs=[load_arr(t,comp) if isinstance(t,str) else None for t in ts]
        M,idx=cc_matrix(arrs); im=ax.imshow(np.ma.masked_invalid(M),vmin=0,vmax=1,cmap=cmap,origin="upper")
        ax.set(title=f"HH{comp} CC",xlabel="event (time order)")
        if j==0: ax.set_ylabel("event (time order)")
        fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
        if len(idx)>=2:
            off=M.copy(); np.fill_diagonal(off,np.nan); medtxt.append(f"HH{comp} {np.nanmedian(off):.2f}")
    fig.suptitle(f"{TITLE[v]} — KG.HDB waveform gather + CC matrices ({CCBAND[0]}–{CCBAND[1]} Hz), time-ordered | median off-diag CC: {', '.join(medtxt)}",y=1.01)
    fig.tight_layout(); plt.show()
    print(f"{v}: {nP}/{n} events with a KG.HDB P pick; median off-diagonal CC {', '.join(medtxt) if medtxt else 'n/a (too few)'}")""")

# ------------------------------------------------------------------ §4b PCA plane fits
md(r"""## 4b · PCA plane fits — strike / dip / length / width (bootstrap errors < 100 m)

The same error-gated PCA plane fit used in nb33 §3b, for each of the 8 volumes: **only events with 95% bootstrap
half-width < 100 m on every axis** enter the fit; PC1/PC2 span the plane (length / width), PC3 is the normal
(strike / dip); 95% CIs from re-fitting the same events on every bootstrap replica. The shape gate (planar iff
L2≥0.4·L1 and L3≤0.35·L2) is printed for each — where it says *blob* the strike/dip are indicative only. at least one volume the gate calls a genuine thin **plane**, so its strike/dip are physically meaningful; the small
the smallest volumes have too few gated events to fit. Below the table: the clear map-view-strike + across-strike-
dip read-out for every volume with a fittable plane.""")
co(r"""def pca_plane(E,N,Z):
    P=np.c_[np.asarray(E,float),np.asarray(N,float),np.asarray(Z,float)]
    if len(P)<3: return None
    c=P.mean(0); Q=P-c; sv,Vt=np.linalg.svd(Q,full_matrices=False)[1:]; pr=Q@Vt.T
    L1,L2,L3=(pr.max(0)-pr.min(0))
    n_=Vt[2]; n_=-n_ if n_[2]<0 else n_
    dip=float(np.degrees(np.arccos(abs(n_[2])/np.linalg.norm(n_))))
    strike=float((np.degrees(np.arctan2(n_[0],n_[1]))+90)%180)
    l1,l2,l3=2*sv/np.sqrt(len(P)-1); planar=bool(l2>=0.4*l1 and l3<=0.35*l2)
    return dict(strike=strike,dip=dip,length=float(L1),width=float(L2),thickness=float(L3),planar=planar,
                shape=("planar" if planar else "linear" if l2<0.4*l1 else "blob"),center=c,pc1=Vt[0],pc2=Vt[1],normal=n_)
def branch(vals,ref):
    v=np.asarray(vals,float); return np.where(v-ref>90,v-180,np.where(v-ref<-90,v+180,v))
ERRGATE=100.0; PF={}
for v in VOLS:
    m=D[v]["rel"]; S=D[v]["S"]
    emax=np.nanmax(np.c_[m.ex95,m.ey95,m.ez95],axis=1); keep=(emax<ERRGATE)&m.ez95.notna()
    ids=set(m.loc[keep,"id"].astype(int)); sub=m[keep]
    fit=pca_plane(sub.x.values,sub.y.values,sub.depth.values*1000.0) if keep.sum()>=3 else None
    reps={"strike":[],"dip":[],"length":[],"width":[]}
    if fit is not None:
        for ri,gS in S.groupby("ri"):
            gg=gS[gS.id.isin(ids)]
            if len(gg)<3: continue
            f=pca_plane(gg.x.values,gg.y.values,gg.zz.values)
            if f: [reps[k].append(f[k]) for k in reps]
    ci={}
    if fit is not None and len(reps["strike"])>=10:
        ci["strike"]=[float(np.percentile(branch(reps["strike"],fit["strike"]),p)) for p in (2.5,97.5)]
        for k in ("dip","length","width"): ci[k]=[float(np.percentile(reps[k],p)) for p in (2.5,97.5)]
    PF[v]=dict(fit=fit,ci=ci,n=int(keep.sum()),ids=ids)
rows=[]
for v in VOLS:
    P=PF[v]; f=P["fit"]
    if f is None: rows.append(dict(vol=v,n_gated=P["n"],note="<3 gated events")); continue
    ci=P["ci"]
    rows.append(dict(vol=v,n_gated=P["n"],strike=round(f["strike"]),
        strike95=(f"{ci['strike'][0]:.0f}-{ci['strike'][1]:.0f}" if ci.get("strike") else "-"),
        dip=round(f["dip"]),dip95=(f"{ci['dip'][0]:.0f}-{ci['dip'][1]:.0f}" if ci.get("dip") else "-"),
        length_m=round(f["length"]),width_m=round(f["width"]),thick_m=round(f["thickness"]),shape=f["shape"]))
print(pd.DataFrame(rows).to_string(index=False))
print("\nStrike/dip physically meaningful only where shape=planar. Blobs = indicative; tiny clusters unfittable.")""")

md(r"""Fault-frame read-out (benchmarked to `analysis/relocation` batch_summary §8b) for each fittable volume:
(A) map view (strike solid / across-strike dashed), (B) across-strike depth section B–B′ (dashed = dip, +across =
down-dip = deeper), (C) along-dip view A–A′ with 10 MPa rupture circles to scale. Hollow circles coloured by
origin time (per-panel bar), sized by ML, 95% bootstrap bars in-frame. Meaningful where shape=planar.""")
co(r"""from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
def _srad9(m,dsig=10e6): M0=10.0**(1.5*np.asarray(m,float)+9.1); return (7.0*M0/(16.0*dsig))**(1/3.)
def _ff_project(fit,sel,S):
    nE,nN=float(fit["normal"][0]),float(fit["normal"][1]); h=np.hypot(nE,nN)
    ua=np.array([-nE/h,-nN/h]) if h>1e-9 else np.array([1.,0.]); uv=np.array([-ua[1],ua[0]])
    x0,y0,z0=sel.x.mean(),sel.y.mean(),sel.depth.mean()*1000
    e=sel.x.values-x0; n=sel.y.values-y0; zk=(sel.depth.values*1000-z0)/1000
    across=(e*ua[0]+n*ua[1])/1000; along=(e*uv[0]+n*uv[1])/1000
    cd,sd=np.cos(np.radians(fit["dip"])),np.sin(np.radians(fit["dip"])); along_dip=across*cd+zk*sd
    dirs={"al":np.array([uv[0],uv[1],0.]),"ac":np.array([ua[0],ua[1],0.]),"dp":np.array([0.,0.,1.]),
          "ad":np.array([ua[0]*cd,ua[1]*cd,sd]),"e":np.array([1.,0.,0.]),"n":np.array([0.,1.,0.])}
    sig={k:np.full(len(sel),np.nan) for k in dirs}
    for i,eid in enumerate(sel.id.astype(int)):
        s=S[S.id==eid]
        if len(s)>=10:
            Pm=np.c_[s.x-s.x.mean(),s.y-s.y.mean(),s.zz-s.zz.mean()]
            for k,vv in dirs.items(): sig[k][i]=(np.percentile(Pm@vv,97.5)-np.percentile(Pm@vv,2.5))/2/1000
    return dict(along=along,across=across,dep=zk,along_dip=along_dip,e=e/1000,n=n/1000,ua=ua,uv=uv,sig=sig)
def _cbar(fig,ax,nrm):
    sm=plt.cm.ScalarMappable(norm=nrm,cmap="coolwarm"); sm.set_array([])
    cb=fig.colorbar(sm,ax=ax,fraction=0.046,pad=0.03); tk=np.linspace(nrm.vmin,nrm.vmax,3)
    cb.set_ticks(tk); cb.set_ticklabels([mdates.num2date(t).strftime("%y-%m") for t in tk]); cb.ax.tick_params(labelsize=6)
def fault_frame_fig(lab,fit,sel,S,ci):
    Pj=_ff_project(fit,sel,S); cv=mdates.date2num(pd.to_datetime(sel.time).dt.to_pydatetime())
    nrm=mcolors.Normalize(cv.min(),cv.max() if cv.max()>cv.min() else cv.min()+1); rgba=plt.get_cmap("coolwarm")(nrm(cv))
    sz=mag_size(sel.magU)*0.45; rk=_srad9(sel.magU.values)/1000; ud=fit["dip"]
    fig,axs=plt.subplots(1,3,figsize=(15.5,5.0),constrained_layout=True)
    ax=axs[0]; X,Y=Pj["e"],Pj["n"]; lim=1.12*max(np.abs(X).max(),np.abs(Y).max(),1e-3)
    ax.errorbar(X,Y,xerr=Pj["sig"]["e"],yerr=Pj["sig"]["n"],fmt="none",ecolor="0.55",elinewidth=0.5,capsize=1,zorder=3)
    ax.scatter(X,Y,s=sz,facecolors="none",edgecolors=rgba,linewidth=1.2,zorder=4)
    uv,ua=Pj["uv"],Pj["ua"]
    ax.plot([-lim*uv[0],lim*uv[0]],[-lim*uv[1],lim*uv[1]],"0.35",lw=1.2,zorder=2)
    ax.plot([-lim*ua[0],lim*ua[0]],[-lim*ua[1],lim*ua[1]],"0.35",lw=1.0,ls="--",zorder=2)
    ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel="E (km)",ylabel="N (km)",title="map view (strike)")
    ax=axs[1]; X,Y=Pj["across"],Pj["dep"]; lim=1.12*max(np.abs(X).max(),np.abs(Y).max(),1e-3)
    ax.errorbar(X,Y,xerr=Pj["sig"]["ac"],yerr=Pj["sig"]["dp"],fmt="none",ecolor="0.55",elinewidth=0.5,capsize=1,zorder=3)
    ax.scatter(X,Y,s=sz,facecolors="none",edgecolors=rgba,linewidth=1.2,zorder=4)
    xx=np.linspace(-lim,lim,30); ax.plot(xx,np.tan(np.radians(ud))*xx,"k--",lw=0.9,zorder=1)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.invert_yaxis()
    ax.text(0.04,0.06,"B",transform=ax.transAxes,fontsize=12,fontweight="bold"); ax.text(0.90,0.06,"B'",transform=ax.transAxes,fontsize=12,fontweight="bold")
    ax.set(xlabel="across-strike, down-dip → (km)",ylabel="depth (km)",title=f"⟂ strike section (dip {ud:.0f}°)")
    ax=axs[2]; X,Y=Pj["along"],Pj["along_dip"]; lim=1.12*max((np.abs(X)+rk).max(),(np.abs(Y)+rk).max(),1e-3)
    ax.add_collection(PatchCollection([Circle((X[i],Y[i]),rk[i]) for i in range(len(X))],facecolors="none",edgecolors=rgba,linewidths=1.2,zorder=4))
    ax.errorbar(X,Y,xerr=Pj["sig"]["al"],yerr=Pj["sig"]["ad"],fmt="none",ecolor="0.55",elinewidth=0.5,capsize=1,zorder=3)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.invert_yaxis()
    ax.text(0.04,0.06,"A",transform=ax.transAxes,fontsize=12,fontweight="bold"); ax.text(0.90,0.06,"A'",transform=ax.transAxes,fontsize=12,fontweight="bold")
    ax.set(xlabel="along-strike (km)",ylabel="along-dip (km)",title="along-dip view (10 MPa circles)")
    for ax in axs: ax.set_aspect("equal","box"); ax.grid(True,ls=":",alpha=0.6); ax.set_facecolor("#FAFAFA"); ax.tick_params(labelsize=8)
    _cbar(fig,axs[2],nrm)
    s_ci=(f" [{ci['strike'][0]:.0f}–{ci['strike'][1]:.0f}]" if ci.get('strike') else "")
    d_ci=(f" [{ci['dip'][0]:.0f}–{ci['dip'][1]:.0f}]" if ci.get('dip') else "")
    fig.suptitle(f"{lab} — strike {fit['strike']:.0f}°{s_ci} / dip {fit['dip']:.0f}°{d_ci} | "
                 f"L {fit['length']:.0f} × W {fit['width']:.0f} × thick {fit['thickness']:.0f} m | shape {fit['shape']}",fontsize=12)
    plt.show()
for v in VOLS:
    P=PF[v]; fit=P["fit"]
    if fit is None or P["n"]<4: continue
    m=D[v]["rel"]; emax=np.nanmax(np.c_[m.ex95,m.ey95,m.ez95],axis=1)
    sel=m[(emax<ERRGATE)&m.ez95.notna()&(m.id.isin(P["ids"]))].reset_index(drop=True)
    fault_frame_fig(TITLE[v],fit,sel,D[v]["S"],P["ci"])
print("Fault-frame views benchmarked to analysis/relocation batch_summary §8b (map / across-strike B-B' / along-dip A-A').")""")

md(r"""**Strike / dip statistics across the fittable volumes** (batch_summary style): strike rose (planar/linear fits only), dip histogram, and a strike-vs-dip scatter sized by n and coloured by planarity (L3/L2, greener = flatter = more sheet-like).""")
co(r"""GD=[]
for v in VOLS:
    f=PF[v]["fit"]
    if f is None: continue
    l1,l2,l3=f["length"],f["width"],f["thickness"]; flat=(l3/l2 if l2>0 else 1.0)
    GD.append(dict(vol=v,strike=f["strike"],dip=f["dip"],n=PF[v]["n"],flat=flat,shape=f["shape"]))
GD=pd.DataFrame(GD)
fig=plt.figure(figsize=(14,4.2))
ax1=fig.add_subplot(1,3,1,projection="polar"); ax1.set_theta_zero_location("N"); ax1.set_theta_direction(-1)
ax1.hist(np.radians(np.concatenate([GD.strike,GD.strike+180])),bins=np.radians(np.arange(0,361,20)),color="steelblue",edgecolor="k",linewidth=0.4)
ax1.set_title(f"Strike rose (n={len(GD)} fits)",pad=14)
ax2=fig.add_subplot(1,3,2); ax2.hist(GD.dip,bins=np.arange(0,91,10),color="indianred",edgecolor="k",linewidth=0.4)
ax2.axvline(GD.dip.median(),color="k",ls="--",label=f"median {GD.dip.median():.0f}°"); ax2.legend(fontsize=8)
ax2.set(xlabel="Dip (°)",ylabel="Volumes",title="Dip distribution")
ax3=fig.add_subplot(1,3,3)
sc=ax3.scatter(GD.strike,GD.dip,s=25+GD.n*3,c=GD.flat,cmap="RdYlGn_r",edgecolor="k",linewidths=0.4,vmin=0,vmax=1)
for _,r in GD.iterrows(): ax3.annotate(r.vol,(r.strike,r.dip),fontsize=7,ha="left",va="bottom")
ax3.set(xlabel="Strike (°)",ylabel="Dip (°)",title="Strike vs dip (size∝n, colour=L3/L2)",xlim=(0,180),ylim=(0,90))
fig.colorbar(sc,ax=ax3,fraction=0.046,pad=0.04,label="L3/L2 (flatness)")
fig.tight_layout(); plt.show()
print(GD.to_string(index=False))""")

# ------------------------------------------------------------------ §5 summary
md(r"""## 5 · Summary""")
co(r"""print("="*140); print("PER-VOLUME LSQR-CND RELOCATION (whole-box seed) + GLOBAL BOOTSTRAP — 8 largest UF clusters by member count (nb32 set)".center(140)); print("="*140)
print(SUM.to_string(index=False))
print("\nTAKE-HOMES")
# biggest SVD absolute-depth drift, chosen from the data (labels change when the catalog is re-relocated)
_drift={v:abs(np.nanmedian(D[v]['svd'].depth)-np.nanmedian(D[v]['rel'].depth)) for v in VOLS}
_vd=max(_drift,key=_drift.get)
print(" - Reported solver = LSQR-CND on the whole-box seed (physical absolute depth); undamped SVD wanders down the")
print(f"   DD null-space depth direction — largest here in {_vd} (SVD %.1f km vs physical %.1f km). Same relative shape."%(
      np.nanmedian(D[_vd]['svd'].depth),np.nanmedian(D[_vd]['rel'].depth)))
_ez={v:D[v]['rel'].ez95.median() for v in VOLS}
_well=[v for v in VOLS if _ez[v]<30]; _weak=[v for v in VOLS if _ez[v]>=80]
print(f" - Global bootstrap ez95 = RELATIVE precision. Well-constrained (ez95<30 m): {', '.join(_well) or 'none'};")
print(f"   weakly-constrained (ez95>=80 m, few events): {', '.join(_weak) or 'none'} — read their structure with care.")
shp={v:D[v]['summ']['fit_main']['shape'] for v in VOLS}
print(f" - Shape gate: {sum(s=='planar' for s in shp.values())} planar, {sum(s=='linear' for s in shp.values())} linear, "
      f"{sum(s=='blob' for s in shp.values())} blob -> "+", ".join(f"{v}:{s}" for v,s in shp.items()))
print(" - §4 CC matrices show which volumes are coherent repeater families (bright blocks) vs diffuse — high")
print("   within-volume CC + long duration = a chronically reactivating patch (the known repeater site is in here).")
print("\nNEXT: HDBSCAN/DBSCAN density volumes; Phase-2 relative-amplitude ML for the ML-less members; repeater deep-dive.")""")

nb["cells"]=C
import os
os.chdir("/home/msseo/works/02.Ulsan_Fault_detection/analysis/reloc_analysis")
nbf.write(nb,"35.UF_cluster_svd_top8count.ipynb")
print("wrote 35.UF_cluster_svd_top8count.ipynb",len(C),"cells")
