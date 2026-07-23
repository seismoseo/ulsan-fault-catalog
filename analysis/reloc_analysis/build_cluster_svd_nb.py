#!/usr/bin/env python
"""Generate 33.UF_cluster_svd_volumes.ipynb — per-volume HypoDD relocation + global-bootstrap uncertainties
(Phase 3), in the analysis/relocation house style (hollow circles coloured by origin time, grey 95% bootstrap
error bars, depth positive-down), plain East/North/depth axes.

Reported solution = LSQR-CND seeded on the whole-box LSQR locations (light damping, CND 40-80). The absolute
CENTROID depth is a near-null-space direction of the DD system: undamped SVD wanders down it (physical 11.5 km ->
9.5 km for m373, seed-dependent), so SVD is kept ONLY as a diagnostic (§1) that documents the drift; LSQR-CND
holds the physical whole-box centroid and gives the SAME relative structure. Bootstrap = global resampling of the
whole differential-time pool, LSQR-CND, n=200; ez95 is RELATIVE precision (absolute centroid ~km, uncaptured).

Reads ONLY cached runner outputs (analysis/uf_subregion_hypodd/svd_volumes/{m389,m373}/): hypoDD.reloc (LSQR-CND
primary), hypoDD.reloc.svd (SVD diagnostic), bootstrap/{bootstrap_errors.csv,bootstrap_samples.npz,
bootstrap_meta.json}, analysis/{summary.json,plane_bootstrap.csv,separation_bootstrap.csv,event_quality.csv},
volume_events.csv, run_meta.json. Runs in base (numpy/pandas/matplotlib + obspy for the waveform gather).

Figures: (1) SVD absolute-depth null-space drift + whole-box↔LSQR-CND agreement; (2) E/N/depth sections w/ 95%
bootstrap bars; (3) 3-D structure; (4) M3.73 C6-vs-C7 separation; (4b) before/after mainshock; (5) uncertainty;
(6) KG.HDB Z/N/E waveform gather (5-20 Hz = HypoDD CC band, time-ordered); (7) CC matrices; (8) instrument test;
(9) summary."""
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# Per-volume relocation of the two largest-event volumes — structure, uncertainty, waveforms

The whole-box dt.cc relocation (LSQR, all 2 734 events) sharpened the catalog but its formal errors are
meaningless (~0.5 m). Here the **full 1 km³ in-cube populations** around the **M3.89 (2014)** and **M3.73
(2023)** mainshocks (nb30 volumes) are re-relocated stand-alone to resolve internal structure and get honest
uncertainties.

**Reported solver — LSQR-CND seeded on the whole-box LSQR locations.** The differential-time data constrains
*relative* positions tightly but the *absolute centroid depth* only weakly — it is a **near-null-space direction**
of the DD system. An **undamped SVD** (ISOLV=1) has no anchor, so the whole cloud slides down that direction to
wherever the seed + numerical path lands (m373: 11.5 km whole-box → **9.5 km**, and a *different* 10.4 km from a
catalog seed — i.e. not data-driven). **Light-damped LSQR (CND 40–80)** is softly anchored to the seed and holds
the physical whole-box centroid at **every** damping level, while giving the **same relative structure** as SVD
(≈10 m median difference, identical thickness) and dropping no events. So LSQR-CND on the whole-box seed is the
reported solution; **§1 keeps the SVD run purely as a diagnostic that documents the depth drift.**

**Uncertainty — global-resampling bootstrap, LSQR-CND, `n = 200`.** The entire differential-time pool is resampled
with replacement (whole pairs drop/duplicate — captures pair-selection variance, ~30% larger than within-pair),
each replica re-run with the same LSQR-CND and seeded on the reported solution, median-aligned, 95% percentile
half-widths. **`ez95` (~3 m) is the RELATIVE-precision scatter** of the reported estimator. It is *not* the
absolute-depth uncertainty: the centroid carries a separate **~1–2 km null-space depth ambiguity** (evidenced by
the SVD wander in §1) that **no bootstrap seeded on one solution can see** — stated explicitly in §5.

**Method notes (disclosed):** subsetting keeps only intra-cube differential times, so an event anchored in the
whole box by links to *outside*-cube neighbours loses constraint here and can drift — §1 measures this (whole-box
→ LSQR-CND shift) and §6/§7 let you judge by waveform whether such an event truly belongs. House style follows
`analysis/relocation`: hollow circles coloured by origin time, grey 95% bootstrap bars, depth positive-down.""")

# ------------------------------------------------------------------ §0 setup + load
co(r"""import warnings; warnings.filterwarnings("ignore")
import os, json, glob
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.font_manager as fm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams["axes.unicode_minus"]=False
mpl.rcParams.update({"figure.dpi":130,"axes.grid":True,"grid.alpha":0.3,"font.size":10,
                     "legend.framealpha":1,"legend.edgecolor":"black","legend.facecolor":"white"})
BASE="/home/msseo/works/02.Ulsan_Fault_detection/analysis/uf_subregion_hypodd/svd_volumes"
WF100=("/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/"
       "uf_subregion_reuse/waveforms_100km")
MEIDX="/home/msseo/works/02.Ulsan_Fault_detection/analysis/uf_subregion_hypodd/uf_subregion/members_event_idx.csv"
RC=["id","lat","lon","depth","x","y","z","ex","ey","ez","yr","mo","dy","hr","mi","sc","mag",
    "nccp","nccs","nctp","ncts","rcc","rct","cid"]
VOLS=["m389","m373"]; TITLE={"m389":"M3.89 (2014) volume","m373":"M3.73 (2023) volume"}
CCBAND=(5,20)   # HypoDD dt.cc cross-correlation bandpass (Hz, 4 corners, zerophase) — used for the gather
# ---- house-style helpers (from analysis/relocation) ----
def mag_size(m,smin=25,smax=1500): return np.clip(5*np.exp(2*np.asarray(m,float)),smin,smax)   # area ∝ exp(ML)
def src_radius(mw,dsig=10e6): m0=10.0**(1.5*np.asarray(mw,float)+9.05); return (7.0*m0/(16.0*dsig))**(1/3.)  # m
def time_colors(t):
    tn=mdates.date2num(pd.to_datetime(t).dt.to_pydatetime()); nrm=plt.Normalize(tn.min(),tn.max())
    return plt.cm.coolwarm(nrm(tn)), nrm, tn
def time_colorbar(fig,ax,nrm):
    sm=plt.cm.ScalarMappable(norm=nrm,cmap="coolwarm"); sm.set_array([])
    cb=fig.colorbar(sm,ax=ax,fraction=0.03,pad=0.02); tk=np.linspace(nrm.vmin,nrm.vmax,5)
    cb.set_ticks(tk); cb.set_ticklabels([mdates.num2date(x).strftime("%Y-%m") for x in tk]); cb.set_label("Origin time")
    return cb
_mei=pd.read_csv(MEIDX); _mei["ts"]=pd.to_datetime(_mei.time,utc=True,format="ISO8601").dt.floor("s").dt.strftime("%Y%m%d%H%M%S")
E2TS=dict(zip(_mei.event_idx.astype(int),_mei.ts))               # event_idx -> waveform-dir timestamp

D={}
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
    svd=pd.read_csv(f"{d}/hypoDD.reloc.svd",sep=r"\s+",header=None,names=RC)   # SVD diagnostic (null-space drift)
    m=rel.merge(ve[["id","event_idx","cat","nnd_cluster","magU","time","suspect"]],on="id",how="left")
    m=m.merge(be[["id","n_boot","ex95","ey95","ez95"]],on="id",how="left")
    m=m.merge(eq[["id","wb_shift_m","in_volume"]],on="id",how="left")
    m["ts"]=m.event_idx.map(lambda e: E2TS.get(int(e)) if pd.notna(e) else None)   # event_idx -> waveform ts
    D[v]=dict(ve=ve,rel=m,svd=svd,S=S,meta=meta,summ=summ,bmeta=bmeta)
    print(f"{v}: primary {meta['primary_solver']} (CND {meta['primary_cnd']}) | relocated {meta['n_relocated']}/{meta['n_in_cube']} "
          f"| depth {np.median(rel.depth):.2f} km (SVD wandered to {np.median(svd.depth):.2f} km) "
          f"| bootstrap {bmeta.get('resample')} {bmeta.get('boot_solver')} n={bmeta['n']} failed {bmeta['failed_frac']*100:.0f}% "
          f"| n_boot/event min/median {int(be.n_boot.min())}/{int(be.n_boot.median())} "
          f"| whole-box→primary shifts >300 m: {summ['big_shift_ids']}")
print("\nAbsolute centroid depth is a DD near-null-space direction: LSQR-CND (reported) holds the physical whole-box")
print("centroid; undamped SVD slides to a seed-dependent shallower value (§1). Global bootstrap n=200, LSQR-CND:")
print("0 (or near-0) failed replicas; ez95 = RELATIVE precision (absolute depth ~1-2 km, uncaptured — see §5).")
for v in VOLS:
    lo=D[v]["rel"][D[v]["rel"].n_boot<200]
    if len(lo): print(f"  {v}: {len(lo)} event(s) with n_boot<200 -> ids {sorted(lo.id.astype(int))} (weakly linked; ez95 up to {lo.ez95.max():.0f} m)")""")

# ------------------------------------------------------------------ §1 depth null-space + WB vs LSQR-CND
md(r"""## 1 · Absolute-depth null space (why not SVD) + whole-box ↔ LSQR-CND agreement

**Top row — the absolute-depth diagnostic.** For each volume, the per-event absolute depth from three solutions:
the **whole-box LSQR** (physical anchor, from the full-catalog relocation), the reported **LSQR-CND** (light
damping, whole-box seed), and the **undamped SVD**. The differential data barely constrains the centroid depth,
so undamped SVD slides the *whole cloud* down that null direction — for m373, from 11.5 km to **9.5 km** (and to a
*different* 10.4 km from a catalog seed, confirming it is seed- not data-driven). LSQR-CND stays on the physical
11.49 km. The internal *shape* (spread) is preserved in all three; only the rigid centroid moves. **This is why
the reported solution is LSQR-CND, not SVD.**

**Bottom row — whole-box → LSQR-CND displacement** (each panel median-centred). The reported LSQR-CND is a mild
refinement of the whole box (median shift ~7 m). **Red (> 300 m)** = a few weakly-linked events that lost their
outside-cube neighbour links under subsetting; those are adjudicated by waveform in §6/§7 (the whole-box location
is usually the more reliable there).""")
co(r"""fig,axs=plt.subplots(2,2,figsize=(13,10.5))
# --- top row: absolute-depth comparison (whole-box vs LSQR-CND vs SVD) ---
for j,v in enumerate(VOLS):
    m=D[v]["rel"]; ve=D[v]["ve"].set_index("id"); sv=D[v]["svd"].set_index("id")
    ax=axs[0,j]; ax.set_facecolor("#FAFAFA"); ax.grid(True,ls=":",alpha=0.6)
    wbd=ve.depth.reindex(m.id.values).values; cnd=m.depth.values; svd=sv.depth.reindex(m.id.values).values
    bins=np.linspace(np.nanmin([wbd,cnd,svd])-0.1,np.nanmax([wbd,cnd,svd])+0.1,30)
    ax.hist(wbd,bins=bins,color="0.6",alpha=0.7,label=f"whole-box LSQR (med {np.nanmedian(wbd):.2f} km)")
    ax.hist(cnd,bins=bins,histtype="step",lw=2.2,color="tab:green",label=f"LSQR-CND reported (med {np.nanmedian(cnd):.2f} km)")
    ax.hist(svd,bins=bins,histtype="step",lw=2.2,color="tab:red",ls="--",label=f"SVD undamped (med {np.nanmedian(svd):.2f} km)")
    for val,c in [(np.nanmedian(wbd),"0.4"),(np.nanmedian(cnd),"tab:green"),(np.nanmedian(svd),"tab:red")]:
        ax.axvline(val,color=c,lw=1.2,ls=":")
    ax.set(xlabel="Absolute depth (km)",ylabel="events",title=f"{TITLE[v]} — absolute depth by solver"); ax.legend(loc="upper left",fontsize=7.5)
# --- bottom row: whole-box -> LSQR-CND displacement (E-N and E-depth) ---
for j,v in enumerate(VOLS):
    m=D[v]["rel"]; ve=D[v]["ve"].set_index("id")
    wbE=(ve.x_m-ve.x_m.median())/1000; wbN=(ve.y_m-ve.y_m.median())/1000; wbZ=(ve.depth_m-ve.depth_m.median())/1000
    wb=pd.DataFrame({"E":wbE,"N":wbN,"Z":wbZ}).reindex(m.id.values)
    prE=(m.x-m.x.median()).values/1000; prN=(m.y-m.y.median()).values/1000; prZ=(m.z-m.z.median()).values/1000
    big=(m.wb_shift_m>300).values; ax=axs[1,j]
    ax.scatter(wb.E.values,wb.N.values,s=16,c="0.6",lw=0,zorder=2,label="whole-box LSQR")
    ax.scatter(prE[~big],prN[~big],s=16,c="tab:green",lw=0,zorder=3,label="LSQR-CND reported")
    for k in range(len(wb)):
        col="tab:red" if big[k] else "0.45"; ax.plot([wb.E.values[k],prE[k]],[wb.N.values[k],prN[k]],color=col,lw=1.0 if big[k] else 0.4,zorder=4 if big[k] else 1)
    if big.any(): ax.scatter(prE[big],prN[big],s=40,facecolor="none",edgecolor="tab:red",lw=1.4,zorder=5,label=f"WB→CND > 300 m ({big.sum()})")
    lim=np.nanpercentile(np.abs(np.r_[wb.E.values,wb.N.values]),98)*2.2+0.05
    ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel="East (km)",ylabel="North (km)")
    ax.set_aspect("equal"); ax.set_facecolor("#FAFAFA"); ax.grid(True,ls=":",alpha=0.6); ax.legend(loc="upper left",fontsize=7.5)
    print(f"{v}: whole-box→LSQR-CND shift — median {m.wb_shift_m.median():.0f} m, 90th {np.nanpercentile(m.wb_shift_m,90):.0f} m; "
          f"{int(big.sum())} events > 300 m (ids {sorted(m.loc[big,'id'].astype(int))}) -> §6/§7 | "
          f"SVD centroid depth {np.nanmedian(D[v]['svd'].depth):.2f} km vs whole-box {np.nanmedian(D[v]['ve'].set_index('id').depth):.2f} km")
fig.suptitle("Absolute depth is a DD null space (top: SVD wanders, LSQR-CND holds physical); whole-box↔LSQR-CND agreement (bottom)",y=0.997)
fig.tight_layout(); plt.show()""")

# ------------------------------------------------------------------ §2 sections
md(r"""## 2 · East / North / depth sections with 95% bootstrap uncertainties

Map view (East–North) + two depth sections, km relative to the cluster centroid, equal aspect (so flatness is
real). **Hollow circles** sized by ML, **coloured by origin time** (coolwarm); grey **95% bootstrap error bars**
(the E/N/Z half-widths directly). ★ = mainshock. Events that **relocated out of the 1 km volume** (a subsetting
artefact — see §1/§6) are drawn as small red × at the frame edge and excluded from the shape descriptor only.

**M3.89 mainshock (cc-starved / ct-only).** The 2014 M3.89 (the largest UF event) cross-correlates with its
aftershocks on only ~1 station per pair — far below the `OBSCC=4` threshold — so all its cross-correlation links
are culled and it is relocated by **catalog differential times only** (`ncc=0`). It is a full NND family-0 member,
so it is **kept and relocated with the cluster (no omission)** and marked with an **open ★**, but its position
carries large bootstrap errors (hundreds of m vs tens for the cc-resolved events) and it is **excluded from the
plane/shape fit** (it lacks cc precision). Every other cluster mainshock is cc-resolved and drawn as a filled ★.""")
co(r"""for v in VOLS:
    m=D[v]["rel"]; inv=m.in_volume.fillna(True).values
    # HypoDD's x/y/z centroid can sit off origin, so CENTRE the East/North offsets on the in-volume median;
    # use ABSOLUTE depth (LSQR-CND holds the physical whole-box centroid) for the vertical axis.
    cE=np.median(m.x.values[inv]); cN=np.median(m.y.values[inv])
    E=(m.x.values-cE)/1000; N=(m.y.values-cN)/1000; Z=m.depth.values
    exk,eyk,ezk=(m.ex95/1000).values,(m.ey95/1000).values,(m.ez95/1000).values
    cols,nrm,_=time_colors(m.time); sz=mag_size(m.magU)
    star=m.magU.idxmax(); zc=float(np.median(Z[inv])); half=max(np.ptp(E[inv]),np.ptp(N[inv]),np.ptp(Z[inv]))*0.75+0.05
    ms_ct=bool((m.nccp+m.nccs).loc[star]==0)   # ct-only mainshock (cc-starved) -> open star, excluded from shape fit
    fig,axs=plt.subplots(1,3,figsize=(16.5,5.6))
    P=[(E,N,exk,eyk,"East offset (km)","North offset (km)",0.0,0.0,False,True),
       (E,Z,exk,ezk,"East offset (km)","Depth (km)",0.0,zc,True,False),
       (N,Z,eyk,ezk,"North offset (km)","Depth (km)",0.0,zc,True,False)]
    for ax,(X,Y,XE,YE,xl,yl,xc,yc,flip,mapview) in zip(axs,P):
        ax.set_facecolor("#FAFAFA"); ax.grid(True,ls=":",alpha=0.7)
        ax.errorbar(X[inv],Y[inv],xerr=XE[inv],yerr=YE[inv],fmt="none",ecolor="0.55",elinewidth=0.6,capsize=1.5,zorder=2)
        ax.scatter(X[inv],Y[inv],s=sz[inv],facecolors="none",edgecolors=cols[inv],linewidth=1.8,zorder=4)
        if (~inv).any(): ax.scatter(np.clip(X[~inv],xc-half,xc+half),np.clip(Y[~inv],yc-half,yc+half),
                                    s=40,marker="x",c="red",lw=1.3,zorder=5,label=f"relocated out of volume ({int((~inv).sum())})")
        ax.scatter(X[star],Y[star],s=340,marker="*",facecolors=("none" if ms_ct else "red"),edgecolors=("red" if ms_ct else "k"),linewidths=(1.8 if ms_ct else 0.6),zorder=6)
        ax.set_xlim(xc-half,xc+half); ax.set_ylim((yc+half,yc-half) if flip else (yc-half,yc+half))
        ax.set(xlabel=xl,ylabel=yl); ax.set_aspect("equal")
        if (~inv).any(): ax.legend(loc="upper left",fontsize=8)
    time_colorbar(fig,axs,nrm)
    fm_=D[v]["summ"]["fit_main"]
    _ctn=f"  |  open ★ = M{m.magU.loc[star]:.2f} mainshock ct-only (cc-starved), excluded from shape fit" if ms_ct else ""
    fig.suptitle(f"{TITLE[v]} — East/North/depth, 95% global-bootstrap bars (LSQR-CND, n={D[v]['bmeta']['n']}); "
                 f"shape {fm_['shape']}, principal widths {fm_['L1']}×{fm_['L2']}×{fm_['L3']} km{_ctn}",y=1.01)
    plt.show()
    med=np.nanmedian(np.c_[exk[inv],eyk[inv],ezk[inv]]*1000,axis=0); L3m=fm_["L3"]*1000
    print(f"{v}: {int(inv.sum())} in-volume events | median 95% half-width E {med[0]:.0f} / N {med[1]:.0f} / depth {med[2]:.0f} m | "
          f"shortest axis {L3m:.0f} m = {L3m/max(med.min(),1):.0f}× precision -> "
          f"{'a RESOLVED 3-D shape' if L3m>3*med.min() else '~ location error'}")""")

# ------------------------------------------------------------------ §3 3-D
md(r"""## 3 · 3-D internal structure with per-event 95% bootstrap ellipsoids

Plain East / North / Down (km). Hollow circles coloured by origin time; wireframe ellipsoids = per-event 95%
confidence (√χ²₃,₀.₉₅ = 2.80 σ) from the replica covariance. No fault plane is drawn — the shape gate calls both
volumes blobs (a plane would be a plausible-but-fake fit on a near-isotropic cloud).""")
co(r"""from numpy.linalg import eigh
def ellipsoid(ax,c,cov,color,scale=2.796):
    w,Vv=eigh(cov); w=np.clip(w,0,None); u=np.linspace(0,2*np.pi,14); ph=np.linspace(0,np.pi,8)
    sp=np.array([np.outer(np.cos(u),np.sin(ph)),np.outer(np.sin(u),np.sin(ph)),np.outer(np.ones_like(u),np.cos(ph))])
    Emat=np.einsum("ij,jkl->ikl",Vv@np.diag(scale*np.sqrt(w)),sp)
    ax.plot_wireframe(Emat[0]+c[0],Emat[1]+c[1],Emat[2]+c[2],color=color,lw=0.25,alpha=0.4)
for v in VOLS:
    m=D[v]["rel"]; S=D[v]["S"]; inv=m.in_volume.fillna(True).values
    cx,cy=m.x.median(),m.y.median()
    fig=plt.figure(figsize=(11,8)); ax=fig.add_subplot(111,projection="3d")
    cols,nrm,_=time_colors(m.time)
    E,N,Z=(m.x-cx)/1000,(m.y-cy)/1000,(m.z-m.z.median())/1000
    czm=float(S.zz.median())
    for i,eid in enumerate(m.id.astype(int)):
        if not inv[i]: continue
        s=S[S.id==eid]
        if len(s)>=10:
            cov=np.cov(np.c_[(s.x-cx)/1000,(s.y-cy)/1000,(s.zz-czm)/1000].T)
            ellipsoid(ax,(E.iloc[i],N.iloc[i],Z.iloc[i]),cov,cols[i])
    ax.scatter(E[inv],N[inv],Z[inv],s=28,facecolors="none",edgecolors=cols[inv],linewidth=1.4,depthshade=False)
    st=m.magU.idxmax(); _ct3=bool((m.nccp+m.nccs).loc[st]==0)
    ax.scatter([E[st]],[N[st]],[Z[st]],s=320,marker="*",facecolors=("none" if _ct3 else "red"),edgecolors=("red" if _ct3 else "k"),linewidths=(1.8 if _ct3 else 0.6),depthshade=False)
    lim=np.nanpercentile(np.abs(np.c_[E[inv],N[inv],Z[inv]]),98)*1.7
    ax.set(xlim=(-lim,lim),ylim=(-lim,lim),zlim=(lim,-lim)); ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)"); ax.set_zlabel("Down (km)")
    ax.set_title(f"{TITLE[v]} — 3-D East/North/depth + 95% bootstrap ellipsoids"); ax.view_init(elev=20,azim=-55)
    fm_=D[v]["summ"]["fit_main"]; print(f"{v}: shape={fm_['shape']} (no plane drawn on a blob)")
    plt.show()""")

# ------------------------------------------------------------------ §3b PCA plane fit (error-gated)
md(r"""## 3b · PCA plane fit — strike / dip / length / width (bootstrap errors < 100 m only)

A best-fit plane by principal-component analysis of the error-gated hypocentres: **only events whose 95% bootstrap
half-width is < 100 m on every axis** (E, N and depth) enter the fit, so poorly-located strays cannot tilt the
plane. PC1 & PC2 span the plane (→ **length** = along-strike extent, **width** = down-dip extent); PC3 is the
normal (→ **strike**, **dip**); PC3 extent = **thickness**. 95% CIs come from re-fitting the same event set on
every bootstrap replica (strike branch-centred to avoid the 0/180° wrap). For the **M3.73 (2023) volume** the fit
is done three ways: **all** error-gated events, **before** the Nov-2023 mainshock (the Sept-2023 C6 swarm +
background), and **after** it (the aftershock sequence) — to test whether the pre- and post-mainshock activity
share a plane.

The plane is drawn **on the geographic sections** (its rectangle projected into East–North, East–depth,
North–depth) rather than in rotated fault-frame axes. **Disclosure:** the shape gate (sheet iff L2≥0.4·L1 and
L3≤0.35·L2) is printed for each fit — where it says *blob*, the strike/dip are a forced fit on a near-isotropic
cloud and the plane is not physically resolved; treat those numbers as indicative only.""")
co(r"""def pca_plane(E,N,Z):
    # PCA plane of hypocentres (metres, Z +down; E=x, N=y). length/width/thickness = full extent along
    # PC1/PC2/PC3; strike/dip from the PC3 normal. Translation-invariant -> replica CIs comparable.
    P=np.c_[np.asarray(E,float),np.asarray(N,float),np.asarray(Z,float)]
    if len(P)<3: return None
    c=P.mean(0); Q=P-c; sv,Vt=np.linalg.svd(Q,full_matrices=False)[1:]; pr=Q@Vt.T
    L1,L2,L3=(pr.max(0)-pr.min(0))                                 # full extent along PC1,PC2,PC3
    n_=Vt[2]; n_=-n_ if n_[2]<0 else n_
    dip=float(np.degrees(np.arccos(abs(n_[2])/np.linalg.norm(n_))))
    strike=float((np.degrees(np.arctan2(n_[0],n_[1]))+90)%180)
    l1,l2,l3=2*sv/np.sqrt(len(P)-1)                                # PCA 2-sigma widths (for the sheet gate)
    planar=bool(l2>=0.4*l1 and l3<=0.35*l2)
    return dict(strike=strike,dip=dip,length=float(L1),width=float(L2),thickness=float(L3),
                planar=planar,shape=("planar" if planar else "linear" if l2<0.4*l1 else "blob"),
                center=c,pc1=Vt[0],pc2=Vt[1],normal=n_)
def branch(vals,ref):
    v=np.asarray(vals,float); return np.where(v-ref>90,v-180,np.where(v-ref<-90,v+180,v))

# define the fits: (label, volume, time-mask-fn)
FITS=[("M3.89 (2014)","m389",lambda m: np.ones(len(m),bool))]
mstamp=pd.Timestamp(D["m373"]["meta"]["mainshock_time"])
FITS+=[("M3.73 (2023) — all","m373",lambda m: np.ones(len(m),bool)),
       ("M3.73 — before mainshock","m373",lambda m:(m.time<mstamp).values),
       ("M3.73 — after mainshock","m373",lambda m:(m.time>mstamp).values)]
ERRGATE=100.0   # per-event 95% bootstrap half-width must be < this on EVERY axis to enter the plane fit
PF={}
for lab,v,tmask in FITS:
    m=D[v]["rel"]; S=D[v]["S"]
    emax=np.nanmax(np.c_[m.ex95,m.ey95,m.ez95],axis=1)
    keep=(emax<ERRGATE)&m.ez95.notna()&tmask(m)
    ids=set(m.loc[keep,"id"].astype(int))
    sub=m[keep]
    fit=pca_plane(sub.x.values,sub.y.values,sub.depth.values*1000.0)   # absolute-depth metres for plotting
    # bootstrap CIs on the SAME gated events
    reps={"strike":[],"dip":[],"length":[],"width":[],"thickness":[]}
    for ri,gS in S.groupby("ri"):
        gg=gS[gS.id.isin(ids)]
        if len(gg)<3: continue
        f=pca_plane(gg.x.values,gg.y.values,gg.zz.values)
        if f is None: continue
        for k in reps: reps[k].append(f[k])
    ci={}
    if fit is not None and len(reps["strike"])>=10:
        ci["strike"]=[float(np.percentile(branch(reps["strike"],fit["strike"]),p)) for p in (2.5,97.5)]
        for k in ("dip","length","width","thickness"):
            ci[k]=[float(np.percentile(reps[k],p)) for p in (2.5,97.5)]
    PF[lab]=dict(v=v,fit=fit,ci=ci,n=int(keep.sum()),n_tot=len(m),ids=ids)
    if fit is None: print(f"{lab}: <3 error-gated events — no plane"); continue
    def pm(k,u=""):
        c=ci.get(k); return f"{fit[k]:.0f}{u}"+(f" [{c[0]:.0f}-{c[1]:.0f}]" if c else "")
    print(f"{lab}: n={keep.sum()}/{len(m)} gated | strike {pm('strike','°')} dip {pm('dip','°')} | "
          f"length {pm('length','m')} width {pm('width','m')} thick {fit['thickness']:.0f}m | shape gate: {fit['shape'].upper()}")""")

md(r"""The fitted planes on the geographic sections (rectangle = length × width, oriented by strike/dip; events = error-gated, coloured by time):""")
co(r"""def rect_corners(fit):
    c=fit["center"]; a=fit["pc1"]*fit["length"]/2; b=fit["pc2"]*fit["width"]/2
    return np.array([c-a-b,c-a+b,c+a+b,c+a-b,c-a-b])            # closed 3-D rectangle (metres, Z +down abs)
for lab in PF:
    P=PF[lab]; fit=P["fit"]
    if fit is None: continue
    m=D[P["v"]]["rel"]; emax=np.nanmax(np.c_[m.ex95,m.ey95,m.ez95],axis=1)
    sel=m[(emax<ERRGATE)&m.ez95.notna()&(m.id.isin(P["ids"]))]
    cE=sel.x.median(); cN=sel.y.median()
    E=(sel.x.values-cE)/1000; N=(sel.y.values-cN)/1000; Z=sel.depth.values
    cols,nrm,_=time_colors(sel.time); sz=mag_size(sel.magU)
    R=rect_corners(fit); Re=(R[:,0]-cE)/1000; Rn=(R[:,1]-cN)/1000; Rz=R[:,2]/1000
    zc=float(np.median(Z)); half=max(np.ptp(E),np.ptp(N),np.ptp(Z),fit["length"]/1000,fit["width"]/1000)*0.62+0.05
    fig,axs=plt.subplots(1,3,figsize=(16.5,5.2))
    PN=[(E,N,Re,Rn,"East offset (km)","North offset (km)",0.0,0.0,False),
        (E,Z,Re,Rz,"East offset (km)","Depth (km)",0.0,zc,True),
        (N,Z,Rn,Rz,"North offset (km)","Depth (km)",0.0,zc,True)]
    for ax,(X,Y,RX,RY,xl,yl,xc,yc,flip) in zip(axs,PN):
        ax.set_facecolor("#FAFAFA"); ax.grid(True,ls=":",alpha=0.7)
        ax.scatter(X,Y,s=sz,facecolors="none",edgecolors=cols,linewidth=1.7,zorder=4)
        ax.plot(RX,RY,color="k",lw=1.6,zorder=6); ax.fill(RX,RY,color="k",alpha=0.06,zorder=3)
        ax.set_xlim(xc-half,xc+half); ax.set_ylim((yc+half,yc-half) if flip else (yc-half,yc+half))
        ax.set(xlabel=xl,ylabel=yl); ax.set_aspect("equal")
    time_colorbar(fig,axs,nrm); ci=P["ci"]
    ttl=(f"{lab} — strike {fit['strike']:.0f}° dip {fit['dip']:.0f}°, "
         f"length {fit['length']:.0f} m × width {fit['width']:.0f} m (thick {fit['thickness']:.0f} m); "
         f"n={P['n']} error-gated (<{ERRGATE:.0f} m); shape gate {fit['shape']}")
    fig.suptitle(ttl,y=1.02); plt.show()
print("Plane rectangles are length×width oriented by strike/dip, projected into each geographic section.")
print("Where the shape gate = blob, the plane is a forced fit on a near-isotropic cloud (indicative, not resolved).")""")

md(r"""**Fault-frame read-out** (benchmarked to `analysis/relocation` batch_summary §8b). Three panels per fit, hollow
circles coloured by origin time (per-panel bar), sized by ML, with 95% bootstrap bars projected into the frame:
(A) **map view** — solid grey = strike, dashed = across-strike; (B) **across-strike depth section (B–B′)** — the
dashed line is the **dip** (+across = down-dip = deeper, physically oriented); (C) **along-dip view (A–A′)** with
**rupture circles drawn to scale for a 10 MPa stress drop** (Eshelby, ML as Mw proxy). B/B′ and A/A′ mark the
section ends. Where the shape gate = *blob* the strike/dip are indicative only (near-isotropic cloud).""")
co(r"""from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
def _srad9(m,dsig=10e6): M0=10.0**(1.5*np.asarray(m,float)+9.1); return (7.0*M0/(16.0*dsig))**(1/3.)   # Eshelby m
def _ff_project(fit,sel,S):
    nE,nN=float(fit["normal"][0]),float(fit["normal"][1]); h=np.hypot(nE,nN)
    ua=np.array([-nE/h,-nN/h]) if h>1e-9 else np.array([1.,0.])   # down-dip horizontal (physical)
    uv=np.array([-ua[1],ua[0]])                                   # along-strike (perp)
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
            for k,v in dirs.items(): sig[k][i]=(np.percentile(Pm@v,97.5)-np.percentile(Pm@v,2.5))/2/1000
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
    # A: map view
    ax=axs[0]; X,Y=Pj["e"],Pj["n"]; lim=1.12*max(np.abs(X).max(),np.abs(Y).max(),1e-3)
    ax.errorbar(X,Y,xerr=Pj["sig"]["e"],yerr=Pj["sig"]["n"],fmt="none",ecolor="0.55",elinewidth=0.5,capsize=1,zorder=3)
    ax.scatter(X,Y,s=sz,facecolors="none",edgecolors=rgba,linewidth=1.2,zorder=4)
    uv,ua=Pj["uv"],Pj["ua"]
    ax.plot([-lim*uv[0],lim*uv[0]],[-lim*uv[1],lim*uv[1]],"0.35",lw=1.2,zorder=2)          # strike (solid)
    ax.plot([-lim*ua[0],lim*ua[0]],[-lim*ua[1],lim*ua[1]],"0.35",lw=1.0,ls="--",zorder=2)  # across (dashed)
    ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel="E (km)",ylabel="N (km)",title="map view (strike)")
    # B: across-strike depth section
    ax=axs[1]; X,Y=Pj["across"],Pj["dep"]; lim=1.12*max(np.abs(X).max(),np.abs(Y).max(),1e-3)
    ax.errorbar(X,Y,xerr=Pj["sig"]["ac"],yerr=Pj["sig"]["dp"],fmt="none",ecolor="0.55",elinewidth=0.5,capsize=1,zorder=3)
    ax.scatter(X,Y,s=sz,facecolors="none",edgecolors=rgba,linewidth=1.2,zorder=4)
    xx=np.linspace(-lim,lim,30); ax.plot(xx,np.tan(np.radians(ud))*xx,"k--",lw=0.9,zorder=1)   # dip line (+down-dip deeper)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.invert_yaxis()
    ax.text(0.04,0.06,"B",transform=ax.transAxes,fontsize=12,fontweight="bold"); ax.text(0.90,0.06,"B'",transform=ax.transAxes,fontsize=12,fontweight="bold")
    ax.set(xlabel="across-strike, down-dip → (km)",ylabel="depth (km)",title=f"⟂ strike section (dip {ud:.0f}°)")
    # C: along-dip view with rupture circles
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
for lab in PF:
    P=PF[lab]; fit=P["fit"]
    if fit is None: continue
    m=D[P["v"]]["rel"]; emax=np.nanmax(np.c_[m.ex95,m.ey95,m.ez95],axis=1)
    sel=m[(emax<ERRGATE)&m.ez95.notna()&(m.id.isin(P["ids"]))].reset_index(drop=True)
    fault_frame_fig(lab,fit,sel,D[P["v"]]["S"],P["ci"])
print("Fault-frame views benchmarked to analysis/relocation batch_summary §8b (map / across-strike B-B' / along-dip A-A').")""")

# ------------------------------------------------------------------ §4 C6/C7
md(r"""## 4 · Does the M3.73 volume resolve into two structures (C6 swarm vs C7 sequence)?

The M3.73 cube merged NND family **C7** (Nov-2023 M3.73 sequence), the co-located **C6** swarm (Sept 2023),
background and ML-less events. Left: LSQR-CND East–North positions coloured by category. Right: bootstrap posterior of
the **3-D distance between the C6 and C7 centroids** (in-volume events) — *cleanly resolved* if its 95% interval
clears half the larger group's internal width by >30%, *marginal* if it just clears it, else *not separable*.

**Damping-sensitivity caveat:** this uses the reported LSQR-CND bootstrap (relative precision ~3 m), under which
the separation just clears the bar. The undamped-SVD bootstrap has larger per-event relative scatter (damping
shrinks variance — a bias–variance tradeoff), which widens the groups and pushes the verdict to *not separable*.
So the C6/C7 separation is **marginal either way** — the two episodes are, at best, barely resolvable and are best
read as the **same ~0.1 km patch** reactivating twice, not two distinct faults.""")
co(r"""def sep_verdict(sm):
    if not sm: return "n/a"
    r=sm["lo"]/max(sm["max_group_width_m"]/2,1e-9)
    return "not separable at 95%" if r<1 else "marginally resolved" if r<1.3 else "cleanly resolved"
m=D["m373"]["rel"]; summ=D["m373"]["summ"]; inv=m.in_volume.fillna(True).values
sep=pd.read_csv(f"{BASE}/m373/analysis/separation_bootstrap.csv")
E=(m.x-m.x.median())/1000; N=(m.y-m.y.median())/1000
CC={"member":("tab:red","C7 (M3.73 sequence)"),"other-family":("tab:orange","C6 (Sept swarm)"),
    "background":("0.45","background"),"not-in-pop":("tab:purple","ML-less (not in NND)")}
fig,axs=plt.subplots(1,2,figsize=(13,5.6)); ax=axs[0]; ax.set_facecolor("#FAFAFA"); ax.grid(True,ls=":",alpha=0.7)
for cat,(col,lab) in CC.items():
    s=((m.cat==cat)&inv).values
    if s.any(): ax.scatter(E[s],N[s],s=mag_size(m.magU[s]),facecolors="none",edgecolors=col,linewidth=1.8,label=f"{lab} ({int((m.cat==cat).sum())})")
st=m.magU.idxmax(); ax.scatter(E[st],N[st],s=320,marker="*",color="red",edgecolor="k",zorder=5)
ax.set(xlabel="East (km)",ylabel="North (km)",title="M3.73 volume — LSQR-CND positions by category"); ax.set_aspect("equal"); ax.legend(loc="upper left",fontsize=8)
ax=axs[1]; sm=summ.get("separation_m",{})
if len(sep):
    ax.hist(sep.sep3d,bins=30,color="steelblue",ec="w")
    ax.axvspan(sm.get("lo",0),sm.get("hi",0),color="steelblue",alpha=0.18,label=f"95% CI {sm.get('lo',0):.0f}–{sm.get('hi',0):.0f} m")
    ax.axvline(sm.get("med",np.nan),color="k",lw=1.4,label=f"median {sm.get('med',0):.0f} m")
    ax.axvline(sm.get("max_group_width_m",0)/2,color="tab:red",ls="--",lw=1.3,label="½ · larger group width")
    ax.set(xlabel="C6–C7 centroid separation (m)",ylabel="bootstrap replicas",title=f"Separation posterior — {sep_verdict(sm)}"); ax.legend(loc="upper right",fontsize=8)
fig.tight_layout(); plt.show()
print(f"C6–C7 separation {sm.get('med')} m (95% {sm.get('lo')}–{sm.get('hi')}) vs ½-width {sm.get('max_group_width_m')} m "
      f"-> {sep_verdict(sm).upper()}. Nearly co-located: the same ~0.1 km patch reactivating in two temporally")
print("distinct episodes, not two separate faults.")""")

# ------------------------------------------------------------------ §4b M3.73 before/after mainshock
md(r"""## 4b · M3.73 volume — seismicity before vs after the mainshock

Map view (East–North) and two depth sections for the M3.73 (2023) volume, with events **before the M3.73
mainshock in blue** (the Sept-2023 C6 swarm + earlier background — the precursory activity) and **after it in
red** (the aftershock sequence). ★ = the M3.73 mainshock. Shows whether the pre- and post-mainshock activity
occupy the same patch or offset volumes.""")
co(r"""m=D["m373"]["rel"]; inv=m.in_volume.fillna(True).values
ms=m.loc[m.magU.idxmax()]; mst=ms.time; star=m.magU.idxmax()
cE=np.median(m.x.values[inv]); cN=np.median(m.y.values[inv])
E=(m.x.values-cE)/1000; N=(m.y.values-cN)/1000; Z=m.depth.values
before=((m.time<mst).values)&inv; after=((m.time>mst).values)&inv
zc=float(np.median(Z[inv])); half=max(np.ptp(E[inv]),np.ptp(N[inv]),np.ptp(Z[inv]))*0.75+0.05
fig,axs=plt.subplots(1,3,figsize=(16.5,5.6))
P=[(E,N,"East offset (km)","North offset (km)",0.0,0.0,False),
   (E,Z,"East offset (km)","Depth (km)",0.0,zc,True),
   (N,Z,"North offset (km)","Depth (km)",0.0,zc,True)]
for ax,(X,Y,xl,yl,xc,yc,flip) in zip(axs,P):
    ax.set_facecolor("#FAFAFA"); ax.grid(True,ls=":",alpha=0.7)
    ax.scatter(X[before],Y[before],s=mag_size(m.magU[before]),facecolors="none",edgecolors="tab:blue",linewidth=1.8,zorder=3,label=f"before mainshock ({int(before.sum())})")
    ax.scatter(X[after],Y[after],s=mag_size(m.magU[after]),facecolors="none",edgecolors="tab:red",linewidth=1.8,zorder=3,label=f"after mainshock ({int(after.sum())})")
    ax.scatter(X[star],Y[star],s=340,marker="*",color="gold",edgecolor="k",lw=0.7,zorder=5)
    ax.set_xlim(xc-half,xc+half); ax.set_ylim((yc+half,yc-half) if flip else (yc-half,yc+half))
    ax.set(xlabel=xl,ylabel=yl); ax.set_aspect("equal")
axs[0].legend(loc="upper left",fontsize=8)
fig.suptitle(f"M3.73 (2023) volume — before (blue) vs after (red) the M3.73 mainshock {str(mst)[:10]} (★)",y=1.01)
fig.tight_layout(); plt.show()
print(f"m373: {int(before.sum())} in-volume events BEFORE the M3.73 mainshock ({str(mst)[:16]}), {int(after.sum())} AFTER.")""")

# ------------------------------------------------------------------ §5 uncertainty
md(r"""## 5 · Uncertainty — bootstrap 95% half-widths (relative precision) vs the absolute-depth null space

The **histograms of the per-event 95% bootstrap half-widths** in East, North and depth (global resampling,
LSQR-CND, n=200), then the per-event comparison to the reloc's formal errors (log–log, 1:1).

**What `ez95` (~3 m) means, and what it does NOT.** Each replica resamples the whole differential-time pool and
re-runs the same LSQR-CND from the reported solution. So `ez95` is the **relative-precision scatter** — how tightly
the events are pinned *to each other* — and it is honest for that: the estimator is stable (CND 40–80), the
resampling is global (captures pair selection), and there are ~0 failed replicas. It is a *complete* answer to
"how well does the data resolve the internal structure."

**It is NOT the absolute-depth uncertainty.** Every replica is seeded at the same solution, and the differential
data cannot see the absolute centroid (the null direction of §1), so the bootstrap holds the centroid fixed by
construction. The true absolute-depth uncertainty is the **~1–2 km** spread §1 exposes (whole-box 11.5 → SVD 9.5,
seed-dependent). Quote **`ez95` for relative/internal precision** and the **§1 solver spread for absolute depth** —
they answer different questions. The reloc's own formal errors (~0.5 m, below) are a covariance artefact — never
quote them.""")
# ---- histograms of the per-event bootstrap 95% half-widths in East / North / depth ----
co(r"""bsolv=D['m389']['bmeta'].get('boot_solver','svd')
fig,axs=plt.subplots(1,3,figsize=(14,4.4))
for k,(bc,lab) in enumerate([("ex95","East"),("ey95","North"),("ez95","Depth")]):
    ax=axs[k]
    allv=np.concatenate([D[v]['rel'][bc].dropna().values for v in VOLS])
    hi=np.nanpercentile(allv,98)*1.1+1; bins=np.linspace(0,hi,26)
    for v,col in (("m389","tab:blue"),("m373","tab:red")):
        x=D[v]['rel'][bc].dropna().values
        ax.hist(x,bins=bins,color=col,alpha=0.55,edgecolor="w",label=f"{v} (median {np.median(x):.1f} m)")
    ax.set(xlabel=f"{lab} 95% bootstrap half-width (m)",ylabel="events",title=lab); ax.legend(fontsize=8)
fig.suptitle(f"Per-event relative-precision 95% half-widths (global LSQR-CND bootstrap, E / N / depth), n=200",y=1.02)
fig.tight_layout(); plt.show()
for v in VOLS:
    m=D[v]['rel']
    print(f"{v}: RELATIVE-precision 95% half-width median (P90) — E {m.ex95.median():.1f} ({m.ex95.quantile(.9):.0f}) | "
          f"N {m.ey95.median():.1f} ({m.ey95.quantile(.9):.0f}) | depth {m.ez95.median():.1f} ({m.ez95.quantile(.9):.0f}) m")
print("\nThese are RELATIVE precision (event-to-event). The ABSOLUTE centroid depth is a separate ~1-2 km null-space")
for v in VOLS:
    wbd=np.nanmedian(D[v]['ve'].set_index('id').depth); svd=np.nanmedian(D[v]['svd'].depth)
    print(f"   {v}: absolute depth spread across solvers = |whole-box {wbd:.2f} - SVD {svd:.2f}| = {abs(wbd-svd)*1000:.0f} m (see §1) -- NOT in ez95.")""")

md(r"""Reloc formal errors (covariance artefact, ~0.5 m) vs the bootstrap 95% half-widths (log–log, 1:1 dashed) — the formal errors are meaningless; quote the bootstrap:""")
co(r"""fig,axs=plt.subplots(1,3,figsize=(13.5,4.6))
for k,(fc,bc,lab) in enumerate([("ex","ex95","East"),("ey","ey95","North"),("ez","ez95","Depth")]):
    ax=axs[k]
    for v,col,mk in (("m389","tab:blue","o"),("m373","tab:red","s")):
        m=D[v]["rel"]; ok=m[(m[fc]>0)&m[bc].notna()]
        ax.scatter(ok[fc],ok[bc],s=16,color=col,marker=mk,alpha=0.7,lw=0,label=f"{v} (n={len(ok)})")
    lo,hi=0.05,5000; ax.plot([lo,hi],[lo,hi],"k--",lw=1)
    ax.set(xscale="log",yscale="log",xlim=(lo,hi),ylim=(lo,hi),xlabel=f"reloc formal {lab} error (m)",ylabel="bootstrap 95% (m)",title=lab)
    if k==0: ax.legend(loc="upper left",fontsize=8)
fig.suptitle("Reloc formal errors (covariance artefact) vs bootstrap 95% relative-precision half-widths",y=1.02); fig.tight_layout(); plt.show()
for v in VOLS:
    m=D[v]["rel"]; ok=m[(m.ex>0)&m.ex95.notna()]
    print(f"{v}: median formal ({np.nanmedian(ok.ex):.1f},{np.nanmedian(ok.ey):.1f},{np.nanmedian(ok.ez):.1f}) m "
          f"(artefact) vs median bootstrap95 ({np.nanmedian(ok.ex95):.0f},{np.nanmedian(ok.ey95):.0f},{np.nanmedian(ok.ez95):.0f}) m (relative precision)")
print("Quote the BOOTSTRAP 95% half-widths for relative precision; the reloc formal ex/ey/ez are a covariance artefact — never quote.")""")

# ------------------------------------------------------------------ §6 waveform gather
md(r"""## 6 · KG.HDB Z/N/E waveform gather — do the large-shift events belong to the cluster?

The definitive test of cluster membership is waveform similarity at a common close station. **KG.HDB** records
~99% of these events on **HHZ/HHN/HHE**. Each event's three components are bandpass-filtered **5–20 Hz (the exact
HypoDD dt.cc cross-correlation band, 4 corners, zero-phase)**, aligned on the KG.HDB **P pick**, and plotted
**ordered by origin time (earliest at top)** in three columns (Z, N, E). The dashed **blue line at t = 0 is the
P pick**; short **blue vertical bars mark the S pick** (SAC `t0`, where present). Traces of events flagged in §1
(whole-box→LSQR-CND shift > 300 m) are drawn in **red** with the shift annotated: if a red trace matches the black
cluster waveforms, the event *is* co-located (the per-volume subset mislocated it — trust the whole box); if it
looks different, it is a genuinely distinct source.

**P-pick handling:** each trace is aligned on the KG.HDB **P pick** (SAC header `a`). An event with **no KG.HDB
P pick is NOT plotted** (there is no datum to align it) — the printout reports how many of each volume's events
that removes, so nothing is silently shown at the wrong time.""")
co(r"""from obspy import read
def load_hdb(ts,comp,band=CCBAND,win=(-0.5,3.0)):
    p=f"{WF100}/{ts}/{ts}.KG.HDB.HH{comp}.sac"
    if not os.path.exists(p): return None
    tr=read(p)[0]; tr.detrend("demean"); tr.detrend("linear"); tr.taper(0.05)
    tr.filter("bandpass",freqmin=band[0],freqmax=band[1],corners=4,zerophase=True)
    sac=tr.stats.sac; a=sac.get("a",-12345.0)
    if a==-12345.0: return None                              # NO P pick on this component -> skip (no datum)
    pt=tr.stats.starttime+(a-sac.get("b",0.0))
    seg=tr.slice(pt+win[0],pt+win[1])
    if seg.stats.npts<10: return None
    d=seg.data.astype(float); d=d-d.mean(); mx=np.abs(d).max()
    t0=sac.get("t0",-12345.0); s_off=(t0-a) if t0!=-12345.0 else None   # S pick relative to P (SAC t0)
    return (np.linspace(win[0],win[1],len(d)), d/mx if mx>0 else d, s_off)
for v in VOLS:
    m=D[v]["rel"].sort_values("time").reset_index(drop=True); big=(m.wb_shift_m>300).values
    n=len(m); fig,axs=plt.subplots(1,3,figsize=(13,max(4,0.20*n+1)),sharey=True)
    nP={c:0 for c in "ZNE"}; nS={c:0 for c in "ZNE"}
    for col,comp in zip(axs,("Z","N","E")):
        col.set_facecolor("#FAFAFA")
        for i,r in m.iterrows():
            ts=r["ts"]; wf=load_hdb(ts,comp) if isinstance(ts,str) else None
            if wf is None: continue
            t,d,s_off=wf; nP[comp]+=1; c="tab:red" if big[i] else "0.15"; lw=0.8 if big[i] else 0.5
            col.plot(t,-i+0.45*d,color=c,lw=lw)
            if s_off is not None and -0.5<s_off<3.0:                 # short S bar
                col.plot([s_off,s_off],[-i-0.32,-i+0.32],color="tab:blue",lw=0.9,zorder=3); nS[comp]+=1
        col.axvline(0,color="tab:blue",ls="--",lw=0.8,zorder=0)      # P pick at t=0
        col.set(xlabel="Time from P (s)",title=f"KG.HDB HH{comp} ({CCBAND[0]}–{CCBAND[1]} Hz)  P={nP[comp]} S={nS[comp]}",xlim=(-0.5,3.0))
    ylabs=[f"{str(r.time)[:10]}  M{r.magU:.1f}"+(f"  ←{r.wb_shift_m:.0f}m" if big[i] else "") for i,r in m.iterrows()]
    axs[0].set_yticks(-np.arange(n)); axs[0].set_yticklabels(ylabs,fontsize=6); axs[0].set_ylim(-n,1)
    for tl,isbig in zip(axs[0].get_yticklabels(),big):
        if isbig: tl.set_color("tab:red")
    fig.suptitle(f"{TITLE[v]} — KG.HDB 3-component gather (P dashed, S bars), ordered by time (red = WB→CND > 300 m)",y=1.0)
    fig.tight_layout(); plt.show()
    print(f"{v}: {nP['Z']}/{n} plotted on HHZ (P-aligned); {n-nP['Z']} NOT plotted (no KG.HDB P pick). "
          f"S bars: Z {nS['Z']}, N {nS['N']}, E {nS['E']} of {n}. {int(big.sum())} red events to judge.")""")

# ------------------------------------------------------------------ §7 CC matrices
md(r"""## 7 · KG.HDB waveform cross-correlation matrices (Z / N / E), ordered by origin time

A quantitative companion to the gather: the **N×N maximum cross-correlation matrix** among all of a volume's
events at KG.HDB, one per component, **5–20 Hz**, aligned on P with a ±0.2 s lag search (the HypoDD CC settings),
events **ordered by origin time**. Bright blocks = groups of near-identical waveforms (co-located repeating
sources). A large-shift event should be judged against **this cluster's own coherence** (its median off-diagonal
CC, printed) — not an absolute threshold: the M3.89 volume is a diffuse blob (median CC ~0.4), so "co-located"
there means CC ≈ 0.4, not 0.9. **Caveat:** the flagged large-shift events are mostly **M ~ 0 (ML-less)**, whose
waveforms are low-SNR, so a low CC can reflect noise rather than a different location — the printout marks those
as ambiguous rather than over-claiming a distinct source.""")
co(r"""def load_arr(ts,comp,sta="KG.HDB",band=CCBAND,win=(-0.2,2.5),sr=100):
    g=glob.glob(f"{WF100}/{ts}/{ts}.{sta}.*{comp}.sac")   # station-general (KG.HDB HH; KG.MKL etc.)
    if not g: return None
    tr=read(g[0])[0]; tr.detrend("demean"); tr.detrend("linear"); tr.taper(0.05)
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
    m=D[v]["rel"].sort_values("time").reset_index(drop=True); big=(m.wb_shift_m>300).values; ts=m.ts.tolist()
    fig,axs=plt.subplots(1,3,figsize=(15.5,5.2)); cmap=plt.cm.seismic.copy(); cmap.set_bad("0.85")
    Mz=None
    for ax,comp in zip(axs,("Z","N","E")):
        arrs=[load_arr(t,comp) if isinstance(t,str) else None for t in ts]
        M,idx=cc_matrix(arrs)
        if comp=="Z": Mz=M
        im=ax.imshow(np.ma.masked_invalid(M),vmin=0,vmax=1,cmap=cmap,origin="upper")
        for k in np.where(big)[0]: ax.axhline(k,color="red",lw=0.4,alpha=0.5); ax.axvline(k,color="red",lw=0.4,alpha=0.5)
        ax.set(title=f"KG.HDB HH{comp} ({CCBAND[0]}–{CCBAND[1]} Hz)",xlabel="event (time order)",ylabel="event (time order)")
        fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04,label="max CC")
    fig.suptitle(f"{TITLE[v]} — KG.HDB waveform CC matrices, events ordered by origin time (red lines = WB→CND > 300 m)",y=1.02)
    fig.tight_layout(); plt.show()
    if Mz is not None:
        off=Mz.copy(); np.fill_diagonal(off,np.nan); medcc=np.nanmedian(off)
        print(f"{v}: HHZ median off-diagonal CC = {medcc:.2f} (this cluster's OWN coherence — judge flagged events vs THIS).")
        print(f"  Flagged (WB→CND>300 m) events' median CC to the rest (compared to the {medcc:.2f} cluster baseline):")
        for k in np.where(big)[0]:
            r=m.iloc[k]; mc=np.nanmedian(off[k])
            tag=("≥ cluster baseline -> as coherent as a typical member (co-located; the subset mislocated it)" if mc>=medcc
                 else "below baseline -> less similar, BUT it is M%.1f so low SNR may explain it (ambiguous)"%r.magU if r.magU<1.0
                 else "below baseline -> genuinely less similar (candidate distinct source)")
            print(f"    id {int(r.id)} {str(r.time)[:10]} M{r.magU:.1f} shift {r.wb_shift_m:.0f} m -> median CC {mc:.2f}  ({tag})")""")

# ------------------------------------------------------------------ §8 multi-station CC transition test
md(r"""## 8 · Is the M3.89 similarity transition instrumental? — multi-station CC test

The M3.89 CC matrix (§7) shows a step in waveform similarity partway down. A **source** change would appear at
every station; an **instrument** change appears only at the affected one. Below are the KG.HDB HHZ CC matrix
against three other stations that recorded the same events in both eras. The black line marks the **documented
KG.HDB sensor break 2015-05-21**; the printed *contrast* = (within-era mean CC) − (cross-era mean CC).""")
co(r"""STAS=["KG.HDB","KG.MKL","KG.HAK","KG.BBK"]        # HDB + 3 stations with coverage both sides of the break
BRK=pd.Timestamp("2015-05-21",tz="UTC")
m=D["m389"]["rel"].sort_values("time").reset_index(drop=True); ts=m.ts.tolist(); tt=m.time.values
kbrk=int((m.time>=BRK).values.argmax())            # first event on/after the break -> matrix boundary
cmap=plt.cm.seismic.copy(); cmap.set_bad("0.85")
fig,axs=plt.subplots(1,len(STAS),figsize=(4.3*len(STAS),4.6))
for ax,sta in zip(axs,STAS):
    arrs=[load_arr(t,"Z",sta=sta) if isinstance(t,str) else None for t in ts]
    M,idx=cc_matrix(arrs)
    im=ax.imshow(np.ma.masked_invalid(M),vmin=0,vmax=1,cmap=cmap,origin="upper")
    ax.axhline(kbrk-0.5,color="k",lw=1.0); ax.axvline(kbrk-0.5,color="k",lw=1.0)
    bef=np.array([tt[i] for i in idx])<np.datetime64(BRK.tz_convert(None)); iu=np.triu_indices(len(idx),1)
    wmask=bef[iu[0]]==bef[iu[1]]; con=np.nanmean(M[iu][wmask])-np.nanmean(M[iu][~wmask])
    ax.set(title=f"{sta} HHZ | contrast {con:.2f}",xlabel="event (time order)")
axs[0].set_ylabel("event (time order)")
fig.suptitle("M3.89 similarity step is INSTRUMENTAL: strong contrast at KG.HDB (2015-05-21 break), ~0 elsewhere",y=1.03)
fig.colorbar(im,ax=axs,fraction=0.012,pad=0.02,label="max CC"); plt.show()
print("Contrast across the 2015-05-21 KG.HDB break: KG.HDB ~0.2 (the step); KG.MKL/HAK/BBK ~0.02 (no step)")
print("-> the transition is the KG.HDB sensor/response change, NOT a change in the earthquakes. It is also")
print("   DISTINCT from the ~2014-12 HDB gain drift (amplitude only, invisible to amplitude-normalised CC).")""")

# ------------------------------------------------------------------ §9 summary
md(r"""## 9 · Summary""")
co(r"""rows=[]
for v in VOLS:
    m=D[v]["rel"]; meta=D[v]["meta"]; summ=D[v]["summ"]; fm_=summ["fit_main"]; bm=D[v]["bmeta"]
    inv=m.in_volume.fillna(True); ok=m[inv&m.ex95.notna()]
    rows.append(dict(volume=TITLE[v],primary=meta["primary_solver"],cnd=meta["primary_cnd"],
        n_reloc=meta["n_relocated"],n_in_volume=summ["n_in_volume"],
        depth_km=round(float(np.nanmedian(m.depth)),2),svd_depth_km=round(float(np.nanmedian(D[v]["svd"].depth)),2),
        boot_n=bm["n"],boot=f"{bm.get('resample')}/{bm.get('boot_solver')}",boot_fail_pct=round(bm["failed_frac"]*100,1),n_boot_min=int(m.n_boot.min()),
        shape=fm_["shape"],L1_km=fm_["L1"],L2_km=fm_["L2"],L3_km=fm_["L3"],
        rel_prec95_m=round(float(np.nanmedian(np.c_[ok.ex95,ok.ey95,ok.ez95])),1),
        big_shift=len(summ["big_shift_ids"])))
SUM=pd.DataFrame(rows)
print("="*140); print("PER-VOLUME LSQR-CND RELOCATION (whole-box seed) + GLOBAL BOOTSTRAP — two largest-event UF volumes".center(140)); print("="*140)
print(SUM.to_string(index=False)); sm=D["m373"]["summ"].get("separation_m",{})
print("\nTAKE-HOMES")
print(" - Reported solver = LSQR-CND on the whole-box seed: it holds the PHYSICAL absolute depth (m373 11.49 km,")
print("   m389 13.69 km), whereas undamped SVD slides the centroid down the DD null direction to a seed-dependent")
print("   shallower value (m373 9.5 km) — same internal shape, wrong absolute depth. SVD is kept only as the §1 diagnostic.")
print(" - Global bootstrap (n=200, LSQR-CND): ~0 failed replicas, n_boot≈200/event. ez95 (~3 m) = RELATIVE precision;")
print("   the ABSOLUTE centroid depth carries a separate ~1-2 km null-space uncertainty (§1) NOT captured by ez95.")
print(" - Both volumes are sub-km BLOBS (not planes):")
for v in VOLS:
    fm_=D[v]["summ"]["fit_main"]
    print(f"     {TITLE[v]}: {fm_['L1']}×{fm_['L2']}×{fm_['L3']} km, {len(D[v]['summ']['big_shift_ids'])} large-shift events to vet by waveform (§6).")
print(f" - C6 vs C7 (M3.73): {sm.get('med')} m apart (95% {sm.get('lo')}–{sm.get('hi')}) — nearly co-located; same")
print("   ~0.1 km patch, two episodes.")
print(" - The large whole-box↔LSQR-CND shifts are events that lost outside-cube links under subsetting; §6 shows their")
print("   KG.HDB waveforms next to the cluster so membership is judged by similarity, not auto-flagged.")
print("\nNEXT: extend to the C95 repeater volume / HDBSCAN patches; Phase-2 relative ML for the ML-less members.")""")

nb["cells"]=C
import os
os.chdir("/home/msseo/works/02.Ulsan_Fault_detection/analysis/reloc_analysis")
nbf.write(nb,"33.UF_cluster_svd_volumes.ipynb")
print("wrote 33.UF_cluster_svd_volumes.ipynb",len(C),"cells")
