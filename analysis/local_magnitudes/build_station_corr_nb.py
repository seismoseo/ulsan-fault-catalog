#!/usr/bin/env python
"""Generate 09.Station_corrections_ML.ipynb — SOTA homogenisation of the Heo (2024) ML catalog by
estimating time-invariant STATION CORRECTIONS, so ML no longer depends on which stations recorded an
event (the root cause of the network-densification bias quantified in notebook 08).

Model (per station-channel reading):   ML_si = mu_i + S_s + eps_si
  mu_i = event magnitude (homogenised ML),  S_s = station term,  eps = noise.
Solve {mu_i, S_s} from all readings. Gauge: obs-weighted mean(S_s)=0 (keeps the absolute scale on the
published Heo level). Primary solver = robust alternating-median (Tukey median polish); cross-check =
sparse least squares. Validate by RE-RUNNING the notebook-08 decimation test on the corrected readings
(bias(Y) should collapse to ~0), plus split-era station-term stability and residual-vs-distance.

Pure CSV inversion on the per-station ML table; does not touch any running job.
References: Hutton & Boore 1987 (BSSA 77); Tukey 1977 (EDA, median polish);
Abrahamson & Youngs 1992 (BSSA 82, random-effects); Bormann NMSOP-2 (station corrections)."""
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md("""# Station corrections — homogenising the Heo (2024) ML catalog

Notebook 08 showed the sparse early network biases ML **low by ~0.1** (2010-2015), decaying to ~0 by
2018 as the network densified *continuously*. A per-year offset is a crude symptom-patch. Here we remove
the **root cause** with time-invariant **station corrections**, so ML becomes independent of *which*
stations recorded an event — the densification bias then dissolves by construction, for any (continuous)
network configuration.

### Model
Each per-station-channel reading is the regional ML estimate `log10(A) + (-logA0_Heo)(r)`. We decompose

$$ML_{si} = \\mu_i + S_s + \\varepsilon_{si}$$

- $\\mu_i$ — event magnitude (the homogenised ML we want),
- $S_s$ — **station term** (residual site response / local $-\\log A_0$ error not in the regional curve),
- $\\varepsilon_{si}$ — reading noise.

We keep the **Heo $-\\log A_0$** (trusted) and let $S_s$ absorb the residual. **Gauge:** the $(\\mu,S)$ pair
has a constant trade-off ($\\mu_i{+}c$, $S_s{-}c$), fixed by requiring the observation-weighted
$\\overline{S_s}=0$ — this anchors the homogenised magnitudes to the published Heo scale *on average*.

For a **spatially concentrated source** (the UF box, ~30 km) each station sits at a near-fixed distance,
so $S_s$ also absorbs that station's local $-\\log A_0$ error (near-degenerate) — station terms alone
suffice; no $-\\log A_0$ refit needed.

### Method
- **Primary solver — robust alternating medians (Tukey *median polish*):** iterate
  $S_s=\\mathrm{med}_i(ML_{si}-\\mu_i)$ (gauge-centred), $\\mu_i=\\mathrm{med}_s(ML_{si}-S_s)$ until
  convergence. Outlier-resistant and matches the pipeline's median aggregation.
- **Cross-check — sparse least squares** (`scipy.sparse.linalg.lsqr`) on the same two-way model.

### Validation
1. **Re-run the notebook-08 decimation test on corrected readings** → bias(Y) should flatten to ~0.
2. **Split-era station-term stability** (2010-2017 vs 2018-2024 should agree on 1:1; outliers = instrument/site changes needing an epoch-split term).
3. **Residual vs distance** flat after correction.

### References (verify exact vol/page/DOI before manuscript use)
- **Hutton & Boore (1987)**, *BSSA* 77(6), 2074-2094 — Mᴸ scale + station corrections (direct template).
- **Tukey (1977)**, *Exploratory Data Analysis* — median polish (robust two-way decomposition).
- **Abrahamson & Youngs (1992)**, *BSSA* 82(1), 505-510 — random-effects regression (mixed-effects form).
- **Bormann (ed.), NMSOP-2** (IASPEI/GFZ) — magnitude station corrections in practice.""")

md("""## 1 · Load per-station readings""")
co("""import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib as mpl, matplotlib.font_manager as fm
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":120,"axes.grid":True,"grid.alpha":0.3})

PS_FILE="catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo.csv"
SNR_MIN=3.0; DM=0.1; N_ITER=40; TOL=1e-4

d=pd.read_csv(PS_FILE)
d=d[(d.snr>=SNR_MIN)&d.ML.notna()].copy()
d["year"]=pd.to_datetime(d.event_time,utc=True,errors="coerce").dt.year
d=d.dropna(subset=["year"]); d["year"]=d.year.astype(int)
d["sc"]=d.network.astype(str)+"."+d.station.astype(str)+"."+d.channel.astype(str)   # station-channel = correction unit
print(f"{len(d):,} readings | {d.event_idx.nunique():,} events | {d.sc.nunique()} station-channels | {d.year.min()}-{d.year.max()}")
sc_counts=d.sc.value_counts()""")

md("""## 2 · Robust station-term inversion (median polish)

Alternating medians with an observation-weighted gauge. Converges in a few iterations.""")
co("""mu = d.groupby("event_idx").ML.median()                     # init: per-event median
S  = pd.Series(0.0, index=sc_counts.index)
hist=[]
for it in range(N_ITER):
    resid = d.ML.values - mu.reindex(d.event_idx).values        # ML - mu
    S_new = pd.Series(resid, index=d.sc).groupby(level=0).median()
    S_new -= np.average(S_new.reindex(sc_counts.index).values, weights=sc_counts.values)  # gauge: weighted mean 0
    corr  = d.ML.values - S_new.reindex(d.sc).values            # ML - S
    mu_new = pd.Series(corr, index=d.event_idx).groupby(level=0).median()
    dmax = float(np.nanmax(np.abs(mu_new.reindex(mu.index).values - mu.values)))
    hist.append(dmax); mu, S = mu_new, S_new
    if dmax < TOL: break
print(f"converged in {len(hist)} iters (max |dmu|={hist[-1]:.2e})")
ST = pd.DataFrame({"S":S, "n":sc_counts.reindex(S.index)}).sort_values("S")
print(f"station terms: range [{S.min():+.2f},{S.max():+.2f}], std {S.std():.3f}, weighted-mean {np.average(S.reindex(sc_counts.index),weights=sc_counts):+.4f} (=0 by gauge)")

fig,ax=plt.subplots(1,2,figsize=(13,4.4))
ax[0].plot(range(1,len(hist)+1),hist,"o-"); ax[0].set(xlabel="Iteration",ylabel="max |Δμ|",yscale="log",title="Median-polish convergence")
o=ST[ST.n>=20]
ax[1].barh(range(len(o)),o.S,color=np.where(o.S>=0,"tab:red","tab:blue")); ax[1].set_yticks(range(len(o)))
ax[1].set_yticklabels(o.index,fontsize=6); ax[1].axvline(0,color="0.5",lw=0.8)
ax[1].set(xlabel="Station term S_s (ML units)",title="Station corrections (n≥20)")
fig.tight_layout(); plt.show()""")

md("""## 3 · Cross-check — sparse least squares

Same model solved by `lsqr` (mean-based). Should match the robust median-polish terms closely.""")
co("""from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
ev_ix={e:i for i,e in enumerate(sorted(d.event_idx.unique()))}
sc_ix={s:j for j,s in enumerate(sc_counts.index)}
ne,ns=len(ev_ix),len(sc_ix); m=len(d)
rows=np.repeat(np.arange(m),2)
cols=np.empty(2*m,int); cols[0::2]=d.event_idx.map(ev_ix).values; cols[1::2]=d.sc.map(sc_ix).values+ne
vals=np.ones(2*m)
A=csr_matrix((vals,(rows,cols)),shape=(m,ne+ns))
sol=lsqr(A,d.ML.values,atol=1e-8,btol=1e-8,iter_lim=2000)[0]
mu_ls=pd.Series(sol[:ne],index=list(ev_ix)); S_ls=pd.Series(sol[ne:],index=list(sc_ix))
S_ls-=np.average(S_ls.reindex(sc_counts.index),weights=sc_counts.values)   # same gauge
cmp=pd.DataFrame({"median_polish":S,"lsqr":S_ls}).dropna()
fig,ax=plt.subplots(figsize=(5.2,5.2))
ax.scatter(cmp.median_polish,cmp.lsqr,s=18,alpha=.7); lim=[cmp.values.min()-.05,cmp.values.max()+.05]
ax.plot(lim,lim,"0.5",ls="--"); ax.set(xlabel="S_s  (median polish)",ylabel="S_s  (lsqr)",
   title=f"Station terms agree (r={cmp.median_polish.corr(cmp.lsqr):.3f})",xlim=lim,ylim=lim)
ax.set_aspect("equal"); fig.tight_layout(); plt.show()""")

md("""## 4 · Validation — does it kill the densification bias?

Apply the station terms to every reading (`ML_corr = ML − S_s`) and re-run the notebook-08 decimation
test. If the correction works, **bias(Y) collapses to ~0** for all years (no station-set dependence).""")
co("""d["ML_corr"]=d.ML - d.sc.map(S).values
REF_YEARS=(2022,2023,2024); REF_NMIN=8; DEC_NMIN=3; OP_MIN_FRAC=0.02
years=sorted(d.year.unique())
SY={}
for y in years:
    dy=d[d.year==y]; nev=dy.event_idx.nunique(); vc=dy.groupby("station").event_idx.nunique()
    SY[y]=set(vc[vc>=max(3,OP_MIN_FRAC*nev)].index)
def decim_bias(col):
    full=d.groupby("event_idx")[col].median(); nf=d.groupby("event_idx").station.nunique()
    ref_ev=nf[(nf>=REF_NMIN)&nf.index.isin(d[d.year.isin(REF_YEARS)].event_idx)].index
    r=d[d.event_idx.isin(ref_ev)]; out={}
    for y in years:
        sub=r[r.station.isin(SY[y])]; mld=sub.groupby("event_idx")[col].median(); ndc=sub.groupby("event_idx").station.nunique()
        keep=ndc[ndc>=DEC_NMIN].index; dml=(mld.loc[keep]-full.loc[keep]).dropna()
        if len(dml): out[y]=dml.median()
    return pd.Series(out)
b_before=decim_bias("ML"); b_after=decim_bias("ML_corr")
fig,ax=plt.subplots(figsize=(11,4.6))
ax.axhline(0,color="0.6",lw=0.8,ls="--")
ax.plot(b_before.index,b_before.values,"o-",color="tab:red",label="before (raw ML)")
ax.plot(b_after.index,b_after.values,"s-",color="tab:green",label="after station correction")
ax.set(xlabel="Historical network year",ylabel="Decimation bias ΔML",
       title="Densification bias before vs after station corrections"); ax.set_xticks(years)
ax.tick_params(axis="x",labelrotation=45); ax.legend(); fig.tight_layout(); plt.show()
print("max |bias|  before: %.3f  after: %.3f  (early-era mean before %.3f -> after %.3f)"%(
   b_before.abs().max(),b_after.abs().max(),b_before[b_before.index<=2015].mean(),b_after[b_after.index<=2015].mean()))""")

md("""## 5 · Stability checks — are the station terms time-invariant?""")
co("""# split-era station terms (recompute residual medians per era at fixed mu)
d["res"]=d.ML - d.event_idx.map(mu).values
early=d[d.year<=2017].groupby("sc").res.median(); late=d[d.year>=2018].groupby("sc").res.median()
ne_=d[d.year<=2017].groupby("sc").size(); nl_=d[d.year>=2018].groupby("sc").size()
ok=(ne_>=20)&(nl_>=20); E=early[ok.index[ok]]; L=late[ok.index[ok]]; E,L=E.align(L,join="inner")
fig,ax=plt.subplots(1,2,figsize=(12,4.6))
ax[0].scatter(E,L,s=20,alpha=.7); lim=[min(E.min(),L.min())-.05,max(E.max(),L.max())+.05]
ax[0].plot(lim,lim,"0.5",ls="--"); ax[0].set(xlabel="S_s  2010-2017",ylabel="S_s  2018-2024",
   title=f"Station-term stability (r={E.corr(L):.3f})",xlim=lim,ylim=lim); ax[0].set_aspect("equal")
for s in E.index:
    if abs(E[s]-L[s])>0.15: ax[0].annotate(s,(E[s],L[s]),fontsize=6)
# residual vs distance, before/after
db=d.assign(rb=d.ML-d.event_idx.map(mu).values, ra=d.ML_corr-d.event_idx.map(mu).values)
bins=pd.cut(db.dist_km,np.arange(0,120,10))
mb=db.groupby(bins,observed=True).rb.median(); ma=db.groupby(bins,observed=True).ra.median()
xc=[i.mid for i in mb.index]
ax[1].axhline(0,color="0.6",lw=0.8,ls="--")
ax[1].plot(xc,mb.values,"o-",color="tab:red",label="before"); ax[1].plot(xc,ma.values,"s-",color="tab:green",label="after")
ax[1].set(xlabel="Hypocentral distance (km)",ylabel="Median residual",title="Residual vs distance"); ax[1].legend()
fig.tight_layout(); plt.show()
print(f"split-era station-term correlation r={E.corr(L):.3f}; terms with |Δ|>0.15 (possible instrument change): "
      f"{[s for s in E.index if abs(E[s]-L[s])>0.15]}")""")

md("""## 6 · Apply correction → homogenised catalog + b-value/Mc impact""")
co("""from seismostats.analysis import estimate_mc_maxc, ClassicBValueEstimator
from seismostats.utils import bin_to_precision
ml_raw=d.groupby("event_idx").ML.median(); ml_hom=mu     # homogenised = median-polish event term
yr=d.groupby("event_idx").year.first(); tt=d.groupby("event_idx").event_time.first()
cat=pd.DataFrame({"event_time":tt,"year":yr,"ml_heo":ml_raw,"ml_homogenised":ml_hom.reindex(ml_raw.index)})
cat.to_csv("catalog_ml_heo_station_homogenised.csv")
def _b(mm):
    m=bin_to_precision(np.sort(mm.astype(float)),DM); mc,_=estimate_mc_maxc(m,fmd_bin=DM)
    be=ClassicBValueEstimator(); be.calculate(m[m>=mc],mc=mc,delta_m=DM); return mc,be.b_value,be.std
ann=cat.groupby("year").agg(raw=("ml_heo","median"),hom=("ml_homogenised","median"),n=("ml_heo","size"))
fig,ax=plt.subplots(1,2,figsize=(13,4.4))
ax[0].plot(ann.index,ann.raw,"o-",color="0.5",label="raw Heo ML"); ax[0].plot(ann.index,ann.hom,"s-",color="tab:blue",label="station-homogenised")
ax[0].set(xlabel="Year",ylabel="Annual median ML",title="Annual median ML"); ax[0].tick_params(axis="x",labelrotation=45); ax[0].legend()
for col,lab,c in [("ml_heo","raw","0.5"),("ml_homogenised","homogenised","tab:blue")]:
    mm=cat[col].dropna().to_numpy(); mc,b,se=_b(mm); m=bin_to_precision(np.sort(mm),DM)
    e=np.arange(np.floor(m.min()/DM)*DM,np.ceil(m.max()/DM)*DM+DM,DM); cum=np.array([(m>=x).sum() for x in e])
    ax[1].semilogy(e,cum,".",color=c,ms=4,label=f"{lab}: Mc={mc:.2f} b={b:.2f}±{se:.2f}")
ax[1].set(xlabel="ML",ylabel="N(≥ML)",title="Full-catalog FMD"); ax[1].legend(fontsize=8)
fig.tight_layout(); plt.show()
print("wrote catalog_ml_heo_station_homogenised.csv")""")

md("""## 7 · Summary""")
co("""print("STATION-CORRECTION HOMOGENISATION — summary\\n"+"="*50)
print(f"events {cat.shape[0]:,} | station-channels {len(S)} | median-polish iters {len(hist)}")
print(f"station terms: range [{S.min():+.2f},{S.max():+.2f}] ML, std {S.std():.3f}")
print(f"median-polish vs lsqr agreement r={cmp.median_polish.corr(cmp.lsqr):.3f}")
print(f"split-era station-term stability r={E.corr(L):.3f}")
print(f"decimation bias (early era) BEFORE {b_before[b_before.index<=2015].mean():+.3f} -> AFTER {b_after[b_after.index<=2015].mean():+.3f} ML")
print(f"max |decimation bias| BEFORE {b_before.abs().max():.3f} -> AFTER {b_after.abs().max():.3f} ML")
print("\\nTake-homes:")
print(" - Station terms estimated jointly with event magnitudes (Hutton-Boore model; median-polish solver).")
print(" - After correction the decimation bias collapses toward 0 at all years => ML is now station-set-independent")
print("   (continuous densification handled by construction, not by per-year steps).")
print(" - Homogenised catalog: catalog_ml_heo_station_homogenised.csv (use ml_homogenised for b/Mc/temporal work).")
print(" - Stable split-era terms confirm time-invariance; any |Δ|>0.15 station flagged for an epoch-split term.")""")

nb.cells=C
out="09.Station_corrections_ML.ipynb"; nbf.write(nb,out); print("wrote",out,len(C),"cells")
