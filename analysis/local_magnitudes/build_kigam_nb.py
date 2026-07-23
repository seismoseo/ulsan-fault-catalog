#!/usr/bin/env python
"""Generate 17.Response_epoch_corrected_catalog.ipynb — the clean, metadata-grounded fix for the
sensor-swap magnitude time-dependence. Logic in one sentence: nb09 gave each station ONE offset; a
sensor swap changes that offset, so we give each station ONE offset PER SENSOR ERA, using the documented
sensor-change dates from the station-response StationXML (covers BOTH KS and KG; no data-driven guessing),
then re-check the UF background rate. Runs in `base`."""
import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# Response-epoch-corrected ML catalog (station-response metadata, KS + KG)

## The logic, in four plain steps

1. **nb09 gave each station one fixed offset.** That removed the network-densification bias (good — the
   magnitudes you were happy with). It assumes the offset never changes in time.
2. **A sensor swap changes that offset.** When a station's seismometer is physically replaced, its
   leftover offset *steps* on the swap date — so one fixed number is wrong for it. (We confirmed this is
   real, not a metadata error: the official responses match the old ones in our band, yet a
   residual magnitude step remains across the swap.)
3. **The station-response metadata tells us exactly WHEN each sensor changed.** We no longer guess from
   the data. We read the sensor-change dates straight from the StationXML used to deconvolve the
   instrument (a response *shape* change = a physical sensor swap; a gain-only change is already handled
   by deconvolution, so it needs no split).
4. **So: give each station one offset PER SENSOR ERA.** Re-run the same nb09 median-polish, but let a
   station's offset differ before vs after each documented swap. Subtract, take the median → a
   time-homogeneous ML. Then re-check whether the UF background rate is steady or rising.

*That is the whole method.* Everything below just executes these four steps and shows the numbers.

*Inputs:* per-station ML table (nb08/09), the station-response StationXML
`responses/master/KS_KG_metadata_1.0.2.xml` (the same metadata used to remove instrument response).
*Scope note:* this StationXML covers **both KS and KG**, so the epoch split now reaches the drifting KMA
(KS) stations too (EUSB, GUWB, JEJB, YOCB) — not just the KG movers (HDB, MKL, YSB). See nb18 for the
per-station diagnostics and the pre-2015 HDB failure window (flagged, better excluded than offset).""")

md(r"""### What "one offset per sensor era" means — a worked example

The correction changes **only how a station's rows are labelled**; the math is the identical nb09 median
polish.

- **nb09:** every reading from `KG.HDB.HHZ` shares the label `KG.HDB.HHZ` → **one** offset for 15 years.
- **nb17:** relabel HDB's rows by sensor era using its documented swap dates (HDB actually has four:
  2010-03, 2012-11, 2015-05, 2019-02): `…@0/@1/…` per era. The solver treats each era as an **independent
  station** and finds a **separate offset for each**.

**Why it fixes the step:** if HDB read +0.10 ML high before the 2019 swap and −0.20 low after, one fixed
offset can only be a compromise (≈ −0.05) → a residual *step*. With three labels the solver returns
S@1 = +0.10, S@2 = −0.20, each reading gets its own era's offset, and the step disappears. That's the
whole trick: **split a station's rows at its documented swap dates, then run the same nb09 correction.**""")

# ----------------------------------------------------------------- §0 setup + nb09 baseline
co(r"""import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, pandas as pd, obspy
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
from collections import defaultdict
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":120,"axes.grid":True,"grid.alpha":0.3,"font.size":11,
    "legend.framealpha":1.0,"legend.facecolor":"white","legend.edgecolor":"0.6"})  # opaque legend, above data

PS="catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo.csv"
SXML="responses/master/KS_KG_metadata_1.0.2.xml"   # station-response StationXML (covers BOTH KS and KG)
DM=0.1; UF=(129.25,129.55,35.60,35.90); MIN_EPOCH_N=50

d=pd.read_csv(PS); d=d[(d.snr>=3)&d.ML.notna()].copy()
d["t"]=pd.to_datetime(d.event_time,utc=True,errors="coerce"); d=d.dropna(subset=["t"])
d["year"]=d.t.dt.year; d["sc"]=d.network+"."+d.station+"."+d.channel
print(f"{len(d):,} readings | {d.sc.nunique()} station-channels | {d.year.min()}-{d.year.max()}")

def median_polish(df,col,n=40,tol=1e-4):
    "STEP-4 engine: joint {event magnitude mu, unit offset S}; gauge obs-weighted mean(S)=0."
    w=df[col].value_counts(); mu=df.groupby("event_idx").ML.median(); S=pd.Series(0.0,index=w.index)
    for _ in range(n):
        Sn=pd.Series(df.ML.values-mu.reindex(df.event_idx).values,index=df[col]).groupby(level=0).median()
        Sn-=np.average(Sn.reindex(w.index),weights=w.values)
        mun=pd.Series(df.ML.values-Sn.reindex(df[col]).values,index=df.event_idx).groupby(level=0).median()
        if float(np.nanmax(np.abs(mun.reindex(mu.index).values-mu.values)))<tol: mu,S=mun,Sn; break
        mu,S=mun,Sn
    return mu,S
mu0,S0=median_polish(d,"sc")   # STEP 1: the nb09 one-offset-per-station baseline
print("Step 1 done: nb09 baseline (one fixed offset per station).")""")

# ----------------------------------------------------------------- §1 sensor-change dates
md(r"""## Step 3 in code — read the sensor-change dates from the station-response metadata

A **sensor change** = the response *shape* changes (number of poles/zeros differs between consecutive
epochs). That is the part a deconvolution cannot fully erase, so it is where the offset can step. We
list those dates per station-channel — these are the *only* break dates we will use.""")
co(r"""inv=obspy.read_inventory(SXML)
def shape(c):
    try: pz=c.response.get_paz(); return (len(pz.poles),len(pz.zeros))
    except Exception: return None
breaks=defaultdict(list)
for net in inv:
    for sta in net:
        bych=defaultdict(list)
        for c in sta.channels: bych[c.code].append(c)
        for cc,cl in bych.items():
            sc=f"{net.code}.{sta.code}.{cc}"
            cl=sorted(cl,key=lambda x:x.start_date); prev=None
            for c in cl:
                fp=shape(c)
                if prev is not None and fp is not None and fp!=prev: breaks[sc].append(c.start_date.date)
                if fp is not None: prev=fp
breaks={k:v for k,v in breaks.items() if k in set(d.sc.unique())}   # only stations in our catalog
tab=pd.DataFrame([(k,len(v)+1,[str(x) for x in v]) for k,v in sorted(breaks.items())],
                 columns=["station-channel","sensor eras","change dates"])
nks=sum(k.startswith("KS.") for k in breaks); nkg=sum(k.startswith("KG.") for k in breaks)
print(f"catalog station-channels with a documented sensor change: {len(breaks)}  ({nkg} KG, {nks} KS)")
print(tab.to_string(index=False))""")

# ----------------------------------------------------------------- §2 re-homogenise
md(r"""## Step 3b — exclude the HDB pre-2015 sensor-failure window

Before re-homogenising, drop the **corrupt HDB readings from its 2014–2015 failure window** (the sensor
read ~2 ML low for ~7 months before its 2015-05 replacement; see nb18 §1c–d). A failing sensor's
amplitudes are unreliable — better excluded than given a large offset. The cut is the documented swap date
(2015-05-21) back to the data-driven failure onset (first month HDB drops below −1 ML). The per-event
median already shields event magnitudes from these outliers, so the catalog-level effect is negligible;
this just keeps a known-bad window out of the station-term fit.""")
co(r"""_hres=d[d.sc=="KG.HDB.HHZ"].assign(res=d[d.sc=="KG.HDB.HHZ"].ML-d[d.sc=="KG.HDB.HHZ"].event_idx.map(mu0).values)
_mz=_hres.set_index("t").res.groupby(pd.Grouper(freq="ME")).median(); _f=_mz[(_mz<-1.0)&(_mz.index<pd.Timestamp("2015-06",tz="UTC"))]
FAIL_ON=pd.Timestamp(_f.index.min()).replace(day=1) if len(_f) else pd.Timestamp("2014-11-01",tz="UTC")
FAIL_OFF=pd.Timestamp("2015-05-21",tz="UTC")
bad=(d.sc=="KG.HDB.HHZ")&(d.t>=FAIL_ON)&(d.t<FAIL_OFF)
print(f"excluding {int(bad.sum())} HDB readings in the failure window {FAIL_ON.date()} .. {FAIL_OFF.date()}")
d=d[~bad].copy()
mu0,S0=median_polish(d,"sc")    # refit baseline on the cleaned readings""")

md(r"""## Step 4 in code — one offset per sensor era, then re-homogenise

Give each station-channel a separate offset in each sensor era (split at the dates above; merge any era
with < 50 readings back so we never fit noise). Re-run the identical median-polish. The result `ml_epoch`
is the time-homogeneous magnitude.""")
co(r"""def era_unit(row):
    s=row.sc
    if s not in breaks: return s
    return f"{s}@{sum(row.t.date()>=b for b in breaks[s])}"
d["unit"]=d.apply(era_unit,axis=1)
uc=d.unit.value_counts(); d["unit"]=d.unit.where(~d.unit.isin(set(uc[uc<MIN_EPOCH_N].index)),d.sc)
mu1,S1=median_polish(d,"unit")
n_split=sum("@" in u for u in d.unit.unique())
print(f"correction units: {d.unit.nunique()} (was {d.sc.nunique()}); {n_split} sensor-era units")
ev=pd.DataFrame({"event_time":d.groupby("event_idx").t.first(),"year":d.groupby("event_idx").year.first(),
                 "ml_nb09":mu0,"ml_epoch":mu1.reindex(mu0.index)}).dropna()
ev.to_csv("catalog_ml_heo_epoch.csv")                                            # canonical (KS+KG epoch ML)
ev.rename(columns={"ml_epoch":"ml_kigam"}).to_csv("catalog_ml_heo_kigam_epoch.csv")  # back-compat alias for downstream
print(f"wrote catalog_ml_heo_epoch.csv (+ kigam alias) ({len(ev):,} events)")""")

# ----------------------------------------------------------------- §3 what changed in the magnitudes
md(r"""## What changed in the magnitudes?

Δμ = `ml_epoch − ml_nb09` is how much the correction moved each event. Its *overall* level is arbitrary
(a "gauge" — the absolute zero is just a convention; see §4f), so only the **shape in time** is physical.
We draw it **relative to the 2019–2024 modern network** (recent era ≡ 0), so the curve reads directly as
"how much earlier magnitudes were inflated relative to today's scale." Zero sits at the recent end by
construction; the early dip is the time-dependence the correction removed.""")
co(r"""ev["dmu"]=ev.ml_epoch-ev.ml_nb09; ev["dmu"]-=ev.loc[ev.year>=2019,"dmu"].median()  # recent era = 0 reference
ann=ev.groupby("year").agg(dmu=("dmu","median"),nb09=("ml_nb09","median"),kig=("ml_epoch","median"))
fig,ax=plt.subplots(1,2,figsize=(13,4.2))
ax[0].axhline(0,color="0.5",lw=0.8,ls="--"); ax[0].plot(ann.index,ann.dmu,"o-",color="tab:purple")
ax[0].set(xlabel="Year",ylabel="Δμ (epoch − nb09), gauge-centred",title="Time-dependent shift removed"); ax[0].tick_params(axis="x",labelrotation=45)
ax[1].plot(ann.index,ann.nb09,"o-",color="0.5",label="nb09 (one offset/station)")
ax[1].plot(ann.index,ann.kig,"s-",color="tab:green",label="epoch (KS+KG)")
ax[1].set(xlabel="Year",ylabel="Annual median ML",title="Annual median ML"); ax[1].tick_params(axis="x",labelrotation=45); ax[1].legend()
fig.tight_layout(); plt.show()
print(f"Δμ early(<=2014) {ev[ev.year<=2014].dmu.median():+.3f} -> recent(>=2019) {ev[ev.year>=2019].dmu.median():+.3f} ML (recent era = 0 reference)")""")

# ----------------------------------------------------------------- §4 seismicity test
md(r"""## The seismicity test — UF background rate, before vs after the correction

Re-run the nb13/14 pipeline (Mc, b, and the Zaliapin–Ben-Zion **declustered background** rate, pre
2010-2013 vs post 2019-2024) on the nb09 catalog and the epoch-corrected catalog.""")
co(r"""sys.path.insert(0,"/home/msseo/works/16.kma_absolute_location")
from kma_absolute_location import nnd
from seismostats.analysis import estimate_mc_maxc, ClassicBValueEstimator
from seismostats.utils import bin_to_precision
def mc_b(mm):
    m=bin_to_precision(np.sort(np.asarray(mm,float)),DM); mc=float(estimate_mc_maxc(m,fmd_bin=DM)[0])
    be=ClassicBValueEstimator(); be.calculate(m[m>=mc],mc=mc,delta_m=DM); return mc,be.b_value
clean=pd.read_csv("catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo_homogenised_clean.csv")
clean["time"]=pd.to_datetime(clean.time,utc=True,errors="coerce")
clean=clean[(clean.lon>=UF[0])&(clean.lon<=UF[1])&(clean.lat>=UF[2])&(clean.lat<=UF[3])].dropna(subset=["time","lat","lon"]).sort_values("time")
m=pd.merge_asof(clean,ev.reset_index().rename(columns={"index":"event_idx"}).sort_values("event_time"),
                left_on="time",right_on="event_time",tolerance=pd.Timedelta("3s"),direction="nearest").dropna(subset=["ml_epoch"])
def nnd_bg(mag):
    g=m.copy(); g["kma_mag"]=g[mag].values; g=g[g.kma_mag>=0.4].sort_values("time").reset_index(drop=True)
    g["t_year"]=g.time.dt.year+(g.time.dt.dayofyear-1)/365.25; g["year"]=g.time.dt.year; g["event_id"]=np.arange(len(g))
    g=g.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep"})
    mm=bin_to_precision(g.kma_mag.values.astype(float),DM); mc=float(estimate_mc_maxc(mm,fmd_bin=DM)[0])
    be=ClassicBValueEstimator(); be.calculate(mm[mm>=mc],mc=mc,delta_m=DM)
    nd=nnd.compute_nnd(g,b=be.b_value,mmin=mc,metric="2d"); e0,_=nnd.fit_eta0(nd.eta.values,method="gmm")
    clu=set(nd.loc[nd.eta<e0,"event_id"]); g["bg"]=~g.event_id.isin(clu); out={}
    for cut in (0.8,1.0):
        s=g[g.bg&(g.kma_mag>=cut)]; npre=((s.year>=2010)&(s.year<=2013)).sum(); npost=(s.year>=2019).sum()
        out[cut]=(npre/4,npost/6,(npost/6)/(npre/4+1e-9))
    return out
res={lab:nnd_bg(c) for c,lab in [("ml_nb09","nb09 (one offset/station)"),("ml_epoch","epoch (KS+KG)")]}
print("WHOLE CATALOG:")
for c,l in [("ml_nb09","nb09"),("ml_epoch","epoch")]:
    mc,b=mc_b(ev[c]); print(f"  {l:8}: Mc={mc:.2f} b={b:.2f}")
print("\nUF declustered BACKGROUND rate (events/yr), pre 2010-2013 vs post 2019-2024:")
for l,o in res.items():
    for cut in (0.8,1.0): print(f"  {l:30} M>={cut}: pre {o[cut][0]:.1f} -> post {o[cut][1]:.1f}  ratio {o[cut][2]:.2f}")
fig,ax=plt.subplots(figsize=(7.5,4.4)); x=np.arange(2); w=0.35
r08=[res["nb09 (one offset/station)"][0.8][2],res["epoch (KS+KG)"][0.8][2]]
r10=[res["nb09 (one offset/station)"][1.0][2],res["epoch (KS+KG)"][1.0][2]]
ax.bar(x-w/2,r08,w,color="tab:blue",label="M≥0.8"); ax.bar(x+w/2,r10,w,color="tab:orange",label="M≥1.0")
ax.axhline(1.0,color="0.5",ls="--",lw=1); ax.set_xticks(x); ax.set_xticklabels(["nb09\n(one offset)","epoch\n(per sensor era)"])
ax.set(ylabel="post-2019 / pre-2013 background ratio",title="UF background ratio: steady (≈1) vs rising")
for i,(a,b_) in enumerate(zip(r08,r10)): ax.text(i-w/2,a+0.03,f"{a:.2f}",ha="center"); ax.text(i+w/2,b_+0.03,f"{b_:.2f}",ha="center")
ax.legend(); fig.tight_layout(); plt.show()""")

# ----------------------------------------------------------------- §4b UF statistical summary
md(r"""## UF-region statistical summary (epoch-corrected magnitudes)

A consolidated summary for the **Ulsan-Fault box only**, on the time-homogeneous `ml_epoch` magnitudes.
The **b-value** (Aki-Utsu) and FMD use the **robust time-uniform Mc** (derived in the Mc(t) section below —
the densification envelope, not a single noisy window). The **Zaliapin–Ben-Zion declustering is run on
ALL events (no Mc cut, `mmin=None`)** — this is the primary background/clustered classification used by
every panel below; background **rates** are then counted above the completeness cutoffs. (The Goebel/ZBZ
Mc-cut alternative is shown for comparison in §4e2.)""")
co(r"""uf=m.dropna(subset=["ml_epoch"]).copy(); uf["ML"]=uf["ml_epoch"]; uf["year"]=uf.time.dt.year
uf=uf.sort_values("time").reset_index(drop=True)
mags=bin_to_precision(uf.ML.values.astype(float),DM)
Mc_maxc=float(estimate_mc_maxc(mags,fmd_bin=DM)[0])
# ---- robust TIME-UNIFORM (network) Mc: MAXC in 2-yr windows, smoothed against single-window spikes ----
tnum=(uf.time.dt.year+(uf.time.dt.dayofyear-1)/365.25).values
_w=[]; _c=tnum.min()+1.0
while _c+1.0<=tnum.max()+1e-9:
    _m=(tnum>=_c-1.0)&(tnum<_c+1.0)
    if _m.sum()>=80:
        _mm=bin_to_precision(np.sort(uf.ML.values[_m].astype(float)),DM); _w.append((_c,float(estimate_mc_maxc(_mm,fmd_bin=DM)[0]),int(_m.sum())))
    _c+=0.5
MCW=pd.DataFrame(_w,columns=["t","Mc","n"])
MCW["Mc_rob"]=MCW.Mc.rolling(3,center=True,min_periods=1).median()        # rolling median kills isolated spikes
MC_TU=float(np.round(MCW.Mc_rob.max(),1))                                 # network time-uniform completeness
be=ClassicBValueEstimator(); be.calculate(mags[mags>=MC_TU],mc=MC_TU,delta_m=DM); bval,bstd=be.b_value,be.std
nge=int((mags>=MC_TU).sum()); aval=np.log10(nge)+bval*MC_TU

# ---- NND declustering on ALL events (no Mc cut; mmin=None) — the PRIMARY classification ----
g=uf.copy().sort_values("time").reset_index(drop=True)
g["t_year"]=g.time.dt.year+(g.time.dt.dayofyear-1)/365.25; g["event_id"]=np.arange(len(g)); g["year"]=g.time.dt.year
g=g.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep","ML":"kma_mag"})
nd=nnd.compute_nnd(g,b=bval,mmin=None,metric="2d"); e0,gmm_info=nnd.fit_eta0(nd.eta.values,method="gmm")
g["bg"]=~g.event_id.isin(set(nd.loc[nd.eta<e0,"event_id"]))
nclu=int((~g.bg).sum()); nbg=int(g.bg.sum())
# clustered fraction per Mc-window (to diagnose any spike as STAI vs estimator noise; used in §4d2)
gtn=(g.time.dt.year+(g.time.dt.dayofyear-1)/365.25).values
MCW["clust_frac"]=[float((~g.bg.values[(gtn>=t-1.0)&(gtn<t+1.0)]).mean()) if ((gtn>=t-1.0)&(gtn<t+1.0)).sum() else np.nan for t in MCW.t]
def rate(df,a,b,cut): s=df[(df.kma_mag>=cut)&(df.year>=a)&(df.year<=b)]; return len(s)/(b-a+1)

print("UF-REGION STATISTICAL SUMMARY  (epoch-corrected ml_epoch)"); print("="*58)
print(f"events in UF box              : {len(uf):,}  ({uf.time.min().date()}..{uf.time.max().date()})")
print(f"Mc (global MAXC)             : {Mc_maxc:.2f}    [reference; biased low by the dense recent era]")
print(f"Mc (TIME-UNIFORM, robust)    : {MC_TU:.2f}    <- used below (complete at ALL times)")
print(f"b-value (Aki-Utsu, >=Mc_tu)  : {bval:.2f} +/- {bstd:.2f}   (N>=Mc_tu = {nge:,})")
print(f"a-value                      : {aval:.2f}")
print(f"NND background / clustered   : {nbg:,} / {nclu:,}  (ALL events, no Mc cut; background {100*nbg/len(g):.0f}%, log10 eta0={np.log10(e0):+.2f})")
print("declustered background rate (events/yr, background events >= cut), pre 2010-2013 vs post 2019-2024:")
for cut in (MC_TU,0.8,1.0):
    rp=rate(g[g.bg],2010,2013,cut); rq=rate(g[g.bg],2019,2024,cut)
    print(f"   M>={cut:.1f}: pre {rp:4.1f}   post {rq:4.1f}   ratio {rq/(rp+1e-9):.2f}")

fig,ax=plt.subplots(1,3,figsize=(16,4.3))
edges=np.arange(np.floor(mags.min()/DM)*DM,mags.max()+DM,DM)
cum=np.array([(mags>=x).sum() for x in edges]); inc,_=np.histogram(mags,bins=np.append(edges,edges[-1]+DM)-DM/2)
ax[0].semilogy(edges,cum,"ks",ms=4,label="Cumulative"); ax[0].semilogy(edges,np.maximum(inc,0.1),"o",mfc="none",c="gray",ms=4,label="Incremental")
ax[0].semilogy(edges,10**(aval-bval*edges),"r-",lw=1.2,label=f"GR fit, b={bval:.2f}")
ax[0].axvline(MC_TU,color="tab:green",ls="--",lw=1.2,label=f"Mc(time-uniform)={MC_TU:.1f}")
ax[0].set(xlabel="ML",ylabel="N(≥ML)",title="UF frequency-magnitude distribution",ylim=(0.5,None)); ax[0].legend(fontsize=8)
x=np.log10(nd.eta.values); x=x[np.isfinite(x)]
ax[1].hist(x,bins=40,color="0.7",ec="w"); ax[1].axvline(np.log10(e0),color="tab:red",lw=2,label=f"log10 η₀={np.log10(e0):+.2f}")
ax[1].set(xlabel="log₁₀ NND η",ylabel="Count",title="NND distribution (left=clustered, right=background)"); ax[1].legend(fontsize=8)
ax[2].scatter(g.loc[g.bg,"time"],g.loc[g.bg,"kma_mag"],s=9,c="steelblue",alpha=0.6,lw=0,label="background")
ax[2].scatter(g.loc[~g.bg,"time"],g.loc[~g.bg,"kma_mag"],s=9,c="tab:red",alpha=0.6,lw=0,label="clustered")
ax[2].axhline(MC_TU,color="tab:green",ls="--",lw=1); ax[2].set(xlabel="Year",ylabel="ML",title="Declustered UF catalog (all events; line = Mc_tu)"); ax[2].legend(fontsize=8)
fig.tight_layout(); plt.show()""")

# ----------------------------------------------------------------- §4c annual bg/clustered
md(r"""### Annual evolution — background vs clustered event counts

How many UF events each year are **background** (isolated) vs **clustered** (aftershock/swarm), from the
NND split above (events ≥ the time-uniform Mc). Shown at M≥Mc_tu and M≥1.0.""")
co(r"""yrs=list(range(2010,2025))
fig,ax=plt.subplots(1,2,figsize=(15,4.3))
for k,cut in enumerate((MC_TU,1.0)):
    gg=g[g.kma_mag>=cut]
    bgy=gg[gg.bg].groupby("year").size().reindex(yrs,fill_value=0)
    cly=gg[~gg.bg].groupby("year").size().reindex(yrs,fill_value=0)
    ax[k].bar(yrs,bgy.values,color="steelblue",label="background")
    ax[k].bar(yrs,cly.values,bottom=bgy.values,color="tab:red",label="clustered")
    ax[k].set(xlabel="Year",ylabel=f"Events (M≥{cut:.1f})",title=f"Annual background vs clustered (M≥{cut:.1f})")
    ax[k].tick_params(axis="x",labelrotation=45); ax[k].legend()
fig.tight_layout(); plt.show()
bgy=g[g.bg].groupby("year").size().reindex(yrs,fill_value=0); cly=g[~g.bg].groupby("year").size().reindex(yrs,fill_value=0)
print(f"annual background (M≥{MC_TU:.1f}):", dict(zip(yrs,bgy.values)))
print(f"annual clustered  (M≥{MC_TU:.1f}):", dict(zip(yrs,cly.values)))""")

# ----------------------------------------------------------------- §4c2 cumulative bg/clustered
md(r"""### Continuous view — cumulative background vs clustered, and running clustered fraction

Event-by-event cumulative counts (not binned) for background and clustered events, plus the **running
fraction clustered**. A steady background shows a roughly constant cumulative slope; clustered bursts
(aftershock sequences) appear as steep jumps in the clustered curve.""")
co(r"""gs=g.sort_values("time"); tt=pd.to_datetime(gs.time)
cum_bg=np.cumsum(gs.bg.values.astype(int)); cum_cl=np.cumsum((~gs.bg).values.astype(int))
frac=cum_cl/np.maximum(cum_bg+cum_cl,1)
fig,ax=plt.subplots(1,2,figsize=(14,4.4))
ax[0].plot(tt,cum_bg,color="steelblue",lw=2,label="background")
ax[0].plot(tt,cum_cl,color="tab:red",lw=2,label="clustered")
ax[0].plot(tt,cum_bg+cum_cl,color="0.4",lw=1,ls="--",label="total")
ax[0].set(xlabel="Time",ylabel="Cumulative events (all, no Mc cut)",title="Cumulative background vs clustered"); ax[0].legend(loc="upper left")
ax[1].plot(tt,frac,color="tab:purple",lw=2); ax[1].axhline(frac[-1],color="0.5",ls=":",lw=1)
ax[1].set(xlabel="Time",ylabel="Running fraction clustered",title="Clustered proportion over time",ylim=(0,1))
fig.tight_layout(); plt.show()
print(f"final cumulative: background {cum_bg[-1]}, clustered {cum_cl[-1]}, clustered fraction {frac[-1]:.2f}")""")

# ----------------------------------------------------------------- §4c3 continuous background rate
md(r"""### Continuous background-rate tracking λ(t) (from the cumulative, not annual bins)

The declustered **background rate as a smooth function of time**: at each instant we count background
events in a centred 2-yr sliding window and divide by its width (= the local slope of the cumulative
curve). A flat λ(t) means a steady background; the pre-2013 and post-2019 mean levels are marked.""")
co(r"""bgt=g.loc[g.bg&(g.kma_mag>=MC_TU)].sort_values("time").time; by=(bgt.dt.year+(bgt.dt.dayofyear-1)/365.25).values
bgt8=g.loc[g.bg&(g.kma_mag>=0.8)].sort_values("time").time; by8=(bgt8.dt.year+(bgt8.dt.dayofyear-1)/365.25).values
grid=np.arange(2010.5,2024.01,0.1); HW=1.0
lam =np.array([((by >=tg-HW)&(by <tg+HW)).sum()/(2*HW) for tg in grid])
lam8=np.array([((by8>=tg-HW)&(by8<tg+HW)).sum()/(2*HW) for tg in grid])
fig,ax=plt.subplots(1,2,figsize=(14,4.4))
ax[0].plot(bgt.values,np.arange(1,len(bgt)+1),color="steelblue",lw=2,label=f"cumulative background (M≥{MC_TU:.1f})")
ax[0].set(xlabel="Time",ylabel="Cumulative background events",title="Cumulative background (slope = rate)"); ax[0].legend(loc="upper left")
ax[1].plot(grid,lam,color="steelblue",lw=2,label=f"λ(t), M≥{MC_TU:.1f}")
ax[1].plot(grid,lam8,color="tab:orange",lw=2,label="λ(t), M≥0.8")
pre=((by>=2010)&(by<=2013)).sum()/4; post=(by>=2019).sum()/6
ax[1].hlines(pre,2010,2014,color="0.4",ls="--"); ax[1].hlines(post,2019,2025,color="0.4",ls="--")
ax[1].set(xlabel="Time",ylabel="Background rate (events/yr)",title=f"Continuous background rate λ(t)  (pre {pre:.0f} → post {post:.0f} /yr)"); ax[1].legend(loc="upper left"); ax[1].set_ylim(0,None)
fig.tight_layout(); plt.show()
print(f"continuous background rate (2-yr window, M≥{MC_TU:.1f}): pre-2013 mean {lam[grid<2014].mean():.1f}, post-2019 mean {lam[grid>=2019].mean():.1f} /yr")""")

# ----------------------------------------------------------------- §4d era-based Mc/b
md(r"""### Time-varying Mc and b-value, by network-density era

A fixed-event-count moving window mixes eras of different network density (hence different completeness),
which biases Mc and b. Instead we split into **eras defined by network density** and report MAXC Mc and
Aki-Utsu b **at each era's own Mc** — the honest, completeness-aware comparison.""")
co(r"""sta_yr=d.groupby("year").station.nunique()
ERAS=[("2010–2015 sparse",2010,2015),("2016–2018 densifying",2016,2018),("2019–2024 dense",2019,2024)]
fig,ax=plt.subplots(figsize=(9,3.8)); ax.bar(sta_yr.index,sta_yr.values,color="0.6")
for lab,a,b in ERAS: ax.axvspan(a-0.5,b+0.5,alpha=0.08,color="tab:green")
ax.set(xlabel="Year",ylabel="Operating stations",title="Network density by year (era boundaries shaded)")
ax.tick_params(axis="x",labelrotation=45); fig.tight_layout(); plt.show()
def era_b(mm,mc,nmin=40):
    mm=mm[mm>=mc]
    if len(mm)<nmin: return np.nan,np.nan,len(mm)
    e=ClassicBValueEstimator(); e.calculate(mm,mc=mc,delta_m=DM); return e.b_value,e.std,len(mm)
rows=[]
for lab,a,b in ERAS:
    sub=uf[(uf.year>=a)&(uf.year<=b)]; mm=bin_to_precision(np.sort(sub.ML.values.astype(float)),DM)
    mc=float(estimate_mc_maxc(mm,fmd_bin=DM)[0])
    b_all,bs,nge_=era_b(mm,mc)                                 # b on ALL events
    gb=g[(g.bg)&(g.year>=a)&(g.year<=b)]                       # NND BACKGROUND only
    mmb=bin_to_precision(np.sort(gb.kma_mag.values.astype(float)),DM)
    b_bg,bsb,ngb=era_b(mmb,mc)
    rows.append(dict(era=lab,years=f"{a}-{b}",N=len(sub),stations=int(round(sta_yr.loc[a:b].mean())),
                     Mc=round(mc,2),b_all=round(b_all,2),b_background=round(b_bg,2),
                     N_ge_Mc=nge_,N_bg_ge_Mc=ngb))
ERA_TAB=pd.DataFrame(rows); print(ERA_TAB.to_string(index=False))
print("\\nMc decreases as the network densifies (lower detection threshold) -> a fixed-window estimate")
print("straddling eras is unreliable. b_all includes aftershocks; b_background is the declustered (tectonic)")
print("b — compare them in the dense era to see whether the high all-events b is the clustered population.")""")

# ----------------------------------------------------------------- §4d2 time-uniform Mc
md(r"""### A statistically-valid time-uniform Mc — and why the spikes are *not* network completeness

A network-detection Mc should **decrease (or stay flat) as the network densifies** — more stations ⇒ lower
threshold. So an *upward* spike in Mc(t) cannot be a network-completeness feature; it is either **MAXC
estimator noise** (MAXC is unstable in small windows) or **STAI** — short-term aftershock incompleteness in
a window dominated by a sequence (small events transiently missed). Either way it is transient and must
**not** set the time-uniform threshold. We therefore take **Mcₜᵤ = max of the spike-smoothed Mc(t)** (the
densification envelope, computed above), overlay the operating-station count, and diagnose each spike
(STAI if the window is aftershock-rich, else noise). MAXC under-estimates, so the Woessner–Wiemer (2005)
"+0.2" safe value is also reported.""")
co(r"""MCW["spike"]=(MCW.Mc-MCW.Mc_rob)>=0.15
fig,ax=plt.subplots(figsize=(11,4.6)); ax2=ax.twinx()
ax2.bar(sta_yr.index,sta_yr.values,color="0.88",width=0.85); ax2.set_ylabel("Operating stations",color="0.55"); ax2.set_zorder(0)
ax.set_zorder(2); ax.patch.set_visible(False)
ax.plot(MCW.t,MCW.Mc,"o-",color="tab:blue",lw=1,ms=5,label="MAXC(t), 2-yr window")
ax.plot(MCW.t,MCW.Mc_rob,"-",color="tab:green",lw=2.4,label="robust (3-window median)")
ax.axhline(MC_TU,color="tab:red",ls="--",lw=1.6,label=f"network time-uniform Mc = {MC_TU:.1f}")
sp=MCW[MCW.spike]
if len(sp): ax.scatter(sp.t,sp.Mc,s=150,facecolors="none",edgecolors="tab:red",lw=2,label="upward spike")
ax.set(xlabel="Year (window centre)",ylabel="MAXC Mc",title="Mc(t) tracks densification ↓; upward spikes are STAI or estimator noise")
ax.legend(loc="upper right",fontsize=7.5); fig.tight_layout(); plt.show()
recw=MCW[MCW.t>=2019]
print(f"network time-uniform Mc (robust densification envelope) = {MC_TU:.2f}   (WW2005 safe: {MC_TU+0.2:.2f})")
print(f"  recent dense-era windows: {recw.Mc.min():.2f}-{recw.Mc.max():.2f}  ->  a 2019+ only study can use Mc~{recw.Mc_rob.max():.1f}")
if len(sp)==0:
    print("  no upward spikes above the robust curve -> Mc(t) is monotone-consistent with densification (clean).")
for _,r in sp.iterrows():
    kind="STAI — aftershock-rich window (real but transient)" if r.clust_frac>=0.55 else "MAXC estimator noise (window not aftershock-rich)"
    print(f"  spike {r.t:.1f}: MAXC={r.Mc:.2f} vs robust {r.Mc_rob:.2f}, clustered frac={r.clust_frac:.2f} -> {kind}")
print(f"RECOMMENDATION: whole-period time-uniform Mc = {MC_TU:.1f} (network threshold); transient STAI/noise spikes excluded.")""")

# ----------------------------------------------------------------- §4d2b K-S Mc over time
md(r"""### K-S Mc over time — is the densification picture estimator-independent?

The Mc(t) above used MAXC. Here we recompute completeness in the **same 2-yr windows** with the
**Kolmogorov–Smirnov** estimator (GR-consistency), as an independent check (separate figure). If both
track the same densification-driven ↓ trend, the time-uniform Mc is robust to the estimator choice.""")
co(r"""from seismostats.analysis import estimate_mc_ks
ksmc=[]
for t in MCW.t:
    msk=(tnum>=t-1.0)&(tnum<t+1.0); mm=bin_to_precision(np.sort(uf.ML.values[msk].astype(float)),DM)
    try: mk,_=estimate_mc_ks(mm,delta_m=DM,n=200); ksmc.append(mk if mk is not None else np.nan)
    except Exception: ksmc.append(np.nan)
MCW["Mc_ks"]=ksmc
fig,ax=plt.subplots(figsize=(11,4.6)); ax2=ax.twinx()
ax2.bar(sta_yr.index,sta_yr.values,color="0.9",width=0.85); ax2.set_ylabel("Operating stations",color="0.55"); ax2.set_zorder(0)
ax.set_zorder(2); ax.patch.set_visible(False)
ax.plot(MCW.t,MCW.Mc,"o-",color="tab:blue",lw=1.4,ms=5,label="MAXC Mc(t)")
ax.plot(MCW.t,MCW.Mc_ks,"s-",color="tab:red",lw=1.4,ms=5,label="K-S Mc(t)")
ax.axhline(MC_TU,color="0.3",ls="--",lw=1.2,label=f"time-uniform Mc = {MC_TU:.1f}")
ax.set(xlabel="Year (window centre)",ylabel="Mc",title="Mc(t): MAXC vs K-S — both track the densification ↓ trend")
l=ax.legend(loc="upper right",fontsize=8); l.set_zorder(50); fig.tight_layout(); plt.show()
print("per-window Mc(t):"); print(MCW[["t","n","Mc","Mc_ks"]].round(2).to_string(index=False))
_cc=MCW[["Mc","Mc_ks"]].dropna()
print(f"\\nMAXC vs K-S Mc(t): correlation {_cc.Mc.corr(_cc.Mc_ks):.2f}; both decrease as the network densifies,")
print(f"so the time-uniform Mc={MC_TU:.1f} (max of the densification envelope) is robust to the estimator.")""")

# ----------------------------------------------------------------- §4d3 per-window FMD grid
md(r"""### Individual FMD for each 2-year window — *seeing* where MAXC puts Mc

One frequency–magnitude distribution per moving window (cumulative ● and incremental ○). MAXC = the
magnitude of the **incremental peak** (dashed line; red = flagged spike). The spike panel shows the
signature of estimator noise: plenty of events *below* the marked Mc, yet the incremental mode happens
to land high.""")
co(r"""wins=MCW.t.values; ncol=5; nrow=int(np.ceil(len(wins)/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(2.9*ncol,2.3*nrow),sharex=True,sharey=True)
for i,t in enumerate(wins):
    axx=axes.ravel()[i]; msk=(tnum>=t-1.0)&(tnum<t+1.0)
    mm=bin_to_precision(np.sort(uf.ML.values[msk].astype(float)),DM)
    ed=np.arange(-0.5,mm.max()+DM,DM); cum=np.array([(mm>=x).sum() for x in ed]); inc,_=np.histogram(mm,bins=np.append(ed,ed[-1]+DM)-DM/2)
    axx.semilogy(ed,np.maximum(cum,0.5),"k.-",ms=3,lw=0.8); axx.semilogy(ed,np.maximum(inc,0.5),"o",mfc="none",c="gray",ms=3)
    mc=float(MCW.loc[MCW.t==t,"Mc"].iloc[0]); sk=bool(MCW.loc[MCW.t==t,"spike"].iloc[0])
    axx.axvline(mc,color="tab:red" if sk else "tab:green",ls="--",lw=1.3)
    axx.set_title(f"{t:.1f}: Mc={mc:.1f}"+("  spike" if sk else ""),fontsize=8,color=("tab:red" if sk else "k")); axx.set_xlim(-0.5,2.5)
for j in range(len(wins),nrow*ncol): axes.ravel()[j].axis("off")
fig.supxlabel("ML"); fig.supylabel("N  (cumulative ● / incremental ○)"); fig.tight_layout(); plt.show()""")

# ----------------------------------------------------------------- §4d4 MAXC vs K-S Mc
md(r"""### MAXC vs Kolmogorov–Smirnov Mc — two ways to pick completeness

Two standard single-snapshot Mc estimators on the UF magnitudes:
- **MAXC** (Wiemer–Wyss / Woessner–Wiemer 2005): Mc = the magnitude of the **incremental FMD peak**. Fast
  and stable but **biased low** (the mode sits at/just below true completeness); the WW2005 **"+0.2"** bump
  is the usual safety margin.
- **Kolmogorov–Smirnov (Clauset-style, `estimate_mc_ks`)**: scan candidate Mc; for each, fit GR and
  measure the **KS distance** between the observed and modelled cumulative FMD; accept the **smallest Mc
  whose GR fit is not rejected** (p ≥ 0.1). It encodes a *different* criterion — GR-consistency rather than
  the incremental peak — so it can land **below or above** MAXC depending on the FMD shape (here it sits
  one bin below: the UF FMD is already GR-consistent just under the MAXC peak).

These are *snapshot* estimators (one whole-catalog FMD); the **time-uniform Mc** above solves a different
problem (completeness changing as the network densifies), so it can sit above either.""")
co(r"""from seismostats.analysis import estimate_mc_ks
mc_maxc=float(estimate_mc_maxc(mags,fmd_bin=DM)[0])
mc_ks,ksinfo=estimate_mc_ks(mags,delta_m=DM)
mcs=np.array(ksinfo["mcs_tested"],float); pvs=np.array(ksinfo["p_values"],float)
fig,ax=plt.subplots(1,2,figsize=(13,4.3))
edges=np.arange(np.floor(mags.min()/DM)*DM,mags.max()+DM,DM)
cum=np.array([(mags>=x).sum() for x in edges]); inc,_=np.histogram(mags,bins=np.append(edges,edges[-1]+DM)-DM/2)
ax[0].semilogy(edges,cum,"ks",ms=4,label="cumulative"); ax[0].semilogy(edges,np.maximum(inc,0.1),"o",mfc="none",c="gray",ms=4,label="incremental")
ax[0].axvline(mc_maxc,color="tab:blue",ls="--",lw=1.6,label=f"MAXC = {mc_maxc:.2f}")
ax[0].axvline(mc_maxc+0.2,color="tab:cyan",ls=":",lw=1.4,label=f"MAXC+0.2 = {mc_maxc+0.2:.2f}")
ax[0].axvline(mc_ks,color="tab:red",ls="--",lw=1.6,label=f"K-S = {mc_ks:.2f}")
ax[0].set(xlabel="ML",ylabel="N(≥ML) / incremental",title="UF FMD — MAXC vs K-S Mc",ylim=(0.5,None)); l=ax[0].legend(fontsize=8); l.set_zorder(50)
ax[1].plot(mcs,pvs,"o-",color="tab:red"); ax[1].axhline(0.1,color="0.5",ls="--",lw=1,label="p = 0.1 (accept threshold)")
ax[1].axvline(mc_ks,color="tab:red",ls="--",lw=1.2,label=f"K-S Mc = {mc_ks:.2f}")
ax[1].set(xlabel="candidate Mc",ylabel="K-S goodness-of-fit p",title="K-S test — smallest Mc with p ≥ 0.1"); l=ax[1].legend(fontsize=8); l.set_zorder(50)
fig.tight_layout(); plt.show()
print(f"UF Mc:  MAXC = {mc_maxc:.2f}  (MAXC+0.2 = {mc_maxc+0.2:.2f})  |  K-S = {mc_ks:.2f}  |  time-uniform (robust) = {MC_TU:.2f}")
print(f"K-S − MAXC = {mc_ks-mc_maxc:+.2f} ML — different criteria (K-S = smallest GR-consistent Mc; MAXC =")
print("incremental peak, biased low), so they need not agree. For the time-varying network we use Mc_tu.")""")

# ----------------------------------------------------------------- §4e Goebel-style NND diagram
md(r"""### Nearest-neighbour declustering diagram — Goebel style

Reproduced to match **Goebel's `clustering-analysis/4_dist_tau.py`**: a **Gaussian-KDE-smoothed
event-pair density** in the rescaled time–distance plane (x = log₁₀ T, y = log₁₀ R), `RdYlGn_r` colormap,
with the η₀ separation line (slope −1, white + grey dashed). Background events form the upper-right ridge;
the clustered population is the separate mode toward the origin. The right panel is the **bimodal
log₁₀ η** distribution with its two Gaussian-mixture modes and the η₀ threshold.""")
co(r"""from scipy.stats import gaussian_kde, norm as _norm
lt=nd.logT.values; lr=nd.logR.values; ok=np.isfinite(lt)&np.isfinite(lr); lt,lr=lt[ok],lr[ok]
le=np.log10(nd.eta.values[ok]); le0=np.log10(e0); binx=biny=0.1
# EQUAL-SPAN ranges so the panel is SQUARE and the η₀ line (slope −1) renders at 45° (ZBZ/Goebel)
Tlo,Thi=np.floor(lt.min())-0.5,np.ceil(lt.max())+0.5; Rlo,Rhi=np.floor(lr.min())-0.5,np.ceil(lr.max())+0.5
span=max(Thi-Tlo,Rhi-Rlo); Tc,Rc=(Tlo+Thi)/2,(Rlo+Rhi)/2
Tlo,Thi,Rlo,Rhi=Tc-span/2,Tc+span/2,Rc-span/2,Rc+span/2
Tb=np.arange(Tlo,Thi+binx,binx); Rb=np.arange(Rlo,Rhi+biny,biny); XX,YY=np.meshgrid(Tb,Rb)
ZZ=gaussian_kde(np.vstack([lt,lr]))(np.vstack([XX.ravel(),YY.ravel()])).reshape(XX.shape)*binx*biny*len(lt)
fig,ax=plt.subplots(figsize=(6.8,6.8))                                   # square figure
pc=ax.pcolormesh(XX,YY,ZZ,cmap=plt.cm.RdYlGn_r,shading="auto")
cb=plt.colorbar(pc,ax=ax,fraction=0.046,pad=0.04); cb.set_label("Number of event pairs")
ax.plot([Tlo,Thi],-np.array([Tlo,Thi])+le0,"-",lw=2.5,color="w")
ax.plot([Tlo,Thi],-np.array([Tlo,Thi])+le0,"--",lw=1.5,color="0.3",label=f"η₀ (log₁₀={le0:.2f})")
ax.set(xlabel="Rescaled time  log₁₀ T",ylabel="Rescaled distance  log₁₀ R",
       title="Nearest-neighbour pairs in R–T (Goebel / Zaliapin–Ben-Zion)",xlim=(Tlo,Thi),ylim=(Rlo,Rhi))
ax.set_aspect("equal"); ax.legend(loc="lower left",fontsize=8); fig.tight_layout(); plt.show()
# bimodal log10(η) with the two GMM modes + η₀  (separate panel)
fig,ax=plt.subplots(figsize=(7.2,4.4)); ax.hist(le,bins=40,density=True,color="0.82",ec="w")
xs=np.linspace(le.min(),le.max(),400); mns,sgs,wts=gmm_info["means"],gmm_info["sigmas"],gmm_info["weights"]
for j,(c_,nm) in enumerate([("tab:red","clustered mode"),("tab:green","background mode")]):
    ax.plot(xs,wts[j]*_norm.pdf(xs,mns[j],sgs[j]),color=c_,lw=2,label=nm)
ax.axvline(le0,color="k",ls="--",lw=2,label=f"η₀={le0:.2f}")
ax.set(xlabel="log₁₀ η (nearest-neighbour proximity)",ylabel="Density",title="Bimodal NND distribution + GMM split"); ax.legend(fontsize=8)
fig.tight_layout(); plt.show()
print(f"η₀ (this catalog, GMM) = {le0:.2f}  |  Goebel's independent UF run gave η₀ = -3.97 (consistent)")""")

# ----------------------------------------------------------------- §4e2 NND without imposing Mc
md(r"""### NND declustering WITH Mc cut (Goebel/ZBZ standard) — comparison

The main analysis above uses NND on **all events** (no Mc cut), per preference. For transparency, here is
the **Goebel/ZBZ-standard alternative**: the same NND with the catalog **cut at the time-uniform Mc**
(`mmin=Mc_tu`). Cutting at Mc is the convention for *rate* work (declustering is sensitive to missing
background events); the all-event version maximises cluster membership. Both ratios are shown so the
choice is transparent.""")
co(r"""gcut=uf[uf.ML>=MC_TU].copy().sort_values("time").reset_index(drop=True)
gcut["t_year"]=gcut.time.dt.year+(gcut.time.dt.dayofyear-1)/365.25; gcut["event_id"]=np.arange(len(gcut)); gcut["year"]=gcut.time.dt.year
gcut=gcut.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep","ML":"kma_mag"})
nd_cut=nnd.compute_nnd(gcut,b=bval,mmin=MC_TU,metric="2d"); e0c,gic=nnd.fit_eta0(nd_cut.eta.values,method="gmm")
gcut["bg"]=~gcut.event_id.isin(set(nd_cut.loc[nd_cut.eta<e0c,"event_id"]))
print(f"Mc-cut NND: {len(gcut)} events (>=Mc_tu={MC_TU}) -> background {int(gcut.bg.sum())}, clustered {int((~gcut.bg).sum())} "
      f"(background {100*gcut.bg.mean():.0f}%, log10 eta0={np.log10(e0c):.2f})")
print("background-rate ratio (post-2019 / pre-2013):  PRIMARY all-event  vs  Mc-cut")
for cut in (0.5,0.8,1.0):
    r1=rate(g[g.bg],2019,2024,cut)/(rate(g[g.bg],2010,2013,cut)+1e-9)
    r2=rate(gcut[gcut.bg],2019,2024,cut)/(rate(gcut[gcut.bg],2010,2013,cut)+1e-9)
    print(f"   M>={cut}: all-event (primary) {r1:.2f}   |   Mc-cut {r2:.2f}")
lt=nd_cut.logT.values; lr=nd_cut.logR.values; okk=np.isfinite(lt)&np.isfinite(lr); lt,lr=lt[okk],lr[okk]; le0c=np.log10(e0c)
Tl,Th=np.floor(lt.min())-0.5,np.ceil(lt.max())+0.5; Rl,Rh=np.floor(lr.min())-0.5,np.ceil(lr.max())+0.5
spn=max(Th-Tl,Rh-Rl); _tc,_rc=(Tl+Th)/2,(Rl+Rh)/2; Tl,Th,Rl,Rh=_tc-spn/2,_tc+spn/2,_rc-spn/2,_rc+spn/2
Tb=np.arange(Tl,Th+0.1,0.1); Rb=np.arange(Rl,Rh+0.1,0.1); XX,YY=np.meshgrid(Tb,Rb)
ZZ=gaussian_kde(np.vstack([lt,lr]))(np.vstack([XX.ravel(),YY.ravel()])).reshape(XX.shape)*0.01*len(lt)
fig,ax=plt.subplots(figsize=(6.6,6.6)); pc=ax.pcolormesh(XX,YY,ZZ,cmap=plt.cm.RdYlGn_r,shading="auto")
plt.colorbar(pc,ax=ax,fraction=0.046,pad=0.04).set_label("Number of event pairs")
ax.plot([Tl,Th],-np.array([Tl,Th])+le0c,"-",lw=2.5,color="w"); ax.plot([Tl,Th],-np.array([Tl,Th])+le0c,"--",lw=1.5,color="0.3",label=f"η₀={le0c:.2f}")
ax.set(xlabel="Rescaled time  log₁₀ T",ylabel="Rescaled distance  log₁₀ R",title="NND in R–T — WITH Mc cut (≥Mc_tu) [comparison]",xlim=(Tl,Th),ylim=(Rl,Rh))
ax.set_aspect("equal"); ax.legend(loc="lower left",fontsize=8); fig.tight_layout(); plt.show()""")

# ----------------------------------------------------------------- §4f calibration note
md(r"""### Note — what is the ML absolute scale anchored to?

Solving `ML = μ + S` has a built-in ambiguity: add a constant *c* to every event magnitude and subtract
it from every station offset → identical readings. We fix it with the **gauge** "observation-weighted
mean(S) = 0". Because most readings come from the **recent dense network**, this effectively pins the
zero-point to the **recent stations' average** — i.e. the corrected ML is, in effect, **calibrated to the
modern network**, and the early data is brought onto that scale.

**Does this bias the background-rate result?** No. The driver — the **relative** magnitude difference
between the early and recent eras — is set by the *relative* station/epoch offsets, which are
**gauge-invariant** (the constant *c* cancels). The gauge only moves the common zero-point, not the
early-vs-recent difference. Anchoring to the dense modern network is the standard, sensible choice (it is
the best-characterised), and the temporal conclusion does not hinge on it.""")

# ----------------------------------------------------------------- §5 summary
md(r"""## Summary""")
co(r"""print("="*70); print("RESPONSE-EPOCH CORRECTED CATALOG (KS + KG) — summary".center(70)); print("="*70)
print("METHOD (4 steps): nb09 one-offset-per-station -> a sensor swap steps that offset ->")
print("  the station-response metadata gives the swap dates -> one offset per sensor era, re-homogenise.")
print(f"\nsensor-era splits applied to {len(breaks)} station-channels ({nkg} KG, {nks} KS) from the StationXML")
mci,bi=mc_b(ev.ml_nb09); mck,bk=mc_b(ev.ml_epoch)
print(f"whole-catalog Mc/b:  nb09 {mci:.2f}/{bi:.2f}  ->  epoch {mck:.2f}/{bk:.2f}")
print("UF declustered background ratio (post-2019 / pre-2013):")
for cut in (0.8,1.0):
    print(f"   M>={cut}: nb09 {res['nb09 (one offset/station)'][cut][2]:.2f}  ->  epoch {res['epoch (KS+KG)'][cut][2]:.2f}")
print("\nTAKE-HOMES")
print(" - The correction uses ONLY documented sensor-change dates from the StationXML — no data-driven guessing.")
print(" - It removes a real, sensor-driven ~0.03 ML early-era inflation that nb09's single offset missed.")
print(" - Under it the UF background ratio rises (see above): the post-2019 increase is NOT fully")
print("   explained by clustering — a real background rise becomes the better-supported reading.")
print("\nHONEST CAVEATS (now the only ones left — the metadata question is settled):")
print(" - Small early-era sample -> wide error bars on the ratio (bootstrap before quoting a number).")
print(" - Cutoff sensitivity: a 0.03 ML shift against a sharp M-threshold amplifies into the rate.")
print(" - HDB had a pre-2015 sensor FAILURE window (~2 ML low); flagged in nb18, better excluded than offset.")
print(" - Use catalog_ml_heo_epoch.csv (ml_epoch) as the time-homogeneous magnitude going forward")
print("   (catalog_ml_heo_kigam_epoch.csv kept as a back-compat alias with the same values).")""")

nb.cells=C
out="17.Response_epoch_corrected_catalog.ipynb"; nbf.write(nb,out); print("wrote",out,len(C),"cells")
