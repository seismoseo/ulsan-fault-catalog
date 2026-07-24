#!/usr/bin/env python
"""Generate 23.UF_constant_network_ML.ipynb — the temporally-homogeneous (fixed reference network)
magnitude catalog and its verification. Demonstrates that holding the station set constant across
2010-2024 removes the network-geometry magnitude artifact at its source (no distance cap needed), and
separates the two distinct quantities: (1) detection completeness Mc(t) that genuinely steps down with
densification, vs (2) the magnitude SCALE that must be temporally stationary for secular studies.
Runs in `base`, cwd = local_magnitudes. Reads catalog_ml_heo_const.csv (from build_constant_network_ml.py)."""
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# Ulsan-Fault magnitudes on a constant reference network

**The problem.** The KS/KG network densified over 2010-2024 (≈12 → 38 stations within 60 km of the box).
Stations added after 2016/2019 sit far from the cluster and **over-correct** under the regional
distance-attenuation term — they read systematically high, **inflating event ML after 2019** (UF annual
median trends **+0.018 ML/yr**). A distance cap (≤60 km) removes the inflation but **censors the genuine
post-2019 completeness gain** (it drops the smallest events, forcing the magnitude floor artificially
flat). The two effects are entangled, so no per-event distance cut separates them.

**The fix (this notebook).** Measure every event with the **same fixed set of stations** for the whole
period. The scale then cannot drift with the network. Exactly **5 station-channels operate across the
full span** and sit ≤50 km from the box — the persistent anchors:

| anchor | median dist | role |
|---|---|---|
| KG.MKL.HHZ | 16 km | near |
| KG.HDB.HHZ | 24 km | near (epoch drift) |
| KG.YSB.HHZ | 38 km | (epoch drift) |
| KG.CGD.ELZ | 39 km | short-period anchor |
| KG.CHS.HHZ | 49 km | far anchor |

Epoch-dependent drift (HDB, YSB) is handled by the **same** median-polish + changepoint epoch-split
machinery as the time-dependent station-correction notebook, restricted to these 5.

**The reframing this makes explicit — two different quantities, do not conflate:**
1. **Detection completeness `Mc(t)`** *should* step down at 2016 and 2019 (the full evolving network
   detects smaller events) — shown in §4 from the full catalog.
2. **Magnitude-scale homogeneity** *must be stationary* for valid rate/`b` comparison — delivered by the
   constant network here.""")

# ----------------------------------------------------------------- setup
co(r"""import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":120,"axes.grid":True,"grid.alpha":0.3,"font.size":11})
from seismostats.analysis import estimate_mc_maxc, ClassicBValueEstimator
from seismostats.utils import bin_to_precision
DM=0.1; UF=[129.25,129.55,35.60,35.90]
ANCHORS=["KG.MKL.HHZ","KG.HDB.HHZ","KG.YSB.HHZ","KG.CGD.ELZ","KG.CHS.HHZ"]
SNR_PP_MIN=2.0
MAINSHOCKS={"Gyeongju 2016":pd.Timestamp("2016-09-12",tz="UTC"),"Pohang 2017":pd.Timestamp("2017-11-15",tz="UTC")}

# constant-network catalog (built by build_constant_network_ml.py)
ev=pd.read_csv("catalog_ml_heo_const.csv"); ev["event_time"]=pd.to_datetime(ev.event_time,utc=True,errors="coerce")
ev=ev.dropna(subset=["event_time"]).sort_values("event_time")
# clean catalog for locations + the full-network ml_all / capped magnitude (for comparison)
clean=pd.read_csv("catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo_homogenised_clean.csv")
clean["time"]=pd.to_datetime(clean.time,utc=True,errors="coerce"); clean=clean.dropna(subset=["time"]).sort_values("time")
m=pd.merge_asof(ev,clean[["time","lat","lon","magnitude","ml_all"]],left_on="event_time",right_on="time",
                tolerance=pd.Timedelta("3s"),direction="nearest")
uf=m[(m.lon>=UF[0])&(m.lon<=UF[1])&(m.lat>=UF[2])&(m.lat<=UF[3])].copy(); uf["year"]=uf.event_time.dt.year
rel=uf[uf.n_const>=3].copy()
def slope(df,col):   # annual-median linear slope over 2017-2024 (ML/yr)
    md_=[df.loc[df.year==y,col].median() for y in np.arange(2017,2025)]; return float(np.polyfit(np.arange(2017,2025),md_,1)[0])
print(f"constant-net events (region-wide): {len(ev):,}  | UF box: {len(uf):,}  | UF reliable (n_const>=3): {len(rel):,}")""")

# ----------------------------------------------------------------- §1 anchors + residual series
md(r"""## 1 · The 5 anchors and their residual stability

Per-station residual `r = ML − μ_event − S_station` after the time-invariant fit. A flat zero-mean band
means the station's term is time-invariant; a **step/ramp** is real drift that the epoch-split must
absorb. The blue dotted lines mark the 2016 Gyeongju / 2017 Pohang mainshocks (population-driven steps to
treat cautiously).""")
co(r"""ps=pd.read_csv("catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo_uncapped.csv")
ps["sc"]=ps.network+"."+ps.station+"."+ps.channel
d=ps[ps.sc.isin(ANCHORS)&(ps.snr_pp>=SNR_PP_MIN)&ps.ML.notna()].copy()
d["t"]=pd.to_datetime(d.event_time,utc=True,errors="coerce"); d=d.dropna(subset=["t"]); d["year"]=d.t.dt.year
T0=d.t.min(); d["tnum"]=(d.t-T0).dt.total_seconds()/86400/365.25
def median_polish(df,unit_col,n_iter=60,tol=1e-4):
    w=df[unit_col].value_counts(); mu=df.groupby("event_idx").ML.median(); S=pd.Series(0.0,index=w.index)
    for _ in range(n_iter):
        res=df.ML.values-mu.reindex(df.event_idx).values
        Sn=pd.Series(res,index=df[unit_col]).groupby(level=0).median(); Sn-=np.average(Sn.reindex(w.index).values,weights=w.values)
        corr=df.ML.values-Sn.reindex(df[unit_col]).values; mun=pd.Series(corr,index=df.event_idx).groupby(level=0).median()
        dm=float(np.nanmax(np.abs(mun.reindex(mu.index).values-mu.values))); mu,S=mun,Sn
        if dm<tol: break
    return mu,S
mu0,S0=median_polish(d,"sc"); d["res"]=d.ML-d.event_idx.map(mu0).values-d.sc.map(S0).values
fig,axes=plt.subplots(2,3,figsize=(16,7),sharex=True,sharey=True)
for ax,s in zip(axes.ravel(),ANCHORS):
    sub=d[d.sc==s].sort_values("t"); ax.scatter(sub.t,sub.res,s=3,alpha=0.15,color="0.5")
    q=sub.set_index("t").res.rolling("180D",min_periods=20).median(); ax.plot(q.index,q.values,color="tab:red",lw=1.6)
    ax.axhline(0,color="0.3",lw=0.8,ls="--")
    for nm,tt in MAINSHOCKS.items(): ax.axvline(tt,color="tab:blue",lw=0.8,ls=":")
    ax.set_title(f"{s}  (n={len(sub)}, S={S0[s]:+.2f})",fontsize=10); ax.set_ylim(-1,1); ax.set_ylabel("residual ML")
axes.ravel()[-1].axis("off")
fig.suptitle("Constant-network anchors — post-correction residual stability",y=1.0); fig.tight_layout(); plt.show()
print("time-invariant anchor terms:", {s.split('.')[1]:round(float(S0[s]),2) for s in ANCHORS})""")

# ----------------------------------------------------------------- §2 documented epochs
md(r"""## 2 · Epoch handling — the documented sensor breaks (same recipe as nb17/nb18)

Drift is corrected with the **documented sensor-shape change dates** (`responses/sensor_breaks_master.json`,
derived from the StationXML response pole/zero counts) — **not** ad-hoc data-driven steps. The anchors
carry rich histories: **HDB 4 breaks** (2010-03, 2012-11, 2015-05, 2019-02), **YSB 6 breaks**, CHS 1, MKL 1.
Each station-channel is split into eras (`sc@epoch`); epochs with < 50 readings merge back. HDB's
**2014–2015 sensor-failure window** (residual ≪ 0; see the record sections in nb24) is given its **own
epoch** and corrected (offset ≈ −1.9 ML) — kept, not excluded. With HDB 1-of-5 and the failure ≈ 7
readings, the event ML is identical either way (b, slope, n all unchanged); this just keeps the data.""")
co(r"""import json
breaks_str=json.load(open("/home/msseo/works/02.Ulsan_Fault_detection/data/metadata/responses/sensor_breaks_master.json"))
breaks={k:[pd.Timestamp(x).date() for x in v] for k,v in breaks_str.items() if k in ANCHORS}
print("documented epoch breaks:")
for s in ANCHORS: print(f"  {s:14} {breaks.get(s,[])}")
d["res_raw"]=d.ML-d.event_idx.map(mu0).values     # raw station residual, to locate the failure window
# HDB sensor-FAILURE window: monthly residual << 0 in 2014-2015 -> its OWN epoch (corrected), not excluded
_hm=d[d.sc=="KG.HDB.HHZ"].set_index("t").res_raw.groupby(pd.Grouper(freq="ME")).median()
_f=_hm[(_hm<-1.0)&(_hm.index<pd.Timestamp("2015-06",tz="UTC"))]
FAIL_ON=pd.Timestamp(_f.index.min()).replace(day=1) if len(_f) else pd.Timestamp("2014-12-01",tz="UTC"); FAIL_OFF=pd.Timestamp("2015-05-21",tz="UTC")
breaks["KG.HDB.HHZ"]=sorted(set(breaks.get("KG.HDB.HHZ",[])+[FAIL_ON.date()]))   # failure onset = extra break -> own epoch
MIN_EPOCH_N=50
def era_unit(row):
    s=row.sc
    return s if s not in breaks else f"{s}@e{sum(row.t.date()>=bd for bd in breaks[s])}"
FAIL_UNIT=f"KG.HDB.HHZ@e{sum(pd.Timestamp('2015-02-01').date()>=bd for bd in breaks['KG.HDB.HHZ'])}"
u=d.apply(era_unit,axis=1); uc=u.value_counts(); small=set(uc[uc<MIN_EPOCH_N].index)-{FAIL_UNIT}  # protect failure epoch
d["unit"]=u.where(~u.isin(small),d.sc)
mu_e,S_e=median_polish(d,"unit"); dd=d
dd["res_inv"]  =dd.ML-dd.event_idx.map(mu0).values-dd.sc.map(S0).values        # after single offset
dd["res_epoch"]=dd.ML-dd.event_idx.map(mu_e).values-dd.unit.map(S_e).values    # after epoch split
print(f"\nHDB failure epoch {FAIL_UNIT} [{FAIL_ON.date()}..{FAIL_OFF.date()}]: offset {S_e.get(FAIL_UNIT,float('nan')):+.2f} ML, n={int((dd.unit==FAIL_UNIT).sum())}")
print(f"epoch units: {dd.unit.nunique()}")
for s in ("KG.HDB.HHZ","KG.YSB.HHZ"):
    sub=dd[dd.sc==s]; print(f"  {s}: median |residual|  single {sub.res_inv.abs().median():.3f} -> epoch {sub.res_epoch.abs().median():.3f} ML")""")

# ----------------------------------------------------------------- §2b epoch flattening
md(r"""### 2b · How the epoch split flattens the drifting stations

For the two anchors with rich histories — HDB and YSB — the 180-day rolling median of the per-station
residual. *Before* (one constant offset, red) it walks across the documented breaks (blue dotted);
*after* (epoch-split, green) each era gets its own offset and the band sits flat on zero. The HDB
**failure window** (red shading) is excluded, so the severe 2014–2015 dip is removed rather than smeared
into the offset. **Right:** the UF event-level annual median — single-offset vs epoch-corrected — showing
the scale flatten once HDB's full history is corrected.""")
co(r"""fig,ax=plt.subplots(1,3,figsize=(17,4.4))
for a,s in zip(ax[:2],["KG.HDB.HHZ","KG.YSB.HHZ"]):
    sub=dd[dd.sc==s].sort_values("t").set_index("t")
    a.scatter(sub.index,sub.res_inv,s=3,alpha=0.10,color="0.6")
    a.plot(sub.index,sub.res_inv.rolling("180D",min_periods=15).median(),color="tab:red",lw=1.9,label="before (one offset)")
    a.plot(sub.index,sub.res_epoch.rolling("180D",min_periods=15).median(),color="tab:green",lw=1.9,label="after (epoch-split)")
    a.axhline(0,color="0.3",lw=0.8,ls="--")
    for bd in breaks.get(s,[]): a.axvline(pd.Timestamp(bd,tz="UTC"),color="tab:blue",lw=0.9,ls=":")
    if s=="KG.HDB.HHZ": a.axvspan(FAIL_ON,FAIL_OFF,color="tab:red",alpha=0.13,label="failure epoch (offset ≈−1.9)")
    a.set(title=f"{s.split('.')[1]} residual (dotted = documented breaks)",xlabel="Year",ylabel="residual ML (180-day median)",ylim=(-0.8,0.8)); a.legend(fontsize=8.5)
yy=np.arange(2012,2025)
mi=[rel.loc[rel.year==y,"ml_const_inv"].median() for y in yy]; me=[rel.loc[rel.year==y,"ml_const"].median() for y in yy]
ax[2].plot(yy,mi,"o-",color="0.5",label=f"single offset (slope {slope(rel,'ml_const_inv'):+.4f})")
ax[2].plot(yy,me,"s-",color="tab:green",lw=2,label=f"epoch-corrected (slope {slope(rel,'ml_const'):+.4f})")
ax[2].set(title="UF annual median — epoch flattening",xlabel="Year",ylabel="median ML"); ax[2].legend(fontsize=8.5); ax[2].tick_params(axis="x",labelrotation=45)
fig.suptitle("Documented sensor epochs + HDB failure-window exclusion flatten the anchors",y=1.02); fig.tight_layout(); plt.show()
print(f"UF median slope 2017-24:  single offset {slope(rel,'ml_const_inv'):+.4f}  ->  epoch-corrected {slope(rel,'ml_const'):+.4f} ML/yr")""")

# ----------------------------------------------------------------- §3 the acceptance test (stationarity)
md(r"""## 3 · Acceptance test — is the magnitude scale temporally stationary?

The decisive diagnostic is the **rate ratio vs threshold**. A *magnitude-scale artifact* (inflation)
keeps the post/pre ratio **high at all thresholds**; a *completeness artifact* makes it **collapse** at
high thresholds. A *stationary scale* gives a **roughly constant** ratio (the real rate difference,
identical at every magnitude). We compare the uncapped, ≤60 km-capped, and constant-network magnitudes.""")
co(r"""def ladder(df,col):
    out={}
    for thr in (0.8,1.0,1.3,1.5,2.0):
        e=[df[(df.year>=a)&(df.year<=b)&(df[col]>=thr)].shape[0]/(b-a+1) for a,b in [(2010,2013),(2014,2018),(2019,2024)]]
        out[thr]=(e,e[2]/(e[0]+1e-9))
    return out
ufA=uf.copy(); ufA["year"]=ufA.event_time.dt.year
L_unc=ladder(ufA,"ml_all"); L_cap=ladder(ufA.dropna(subset=["magnitude"]),"magnitude"); L_con=ladder(rel,"ml_const")
thrs=[0.8,1.0,1.3,1.5,2.0]
fig,ax=plt.subplots(1,2,figsize=(14,4.6))
ax[0].plot(thrs,[L_unc[t][1] for t in thrs],"o-",color="tab:red",label="uncapped (full network)")
ax[0].plot(thrs,[L_cap[t][1] for t in thrs],"s-",color="tab:orange",label="≤60 km cap")
ax[0].plot(thrs,[L_con[t][1] for t in thrs],"D-",color="tab:green",lw=2,label="constant network")
ax[0].axhline(1,color="0.5",ls="--",lw=1)
ax[0].set(xlabel="Magnitude threshold",ylabel="Rate ratio  post-2019 / pre-2013",
          title="Rate ratio vs threshold (flat = stationary scale)"); ax[0].legend()
# annual median trend
yy=np.arange(2014,2025)
for col,lab,c in [("ml_all","uncapped",'tab:red'),("magnitude","≤60 km cap",'tab:orange'),("ml_const","constant net",'tab:green')]:
    src=rel if col=="ml_const" else ufA
    md_=[src.loc[src.year==y,col].median() for y in yy]; ax[1].plot(yy,md_,"o-",color=c,label=lab)
ax[1].set(xlabel="Year",ylabel="Annual median ML",title="Annual median ML (UF box)"); ax[1].legend(); ax[1].tick_params(axis="x",labelrotation=45)
fig.tight_layout(); plt.show()
def slope(df,col):
    md_=[df.loc[df.year==y,col].median() for y in np.arange(2017,2025)]; return float(np.polyfit(np.arange(2017,2025),md_,1)[0])
print(f"UF annual-median slope 2017-2024:  uncapped {slope(ufA,'ml_all'):+.4f}  |  cap {slope(ufA.dropna(subset=['magnitude']),'magnitude'):+.4f}  |  constant-net {slope(rel,'ml_const'):+.4f}  ML/yr")
print("\nRate ratio post-2019/pre-2013 by threshold:")
print(f"{'thr':>5} {'uncapped':>9} {'cap':>7} {'const':>7}")
for t in thrs: print(f"{t:>5} {L_unc[t][1]:9.2f} {L_cap[t][1]:7.2f} {L_con[t][1]:7.2f}")""")

# ----------------------------------------------------------------- §4 the two quantities
md(r"""## 3b · Time–magnitude and the ML lower bound over time

The clearest view of "what densification does to the magnitudes": a time–magnitude scatter with rolling
1-yr envelopes (5th/10th percentile = the lower bound, plus the median). **Left:** the full evolving
network (`ml_all`) — the lower bound **steps down** in two stages (2016, 2019). **Right:** the constant
reference network (`ml_const`) — the lower bound is **flat**, because the same stations set the detection
limit for every event.

**Why the 5/10th-percentile floor drops but the annual median does not** (on the full network): they
measure different things.
- The **percentile floor** is the catalog's lower *edge* — it tracks **detection completeness** directly.
  When the dense network occasionally catches one more sub-`Mc` event, the 5th-percentile boundary slides
  down immediately, even if only a handful of events populate that tail.
- The **median** is central tendency, ≈ `Mc + log₁₀2 / b`, anchored by the **many** events near `Mc`
  (Gutenberg–Richter is bottom-heavy). It only moves if the **bulk** completeness `Mc` moves. After the
  2016 densification the bulk `Mc` plateaued (~0.5) and the 2019 additions extended mainly the **extreme
  tail** (best-case near-station sensitivity), not the typical/bulk sensitivity — so the floor kept
  dropping while the median stayed put. (This is also why MAXC `Mc`, which follows the bulk peak, misses
  the 2019 step.)""")
co(r"""cuf=clean[(clean.lon>=UF[0])&(clean.lon<=UF[1])&(clean.lat>=UF[2])&(clean.lat<=UF[3])].dropna(subset=["time"]).copy()
def env(times,vals):
    s=pd.Series(np.asarray(vals,float),index=pd.DatetimeIndex(times)).sort_index().dropna()
    return (s.rolling("365D",min_periods=30).quantile(0.05),
            s.rolling("365D",min_periods=30).quantile(0.10),
            s.rolling("365D",min_periods=30).median())
fig,ax=plt.subplots(1,2,figsize=(15,5),sharey=True)
sub=cuf.dropna(subset=["ml_all"]).sort_values("time")
p5,p10,pm=env(sub.time,sub.ml_all.values)
ax[0].scatter(sub.time,sub.ml_all,s=4,alpha=0.12,color="0.6")
ax[0].plot(p5.index,p5,color="tab:purple",lw=1.9,label="5th pct (floor)")
ax[0].plot(p10.index,p10,color="tab:blue",lw=1.4,label="10th pct")
ax[0].plot(pm.index,pm,color="tab:red",lw=1.9,label="median")
ax[0].set(title="Full evolving network (ml_all) — floor steps DOWN",xlabel="Year",ylabel="ML",ylim=(-1.2,4.0))
sub2=rel.dropna(subset=["ml_const"]).sort_values("event_time")
q5,q10,qm=env(sub2.event_time,sub2.ml_const.values)
ax[1].scatter(sub2.event_time,sub2.ml_const,s=5,alpha=0.18,color="0.6")
ax[1].plot(q5.index,q5,color="tab:purple",lw=1.9,label="5th pct (floor)")
ax[1].plot(q10.index,q10,color="tab:blue",lw=1.4,label="10th pct")
ax[1].plot(qm.index,qm,color="tab:red",lw=1.9,label="median")
ax[1].set(title="Constant reference network (ml_const) — floor FLAT",xlabel="Year",ylim=(-1.2,4.0))
for a in ax:
    for yr in (2016,2019): a.axvline(pd.Timestamp(f"{yr}-01-01",tz="UTC"),color="0.4",ls=":",lw=1)
    a.legend(loc="upper right",fontsize=9)
fig.suptitle("Time–magnitude: lower bound steps down on the full network, stays flat on the constant network",y=1.0)
fig.tight_layout(); plt.show()
def band(s,t,col,a,b):
    x=s[(s[t].dt.year>=a)&(s[t].dt.year<=b)][col]; return x.quantile(.05),x.median()
for lab,a,b in [("2012-2015",2012,2015),("2016-2018",2016,2018),("2019-2024",2019,2024)]:
    f5,fm=band(cuf.dropna(subset=["ml_all"]),"time","ml_all",a,b); c5,cm=band(rel,"event_time","ml_const",a,b)
    print(f"{lab}: full-net p5 {f5:+.2f} median {fm:+.2f}  |  const-net p5 {c5:+.2f} median {cm:+.2f}")""")

md(r"""## 4 · The two quantities, separated

**(a) Detection completeness** from the *full evolving network* (uncapped `ml_all`) — the productivity-
robust 5th-percentile floor in a sliding window. This is what *should* step down with densification, and
it does: **two steps, 2016 and 2019**. (MAXC `Mc` saturates ~0.5 and is confounded by the 2016 aftershock
flood, so it cannot show the second step — the percentile floor can.)

**(b) Magnitude scale** from the *constant network* — flat by construction, so rates/`b` are comparable
across time.""")
co(r"""fig,ax=plt.subplots(1,2,figsize=(14,4.6))
# (a) full-network completeness staircase
ca=clean[(clean.lon>=UF[0])&(clean.lon<=UF[1])&(clean.lat>=UF[2])&(clean.lat<=UF[3])].dropna(subset=["time","ml_all"]).copy()
t=(ca.time.dt.year+(ca.time.dt.dayofyear-1)/365.25).values; v=ca.ml_all.values
g=2011.0; HW=0.75; rows=[]
while g<=2024.3:
    msk=(t>=g-HW)&(t<g+HW); s=v[msk]; s=s[~np.isnan(s)]
    if len(s)>=60: rows.append((g,np.percentile(s,5),np.percentile(s,10)))
    g+=0.5
S=pd.DataFrame(rows,columns=["t","p5","p10"])
for c in ("p5","p10"): S[c]=S[c].rolling(3,center=True,min_periods=1).median()
ax[0].plot(S.t,S.p5,"o-",color="tab:purple",label="5th-pct floor")
ax[0].plot(S.t,S.p10,"s-",color="0.5",label="10th-pct floor")
for yr in (2016,2019): ax[0].axvline(yr,color="tab:blue",ls=":",lw=1)
ax[0].set(xlabel="Year",ylabel="ML floor (full network)",title="(a) Detection completeness — steps DOWN at 2016 & 2019"); ax[0].legend()
# (b) constant-network stationary scale: annual median + IQR
yy=np.arange(2012,2025); med=[rel.loc[rel.year==y,"ml_const"].median() for y in yy]
q1=[rel.loc[rel.year==y,"ml_const"].quantile(.25) for y in yy]; q3=[rel.loc[rel.year==y,"ml_const"].quantile(.75) for y in yy]
ax[1].fill_between(yy,q1,q3,alpha=0.2,color="tab:green",label="IQR")
ax[1].plot(yy,med,"o-",color="tab:green",lw=2,label="annual median")
ax[1].axhline(np.median(rel.ml_const),color="0.4",ls="--",lw=1)
ax[1].set(xlabel="Year",ylabel="constant-network ML",title="(b) Magnitude scale — stationary by construction"); ax[1].legend(); ax[1].tick_params(axis="x",labelrotation=45)
fig.tight_layout(); plt.show()
print("(a) full-network p5 floor:", {int(r.t):round(r.p5,2) for _,r in S.iterrows() if r.t in (2012,2015,2017,2020,2023)})
print("(b) constant-net median is flat: slope", round(slope(rel,'ml_const'),4),"ML/yr (2017-2024)")""")

# ----------------------------------------------------------------- §5 FMD
md(r"""## 4c · Moving-window Mc on the constant network — is completeness time-invariant?

The constant network's whole premise is a **time-invariant completeness**. Direct test: Mc in a sliding
2-yr window on the constant-network magnitudes (`ml_const`, n_const ≥ 3), by **two independent estimators**
— **MAXC** (maximum curvature) and **K-S** (Clauset / Kolmogorov–Smirnov goodness-of-fit). If the constant
network works, both should stay ~flat near the bulk Mc (~0.8) across 2010–2024 — unlike the full evolving
network (grey dashed), whose MAXC Mc steps down at 2016.""")
co(r"""from seismostats.analysis import estimate_mc_ks
def _mc(method,s):
    s=np.asarray(s,float); s=s[~np.isnan(s)]
    if len(s)<40: return np.nan
    m=bin_to_precision(np.sort(s),DM)
    try:
        if method=="maxc": return float(estimate_mc_maxc(m,fmd_bin=DM)[0])
        return float(np.asarray(estimate_mc_ks(m,delta_m=DM)[0]).ravel()[0])
    except Exception: return np.nan
def mcwin(df,tcol,vcol,HW=1.0,step=0.5,minN=50):
    t=(df[tcol].dt.year+(df[tcol].dt.dayofyear-1)/365.25).values; v=df[vcol].values
    g=2011.0; rows=[]
    while g<=2024.3:
        s=v[(t>=g-HW)&(t<g+HW)]; s=s[~np.isnan(s)]
        if len(s)>=minN: rows.append((g,_mc("maxc",s),_mc("ks",s),len(s)))
        g+=step
    return pd.DataFrame(rows,columns=["t","maxc","ks","n"])
CW=mcwin(rel,"event_time","ml_const")          # constant network
FW=mcwin(cuf,"time","ml_all")                  # full evolving network (cuf from 4a), for contrast
fig,ax=plt.subplots(figsize=(10,4.6))
ax.plot(CW.t,CW.maxc,"o-",color="tab:green",lw=2,label="constant net — MAXC")
ax.plot(CW.t,CW.ks,"s-",color="tab:blue",lw=2,label="constant net — K-S")
ax.plot(FW.t,FW.maxc,"--",color="0.55",lw=1.6,label="full net — MAXC (contrast)")
for yr in (2016,2019): ax.axvline(yr,color="0.7",ls=":",lw=1)
ax.set(xlabel="Year (window centre)",ylabel="Mc (ML)",ylim=(0.0,1.35),
       title="Moving-window Mc — constant network stays ~flat (MAXC & K-S); full network steps down at 2016")
ax.legend(fontsize=9,ncol=2); fig.tight_layout(); plt.show()
print(f"constant net Mc(t):  MAXC {CW.maxc.mean():.2f}±{CW.maxc.std():.2f}   K-S {CW.ks.mean():.2f}±{CW.ks.std():.2f}   (range MAXC {CW.maxc.min():.2f}-{CW.maxc.max():.2f})")
print(f"full net Mc(t):      MAXC {FW.maxc.mean():.2f}±{FW.maxc.std():.2f}   (2011 {FW.maxc.iloc[0]:.2f} -> 2023 {FW.maxc.iloc[-1]:.2f}; steps down at 2016)")
print(CW.round(2).to_string(index=False))""")

md(r"""## 4d · Same events, different network — what if we estimate ML with ALL stations?

A direct control for your question: take the **identical event set** (n_const ≥ 3) and recompute the
magnitude two ways — the **constant 5 stations** (`ml_const`) vs **all available stations at the time**
(`ml_all`, the full evolving network). Only the *measurement* network differs; the events are fixed. This
isolates whether the network-geometry drift lives in the event SELECTION or in the MAGNITUDE ESTIMATION.""")
co(r"""rr=rel.dropna(subset=["ml_const","ml_all"]).copy()
fig,ax=plt.subplots(1,2,figsize=(14,4.6))
yy=np.arange(2012,2025)
for col,c,lab in [("ml_const","tab:green","constant 5 stns"),("ml_all","tab:red","all stations")]:
    mdv=[rr.loc[rr.year==y,col].median() for y in yy]
    ax[0].plot(yy,mdv,"o-",color=c,lw=2,label=f"{lab} (slope {slope(rr,col):+.4f})")
ax[0].set(xlabel="Year",ylabel="annual median ML (same events)",title="Same events — constant vs full-network ML")
ax[0].legend(fontsize=9); ax[0].tick_params(axis="x",labelrotation=45)
dif=(rr.ml_all-rr.ml_const).values; t=(rr.event_time.dt.year+(rr.event_time.dt.dayofyear-1)/365.25).values
o=np.argsort(t)
ax[1].scatter(t,dif,s=6,alpha=0.2,color="0.5")
roll=pd.Series(dif[o]).rolling(120,min_periods=20,center=True).median()
ax[1].plot(t[o],roll.values,color="tab:purple",lw=2,label="rolling median")
ax[1].axhline(0,color="0.4",ls="--",lw=1)
for yr in (2016,2019): ax[1].axvline(yr,color="0.7",ls=":",lw=1)
ax[1].set(xlabel="Year",ylabel="ml_all − ml_const (same event)",title="Full-network minus constant-network ML",ylim=(-0.6,0.6))
ax[1].legend(fontsize=9); fig.tight_layout(); plt.show()
print(f"slope 2017-24 (same events):  constant {slope(rr,'ml_const'):+.4f}   full network {slope(rr,'ml_all'):+.4f} ML/yr")
for lab,a,b in [("2010-2015",2010,2015),("2016-2018",2016,2018),("2019-2024",2019,2024)]:
    s=rr[(rr.year>=a)&(rr.year<=b)]; print(f"  {lab}: median(ml_all-ml_const) {(s.ml_all-s.ml_const).median():+.3f}")
def _bb(col):
    mg=bin_to_precision(np.sort(rr[col].dropna().values),DM); be=ClassicBValueEstimator(); be.calculate(mg[mg>=0.8],mc=0.8,delta_m=DM); return be.b_value
print(f"  b (Mc 0.8): constant {_bb('ml_const'):.2f}   full network {_bb('ml_all'):.2f}")""")

md(r"""### 4d-ii · Time–magnitude floor, SAME events, both networks

The §3b time–magnitude view, now restricted to the **identical event set** (n_const ≥ 3), measured by the
full network (`ml_all`, left) vs the constant network (`ml_const`, right). Because the events are the same,
the two floors are **nearly identical** — proving the dramatic full-network floor *drop* in §3b came from
the **extra small events** the dense network detects, **not** from measuring these events differently.""")
co(r"""def env(times,vals):
    s=pd.Series(np.asarray(vals,float),index=pd.DatetimeIndex(times)).sort_index().dropna()
    return (s.rolling("365D",min_periods=20).quantile(0.05),s.rolling("365D",min_periods=20).quantile(0.10),s.rolling("365D",min_periods=20).median())
fig,ax=plt.subplots(1,2,figsize=(15,5),sharey=True)
for a,(col,lab) in zip(ax,[("ml_all","Full network (all stns) — same events"),("ml_const","Constant network (5 stns) — same events")]):
    sub=rr.dropna(subset=[col]).sort_values("event_time")
    p5,p10,pm=env(sub.event_time,sub[col].values)
    a.scatter(sub.event_time,sub[col],s=6,alpha=0.18,color="0.6")
    a.plot(p5.index,p5,color="tab:purple",lw=1.9,label="5th pct (floor)")
    a.plot(p10.index,p10,color="tab:blue",lw=1.4,label="10th pct")
    a.plot(pm.index,pm,color="tab:red",lw=1.9,label="median")
    for yr in (2016,2019): a.axvline(pd.Timestamp(f"{yr}-01-01",tz="UTC"),color="0.4",ls=":",lw=1)
    a.set(title=lab,xlabel="Year",ylabel="ML",ylim=(-1.2,4.0)); a.legend(loc="upper right",fontsize=9)
fig.suptitle("Same events, two networks — floors nearly identical (§3b's full-net floor drop came from EXTRA events)",y=1.0,fontsize=10.5)
fig.tight_layout(); plt.show()
for lab,a,b in [("2012-2015",2012,2015),("2016-2018",2016,2018),("2019-2024",2019,2024)]:
    s=rr[(rr.year>=a)&(rr.year<=b)]
    print(f"{lab}: full-net p5 {s.ml_all.quantile(.05):+.2f} med {s.ml_all.median():+.2f}  |  const p5 {s.ml_const.quantile(.05):+.2f} med {s.ml_const.median():+.2f}")""")

md(r"""## 4e · Direct evidence for the distance-dependent over-correction

The foundation under everything above. **(a)** Each reading's deviation from its event's *near-station
truth* (median of stations < 30 km), **after removing each station's constant site term**, plotted vs
source distance: it climbs monotonically (≈ 0 within 30 km → **+0.14 at 60–80 km → +0.26 beyond**),
proving the regional Heo −logA₀ **over-corrects long paths**. The constant-net anchors (green ticks, ≤ 49 km)
sit in the low-bias zone. **(b)** the consequence for UF: full-network minus near-truth ML is ≈ 0 before
2019 and **+0.04 after**, tracking the rising fraction of far readings. **(c)** a UF-only station/epoch
correction (small box → per-station distance ≈ constant) **reduces** the drift (slope +0.018 → +0.011,
b 0.90 → 1.03) but cannot fully remove it — the residual within-box distance spread and post-2019-only
stations remain. This is why a *fixed near network* (or a genuine distance recalibration) is needed, not
just per-station correction.""")
co(r"""import json
psf=pd.read_csv("catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo_uncapped.csv")
psf=psf[(psf.snr_pp>=2)&psf.ML.notna()&psf.dist_km.notna()].copy()
psf["t"]=pd.to_datetime(psf.event_time,utc=True,errors="coerce"); psf=psf.dropna(subset=["t"]); psf["sc"]=psf.network+"."+psf.station+"."+psf.channel
# (a) bias curve: deviation from near-truth (<30 km, >=2 near), per-station site term removed
nref=psf[psf.dist_km<30].groupby("event_idx").ML.agg(["median","size"]); nref=nref[nref["size"]>=2]["median"].rename("ref")
bb=psf.merge(nref,on="event_idx"); bb["dev"]=bb.ML-bb.ref; bb["dev_c"]=bb.dev-bb.groupby("sc").dev.transform("median")
ed=[0,15,30,45,60,80,120]; xb=[(ed[i]+ed[i+1])/2 for i in range(len(ed)-1)]
yb=[bb[(bb.dist_km>=lo)&(bb.dist_km<hi)].dev_c.median() for lo,hi in zip(ed[:-1],ed[1:])]
# UF readings
ufset=set(np.round(clean[(clean.lon.between(UF[0],UF[1]))&(clean.lat.between(UF[2],UF[3]))].time.astype("int64")/1e9).astype(int))
psf["uf"]=np.round(psf.t.astype("int64")/1e9).astype(int).isin(ufset); d=psf[psf.uf].copy(); d["year"]=d.t.dt.year
near=d[d.dist_km<30].groupby("event_idx").ML.median().rename("ml_near"); full=d.groupby("event_idx").ML.median().rename("ml_full")
et=d.groupby("event_idx").t.first(); bz=pd.concat([near,full,et.rename("t")],axis=1).dropna(); bz["year"]=bz.t.dt.year
yrs=np.arange(2012,2025)
infl=[(bz[bz.year==y].ml_full-bz[bz.year==y].ml_near).median() for y in yrs]
farf=[(d[d.year==y].dist_km>45).mean() for y in yrs]
# (c) UF-only correction
breaks=json.load(open("/home/msseo/works/02.Ulsan_Fault_detection/data/metadata/responses/sensor_breaks_master.json")); breaks={k:[pd.Timestamp(x).date() for x in v] for k,v in breaks.items()}
def _eu(r):
    s=r.sc; return s if s not in breaks else f"{s}@e{sum(r.t.date()>=b for b in breaks[s])}"
u=d.apply(_eu,axis=1); uc=u.value_counts(); d["unit"]=u.where(~u.isin(set(uc[uc<50].index)),d.sc)
mu_raw=d.groupby("event_idx").ML.median(); mu_uf,_=median_polish(d,"unit"); y2=d.groupby("event_idx").t.first().dt.year
rawm=[mu_raw[y2.reindex(mu_raw.index)==y].median() for y in yrs]; ufm=[mu_uf[y2.reindex(mu_uf.index)==y].median() for y in yrs]
def _sl(v): m=np.asarray(v); ix=[list(yrs).index(y) for y in range(2017,2025)]; return float(np.polyfit(np.arange(2017,2025),m[ix],1)[0])
fig,ax=plt.subplots(1,3,figsize=(17,4.7))
ax[0].scatter(bb.dist_km,bb.dev_c,s=2,alpha=0.03,color="0.6")
ax[0].plot(xb,yb,"o-",color="tab:red",lw=2.2,label="median deviation"); ax[0].axhline(0,color="0.4",ls="--",lw=1)
for a_ in (16,24,38,39,49): ax[0].axvline(a_,color="tab:green",lw=0.8,ls=":")
ax[0].set(xlim=(0,125),ylim=(-0.4,0.5),xlabel="Source distance (km)",ylabel="ML − near-truth (site-removed)",title="(a) −logA₀ over-corrects far paths"); ax[0].legend(fontsize=8,loc="upper left")
ax[1].plot(yrs,infl,"o-",color="tab:purple",lw=2,label="ml_full − ml_near (UF)"); ax[1].axhline(0,color="0.4",ls="--",lw=1)
for yr in (2016,2019): ax[1].axvline(yr,color="0.7",ls=":",lw=1)
ax[1].set(xlabel="Year",ylabel="full − near ML",title="(b) Inflation appears post-2019"); ax[1].legend(loc="upper left",fontsize=8)
axt=ax[1].twinx(); axt.plot(yrs,farf,"s--",color="0.55",lw=1.1); axt.set_ylabel("frac readings > 45 km",color="0.45")
ax[2].plot(yrs,rawm,"o-",color="tab:red",lw=2,label=f"RAW (slope {_sl(rawm):+.3f})")
ax[2].plot(yrs,ufm,"s-",color="tab:blue",lw=2,label=f"UF-only corr ({_sl(ufm):+.3f})")
for yr in (2016,2019): ax[2].axvline(yr,color="0.7",ls=":",lw=1)
ax[2].set(xlabel="Year",ylabel="annual median ML",title="(c) UF-only correction helps, not enough"); ax[2].legend(fontsize=8,loc="upper left")
fig.suptitle("Evidence: Heo −logA₀ over-corrects far paths → post-2019 inflation; per-station correction only partly fixes it",y=1.02,fontsize=11)
fig.tight_layout(); plt.show()
print("deviation from near-truth (site-removed) by distance:",{f"{lo}-{hi}":round(v,3) for (lo,hi),v in zip(zip(ed[:-1],ed[1:]),yb)})
print(f"UF inflation (ml_full-ml_near): pre-2019 ~{np.nanmedian(infl[:7]):+.3f}, post-2019 ~{np.nanmedian(infl[7:]):+.3f}")
print(f"UF-region temporal slope 2017-24:  RAW {_sl(rawm):+.4f}  ->  UF-only corrected {_sl(ufm):+.4f}  (constant net -0.005)")""")

md(r"""## 5 · UF frequency-magnitude distribution on the homogeneous scale

`b` on the temporally-homogeneous constant-network magnitudes (the value to quote for secular/`b`-value
interpretation).""")
co(r"""mags=bin_to_precision(np.sort(rel.ml_const.dropna().values),DM)
mc=float(estimate_mc_maxc(mags,fmd_bin=DM)[0]); be=ClassicBValueEstimator(); be.calculate(mags[mags>=mc],mc=mc,delta_m=DM)
aval=np.log10((mags>=mc).sum())+be.b_value*mc
edges=np.arange(np.floor(mags.min()/DM)*DM,mags.max()+DM,DM)
cum=np.array([(mags>=x).sum() for x in edges]); inc,_=np.histogram(mags,bins=np.append(edges,edges[-1]+DM)-DM/2)
fig,ax=plt.subplots(figsize=(7.5,5))
ax.semilogy(edges,cum,"ks",ms=4,label="Cumulative"); ax.semilogy(edges,np.maximum(inc,0.1),"o",mfc="none",c="gray",ms=4,label="Incremental")
ax.semilogy(edges,10**(aval-be.b_value*edges),"r-",lw=1.2,label=f"GR fit b={be.b_value:.2f}±{be.std:.2f}")
ax.axvline(mc,color="tab:green",ls="--",lw=1.2,label=f"Mc={mc:.1f}")
ax.set(xlabel="ML (constant network)",ylabel="N(≥ML)",title="UF FMD — constant reference network",ylim=(0.5,None)); ax.legend()
fig.tight_layout(); plt.show()
print(f"UF constant-network FMD: Mc={mc:.2f}  b={be.b_value:.2f}±{be.std:.2f}  (N≥Mc={int((mags>=mc).sum())})")""")

# ----------------------------------------------------------------- §6 summary
md(r"""## 6 · Summary

**Computed results (UF box, epoch-corrected constant network, 5 anchors, n_const ≥ 3):**

| quantity | value |
|---|---|
| reliable UF events (n_const ≥ 3) | ≈ 890 |
| annual-median slope 2017–2024 | ≈ **−0.005 ML/yr** (stationary; uncapped +0.018, cap −0.0015) |
| rate ratio post-2019/pre-2013 | ≈ **1.7, flat across M≥0.8…1.5** (scale-consistent, real) |
| **UF b (Mc 0.80)** | **≈ 1.07 ± 0.06** (stable 1.07/1.08/1.07 at Mc 0.8/0.9/1.0) |
| moving-window Mc(t) | **flat: MAXC 0.81±0.11, K-S 0.46±0.10** (no 2016/2019 step; §4c) |
| epoch handling | documented breaks (HDB 4, YSB 6, …) + HDB failure window as its own epoch (−1.94) |

**Take-homes**

1. **The post-2019 ML inflation was real** (uncapped median +0.018 ML/yr; far new stations over-correct).
   It is a *network-geometry* artifact of the **evolving** station set, not of the Heo formula.
2. **A distance cap is the wrong tool** — it removes the inflation but censors the smallest post-2019
   events, forcing the floor flat. A flat floor is itself an artifact.
3. **A constant reference network is the right tool** — same 5 stations measure every event, so the
   magnitude *scale* cannot drift. The result is temporally stationary (slope ≈ 0; rate ratio flat across
   thresholds) **without** discarding small-event sensitivity per se.
4. **Two quantities must be kept separate.** Detection completeness *does* step down at 2016 **and 2019**
   (full-network p5 floor, §4a) — densification genuinely lets the network see smaller events. The
   magnitude *scale* must be held stationary (constant network, §4b) for `b`/rate work. Conflating them is
   what produced both the apparent post-2019 inflation and the misleading "flat" cap.
5. **Completeness is time-invariant** (§4c): constant-network moving-window Mc is flat under *both* MAXC
   (0.81±0.11) and K-S (0.46±0.10) — no 2016/2019 step — whereas the full network's MAXC Mc steps down at
   2016. The MAXC level matches the bulk Mc (0.80).
6. **It's the measurement network, not the event selection** (§4d): on the *identical* n_const≥3 events,
   estimating ML with all stations re-introduces a mild upward drift (slope +0.006 vs −0.005 ML/yr) — the
   network-geometry inflation lives in the *measurement*. b is unchanged (1.06 vs 1.07).
7. **For secular UF studies, quote `b ≈ 1.07` (Mc 0.80) on the constant-network catalog.** The full
   catalog (with `ml_all` for location) remains the right input for relocation/HypoDD.""")

nb["cells"]=C
import os
os.chdir("/home/msseo/works/02.Ulsan_Fault_detection/analysis/local_magnitudes")
nbf.write(nb,"23.UF_constant_network_ML.ipynb")
print("wrote 23.UF_constant_network_ML.ipynb with",len(C),"cells")
