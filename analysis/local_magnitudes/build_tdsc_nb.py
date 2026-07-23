#!/usr/bin/env python
"""Generate 15.Time_dependent_station_corrections.ipynb — test the time-invariance assumption behind
the nb09 station corrections, and replace it with EPOCH-SPLIT (piecewise-constant in time) station
terms where the data demand it. Converts the moderate split-era stability (r=0.54, nb09) into a
measured, corrected result.

Pipeline:
 1. time-invariant median-polish baseline (= nb09), residual per reading r = ML - mu_event - S_station
 2. per-station residual time series + changepoint detection (own binary segmentation, robust median
    step, permutation significance) -> detected break dates + step sizes; flag mainshock-coincident
 3. epoch-split re-homogenisation: correction unit = (station-channel, epoch); re-run median polish
 4. impact on event magnitudes (Delta-mu vs time), and on Mc / b-value / UF-box pre-vs-post-2019 rate
 5. independent internal cross-check: anchor-only catalog temporal stability (invariant vs epoch-split)
Runs in `base`. References: Hutton & Boore 1987; Pérez et al. / Killick et al. 2012 (PELT changepoint);
Bormann NMSOP-2 (station corrections); Kumazawa & Ogata 2014 (non-stationarity in seismicity)."""
import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# Time-dependent station corrections — testing (and fixing) the time-invariance assumption

Notebook 09 estimated **time-invariant** station terms S_s and homogenised the Heo (2024) ML catalog.
Its own stability check found only **moderate** split-era agreement (r = 0.54) and flagged 3 stations
with |ΔS| > 0.15. A time-invariant term **cannot** absorb a real hardware/response change (sensor swap,
gain change, datalogger replacement, site change) — and such a change produces a **time-dependent ML
bias that survives homogenisation** and can masquerade as a real b / Mc / rate trend. That is precisely
the artifact the seismicity-statistics work (nb13/14) must be immune to.

**This notebook** (a) detects *when* each station's term changes, (b) re-homogenises with **epoch-split**
(piecewise-constant) terms, and (c) measures whether the change matters for the catalog statistics —
turning "we hope it's time-invariant (r=0.54)" into "we measured where it isn't, corrected it, and
quantified the impact."

**Why it matters here:** only **5 stations are persistent anchors** (operating ≥12 of 15 yr) that carry
the sparse early era — and 2 of them (KG.MKL, KG.HDB) are exactly the flagged drifters. The early-era
magnitudes lean on these few stations, so their drift has leverage.

*Method:* residual r_si = ML_si − μ_i − S_s after the nb09 fit; changepoints by binary segmentation
(robust median step + permutation test); epoch-split re-fit by median polish. *Caveat up front:* a
data-driven step can be a real instrument change **or** an apparent step from a changing event
population (e.g. a big sequence) — we flag break dates coincident with the 2016 Gyeongju / 2017 Pohang
mainshocks and treat those cautiously; metadata (StationXML response epochs) should confirm the rest.""")

# ----------------------------------------------------------------- §0 baseline
co(r"""import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib as mpl, matplotlib.font_manager as fm
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":120,"axes.grid":True,"grid.alpha":0.3,"font.size":11})

PS="catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo.csv"
SNR_MIN=3.0; DM=0.1
MAINSHOCKS={"Gyeongju 2016":pd.Timestamp("2016-09-12",tz="UTC"),
            "Pohang 2017":pd.Timestamp("2017-11-15",tz="UTC")}

d=pd.read_csv(PS); d=d[(d.snr>=SNR_MIN)&d.ML.notna()].copy()
d["t"]=pd.to_datetime(d.event_time,utc=True,errors="coerce"); d=d.dropna(subset=["t"])
d["year"]=d.t.dt.year; d["sc"]=d.network+"."+d.station+"."+d.channel
T0=d.t.min(); d["tnum"]=(d.t-T0).dt.total_seconds()/86400/365.25
scc=d.sc.value_counts()
print(f"{len(d):,} readings | {d.sc.nunique()} station-channels | {d.year.min()}-{d.year.max()}")

def median_polish(df, unit_col, n_iter=40, tol=1e-4):
    "joint {event mu, unit term S} by alternating medians; gauge: obs-weighted mean(S)=0."
    w=df[unit_col].value_counts()
    mu=df.groupby("event_idx").ML.median(); S=pd.Series(0.0,index=w.index); hist=[]
    for it in range(n_iter):
        res=df.ML.values-mu.reindex(df.event_idx).values
        Sn=pd.Series(res,index=df[unit_col]).groupby(level=0).median()
        Sn-=np.average(Sn.reindex(w.index).values,weights=w.values)
        corr=df.ML.values-Sn.reindex(df[unit_col]).values
        mun=pd.Series(corr,index=df.event_idx).groupby(level=0).median()
        dm=float(np.nanmax(np.abs(mun.reindex(mu.index).values-mu.values))); mu,S=mun,Sn
        hist.append(dm)
        if dm<tol: break
    return mu,S,hist

mu0,S0,hist0=median_polish(d,"sc")
d["res"]=d.ML - d.event_idx.map(mu0).values - d.sc.map(S0).values
print(f"time-invariant baseline (nb09): {len(hist0)} iters, station-term std {S0.std():.3f}")""")

# ----------------------------------------------------------------- §1 residual series
md(r"""## 1 · Per-station residual time series

After the time-invariant correction, each station's residual r = ML − μ_event − S_station should be a
flat, zero-mean band in time. A **step or ramp** is the signature of a term that changed — i.e. the
time-invariant S is wrong for that station. Below are the 5 persistent anchors (they carry the early
era) plus any other high-count movers.""")
co(r"""anchors=["KG.CGD.ELZ","KG.YSB.HHZ","KG.MKL.HHZ","KG.CHS.HHZ","KG.HDB.HHZ"]
fig,axes=plt.subplots(2,3,figsize=(16,7),sharex=True,sharey=True)
for ax,s in zip(axes.ravel(),anchors):
    sub=d[d.sc==s].sort_values("t")
    ax.scatter(sub.t,sub.res,s=3,alpha=0.15,color="0.5")
    q=sub.set_index("t").res.rolling("180D",min_periods=20).median()
    ax.plot(q.index,q.values,color="tab:red",lw=1.6,label="180-day median")
    ax.axhline(0,color="0.3",lw=0.8,ls="--")
    for nm,tt in MAINSHOCKS.items(): ax.axvline(tt,color="tab:blue",lw=0.8,ls=":")
    ax.set_title(f"{s}  (n={len(sub)})",fontsize=10); ax.set_ylim(-1.0,1.0)
    ax.set_ylabel("residual ML")
axes.ravel()[-1].axis("off")
fig.suptitle("Post-correction residual time series — flat band = time-invariant; step/ramp = drift",y=1.0)
fig.tight_layout(); plt.show()
print("blue dotted = Gyeongju 2016 / Pohang 2017 mainshocks")""")

# ----------------------------------------------------------------- §2 changepoints
md(r"""## 2 · Changepoint detection

Binary segmentation: at each level find the split maximising the robust median step, keep it only if a
**permutation test** rejects "no step" (p < 0.01) and the step exceeds 0.08 ML and both segments have
≥ 300 readings; recurse (≤ 3 breaks/station). Data-driven, so it catches **undocumented** changes —
the dangerous ones — not just those in metadata. Break dates within ±45 d of a mainshock are flagged as
possibly population-driven rather than instrumental.""")
co(r"""rng=np.random.default_rng(0)
def _best(tn,rv,lo,hi,minseg):
    "best split within sorted-array index range [lo,hi); returns absolute split index."
    n=hi-lo
    if n<2*minseg: return None
    sub=rv[lo:hi]; subt=tn[lo:hi]; best=None
    grid=max(1,(n-2*minseg)//250 or 1)
    for k in range(minseg,n-minseg,grid):
        dd=abs(np.median(sub[:k])-np.median(sub[k:]))
        if best is None or dd>best[1]: best=(k,dd)
    k,_=best; obs=abs(np.median(sub[k:])-np.median(sub[:k])); c=0; NP=300
    for _ in range(NP):
        p=rng.permutation(sub); c+=abs(np.median(p[:k])-np.median(p[k:]))>=obs
    return dict(kabs=lo+k,tcut=subt[k],step=float(np.median(sub[k:])-np.median(sub[:k])),p=(c+1)/(NP+1))
def find_cps(tn,rv,minseg=300,min_step=0.08,maxcp=3):
    o=np.argsort(tn); tn=tn[o]; rv=rv[o]               # sort once; segment by index ranges
    segs=[(0,len(rv))]; cps=[]
    for _ in range(maxcp):
        best=None
        for si,(lo,hi) in enumerate(segs):
            bs=_best(tn,rv,lo,hi,minseg)
            if bs and bs["p"]<0.01 and abs(bs["step"])>=min_step and (best is None or abs(bs["step"])>abs(best[0]["step"])):
                best=(bs,si)
        if best is None: break
        bs,si=best; lo,hi=segs.pop(si); k=bs["kabs"]; cps.append(bs); segs+=[(lo,k),(k,hi)]
    return sorted(cps,key=lambda x:x["tcut"])

rows=[]
for s in scc.index:
    sub=d[d.sc==s]
    if len(sub)<700: continue
    for cp in find_cps(sub.tnum.values,sub.res.values):
        dt=T0+pd.Timedelta(days=cp["tcut"]*365.25)
        near=[nm for nm,tt in MAINSHOCKS.items() if abs((dt-tt).days)<=45]
        rows.append(dict(sc=s,date=dt.date(),step=round(cp["step"],3),p=round(cp["p"],3),
                         n=len(sub),flag=("MAINSHOCK? "+near[0]) if near else ""))
CP=pd.DataFrame(rows).sort_values("step",key=abs,ascending=False)
print(f"{CP.sc.nunique()} station-channels show >=1 significant changepoint (of {(scc>=700).sum()} tested):")
print(CP.to_string(index=False))""")

# ----------------------------------------------------------------- §3 epoch-split refit
md(r"""## 3 · Epoch-split re-homogenisation

Each station-channel with a detected break is split into epochs (correction unit = `sc@epoch`); epochs
with < 50 readings are merged back. Re-run median polish on the expanded unit set → an epoch-split
homogenised catalog. Mainshock-coincident breaks are **kept** here (they still describe a real change in
that station's readings) but revisited in the impact section.""")
co(r"""MIN_EPOCH_N=50
def make_units(bmap):
    def eu(row):
        s=row.sc
        if s not in bmap: return s
        return f"{s}@e{sum(row.t.date()>=bd for bd in bmap[s])}"
    u=d.apply(eu,axis=1)
    uc=u.value_counts(); small=set(uc[uc<MIN_EPOCH_N].index)
    return u.where(~u.isin(small), d.sc)
breaks_full={s:sorted(g.date.tolist()) for s,g in CP.groupby("sc")}                  # all detected breaks
# CONSERVATIVE: 1 largest, non-mainshock, |step|>=0.15 step per station (most defensible as instrumental)
cons=CP[(CP.flag=="")&(CP.step.abs()>=0.15)].sort_values("step",key=abs,ascending=False).groupby("sc").head(1)
breaks_cons={s:sorted(g.date.tolist()) for s,g in cons.groupby("sc")}
d["unit"]=make_units(breaks_full);  mu1,S1,hist1=median_polish(d,"unit")
d["unit_c"]=make_units(breaks_cons); mu1c,S1c,_=median_polish(d,"unit_c")
print(f"FULL epoch-split:  {d.unit.nunique()} units, all {len(CP)} breaks -> {sorted(breaks_full)}")
print(f"CONSERVATIVE:      {d.unit_c.nunique()} units, 1 large non-mainshock step/station -> {sorted(breaks_cons)}")
ev=pd.DataFrame({"event_time":d.groupby('event_idx').t.first(),"year":d.groupby('event_idx').year.first(),
                 "ml_invariant":mu0,"ml_epoch":mu1.reindex(mu0.index),
                 "ml_epoch_cons":mu1c.reindex(mu0.index)}).dropna()
ev.to_csv("catalog_ml_heo_epoch_split.csv")
print(f"wrote catalog_ml_heo_epoch_split.csv ({len(ev):,} events)")""")

# ----------------------------------------------------------------- §4 impact on magnitudes
md(r"""## 4 · Impact on event magnitudes — is there a *time-dependent* shift?

The test that matters: Δμ = μ(epoch-split) − μ(time-invariant) vs time. A constant offset is irrelevant
(gauge). A **temporal trend/step in Δμ** is exactly the time-dependent bias the epoch-split removes — and
the magnitude of that trend is how much the time-invariant catalog was wrong in time.""")
co(r"""ev["dmu"]=ev.ml_epoch-ev.ml_invariant
ev["dmu"]-=ev.dmu.median()   # remove constant gauge offset; only time structure matters
ann=ev.groupby("year").agg(dmu_med=("dmu","median"),
                           inv=("ml_invariant","median"),epo=("ml_epoch","median"),n=("dmu","size"))
fig,ax=plt.subplots(1,2,figsize=(13,4.4))
ax[0].axhline(0,color="0.5",lw=0.8,ls="--")
ax[0].plot(ann.index,ann.dmu_med,"o-",color="tab:purple")
ax[0].set(xlabel="Year",ylabel="Δμ = epoch − invariant (gauge-centred)",
          title="Time-dependent magnitude shift removed by epoch-split"); ax[0].tick_params(axis="x",labelrotation=45)
ax[1].plot(ann.index,ann.inv,"o-",color="0.5",label="time-invariant (nb09)")
ax[1].plot(ann.index,ann.epo,"s-",color="tab:green",label="epoch-split")
ax[1].set(xlabel="Year",ylabel="Annual median ML",title="Annual median ML"); ax[1].tick_params(axis="x",labelrotation=45); ax[1].legend()
fig.tight_layout(); plt.show()
print(f"Δμ time span: early(<=2014) {ev[ev.year<=2014].dmu.median():+.3f} | "
      f"recent(>=2019) {ev[ev.year>=2019].dmu.median():+.3f} ML  (gauge-centred)")
print(f"max |annual Δμ| = {ann.dmu_med.abs().max():.3f} ML at {ann.dmu_med.abs().idxmax()}")""")

# ----------------------------------------------------------------- §5 impact on stats
md(r"""## 5 · Impact on the seismicity statistics — and a warning

**This is the safety test.** A correction that *reduces* time-dependence should leave the whole-catalog
Mc/b roughly intact and only gently adjust rates. If instead it **swings** them, the "correction" is
likely absorbing real signal (event-population / spatial structure), not instrument response. We compare
three catalogs: time-invariant (nb09), **conservative** epoch-split (one large non-mainshock step per
station), and **full** epoch-split (every detected break).""")
co(r"""from seismostats.analysis import estimate_mc_maxc, ClassicBValueEstimator
from seismostats.utils import bin_to_precision
def mc_b(mm):
    m=bin_to_precision(np.sort(np.asarray(mm,float)),DM); mc=float(estimate_mc_maxc(m,fmd_bin=DM)[0])
    be=ClassicBValueEstimator(); be.calculate(m[m>=mc],mc=mc,delta_m=DM); return mc,be.b_value,be.std
COLS=[("ml_invariant","time-invariant (nb09)"),("ml_epoch_cons","epoch-split CONSERVATIVE"),
      ("ml_epoch","epoch-split FULL")]
print("WHOLE-CATALOG Mc / b:")
for col,lab in COLS:
    mc,b,s=mc_b(ev[col]); print(f"  {lab:28}: Mc={mc:.2f}  b={b:.2f}±{s:.2f}")

clean=pd.read_csv("catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo_homogenised_clean.csv")
clean["time"]=pd.to_datetime(clean.time,utc=True,errors="coerce")
UF=(129.25,129.55,35.60,35.90)
clean=clean[(clean.lon>=UF[0])&(clean.lon<=UF[1])&(clean.lat>=UF[2])&(clean.lat<=UF[3])].dropna(subset=["time"]).sort_values("time")
e2=ev.reset_index().rename(columns={"index":"event_idx"}).sort_values("event_time")
mrg=pd.merge_asof(clean,e2,left_on="time",right_on="event_time",tolerance=pd.Timedelta("3s"),direction="nearest").dropna(subset=["ml_epoch"])
print(f"\nUF-box matched: {len(mrg):,} (corr ml_invariant vs catalog magnitude: {mrg.ml_invariant.corr(mrg.magnitude):.3f})")
PRE=(2010,2013); POST=(2019,2024); Tpre=4; Tpost=6; yr=mrg.time.dt.year
print("UF-box RAW rate (events/yr), pre 2010-2013 vs post 2019-2024 (raw = before declustering):")
for col,lab in [("ml_invariant","time-invariant"),("ml_epoch_cons","conservative"),("ml_epoch","full")]:
    for cut in (0.8,1.0):
        npre=((mrg[col]>=cut)&(yr>=PRE[0])&(yr<=PRE[1])).sum(); npost=((mrg[col]>=cut)&(yr>=POST[0])&(yr<=POST[1])).sum()
        print(f"  {lab:14} M>={cut}: pre {npre/Tpre:4.1f} -> post {npost/Tpost:4.1f}  ratio {(npost/Tpost)/(npre/Tpre+1e-9):.2f}")
print("NOTE: a 'correction' that swings b by >~0.1 or the ratio by ~50% is reshaping the FMD — a red flag")
print("      that the data-driven steps are confounded with the event population, not pure instrument response.")""")

# ----------------------------------------------------------------- §6 cross-check
md(r"""## 6 · Independent cross-check — anchor-only catalog temporal stability

Build event magnitudes from the **persistent anchors only** (fixed network composition → immune to
densification by construction), under both the time-invariant and the epoch-split terms. If the
epoch-split is doing its job, the anchor-only vs full-catalog discrepancy that drifts in time under the
invariant terms should **shrink** under epoch-split. *Caveat:* a better anchor fit can also just reflect
the extra free parameters of epoch-splitting (overfitting), so this is supportive, not proof.""")
co(r"""def anchor_series(unit_col, S):
    da=d[d.sc.isin(anchors)]
    mlc=(da.ML - da[unit_col].map(S).values)
    g=pd.DataFrame({"event_idx":da.event_idx.values,"mlc":mlc,"sc":da.sc.values})
    n=g.groupby("event_idx").sc.nunique(); keep=n[n>=2].index
    return g[g.event_idx.isin(keep)].groupby("event_idx").mlc.median()
a_inv=anchor_series("sc",S0); a_epo=anchor_series("unit",S1)
yrmap=d.groupby("event_idx").year.first()
def annual_diff(a_series, full_series):
    df=pd.DataFrame({"a":a_series,"f":full_series.reindex(a_series.index),"year":yrmap.reindex(a_series.index)}).dropna()
    df["d"]=df.a-df.f; df["d"]-=df.d.median()
    return df.groupby("year").d.median()
di=annual_diff(a_inv,mu0); de=annual_diff(a_epo,mu1)
fig,ax=plt.subplots(figsize=(9,4.4))
ax.axhline(0,color="0.5",lw=0.8,ls="--")
ax.plot(di.index,di.values,"o-",color="tab:red",label="time-invariant terms")
ax.plot(de.index,de.values,"s-",color="tab:green",label="epoch-split terms")
ax.set(xlabel="Year",ylabel="anchor-only − full (gauge-centred)",
       title="Anchor-only vs full catalog drift — epoch-split should flatten it"); ax.legend(); ax.tick_params(axis="x",labelrotation=45)
fig.tight_layout(); plt.show()
print(f"recent(>=2019) anchor−full drift:  invariant {di[di.index>=2019].mean():+.3f}  ->  epoch-split {de[de.index>=2019].mean():+.3f} ML")
print(f"std of annual drift:  invariant {di.std():.3f}  ->  epoch-split {de.std():.3f} ML")""")

# ----------------------------------------------------------------- §7 summary
md(r"""## 7 · Comprehensive summary""")
co(r"""print("="*74); print("TIME-DEPENDENT STATION CORRECTIONS — summary".center(74)); print("="*74)
print(f"readings {len(d):,} | station-channels {d.sc.nunique()} | persistent anchors {len(anchors)}")
print(f"\nCHANGEPOINTS: {CP.sc.nunique()} station-channels show a significant term change.")
for _,r in CP.iterrows():
    print(f"   {r.sc:14} {str(r.date):>12}  step {r.step:+.2f} ML  (p={r.p}, n={r.n}) {r.flag}")
mci,bi,_=mc_b(ev.ml_invariant); mcc,bc,_=mc_b(ev.ml_epoch_cons); mce,be_,_=mc_b(ev.ml_epoch)
print(f"\nWHOLE-CATALOG Mc/b:  invariant Mc={mci:.2f} b={bi:.2f} | conservative Mc={mcc:.2f} b={bc:.2f} | full Mc={mce:.2f} b={be_:.2f}")
print(f"Δμ (full − invariant) temporal: early(<=2014) {ev[ev.year<=2014].dmu.median():+.3f} -> recent(>=2019) {ev[ev.year>=2019].dmu.median():+.3f} ML (gauge-centred)")
print(f"anchor-only drift (recent): invariant {di[di.index>=2019].mean():+.3f} -> full epoch-split {de[de.index>=2019].mean():+.3f} ML")
print("\nTAKE-HOMES")
print(" - Time-invariance IS violated: significant residual steps in HDB, MKL, YSB, GUWB, POHB, DUC")
print("   (HDB/MKL up to ~0.5-0.6 ML). The nb09 single-term-per-station model is genuinely incomplete.")
print(" - BUT naive data-driven epoch-split is NOT a safe fix here: HDB & MKL show implausible 3x")
print("   ALTERNATING +-0.5 steps, and the FULL split SWINGS whole-catalog b and the UF raw ratio (see")
print("   above) -> it is RESHAPING the FMD, i.e. absorbing the time-varying event population/geometry")
print("   (the station-term <-> distance degeneracy nb09 flagged), not just instrument response.")
print(" - Verdict: the ML scale has REAL ~0.04-0.1 ML time-dependence, but correcting it reliably needs")
print("   StationXML response-epoch / metadata confirmation of each break. Data-driven splits over-correct.")
print(" - So nb13/14 should stand on the TIME-INVARIANT catalog, carrying this as the key open caveat;")
print("   do NOT adopt the epoch-split catalog for statistics until breaks are metadata-confirmed.")
print(" - Improved anchor consistency under splitting partly reflects added free parameters (overfitting).")
print("\nCAVEATS")
print(" - SNR>=3 censoring + heteroscedastic early-era magnitude error are NOT addressed here (separate)")
print("   low-end/time-dependent effects; this notebook only removes the station-term time-dependence.")
print(" - Anchor set itself contains drifters (HDB/MKL); the epoch-split anchor cross-check is the clean version.")
print(" - Changepoint dates are data-driven; metadata (response epochs) should corroborate the instrumental ones.")""")

nb.cells=C
out="15.Time_dependent_station_corrections.ipynb"; nbf.write(nb,out); print("wrote",out,len(C),"cells")
