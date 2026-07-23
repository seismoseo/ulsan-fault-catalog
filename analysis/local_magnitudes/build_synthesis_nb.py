#!/usr/bin/env python
"""Generate 16.Magnitude_timedependence_synthesis.ipynb — ONE clear, pedagogical synthesis of the whole
station-correction / instrument-response / magnitude-time-dependence investigation, written to be
understood end-to-end. It connects: (1) how ML is measured, (2) time-invariant station corrections
(nb09) and why they are incomplete, (3) what the StationXML response metadata says, (4) which residual
magnitude steps are real instrument/response changes vs spurious, and (5) the impact on the UF
background-rate conclusion (nb13/14). Ends with a plain verdict and 3 decision options.
Self-contained; runs in `base`."""
import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# Magnitude time-dependence: station corrections, instrument response, and what it means

**Read this first — the whole story in five sentences.**
1. Our seismicity statistics (Mc, b-value, rates) are only as trustworthy as the **time-homogeneity of
   the ML magnitudes**, so a magnitude that drifts with time would fake a real trend.
2. The nb09 **station corrections** assume each station's term is **constant in time** — but the data
   show that assumption is **violated** (residual magnitude *steps* at a few key stations).
3. The **StationXML** explains why: those stations had **documented instrument/response changes**, and
   while the big scalar *gain* jumps were correctly removed, **response-*shape* changes left residual
   magnitude steps** (the response metadata is imperfect there).
4. **Some** detected steps are real (they sit on documented response changes); **some are spurious**
   (they sit on nothing, or on the 2017 Pohang mainshock — i.e. the changing earthquake population).
5. **Correcting** the real ones lowers early-era magnitudes by ~0.03 ML and **raises the UF declustered
   background ratio from ~1.2 to ~1.7** — so the earlier "background steady" conclusion (nb13/14) is
   **not robust**, and this is the key thing to resolve before publication.

This notebook shows each step with the actual numbers and one decisive figure, then lists the options.""")

# ----------------------------------------------------------------- setup
co(r"""import warnings; warnings.filterwarnings("ignore")
import os, sys, numpy as np, pandas as pd, obspy
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
from collections import defaultdict
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":120,"axes.grid":True,"grid.alpha":0.3,"font.size":11})

PS="catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo.csv"
XML="responses/master/KS_KG_metadata_1.0.2.xml"
DM=0.1; UF=(129.25,129.55,35.60,35.90)
MAINSHOCKS={"Gyeongju 2016":pd.Timestamp("2016-09-12",tz="UTC"),"Pohang 2017":pd.Timestamp("2017-11-15",tz="UTC")}

d=pd.read_csv(PS); d=d[(d.snr>=3)&d.ML.notna()].copy()
d["t"]=pd.to_datetime(d.event_time,utc=True,errors="coerce"); d=d.dropna(subset=["t"])
d["year"]=d.t.dt.year; d["sc"]=d.network+"."+d.station+"."+d.channel
T0=d.t.min(); d["tnum"]=(d.t-T0).dt.total_seconds()/86400/365.25
print(f"{len(d):,} per-station readings | {d.sc.nunique()} station-channels | {d.year.min()}-{d.year.max()}")

def median_polish(df,col,n=40,tol=1e-4):
    "joint {event magnitude mu, station/unit term S}; gauge obs-weighted mean(S)=0."
    w=df[col].value_counts(); mu=df.groupby("event_idx").ML.median(); S=pd.Series(0.0,index=w.index)
    for _ in range(n):
        Sn=pd.Series(df.ML.values-mu.reindex(df.event_idx).values,index=df[col]).groupby(level=0).median()
        Sn-=np.average(Sn.reindex(w.index),weights=w.values)
        mun=pd.Series(df.ML.values-Sn.reindex(df[col]).values,index=df.event_idx).groupby(level=0).median()
        if float(np.nanmax(np.abs(mun.reindex(mu.index).values-mu.values)))<tol: mu,S=mun,Sn; break
        mu,S=mun,Sn
    return mu,S
mu0,S0=median_polish(d,"sc")
d["res"]=d.ML - d.event_idx.map(mu0).values - d.sc.map(S0).values   # residual after time-invariant correction""")

# ----------------------------------------------------------------- §1
md(r"""## 1 · Two layers: how a magnitude is built

**Layer A — measurement** (per station, per event): remove the instrument response (deconvolution
using the StationXML), simulate a Wood-Anderson seismograph, read the peak amplitude *A*, then
`ML = log10(A) + (−logA0_Heo)(distance)`. **Key idea:** if the StationXML response is *correct*, this
step already removes any instrument/gain change — the magnitude should not care which hardware was
installed.

**Layer B — station corrections** (nb09): even with perfect responses, each site has a small fixed
offset (local geology). We model every reading as `ML = μ_event + S_station + noise` and solve for the
event magnitudes μ and station terms S. **The assumption we are testing:** S is *constant in time*.

A station term that secretly *changes in time* (because Layer A's response was imperfect) would put a
**step** into the residuals — and bias the magnitudes of whichever events that station recorded.""")

# ----------------------------------------------------------------- §2 residual steps
md(r"""## 2 · The symptom — residual magnitude *steps*

After the time-invariant correction, each station's residual should be a flat zero band in time. Three
high-count anchors show clear **steps** instead — the time-invariant model is incomplete for them.""")
co(r"""rng=np.random.default_rng(0)
def _best(tn,rv,lo,hi,ms):
    n=hi-lo
    if n<2*ms: return None
    sub=rv[lo:hi]; subt=tn[lo:hi]; best=None; grid=max(1,(n-2*ms)//250 or 1)
    for k in range(ms,n-ms,grid):
        dd=abs(np.median(sub[:k])-np.median(sub[k:]))
        if best is None or dd>best[1]: best=(k,dd)
    k,_=best; obs=abs(np.median(sub[k:])-np.median(sub[:k])); c=0
    for _ in range(300):
        p=rng.permutation(sub); c+=abs(np.median(p[:k])-np.median(p[k:]))>=obs
    return dict(kabs=lo+k,tcut=subt[k],step=float(np.median(sub[k:])-np.median(sub[:k])),p=(c+1)/301)
def find_cps(tn,rv,ms=300,min_step=0.08,maxcp=3):
    o=np.argsort(tn); tn=tn[o]; rv=rv[o]; segs=[(0,len(rv))]; cps=[]
    for _ in range(maxcp):
        best=None
        for si,(lo,hi) in enumerate(segs):
            bs=_best(tn,rv,lo,hi,ms)
            if bs and bs["p"]<0.01 and abs(bs["step"])>=min_step and (best is None or abs(bs["step"])>abs(best[0]["step"])): best=(bs,si)
        if best is None: break
        bs,si=best; lo,hi=segs.pop(si); cps.append(bs); segs+=[(lo,bs["kabs"]),(bs["kabs"],hi)]
    return sorted(cps,key=lambda x:x["tcut"])
# detect on all well-sampled station-channels
DET={}
for s in d.sc.value_counts().index:
    sub=d[d.sc==s]
    if len(sub)<700: continue
    cps=find_cps(sub.tnum.values,sub.res.values)
    if cps: DET[s]=[(T0+pd.Timedelta(days=cp["tcut"]*365.25)).date() for cp in cps], [round(cp["step"],2) for cp in cps]

show=["KG.HDB.HHZ","KG.MKL.HHZ","KG.YSB.HHZ"]
fig,axes=plt.subplots(1,3,figsize=(16,4.2),sharey=True)
for ax,s in zip(axes,show):
    sub=d[d.sc==s].sort_values("t")
    ax.scatter(sub.t,sub.res,s=3,alpha=0.12,color="0.6")
    q=sub.set_index("t").res.rolling("180D",min_periods=20).median()
    ax.plot(q.index,q.values,color="k",lw=1.5,label="180-day median residual")
    ax.axhline(0,color="0.4",lw=0.8,ls="--")
    if s in DET:
        for dt,st in zip(*DET[s]): ax.axvline(pd.Timestamp(dt,tz="UTC"),color="tab:red",lw=1.4)
    for nm,tt in MAINSHOCKS.items(): ax.axvline(tt,color="tab:blue",lw=0.8,ls=":")
    ax.set_title(f"{s}",fontsize=11); ax.set_ylim(-1,1); ax.set_xlabel("Year")
axes[0].set_ylabel("Residual ML (after nb09 correction)"); axes[0].legend(fontsize=8,loc="lower left")
fig.suptitle("Residual steps (red = detected change; blue dotted = Gyeongju/Pohang mainshocks)",y=1.02)
fig.tight_layout(); plt.show()
print("Detected changepoints (date, step in ML):")
for s,(dts,sts) in DET.items(): print(f"  {s}: "+", ".join(f"{x} ({y:+.2f})" for x,y in zip(dts,sts)))""")

# ----------------------------------------------------------------- §3 what the XML says
md(r"""## 3 · What the StationXML says — gain vs response-shape changes

For each station we read its documented **response epochs**. Two kinds of change matter:
- **Gain (scalar sensitivity) change** — e.g. a 4× jump. The deconvolution uses the time-appropriate
  response, so this is **already corrected** → it should *not* leave a residual step.
- **Response-shape change** (poles/zeros differ at the same gain) — a sensor/response swap. If the
  metadata for it is imperfect, the deconvolution leaves a **residual step**.""")
co(r"""inv=obspy.read_inventory(XML)
def fingerprint(c):
    try:
        s=c.response.instrument_sensitivity.value; npz=len(c.response.get_paz().poles); return s,npz
    except Exception: return np.nan,np.nan
rows=[]
for net in inv:
    for sta in net:
        chans=defaultdict(list)
        for c in sta.channels: chans[c.code].append(c)
        for cc,cl in chans.items():
            sc=f"{net.code}.{sta.code}.{cc}"
            if sc not in set(d.sc.unique()): continue
            cl=sorted(cl,key=lambda x:x.start_date); prev=None
            for c in cl:
                s,npz=fingerprint(c)
                if prev is not None:
                    gain_ch=abs(np.log10(s)-np.log10(prev[0]))>0.05 if (s and prev[0]) else False
                    shape_ch=(npz!=prev[1])
                    if gain_ch or shape_ch:
                        rows.append(dict(sc=sc,date=c.start_date.date,kind=("GAIN" if gain_ch else "")+("+SHAPE" if shape_ch else ""),
                                         sens=f"{prev[0]:.2e}->{s:.2e}",npoles=f"{prev[1]}->{npz}"))
                prev=(s,npz)
RESP=pd.DataFrame(rows)
print("Documented response changes for catalog station-channels (gain already corrected; shape may leave residual):")
print(RESP.to_string(index=False))""")

# ----------------------------------------------------------------- §4 alignment verdict
md(r"""## 4 · The verdict on each detected step — real or spurious?

Match each **detected residual step** to the nearest **documented response change** (±90 days). A step
on a response change is a **real magnitude inhomogeneity**; a step on nothing (or on the Pohang
mainshock) is **spurious** (the changing earthquake population, via the station-term↔distance
degeneracy nb09 noted).""")
co(r"""def classify(sc,dt):
    cand=RESP[RESP.sc==sc]
    if len(cand):
        dd=(pd.to_datetime(cand.date)-pd.Timestamp(dt)).abs().dt.days
        if dd.min()<=90:
            r=cand.iloc[int(dd.values.argmin())]; return f"REAL — {r.kind} response change {r.date}"
    for nm,tt in MAINSHOCKS.items():
        if abs((pd.Timestamp(dt,tz='UTC')-tt).days)<=45: return f"SPURIOUS — {nm} mainshock (population)"
    return "SPURIOUS — no documented change"
ver=[]
for s,(dts,sts) in DET.items():
    for dt,st in zip(dts,sts): ver.append(dict(sc=s,date=dt,step=st,verdict=classify(s,dt)))
VER=pd.DataFrame(ver).sort_values("verdict")
print(VER.to_string(index=False))
nreal=VER.verdict.str.startswith("REAL").sum()
print(f"\n{nreal} of {len(VER)} detected steps align with a documented response change (REAL); the rest are spurious.")""")

# ----------------------------------------------------------------- §5 impact
md(r"""## 5 · Why it matters — impact on the UF background-rate conclusion

Now build a **response-aware** catalog: give each documented-response epoch its **own** station term
(split only at real changes), re-homogenise, and re-run the nb13/14 pipeline (Mc, b, and the
Zaliapin–Ben-Zion **declustered background** rate, pre 2010-2013 vs post 2019-2024).""")
co(r"""sys.path.insert(0,"/home/msseo/works/16.kma_absolute_location")
from kma_absolute_location import nnd
from seismostats.analysis import estimate_mc_maxc, ClassicBValueEstimator
from seismostats.utils import bin_to_precision
# epoch boundaries = documented response-change dates (from §3)
bnd=defaultdict(list)
for _,r in RESP.iterrows(): bnd[r.sc].append(pd.Timestamp(r.date).date())
d["um"]=d.apply(lambda r: f"{r.sc}@e{sum(r.t.date()>=b for b in bnd[r.sc])}" if r.sc in bnd else r.sc,axis=1)
uc=d.um.value_counts(); d["um"]=d.um.where(~d.um.isin(set(uc[uc<50].index)),d.sc)
mu_c,_=median_polish(d,"um")
ev=pd.DataFrame({"t":d.groupby("event_idx").t.first(),"inv":mu0,"corr":mu_c.reindex(mu0.index)}).dropna()

def mc_b(mm):
    m=bin_to_precision(np.sort(np.asarray(mm,float)),DM); mc=float(estimate_mc_maxc(m,fmd_bin=DM)[0])
    be=ClassicBValueEstimator(); be.calculate(m[m>=mc],mc=mc,delta_m=DM); return mc,be.b_value
clean=pd.read_csv("catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo_homogenised_clean.csv")
clean["time"]=pd.to_datetime(clean.time,utc=True,errors="coerce")
clean=clean[(clean.lon>=UF[0])&(clean.lon<=UF[1])&(clean.lat>=UF[2])&(clean.lat<=UF[3])].dropna(subset=["time","lat","lon"]).sort_values("time")
m=pd.merge_asof(clean,ev.reset_index().sort_values("t"),left_on="time",right_on="t",tolerance=pd.Timedelta("3s"),direction="nearest").dropna(subset=["corr"])
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
res={lab:nnd_bg(c) for c,lab in [("inv","time-invariant (nb09)"),("corr","response-aware")]}
print("WHOLE-CATALOG:")
for c,l in [("inv","time-invariant"),("corr","response-aware")]:
    mc,b=mc_b(ev[c]); print(f"  {l:18}: Mc={mc:.2f} b={b:.2f}")
print("\nUF declustered BACKGROUND rate (events/yr), pre 2010-2013 vs post 2019-2024:")
for l,o in res.items():
    for cut in (0.8,1.0): print(f"  {l:22} M>={cut}: pre {o[cut][0]:.1f} -> post {o[cut][1]:.1f}  ratio {o[cut][2]:.2f}")

fig,ax=plt.subplots(figsize=(7.5,4.4))
labs=["time-invariant\n(nb09 / nb13-14)","response-aware\n(this notebook)"]; x=np.arange(2); w=0.35
r08=[res["time-invariant (nb09)"][0.8][2],res["response-aware"][0.8][2]]
r10=[res["time-invariant (nb09)"][1.0][2],res["response-aware"][1.0][2]]
ax.bar(x-w/2,r08,w,color="tab:blue",label="M≥0.8"); ax.bar(x+w/2,r10,w,color="tab:orange",label="M≥1.0")
ax.axhline(1.0,color="0.5",ls="--",lw=1); ax.set_xticks(x); ax.set_xticklabels(labs)
ax.set(ylabel="post-2019 / pre-2013 background ratio",title="Background-rate ratio: steady (≈1) vs increase")
for i,(a,b_) in enumerate(zip(r08,r10)): ax.text(i-w/2,a+0.03,f"{a:.2f}",ha="center"); ax.text(i+w/2,b_+0.03,f"{b_:.2f}",ha="center")
ax.legend(); fig.tight_layout(); plt.show()""")

# ----------------------------------------------------------------- §6 verdict
md(r"""## 6 · Plain-language verdict and the decision

**What we now know for certain**
- The nb09 time-invariant station correction is **incomplete**: a few KG anchor stations (HDB, MKL,
  YSB) have **real residual magnitude steps** that sit on **documented response-shape changes** in the
  StationXML. The big scalar *gain* jumps were already handled; the *shape* changes were not.
- A few other detected steps are **spurious** (no documented change, or the Pohang mainshock).
- Correcting the real ones lowers early-era ML by ~0.03 and **raises the UF declustered background
  ratio from ~1.2 to ~1.7 (M≥0.8) / ~1.5 to ~2.2 (M≥1.0)** — while the UF **b-value stays stable**,
  so it is a genuine magnitude correction near the cutoff, not a fitting artifact.

**What this means:** the earlier "**background steady**" headline (nb13/14) is **not robust** to a
defensible response correction — the UF post-2019 background may have **genuinely increased ~1.7×**.

**What is still uncertain (why we can't just declare it yet)**
- The response metadata looks partly **inconsistent** (pole counts oscillate 6→11→6→11→13), so some
  "shape changes" may be *metadata-encoding* artifacts rather than physical sensor swaps.
- The **early-era sample is tiny** (~10-50 events), so the ratios have wide error bars.
- The result is **sensitive to the cutoff** (a 0.03 ML shift against a sharp threshold moves rates a lot).

**The decision — three options (in increasing rigour):**
1. **Quick / defensible-now:** report nb13/14 on the time-invariant catalog **with this as a stated
   caveat** (background ratio 1.2–1.7 depending on response treatment); flag HDB/MKL/YSB epochs.
2. **Metadata fix (recommended):** verify the HDB-2015/2019, YSB-2016, MKL-2015 response changes against
   KMA/NECIS instrument logs; **re-deconvolve ML with corrected responses** for those epochs (the root
   fix), then re-run nb13/14. This resolves real-vs-encoding cleanly.
3. **Full:** option 2 **plus** bootstrap CIs on the small early sample **plus** re-run on the dt.cc
   relocations once they finish — the publication-grade version.""")
co(r"""print("="*72); print("SYNTHESIS — magnitude time-dependence".center(72)); print("="*72)
print(f"readings {len(d):,} | station-channels {d.sc.nunique()} | documented response changes {len(RESP)}")
print(f"detected residual steps: {len(VER)} ({VER.verdict.str.startswith('REAL').sum()} REAL on response changes, "
      f"{VER.verdict.str.startswith('SPURIOUS').sum()} spurious)")
mci,bi=mc_b(ev["inv"]); mcc,bc=mc_b(ev["corr"])
print(f"whole-catalog Mc/b:  invariant {mci:.2f}/{bi:.2f}  ->  response-aware {mcc:.2f}/{bc:.2f}")
print("UF declustered background ratio (post-2019/pre-2013):")
for cut in (0.8,1.0):
    print(f"   M>={cut}: time-invariant {res['time-invariant (nb09)'][cut][2]:.2f}  ->  response-aware {res['response-aware'][cut][2]:.2f}")
print("\nHEADLINE: 'background steady' is NOT robust to a defensible response correction — the UF post-2019")
print("background may be genuinely higher (~1.7x). Resolve via option 2 (re-deconvolve affected epochs) before publishing.")""")

nb.cells=C
out="16.Magnitude_timedependence_synthesis.ipynb"; nbf.write(nb,out); print("wrote",out,len(C),"cells")
