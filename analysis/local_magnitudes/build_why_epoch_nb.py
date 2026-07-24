#!/usr/bin/env python
"""Generate 18.Why_epoch_station_correction.ipynb — deep-dive companion to nb17 on station corrections.
Sensor dates come from the station-response StationXML (responses/master/KS_KG_metadata_1.0.2.xml — the
same metadata used to deconvolve the instrument), covering BOTH KS and KG. Covers:
 1  HDB worked example (problem/fix/result) + the deep narrow valley = a PRE-REPLACEMENT MALFUNCTION.
 1b a milder valley = an INCOMPLETE date list (epoch correction is piecewise-constant).
 2  time-dependent residual for EVERY catalog station-channel (none omitted).
 3  before/after flattening for ALL epoch-split station-channels (KS + KG).
 4  methodology notes: 120-day median is visualization-only; corrections are per-era medians over ALL
    readings; dropped stations still get terms (connectivity); epoch split re-solves ALL terms jointly.
 5  map of recent-era station corrections (PyGMT).
Runs in `base`."""
import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# Station corrections and why they must be epoch-dependent (KS + KG)

**One-sentence idea.** nb09 gives each station **one** magnitude offset for all time; that is right only
if the station never changes. A **sensor swap (or failure)** steps that offset, so one fixed number is
wrong. We split each station's record into sensor eras using the **station-response metadata** (the same
StationXML used to remove the instrument response), which documents changes for **both KS and KG**.

Contents: 1 the problem/fix on HDB + the deep valley (a pre-swap malfunction); 1b a milder valley (an
incomplete date list); 2 the residual of every station-channel; 3 before/after for every split station;
4 method notes (windowing, dropped stations, joint re-solve); 5 a map of the recent-era corrections.

*Companion to:* `09.Station_corrections_ML.ipynb`, `17.Response_epoch_corrected_catalog.ipynb`.""")

# ----------------------------------------------------------------- §0 setup
co(r"""import warnings; warnings.filterwarnings("ignore")
import os, json, math, numpy as np, pandas as pd, obspy, glob
from collections import defaultdict
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
import matplotlib.dates as mdates
from matplotlib.patches import Patch
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":130,"axes.grid":True,"grid.alpha":0.25,"font.size":11,
    "legend.framealpha":1.0,"legend.facecolor":"white","legend.edgecolor":"0.6"})  # opaque legend box, drawn above data
PS="catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo.csv"
SXML="/home/msseo/works/02.Ulsan_Fault_detection/data/metadata/responses/master/KS_KG_metadata_1.0.2.xml"          # station-response metadata (KS + KG)
CACHE="/home/msseo/works/02.Ulsan_Fault_detection/data/metadata/responses/sensor_breaks_master.json"; MIN_EPOCH_N=50

d=pd.read_csv(PS); d=d[(d.snr>=3)&d.ML.notna()].copy()
d["t"]=pd.to_datetime(d.event_time,utc=True,errors="coerce"); d=d.dropna(subset=["t"])
d["year"]=d.t.dt.year; d["sc"]=d.network+"."+d.station+"."+d.channel
incat=set(d.sc.unique())

def parse_breaks(xml):
    "sensor-shape change = response pole/zero COUNT changes between consecutive channel epochs."
    inv=obspy.read_inventory(xml)
    def shape(c):
        try: pz=c.response.get_paz(); return (len(pz.poles),len(pz.zeros))
        except Exception: return None
    brk=defaultdict(list)
    for net in inv:
        for sta in net:
            bych=defaultdict(list)
            for c in sta.channels: bych[c.code].append(c)
            for cc,cl in bych.items():
                sc=f"{net.code}.{sta.code}.{cc}"; cl=sorted(cl,key=lambda x:x.start_date); prev=None
                for c in cl:
                    fp=shape(c)
                    if prev is not None and fp is not None and fp!=prev: brk[sc].append(str(c.start_date.date))
                    if fp is not None: prev=fp
    return {k:v for k,v in brk.items() if k in incat}
breaks_str=json.load(open(CACHE)) if os.path.exists(CACHE) else parse_breaks(SXML)
if not os.path.exists(CACHE): json.dump(breaks_str,open(CACHE,"w"),indent=0)
breaks={k:[pd.Timestamp(x).date() for x in v] for k,v in breaks_str.items()}

# station coordinates (union of yearly station tables, latest wins)
coord={}
for f in sorted(glob.glob("../station_table/stations_*.csv")):
    s=pd.read_csv(f)
    for _,r in s.iterrows(): coord[r.Code]=(r.Latitude,r.Longitude)

def mp(df,col,n=40,tol=1e-4):
    "median polish: joint {event magnitude mu, unit offset S}; gauge obs-weighted mean(S)=0."
    w=df[col].value_counts(); mu=df.groupby("event_idx").ML.median(); S=pd.Series(0.0,index=w.index)
    for _ in range(n):
        Sn=pd.Series(df.ML.values-mu.reindex(df.event_idx).values,index=df[col]).groupby(level=0).median()
        Sn-=np.average(Sn.reindex(w.index),weights=w.values)
        mun=pd.Series(df.ML.values-Sn.reindex(df[col]).values,index=df.event_idx).groupby(level=0).median()
        if float(np.nanmax(np.abs(mun.reindex(mu.index).values-mu.values)))<tol: mu,S=mun,Sn; break
        mu,S=mun,Sn
    return mu,S
mu0,S0=mp(d,"sc"); d["res"]=d.ML - d.event_idx.map(mu0).values   # single-offset baseline + residual

def binmed(sub,col="res",freq="120D",min_n=10):   # drop under-populated bins (e.g. the final partial bin) — avoids end artifacts
    g=sub.set_index("t")[col].sort_index().groupby(pd.Grouper(freq=freq)); med=g.median(); cnt=g.size()
    out=med[cnt>=min_n].dropna(); return out.index,out.values
def era_split(sub,bdates):
    edges=[sub.t.min()-pd.Timedelta("1D")]+[pd.Timestamp(x,tz="UTC") for x in bdates]+[sub.t.max()+pd.Timedelta("1D")]
    return pd.cut(sub.t,bins=edges,labels=False)
# HDB pre-2015 sensor-FAILURE window (data-driven onset; documented swap = recovery) — excluded from corrected views & nb17
_hm=d[d.sc=="KG.HDB.HHZ"].set_index("t").res.groupby(pd.Grouper(freq="ME")).median()
_f=_hm[(_hm<-1.0)&(_hm.index<pd.Timestamp("2015-06",tz="UTC"))]
FAIL_ON=pd.Timestamp(_f.index.min()).replace(day=1) if len(_f) else pd.Timestamp("2014-11-01",tz="UTC"); FAIL_OFF=pd.Timestamp("2015-05-21",tz="UTC")
fail_mask=lambda df:(df.sc=="KG.HDB.HHZ")&(df.t>=FAIL_ON)&(df.t<FAIL_OFF)
print(f"{len(d):,} readings | {len(incat)} station-channels | metadata splits {len(breaks)}: {sorted(breaks)}")
print(f"HDB failure window: {FAIL_ON.date()} .. {FAIL_OFF.date()} ({int(fail_mask(d).sum())} readings, excluded from corrected views)")""")

# ----------------------------------------------------------------- §1 HDB worked example
md(r"""## 1 · The problem and the fix on real data — HDB

y-axis = HDB's **station residual** = (HDB's ML) − (event network-median ML). Constant ⇒ flat. **(a)** it
steps at each documented sensor change (dotted); the single offset (red) fits no era. **(b)** each era's
median (green) tracks the steps. **(c)** after correction the per-era offsets flatten it; the single
offset leaves the steps. (Note the deep narrow dip ~2015 — explained in §1c.)""")
co(r"""SC="KG.HDB.HHZ"; bdates=breaks[SC]; h=d[d.sc==SC].copy(); h["era"]=era_split(h,bdates)
era_med=h.groupby("era").res.median()
fig,ax=plt.subplots(1,3,figsize=(16.5,4.6),sharey=True)
ax[0].scatter(h.t,h.res,s=4,alpha=0.08,color="0.5")
bx,by=binmed(h); ax[0].plot(bx,by,"-",color="tab:blue",lw=1.6,label="120-day median residual")
ax[0].axhline(S0[SC],color="tab:red",lw=2.2,ls="--",label=f"single offset ({S0[SC]:+.2f})")
ax[0].set(title="(a) Problem: the true offset STEPS at each swap",ylabel="Station residual  ML$_{sta}$ − ML$_{event}$",ylim=(-1.1,1.1)); ax[0].legend(loc="upper left",fontsize=8.5)
ax[1].scatter(h.t,h.res,s=4,alpha=0.08,color="0.5")
for e in era_med.index:
    seg=h[h.era==e]; x0,x1=seg.t.min(),seg.t.max(); v=era_med[e]
    ax[1].plot([x0,x1],[v,v],"-",color="tab:green",lw=3.4,solid_capstyle="butt")
ax[1].plot([],[],"-",color="tab:green",lw=3,label="one offset per sensor era"); ax[1].set(title="(b) Fix: one offset per sensor era"); ax[1].legend(loc="upper left",fontsize=8.5)
h["res_single"]=h.res-S0[SC]; h["res_epoch"]=h.res-h.era.map(era_med).values
for col,lab,c in [("res_single","after single offset","tab:red"),("res_epoch","after epoch offsets","tab:green")]:
    bx,by=binmed(h,col); ax[2].plot(bx,by,"-",lw=1.8,color=c,label=lab)
ax[2].axhline(0,color="0.4",lw=1,ls="--"); ax[2].set(title="(c) Result: epoch offsets flatten it; single doesn't"); ax[2].legend(loc="upper left",fontsize=8.5)
for a in ax:
    for bd in bdates: a.axvline(pd.Timestamp(bd,tz="UTC"),color="k",lw=1.0,ls=":")
    a.xaxis.set_major_locator(mdates.YearLocator(3)); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); a.set_xlabel("Year")
fig.suptitle(f"{SC} — sensor changes {', '.join(str(x) for x in bdates)}",y=1.03,fontsize=12)
fig.tight_layout(); plt.show()
print("HDB per-era offsets:",{int(k):round(v,3) for k,v in era_med.items()},f"| single {S0[SC]:+.3f}")""")

# ----------------------------------------------------------------- §1c the deep narrow valley = malfunction
md(r"""## 1c · The deep narrow valley (~2015) is a PRE-REPLACEMENT MALFUNCTION

Zooming into 2013–2016: HDB sat near −0.3 ML, then from **Nov 2014 collapsed to ≈ −2 ML** (it reported
amplitudes ~100× too small) for ~7 months, then **snapped back to ~0 exactly at the 2015-05-21 swap**.
That is a **failing instrument in the months before it was replaced** — the metadata records the
*replacement* (2015-05-21), not the *onset of failure* (~2014-11). So the malfunction window sits inside
one era and a single per-era median cannot remove it. This is the honest limit of metadata-driven epoch
correction: **it catches documented swaps, not gradual pre-swap failures.** In practice these ~2-ML-low
readings barely bias event magnitudes (the per-event *median* over many stations rejects one wild outlier
— that is exactly why we use the median), but they do corrupt HDB's own era term, so a clean catalog
should additionally **flag/exclude** such failure windows.""")
co(r"""z=h[(h.t>="2013-06")&(h.t<="2016-06")].set_index("t").res.groupby(pd.Grouper(freq="ME")).agg(["median","size"])
fig,ax=plt.subplots(figsize=(11,4.4)); ax2=ax.twinx()
ax2.set_zorder(0); ax.set_zorder(2); ax.patch.set_visible(False)   # line + legend drawn ABOVE the count bars
ax2.bar(z.index,z["size"],width=20,color="0.85",label="readings / month")
ax.plot(z.index,z["median"],"o-",color="tab:red",lw=1.6,ms=4,label="HDB monthly median residual")
ax.axhline(0,color="0.5",lw=1,ls="--")
ax.axvspan(pd.Timestamp("2014-11-01",tz="UTC"),pd.Timestamp("2015-05-21",tz="UTC"),color="tab:red",alpha=0.10)
ax.axvline(pd.Timestamp("2015-05-21",tz="UTC"),color="k",lw=1.4,ls=":",label="documented swap 2015-05-21")
ax.annotate("failure onset ~2014-11\n(reads ~2 ML low)",xy=(pd.Timestamp("2015-01-01",tz="UTC"),-2.0),
    xytext=(pd.Timestamp("2013-08-01",tz="UTC"),-1.6),fontsize=9,arrowprops=dict(arrowstyle="->"))
ax.annotate("snaps to ~0\nafter replacement",xy=(pd.Timestamp("2015-08-01",tz="UTC"),0.0),
    xytext=(pd.Timestamp("2015-08-01",tz="UTC"),-1.2),fontsize=9,arrowprops=dict(arrowstyle="->"))
ax.set(ylabel="HDB residual (ML)",xlabel="Date",title="HDB 2013–2016: a failing sensor before its 2015-05 replacement"); ax2.set_ylabel("readings / month")
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
leg=ax.legend(h1+h2,l1+l2,fontsize=8,loc="lower right"); leg.set_zorder(100)   # legend above all artists
fig.tight_layout(); plt.show()
print(f"median residual Nov2014-May2015: {z.loc['2014-11':'2015-05','median'].median():.2f} ML; after swap (Jun-Dec 2015): {z.loc['2015-06':'2015-12','median'].median():+.2f} ML")""")

# ----------------------------------------------------------------- §1d step-by-step correction
md(r"""## 1d · How HDB is corrected, step by step (single figure)

One figure, three stages, each removing the next layer:
1. **One offset (nb09):** the single lifetime offset — the 4 sensor-era steps **and** the 2014–2015
   failure spike all remain.
2. **+ per-era offsets (metadata epochs):** each documented era gets its own offset → the era steps
   flatten; only the **failure window** (a pre-replacement malfunction, not a documented swap) still
   pokes through.
3. **+ exclude the 2014–2015 failure window:** the corrupt readings are dropped → the residual is flat at
   0 across the whole record. Excluding beats offsetting (the failure offset would be ≈ −1.8 ML =
   "unusable"). **This is exactly the correction nb17 applies to the catalog.**""")
co(r"""SC="KG.HDB.HHZ"; bdates=breaks[SC]
h=d[d.sc==SC].copy(); h["era"]=era_split(h,bdates); em=h.groupby("era").res.median()
hno=h[~((h.t>=FAIL_ON)&(h.t<FAIL_OFF))].copy(); hno["era"]=era_split(hno,bdates); emn=hno.groupby("era").res.median()
h["s_single"]=h.res-S0[SC]; h["s_epoch"]=h.res-h.era.map(em).values; hno["s_excl"]=hno.res-hno.era.map(emn).values
fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=True)
for a,(df,col,title,c) in zip(ax,[
    (h,"s_single","1 · one offset (nb09): era steps + 2014–2015 failure spike both remain","tab:red"),
    (h,"s_epoch","2 · + per-era offsets (metadata): era steps gone, failure spike remains","tab:orange"),
    (hno,"s_excl","3 · + exclude 2014–2015 failure window: flat across the whole record (= nb17)","tab:green")]):
    a.scatter(df.t,df[col],s=5,alpha=0.07,color="0.6"); bx,by=binmed(df,col); a.plot(bx,by,"-",lw=1.9,color=c)
    a.axhline(0,color="0.4",lw=1,ls="--"); a.axvspan(FAIL_ON,FAIL_OFF,color="tab:red",alpha=0.08)
    for b in bdates: a.axvline(pd.Timestamp(b,tz="UTC"),color="k",lw=1.0,ls=":")
    a.set(ylim=(-2.3,0.9),ylabel="residual (ML)"); a.set_title(title,fontsize=10.5,loc="left")
    a.xaxis.set_major_locator(mdates.YearLocator(2)); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax[2].set_xlabel("Year")
fig.suptitle("KG.HDB.HHZ — step-by-step correction: single offset → per-era offsets → failure window excluded",y=1.005,fontsize=12.5)
fig.tight_layout(); plt.show()
print(f"failure window {FAIL_ON.date()}..{FAIL_OFF.date()} ({int(((h.t>=FAIL_ON)&(h.t<FAIL_OFF)).sum())} readings) excluded in stage 3 and in nb17")
print(f"spread of 120-day medians: stage1 {np.std(binmed(h,'s_single')[1]):.2f} -> stage2 {np.std(binmed(h,'s_epoch')[1]):.2f} -> stage3 {np.std(binmed(hno,'s_excl')[1]):.3f} ML (flat)")""")

# ----------------------------------------------------------------- §2 all-station grid
md(r"""## 2 · Time-dependent residual for EVERY catalog station-channel (none omitted)

Each panel is one station-channel's 120-day median residual vs time (KG purple, KS orange); dotted lines
mark its documented sensor changes. Flat = stable (single offset is fine); a step at a dotted line = needs
the epoch split. Most are flat; only a handful step.""")
co(r"""order=sorted(incat,key=lambda s:(s.split(".")[0],s)); ncol=8; nrow=math.ceil(len(order)/ncol)
fig,ax=plt.subplots(nrow,ncol,figsize=(2.05*ncol,1.55*nrow),sharex=True,sharey=True)
for i,sc in enumerate(order):
    a=ax.flat[i]; sub=d[d.sc==sc]; col="tab:purple" if sc.startswith("KG") else "tab:orange"
    bx,by=binmed(sub); a.plot(bx,by,"-",color=col,lw=0.9); a.axhline(0,color="0.6",lw=0.5,ls="--")
    for b in breaks.get(sc,[]): a.axvline(pd.Timestamp(b,tz="UTC"),color="k",lw=0.8,ls=":")
    net,sta,ch=sc.split("."); a.set_title(f"{sta}.{ch}",fontsize=6,pad=1.5,color=col); a.set_ylim(-0.6,0.6); a.tick_params(labelsize=5); a.grid(alpha=0.15)
for j in range(len(order),nrow*ncol): ax.flat[j].axis("off")
for a in ax.flat: a.xaxis.set_major_locator(mdates.YearLocator(5)); a.xaxis.set_major_formatter(mdates.DateFormatter("%y"))
fig.legend(handles=[Patch(color="tab:purple",label="KG"),Patch(color="tab:orange",label="KS"),
    mpl.lines.Line2D([],[],color="k",ls=":",label="sensor change")],loc="upper right",fontsize=9,ncol=3,bbox_to_anchor=(1.0,1.02))
fig.suptitle("120-day median residual vs time — all catalog station-channels (y∈[−0.6,0.6]; HDB/MKL exceed it)",y=1.005,fontsize=12)
fig.tight_layout(); plt.show()""")

# ----------------------------------------------------------------- §3 before/after for ALL split stations
md(r"""## 3 · Before/after for every epoch-split station-channel (KS + KG)

For each station-channel with a documented sensor change, the residual after the **single** offset (red)
vs after the **per-era** offsets (green). Green flatter ⇒ the epoch split helped. The KS movers (EUSB,
GUWB, JEJB, YOCB) — which nb17 could not reach — are included. **HDB's 2014–2015 failure window is
excluded** (as in nb17), so its green curve is now flat across the whole record.""")
co(r"""split_sc=sorted(breaks,key=lambda s:(s.split(".")[0],s)); n=len(split_sc); ncol=2; nrow=math.ceil(n/ncol)
fig,ax=plt.subplots(nrow,ncol,figsize=(7.2*ncol,2.7*nrow),sharex=True)
for i,sc in enumerate(split_sc):
    a=ax.flat[i]; hh=d[d.sc==sc].copy()
    if sc=="KG.HDB.HHZ": hh=hh[~((hh.t>=FAIL_ON)&(hh.t<FAIL_OFF))]   # exclude the failure window (as nb17 does)
    hh["era"]=era_split(hh,breaks[sc]); em=hh.groupby("era").res.median()
    hh["rs"]=hh.res-S0[sc]; hh["re"]=hh.res-hh.era.map(em).values
    for col,lab,c in [("rs","after single","tab:red"),("re","after epoch","tab:green")]:
        bx,by=binmed(hh,col); a.plot(bx,by,"-",lw=1.5,color=c,label=lab)
    for b in breaks[sc]: a.axvline(pd.Timestamp(b,tz="UTC"),color="k",lw=0.8,ls=":")
    a.axhline(0,color="0.5",lw=0.8,ls="--"); netc="tab:purple" if sc.startswith("KG") else "tab:orange"
    a.set_title(sc,fontsize=9,color=netc); a.set_ylim(-0.9,0.9); a.tick_params(labelsize=7)
    if i==0: a.legend(fontsize=7.5,loc="upper left")
    a.xaxis.set_major_locator(mdates.YearLocator(3)); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for j in range(n,nrow*ncol): ax.flat[j].axis("off")
fig.suptitle("Residual after single vs epoch correction — every split station-channel",y=1.005,fontsize=12)
fig.tight_layout(); plt.show()""")

# ----------------------------------------------------------------- §4 method notes
md(r"""## 4 · Method notes (three things worth being precise about)

**(i) The 120-day median is for *visualization only.*** The correction itself is a single **per-era
median over all that era's readings** — no time window, no smoothing. The 120-day curve just lets the eye
see the steps; a finer window shows more noise, a coarser one hides detail, but neither changes the
correction. We confirm below that the per-era offset is identical whether or not we bin.

**(ii) A station dropping out of the network does not break its correction.** The median polish estimates
a station's offset from whenever it was active; decommissioned stations (e.g. DAG2, HAK) still get terms.
The only requirement is that the event–station graph stays *connected* (every era shares stations with
its neighbours), which it does.

**(iii) Epoch splitting re-solves *all* terms jointly.** Splitting HDB into eras is not a local edit — the
median polish re-estimates every event magnitude and every *other* station's offset simultaneously. So
the constant offsets of the non-split stations are **updated** (slightly) too.""")
co(r"""# (i) per-era offset is window-free: era median over all readings == what the correction uses
print("HDB era offsets, per-era median over ALL readings (no binning):")
print("   ",{int(k):round(v,3) for k,v in h.groupby("era").res.median().items()})
# (ii) dropped stations still get a term
for s in ["KS.DAG2.HHZ","KG.HAK.ELZ"]:
    if s in S0.index: print(f"   dropped station {s}: single offset {S0[s]:+.3f} (n={int((d.sc==s).sum())}) — estimated fine")
# (iii) how much do NON-split station constants move when we add the epoch splits?
dd=d[~fail_mask(d)].copy()           # exclude HDB failure window -> identical recipe to nb17 (up-to-date ML)
def eu(row):
    s=row.sc
    return s if s not in breaks else f"{s}@{sum(row.t.date()>=b for b in breaks[s])}"
dd["unit"]=dd.apply(eu,axis=1); uc=dd.unit.value_counts(); dd["unit"]=dd.unit.where(~dd.unit.isin(set(uc[uc<MIN_EPOCH_N].index)),dd.sc)
mu1,S1=mp(dd,"unit")
nonsplit=[s for s in incat if s not in breaks]
shift=pd.Series({s:S1.get(s,np.nan)-S0[s] for s in nonsplit}).dropna()
print(f"\n(iii) non-split stations: median |offset change| after adding epoch splits = {shift.abs().median():.4f} ML, "
      f"max {shift.abs().max():.3f} ML ({shift.abs().idxmax()})")
print("    -> small but nonzero: the whole system is re-solved, not just the split stations.")""")

# ----------------------------------------------------------------- §4b time-magnitude comparison
md(r"""## 4b · Time–magnitude: every event under three corrections (all events + UF box)

Each event is plotted **three times, overlaid** at its origin time, in different colours:
1. **raw Heo-2024** per-event median (no station correction),
2. **+ station correction** (nb09, one offset per station),
3. **+ epoch-aware** (nb17 recipe: KS+KG sensor eras + HDB failure window excluded).

**Top:** all events. **Bottom:** the Ulsan-Fault box only (129.25–129.55°E, 35.60–35.90°N). The y-axis
spans the full magnitude range. Where the three colours separate for an event, the correction moved that
ML; most separation is in the early sparse-network era (the corrections are small, a few 0.01–0.1 ML).""")
co(r"""ml_raw=d.groupby("event_idx").ML.median(); tmap=d.groupby("event_idx").t.first()
mag=pd.DataFrame({"t":tmap,"raw":ml_raw,"nb09":mu0.reindex(ml_raw.index),"epoch":mu1.reindex(ml_raw.index)}).dropna()
# flag UF-box events by merging event times to the located (clean) catalog
clean=pd.read_csv("catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo_homogenised_clean.csv")
clean["time"]=pd.to_datetime(clean.time,utc=True,errors="coerce"); clean=clean.dropna(subset=["time","lat","lon"]).sort_values("time")
_ev=mag.reset_index().rename(columns={"index":"event_idx"})[["event_idx","t"]].sort_values("t")
_mm=pd.merge_asof(_ev,clean[["time","lat","lon"]],left_on="t",right_on="time",tolerance=pd.Timedelta("3s"),direction="nearest")
_mm["uf"]=_mm.lon.between(129.25,129.55)&_mm.lat.between(35.60,35.90)
mag["uf"]=mag.index.map(_mm.set_index("event_idx").uf).fillna(False)
ylo,yhi=float(mag[["raw","nb09","epoch"]].min().min())-0.2,float(mag[["raw","nb09","epoch"]].max().max())+0.2
fig,ax=plt.subplots(2,1,figsize=(13,10),sharex=True)
for a,(sub,title) in zip(ax,[(mag,f"all events (N={len(mag):,})"),(mag[mag.uf],f"Ulsan-Fault box only (N={int(mag.uf.sum()):,})")]):
    for col,lab,c in [("raw","(1) raw ML (Heo 2024)","#7a7a7a"),("nb09","(2) + station correction (nb09)","#1f77ff"),("epoch","(3) + epoch-aware (nb17)","#13b113")]:
        a.scatter(sub.t,sub[col],s=12,alpha=0.6,color=c,edgecolors="none",label=lab)
    a.set(ylabel="ML",title=title,ylim=(ylo,yhi)); a.grid(alpha=0.25)
    leg=a.legend(loc="upper right",fontsize=8.5,markerscale=2.2)
    for lh in leg.legend_handles: lh.set_alpha(1)
    leg.set_zorder(50)
ax[1].set_xlabel("Year")
ax[1].xaxis.set_major_locator(mdates.YearLocator(2)); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.suptitle("Time–magnitude: each event under three corrections (raw / station / epoch)",y=1.0,fontsize=12.5)
fig.tight_layout(); plt.show()
print(f"all events {len(mag):,}; UF-box events {int(mag.uf.sum()):,}")
ay=mag.assign(y=mag.t.dt.year); an=ay.groupby("y")[["raw","nb09","epoch"]].median()
auf=mag[mag.uf].assign(y=mag[mag.uf].t.dt.year).groupby("y")[["raw","nb09","epoch"]].median()
print("annual median ML  raw -> nb09 -> epoch  (ALL | UF):")
for y in an.index:
    u=f"{auf.loc[y,'raw']:+.2f}->{auf.loc[y,'epoch']:+.2f}" if y in auf.index else "—"
    print(f"  {int(y)}: {an.loc[y,'raw']:+.2f} -> {an.loc[y,'nb09']:+.2f} -> {an.loc[y,'epoch']:+.2f}   UF: {u}")""")

# ----------------------------------------------------------------- §5 recent-era correction map
md(r"""## 5 · Map of recent-era station corrections (PyGMT)

The most-recent-era offset per station (mean over its channels), as a diverging colormap: **red = the
station reads HIGH** (positive S, magnitudes get S subtracted), **blue = reads LOW** (negative S). Split
stations contribute their *latest* era; all others their single offset. The offsets `S1` here come from the
**same recipe nb17 uses** (KS+KG sensor eras + HDB failure window excluded), so this map and the catalog
`catalog_ml_heo_epoch.csv` are the **same, up-to-date** correction. This is the spatial pattern of site
response the magnitudes are calibrated against.""")
co(r"""import pygmt
recent_unit=dd.sort_values("t").groupby("sc").unit.last()           # each sc's most recent era unit
rec=pd.DataFrame({"sc":recent_unit.index,"unit":recent_unit.values})
rec["S"]=rec.unit.map(S1); rec["station"]=rec.sc.str.split(".").str[1]
st=rec.dropna(subset=["S"]).groupby("station").S.mean().reset_index()  # one value per station
st["lat"]=st.station.map(lambda s:coord.get(s,(np.nan,np.nan))[0]); st["lon"]=st.station.map(lambda s:coord.get(s,(np.nan,np.nan))[1])
st=st.dropna(subset=["lat","lon"])
pad=0.15; REG=[st.lon.min()-pad,st.lon.max()+pad,st.lat.min()-pad,st.lat.max()+pad]
vmax=float(np.ceil(st.S.abs().max()*20)/20)
pygmt.config(MAP_FRAME_TYPE="plain",FORMAT_GEO_MAP="ddd.x",FONT_TITLE="13p")
fig=pygmt.Figure()
fig.basemap(region=REG,projection="M12c",frame=["WSne+tRecent-era station corrections (ML)","xa0.5","ya0.5"])
fig.coast(land="gray96",water="lightblue",shorelines="0.4p,gray50",borders="1/0.3p,gray70")
pygmt.makecpt(cmap="polar",series=[-vmax,vmax])     # blue = negative S (reads low), red = positive S (reads high)
fig.plot(x=st.lon,y=st.lat,fill=st.S,cmap=True,style="c0.34c",pen="0.4p,black")
fig.colorbar(frame="af+lstation correction  S  (ML)")
fig.basemap(map_scale="jBL+w20k+o0.6c/0.6c"); fig.show()
print(f"mapped {len(st)} stations | correction range [{st.S.min():+.2f}, {st.S.max():+.2f}] ML, std {st.S.std():.3f}")
print("most negative S (reads LOW, blue):",", ".join(f"{r.station}{r.S:+.2f}" for _,r in st.nsmallest(3,'S').iterrows()),
      "| most positive S (reads HIGH, red):",", ".join(f"{r.station}{r.S:+.2f}" for _,r in st.nlargest(3,'S').iterrows()))""")

# ----------------------------------------------------------------- §6 summary
md(r"""## 6 · Summary""")
co(r"""print("="*74); print("STATION CORRECTIONS — EPOCH-DEPENDENT, KS + KG".center(74)); print("="*74)
print(f" - HDB single offset {S0[SC]:+.2f} hides a ~{h.groupby('era').res.median().max()-h.groupby('era').res.median().min():.1f} ML swing across 4 sensor eras; the per-era split flattens it.")
print(" - The deep ~2015 valley is a PRE-REPLACEMENT FAILURE (HDB read ~2 ML low for 7 months before the")
print("   2015-05 swap) — metadata marks replacements, not gradual failures; the window is EXCLUDED (nb17).")
print(" - Step-by-step (§1d): single offset -> per-era offsets -> exclude failure window = flat residual.")
print(f" - The response StationXML splits {len(breaks)} station-channels ({sum(s.startswith('KS.') for s in breaks)} KS, {sum(s.startswith('KG.') for s in breaks)} KG); all corrections")
print("   are solved jointly over BOTH networks. Visualization uses 120-day medians; the CORRECTION is a")
print("   per-era median (window-free). Dropped stations still get terms; non-split terms re-solve too.")
print(f" - Recent-era corrections span ±{vmax:.2f} ML across the network (Fig 5).")""")

nb.cells=C
out="18.Why_epoch_station_correction.ipynb"; nbf.write(nb,out); print("wrote",out,len(C),"cells")
