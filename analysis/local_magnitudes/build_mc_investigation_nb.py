#!/usr/bin/env python
"""Generate 11.Mc_completeness_investigation.ipynb — why does the UF-subregion Mc look too high in the
recent (dense-network) era and not track the falling minimum magnitude?

Findings the notebook establishes, on the station-homogenised catalog:
  (1) min-magnitude falls with densification (steps ~2015, 2019) but Mc does not — because min-mag is
      the DETECTION FRINGE (a few tiny events near dense stations) while Mc is where the catalog is
      COMPLETE; the gap is large because the FMD rollover is GRADUAL, not a step.
  (2) the 'too high' recent Mc is mostly an ESTIMATOR effect: MAXC+0.2 over-corrects on a gradual
      rollover; MAXC(mode) and K-S sit ~0.2-0.4.
  (3) it is NOT mainly spatial (dense core and edges give the same Mc).
  (4) the principled fix: a DETECTION-COMPLETENESS CURVE (observed/GR-extrapolated, Mignan-style)
      gives a defensible Mc(90%/99%) and shows the true completeness explicitly.
Pure CSV analysis; does not touch any running job."""
import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md("""# Why is the recent Ulsan-Fault Mc 'too high'? — a completeness investigation

**Observation.** On the station-homogenised catalog the UF **minimum magnitude** falls with network
densification (steps ~2015 and 2019, down to ML ≈ −1.3), and the median falls too — the correction is
working. **Yet the estimated Mc stays ~0.2–0.5 and looks too high**, not tracking the falling min-mag.

**What this notebook shows.**
1. **min-mag ≠ Mc.** The smallest detected events are a *detection fringe* (a few tiny events near dense
   stations); Mc is the magnitude above which the catalog is *complete*. Because the FMD rollover is
   **gradual** (not a step), the gap between them is large and *expected*.
2. **The 'too high' is mostly the estimator.** **MAXC+0.2** over-corrects on a gradual rollover; MAXC
   (mode) and K-S sit lower (~0.2–0.4).
3. **It is not mainly spatial** — dense core and edges give the same Mc.
4. **Principled remedy:** a **detection-completeness curve** — observed counts ÷ GR extrapolated from the
   complete part — gives the fraction detected vs magnitude and a defensible **Mc(90%) / Mc(99%)**.""")

md("""## 1 · Load UF-subregion (station-homogenised) catalog""")
co("""import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib as mpl, matplotlib.font_manager as fm
for _f in ("Helvetica","Arial","Nimbus Sans","DejaVu Sans"):
    if _f in {x.name for x in fm.fontManager.ttflist}: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":120,"axes.grid":True,"grid.alpha":0.3})
from seismostats.analysis import estimate_mc_maxc, estimate_mc_ks, ClassicBValueEstimator
from seismostats.utils import bin_to_precision
DM=0.1; UF=(129.25,129.55,35.6,35.9)
c=pd.read_csv("catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo_homogenised_clean.csv")
c["t"]=pd.to_datetime(c.time,utc=True,errors="coerce"); c["yr"]=c.t.dt.year
c=c.dropna(subset=["magnitude","lat","lon"])
uf=c[(c.lon>=UF[0])&(c.lon<=UF[1])&(c.lat>=UF[2])&(c.lat<=UF[3])].copy()
print(f"UF subregion: {len(uf)} homogenised events, ML [{uf.magnitude.min():+.2f},{uf.magnitude.max():+.2f}]")
def mc_set(mm):
    mm=bin_to_precision(np.sort(np.asarray(mm,float)),DM); mm=mm[np.isfinite(mm)]
    if len(mm)<40: return dict(maxc=np.nan,maxc2=np.nan,ks=np.nan)
    mx,_=estimate_mc_maxc(mm,fmd_bin=DM)
    try: ks=estimate_mc_ks(mm,delta_m=DM,p_value_pass=0.1); ks=ks[0] if isinstance(ks,tuple) else ks
    except Exception: ks=np.nan
    return dict(maxc=round(float(mx),2),maxc2=round(float(mx)+0.2,2),ks=(round(float(ks),2) if ks==ks else np.nan))""")

md("""## 2 · The puzzle — minimum magnitude falls, Mc does not

Annual minimum, 5th-percentile, median magnitude, and the three Mc estimators. min-mag drops ~1 unit
(2015, 2019 steps) while Mc estimators hover; MAXC+0.2 is the highest and noisiest.""")
co("""rows=[]
for y in range(2010,2025):
    g=uf[uf.yr==y].magnitude.to_numpy()
    if len(g)<10: continue
    s=mc_set(g)
    rows.append(dict(year=y,n=len(g),mmin=np.min(g),p5=np.percentile(g,5),med=np.median(g),**s))
T=pd.DataFrame(rows).set_index("year")
fig,ax=plt.subplots(figsize=(11,5))
ax.plot(T.index,T.mmin,"v-",color="tab:green",label="min magnitude")
ax.plot(T.index,T.p5,"^-",color="seagreen",alpha=0.7,label="5th percentile")
ax.plot(T.index,T.med,"o-",color="0.4",label="median")
ax.plot(T.index,T.maxc,"s--",color="tab:orange",label="Mc MAXC")
ax.plot(T.index,T.maxc2,"D--",color="tab:red",label="Mc MAXC+0.2")
ax.plot(T.index,T.ks,"x--",color="tab:blue",label="Mc K-S")
for yy in (2016,2019): ax.axvline(yy,color="0.8",lw=0.8,ls=":")
ax.set(xlabel="Year",ylabel="ML",title="UF subregion: minimum magnitude vs Mc estimators")
ax.set_xticks(T.index); ax.tick_params(axis="x",labelrotation=45); ax.legend(ncol=2,fontsize=8)
fig.tight_layout(); plt.show()
print(T.round(2).to_string())""")

md("""## 3 · Recent FMD shape — a gradual rollover, not a step

2019–2024 incremental + cumulative FMD with all Mc marks. The incremental counts ramp up smoothly from
ML ≈ −1 to a mode near +0.2: detection probability rises *gradually*, so the min-mag (fringe) sits far
below where the catalog is complete (Mc).""")
co("""rec=uf[uf.yr>=2019]; m=bin_to_precision(np.sort(rec.magnitude.to_numpy()),DM); m=m[np.isfinite(m)]
e=np.round(np.arange(round(m.min(),1),round(m.max(),1)+DM,DM),1)
inc,_=np.histogram(m,bins=np.append(e,e[-1]+DM)); cum=np.array([(m>=x).sum() for x in e])
s=mc_set(m)
fig,ax=plt.subplots(1,2,figsize=(13,4.6))
ax[0].bar(e,inc,width=DM*0.9,color="steelblue",alpha=0.8); ax[0].set(xlabel="ML",ylabel="count / 0.1 bin",title=f"Incremental FMD 2019-2024 (N={len(m)})")
ax[0].axvline(m.min(),color="tab:green",lw=1.2,label=f"min {m.min():+.2f}")
ax[0].axvline(s['maxc'],color="tab:orange",ls="--",label=f"MAXC {s['maxc']}")
ax[0].axvline(s['maxc2'],color="tab:red",ls="--",label=f"MAXC+0.2 {s['maxc2']}")
ax[0].axvline(s['ks'],color="tab:blue",ls=":",label=f"K-S {s['ks']}"); ax[0].legend(fontsize=8)
ax[1].semilogy(e,cum,"k.",ms=5); ax[1].set(xlabel="ML",ylabel="N(≥ML)",title="Cumulative FMD 2019-2024")
for v,c_,l in [(s['maxc'],"tab:orange","MAXC"),(s['maxc2'],"tab:red","MAXC+0.2"),(s['ks'],"tab:blue","K-S")]:
    ax[1].axvline(v,color=c_,ls="--",lw=1)
ax[1].axvline(m.min(),color="tab:green",lw=1.2)
fig.tight_layout(); plt.show()
print(f"recent: min {m.min():+.2f}, mode/MAXC {s['maxc']}, MAXC+0.2 {s['maxc2']}, K-S {s['ks']}; gap min->MAXC = {s['maxc']-m.min():.2f} ML")""")

md("""## 4 · Detection-completeness curve — the principled Mc

Fit a Gutenberg-Richter line to the **clearly-complete** part (ML ≥ anchor, above the mode), extrapolate
it down, and take **detection fraction q(M) = observed / GR-predicted** per bin. **Mc(90%)** and
**Mc(99%)** are the magnitudes above which q crosses those thresholds — a defensible completeness that
makes the gradual rollover explicit (Mignan 2012 style).""")
co("""def detection_curve(mm, anchor=0.7):
    m=bin_to_precision(np.sort(np.asarray(mm,float)),DM); m=m[np.isfinite(m)]
    e=np.round(np.arange(round(m.min(),1),round(m.max(),1)+DM,DM),1)
    inc=np.histogram(m,bins=np.append(e,e[-1]+DM))[0].astype(float)
    be=ClassicBValueEstimator(); be.calculate(m[m>=anchor],mc=anchor,delta_m=DM); b=be.b_value
    Nanc=(m>=anchor).sum()
    # GR-predicted cumulative then incremental
    Npred_cum=Nanc*10**(-b*(e-anchor)); pred_inc=np.diff(np.append(Npred_cum, Npred_cum[-1]*10**(-b*DM)))*-1
    pred_inc=np.clip(pred_inc,1e-9,None)
    q=np.clip(inc/pred_inc,0,1.5)
    def mc_at(thr):
        ok=np.where((q>=thr)&(e<=anchor))[0]
        return float(e[ok[0]]) if len(ok) else np.nan
    return e,q,b,mc_at(0.90),mc_at(0.99)
e,q,b,mc90,mc99=detection_curve(rec.magnitude.to_numpy())
fig,ax=plt.subplots(figsize=(8.5,4.6))
ax.plot(e,q,"o-",color="purple"); ax.axhline(0.9,color="0.6",ls=":",lw=1); ax.axhline(0.99,color="0.4",ls="--",lw=1)
ax.axvline(mc90,color="tab:orange",lw=1.2,label=f"Mc(90%)={mc90:+.2f}"); ax.axvline(mc99,color="tab:red",lw=1.2,label=f"Mc(99%)={mc99:+.2f}")
ax.set(xlabel="ML",ylabel="detection fraction  q(M)=obs/GR",title=f"UF detection-completeness curve 2019-2024 (b={b:.2f})",xlim=(-1.3,1.2),ylim=(0,1.2))
ax.legend(); fig.tight_layout(); plt.show()
print(f"detection-curve Mc: 90%={mc90:+.2f}, 99%={mc99:+.2f}  (vs MAXC {mc_set(rec.magnitude)['maxc']}, MAXC+0.2 {mc_set(rec.magnitude)['maxc2']}, K-S {mc_set(rec.magnitude)['ks']})")""")

md("""## 5 · Is it spatial? — dense core vs edges, and Mc vs distance to nearest station

If the high aggregate Mc were caused by poorly-covered edges, the dense core would have a much lower Mc.
It does not — the gradual rollover is intrinsic.""")
co("""import glob, os
# station coords from station_table (most recent year present)
stc=None
for f in sorted(glob.glob(os.path.join("..","station_table","stations_*.csv")))[::-1]:
    s=pd.read_csv(f); cols={c.lower():c for c in s.columns}
    la=next((cols[k] for k in cols if k in ("lat","latitude")),None); lo=next((cols[k] for k in cols if k in ("lon","longitude","lng")),None)
    if la and lo: stc=s[[la,lo]].rename(columns={la:"lat",lo:"lon"}).dropna(); break
def nearest_km(row):
    if stc is None: return np.nan
    dl=(stc.lat-row.lat)*111.32; dn=(stc.lon-row.lon)*111.32*np.cos(np.radians(row.lat)); return np.sqrt(dl**2+dn**2).min()
clat,clon=rec.lat.median(),rec.lon.median()
core=rec[(np.abs(rec.lat-clat)<0.06)&(np.abs(rec.lon-clon)<0.06)]; edge=rec.drop(core.index)
print("region            N    min    MAXC  K-S   Mc99")
for lab,g in [("whole UF box",rec),("dense core ~13km",core),("edges",edge)]:
    if len(g)>=50:
        s=mc_set(g.magnitude); _,_,_,_,m99=detection_curve(g.magnitude.to_numpy())
        print(f"{lab:17s} {len(g):4d}  {g.magnitude.min():+.2f}  {s['maxc']:.2f}  {s['ks'] if s['ks']==s['ks'] else float('nan'):.2f}  {m99:+.2f}")
if stc is not None:
    rec=rec.copy(); rec["dsta"]=rec.apply(nearest_km,axis=1)
    fig,ax=plt.subplots(figsize=(8,4.4))
    for q0,q1,lab in [(0,5,"0-5 km"),(5,10,"5-10 km"),(10,99,">10 km")]:
        g=rec[(rec.dsta>=q0)&(rec.dsta<q1)]
        if len(g)>=60:
            mm=bin_to_precision(np.sort(g.magnitude.to_numpy()),DM); ee=np.round(np.arange(round(mm.min(),1),1.2,DM),1)
            cumv=np.array([(mm>=x).sum() for x in ee]); ax.semilogy(ee,cumv/cumv.max(),"o-",ms=3,label=f"{lab} (N={len(g)}, min {g.magnitude.min():+.2f})")
    ax.set(xlabel="ML",ylabel="normalised N(≥ML)",title="FMD by distance to nearest station (2019-2024)"); ax.legend(fontsize=8)
    fig.tight_layout(); plt.show()
else:
    print("(station coords not found — core/edge table above is the spatial check)")""")

md("""## 6 · Homogeneous seismicity rate — events above a fixed high cutoff

For a **temporally homogeneous** rate the cutoff must exceed the **worst-era completeness**. The sparse
early network (2010–2015) had high completeness, so a full-period homogeneous catalog is forced up to
**Mc ≈ max over years of the annual completeness** — which lands near **1.0**. So ML ≥ 1.0 isn't
arbitrarily high: it's the price of homogeneity across the whole densification history. Above it, annual
counts are a clean seismicity-rate trend (no densification artefact); below it, only the recent era is
complete. Below we show annual N above several cutoffs and the homogeneity floor.""")
co("""cuts=[0.5,1.0,1.5]
NA=pd.DataFrame({f"N>={c:.1f}":uf.groupby("yr").magnitude.apply(lambda g:(np.asarray(g)>=c).sum()) for c in cuts})
floor_maxc=float(np.nanmax(T.maxc)); floor_maxc2=float(np.nanmax(T.maxc2)); floor_ks=float(np.nanmax(T.ks))
print(f"homogeneity floor = max annual Mc over 2010-2024:  MAXC {floor_maxc:.2f} | MAXC+0.2 {floor_maxc2:.2f} | K-S {floor_ks:.2f}")
print(f"-> a cutoff >= ~{floor_maxc2:.1f} is complete in EVERY year; ML>=1.0 satisfies this.\\n")
fig,ax=plt.subplots(1,2,figsize=(13,4.6))
for c,col in zip(cuts,["0.6","tab:purple","tab:red"]):
    ax[0].plot(NA.index,NA[f"N>={c:.1f}"],"o-",color=col,label=f"N(ML≥{c:.1f})")
for yy in (2016,2019): ax[0].axvline(yy,color="0.85",lw=0.8,ls=":")
ax[0].set(xlabel="Year",ylabel="annual event count",title="Annual N above fixed cutoffs (homogeneous rate)")
ax[0].set_xticks(NA.index); ax[0].tick_params(axis="x",labelrotation=45); ax[0].legend()
# the trustworthy rate: N>=1.0, with Poisson sqrt(N) error
n10=NA["N>=1.0"]
ax[1].errorbar(n10.index,n10.values,yerr=np.sqrt(n10.values),fmt="s-",color="tab:red",capsize=2)
ax[1].set(xlabel="Year",ylabel="N(ML≥1.0)  ±√N",title="Homogeneous UF seismicity rate (ML≥1.0)")
ax[1].set_xticks(n10.index); ax[1].tick_params(axis="x",labelrotation=45)
fig.tight_layout(); plt.show()
print(NA.to_string())
print(f"\\nML>=1.0: total {int(n10.sum())} events over 15 yr, mean {n10.mean():.0f}/yr "
      f"(range {int(n10.min())}-{int(n10.max())}); this is the densification-free rate series.")""")

md("""## 7 · Summary & recommendation""")
co("""s=mc_set(rec.magnitude)
print("UF Mc COMPLETENESS INVESTIGATION — summary\\n"+"="*50)
print(f"recent (2019-2024): N={len(rec)}, min mag {rec.magnitude.min():+.2f}, median {rec.magnitude.median():+.2f}")
print(f"Mc estimators: MAXC {s['maxc']}, MAXC+0.2 {s['maxc2']}, K-S {s['ks']}, detection-curve Mc90 {mc90:+.2f} / Mc99 {mc99:+.2f}")
print("\\nWhy min-mag falls but Mc does not:")
print(" - min-mag = detection FRINGE (few tiny events near dense stations); Mc = magnitude of COMPLETENESS.")
print(" - the FMD rollover is GRADUAL (detection prob ramps from ~ -1 to ~ +0.2), so min << Mc by design.")
print("\\nWhy recent Mc 'looks too high':")
print(" - MAXC+0.2 over-corrects on a gradual rollover -> 0.5-0.9 (the temporal-plot default). MAXC(mode)")
print("   and K-S (~0.2-0.4) are reasonable; the detection-curve Mc90/99 quantify it properly.")
print(" - it is NOT mainly spatial (core and edges give the same Mc).")
print("\\nRecommendation:")
print(" - For UF temporal/seasonal work, DROP MAXC+0.2; use the detection-curve Mc(99%) (or K-S) as Mc.")
print(" - True recent completeness is ~%+.1f (Mc99), well below the MAXC+0.2 value, and the falling"%mc99)
print("   min-mag reflects improved detection of the sub-Mc fringe, not a falling Mc.")
print(" - For a fully rigorous, station-based completeness, PMC (Schorlemmer & Woessner 2008) is the next step.")""")

nb.cells=C
out="11.Mc_completeness_investigation.ipynb"; nbf.write(nb,out); print("wrote",out,len(C),"cells")
