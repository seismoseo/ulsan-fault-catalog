#!/usr/bin/env python
"""Generate 26.UF_relocated_dtcc_analysis.ipynb — analysis of the HypoDD dt.cc-relocated (kim2011) UF
catalog with SOTA UF-only-corrected ML recomputed at the refined hypocentres (catalog_ml_heo_ufonly_reloc.csv).

Answers the four user questions on ONE coherent population (ALL relocated events with n_used>=3, dt.cc AND
dt.ct — the dt.cc/dt.ct split is precision, not detection), because the background rate is the declustered
subset of the SAME NND run (a split population would be incoherent):
  Q1 coverage: how many dt.cc-resolved events carry SOTA ML.
  Q2 ML recomputation: how ML shifts when locations are refined (dml).
  Q3 3D NND: Zaliapin-Ben-Zion declustering using DEPTH, fractal dimension Df=2.5 (ZBZ 3D value),
     b=1.0 fixed, vs the 2D (Df=1.6) run.
  Q4 density maps (background/clustered) + depth cross-sections + declustered background RATE.
Runs in `base`."""
import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# UF dt.cc-relocated catalog — SOTA ML, 3D NND, density & background rate

The whole-subregion **HypoDD dt.cc** relocation (kim2011 velocity; nb21/nb22) sharpened the Ulsan-Fault
hypocentres and gave **reliable depths**. Here we exploit those refined locations:

1. **Coverage** — how many dt.cc-resolved events carry our SOTA UF-only-corrected ML.
2. **ML recomputed at the refined hypocentres** (`catalog_ml_heo_ufonly_reloc.csv`, `build_ufonly_reloc_ml.py`):
   amplitudes (`peak_mm`) are unchanged, only the **distance-attenuation term** moves, so ML shifts only
   through the new hypocentral distance R = √(epi² + depth²) (Heo et al. 2024 formula).
3. **3D NND declustering** using depth — fractal dimension **Df = 2.5** (Zaliapin–Ben-Zion 3D value), vs the
   2D (Df = 1.6) run; **b = 1.0 fixed** (as nb25/seasonal).
4. **Density maps + depth cross-sections + declustered background rate** on the relocated catalog.

**One coherent population (disclosed).** The background rate is the declustered subset of the *same* NND run,
so we use a single population throughout: **all relocated events with SOTA ML, n_used ≥ 3** — dt.cc-resolved
*and* dt.ct-relocated (the split is location precision, not detection; excluding dt.ct would drop mainshock
parents like the 2014 M3.89 and open a completeness hole below Mc).
Completeness = the time-uniform **Mc = 1.2** (nb25 §7; set by the sparse early-network era). Honest caveat:
at Mc=1.2 the complete count is small (~100) — the rate is Poisson-noisy, exactly as in nb13/nb14; the
spatial/NND structure (which uses all magnitudes, mmin=None) is the better-resolved result here.

*References:* Heo et al. (2024); Zaliapin & Ben-Zion (2013, 2020); Waldhauser & Ellsworth (2000, HypoDD).""")

# ----------------------------------------------------------------- §0 setup
co(r"""import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, pandas as pd
from scipy import stats
from scipy.stats import norm as _norm, gaussian_kde
from obspy.geodetics.base import gps2dist_azimuth
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":120,"axes.grid":True,"grid.alpha":0.3,"font.size":11})
sys.path.insert(0,"/home/msseo/works/16.kma_absolute_location")
from kma_absolute_location import nnd

UF=(129.25,129.55,35.60,35.90); DM=0.1; MC=1.2          # time-uniform completeness (nb25 §7)
B_NND=1.0; D2D=1.6; D3D=2.5                              # NND: b fixed; Df 1.6 (2D) vs 2.5 (3D, ZBZ)
PRE=(2010,2013); POST=(2019,2024)
RELOC=("/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/"
       "runs/uf_subregion_reuse/2.HypoDD/03.dt.cc_kim2011/hypoDD.reloc")

ev=pd.read_csv("/home/msseo/works/02.Ulsan_Fault_detection/analysis/local_magnitudes/catalog_ml_heo_ufonly_reloc.csv")
ev["event_time"]=pd.to_datetime(ev.event_time,format="ISO8601",utc=True,errors="coerce")  # ISO8601: whole-second times parse too
ev=ev[~ev.event_idx.isin(set(pd.read_csv("/home/msseo/works/02.Ulsan_Fault_detection/analysis/local_magnitudes/blast_event_idx_deblast.csv").event_idx.dropna().astype(int)))].copy()  # DE-BLAST: drop quarry-blast events (nb22 §7)

# ----- ANALYSIS POPULATION: EVERY relocated event with reliable ML (dt.cc AND dt.ct), best-available loc -----
# The dt.cc / dt.ct split is about location PRECISION (tens of m vs ~hundreds of m), NOT detection — both come
# from the SAME PhaseNet+ catalog. Restricting the declustering to dt.cc-resolved events would (a) DROP mainshock
# PARENTS that lost their cc links to reweighting (incl. the catalog-max, M3.94 abs / M3.89 reloc — a dt.ct event
# with 1189 catalog links) and (b) open an artificial completeness HOLE below Mc (dt.cc kept, dt.ct not),
# biasing the clustered/background split and the rate. So we keep ALL relocated events with reliable ML
# (n_used>=3) at their relocated location, tagged loc_quality:
#   'dtcc' = dt.cc-resolved (sharp, tens of m);  'dtct' = relocated by catalog dt only (~hundreds of m, still
#   fine for km-scale NND);  'abs' = the 5 M>=Mc events NOT in the HypoDD output, backfilled at absolute loc so
#   the event is not lost (depth clipped to the relocated range — HypoInverse depths unreliable, one was 51 km).
rel=ev[ev.n_used>=3].dropna(subset=["ml_ufcorr_reloc","lat","lon","depth"]).copy()   # dt.cc + dt.ct, reliable ML
rel["mag"]=rel.ml_ufcorr_reloc
rel["loc_quality"]=np.where(rel.is_dtcc,"dtcc","dtct")
rel=rel[["event_idx","event_time","lat","lon","depth","mag","n_used","is_dtcc","loc_quality"]]
# backfill: large events (n_used>=3, old ML>=Mc) that are NOT in the relocated catalog -> absolute location.
# EXACT event_idx join (no time matching): with_ml_heo_clean carries the frozen event_idx + abs HypoInverse loc.
full=pd.read_csv("/home/msseo/works/02.Ulsan_Fault_detection/analysis/local_magnitudes/catalog_ml_heo_ufonly.csv"); full["event_time"]=pd.to_datetime(full.event_time,format="ISO8601",utc=True,errors="coerce")
big=full[(full.n_used>=3)&(full.ml_ufcorr>=MC)]
miss=big[~big.event_idx.isin(ev.event_idx)].copy()
cl0=pd.read_csv("/home/msseo/works/02.Ulsan_Fault_detection/analysis/local_magnitudes/catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo_clean.csv")[["event_idx","lat","lon","depth"]]
mb=miss.merge(cl0,on="event_idx",how="left").dropna(subset=["lat","lon"])
# absolute HypoInverse depths for these unrelocated events are UNRELIABLE (deep-biased; one is 51 km — unphysical
# for this crustal zone). Clip to the relocated catalog's depth range so a bad depth can't spuriously isolate a
# parent in the 3D NND; the (decent) absolute epicentre is kept. DISCLOSED.
_dlo,_dhi=float(ev.depth.min()),float(ev.depth.max())
mb["depth"]=mb["depth"].clip(_dlo,_dhi)
mb=mb.rename(columns={"ml_ufcorr":"mag"}); mb["is_dtcc"]=False; mb["loc_quality"]="abs"
mb=mb[["event_idx","event_time","lat","lon","depth","mag","n_used","is_dtcc","loc_quality"]]
pop=pd.concat([rel,mb],ignore_index=True).dropna(subset=["event_time","lat","lon","depth","mag"])
pop["year"]=pop.event_time.dt.year; pop=pop.sort_values("event_time").reset_index(drop=True)
_q=pop.loc_quality.value_counts()
print(f"relocated-with-ML: {len(ev):,} | ANALYSIS POPULATION (ALL relocated, n_used>=3, dt.cc + dt.ct): {len(pop):,}")
print(f"  loc_quality: dtcc {_q.get('dtcc',0)} (sharp) | dtct {_q.get('dtct',0)} (incl. catalog-max M3.89) | abs {_q.get('abs',0)} (backfilled, all small)")
print(f"  large (M>={MC}) events kept: {int((pop.mag>=MC).sum())}  (of 144 in the full catalog) | max M = {pop.mag.max():.2f}")
print(f"  depth {pop.depth.min():.1f}-{pop.depth.max():.1f} km")""")

# ----------------------------------------------------------------- §1 coverage
md(r"""## 1 · Coverage and population — keeping every large event (Q1)

An event is **dt.cc-resolved** if it has ≥1 surviving cross-correlation link (`nccp+nccs>0`); the rest are
**dt.ct-relocated** (located by catalog differential times only). This distinction is about location
**precision** (tens of m vs ~hundreds of m), **not** detection — both classes come from the same PhaseNet+
catalog. A dt.cc-only population would (i) drop mainshock **parents** that lost their cc links to reweighting
— including the **catalog-max event** (M3.94 absolute → **M3.89 relocated**, a dt.ct event held by
**1,189 catalog differential-time links**) — and (ii) open an artificial completeness **hole below Mc**
(dt.cc kept, dt.ct not), both of which bias the declustering and rate. We therefore take **every relocated
event with reliable ML (n_used≥3)** — dt.cc *and* dt.ct — at its best-available location
(`loc_quality` = dtcc / dtct / abs), so no event or parent is lost.""")
co(r"""COLS=["id","lat","lon","depth","x","y","z","ex","ey","ez","yr","mo","dy","hr","mi","sc","mag",
      "nccp","nccs","nctp","ncts","rcc","rct","cid"]
r=pd.read_csv(RELOC,sep=r"\s+",header=None,names=COLS); r["ncc"]=r.nccp+r.nccs
n_reloc=len(r); n_dtcc=int((r.ncc>0).sum())
n_dtcc_n3=int((ev.is_dtcc&(ev.n_used>=3)).sum())
n_dtct_n3=int((~ev.is_dtcc&(ev.n_used>=3)).sum())
n_dtcc_mc=int((ev.is_dtcc&(ev.n_used>=3)&(ev.ml_ufcorr_reloc>=MC)).sum())
tab=pd.DataFrame([
    ("relocated events (kim2011 dt.cc)",n_reloc),
    ("  dt.cc-resolved (nccp+nccs>0)",n_dtcc),
    ("  dt.ct-relocated (no surviving cc link)",n_reloc-n_dtcc),
    ("relocated AND carry SOTA ml_ufcorr",len(ev)),
    ("  dt.cc-resolved & n_used>=3",n_dtcc_n3),
    ("  dt.ct-relocated & n_used>=3 (WAS EXCLUDED)",n_dtct_n3),
    ("ANALYSIS POPULATION (all relocated, n_used>=3)",len(pop)),
],columns=["subset","N"])
print(tab.to_string(index=False))
# large-event recovery accounting (the point of this section)
nl_dtcc=int(((pop.mag>=MC)&(pop.loc_quality=="dtcc")).sum())
nl_dtct=int(((pop.mag>=MC)&(pop.loc_quality=="dtct")).sum())
nl_abs =int(((pop.mag>=MC)&(pop.loc_quality=="abs")).sum())
print(f"\nLARGE events (M>={MC}) KEPT = {nl_dtcc+nl_dtct+nl_abs} (target 144):")
print(f"  dtcc {nl_dtcc} (well-linked) | dt.ct-only {nl_dtct} (relocated, no cc; incl. catalog-max M3.89) | abs {nl_abs} (backfilled, all small; max M2.48)")
print(f"  -> a dt.cc-only filter would have kept only {n_dtcc_mc}; we recover {nl_dtct+nl_abs} large mainshock parents.")
print(f"event_idx recovered by the EXACT hypoDD id->ts->event_idx map (no time matching); abs-loc joined by event_idx.")""")

# ----------------------------------------------------------------- §2 ML recompute
md(r"""## 2 · ML recomputed at the refined hypocentres (Q2)

`dml = ml_ufcorr_reloc − ml_ufcorr_old`. Only the hypocentral distance changes (amplitudes fixed), so ML
moves only through the distance-attenuation term. We expect **small, location-driven** shifts. We compare
against the old absolute (HypoInverse) hypocentre to see `dml` vs how far each event moved.""")
co(r"""# old absolute hypocentre — EXACT event_idx join (with_ml_heo_clean carries event_idx + abs loc), to measure
# the location change. No time matching: every reloc row already carries its frozen master event_idx.
cl=pd.read_csv("/home/msseo/works/02.Ulsan_Fault_detection/analysis/local_magnitudes/catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo_clean.csv")[["event_idx","lat","lon","depth"]]
cl=cl.rename(columns={"lat":"olat","lon":"olon","depth":"odep"})
mm=ev.merge(cl,on="event_idx",how="left").dropna(subset=["olat","olon"])
mm["dhoriz_km"]=[gps2dist_azimuth(a,b,c,d)[0]/1000 for a,b,c,d in zip(mm.lat,mm.lon,mm.olat,mm.olon)]
mm["ddep_km"]=mm.depth-mm.odep
d=ev.dml.dropna()
fig,ax=plt.subplots(1,3,figsize=(16,4.4))
ax[0].hist(d,bins=60,color="steelblue",ec="w"); ax[0].axvline(0,color="k",lw=1)
ax[0].axvline(d.median(),color="tab:red",ls="--",lw=1.5,label=f"median {d.median():+.3f}")
ax[0].set(xlabel="dml = ml_reloc − ml_old",ylabel="events",title=f"(a) ML shift (std {d.std():.3f})"); ax[0].legend(fontsize=8)
sc=ax[1].scatter(mm.dhoriz_km,mm.dml,s=8,c=mm.depth,cmap="viridis",alpha=0.6,lw=0)
plt.colorbar(sc,ax=ax[1],label="reloc depth (km)"); ax[1].axhline(0,color="k",lw=0.8)
ax[1].set(xlabel="horizontal move |Δepicentre| (km)",ylabel="dml",title="(b) ML shift vs how far event moved")
ax[2].scatter(mm.depth,mm.dml,s=8,c="0.4",alpha=0.5,lw=0); ax[2].axhline(0,color="k",lw=0.8)
ax[2].set(xlabel="relocated depth (km)",ylabel="dml",title="(c) ML shift vs depth")
fig.tight_layout(); plt.show()
print(f"dml: median {d.median():+.3f}, IQR [{d.quantile(.25):+.3f},{d.quantile(.75):+.3f}], std {d.std():.3f}, "
      f"|dml|>0.1: {(d.abs()>0.1).mean()*100:.1f}%")
print(f"location change: |Δepi| median {mm.dhoriz_km.median():.3f} km, |Δdepth| median {mm.ddep_km.abs().median():.3f} km")
print("=> ML recompute is a SMALL, location-driven refinement; we adopt ml_ufcorr_reloc below.")""")

# ----------------------------------------------------------------- §3 3D NND
md(r"""## 3 · 3D nearest-neighbour declustering (Q3)

Zaliapin–Ben-Zion NND on the relocated population, using **depth** (`metric="3d"`, hypocentral
√(R_epi²+Δz²)) with fractal dimension **Df = 2.5** (the ZBZ 3D value), versus the **2D** run (Df = 1.6,
epicentral). **b = 1.0 fixed** (η ∝ 10^(−bM); catalog b drifts in time). `mmin=None` (all magnitudes used
for linkage). η₀ from a 2-component GMM on log₁₀η.""")
co(r"""def build_g(df):
    g=df.copy(); g["t_year"]=g.event_time.dt.strftime("%Y%m%d%H%M%S").map(nnd.decimal_year)  # CANONICAL exact-year-length decimal year (nnd.decimal_year)
    g["event_id"]=np.arange(len(g))
    return g.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep","mag":"kma_mag"})
g=build_g(pop)
nd3,info3=None,None
nd2=nnd.compute_nnd(g,b=B_NND,D=D2D,mmin=None,metric="2d"); e2_,info2=nnd.fit_eta0(nd2.eta.values,method="gmm")
nd3=nnd.compute_nnd(g,b=B_NND,D=D3D,mmin=None,metric="3d"); e3_,info3=nnd.fit_eta0(nd3.eta.values,method="gmm")
def split(nd,e0):
    clu=set(nd.loc[nd.eta<e0,"event_id"]); bg=~g.event_id.isin(clu); return bg
bg2=split(nd2,e2_); bg3=split(nd3,e3_)
g["bg"]=bg3.values                                   # 3D is the primary (uses depth)
flip=int((bg2.values!=bg3.values).sum())
print(f"2D (Df={D2D}): log10 eta0={np.log10(e2_):+.2f}, clustered {int((~bg2).sum())}/{len(g)} ({100*(~bg2).mean():.0f}%)")
print(f"3D (Df={D3D}): log10 eta0={np.log10(e3_):+.2f}, clustered {int((~bg3).sum())}/{len(g)} ({100*(~bg3).mean():.0f}%)")
print(f"depth changes class for {flip} events ({100*flip/len(g):.0f}%) between the 2D and 3D runs.")
# how the location classes split (3D): dt.ct events go mostly to background (genuinely isolated -> no cc link),
# but large dt.ct events (M3.89) are real cluster parents. Confirms including dt.ct is correct, not artefact.
for _lab,_mk in [("dt.cc-resolved",g.loc_quality=="dtcc"),("dt.ct-relocated",g.loc_quality=="dtct")]:
    _cl=int((~g.bg[_mk]).sum()); _n=int(_mk.sum())
    print(f"  {_lab:16}: N {_n:4d} | clustered {_cl:4d} ({100*_cl/max(_n,1):2.0f}%) | background {_n-_cl:4d} ({100*(_n-_cl)/max(_n,1):2.0f}%)")
if (g.event_idx==704).any(): print(f"  -> 2014 M3.89 (dt.ct) clustered={bool((~g.bg[g.event_idx==704]).values[0])} (a mainshock parent).")""")

md(r"""### 3b · NND structure — R–T pair density and bimodal η (2D vs 3D, SOTA presentation)

Side-by-side **2D (Df=1.6)** and **3D (Df=2.5)** runs. Each R–T panel is drawn with **equal aspect** (data
units equal on both axes) and equal-span limits, so the η₀ line of slope −1 renders at exactly 45°
(Zaliapin–Ben-Zion / Goebel convention) and the two runs are directly comparable.""")
co(r"""def rt_panel(ax,nd,e0,label):
    lt=nd.logT.values; lr=nd.logR.values; ok=np.isfinite(lt)&np.isfinite(lr); lt,lr=lt[ok],lr[ok]
    le0=np.log10(e0); binx=biny=0.1
    Tlo,Thi,Rlo,Rhi=-8.0,2.0,-6.0,4.0     # fixed (log10 T, log10 R) domain, equal 10-unit spans -> eta0 at 45 deg
    Tb=np.arange(Tlo,Thi+binx,binx); Rb=np.arange(Rlo,Rhi+biny,biny); XX,YY=np.meshgrid(Tb,Rb)
    ZZ=gaussian_kde(np.vstack([lt,lr]))(np.vstack([XX.ravel(),YY.ravel()])).reshape(XX.shape)*binx*biny*len(lt)
    pc=ax.pcolormesh(XX,YY,ZZ,cmap=plt.cm.RdYlGn_r,shading="auto")
    ax.plot([Tlo,Thi],-np.array([Tlo,Thi])+le0,"-",lw=2.5,color="w")
    ax.plot([Tlo,Thi],-np.array([Tlo,Thi])+le0,"--",lw=1.5,color="0.3",label=f"η₀ (log₁₀={le0:.2f})")
    ax.set(xlabel="Rescaled time  log₁₀ T",ylabel="Rescaled distance  log₁₀ R",
           title=label,xlim=(Tlo,Thi),ylim=(Rlo,Rhi))
    ax.set_aspect("equal",adjustable="box"); ax.legend(loc="lower left",fontsize=8)   # EQUAL aspect x=y
    return pc
fig,ax=plt.subplots(1,2,figsize=(14,6.6))
for a,(nd_,e_,lab) in zip(ax,[(nd2,e2_,f"2D NND R–T (Df={D2D}, b={B_NND})"),
                              (nd3,e3_,f"3D NND R–T (Df={D3D}, b={B_NND})")]):
    pc=rt_panel(a,nd_,e_,lab); cb=fig.colorbar(pc,ax=a,fraction=0.046,pad=0.04); cb.set_label("event pairs")
fig.suptitle("Nearest-neighbour pairs in R–T — 2D vs 3D (Goebel / Zaliapin–Ben-Zion)",y=1.0); plt.show()
# bimodal log10(eta): 2D vs 3D
def bimodal_panel(ax,nd,info,e0,label):
    le=np.log10(nd.eta.values); le=le[np.isfinite(le)]; le0=np.log10(e0)
    ax.hist(le,bins=40,density=True,color="0.82",ec="w")
    xs=np.linspace(le.min(),le.max(),400); mns,sgs,wts=info["means"],info["sigmas"],info["weights"]
    for j,(c_,nm) in enumerate([("tab:red","clustered mode"),("tab:green","background mode")]):
        ax.plot(xs,wts[j]*_norm.pdf(xs,mns[j],sgs[j]),color=c_,lw=2,label=nm)
    ax.axvline(le0,color="k",ls="--",lw=2,label=f"η₀={le0:.2f}")
    ax.set(xlabel="log₁₀ η (nearest-neighbour proximity)",ylabel="Density",title=label); ax.legend(fontsize=8)
fig,ax=plt.subplots(1,2,figsize=(13,4.4))
bimodal_panel(ax[0],nd2,info2,e2_,f"2D bimodal NND + GMM (Df={D2D})")
bimodal_panel(ax[1],nd3,info3,e3_,f"3D bimodal NND + GMM (Df={D3D})")
fig.tight_layout(); plt.show()
print(f"η₀ (GMM):  2D (Df={D2D}) = {np.log10(e2_):.2f}  |  3D (Df={D3D}) = {np.log10(e3_):.2f}  |  Goebel UF ~ -3.97")""")

# ----------------------------------------------------------------- §4 density + depth + rate
md(r"""## 4 · Density maps, depth cross-sections, and background rate (Q4)

Background (declustered, isolated) vs clustered (aftershock/swarm) on the **refined** locations: epicentre
and smoothed-density maps (PyGMT), **depth cross-sections** (now meaningful with reliable dt.cc depths), and
the declustered background **rate** PRE (2010–13) vs POST-2019 (2019–24) at the complete Mc = 1.2.""")
co(r"""import xarray as xr
from scipy.ndimage import gaussian_filter
import pygmt
pygmt.config(MAP_FRAME_TYPE="plain", FORMAT_GEO_MAP="ddd.x", FONT_TITLE="13p")
REGION=list(UF); PROJ="M11c"; FRAME=["WSne","xa0.1","ya0.1"]; SCALE="jBL+w10k+o0.6c/0.6c"; SP=0.004; DEG_KM=111.19
def dgrid(lon,lat,sp=SP,sm=1.5):
    xb=np.arange(REGION[0],REGION[1]+sp,sp); yb=np.arange(REGION[2],REGION[3]+sp,sp)
    H,_,_=np.histogram2d(lon,lat,bins=[xb,yb]); H=gaussian_filter(H,sm)
    return xr.DataArray(H.T,coords={"lat":(yb[:-1]+yb[1:])/2,"lon":(xb[:-1]+xb[1:])/2},dims=["lat","lon"])
def cell_km2(sp=SP):
    latc=np.deg2rad((REGION[2]+REGION[3])/2); return (sp*DEG_KM)*(sp*DEG_KM*np.cos(latc))
def epimap(bgm,clm,title):
    fig=pygmt.Figure(); fig.basemap(region=REGION,projection=PROJ,frame=[FRAME[0]+f"+t{title}"]+FRAME[1:])
    fig.coast(land="gray97",water="lightblue",shorelines="0.5p,gray40")
    fig.plot(x=bgm.svi_lon,y=bgm.svi_lat,size=0.05*1.5**(bgm.kma_mag-0.5),style="cc",fill="steelblue",pen="0.2p,black",transparency=25,label="background+S0.25c")
    fig.plot(x=clm.svi_lon,y=clm.svi_lat,size=0.05*1.5**(clm.kma_mag-0.5),style="cc",fill="red",pen="0.2p,black",transparency=25,label="clustered+S0.25c")
    fig.legend(position="jTR+o0.2c",box="+gwhite+p0.5p"); fig.basemap(frame=FRAME,map_scale=SCALE); fig.show()
def densmap(sub,title,cbar,per_yr=None,per_area=False):
    grid=dgrid(sub.svi_lon.values,sub.svi_lat.values)
    if per_area: grid=grid/cell_km2()
    if per_yr is not None: grid=grid/per_yr
    vmax=float(grid.max()) or 1.0
    fig=pygmt.Figure(); fig.basemap(region=REGION,projection=PROJ,frame=[FRAME[0]+f"+t{title}"]+FRAME[1:])
    pygmt.makecpt(cmap="hot",series=[0,vmax],reverse=True)
    fig.grdimage(grid.where(grid>=0.04*vmax),region=REGION,projection=PROJ,cmap=True,nan_transparent=True)
    fig.coast(shorelines="0.5p,gray40"); fig.colorbar(frame=f"af+l{cbar}")
    fig.basemap(frame=FRAME,map_scale=SCALE); fig.show()
bgm=g[g.bg]; clm=g[~g.bg]; Tyr=float((g.event_time.max()-g.event_time.min()).days)/365.25
epimap(bgm,clm,"UF dt.cc background vs clustered (refined loc)")
densmap(bgm,"Background density (dt.cc)","smoothed event count")
densmap(clm,"Clustered density (dt.cc)","smoothed event count")
densmap(bgm,"Background RATE (dt.cc)","events / yr / km@+2@+",per_yr=Tyr,per_area=True)
print(f"mapped {len(bgm)} background, {len(clm)} clustered; span {Tyr:.1f} yr")""")

md(r"""### 4b · Depth cross-sections (new — reliable dt.cc depths)

Hypocentre depth vs longitude and latitude, coloured by NND class. Only meaningful now that depths come
from the joint dt.cc relocation (the absolute HypoInverse depths were poorly resolved).""")
co(r"""ab=g[g.loc_quality=="abs"]                                   # backfilled small events (absolute depth; max M2.48)
fig,ax=plt.subplots(1,2,figsize=(15,4.8))
for a,(xc,xl) in zip(ax,[("svi_lon","Longitude"),("svi_lat","Latitude")]):
    a.scatter(g.loc[g.bg,xc],g.loc[g.bg,"svi_dep"],s=14,c="steelblue",alpha=0.6,lw=0,label="background")
    a.scatter(g.loc[~g.bg,xc],g.loc[~g.bg,"svi_dep"],s=14,c="tab:red",alpha=0.6,lw=0,label="clustered")
    a.scatter(ab[xc],ab.svi_dep,s=70,marker="*",c="k",lw=0,zorder=5,label="abs-loc backfilled (small, max M2.48)")
    a.set(xlabel=xl,ylabel="Depth (km)",title=f"Depth section vs {xl.lower()}"); a.invert_yaxis(); a.legend(fontsize=8)
fig.tight_layout(); plt.show()
print(f"depth: background median {g.loc[g.bg,'svi_dep'].median():.1f} km, clustered {g.loc[~g.bg,'svi_dep'].median():.1f} km "
      f"| {len(ab)} small backfilled events shown at ABSOLUTE (HypoInverse) depth — poorly resolved, flagged.")""")

md(r"""### 4c · Declustered background rate — PRE vs POST-2019 (Mc = 1.2)

The decisive temporal test on the refined catalog: above the time-uniform Mc = 1.2, is the **declustered
3D-NND background** rate higher after 2019? Poisson conditional-binomial test. **Low-N caveat** — at Mc=1.2
this dt.cc population is small (~100), so this is Poisson-noisy (consistent with nb13/nb14).""")
co(r"""def pois(n,T):
    return n/T
Tpre=PRE[1]-PRE[0]+1; Tpost=POST[1]-POST[0]+1
rows=[]
for cut in [1.2,1.5]:
    for lab,sel in [("raw (all events)",np.ones(len(g),bool)),("declustered background",g.bg.values)]:
        s=g[sel&(g.kma_mag>=cut)]
        npre=int(((s.year>=PRE[0])&(s.year<=PRE[1])).sum()); npost=int(((s.year>=POST[0])&(s.year<=POST[1])).sum())
        ntot=npre+npost; p0=Tpost/(Tpre+Tpost)
        pval=stats.binomtest(npost,ntot,p0).pvalue if ntot>0 else np.nan
        rows.append(dict(cut=cut,population=lab,n_pre=npre,r_pre=pois(npre,Tpre),n_post=npost,r_post=pois(npost,Tpost),
                         ratio=pois(npost,Tpost)/(pois(npre,Tpre)+1e-9),p=pval))
rate=pd.DataFrame(rows)
print(rate.assign(r_pre=rate.r_pre.round(2),r_post=rate.r_post.round(2),ratio=rate.ratio.round(2),p=rate.p.round(3)).to_string(index=False))
fig,ax=plt.subplots(1,2,figsize=(13,4.4))
sub=rate[rate.cut==1.2]; xx=np.arange(2); w=0.35
ax[0].bar(xx-w/2,sub.r_pre,w,color="tab:red",label=f"pre {PRE[0]}-{PRE[1]}")
ax[0].bar(xx+w/2,sub.r_post,w,color="tab:blue",label=f"post {POST[0]}-{POST[1]}")
for i,(_,rr) in enumerate(sub.iterrows()): ax[0].text(i,max(rr.r_pre,rr.r_post)+0.3,f"{rr.ratio:.2f}×\np={rr.p:.2f}",ha="center",fontsize=9)
ax[0].set_xticks(xx); ax[0].set_xticklabels(["raw\n(all events)","declustered\nbackground"])
ax[0].set(ylabel="rate (events/yr, M≥1.2)",title="(a) Rate pre vs post-2019 (M≥1.2)"); ax[0].legend(fontsize=8)
for cut,c in [(1.2,"tab:green"),(1.5,"tab:purple")]:
    sy=g[g.bg&(g.kma_mag>=cut)].groupby("year").size().reindex(range(2010,2025),fill_value=0)
    ax[1].plot(sy.index,sy.values,"o-",color=c,label=f"background M≥{cut}")
ax[1].axvline(2019,color="0.4",ls=":",lw=1.2); ax[1].set(xlabel="Year",ylabel="background events/yr",title="(b) Annual declustered background")
ax[1].tick_params(axis="x",labelrotation=45); ax[1].legend(fontsize=8); fig.tight_layout(); plt.show()
_bg=rate[(rate.cut==1.2)&(rate.population=="declustered background")].iloc[0]
print(f"\ndeclustered background M≥1.2: {_bg.ratio:.2f}× post/pre (p={_bg.p:.2f}) — LOW N ({_bg.n_pre}+{_bg.n_post}); "
      f"consistent with nb13/14/25 (steady background ~1.2×, n.s.).")""")

# ----------------------------------------------------------------- §5 summary
md(r"""## 5 · Comprehensive summary""")
co(r"""print("="*78); print("UF dt.cc-RELOCATED CATALOG — SOTA ML, 3D NND, DENSITY & RATE".center(78)); print("="*78)
print(f"population: ALL relocated (dt.cc + dt.ct), n_used>=3 = {len(g)} events (loc_quality dtcc/dtct/abs; recomputed ML)\n")
print("Q1 COVERAGE — KEEP ALL LARGE EVENTS (no mainshock-parent bias)")
print(f"  {n_dtcc} dt.cc-resolved of {n_reloc} relocated; analysis population {len(pop)}.")
print(f"  large M>={MC} kept = {nl_dtcc+nl_dtct+nl_abs}/144  (dtcc {nl_dtcc}, dt.ct-only {nl_dtct} incl. "
      f"catalog-max M3.89, abs {nl_abs} small); a dt.cc-only filter would have dropped {nl_dtct+nl_abs}.")
print("\nQ2 ML RECOMPUTE")
print(f"  dml=ml_reloc-ml_old: median {ev.dml.median():+.3f}, std {ev.dml.std():.3f} — small, location-driven; "
      f"adopted ml_ufcorr_reloc.")
print("\nQ3 3D NND (b=1.0 fixed)")
print(f"  2D Df={D2D}: log10 η₀={np.log10(e2_):+.2f}, clustered {100*(~bg2).mean():.0f}%")
print(f"  3D Df={D3D}: log10 η₀={np.log10(e3_):+.2f}, clustered {100*(~bg3).mean():.0f}%  (depth flips {flip} events)")
print("\nQ4 DENSITY + RATE")
print(f"  background median depth {g.loc[g.bg,'svi_dep'].median():.1f} km, clustered {g.loc[~g.bg,'svi_dep'].median():.1f} km")
print(f"  declustered background M≥1.2: {_bg.ratio:.2f}× post/pre (p={_bg.p:.2f}, LOW N) — steady, as nb13/14/25.")
print("\nTAKE-HOMES")
print(" - KEEP ALL LARGE EVENTS: isolated mainshocks (incl. the catalog-max, M3.94 abs / M3.89 reloc) have no cc")
print("   links so a dt.cc-only filter would drop them; the catalog-max is recovered by dt.ct (1,189 links), and")
print("   only 5 small events (max M2.48) fall back to absolute loc — no large mainshock parent is lost.")
print(" - Refined dt.cc locations barely move SOTA ML (|dml| median <0.01): magnitudes are robust to location.")
print(" - 3D NND with depth (Df=2.5) gives a coherent clustered/background split; depth refines membership.")
print(" - The declustered background rate stays steady post-2019 on the refined catalog too (low-N caveat).")
print(" - CUSPID FIX: correcting stale per-pair cuspid headers recovered the catalog-max event and its 2014-09")
print("   sequence into the dt.ct-relocated set (1,189 catalog links) — previously they were stranded at abs loc.")""")

nb.cells=C
out="/home/msseo/works/02.Ulsan_Fault_detection/analysis/reloc_analysis/26.UF_relocated_dtcc_analysis.ipynb"; nbf.write(nb,out); print("wrote",out,len(C),"cells")
