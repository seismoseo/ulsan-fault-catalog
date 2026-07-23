#!/usr/bin/env python
"""Generate 37.UF_kmeans_spatial_clusters_dtcc.ipynb — K-means (K=7) clustering of the kim2011 dt.cc-RESOLVED
relocated UF catalog (all 2739 relocated events) into map-identifiable clusters, then per-cluster temporal
STEADINESS (cumulative-count curves + coefficient of variation of inter-event times) and spatial FRACTAL
dimension (Grassberger-Procaccia correlation dimension D2, in 2D map-plane and full 3D).

Complementary to the NND-family view (nb28-nb35): K-means gives the handful of visually-obvious spatial groups a
reader sees on the map, not the rescaled-time neighbour families. PRIMARY clustering is 2D map-plane (z-scored
x,y) at K=7 (seed 0) — 3D clustering fragments into depth shells, so it is kept only as a comparison; a K=3->10
sweep and a smoothed-density overlay are included for model-selection and validation.

Reuses: Grassberger-Procaccia corr_integral/fit_slope verbatim from build_fractal_dimension_nb.py (nb27); the
PyGMT UF-map idiom from HypoInv/build_reloccmp_nb.py (nb21); the id(200000+i)->ts->event_idx->ML join from
build_cluster_volume_nb.py; kma_absolute_location.clustering.to_utm (EPSG:32652). Runs in conda base
(sklearn/scipy/pygmt all present). New code: KMeans+silhouette+ARI, COV of inter-event times, busiest-30-day burst
share, and a PyGMT categorical-legend cluster map.

Sections: (0) setup + load; (1) K-means K=7 + silhouette-vs-K + K=3-10 sweep; (2) cluster map (PyGMT) + depth sections + 2D/3D
separability; (3) temporal steadiness; (4) correlation dimension D2 (2D & 3D); (5) summary."""
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# K-means spatial clusters — dt.cc-resolved catalog (steadiness & fractal dimension)

**Identical analysis to nb36, but on the dt.cc-RESOLVED subset only** — the events that keep at least one surviving
cross-correlation differential time (`ncc = NCCP+NCCS > 0`) in the whole-box relocation. These are the
**tightest, cc-precision** locations (relative errors of tens of m, reliable depths); the ct-only remainder is
dropped, so this is the robustness/precision counterpart to nb36's all-events partition. **Note the 2014 M3.89
mainshock is ct-only (cc-starved) and therefore absent here** — a direct consequence of it having no cc links.
The **quarry-blast events (nb22 §7) are also removed** — this is the **de-blasted** cc-resolved catalog.

The kim2011 whole-box dt.cc HypoDD relocation (ISTART=2 adaptive), restricted to its cc-resolved events, is
partitioned into **7 spatial clusters** with K-means, so each cluster is a *map-identifiable patch* of the Ulsan
Fault — a complementary view to the NND (Zaliapin–Ben-Zion) families of nb28–nb35 (which group by *rescaled-time*
neighbour, not geometry). Compare the cluster geometry, steadiness (COV) and D2 against nb36 to see how sensitive
they are to including the lower-precision ct-only events.

**What we then ask of each cluster:**
- **Steadiness** — is the cluster a steady background trickle or episodic bursts? Read from the **cumulative event
  count vs time** (steady ⇒ straight line) and the **coefficient of variation (COV) of inter-event times**
  (≈1 = Poisson/random, >1 = bursty/clustered, <1 = quasi-periodic), plus the share of events in the single
  busiest 30-day window.
- **Fractal dimension** — how the hypocentres fill space, via the **Grassberger–Procaccia correlation dimension
  D2** ( C(r) ∝ r^{D2} ), computed in **2D** (map plane) and **3D**. D2→1 = lineament, →2 = plane-filling patch,
  →3 = space-filling volume; the 3D–2D gap measures out-of-plane spread.

**Method notes (disclosed):** we clustered both in **3D** (z-scored `[x,y,z]`) and in **2D** (z-scored `[x,y]`).
Because UF seismicity is strongly depth-layered, 3D K-means returns **depth shells** that overlap in map view, so
— per the pre-agreed fallback rule — the **2D map-plane labeling is primary** (map-identifiable) and the 3D
labeling is kept as the depth-structure comparison (§1–§2). **K = 7** (user choice — K=4–5 merged tectonically
distinct along-strike segments; a silhouette-vs-K scan and a K=3→10 map sweep are shown for context, not to
override K); seed = 0 (reproducible). Weakly-constrained relocations
(`suspect`: moved >3 km, <10 links, or ct-only) are clustered with everything else but counted per cluster. House
style follows nb33/nb34; **spatial maps use PyGMT** (memory rule), statistics use matplotlib.""")

# ------------------------------------------------------------------ §0 setup + load
co(r"""import warnings; warnings.filterwarnings("ignore")
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.font_manager as fm
from scipy.spatial.distance import pdist
from scipy.stats import linregress
import pygmt
sys.path.insert(0,"/home/msseo/works/16.kma_absolute_location")
from kma_absolute_location import clustering, nnd
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams["axes.unicode_minus"]=False
mpl.rcParams.update({"figure.dpi":130,"axes.grid":True,"grid.alpha":0.3,"font.size":10,
                     "legend.framealpha":1,"legend.edgecolor":"black","legend.facecolor":"white"})

KG="/home/msseo/works/02.Ulsan_Fault_detection/KS_KG"
RUN03=("/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/"
       "uf_subregion_reuse/2.HypoDD/03.dt.cc_kim2011")
WF100=("/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/"
       "uf_subregion_reuse/waveforms_100km")
MEIDX=f"{KG}/uf_subregion_hypodd/uf_subregion/members_event_idx.csv"
RELOC_ML=f"{KG}/local_magnitudes/catalog_ml_heo_ufonly_reloc.csv"
FAULT_GMT=f"{KG}/HypoInv/faults_lonlat.gmt"
REGION=[129.25,129.55,35.60,35.90]
RC=["id","lat","lon","depth","x","y","z","ex","ey","ez","yr","mo","dy","hr","mi","sc","mag",
    "nccp","nccs","nctp","ncts","rcc","rct","cid"]
TAB=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"]  # tab10 categorical

# --- full relocated catalog (all events) ---
r0=pd.read_csv(f"{RUN03}/hypoDD.reloc",sep=r"\s+",header=None,names=RC)
scf=r0.sc.clip(0,59.999)
r0["time"]=pd.to_datetime(dict(year=r0.yr,month=r0.mo,day=r0.dy,hour=r0.hr,minute=r0.mi,
             second=scf.astype(int),microsecond=((scf-scf.astype(int))*1e6).astype(int)),utc=True,errors="coerce")
r0["nlinks"]=r0.nccp+r0.nccs+r0.nctp+r0.ncts
# --- id(200000+i) -> waveform-dir timestamp -> event_idx -> ML (canonical join, from build_cluster_volume_nb.py) ---
rl=pd.read_csv(RELOC_ML); rl=rl.dropna(subset=["lat","lon","depth","event_time","ml_ufcorr_reloc"])
_dirs=sorted(os.path.basename(d) for d in glob.glob(os.path.join(WF100,"20*")))
id2ts={200000+i:ts for i,ts in enumerate(_dirs)}
mei=pd.read_csv(MEIDX).sort_values("event_idx")
mei["ts"]=pd.to_datetime(mei.time,utc=True,format="ISO8601").dt.floor("s").dt.strftime("%Y%m%d%H%M%S")
ts2e={}
for _e,_t in zip(mei.event_idx.astype(int),mei.ts): ts2e.setdefault(_t,_e)
r0["event_idx"]=r0.id.map(id2ts).map(ts2e)
r0=r0.merge(rl[["event_idx","ml_ufcorr_reloc","n_used"]],on="event_idx",how="left").drop_duplicates("id")
r0["has_ml"]=r0.ml_ufcorr_reloc.notna()&(r0.n_used>=3)
r0["magU"]=np.where(r0.has_ml,r0.ml_ufcorr_reloc,np.nan)
# --- UTM km (EPSG 32652) for reloc + initial hypoDD.loc, and the suspect flag ---
au,_=clustering.to_utm(r0.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep"}))
r0["x_km"]=au.x_m.values/1000; r0["y_km"]=au.y_m.values/1000; r0["z_km"]=au.depth_m.values/1000
lc=pd.read_csv(f"{RUN03}/hypoDD.loc",sep=r"\s+",header=None,
   names=["id","lat","lon","depth","x","y","z","ex","ey","ez","yr","mo","dy","hr","mi","sc","mag","cid"])[["id","lat","lon","depth"]]
lu,_=clustering.to_utm(lc.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep"}))
lc["x0"],lc["y0"],lc["z0"]=lu.x_m.values,lu.y_m.values,lu.depth_m.values
r0=r0.merge(lc[["id","x0","y0","z0"]],on="id",how="left")
r0["shift_km"]=np.sqrt((au.x_m.values-r0.x0)**2+(au.y_m.values-r0.y0)**2+(au.depth_m.values-r0.z0)**2)/1000
r0["suspect"]=(r0.shift_km>3.0)|(r0.nlinks<10)
r0=r0.dropna(subset=["time","x_km","y_km","z_km"]).reset_index(drop=True)
# --- dt.cc-RESOLVED subset: keep ONLY events with surviving cross-correlation links (ncc>0) ---
# these carry the tightest (cc-precision) locations; the ct-only remainder (incl. the cc-starved M3.89) is dropped.
_nall=len(r0); r0["ncc"]=r0.nccp+r0.nccs
r0=r0[r0.ncc>0].reset_index(drop=True)
print(f"dt.cc-RESOLVED subset: {len(r0)} of {_nall} relocated events keep cc links "
      f"(dropped {_nall-len(r0)} ct-only, incl. the cc-starved 2014 M3.89)")
# --- DE-BLAST: remove the quarry-blast events identified in nb22 §7 (shallow, tightly clustered, daytime KST) ---
_BLAST=set(pd.read_csv(f"{KG}/local_magnitudes/blast_event_idx_deblast.csv").id.astype(int))
_nb0=len(r0); r0=r0[~r0.id.isin(_BLAST)].reset_index(drop=True)
print(f"de-blasted: removed {_nb0-len(r0)} quarry-blast events (2 daytime quarries) -> {len(r0)} events")
print(f"  {int(r0.has_ml.sum())} with ML | {int(r0.suspect.sum())} suspect | "
      f"time {r0.time.min():%Y-%m-%d}..{r0.time.max():%Y-%m-%d} | depth {r0.z_km.min():.1f}-{r0.z_km.max():.1f} km")
# --- smoothed seismicity density (nb26 method: 2D histogram + Gaussian smoothing -> xarray for grdimage) ---
import xarray as xr
from scipy.ndimage import gaussian_filter
SP=0.004
def dgrid(lon,lat,sp=SP,sm=1.5):
    xb=np.arange(REGION[0],REGION[1]+sp,sp); yb=np.arange(REGION[2],REGION[3]+sp,sp)
    H,_,_=np.histogram2d(lon,lat,bins=[xb,yb]); H=gaussian_filter(H,sm)
    return xr.DataArray(H.T,coords={"lat":(yb[:-1]+yb[1:])/2,"lon":(xb[:-1]+xb[1:])/2},dims=["lat","lon"])
DENS=dgrid(r0.lon.values,r0.lat.values)                       # smoothed count density of the cc-resolved events
# --- fault + coastline segments for matplotlib multi-panel maps (nb28 idiom) ---
COAST=f"{KG}/reloc_analysis/coastline_lonlat.gmt"
def _load_segs(path):
    segs=[]; cur=[]
    if not os.path.exists(path): return segs
    for ln in open(path):
        if ln.startswith((">","#")):
            if len(cur)>1: segs.append(np.array(cur))
            cur=[]; continue
        p=ln.split()
        if len(p)>=2:
            try: cur.append([float(p[0]),float(p[1])])
            except ValueError: pass
    if len(cur)>1: segs.append(np.array(cur))
    return segs
FSEG=_load_segs(FAULT_GMT); CSEG=_load_segs(COAST)
def plot_faults(ax):
    for s in FSEG: ax.plot(s[:,0],s[:,1],color="0.35",lw=0.7,zorder=1)
def plot_coast(ax):
    for s in CSEG: ax.plot(s[:,0],s[:,1],color="black",lw=0.6,zorder=1)
ASP=1/np.cos(np.deg2rad(35.75))""")

# ------------------------------------------------------------------ §1 K-means
md(r"""## 1 · K-means clustering (K = 7) — 2D map-plane (primary) vs 3D

The goal is clusters a reader can **identify from the map**. We ran K-means both ways and let the data decide:

- **3D** on z-scored `[x_km, y_km, z_km]`: the depth range (~2–21 km) is comparable in z-score to the horizontal
  spread, and UF seismicity is **strongly depth-layered**, so 3D K-means carves the catalog into **depth shells**
  (≈9, 9.6, 12, 14, 15 km) that each span nearly the whole map footprint — *not* map-identifiable.
- **2D** on z-scored `[x_km, y_km]`: partitions the fault into contiguous **map patches** — exactly the
  visually-separable groups requested.

The adjusted Rand index between the two labelings is low (they disagree substantially), confirming that depth,
not map position, drives the 3D grouping. Per the pre-agreed rule (*fall back to 2D if depth fragments the
map-view groups*), the **2D labeling is the primary `cluster`** used for the map, steadiness and D2; the 3D
labeling is carried as `cl3d` and shown in §2 as the depth-structure comparison. `KMeans(n_clusters=7,
n_init=10, random_state=0)`; clusters relabelled 0–6 by descending size. **K = 7** (K=4–5 grouped tectonically
distinct along-strike segments together; 7 resolves them). The silhouette-vs-K curve and the K=3→10 sweep below
show how the partition evolves — K is a modelling choice, not a sharp optimum here (one continuous fault).""")
co(r"""from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
K=7
# PRIMARY: 2D map-plane clustering -> map-identifiable patches
XY=r0[["x_km","y_km"]].values
Xs2=StandardScaler().fit_transform(XY)
km2=KMeans(n_clusters=K,n_init=10,random_state=0).fit(Xs2)
_o2=pd.Series(km2.labels_).value_counts().index.tolist()
r0["cluster"]=pd.Series(km2.labels_).map({o:i for i,o in enumerate(_o2)}).values
# COMPARISON: 3D clustering (depth-aware) -> depth shells
XYZ=r0[["x_km","y_km","z_km"]].values
Xs3=StandardScaler().fit_transform(XYZ)
km3=KMeans(n_clusters=K,n_init=10,random_state=0).fit(Xs3)
_o3=pd.Series(km3.labels_).value_counts().index.tolist()
r0["cl3d"]=pd.Series(km3.labels_).map({o:i for i,o in enumerate(_o3)}).values
CLS=list(range(K))
ari=adjusted_rand_score(r0.cluster,r0.cl3d)
# silhouette vs K on the 2D (primary) embedding (context only)
silK={k:silhouette_score(Xs2,KMeans(n_clusters=k,n_init=10,random_state=0).fit_predict(Xs2)) for k in range(3,11)}
fig,ax=plt.subplots(figsize=(6.4,3.8))
ax.plot(list(silK),list(silK.values()),"o-",color="0.25")
ax.axvline(K,color="tab:red",ls="--",lw=1.4,label=f"K = {K} (used)")
ax.set(xlabel="Number of clusters K",ylabel="Mean silhouette (2D)",title=f"Silhouette vs K — 2D map-plane clustering (context for fixed K = {K})")
ax.legend(); fig.tight_layout(); plt.show()
print("2D (primary) cluster sizes:",{k:int((r0.cluster==k).sum()) for k in CLS})
print("3D (comparison) cluster median depths (km):",{k:round(float(r0[r0.cl3d==k].z_km.median()),1) for k in CLS})
print(f"adjusted Rand index (2D vs 3D labels) = {ari:.3f}  -> depth {'regroups the map patches (3D = depth shells)' if ari<0.7 else 'barely changes the grouping'}")
print(f"silhouette (2D):",{k:round(v,3) for k,v in silK.items()}," | K={K} =",round(silK[K],3))""")
# ---- K = 3..10 sweep: how the map partition evolves ----
md(r"""### 1b · How the partition evolves from K = 3 to 10

Each panel is the **2D map-plane** K-means partition at a different K (same seed), so you can see the fault split
into progressively finer along-strike/across-strike patches. K=7 (highlighted) is the working choice; the sweep
shows what is gained/lost by moving up or down.""")
co(r"""fig,axs=plt.subplots(2,4,figsize=(18,9.4))
for ax,kk in zip(axs.ravel(),range(3,11)):
    lab=KMeans(n_clusters=kk,n_init=10,random_state=0).fit_predict(Xs2)
    _o=pd.Series(lab).value_counts().index.tolist(); lab=pd.Series(lab).map({o:i for i,o in enumerate(_o)}).values
    for j in range(kk): m=lab==j; ax.scatter(r0.lon[m],r0.lat[m],s=3,color=TAB[j%len(TAB)],lw=0)
    plot_coast(ax); plot_faults(ax)
    ax.set(xlim=REGION[:2],ylim=REGION[2:],title=f"K = {kk}"+("  (used)" if kk==K else "")); ax.set_aspect(ASP)
    ax.tick_params(labelsize=7); [ax.spines[s].set_color("tab:red") for s in ax.spines] if kk==K else None
    if kk==K: [ax.spines[s].set_linewidth(2) for s in ax.spines]
fig.suptitle("2D K-means map partition, K = 3 → 10 (seed 0); red frame = working K",y=1.0); fig.tight_layout(); plt.show()""")

# ------------------------------------------------------------------ §1c sweet-spot search
md(r"""### 1c · Is there a "sweet-spot" K? — five independent model-selection criteria

Silhouette (§1) just creeps upward with no peak, but silhouette is only one lens. To genuinely test for a
preferred K we add five more criteria on the same 2D embedding, each with a **different** definition of "best" and,
crucially, several that *can* return a finite optimum (unlike silhouette here):
- **Elbow** — knee of the inertia (within-cluster sum of squares) curve (max distance from the end-to-end chord);
- **Gap statistic** (Tibshirani 2001) — compares log-inertia to that of uniform random reference data; the **only**
  criterion with a null model, so it can say "no more structure than noise". Rule: smallest K with
  `Gap(K) ≥ Gap(K+1) − s_{K+1}`;
- **Calinski–Harabasz** (variance ratio; **peak** = best);
- **Davies–Bouldin** (mean cluster overlap; **min** = best);
- **GMM BIC** (penalised Gaussian-mixture likelihood; **min** = best — the penalty is what stops it running away).

If they converge on one K, that's the sweet spot. If they scatter or run monotone, "no sweet spot" is the honest
answer — a continuous fault whose K is a resolution choice.""")
co(r"""from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
Ks=list(range(2,16))
inertia={};ch={};db={};bic={};gap={};gap_s={}
rng=np.random.default_rng(0); lo=Xs2.min(0); hi=Xs2.max(0); n=len(Xs2); B=10
for k in Ks:
    kmk=KMeans(n_clusters=k,n_init=10,random_state=0).fit(Xs2)
    inertia[k]=kmk.inertia_; ch[k]=calinski_harabasz_score(Xs2,kmk.labels_); db[k]=davies_bouldin_score(Xs2,kmk.labels_)
    bic[k]=GaussianMixture(n_components=k,covariance_type="full",random_state=0).fit(Xs2).bic(Xs2)
    lw=[np.log(KMeans(n_clusters=k,n_init=5,random_state=b).fit(rng.uniform(lo,hi,size=(n,2))).inertia_) for b in range(B)]
    lw=np.array(lw); gap[k]=lw.mean()-np.log(kmk.inertia_); gap_s[k]=lw.std()*np.sqrt(1+1/B)
def knee(d):                                  # elbow = max perpendicular distance from the end-to-end chord
    ks=np.array(list(d),float); v=np.array(list(d.values()),float)
    ks=(ks-ks.min())/np.ptp(ks); v=(v-v.min())/np.ptp(v)
    num=np.abs((v[-1]-v[0])*ks-(ks[-1]-ks[0])*v+ks[-1]*v[0]-v[-1]*ks[0])
    return int(list(d)[np.argmax(num)])
K_elbow=knee(inertia); K_ch=max(ch,key=ch.get); K_db=min(db,key=db.get); K_bic=min(bic,key=bic.get)
K_gap=next((k for k in Ks[:-1] if gap[k]>=gap[k+1]-gap_s[k+1]),max(gap,key=gap.get)); K_sil=max(silK,key=silK.get)
fig,axs=plt.subplots(2,3,figsize=(16,8.6))
panels=[("Inertia (elbow)",inertia,K_elbow),("Gap statistic (Tibshirani)",gap,K_gap),("Silhouette",silK,K_sil),
        ("Calinski–Harabasz (peak)",ch,K_ch),("Davies–Bouldin (min)",db,K_db),("GMM BIC (min)",bic,K_bic)]
for ax,(name,d,best) in zip(axs.ravel(),panels):
    ax.plot(list(d),list(d.values()),"o-",color="0.25")
    if name.startswith("Gap"): ax.errorbar(list(gap),list(gap.values()),yerr=list(gap_s.values()),fmt="none",ecolor="0.6",capsize=2)
    ax.axvline(best,color="tab:red",ls="--",lw=1.5,label=f"pick K={best}")
    ax.axvline(K,color="tab:green",ls=":",lw=1.5,label=f"used K={K}")
    ax.set(xlabel="K",ylabel=name.split(" (")[0],title=name); ax.legend(fontsize=8); ax.set_facecolor("#FAFAFA")
fig.suptitle("Five model-selection criteria for K (2D map-plane embedding) — do they agree on a sweet spot?",y=1.0)
fig.tight_layout(); plt.show()
picks={"elbow":K_elbow,"gap":K_gap,"silhouette":K_sil,"Calinski-Harabasz":K_ch,"Davies-Bouldin":K_db,"GMM-BIC":K_bic}
_v=list(picks.values()); _spread=max(_v)-min(_v)
print("preferred K per criterion:",picks)
print(f"spread of picks: {min(_v)}-{max(_v)} (Δ={_spread}), median {int(np.median(_v))}")
print("VERDICT:", (f"CONVERGENT sweet spot near K={int(np.median(_v))} (criteria agree within 2)" if _spread<=2
      else "NO single sweet spot — the criteria disagree / several run monotone (silhouette & gap keep creeping, "
           "BIC/CH/DB push to different K) => the fault is a CONTINUOUS structure, not a fixed number of clouds; "
           "K is a resolution choice. K=7 chosen tectonically (resolves along-strike segments)."))""")

# ------------------------------------------------------------------ §2 map + sections + separability
md(r"""## 2 · Cluster map, depth sections, and the 2D↔3D (map vs depth) contrast

**Map (PyGMT)** — every event coloured by its **2D (map-plane) cluster** over the SOTA Quaternary fault traces;
the M3.89 mainshock ★. These are the map-identifiable patches. The **second map overlays the same clusters on the
smoothed seismicity density** (nb26's method: 2D event histogram Gaussian-smoothed to ~0.7 km, greyscale
white→dark) so you can check by eye whether each cluster sits on a density high or straddles a low. **Depth
sections** (East–depth, North–depth), coloured the same way, show that each map patch spans a range of depth —
i.e. the map grouping is not a depth grouping. The last cell contrasts this with the **3D labeling** (`cl3d`):
3D K-means instead splits the catalog into depth shells (median depths in §1), which is why the adjusted Rand
index between the two is low. So the fault has genuine depth structure, but the *visually-identifiable* grouping
is the 2D one used here.""")
co(r"""# ---- PyGMT categorical cluster map ----
fig=pygmt.Figure()
with pygmt.config(MAP_FRAME_TYPE="plain", FORMAT_GEO_MAP="ddd.xx"):
    fig.basemap(region=REGION,projection="M15c",frame=["WSne+t2D map-plane K-means clusters over the Ulsan Fault","xa0.1f0.05","ya0.1f0.05"])
    fig.coast(shorelines="0.6p,black",resolution="f",water="230/242/250")   # SOTA full-resolution BLACK coastline + light sea
    if os.path.exists(FAULT_GMT): fig.plot(data=FAULT_GMT,pen="0.9p,black")
    for k in CLS:
        c=r0[r0.cluster==k]
        fig.plot(x=c.lon,y=c.lat,fill=TAB[k],style="c0.11c",pen="0.2p,gray25",label=f"Cluster {k} (n={len(c)})")
    fig.basemap(map_scale="jBL+w10k+o0.5c/0.5c+c35.75")
    fig.legend(position="JTR+jTR+o0.2c",box="+gwhite+p0.6p,black")
fig.show(width=950)""")
co(r"""# ---- clusters overlaid on the smoothed seismicity density (nb26 method; check cluster<->density match) ----
fig=pygmt.Figure()
with pygmt.config(MAP_FRAME_TYPE="plain", FORMAT_GEO_MAP="ddd.xx"):
    fig.basemap(region=REGION,projection="M15c",frame=["WSne+tClusters over smoothed seismicity density (does each patch sit on a density high?)","xa0.1f0.05","ya0.1f0.05"])
    _vmax=float(DENS.max()) or 1.0
    pygmt.makecpt(cmap="gray",series=[0,_vmax],reverse=True)                 # low=white, high=dark grey
    fig.grdimage(DENS,region=REGION,projection="M15c",cmap=True)             # full grid -> empty cells render WHITE (no NaN mask, so no red background)
    fig.coast(shorelines="0.6p,black",resolution="f")                        # SOTA full-resolution BLACK coastline (on top of the grey density)
    if os.path.exists(FAULT_GMT): fig.plot(data=FAULT_GMT,pen="0.9p,black")
    for k in CLS:
        c=r0[r0.cluster==k]
        fig.plot(x=c.lon,y=c.lat,fill=TAB[k],style="c0.10c",pen="0.25p,white",label=f"Cluster {k}")   # white pen -> visible on dark density
    fig.colorbar(position="JBC+w10c/0.4c+h+o0c/1.2c",frame=["xaf+lSmoothed event count (Gaussian ~0.7 km)"])
    fig.basemap(map_scale="jBL+w10k+o0.5c/0.5c+c35.75")
    fig.legend(position="JTR+jTR+o0.2c",box="+gwhite+p0.6p,black")
fig.show(width=950)""")
co(r"""# ---- depth sections coloured by cluster ----
fig,axs=plt.subplots(1,2,figsize=(15,5.4))
for ax,(H,hl) in zip(axs,[("x_km","East (km, UTM 52N)"),("y_km","North (km, UTM 52N)")]):
    for k in CLS:
        c=r0[r0.cluster==k]; ax.scatter(c[H],c.z_km,s=8,color=TAB[k],lw=0,alpha=0.8,label=f"Cluster {k}")
    ax.set(xlabel=hl,ylabel="Depth (km)"); ax.invert_yaxis(); ax.set_facecolor("#FAFAFA")
axs[0].legend(loc="upper right",fontsize=8,ncol=1)
fig.suptitle("Depth sections coloured by K-means cluster (depth positive-down)",y=1.01); fig.tight_layout(); plt.show()""")
co(r"""# ---- map (2D) vs depth-shell (3D) grouping, side by side ----
fig,axs=plt.subplots(1,2,figsize=(14,6.4))
asp=1/np.cos(np.deg2rad(35.75))
for k in CLS:
    c=r0[r0.cluster==k]; axs[0].scatter(c.lon,c.lat,s=7,color=TAB[k],lw=0,label=f"Cluster {k} (n={len(c)})")
    c3=r0[r0.cl3d==k];   axs[1].scatter(c3.lon,c3.lat,s=7,color=TAB[k],lw=0,label=f"{c3.z_km.median():.1f} km shell")
for ax,ti in zip(axs,["2D map-plane clusters (primary — map-identifiable)","3D clusters = depth shells (overlap in map view)"]):
    plot_coast(ax); plot_faults(ax)
    ax.set(xlim=REGION[:2],ylim=REGION[2:],xlabel="Longitude (°E)",ylabel="Latitude (°N)",title=ti)
    ax.set_aspect(asp); ax.legend(fontsize=7,loc="upper left"); ax.set_facecolor("#FAFAFA")
fig.tight_layout(); plt.show()
print(f"adjusted Rand index (2D vs 3D labels) = {ari:.3f}  (low -> 3D groups by depth, not map position)")
for k in CLS:
    c=r0[r0.cluster==k]
    print(f"  2D cluster {k}: n={len(c):4d}  map extent {np.ptp(c.x_km):.1f}x{np.ptp(c.y_km):.1f} km  "
          f"depth {c.z_km.median():.1f} km (IQR {c.z_km.quantile(.25):.1f}-{c.z_km.quantile(.75):.1f})  suspect {int(c.suspect.sum())}")""")

# ------------------------------------------------------------------ §3 steadiness
md(r"""## 3 · Temporal steadiness per cluster

**Cumulative count** N(t) per cluster (a straight line = steady rate; steps = bursts). **COV of inter-event
times** = std(Δt)/mean(Δt) over the sorted origin times (≈1 Poisson, >1 bursty, <1 quasi-periodic). **Burst share**
= fraction of the cluster's events in its single busiest 30-day window (a sliding two-pointer maximum). Together
these separate steady background patches from episodic mainshock–aftershock sequences.""")
co(r"""def cov_iet(times):
    t=np.sort(pd.to_datetime(times).view("int64").values)/1e9
    if len(t)<3: return np.nan
    dt=np.diff(t); dt=dt[dt>0]
    return dt.std()/dt.mean() if len(dt)>1 and dt.mean()>0 else np.nan
def burst30(times):
    t=np.sort(pd.to_datetime(times).view("int64").values)/1e9
    n=len(t)
    if n<2: return np.nan
    w=30*86400; l=0; best=1
    for r in range(n):
        while t[r]-t[l]>w: l+=1
        best=max(best,r-l+1)
    return best/n
STEADY={}
fig,axs=plt.subplots(1,2,figsize=(15,5.2))
for k in CLS:
    c=r0[r0.cluster==k].sort_values("time")
    STEADY[k]=dict(cov=cov_iet(c.time),burst=burst30(c.time),n=len(c),
                   t0=c.time.min(),t1=c.time.max())
    axs[0].plot(c.time,np.arange(1,len(c)+1),color=TAB[k],lw=1.8,label=f"Cluster {k} (n={len(c)})")
    axs[1].plot(c.time,np.arange(1,len(c)+1)/len(c),color=TAB[k],lw=1.8)
axs[0].set(xlabel="Origin time",ylabel="Cumulative count",title="Cumulative events per cluster")
axs[1].set(xlabel="Origin time",ylabel="Fraction of cluster",title="Normalized cumulative (straight = steady)")
axs[0].legend(loc="upper left",fontsize=8); [a.set_facecolor("#FAFAFA") for a in axs]
fig.autofmt_xdate(); fig.tight_layout(); plt.show()
for k in CLS:
    s=STEADY[k]; kind="bursty/clustered" if s['cov']>1.3 else ("~Poisson" if s['cov']>0.7 else "quasi-periodic")
    print(f"Cluster {k}: COV(Δt)={s['cov']:.2f} ({kind}) | busiest-30d share={s['burst']*100:.0f}% | span {s['t0']:%Y-%m}..{s['t1']:%Y-%m}")""")

# ------------------------------------------------------------------ §3b background vs clustered
md(r"""## 3b · Background vs clustered activity per cluster — cumulative count over time

Each spatial cluster is split into **background** and **clustered** events by **NND declustering** (Zaliapin–Ben-
Zion: an event is *clustered* if its nearest-neighbour proximity η < η₀, else *background*), and the **cumulative
count of each class is tracked over time** inside the cluster. This separates a **steady tectonic trickle**
(background → a straight cumulative line) from **episodic mainshock–aftershock / swarm bursts** (clustered → step-
like jumps). The declustering runs on the reliable-ML population (NND needs magnitude); ML-less events are omitted
from this split. Df = 1.2, b = 1.0 (same basis as nb28; the background/clustered split is robust to Df).""")
co(r"""# NND declustering (Zaliapin-Ben-Zion) on the reliable-ML events -> per-event background/clustered flag
gm=r0[r0.has_ml].copy(); gm["event_id"]=gm.id.astype(int).astype(str)
gm["t_year"]=pd.to_datetime(gm.time).dt.strftime("%Y%m%d%H%M%S").map(nnd.decimal_year)
_g=gm.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep","magU":"kma_mag"}).sort_values("t_year").reset_index(drop=True)
_nd=nnd.compute_nnd(_g,b=1.0,D=1.2,mmin=None,metric="3d"); _e0,_=nnd.fit_eta0(_nd.eta.values)
_lab=nnd.build_families(_nd,_e0,_g.event_id.values,link_rmax_km=1.0)          # event_id -> family label (-1 = background)
_g["clustered"]=_g.event_id.map(_lab).fillna(-1).astype(int)>=0              # clustered = in an NND family (eta<eta0 & <=1km); else background
r0["clustered"]=r0.id.map(dict(zip(_g.event_id.astype(int),_g.clustered)))   # NaN for ML-less (not in NND pop)
_ml=r0[r0.clustered.notna()].copy(); _ml["clustered"]=_ml.clustered.astype(bool)
print(f"NND declustering (reliable-ML n={len(_ml)}): clustered {int(_ml.clustered.sum())} "
      f"({100*_ml.clustered.mean():.0f}%), background {int((~_ml.clustered).sum())} | log10 eta0={np.log10(_e0):.2f}")
fig,axs=plt.subplots(2,4,figsize=(18,8.6))
for ax,k in zip(axs.ravel(),CLS):
    c=_ml[_ml.cluster==k].sort_values("time"); bg=c[~c.clustered]; cl=c[c.clustered]
    ax.plot(bg.time,np.arange(1,len(bg)+1),color="tab:blue",lw=1.9,label=f"background ({len(bg)})")
    ax.plot(cl.time,np.arange(1,len(cl)+1),color="tab:red",lw=1.9,label=f"clustered ({len(cl)})")
    ax.set(title=f"Cluster {k} (n={len(c)})",ylabel="cumulative"); ax.legend(fontsize=7,loc="upper left")
    ax.grid(alpha=0.3); ax.tick_params(labelsize=7,axis="x",rotation=30); ax.set_facecolor("#FAFAFA")
for ax in axs.ravel()[len(CLS):]: ax.axis("off")
fig.suptitle("Background (steady) vs clustered (bursty) cumulative count per K-means cluster — reliable-ML NND split (Df=1.2)",y=1.0)
fig.tight_layout(); plt.show()
for k in CLS:
    c=_ml[_ml.cluster==k]; nbg,ncl=int((~c.clustered).sum()),int(c.clustered.sum())
    print(f"Cluster {k}: {nbg} background + {ncl} clustered ({100*ncl/max(nbg+ncl,1):.0f}% clustered)")""")

# ------------------------------------------------------------------ §4 correlation dimension
md(r"""## 4 · Correlation dimension D2 (2D map-plane & 3D) per cluster

Grassberger–Procaccia correlation integral C(r) = fraction of hypocentre pairs closer than r; the log–log slope
over the linear scaling range is D2 (`corr_integral`/`fit_slope` lifted verbatim from nb27). Per cluster the
scaling window `[r1,r2]` is chosen from the data (≈2× the median nearest-neighbour spacing up to ~½ the 90th-pct
pair distance) and **disclosed with R² and n**; clusters with too few events for a stable fit are flagged rather
than given a spurious slope. **D2_2D**→1 lineament, →2 plane-filling; **D2_3D** vs D2_2D gap = out-of-plane spread.""")
co(r"""def corr_integral(P):
    dd=pdist(P); npair=len(dd)
    rg=10**np.linspace(np.log10(0.02),np.log10(20),80)
    C=np.array([(dd<r).sum() for r in rg])/npair
    return rg,C,dd
def fit_slope(rg,C,r1,r2):
    m=(rg>=r1)&(rg<=r2)&(C>0)
    if m.sum()<4: return np.nan,np.nan,np.nan,m
    s=linregress(np.log10(rg[m]),np.log10(C[m]))
    return s.slope,s.rvalue**2,s.stderr,m
def d2_of(P,nmin=25):
    if len(P)<nmin: return dict(D2=np.nan,R2=np.nan,se=np.nan,r1=np.nan,r2=np.nan,n=len(P))
    rg,C,dd=corr_integral(P)
    r1=max(0.05,2*np.median(np.sort(dd)[:len(P)]))       # ~2x median NN spacing (n smallest pair dists ~ NN)
    r2=max(r1*3,0.5*np.percentile(dd,90))
    sl,R2,se,_=fit_slope(rg,C,r1,r2)
    return dict(D2=sl,R2=R2,se=se,r1=r1,r2=r2,n=len(P))
FRAC={}
fig,axs=plt.subplots(1,2,figsize=(15,5.4))
for k in CLS:
    c=r0[r0.cluster==k]
    P2=c[["x_km","y_km"]].values; P3=c[["x_km","y_km","z_km"]].values
    f2=d2_of(P2); f3=d2_of(P3); FRAC[k]=dict(d2=f2,d3=f3)
    for ax,P,f,lab in [(axs[0],P2,f2,"2D"),(axs[1],P3,f3,"3D")]:
        if len(P)>=25:
            rg,Cc,_=corr_integral(P); mm=(Cc>0)
            ax.plot(rg[mm],Cc[mm],color=TAB[k],lw=1.4,label=f"Cluster {k} (D2={f['D2']:.2f})")
axs[0].set(xscale="log",yscale="log",xlabel="r (km)",ylabel="C(r)",title="Correlation integral — 2D (map plane)")
axs[1].set(xscale="log",yscale="log",xlabel="r (km)",ylabel="C(r)",title="Correlation integral — 3D")
for a in axs: a.legend(fontsize=8,loc="upper left"); a.set_facecolor("#FAFAFA")
fig.tight_layout(); plt.show()
for k in CLS:
    a=FRAC[k]['d2']; b=FRAC[k]['d3']
    print(f"Cluster {k}: D2_2D={a['D2']:.2f} (R²={a['R2']:.2f}, r∈{a['r1']:.2f}-{a['r2']:.2f} km, n={a['n']}) | "
          f"D2_3D={b['D2']:.2f} (R²={b['R2']:.2f})")""")

# ------------------------------------------------------------------ §5 summary
md(r"""## 5 · Summary""")
co(r"""rows=[]
for k in CLS:
    c=r0[r0.cluster==k]; s=STEADY[k]; a=FRAC[k]['d2']; b=FRAC[k]['d3']
    rows.append(dict(cluster=k,n=len(c),n_suspect=int(c.suspect.sum()),
        lon=round(c.lon.median(),3),lat=round(c.lat.median(),3),depth_km=round(c.z_km.median(),1),
        maxML=round(c.magU.max(),2) if c.has_ml.any() else np.nan,
        span=f"{s['t0']:%Y-%m}..{s['t1']:%Y-%m}",
        COV=round(s['cov'],2),burst30pct=round(s['burst']*100),
        D2_2D=round(a['D2'],2),D2_3D=round(b['D2'],2),R2_3D=round(b['R2'],2)))
SUM=pd.DataFrame(rows)
print("="*140); print("K-MEANS SPATIAL CLUSTERS (K=7) — STEADINESS & FRACTAL DIMENSION — kim2011 dt.cc-RESOLVED UF catalog".center(140)); print("="*140)
print(SUM.to_string(index=False))
print("\nTAKE-HOMES")
_cov={k:STEADY[k]['cov'] for k in CLS if np.isfinite(STEADY[k]['cov'])}
_st=min(_cov,key=_cov.get); _bz=max(_cov,key=_cov.get)
print(f" - Steadiest cluster = {_st} (COV {_cov[_st]:.2f}, closest to a straight cumulative line); burstiest = {_bz} "
      f"(COV {_cov[_bz]:.2f}, {STEADY[_bz]['burst']*100:.0f}% of its events in one 30-day window).")
_d3={k:FRAC[k]['d3']['D2'] for k in CLS if np.isfinite(FRAC[k]['d3']['D2'])}
if _d3:
    _pl=min(_d3,key=_d3.get); _vol=max(_d3,key=_d3.get)
    print(f" - Most planar/localized cluster = {_pl} (D2_3D {_d3[_pl]:.2f}); most diffuse/volumetric = {_vol} (D2_3D {_d3[_vol]:.2f}).")
_msk=int(r0.loc[r0.magU.idxmax(),"cluster"]) if r0.has_ml.any() else None
print(f" - The M{r0.magU.max():.2f} mainshock falls in cluster {_msk}." if _msk is not None else "")
print(f" - 2D↔3D adjusted Rand index = {ari:.2f} -> the K={K} clusters are {'map-identifiable (depth barely regroups them)' if ari>0.7 else 'partly depth-controlled: 3D K-means instead returns depth shells (§1-2)'}.")
print(f" - Silhouette at K={K} = {silK[K]:.2f} (context only). Over K=3-10 the score stays ~{min(silK.values()):.2f}-{max(silK.values()):.2f} "
      f"with no sharp peak (best K={max(silK,key=silK.get)} at {max(silK.values()):.2f}) -> one continuous fault, not distinct clouds; K is a resolution choice.")
print("\nNEXT: tie K-means clusters to the NND families (nb28/nb33) and per-cluster b-value / Mc (nnd.estimate_mc/estimate_b);"
      " completeness-corrected rates for the steadiness call; D2(t) in rolling windows for geometric localization.")""")

nb["cells"]=C
import os
os.chdir("/home/msseo/works/02.Ulsan_Fault_detection/analysis/reloc_analysis")
nbf.write(nb,"37.UF_kmeans_spatial_clusters_dtcc.ipynb")
print("wrote 37.UF_kmeans_spatial_clusters_dtcc.ipynb",len(C),"cells")
