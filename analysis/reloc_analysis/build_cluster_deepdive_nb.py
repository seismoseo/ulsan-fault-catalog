#!/usr/bin/env python
"""Generate 28.UF_cluster_deepdive.ipynb — Track-2 deep-dive into the LARGEST NND clusters of the
dt.cc-relocated Ulsan-Fault catalog: topology (parent->child families incl. the mainshock parent), spatial
structure (map + depth sections + best-fit-plane geometry), and temporal structure (M-t, cumulative moment,
Omori decay, migration) -> foreshock-mainshock-aftershock vs swarm classification.

POPULATION / NND (identical to the corrected Track-1 NND, Zhigang Fig 4): full relocated catalog with reliable
ML (n_used>=3), 3-D, b=1.0, data-driven **Df=1.2**. Families via nnd.build_families (connected components of the
sub-eta0 parent->child graph) — the family INCLUDES its parent even when the parent's own nearest-neighbour link
is background. spatial_merge gives the fault-patch grouping. Deep-dive on the TOP 5 raw families (build_families
labels 0..K by DESCENDING size, so top 10 = Cluster in {0..9}).

FIRST PASS = characterize only. SVD per-cluster relocation and relative-ML recovery of the ML-less small events
are DEFERRED (Non-goals in the plan); here we only COUNT/overlay the ML-less events (n_used<3) that fall inside
each cluster's space-time envelope, to size the gap.

Disclosed choices: Df=1.2/b=1.0 (Track-1), **sub-day decimal time** (day-resolution force-links same-day events
at eta=0), **link_rmax=1 km** (compact UF; module default 10 km over-chains distant events), classification thresholds (moment-concentration +
timing + migration) stated inline. Runs in base (numpy/scipy/pandas/matplotlib/pygmt)."""
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# Ulsan-Fault cluster deep-dive — the largest NND families (Track 2)

Track 1 (nb26 / Zhigang / nb27) split the dt.cc-relocated catalog into **background** vs **clustered** for rate
& density statistics. Track 2 opens the **individual clusters**: their **topology** (parent-child family tree,
including the mainshock parent), **spatial** structure (fault-plane geometry) and **temporal** structure
(foreshock-mainshock-aftershock vs swarm). This first pass **characterizes** the 10 largest families; dedicated
per-cluster SVD relocation and relative-amplitude ML recovery of the small ML-less members are deferred.

**NND basis (identical to the corrected Track-1 run):** full relocated population with reliable ML
(`n_used >= 3`), 3-D, **b = 1.0**, data-driven **Df = 1.2**. Families = connected components of the sub-η₀
parent→child graph (`nnd.build_families`, link cap **1 km**, sub-day time); the family **includes its parent**. `spatial_merge`
(≤5 km / ≤5 km depth) gives the fault-patch grouping.""")

# ------------------------------------------------------------------ §0 setup + families
co(r"""import warnings; warnings.filterwarnings("ignore")
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.collections import LineCollection
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":130,"axes.grid":True,"grid.alpha":0.3,"font.size":10,
                     "legend.framealpha":1,"legend.edgecolor":"black","legend.facecolor":"white"})
sys.path.insert(0,"/home/msseo/works/16.kma_absolute_location"); from kma_absolute_location import nnd, clustering

KG="/home/msseo/works/02.Ulsan_Fault_detection/KS_KG"
RELOC=f"{KG}/local_magnitudes/catalog_ml_heo_ufonly_reloc.csv"
FAULTS=f"{KG}/HypoInv/faults_lonlat.gmt"
COAST=f"{KG}/reloc_analysis/coastline_lonlat.gmt"
UF=(129.25,129.55,35.60,35.90)
DF_UF=1.2; B_NND=1.0; MC=1.2; LINKR=1.0   # family link-distance cap (km) — 1 km for the compact Ulsan Fault (no event triggers >1 km; cascading short links still join genuine migration; robust: 2 km gave near-identical families). Module default 10 km is whole-KMA.

# fault traces (GMT multiseg, already lon lat) -> list of (lon,lat) arrays for matplotlib
def load_faults(path):
    segs=[]; cur=[]
    if os.path.exists(path):
        for ln in open(path):
            if ln.startswith((">","#")):
                if len(cur)>1: segs.append(np.array(cur));
                cur=[]; continue
            p=ln.split()
            if len(p)>=2:
                try: cur.append((float(p[0]),float(p[1])))
                except ValueError: pass
        if len(cur)>1: segs.append(np.array(cur))
    return segs
FSEG=load_faults(FAULTS); CSEG=load_faults(COAST)     # same GMT multiseg (lon lat) parser works for coastline
def plot_faults(ax):
    for s in FSEG: ax.plot(s[:,0],s[:,1],color="0.35",lw=0.8,zorder=1)
def plot_coast(ax):
    for s in CSEG: ax.plot(s[:,0],s[:,1],color="steelblue",lw=0.7,zorder=1)

# ---- catalog: reliable-ML population (NND) + the ML-less events (n_used<3) for the completeness overlay ----
rl=pd.read_csv(RELOC); rl["event_time"]=pd.to_datetime(rl.event_time,format="ISO8601",utc=True,errors="coerce")
rl=rl[~rl.event_idx.isin(set(pd.read_csv(f"{KG}/local_magnitudes/blast_event_idx_deblast.csv").event_idx.dropna().astype(int)))].copy()  # DE-BLAST: drop quarry-blast events (nb22 §7)
rl=rl.dropna(subset=["lat","lon","depth","event_time","ml_ufcorr_reloc"]).copy()
pop=rl[rl.n_used>=3].copy()                                   # NND population (reliable ML)
mless=rl[rl.n_used<3].copy()                                  # ML-less small events (0<n_used<3) -> overlay only
print(f"reloc catalog with ML: {len(rl)} | NND population (n_used>=3): {len(pop)} | ML-less (n_used<3): {len(mless)}")
print(f"  (a further ~186 relocated events have NO amplitude reading at all -> not in this csv; noted, not shown)")

# ---- build the NND families (Df=1.2, 3D, b=1.0) ----
g=pop.copy()
g["event_id"]=g.event_idx.astype(int).astype(str)
g["t_year"]=g.event_time.dt.strftime("%Y%m%d%H%M%S").map(nnd.decimal_year)  # CANONICAL nnd.decimal_year (exact year length, second precision; day-resolution force-links same-day events at eta=0)
g=g.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep","ml_ufcorr_reloc":"kma_mag"})
g=g.sort_values("t_year").reset_index(drop=True)
nd=nnd.compute_nnd(g,b=B_NND,D=DF_UF,mmin=None,metric="3d")
e0,info=nnd.fit_eta0(nd.eta.values,method="gmm"); LE0=float(np.log10(e0))
labels=nnd.build_families(nd,e0,g.event_id.values,link_rmax_km=LINKR)
merged=nnd.spatial_merge(g,labels,dmax_km=5.0,dz_km=5.0)
g["Cluster"]=g.event_id.map(labels).fillna(-1).astype(int)
g["Cluster_merged"]=g.event_id.map(merged).fillna(-1).astype(int)
n_fam=int(g.Cluster.max())+1; n_clu=int((g.Cluster>=0).sum())
print(f"3D NND (Df={DF_UF}, b={B_NND}): log10 eta0={LE0:.2f} | clustered {n_clu}/{len(g)} ({100*n_clu/len(g):.0f}%) "
      f"| raw families {n_fam} | merged patches {int(g.Cluster_merged.max())+1}")""")

md(r"""## 0 · Family table — the largest clusters

`build_families` labels families 0..K−1 by **descending size**, so the **top 10 = Cluster 0–9**. Per family:
member count, mainshock (largest-ML member) magnitude & time, duration, horizontal extent, depth range. We
verify each top family contains its mainshock (the parent).""")
co(r"""M0=lambda m: 10.0**(1.5*np.asarray(m,float)+9.1)          # seismic moment (N m), ML as Mw proxy
def enu_m(sub):                                                # local ENU metres (UTM 52N) for extent/PCA
    u,_=clustering.to_utm(sub); return u["x_m"].to_numpy(), u["y_m"].to_numpy(), u["depth_m"].to_numpy()
def fam_summary(gg):
    rows=[]
    for k in sorted(x for x in gg.Cluster.unique() if x>=0):
        c=gg[gg.Cluster==k].sort_values("t_year"); ms=c.loc[c.kma_mag.idxmax()]
        x,y,z=enu_m(c); ext=np.hypot(x-x.mean(),y-y.mean()).max()*2/1000.0
        dur=(c.event_time.max()-c.event_time.min()).total_seconds()/86400.0
        rows.append(dict(Cluster=k,n=len(c),mainshock_M=round(float(ms.kma_mag),2),
                         mainshock_time=str(ms.event_time)[:19],dur_days=round(dur,1),
                         extent_km=round(ext,2),dep_min=round(c.svi_dep.min(),1),dep_max=round(c.svi_dep.max(),1),
                         lon=round(c.svi_lon.mean(),4),lat=round(c.svi_lat.mean(),4),
                         merged=int(c.Cluster_merged.mode().iloc[0])))
    return pd.DataFrame(rows)
FAM=fam_summary(g)
print(f"{n_fam} raw families; showing the 15 largest:"); print(FAM.head(15).to_string(index=False))
TOP=list(FAM.sort_values("n",ascending=False).Cluster.head(10))    # top 10 by size (== 0..9)
print(f"\nTOP 5 families for the deep-dive: Cluster {TOP}")
for k in TOP:
    c=g[g.Cluster==k]; assert c.loc[c.kma_mag.idxmax(),"Cluster"]==k       # mainshock IS in the family
print("verified: each top-10 family contains its largest-ML member (the mainshock/parent).")""")

# ------------------------------------------------------------------ §0b R-T scatter
md(r"""### 0b · Nearest-neighbour (log T, log R) scatter — clustered vs background

The Zaliapin–Ben-Zion plane: each event at its nearest-neighbour **rescaled time** log₁₀T (x) and **rescaled
distance** log₁₀R (y). Now coloured to reflect the **actual 1 km-cap clustering**: **red = a real family child**
(η<η₀ *and* link ≤ 1 km), **orange rings = below the diagonal but CUT by the 1 km cap** (η<η₀ yet link > 1 km,
so it falls to background), **grey = η≥η₀**. So the red points are exactly the offspring that end up clustered
after the cap. Same (log T, log R) domain as the nb26/Zhigang density plots, **x fixed to −8…2** (y −6…4,
equal spans → η₀ at 45°). Population = the full relocated set with reliable ML — **dt.cc AND dt.ct**.

*Two things this makes visible:* (i) the **1 km cap** rescues a handful of below-diagonal points (orange) that
are close in rescaled η but physically >1 km away — temporal coincidences the cap correctly rejects; (ii) each
cluster's **mainshock/root** is NOT red here — it sits ABOVE the line (its own nearest earlier neighbour is far),
and `build_families` attaches it to its offspring's cluster (parent-inclusion). So the §2 family sizes exceed
the red count; the §2 spanning trees show the full parent-inclusive families.""")
co(r"""_XL=(-8.0,2.0); _YL=(-6.0,4.0); le0=np.log10(e0)
# reflect the ACTUAL 1 km-cap clustering: a below-diagonal (eta<eta0) event is a real family CHILD only if its
# link also survives R<=LINKR; those with R>LINKR are cut by the cap and fall to background.
clm=nd[(nd.eta<e0)&(nd.R_km<=LINKR)]; cut=nd[(nd.eta<e0)&(nd.R_km>LINKR)]; bgm=nd[nd.eta>=e0]
fig,ax=plt.subplots(figsize=(7.0,6.6))
ax.scatter(bgm.logT,bgm.logR,s=7,c="0.62",lw=0,alpha=0.5,label=f"background, η≥η₀ ({len(bgm)})",zorder=2)
ax.scatter(cut.logT,cut.logR,s=26,c="none",edgecolor="tab:orange",lw=0.9,marker="o",label=f"η<η₀ but CUT by 1 km cap ({len(cut)})",zorder=3)
ax.scatter(clm.logT,clm.logR,s=11,c="tab:red",lw=0,alpha=0.8,label=f"clustered child (η<η₀ & link≤1 km) ({len(clm)})",zorder=4)
ax.plot(_XL,le0-np.array(_XL),"--",lw=1.6,color="k",label=fr"$\eta_0$ (log$_{{10}}$={le0:.2f})",zorder=5)
ax.set(xlim=_XL,ylim=_YL,xlabel=r"Rescaled time  log$_{10}T$",ylabel=r"Rescaled distance  log$_{10}R$",
       title=f"Nearest-neighbour (T, R) — clustered children after 1 km cap (Df={DF_UF}, b={B_NND})")
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="lower left",fontsize=8.5); plt.show()
print(f"NND population {len(g)} = dt.cc {int(g.is_dtcc.sum())} + dt.ct {int((~g.is_dtcc).sum())} | nn-links {len(nd)} | "
      f"clustered children (η<η₀ & ≤1 km) {len(clm)} | η<η₀ but cut by 1 km cap {len(cut)} | η≥η₀ {len(bgm)} | log10 eta0={le0:.2f}")""")

# ------------------------------------------------------------------ §1 overview
md(r"""## 1 · Overview — where the families sit

All clustered events coloured by family (top-10 highlighted), background events light grey, with the SOTA fault
traces; companion depth sections. The size histogram shows how member count is distributed across families, and
how many raw families collapse into each spatially-merged fault patch.""")
co(r"""fig=plt.figure(figsize=(15,5.2))
axm=fig.add_axes([0.05,0.1,0.42,0.82]); axL=fig.add_axes([0.52,0.57,0.44,0.35]); axH=fig.add_axes([0.52,0.1,0.20,0.34]); axS=fig.add_axes([0.78,0.1,0.18,0.34])
bg=g[g.Cluster<0]; cl=g[g.Cluster>=0]
axm.scatter(bg.svi_lon,bg.svi_lat,s=5,c="0.8",lw=0,label=f"background ({len(bg)})",zorder=2)
axm.scatter(cl.svi_lon,cl.svi_lat,s=8,c="0.55",lw=0,zorder=3)
_cols=plt.cm.tab10(np.linspace(0,1,10))
for i,k in enumerate(TOP):
    c=g[g.Cluster==k]; axm.scatter(c.svi_lon,c.svi_lat,s=22,color=_cols[i],lw=0.2,edgecolor="k",zorder=5,label=f"Cluster {k} (n={len(c)})")
    ms=c.loc[c.kma_mag.idxmax()]; axm.scatter(ms.svi_lon,ms.svi_lat,s=180,marker="*",color=_cols[i],edgecolor="k",lw=0.6,zorder=6)
plot_coast(axm); plot_faults(axm); axm.set(xlim=UF[:2],ylim=UF[2:],xlabel="Longitude (°E)",ylabel="Latitude (°N)",title=f"NND families (Df={DF_UF}) — top 10 highlighted, ★=mainshock")
axm.legend(fontsize=7,loc="upper left",ncol=1); axm.set_aspect(1/np.cos(np.deg2rad(35.75)))
# depth section (lon-depth)
axL.scatter(bg.svi_lon,bg.svi_dep,s=4,c="0.8",lw=0); axL.scatter(cl.svi_lon,cl.svi_dep,s=5,c="0.55",lw=0)
for i,k in enumerate(TOP):
    c=g[g.Cluster==k]; axL.scatter(c.svi_lon,c.svi_dep,s=14,color=_cols[i],lw=0)
axL.set(xlim=UF[:2],ylim=(20,0),xlabel="Longitude (°E)",ylabel="Depth (km)",title="Depth section (top-10 coloured)")
# size histogram + raw->merged
sizes=FAM.n.values
axH.hist(sizes,bins=np.logspace(np.log10(2),np.log10(max(sizes)+1),18),color="steelblue",ec="w")
axH.set(xscale="log",xlabel="family size (members)",ylabel="# families",title=f"{n_fam} families")
mg=g[g.Cluster>=0].groupby("Cluster_merged").agg(n_raw=("Cluster","nunique"),n_ev=("event_id","size"))
axS.scatter(mg.n_raw,mg.n_ev,s=18,color="tab:red",lw=0); axS.set(xlabel="raw families in patch",ylabel="events in patch",title="raw→merged")
plt.show()
print(f"largest family n={sizes.max()} | median family n={int(np.median(sizes))} | families with n>=20: {int((sizes>=20).sum())}")
print(f"spatially-merged patches: {int(g.Cluster_merged.max())+1} (largest patch merges {int(mg.n_raw.max())} raw families)")""")

# ------------------------------------------------------------------ §2 per-cluster deep-dive
md(r"""## 2 · Per-cluster deep-dive (top 10)

For each of the 10 largest families: **topology** (parent→child tree + on-map links + the **Zaliapin–Ben-Zion
2013 average leaf depth ⟨d⟩** — a size-aware tree metric: low ⟨d⟩≈1–2 = **burst-like** star of aftershocks,
high/deep = **swarm-like** progressive chaining; independent of the moment-based class below), **spatial** structure
(map coloured by time + depth sections + a 3-D-PCA **shape** — the three principal widths L1≥L2≥L3, with
strike/dip reported ONLY for a genuine thin **sheet** (L2≥0.4·L1 & L3≤0.35·L2); a **linear** streak gives a
trend but no dip; a **blob** gives neither — refusing a plausible-but-fake plane on non-planar clouds),
**temporal** structure (M–t, cumulative count & moment, inter-event times) and a **classification**
(mainshock–aftershock / foreshock–mainshock–aftershock / swarm) from explicit metrics:
- **mainshock moment share** `f_M0` = M₀(largest) / ΣM₀;
- **timing** `f_after` = fraction of members after the mainshock; `f_before` before it;
- **migration** = slope of along-strike position vs time (km/day) and its correlation `r`.

Rule (disclosed, exactly as implemented in `classify()`): `f_M0 < 0.4` → swarm; `f_M0 ≥ 0.6 & f_before ≥ 0.15`
→ foreshock–mainshock–aftershock; `f_M0 ≥ 0.6` (else) → mainshock–aftershock; otherwise mixed/complex. Timing
(`f_after`) and migration are reported as descriptive metrics but do not enter the rule. A grey overlay marks relocated
**ML-less** events (`n_used<3`) inside the family's space-time envelope — the deferred relative-ML targets.""")
co(r"""from scipy.stats import linregress
def plane_geom(x,y,z):
    # 3-D PCA shape of a hypocentre cloud (x=E,y=N,z=Down m). Returns dict: the 3 principal FULL-WIDTHS
    # L1>=L2>=L3 (km) + a SHAPE-GATED strike/dip. A plane (strike+dip) is only defined for a thin SHEET
    # (L2>=0.4*L1 & L3<=0.35*L2); a LINEAR streak gives a strike (trend) but NO dip; a BLOB gives neither.
    # The normal is the least-resolved axis, so strike/dip on a non-sheet cloud is a plausible-but-fake number.
    P=np.c_[x-x.mean(),y-y.mean(),z-z.mean()]
    if len(P)<4: return dict(shape="n/a",strike=None,dip=None,L1=np.nan,L2=np.nan,L3=np.nan)
    S,Vt=np.linalg.svd(P,full_matrices=False)[1:]; sd=S/np.sqrt(len(P)-1)     # std along each PC (m)
    L1,L2,L3=(2*sd/1000)                                                       # full width along each PC (km)
    if L2>=0.4*L1 and L3<=0.35*L2:                                             # thin SHEET -> strike + dip
        n=Vt[2]; n=-n if n[2]<0 else n
        dip=int(round(np.degrees(np.arccos(abs(n[2])/np.linalg.norm(n)))))
        strike=int(round((np.degrees(np.arctan2(n[0],n[1]))+90)%180)); shape="planar"
    elif L2<0.4*L1:                                                            # LINEAR -> trend only, dip undefined
        v=Vt[0]; strike=int(round(np.degrees(np.arctan2(v[0],v[1]))%180)); dip=None; shape="linear"
    else:                                                                      # near-isotropic BLOB
        strike=dip=None; shape="blob"
    return dict(shape=shape,strike=strike,dip=dip,L1=round(L1,2),L2=round(L2,2),L3=round(L3,2))
def migration(sub,x,y):                                         # along principal-strike position vs time
    P=np.c_[x-x.mean(),y-y.mean()];
    if len(P)<3: return np.nan,np.nan
    _,_,Vt=np.linalg.svd(P,full_matrices=False); s=P@Vt[0]/1000.0    # km along PC1
    t=(sub.event_time-sub.event_time.min()).dt.total_seconds().values/86400.0
    if np.ptp(t)<1e-6: return np.nan,np.nan
    lr=linregress(t,s); return round(lr.slope,4),round(lr.rvalue,2)
def omori_p(t_after):                                           # crude log-log rate slope after mainshock
    t=np.sort(t_after[t_after>0])
    if len(t)<8: return np.nan
    edges=np.logspace(np.log10(max(t.min(),1e-3)),np.log10(t.max()),8); ctr=np.sqrt(edges[:-1]*edges[1:])
    cnt,_=np.histogram(t,bins=edges); rate=cnt/np.diff(edges); ok=rate>0
    if ok.sum()<3: return np.nan
    return round(-linregress(np.log10(ctr[ok]),np.log10(rate[ok])).slope,2)

def classify(f_M0,f_after,f_before):
    if f_M0<0.4: return "swarm"                                 # no event dominates the moment
    if f_M0>=0.6 and f_before>=0.15: return "foreshock–mainshock–aftershock"   # dominant + notable foreshocks
    if f_M0>=0.6: return "mainshock–aftershock"                 # dominant, few foreshocks
    return "mixed/complex"
from collections import deque
def build_tree(members):
    # cluster TREE: each event -> its parent (nearest earlier nbr, kept if eta<eta0 & R<=LINKR). Root = member
    # whose parent is outside the family. Returns children{p:[ch]}, parent{ch:p}, depth{node:#links-from-root}, roots.
    ids=set(members); lk=nd[(nd.eta<e0)&(nd.R_km<=LINKR)&(nd.event_id.isin(ids))&(nd.parent_id.isin(ids))]
    children={}; parent={}
    for p,ch in zip(lk.parent_id,lk.event_id): children.setdefault(p,[]).append(ch); parent[ch]=p
    depth={}; roots=[e for e in members if e not in parent]
    for r in roots:
        depth[r]=0; q=deque([r])
        while q:
            uu=q.popleft()
            for v in children.get(uu,[]):
                if v not in depth: depth[v]=depth[uu]+1; q.append(v)
    return children,parent,depth,roots
def leaf_depth(members):
    # Zaliapin & Ben-Zion (2013) AVERAGE LEAF DEPTH <d>: low (~1, star) = BURST-like (aftershocks share a parent);
    # high (deep/chained) = SWARM-like (progressive triggering). Also #generations = max tree depth.
    children,parent,depth,roots=build_tree(members)
    leaves=[e for e in members if e not in children]
    ald=float(np.mean([depth.get(e,0) for e in leaves])) if leaves else np.nan
    return round(ald,2), (max(depth.values()) if depth else 0)
def tree_xpos(members,tsort):
    # tidy-tree horizontal positions (leaf order; children ordered by time within the layout), y = generation.
    children,parent,depth,roots=build_tree(members); xpos={}; cnt=[0]
    def rec(node):
        ch=sorted(children.get(node,[]),key=lambda e: tsort.get(e,0.0))
        if not ch: xpos[node]=cnt[0]; cnt[0]+=1
        else:
            for cc in ch: rec(cc)
            xpos[node]=float(np.mean([xpos[cc] for cc in ch]))
    for r in sorted(roots,key=lambda e: tsort.get(e,0.0)): rec(r)
    return xpos,depth,children

SUMROWS=[]
for i,k in enumerate(TOP):
    c=g[g.Cluster==k].sort_values("event_time").reset_index(drop=True)
    ms=c.loc[c.kma_mag.idxmax()]; t0=c.event_time.min()
    days=(c.event_time-t0).dt.total_seconds().values/86400.0; msday=(ms.event_time-t0).total_seconds()/86400.0
    x,y,z=enu_m(c)
    m0=M0(c.kma_mag.values); f_M0=float(m0[c.kma_mag.idxmax()]/m0.sum())
    f_after=float((c.event_time>ms.event_time).mean()); f_before=float((c.event_time<ms.event_time).mean())
    gm=plane_geom(x,y,z); shape=gm["shape"]; strike=gm["strike"]; dip=gm["dip"]; L1,L2,L3=gm["L1"],gm["L2"],gm["L3"]
    vmig,rmig=migration(c,x,y); pval=omori_p(days-msday); ald,ngen=leaf_depth(list(c.event_id))
    topo="burst-like" if ald<=2.5 else "swarm-like" if ald>=5 else "intermediate"    # ZBZ avg-leaf-depth topology
    gstr=(f"planar strike {strike}°/dip {dip}°" if shape=="planar" else
          f"linear trend {strike}° (dip undefined)" if shape=="linear" else f"{shape} (no plane)")
    cls=classify(f_M0,f_after,f_before)
    # ML-less events inside the family space-time envelope (buffer 1 km + 20% of duration)
    dur=max(days.max(),1e-6); lon0,lon1=c.svi_lon.min(),c.svi_lon.max(); lat0,lat1=c.svi_lat.min(),c.svi_lat.max()
    bufd=0.01                                                    # ~1 km lon/lat buffer
    tt=(mless.event_time-t0).dt.total_seconds()/86400.0
    sel=(mless.lon.between(lon0-bufd,lon1+bufd) & mless.lat.between(lat0-bufd,lat1+bufd) & (tt>=-0.2*dur) & (tt<=1.2*dur))
    inml=mless[sel]; n_mless=len(inml)

    # ---- figure: 3 wide panels (topology | map+links | depth) + temporal row ----
    fig=plt.figure(figsize=(15,7.6)); gs=fig.add_gridspec(2,3,height_ratios=[1.15,1.0],hspace=0.32,wspace=0.28)
    col=plt.cm.viridis(days/dur)
    # (a) SPANNING TREE — parent->child by generation (y=tree depth), NOT time-ordered; node colour=time
    axT=fig.add_subplot(gs[0,0])
    idset=set(c.event_id); tsort=dict(zip(c.event_id,days))
    xpos,depthd,children=tree_xpos(list(c.event_id),tsort)
    for p,chs in children.items():
        for ch in chs: axT.plot([xpos[p],xpos[ch]],[depthd[p],depthd[ch]],color="0.55",lw=0.7,zorder=1)
    _xs=np.array([xpos[e] for e in c.event_id]); _ys=np.array([depthd[e] for e in c.event_id])
    axT.scatter(_xs,_ys,s=18+30*(c.kma_mag-c.kma_mag.min()),c=days,cmap="viridis",edgecolor="k",lw=0.2,zorder=3)
    _mid=ms.event_id; axT.scatter(xpos[_mid],depthd[_mid],s=280,marker="*",color="red",edgecolor="k",lw=0.6,zorder=4)
    axT.invert_yaxis(); axT.set_xticks([])                       # root at top, generations downward
    axT.set(ylabel="generation (tree depth)",title=f"(a) Cluster {k} spanning tree — ⟨leaf depth⟩={ald}, {ngen} gen ({topo})")
    # (b) map coloured by time + parent->child arrows
    axM=fig.add_subplot(gs[0,1]); plot_coast(axM); plot_faults(axM)
    if n_mless: axM.scatter(inml.lon,inml.lat,s=10,c="0.7",marker="x",lw=0.6,zorder=2,label=f"ML-less ({n_mless})")
    lon_by=dict(zip(c.event_id,c.svi_lon)); lat_by=dict(zip(c.event_id,c.svi_lat))
    mseg=[[(lon_by[p],lat_by[p]),(lon_by[ch],lat_by[ch])] for p,chs in children.items() for ch in chs]   # same tree edges as panel (a)
    if mseg: axM.add_collection(LineCollection(mseg,colors="0.6",lw=0.5,zorder=2))
    sc=axM.scatter(c.svi_lon,c.svi_lat,s=18+30*(c.kma_mag-c.kma_mag.min()),c=days,cmap="viridis",zorder=3,edgecolor="k",lw=0.2)
    axM.scatter(ms.svi_lon,ms.svi_lat,s=260,marker="*",color="red",edgecolor="k",lw=0.6,zorder=4)
    _bx=0.2*max(lon1-lon0,0.01)+0.005; _by=0.2*max(lat1-lat0,0.01)+0.005     # tight cluster extent (+buffer), NOT whole region
    axM.set(xlim=(lon0-_bx,lon1+_bx),ylim=(lat0-_by,lat1+_by),xlabel="Longitude (°E)",ylabel="Latitude (°N)",
            title=f"(b) map — colour=time, {gstr}")
    axM.set_aspect(1/np.cos(np.deg2rad(35.75))); cb=fig.colorbar(sc,ax=axM,fraction=0.046,pad=0.02); cb.set_label("days")
    if n_mless: axM.legend(fontsize=7,loc="upper left")
    # (c) depth section along PC1 (strike)
    axD=fig.add_subplot(gs[0,2]); P2=np.c_[x-x.mean(),y-y.mean()]; s1=(P2@np.linalg.svd(P2,full_matrices=False)[2][0])/1000.0
    axD.scatter(s1,c.svi_dep,s=18,c=days,cmap="viridis",edgecolor="k",lw=0.2)
    axD.scatter(s1[c.kma_mag.idxmax()],ms.svi_dep,s=260,marker="*",color="red",edgecolor="k",lw=0.6)
    axD.set(xlabel="along-PC1 (km)",ylabel="Depth (km)",title=f"(c) PC1 section — L1×L2×L3={L1}×{L2}×{L3} km ({shape})",ylim=(c.svi_dep.max()+0.5,c.svi_dep.min()-0.5))
    # (d) M-t stem   (e) cumulative N & moment   (f) inter-event time
    axm2=fig.add_subplot(gs[1,0]); axm2.vlines(days,c.kma_mag.min()-0.3,c.kma_mag,color="0.6",lw=0.6); axm2.scatter(days,c.kma_mag,s=14,c=col,zorder=3)
    axm2.axvline(msday,color="red",ls="--",lw=1); axm2.set(xlabel="days",ylabel="M$_L$",title="(d) M–t")
    axc=fig.add_subplot(gs[1,1]); axc.plot(days,np.arange(1,len(c)+1),color="tab:blue",label="cum. N")
    axc.set(xlabel="days",ylabel="cumulative N"); axc.axvline(msday,color="red",ls="--",lw=1)
    ax2=axc.twinx(); ax2.plot(days,np.cumsum(m0[np.argsort(days)])/m0.sum(),color="tab:red",label="cum. M₀ frac"); ax2.set_ylabel("cum. moment frac",color="tab:red")
    axc.set_title(f"(e) cumulative — f_M0={f_M0:.2f}")
    axi=fig.add_subplot(gs[1,2]); dt=np.diff(np.sort(days))*24; dt=dt[dt>0]
    if len(dt): axi.hist(np.log10(dt),bins=20,color="slategray",ec="w")
    axi.set(xlabel="log₁₀ inter-event time (hr)",ylabel="count",title=f"(f) inter-event — {cls}")
    fig.suptitle(f"Cluster {k}:  n={len(c)}  mainshock M{ms.kma_mag:.2f} ({str(ms.event_time)[:10]})  dur {dur:.1f} d  size {L1}×{L2}×{L3} km ({shape})  →  {cls.upper()}",y=1.0,fontsize=12)
    plt.show()
    print(f"Cluster {k}: n={len(c)} | mainshock M{ms.kma_mag:.2f} | f_M0={f_M0:.2f} f_after={f_after:.2f} f_before={f_before:.2f} "
          f"| ⟨leaf depth⟩={ald} ({ngen} gen, {topo}) | L1xL2xL3={L1}x{L2}x{L3} km {gstr} | migration {vmig} km/d (r={rmig}) | Omori-p {pval} | ML-less {n_mless} | CLASS: {cls}")
    SUMROWS.append(dict(Cluster=k,n=len(c),n_MLless=n_mless,mainshock_M=round(float(ms.kma_mag),2),
                        mainshock_date=str(ms.event_time)[:10],dur_days=round(dur,1),
                        avg_leaf_depth=ald,generations=ngen,topology=topo,
                        L1_km=L1,L2_km=L2,L3_km=L3,shape=shape,strike=strike,dip=dip,
                        dep_min=round(c.svi_dep.min(),1),dep_max=round(c.svi_dep.max(),1),f_M0=round(f_M0,2),
                        f_after=round(f_after,2),migr_kmd=vmig,migr_r=rmig,omori_p=pval,klass=cls))""")

# ------------------------------------------------------------------ §3 ZBZ size vs leaf depth
md(r"""## 3 · Cluster size vs average leaf depth — burst-like vs swarm-like across ALL families

The **Zaliapin–Ben-Zion (2013)** topological diagnostic, for **every** NND family (not just the top 10). Each
family's spanning tree (parent→child links, η<η₀ & R≤1 km) has an **average leaf depth ⟨d⟩** = the mean number of
generations from a leaf back to the root:

- **⟨d⟩ ≈ 1 (a star)** — every event links *directly* to one common parent → a **burst / mainshock-aftershock**
  sequence (aftershocks all triggered by the mainshock).
- **⟨d⟩ large (a deep chain)** — events trigger *sequentially*, each off the previous → a **swarm** (progressive,
  cascading triggering with no single dominant parent).

Every family is plotted (size N on a log axis vs ⟨d⟩), coloured by mainshock M_L. Note the tree metric is
**independent of the moment-based class**, so a large-mainshock (FS-MS-AS) sequence can still be topologically
swarm-like.""")
co(r"""# ---- ZBZ average leaf depth <d> for EVERY family (reuse §2 leaf_depth) ----
_zr=[]
for k in sorted(x for x in g.Cluster.unique() if x>=0):
    mem=g[g.Cluster==k].event_id.values
    ald,gens=leaf_depth(mem)
    _zr.append(dict(Cluster=k,n=len(mem),ald=ald,gens=gens))
Z=pd.DataFrame(_zr).merge(FAM[["Cluster","mainshock_M"]],on="Cluster",how="left")
fig,ax=plt.subplots(figsize=(6.8,6.8))
sc=ax.scatter(Z.n,Z.ald,c=Z.mainshock_M,cmap="viridis",s=38,edgecolor="k",lw=0.3,alpha=0.85,zorder=4)
ax.set_xscale("log"); ax.set_xlim(1.6,Z.n.max()*1.35); ax.set_ylim(-0.3,Z.ald.max()+0.8)
ax.set_box_aspect(1)                                                   # square main panel
ax.set(xlabel="cluster size N (number of members)",ylabel="average leaf depth ⟨d⟩",
       title="ZBZ cluster topology — size vs average leaf depth")
cb=fig.colorbar(sc,ax=ax,fraction=0.046,pad=0.02); cb.set_label("mainshock M$_L$")
ax.text(0.03,0.97,f"{len(Z)} clusters analyzed",transform=ax.transAxes,ha="left",va="top",fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="0.6",lw=0.6))
fig.tight_layout(); plt.show()
print(f"plotted ALL {len(Z)} NND families (size 2..{int(Z.n.max())})")
big=Z[Z.n>=10]
_bu=int((big.ald<=2.0).sum()); _sw=int((big.ald>=5.0).sum())
print(f"of the {len(big)} families with N>=10: {_bu} burst-like (⟨d⟩<=2), {_sw} swarm-like (⟨d⟩>=5), {len(big)-_bu-_sw} intermediate")
for k in TOP[:4]:
    rr=Z[Z.Cluster==k].iloc[0]
    print(f"  C{int(k)} (n={int(rr.n)}, M{rr.mainshock_M}): ⟨d⟩={rr.ald}, {int(rr.gens)} generations")""")

# ------------------------------------------------------------------ §4 summary
md(r"""## 4 · Comparative summary — the ten largest Ulsan-Fault clusters""")
co(r"""SUM=pd.DataFrame(SUMROWS)
print("="*118); print("UF LARGEST NND CLUSTERS — topology / spatial / temporal characterization (Df=1.2, b=1.0)".center(118)); print("="*118)
print(SUM.to_string(index=False))
print("\nTAKE-HOMES")
_ms=SUM[SUM.klass.str.contains("aftershock")]; _sw=SUM[SUM.klass=="swarm"]
print(f" - {len(SUM)} largest families: {len(_ms)} mainshock-driven (incl. FS-MS-AS), {len(_sw)} swarm, {len(SUM)-len(_ms)-len(_sw)} mixed.")
print(f" - TOPOLOGY (ZBZ 2013 avg-leaf-depth ⟨d⟩): {int((SUM.avg_leaf_depth<=2.5).sum())} burst-like (⟨d⟩≤2.5, star), "
      f"{int((SUM.avg_leaf_depth>=5).sum())} swarm-like (⟨d⟩≥5, deep/chained), range {SUM.avg_leaf_depth.min()}–{SUM.avg_leaf_depth.max()}. "
      f"This tree metric is INDEPENDENT of the moment-based class and can disagree — e.g. a moment-dominated")
print(f"   mainshock with a chained (not star) aftershock tree — so report both.")
_np,_nl,_nb=int((SUM["shape"]=="planar").sum()),int((SUM["shape"]=="linear").sum()),int((SUM["shape"]=="blob").sum())
print(f" - GEOMETRY: at these sizes the top-10 are {_np} planar sheet(s), {_nl} linear streak(s), {_nb} blob(s) — "
      f"L1 spans {SUM.L1_km.min()}–{SUM.L1_km.max()} km (only hundreds of m). Strike/dip is reported ONLY for genuine")
print(f"   sheets; linear clusters get a trend, blobs neither. Confirming whether these elongations are REAL vs")
print(f"   within dt.cc location error needs the Phase-3 SVD relocation + error ellipses.")
print(f" - completeness gap: {int(SUM.n_MLless.sum())} ML-less (n_used<3) events fall inside these 5 clusters' envelopes "
      f"(vs {int(SUM.n.sum())} with-ML members) — the Phase-2 relative-magnitude recovery targets.")
_c0=SUM.iloc[0]
print(f" - largest cluster = Cluster {int(_c0.Cluster)} (n={int(_c0.n)}): the catalog-max M{_c0.mainshock_M} "
      f"{_c0.mainshock_date} sequence, classed {_c0.klass}; a {_c0.L1_km}×{_c0.L2_km}×{_c0.L3_km} km {_c0['shape']} "
      f"(NOT a resolvable plane at this scale — strike/dip withheld, needs SVD).")
print("\nNEXT (deferred, per plan): Phase 2 relative-ML recovery of the ML-less members; Phase 3 per-cluster")
print("HypoDD SVD (ISOLV=1) relocation for formal error ellipses on the thicker/curved families.")""")

nb["cells"] = C
import os
os.chdir("/home/msseo/works/02.Ulsan_Fault_detection/analysis/reloc_analysis")
nbf.write(nb, "28.UF_cluster_deepdive.ipynb")
print("wrote 28.UF_cluster_deepdive.ipynb", len(C), "cells")
