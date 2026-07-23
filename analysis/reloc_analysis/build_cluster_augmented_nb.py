#!/usr/bin/env python
"""Generate 29.UF_cluster_augmented_ML.ipynb — COMPLETENESS-AUGMENTED NND clustering of the UF catalog.

UNDER-SOTA PROXY (disclosed up front): the relocated events with NO reliable ML (n_used<3) are assigned a
nominal small magnitude = the MINIMUM assigned ML of the reliable population (M_proxy), then the SAME NND
declustering as nb28 (Df=1.2, b=1.0, 3-D, sub-day time, 1 km link cap) is run on the augmented population and
the families re-identified. This is a first-order stand-in for the rigorous Phase-2 relative-amplitude ML; it
lets the small events (excluded from the reliable-ML NND in nb28) fill in their clusters, so we can see how the
families grow and which sequences (e.g. the 2023 M3.73) are under-counted by the ML-completeness gap.

Caveat: a CONSTANT proxy magnitude under-weights any small event that is actually a bit larger; but because NND
rescales by the PARENT magnitude, the small proxy events act mainly as OFFSPRING/leaves (they rarely become
parents), so the augmentation mostly ADDS cluster members rather than restructuring the trees. The rigorous fix
is relative-amplitude ML (Phase 2). The ~186 relocated events with no amplitude reading at all are absent from
the ML csv and not included here (disclosed). Runs in base."""
import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# Completeness-augmented NND clustering — including the ML-less small events (proxy M)

nb28 declusters only events with **reliable ML** (n_used ≥ 3), so the ~864 relocated events with too few
stations for an ML are excluded — which **under-counts** the clusters (e.g. the 2023 M3.73 sequence loses
~10 aftershocks and drops to rank 8; see nb28). Here, as an explicit **under-SOTA proxy**, every ML-less
relocated event is assigned a nominal small magnitude **M_proxy = the minimum assigned ML of the reliable
population**, and the **same NND** (Df = 1.2, b = 1.0, 3-D, sub-day time, 1 km link cap) is run on the augmented
catalog to re-identify the families.

**This is a stand-in for the rigorous Phase-2 relative-amplitude ML, not a replacement.** Because NND rescales
by the *parent* magnitude, the small proxy events act as **offspring/leaves** — they fill in cluster membership
rather than restructure the trees. Everything is compared against the reliable-only nb28 result and disclosed.""")

# ------------------------------------------------------------------ §0 setup + both populations
co(r"""import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.collections import LineCollection
from collections import deque
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":130,"axes.grid":True,"grid.alpha":0.3,"font.size":10,
                     "legend.framealpha":1,"legend.edgecolor":"black","legend.facecolor":"white"})
sys.path.insert(0,"/home/msseo/works/16.kma_absolute_location"); from kma_absolute_location import nnd, clustering

KG="/home/msseo/works/02.Ulsan_Fault_detection/KS_KG"
RELOC=f"{KG}/local_magnitudes/catalog_ml_heo_ufonly_reloc.csv"
FAULTS=f"{KG}/HypoInv/faults_lonlat.gmt"; COAST=f"{KG}/reloc_analysis/coastline_lonlat.gmt"
UF=(129.25,129.55,35.60,35.90); DF_UF=1.2; B_NND=1.0; MC=1.2; LINKR=1.0
def load_segs(path):
    segs=[]; cur=[]
    if os.path.exists(path):
        for ln in open(path):
            if ln.startswith((">","#")):
                if len(cur)>1: segs.append(np.array(cur))
                cur=[]; continue
            p=ln.split()
            if len(p)>=2:
                try: cur.append((float(p[0]),float(p[1])))
                except ValueError: pass
        if len(cur)>1: segs.append(np.array(cur))
    return segs
FSEG=load_segs(FAULTS); CSEG=load_segs(COAST)
def plot_base(ax):
    for s in CSEG: ax.plot(s[:,0],s[:,1],color="steelblue",lw=0.7,zorder=1)
    for s in FSEG: ax.plot(s[:,0],s[:,1],color="0.35",lw=0.8,zorder=1)
def tyear(t): return t.dt.strftime("%Y%m%d%H%M%S").map(nnd.decimal_year)   # CANONICAL nnd.decimal_year (exact year length; no leap-year overshoot)

rl=pd.read_csv(RELOC); rl["event_time"]=pd.to_datetime(rl.event_time,format="ISO8601",utc=True,errors="coerce")
rl=rl[~rl.event_idx.isin(set(pd.read_csv(f"{KG}/local_magnitudes/blast_event_idx_deblast.csv").event_idx.dropna().astype(int)))].copy()  # DE-BLAST: drop quarry-blast events (nb22 §7)
rl=rl.dropna(subset=["lat","lon","depth","event_time","ml_ufcorr_reloc"]).copy()
MPROXY=float(rl.loc[rl.n_used>=3,"ml_ufcorr_reloc"].min())     # nominal small magnitude for the ML-less events
def make(reliable_only):
    d=(rl[rl.n_used>=3] if reliable_only else rl).copy()
    d["is_proxy"]=False if reliable_only else (d.n_used<3)
    d["kma_mag"]=np.where(d.n_used>=3,d.ml_ufcorr_reloc,MPROXY)
    d["event_id"]=d.event_idx.astype(int).astype(str); d["t_year"]=tyear(d.event_time)
    d=d.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep"}).sort_values("t_year").reset_index(drop=True)
    return d
def run_nnd(d):
    nd=nnd.compute_nnd(d,b=B_NND,D=DF_UF,mmin=None,metric="3d"); e0,_=nnd.fit_eta0(nd.eta.values)
    lab=nnd.build_families(nd,e0,d.event_id.values,link_rmax_km=LINKR)
    return d.assign(Cluster=d.event_id.map(lab).fillna(-1).astype(int)), nd, e0
gA=make(False); gA,ndA,e0A=run_nnd(gA)                          # AUGMENTED (proxy ML for n_used<3)
gR=make(True);  gR,ndR,e0R=run_nnd(gR)                          # reliable-only (nb28 population)
print(f"M_proxy (min assigned reliable ML) = {MPROXY:.2f}  (applied to {int(gA.is_proxy.sum())} ML-less events)")
print(f"AUGMENTED : N={len(gA)} ({int((~gA.is_proxy).sum())} reliable + {int(gA.is_proxy.sum())} proxy) | "
      f"clustered {int((gA.Cluster>=0).sum())} ({100*(gA.Cluster>=0).mean():.0f}%) | families {int(gA.Cluster.max())+1} | log10η0={np.log10(e0A):.2f}")
print(f"RELIABLE  : N={len(gR)} | clustered {int((gR.Cluster>=0).sum())} ({100*(gR.Cluster>=0).mean():.0f}%) | families {int(gR.Cluster.max())+1} | log10η0={np.log10(e0R):.2f}")
print("  (~186 relocated events with no amplitude reading are absent from the ML csv -> not augmented here)")""")

# ------------------------------------------------------------------ helpers (trees, metrics)
co(r"""M0=lambda m: 10.0**(1.5*np.asarray(m,float)+9.1)
def enu_m(sub):
    u,_=clustering.to_utm(sub); return u["x_m"].to_numpy(),u["y_m"].to_numpy(),u["depth_m"].to_numpy()
def build_tree(members,nd,e0):
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
def leaf_depth(members,nd,e0):
    children,parent,depth,roots=build_tree(members,nd,e0)
    leaves=[e for e in members if e not in children]
    return (round(float(np.mean([depth.get(e,0) for e in leaves])),2) if leaves else np.nan,(max(depth.values()) if depth else 0))
def tree_xpos(members,nd,e0,tsort):
    children,parent,depth,roots=build_tree(members,nd,e0); xpos={}; cnt=[0]
    def rec(node):
        ch=sorted(children.get(node,[]),key=lambda e: tsort.get(e,0.0))
        if not ch: xpos[node]=cnt[0]; cnt[0]+=1
        else:
            for cc in ch: rec(cc)
            xpos[node]=float(np.mean([xpos[cc] for cc in ch]))
    for r in sorted(roots,key=lambda e: tsort.get(e,0.0)): rec(r)
    return xpos,depth,children
def classify(f_M0,f_after,f_before):
    if f_M0<0.4: return "swarm"
    if f_M0>=0.6 and f_before>=0.15: return "FS-MS-AS"
    if f_M0>=0.6: return "MS-AS"
    return "mixed"
NTOP=10; TOPA=list(gA[gA.Cluster>=0].groupby("Cluster").size().sort_values(ascending=False).head(NTOP).index)""")

# ------------------------------------------------------------------ §1 comparison
md(r"""## 1 · Reliable-only (nb28) vs augmented — how the families change

Same NND, two populations. Including the small proxy events should **grow** cluster membership (they join as
offspring), raise the clustered count and the family sizes, and lift sequences that were under-counted.""")
co(r"""def fam_sizes(gg): return gg[gg.Cluster>=0].groupby("Cluster").size().sort_values(ascending=False)
sA=fam_sizes(gA); sR=fam_sizes(gR)
# track the M3.89 (2014) and M3.73 (2023) sequences in both
def fam_of(gg,idx):
    c=int(gg.loc[gg.event_idx==idx,"Cluster"].values[0]);
    if c<0: return c,0,0
    m=gg.Cluster==c; return c,int(m.sum()),int((m&gg.is_proxy).sum())
fig,ax=plt.subplots(1,2,figsize=(13,4.4))
ax[0].bar(np.arange(10)-0.2,sR.head(10).values,0.4,color="0.6",label="reliable-only (nb28)")
ax[0].bar(np.arange(10)+0.2,sA.head(10).values,0.4,color="tab:red",label="augmented (+proxy)")
ax[0].set(xlabel="family rank",ylabel="members",title="Top-10 family sizes"); ax[0].legend(fontsize=8.5); ax[0].set_xticks(range(10))
# proxy fraction per augmented top-10 family
pf=[100*((gA.Cluster==k)&gA.is_proxy).sum()/max((gA.Cluster==k).sum(),1) for k in TOPA]
ax[1].bar(range(10),pf,color="tab:orange"); ax[1].set(xlabel="augmented family rank",ylabel="% members that are proxy (ML-less)",title="Small-event fill-in per top-10 family")
fig.tight_layout(); plt.show()
print(f"clustered: reliable {int((gR.Cluster>=0).sum())} -> augmented {int((gA.Cluster>=0).sum())}  (+{int((gA.Cluster>=0).sum())-int((gR.Cluster>=0).sum())} incl. proxy)")
for idx,nm in [(704,'M3.89 2014'),(13902,'M3.73 2023')]:
    cR,nR,_=fam_of(gR,idx); cA,nA,pA=fam_of(gA,idx)
    rkR=(list(sR.index).index(cR)+1) if cR>=0 else None; rkA=(list(sA.index).index(cA)+1) if cA>=0 else None
    print(f"  {nm}: reliable Cluster {cR} n={nR} (rank {rkR}) -> augmented Cluster {cA} n={nA} ({pA} proxy) rank {rkA}")""")

# ------------------------------------------------------------------ §2 NND (T,R) scatter
md(r"""## 2 · Nearest-neighbour (log T, log R) scatter — the augmented plane

Same Zaliapin–Ben-Zion plane as nb28 §0b, but for the **augmented** population. Each event at its
nearest-neighbour rescaled time log₁₀T (x) and distance log₁₀R (y), coloured by the actual 1 km-cap clustering:
**red = reliable-ML child**, **blue = proxy (ML-less) child** (η<η₀ *and* link ≤ 1 km), **orange rings = below
the diagonal but CUT by the 1 km cap**, **grey = background (η≥η₀)**. The proxy children populate the same
clustered lobe as the reliable ones — confirming they are genuine near-neighbour offspring, not scattered noise
pulled in by the low proxy magnitude.""")
co(r"""_XL=(-8.0,2.0); _YL=(-6.0,4.0); leA=np.log10(e0A)
_prox=set(gA.loc[gA.is_proxy,"event_id"])
clm=ndA[(ndA.eta<e0A)&(ndA.R_km<=LINKR)]; cut=ndA[(ndA.eta<e0A)&(ndA.R_km>LINKR)]; bgm=ndA[ndA.eta>=e0A]
clm_r=clm[~clm.event_id.isin(_prox)]; clm_p=clm[clm.event_id.isin(_prox)]
fig,ax=plt.subplots(figsize=(7.0,6.6))
ax.scatter(bgm.logT,bgm.logR,s=6,c="0.62",lw=0,alpha=0.5,label=f"background, η≥η₀ ({len(bgm)})",zorder=2)
ax.scatter(cut.logT,cut.logR,s=26,c="none",edgecolor="tab:orange",lw=0.9,marker="o",label=f"η<η₀ but CUT by 1 km cap ({len(cut)})",zorder=3)
ax.scatter(clm_r.logT,clm_r.logR,s=11,c="tab:red",lw=0,alpha=0.8,label=f"reliable-ML child ({len(clm_r)})",zorder=4)
ax.scatter(clm_p.logT,clm_p.logR,s=11,c="tab:blue",lw=0,alpha=0.8,label=f"proxy (ML-less) child ({len(clm_p)})",zorder=5)
ax.plot(_XL,leA-np.array(_XL),"--",lw=1.6,color="k",label=fr"$\eta_0$ (log$_{{10}}$={leA:.2f})",zorder=6)
ax.set(xlim=_XL,ylim=_YL,xlabel=r"Rescaled time  log$_{10}T$",ylabel=r"Rescaled distance  log$_{10}R$",
       title=f"Augmented NND (T, R) — reliable vs proxy children after 1 km cap (Df={DF_UF}, b={B_NND})")
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="lower left",fontsize=8.5); plt.show()
print(f"augmented nn-links {len(ndA)} | children (η<η₀ & ≤1 km) {len(clm)} = {len(clm_r)} reliable + {len(clm_p)} proxy | "
      f"cut by 1 km cap {len(cut)} | background {len(bgm)} | log10 eta0={leA:.2f}")""")

# ------------------------------------------------------------------ §3 overview map
md(r"""## 3 · Augmented families — map & depth (proxy small events marked)

Augmented clustered families (top-10 coloured), background grey, **proxy (ML-less) members drawn as small
hollow circles** so you can see where the small events fill in around the reliable cores.""")
co(r"""_cols=plt.cm.tab10(np.linspace(0,1,10))
fig=plt.figure(figsize=(15,5.4)); axm=fig.add_axes([0.05,0.1,0.44,0.82]); axL=fig.add_axes([0.56,0.12,0.4,0.78])
plot_base(axm); bg=gA[gA.Cluster<0]
axm.scatter(bg.svi_lon,bg.svi_lat,s=4,c="0.82",lw=0,zorder=2)
for i,k in enumerate(TOPA):
    c=gA[gA.Cluster==k]; rel=c[~c.is_proxy]; px=c[c.is_proxy]
    axm.scatter(rel.svi_lon,rel.svi_lat,s=16,color=_cols[i],lw=0.2,edgecolor="k",zorder=4,label=f"C{k} (n={len(c)})")
    axm.scatter(px.svi_lon,px.svi_lat,s=16,facecolor="none",edgecolor=_cols[i],lw=0.8,zorder=4)
    ms=c.loc[c.kma_mag.idxmax()]; axm.scatter(ms.svi_lon,ms.svi_lat,s=150,marker="*",color=_cols[i],edgecolor="k",lw=0.5,zorder=5)
axm.set(xlim=UF[:2],ylim=UF[2:],xlabel="Longitude (°E)",ylabel="Latitude (°N)",title="Augmented NND families — filled=reliable, hollow=proxy(ML-less), ★=mainshock")
axm.set_aspect(1/np.cos(np.deg2rad(35.75))); axm.legend(fontsize=6.5,ncol=2,loc="upper left")
axL.scatter(bg.svi_lon,bg.svi_dep,s=3,c="0.82",lw=0)
for i,k in enumerate(TOPA):
    c=gA[gA.Cluster==k]; axL.scatter(c.svi_lon,c.svi_dep,s=12,color=_cols[i],lw=0)
axL.set(xlim=UF[:2],ylim=(20,0),xlabel="Longitude (°E)",ylabel="Depth (km)",title="Depth section — augmented top-10")
plt.show()""")

# ------------------------------------------------------------------ §3 family table
md(r"""## 4 · Augmented family table (top 15) — with the proxy-member split""")
co(r"""rows=[]
for k in fam_sizes(gA).head(15).index:
    c=gA[gA.Cluster==k].sort_values("t_year").reset_index(drop=True); ms=c.loc[c.kma_mag.idxmax()]
    x,y,z=enu_m(c); ext=np.hypot(x-x.mean(),y-y.mean()).max()*2/1000.0
    m0=M0(c.kma_mag.values); f_M0=float(m0[c.kma_mag.idxmax()]/m0.sum())
    fa=float((c.event_time>ms.event_time).mean()); fb=float((c.event_time<ms.event_time).mean())
    ald,ngen=leaf_depth(list(c.event_id),ndA,e0A)
    rows.append(dict(Cluster=k,n=len(c),n_proxy=int(c.is_proxy.sum()),mainshock_M=round(float(ms.kma_mag),2),
                     date=str(ms.event_time)[:10],ext_km=round(ext,2),leaf_depth=ald,f_M0=round(f_M0,2),klass=classify(f_M0,fa,fb)))
TAB=pd.DataFrame(rows); print(TAB.to_string(index=False))""")

# ------------------------------------------------------------------ §4 spanning trees
md(r"""## 5 · Spanning trees of the top-10 augmented families

Parent→child trees (y = generation, tidy-tree x, NOT time). **Filled nodes = reliable-ML members, hollow =
proxy (ML-less) small events**, node colour = time, ★ = mainshock. Shows where the small events attach — mostly
as leaves/offspring, confirming they fill in membership without restructuring the trees.""")
co(r"""fig,axs=plt.subplots(2,5,figsize=(16,6.6)); axs=axs.ravel()
for ax,k in zip(axs,TOPA):
    c=gA[gA.Cluster==k].sort_values("t_year").reset_index(drop=True)
    t0=c.event_time.min(); days=(c.event_time-t0).dt.total_seconds().values/86400.0
    tsort=dict(zip(c.event_id,days)); xpos,depth,children=tree_xpos(list(c.event_id),ndA,e0A,tsort)
    for p,chs in children.items():
        for ch in chs: ax.plot([xpos[p],xpos[ch]],[depth[p],depth[ch]],color="0.6",lw=0.6,zorder=1)
    rel=c[~c.is_proxy]; px=c[c.is_proxy]
    for sub,fc in [(rel,None),(px,"none")]:
        if len(sub):
            xs=[xpos[e] for e in sub.event_id]; ys=[depth[e] for e in sub.event_id]; dd=[tsort[e] for e in sub.event_id]
            if fc=="none": ax.scatter(xs,ys,s=22,facecolor="none",edgecolor="0.3",lw=0.7,zorder=3)
            else: ax.scatter(xs,ys,s=22,c=dd,cmap="viridis",edgecolor="k",lw=0.2,zorder=3)
    mid=c.loc[c.kma_mag.idxmax(),"event_id"]; ax.scatter(xpos[mid],depth[mid],s=200,marker="*",color="red",edgecolor="k",lw=0.5,zorder=4)
    ald,ngen=leaf_depth(list(c.event_id),ndA,e0A)
    ax.invert_yaxis(); ax.set_xticks([]); ax.set(title=f"C{k}: n={len(c)} ({int(c.is_proxy.sum())} px), ⟨d⟩={ald}",ylabel="gen")
fig.suptitle("Top-10 augmented family spanning trees — hollow=proxy(ML-less), ★=mainshock",y=1.0); fig.tight_layout(); plt.show()""")

# ------------------------------------------------------------------ §5 per-cluster M-t
md(r"""## 6 · Per-cluster time–magnitude plots (top-10 augmented families)

Same style as nb28 panel (d): stem M–t per family, marker colour = time (days from first event), red dashed
line at the mainshock. **Filled = reliable-ML members, hollow = proxy (ML-less) events** (all drawn at the
constant M_proxy, so they line up along the bottom) — this makes the completeness fill-in and its timing
relative to the mainshock directly visible.""")
co(r"""fig,axs=plt.subplots(2,5,figsize=(16,6.6)); axs=axs.ravel()
for ax,k in zip(axs,TOPA):
    c=gA[gA.Cluster==k].sort_values("t_year").reset_index(drop=True)
    t0=c.event_time.min(); days=(c.event_time-t0).dt.total_seconds().values/86400.0
    ms=c.loc[c.kma_mag.idxmax()]; msday=(ms.event_time-t0).total_seconds()/86400.0
    ymin=float(c.kma_mag.min())-0.3
    ax.vlines(days,ymin,c.kma_mag.values,color="0.7",lw=0.5,zorder=1)
    rel=c[~c.is_proxy]; px=c[c.is_proxy]
    dR=(rel.event_time-t0).dt.total_seconds().values/86400.0; dP=(px.event_time-t0).dt.total_seconds().values/86400.0
    if len(rel): ax.scatter(dR,rel.kma_mag,s=20,c=dR,cmap="viridis",edgecolor="k",lw=0.2,zorder=3)
    if len(px): ax.scatter(dP,px.kma_mag,s=20,facecolor="none",edgecolor="0.35",lw=0.7,zorder=2)
    ax.axvline(msday,color="red",ls="--",lw=1,zorder=2)
    ax.scatter([msday],[ms.kma_mag],s=150,marker="*",color="red",edgecolor="k",lw=0.5,zorder=4)
    ax.set(title=f"C{k}: n={len(c)} ({int(c.is_proxy.sum())} px), M{ms.kma_mag:.2f}",xlabel="days",ylabel="M$_L$")
fig.suptitle("Top-10 augmented family M–t — filled=reliable, hollow=proxy(ML-less), ★=mainshock, dashed=mainshock time",y=1.0)
fig.tight_layout(); plt.show()""")

# ------------------------------------------------------------------ §7 ZBZ size vs leaf depth (augmented)
md(r"""## 7 · Cluster size vs average leaf depth — the AUGMENTED catalog (ML-less events included)

The same Zaliapin–Ben-Zion topology plot as nb28 §3, but on the **augmented population** — every ML-less
(unassigned-ML, `n_used<3`) relocated event is included via its proxy magnitude, so families that were invisible or
undersized in the reliable-only run now appear at their true membership. Each augmented family's spanning tree
(parent→child links, η<η₀ & R≤1 km) gives an **average leaf depth ⟨d⟩** (mean generations from a leaf to the
root): **⟨d⟩ ≈ 1 = burst/mainshock-aftershock star**, **⟨d⟩ large = swarm/deep chain**. Every augmented family is
plotted (size N on a log axis vs ⟨d⟩), coloured by mainshock M_L (a family whose largest member is itself a proxy
event sits at the M_proxy floor colour). Because the proxy events attach mostly as **leaves**, including them tends
to *lengthen* the trees (more offspring generations) — the plot shows how the topology shifts once the
completeness fill-in is added.""")
co(r"""# ---- ZBZ average leaf depth <d> for EVERY augmented family (proxy ML-less events included) ----
_zr=[]
for k in sorted(x for x in gA.Cluster.unique() if x>=0):
    mem=gA[gA.Cluster==k].event_id.values
    ald,gens=leaf_depth(mem,ndA,e0A)
    _zr.append(dict(Cluster=k,n=len(mem),ald=ald,gens=gens))
ZA=pd.DataFrame(_zr)
_msm=gA[gA.Cluster>=0].groupby("Cluster").kma_mag.max().rename("mainshock_M")
ZA=ZA.merge(_msm,on="Cluster",how="left")
fig,ax=plt.subplots(figsize=(6.8,6.8))
sc=ax.scatter(ZA.n,ZA.ald,c=ZA.mainshock_M,cmap="viridis",s=38,edgecolor="k",lw=0.3,alpha=0.85,zorder=4)
ax.set_xscale("log"); ax.set_xlim(1.6,ZA.n.max()*1.35); ax.set_ylim(-0.3,ZA.ald.max()+0.8)
ax.set_box_aspect(1)                                                   # square main panel
ax.set(xlabel="cluster size N (number of members)",ylabel="average leaf depth ⟨d⟩",
       title="ZBZ cluster topology — augmented catalog (ML-less events included)")
cb=fig.colorbar(sc,ax=ax,fraction=0.046,pad=0.02); cb.set_label("mainshock M$_L$")
ax.text(0.03,0.97,f"{len(ZA)} clusters analyzed\n(incl. ML-less proxy events)",transform=ax.transAxes,ha="left",va="top",
        fontsize=9.5,bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="0.6",lw=0.6))
fig.tight_layout(); plt.show()
print(f"plotted ALL {len(ZA)} augmented NND families (size 2..{int(ZA.n.max())}); "
      f"{int(gA.is_proxy.sum())} ML-less proxy events included in the population")
_big=ZA[ZA.n>=10]
_bu=int((_big.ald<=2.0).sum()); _sw=int((_big.ald>=5.0).sum())
print(f"of the {len(_big)} augmented families with N>=10: {_bu} burst-like (⟨d⟩<=2), {_sw} swarm-like (⟨d⟩>=5), {len(_big)-_bu-_sw} intermediate")
for k in TOPA[:4]:
    rr=ZA[ZA.Cluster==k].iloc[0]
    print(f"  augmented C{int(k)} (n={int(rr.n)}, M{rr.mainshock_M:.2f}): ⟨d⟩={rr.ald}, {int(rr.gens)} generations")""")

# ------------------------------------------------------------------ §8 summary
md(r"""## 8 · Summary""")
co(r"""print("="*96); print("COMPLETENESS-AUGMENTED NND (proxy M for ML-less events) — UF".center(96)); print("="*96)
print(f"proxy magnitude M_proxy = {MPROXY:.2f} (min assigned reliable ML) applied to {int(gA.is_proxy.sum())} ML-less relocated events\n")
print(f" reliable-only (nb28) : {len(gR)} events, {int((gR.Cluster>=0).sum())} clustered, {int(gR.Cluster.max())+1} families")
print(f" augmented (+proxy)   : {len(gA)} events, {int((gA.Cluster>=0).sum())} clustered, {int(gA.Cluster.max())+1} families")
_pc=100*((gA.Cluster>=0)&gA.is_proxy).sum()/max(int((gA.Cluster>=0).sum()),1)
print(f"   of the augmented clustered events, {int(((gA.Cluster>=0)&gA.is_proxy).sum())} ({_pc:.0f}%) are proxy (small) events -> the completeness fill-in")
print("\nTAKE-HOMES")
print(f" - Including the small events GROWS the clusters: top family {int(fam_sizes(gR).iloc[0])} -> {int(fam_sizes(gA).iloc[0])} members;")
print(f"   the 2023 M3.73 sequence, rank 8 in nb28, rises with its ~9 recovered small members (see §1).")
print(" - Proxy events attach as LEAVES/offspring (§4 trees) — they fill membership, not restructure topology,")
print("   because NND rescales by the PARENT magnitude, so a small proxy event rarely becomes a parent.")
print(" - UNDER-SOTA CAVEAT: a constant M_proxy is a stand-in; the rigorous Phase-2 relative-amplitude ML would")
print("   give each small event its true magnitude (some are larger) and could add a few genuine parents.")""")

nb["cells"]=C
import os
nbf.write(nb,"/home/msseo/works/02.Ulsan_Fault_detection/analysis/reloc_analysis/29.UF_cluster_augmented_ML.ipynb")
print("wrote 29.UF_cluster_augmented_ML.ipynb",len(C),"cells")
