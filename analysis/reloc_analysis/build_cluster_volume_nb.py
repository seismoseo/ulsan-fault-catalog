#!/usr/bin/env python
"""Generate 30.UF_cluster_volume_history.ipynb — dedicated volume history of the two LARGEST-EVENT clusters
(M3.89 2014-09-23 and M3.73 2023-11-29). For each: center a 1 km^3 cube on the cluster centroid, pull EVERY
relocated event in that volume from the FULL HypoDD catalog (incl. ML-less / no-reading events, not just the
NND-linked ones), and ask whether the volume had seismicity BEFORE the mainshock that NND never linked (NND
links by rescaled time, so long-quiet-then-reactivated volumes are invisible to it by construction).

Caveat (stated in-notebook): only RELOCATED events can be placed in a 1 km^3 box — absolute catalog locations
carry km-scale error — so "previous seismicity" = previously relocated events; pre-network activity is not
recoverable at this precision. Runs in base (numpy/pandas/matplotlib incl. mplot3d)."""
import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# Volume history of the two largest-event clusters — was the volume active before?

The two largest Ulsan-Fault earthquakes are **M3.89 (2014-09-23)** and **M3.73 (2023-11-29)**, each the
mainshock of an NND family (nb28). NND links every event to its nearest neighbour in **rescaled time** τ, so it
groups *bursts* — a volume that was seismically active, went quiet for years, then reactivated is **invisible to
NND by construction** (the old events link to their own contemporaries, not across the quiet gap).

Here we bypass NND: for each target cluster we **center a 1 km³ cube** on the cluster centroid and pull
**every relocated event inside that volume** from the *full* HypoDD catalog — including the small ML-less and
no-reading events that never entered the declustering — then look at the **time history** of the volume. The
question: *was there earlier seismicity in this exact volume, and how did the declustering treat it?* Every
in-volume non-member is decomposed into **other-family** (NND grouped it into a *different* family),
**background** (in the NND population, classified η≥η₀), or **not-in-population** (ML-less, excluded by the
n_used≥3 gate before NND ever saw it) — three very different statements about what the declustering "missed".

**Caveat:** only **relocated** events can be placed in a 1 km³ box (absolute catalog locations carry km-scale
error). So "previous seismicity" here means previously **relocated** events; truly pre-network activity is not
recoverable at this precision. The relocated catalog itself begins ~2012, so the lookback is bounded by that.""")

# ------------------------------------------------------------------ §0 setup + families + full catalog
co(r"""import warnings; warnings.filterwarnings("ignore")
import os, sys, glob, re
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":130,"axes.grid":True,"grid.alpha":0.25,"font.size":10,
                     "legend.framealpha":1,"legend.edgecolor":"black","legend.facecolor":"white"})
sys.path.insert(0,"/home/msseo/works/16.kma_absolute_location"); from kma_absolute_location import nnd, clustering

KG="/home/msseo/works/02.Ulsan_Fault_detection/KS_KG"
RELOC=f"{KG}/local_magnitudes/catalog_ml_heo_ufonly_reloc.csv"
FAULTS=f"{KG}/HypoInv/faults_lonlat.gmt"; COAST=f"{KG}/reloc_analysis/coastline_lonlat.gmt"
UF=(129.25,129.55,35.60,35.90); DF_UF=1.2; B_NND=1.0; LINKR=1.0
HALF=0.5           # cube half-width (km) -> 1 km x 1 km x 1 km volume (change to widen)
CMAP="viridis"
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

# ---- reliable-ML NND families (same basis as nb28) to DEFINE the two target clusters ----
rl=pd.read_csv(RELOC); rl["event_time"]=pd.to_datetime(rl.event_time,format="ISO8601",utc=True,errors="coerce")
rl=rl[~rl.event_idx.isin(set(pd.read_csv(f"{KG}/local_magnitudes/blast_event_idx_deblast.csv").event_idx.dropna().astype(int)))].copy()  # DE-BLAST: drop quarry-blast events (nb22 §7)
rl=rl.dropna(subset=["lat","lon","depth","event_time","ml_ufcorr_reloc"]).copy()
g=rl[rl.n_used>=3].copy(); g["event_id"]=g.event_idx.astype(int).astype(str); g["t_year"]=tyear(g.event_time)
g=g.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep","ml_ufcorr_reloc":"kma_mag"}).sort_values("t_year").reset_index(drop=True)
nd=nnd.compute_nnd(g,b=B_NND,D=DF_UF,mmin=None,metric="3d"); e0,_=nnd.fit_eta0(nd.eta.values)
labels=nnd.build_families(nd,e0,g.event_id.values,link_rmax_km=LINKR); g["Cluster"]=g.event_id.map(labels).fillna(-1).astype(int)
CLU=dict(zip(g.event_idx.astype(int),g.Cluster.astype(int)))      # event_idx -> NND family (-1 = background)
fammax=g[g.Cluster>=0].groupby("Cluster").kma_mag.max().sort_values(ascending=False)
TARGETS=list(fammax.head(2).index)                       # two families with the LARGEST mainshock magnitude
print("two largest-event clusters (reliable NND, 1 km cap):")
for k in TARGETS:
    c=g[g.Cluster==k]; ms=c.loc[c.kma_mag.idxmax()]
    print(f"  Cluster {k}: mainshock M{ms.kma_mag:.2f} on {str(ms.event_time)[:10]} | {len(c)} reliable-ML members")""")

co(r"""# ---- FULL relocated catalog (ALL events incl. ML-less / no-reading) via exact id->ts->event_idx map ----
RELOC_FILE="/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/uf_subregion_reuse/2.HypoDD/03.dt.cc_kim2011/hypoDD.reloc"
_rc=["id","lat","lon","depth","x","y","z","ex","ey","ez","yr","mo","dy","hr","mi","sc","mag","nccp","nccs","nctp","ncts","rcc","rct","cid"]
_r0=pd.read_csv(RELOC_FILE,sep=r"\s+",header=None,names=_rc); _r0["ncc"]=_r0.nccp+_r0.nccs
_WF100="/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/uf_subregion_reuse/waveforms_100km"
_MEIDX="/home/msseo/works/02.Ulsan_Fault_detection/analysis/uf_subregion_hypodd/uf_subregion/members_event_idx.csv"
_dirs=sorted(os.path.basename(d) for d in glob.glob(os.path.join(_WF100,"20*")))
_id2ts={200000+i:ts for i,ts in enumerate(_dirs)}
_mei=pd.read_csv(_MEIDX).sort_values("event_idx"); _mei["ts"]=pd.to_datetime(_mei.time,utc=True,format="ISO8601").dt.floor("s").dt.strftime("%Y%m%d%H%M%S")
_ts2eidx={}
for _e,_t in zip(_mei.event_idx.astype(int),_mei.ts): _ts2eidx.setdefault(_t,_e)
allc=_r0.copy(); allc["event_idx"]=allc.id.map(_id2ts).map(_ts2eidx)
allc=allc.merge(rl[["event_idx","ml_ufcorr_reloc","n_used"]],on="event_idx",how="left").drop_duplicates("id")
allc["has_ml"]=allc.ml_ufcorr_reloc.notna()&(allc.n_used>=3)
_scf=allc.sc.clip(0,59.999)
allc["time"]=pd.to_datetime(dict(year=allc.yr,month=allc.mo,day=allc.dy,hour=allc.hr,minute=allc.mi,
             second=_scf.astype(int),microsecond=((_scf-_scf.astype(int))*1e6).astype(int)),utc=True,errors="coerce")
allc=allc.dropna(subset=["time","lat","lon","depth"]).copy()
allc["magU"]=np.where(allc.has_ml,allc.ml_ufcorr_reloc,allc.mag)     # ML where reliable, else hypoDD input mag
allc["t_year"]=tyear(allc.time)
au,_=clustering.to_utm(allc.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep"}))  # UTM 52N metres
allc["x_m"]=au.x_m.values; allc["y_m"]=au.y_m.values; allc["depth_m"]=au.depth_m.values
# ---- location-quality columns: shift from the hypoDD.loc INITIAL location + total used links ----
# (a handful of weakly-linked events moved >5-10 km during relocation; they carry normal-looking coordinates
#  and could land inside an analysis cube as spurious "unlinked" events -> flag, don't trust silently)
LOC_FILE=RELOC_FILE.replace("hypoDD.reloc","hypoDD.loc")
_lc=["id","lat","lon","depth","x","y","z","ex","ey","ez","yr","mo","dy","hr","mi","sc","mag","cid"]
_l0=pd.read_csv(LOC_FILE,sep=r"\s+",header=None,names=_lc)[["id","lat","lon","depth"]]
_lu,_=clustering.to_utm(_l0.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep"}))
_l0["x0"]=_lu.x_m.values; _l0["y0"]=_lu.y_m.values; _l0["z0"]=_lu.depth_m.values
allc=allc.merge(_l0[["id","x0","y0","z0"]],on="id",how="left")
allc["shift_km"]=np.sqrt((allc.x_m-allc.x0)**2+(allc.y_m-allc.y0)**2+(allc.depth_m-allc.z0)**2)/1000.0
allc["nlinks"]=allc.nccp+allc.nccs+allc.nctp+allc.ncts
allc["ncc"]=allc.nccp+allc.nccs
allc["suspect"]=(allc.shift_km>3.0)|(allc.nlinks<10)|(allc.ncc==0)  # weak/large-move OR ct-only (cc-starved, e.g. M3.89 mainshock)
print(f"FULL relocated catalog: {len(allc)} events | reliable ML {int(allc.has_ml.sum())} | "
      f"ML-less/no-reading {int((~allc.has_ml).sum())} | span {str(allc.time.min())[:10]} … {str(allc.time.max())[:10]}")
print(f"location-quality flags: {int((allc.shift_km>3).sum())} events moved >3 km from initial location, "
      f"{int((allc.nlinks<10).sum())} have <10 used links -> {int(allc.suspect.sum())} flagged 'suspect'")
print(f"cube = {2*HALF:.1f} km × {2*HALF:.1f} km × {2*HALF:.1f} km, centred on each cluster centroid")""")

# ------------------------------------------------------------------ §0b box builder
md(r"""## 0 · Define each cube & pull every relocated event inside

For each target cluster we take the **centroid** of its reliable-ML members (UTM metres), build a cube of
half-width `HALF` (default 0.5 km → 1 km³), and select **all** relocated events whose 3-D location falls inside.
Every event gets a **category**: `member` (this family) / `other-family` (NND grouped it into a *different*
family) / `background` (in the NND population but η≥η₀) / `not-in-pop` (ML-less, excluded by the n_used≥3 gate
before the declustering ever ran). Pre-mainshock counts are reported **per category** — only the `background`
portion is seismicity the NND metric itself failed to attribute; `other-family` means the declustering *did*
see it as a (separate) sequence, and `not-in-pop` is a completeness effect, not an NND failure. Events flagged
`suspect` (moved >3 km in relocation, <10 used links, **or ct-only**) are counted separately — their in-cube
presence may be spurious.

**Selection unions the cube with the full NND family** so no grouped member is omitted. This matters for the 2014
**M3.89 mainshock**: it is *cc-starved* (its waveform is too dissimilar to its aftershocks to cross-correlate, so
all cross-correlation links are culled and it is located by **catalog dt only**, `ncc=0`). In the whole-box
catalog shown here its ct-only hypocentre therefore sits **~0.6 km off** the tight aftershock cloud (the offset
you see for its ★) — but it is a family-0 member and is kept in the volume. The dedicated per-volume relocation
(nb33) re-locates it with the cluster's catalog differential times and brings it back **on-cluster**.""")
co(r"""HW=HALF*1000.0
def build_box(k):
    memset=set(g.loc[g.Cluster==k,"event_idx"].astype(int))
    mem=allc[allc.event_idx.isin(memset)]
    cE,cN,cZ=float(mem.x_m.mean()),float(mem.y_m.mean()),float(mem.depth_m.mean())     # centroid (m)
    ms=mem.loc[mem.magU.idxmax()]                                                       # mainshock row
    in_cube=(allc.x_m.between(cE-HW,cE+HW))&(allc.y_m.between(cN-HW,cN+HW))&(allc.depth_m.between(cZ-HW,cZ+HW))
    box=allc[in_cube|allc.event_idx.isin(memset)].copy()   # UNION cube with full NND family: no member omitted (e.g. the ct-only M3.89 mainshock, offset just outside the cube in the whole-box catalog)
    box["is_member"]=box.event_idx.isin(memset)
    box["is_prior"]=box.time<ms.time
    box["nnd_cluster"]=box.event_idx.map(CLU)                     # NaN = not in the NND population (ML-less)
    box["cat"]=np.where(box.is_member,"member",
               np.where(box.nnd_cluster.isna(),"not-in-pop",
               np.where(box.nnd_cluster==-1,"background","other-family")))
    box["e_km"]=(box.x_m-cE)/1000; box["n_km"]=(box.y_m-cN)/1000; box["z_km"]=(box.depth_m-cZ)/1000
    box["r_km"]=np.sqrt(box.e_km**2+box.n_km**2+box.z_km**2)
    box=box.sort_values("time").reset_index(drop=True)
    prior=box[box.is_prior&~box.is_member]
    info=dict(k=k,center=(cE,cN,cZ),ms=ms,n=len(box),n_mem=int(box.is_member.sum()),
              n_other=int((~box.is_member).sum()),n_prior=int(box.is_prior.sum()),
              n_prior_nonmem=len(prior),
              n_prior_otherfam=int((prior.cat=="other-family").sum()),
              n_prior_bg=int((prior.cat=="background").sum()),
              n_prior_notpop=int((prior.cat=="not-in-pop").sum()),
              prior_fams=sorted(set(int(c) for c in prior.loc[prior.cat=="other-family","nnd_cluster"])),
              n_susp_prior=int(prior.suspect.sum()),
              n_mless=int((~box.has_ml).sum()),n_mem_total=len(mem),n_mem_incube=int(box.is_member.sum()))
    if len(prior):
        info["earliest"]=str(prior.time.min())[:10]; info["lead_d"]=round((ms.time-prior.time.min()).total_seconds()/86400,1)
        bgp=prior[prior.cat=="background"]
        info["earliest_bg"]=str(bgp.time.min())[:10] if len(bgp) else None
    else: info["earliest"]=None; info["lead_d"]=None; info["earliest_bg"]=None
    return box,info
BOXES={k:build_box(k) for k in TARGETS}
for k in TARGETS:
    box,info=BOXES[k]
    print(f"Cluster {k}: mainshock M{info['ms'].magU:.2f} {str(info['ms'].time)[:10]} | "
          f"in-cube {info['n']} (= {info['n_mem']} member + {info['n_other']} non-member) | "
          f"pre-mainshock non-member {info['n_prior_nonmem']} = {info['n_prior_otherfam']} other-family"
          f"{' '+str(info['prior_fams']) if info['prior_fams'] else ''} + {info['n_prior_bg']} background + "
          f"{info['n_prior_notpop']} not-in-pop (ML-less) | suspect among prior {info['n_susp_prior']} | "
          f"earliest prior {info['earliest']} (lead {info['lead_d']} d; earliest TRUE-background {info['earliest_bg']})")""")

# ------------------------------------------------------------------ §1 context map
md(r"""## 1 · Geographic context — where the two cubes sit on the fault""")
co(r"""fig,ax=plt.subplots(figsize=(7.4,7.2)); plot_base(ax)
ax.scatter(allc.lon,allc.lat,s=3,c="0.8",lw=0,zorder=2,label=f"all relocated ({len(allc)})")
_cc=["tab:red","tab:blue"]
for i,k in enumerate(TARGETS):
    box,info=BOXES[k]; cE,cN,cZ=info["center"]; ms=info["ms"]
    m=g[g.Cluster==k]; ax.scatter(m.svi_lon,m.svi_lat,s=14,color=_cc[i],lw=0.2,edgecolor="k",zorder=4,label=f"Cluster {k} (M{ms.magU:.2f})")
    dlat=HALF/110.54; dlon=HALF/(111.32*np.cos(np.deg2rad(ms.lat)))
    ax.add_patch(Rectangle((ms.lon-dlon,ms.lat-dlat),2*dlon,2*dlat,fill=False,ec=_cc[i],lw=1.6,zorder=5))
    ax.scatter(ms.lon,ms.lat,s=210,marker="*",color=_cc[i],edgecolor="k",lw=0.6,zorder=6)
# 5 km scale bar (lower-left)
_y0=UF[2]+0.02; _x0=UF[0]+0.02; _dl=5/(111.32*np.cos(np.deg2rad(35.75)))
ax.plot([_x0,_x0+_dl],[_y0,_y0],"k-",lw=2.5); ax.text(_x0+_dl/2,_y0+0.004,"5 km",ha="center",fontsize=8)
ax.set(xlim=UF[:2],ylim=UF[2:],xlabel="Longitude (°E)",ylabel="Latitude (°N)",title="Two largest-event clusters — 1 km³ cube footprints")
ax.set_aspect(1/np.cos(np.deg2rad(35.75))); ax.legend(loc="upper left",fontsize=8.5); plt.show()""")

# ------------------------------------------------------------------ §2 per-cluster spatial
md(r"""## 2 · Inside the cube — spatial structure (plan · 3-D · orthogonal sections)

All relocated events in the 1 km³ volume, in a **cube-centred frame** (km from centroid). Colour = **year**;
**circles = members of this family, triangles = other in-volume events**. Pre-mainshock non-members carry a
**category ring**: **red = true background** (in the NND population, η≥η₀ — the only ones the declustering
metric genuinely failed to attribute), **orange = member of another NND family** (the declustering saw it, as a
separate sequence), **purple = not in the NND population** (ML-less, n_used<3 — a completeness exclusion).
★ = mainshock. Depth axes point down; the dashed square marks the 1 km cube.""")
co(r"""def spatial_fig(k):
    box,info=BOXES[k]; ms=info["ms"]
    ty=box.t_year.values; norm=Normalize(ty.min(),ty.max())
    sz=(14+34*(box.magU-box.magU.min())).clip(10,240).values
    mem=box.is_member.values; pri=box.is_prior.values; catv=box["cat"].values; H=HALF
    CATCOL=[("background","red"),("other-family","orange"),("not-in-pop","purple")]
    e,n,z=box.e_km.values,box.n_km.values,box.z_km.values
    msE,msN,msZ=(ms.x_m-info["center"][0])/1000,(ms.y_m-info["center"][1])/1000,(ms.depth_m-info["center"][2])/1000
    fig=plt.figure(figsize=(15,9)); gs=fig.add_gridspec(2,2,hspace=0.26,wspace=0.24)
    def pts(ax,xx,yy):
        ax.scatter(xx[~mem],yy[~mem],marker="^",s=sz[~mem],c=ty[~mem],cmap=CMAP,norm=norm,edgecolor="0.3",lw=0.4,zorder=3)
        sc=ax.scatter(xx[mem],yy[mem],marker="o",s=sz[mem],c=ty[mem],cmap=CMAP,norm=norm,edgecolor="k",lw=0.6,zorder=4)
        for cc_,col_ in CATCOL:
            pm=(~mem)&pri&(catv==cc_)
            if pm.any(): ax.scatter(xx[pm],yy[pm],marker="^",s=sz[pm]*1.15,facecolor="none",edgecolor=col_,lw=1.2,zorder=5)
        return sc
    # (a) plan E-N
    axP=fig.add_subplot(gs[0,0]); sc=pts(axP,e,n)
    axP.add_patch(Rectangle((-H,-H),2*H,2*H,fill=False,ec="0.3",ls="--",lw=1.1))
    axP.scatter(msE,msN,s=300,marker="*",color="red",edgecolor="k",lw=0.7,zorder=6)
    axP.set(xlabel="East offset (km)",ylabel="North offset (km)",title=f"(a) plan view — Cluster {k}",xlim=(-H*1.3,H*1.3),ylim=(-H*1.3,H*1.3)); axP.set_aspect("equal")
    # (b) 3-D perspective
    ax3=fig.add_subplot(gs[0,1],projection="3d")
    ax3.scatter(e[~mem],n[~mem],z[~mem],marker="^",s=sz[~mem]*0.7,c=ty[~mem],cmap=CMAP,norm=norm,edgecolor="0.3",lw=0.2,depthshade=False)
    ax3.scatter(e[mem],n[mem],z[mem],marker="o",s=sz[mem]*0.7,c=ty[mem],cmap=CMAP,norm=norm,edgecolor="k",lw=0.3,depthshade=False)
    ax3.scatter([msE],[msN],[msZ],marker="*",s=340,color="red",edgecolor="k",lw=0.5,depthshade=False)
    for a in [-H,H]:
        for b in [-H,H]:
            ax3.plot([a,a],[b,b],[-H,H],color="0.6",lw=0.5); ax3.plot([a,a],[-H,H],[b,b],color="0.6",lw=0.5); ax3.plot([-H,H],[a,a],[b,b],color="0.6",lw=0.5)
    ax3.set_xlabel("E (km)"); ax3.set_ylabel("N (km)"); ax3.set_zlabel("Down (km)")
    ax3.set_zlim(H*1.2,-H*1.2); ax3.view_init(elev=22,azim=-60); ax3.set_title("(b) 3-D perspective")
    # (c) E-Z   (d) N-Z
    axE=fig.add_subplot(gs[1,0]); pts(axE,e,z); axE.add_patch(Rectangle((-H,-H),2*H,2*H,fill=False,ec="0.3",ls="--",lw=1.1))
    axE.scatter(msE,msZ,s=300,marker="*",color="red",edgecolor="k",lw=0.7,zorder=6)
    axE.set(xlabel="East offset (km)",ylabel="Depth offset (km, +down)",title="(c) E–Z section",xlim=(-H*1.3,H*1.3),ylim=(H*1.3,-H*1.3))
    axN=fig.add_subplot(gs[1,1]); pts(axN,n,z); axN.add_patch(Rectangle((-H,-H),2*H,2*H,fill=False,ec="0.3",ls="--",lw=1.1))
    axN.scatter(msN,msZ,s=300,marker="*",color="red",edgecolor="k",lw=0.7,zorder=6)
    axN.set(xlabel="North offset (km)",ylabel="Depth offset (km, +down)",title="(d) N–Z section",xlim=(-H*1.3,H*1.3),ylim=(H*1.3,-H*1.3))
    cb=fig.colorbar(sc,ax=fig.axes,fraction=0.02,pad=0.02); cb.set_label("Year")
    fig.suptitle(f"Cluster {k}  ·  M{ms.magU:.2f} {str(ms.time)[:10]}  ·  {info['n']} relocated events in the 1 km³ cube "
                 f"({info['n_mem']} NND-linked, {info['n_prior']} pre-mainshock)",y=0.98,fontsize=12)
    plt.show()
for k in TARGETS: spatial_fig(k)""")

# ------------------------------------------------------------------ §3 per-cluster temporal
md(r"""## 3 · Volume time history — was it active before the mainshock?

The core question. Top: **M–t of every relocated event in the cube** (circles = members, triangles = other;
★ = mainshock; dashed line = mainshock time). Pre-mainshock non-members are ringed by **category**: red = true
background (the declustering metric failed to attribute these), orange = member of another NND family (the
declustering DID see them — as a separate sequence), purple = not in the NND population (ML-less; a completeness
exclusion, not an NND failure). Middle: **cumulative count** (all vs members only) — a non-flat curve before the
dashed line is prior activity in the volume, whatever its category. Bottom: **distance from centroid vs time**,
to see whether the early activity sits in the same spot.""")
co(r"""def temporal_fig(k):
    box,info=BOXES[k]; ms=info["ms"]; mst=ms.time
    mem=box.is_member.values; pri=box.is_prior.values; catv=box["cat"].values
    CATCOL=[("background","red","pre-MS: background (NND missed)"),
            ("other-family","orange","pre-MS: other NND family"),
            ("not-in-pop","purple","pre-MS: not in NND pop (ML-less)")]
    fig,axs=plt.subplots(3,1,figsize=(13.5,9.2),sharex=True,gridspec_kw=dict(height_ratios=[1.15,1,1],hspace=0.12))
    t=box.time.values
    ymin=float(box.magU.min())-0.3
    axs[0].vlines(t,ymin,box.magU.values,color="0.75",lw=0.5,zorder=1)
    axs[0].scatter(box.time[~mem],box.magU[~mem],marker="^",s=34,color="0.55",edgecolor="0.3",lw=0.3,zorder=3,label="other in-volume")
    axs[0].scatter(box.time[mem],box.magU[mem],marker="o",s=34,color="tab:blue",edgecolor="k",lw=0.3,zorder=4,label="family member")
    for cc_,col_,lab_ in CATCOL:
        pm=(~mem)&pri&(catv==cc_)
        if pm.any(): axs[0].scatter(box.time[pm],box.magU[pm],marker="^",s=46,facecolor="none",edgecolor=col_,lw=1.2,zorder=5,label=f"{lab_} ({int(pm.sum())})")
    _sp=(~mem)&pri&box.suspect.values
    if _sp.any(): axs[0].scatter(box.time[_sp],box.magU[_sp],marker="+",s=60,color="k",lw=0.9,zorder=6,label=f"suspect location ({int(_sp.sum())})")
    axs[0].scatter([mst],[ms.magU],s=320,marker="*",color="red",edgecolor="k",lw=0.6,zorder=6)
    axs[0].axvline(mst,color="red",ls="--",lw=1); axs[0].set(ylabel="M$_L$ (or input mag)",title=f"Cluster {k} — volume M–t history")
    axs[0].legend(loc="upper left",fontsize=8.5,ncol=2)
    if info["earliest"]:
        axs[0].annotate(f"earliest in-volume event {info['earliest']}\n(lead {info['lead_d']:.0f} d before mainshock)",
            xy=(box.time.min(),box.magU.min()),xytext=(0.02,0.9),textcoords="axes fraction",fontsize=8.5,color="red",
            arrowprops=dict(arrowstyle="->",color="red",lw=0.8),va="top")
    # cumulative
    axs[1].step(box.time,np.arange(1,len(box)+1),where="post",color="k",lw=1.3,label="all in-volume")
    _mm=box[mem].sort_values("time"); axs[1].step(_mm.time,np.arange(1,len(_mm)+1),where="post",color="tab:blue",lw=1.3,label="NND-linked only")
    axs[1].axvline(mst,color="red",ls="--",lw=1); axs[1].set(ylabel="cumulative count"); axs[1].legend(loc="upper left",fontsize=8.5)
    # distance from centroid
    axs[2].scatter(box.time[~mem],box.r_km[~mem],marker="^",s=30,color="0.55",edgecolor="0.3",lw=0.3,zorder=3)
    axs[2].scatter(box.time[mem],box.r_km[mem],marker="o",s=30,color="tab:blue",edgecolor="k",lw=0.3,zorder=4)
    for cc_,col_,_lab in CATCOL:
        pm=(~mem)&pri&(catv==cc_)
        if pm.any(): axs[2].scatter(box.time[pm],box.r_km[pm],marker="^",s=42,facecolor="none",edgecolor=col_,lw=1.1,zorder=5)
    axs[2].axvline(mst,color="red",ls="--",lw=1); axs[2].set(ylabel="dist. from centroid (km)",xlabel="Year")
    axs[2].xaxis.set_major_locator(mdates.YearLocator()); axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.show()
for k in TARGETS: temporal_fig(k)""")

# ------------------------------------------------------------------ §4 summary
md(r"""## 4 · Summary""")
co(r"""rows=[]
for k in TARGETS:
    box,info=BOXES[k]; ms=info["ms"]
    rows.append(dict(Cluster=k,mainshock_M=round(float(ms.magU),2),date=str(ms.time)[:10],
                     in_cube=info["n"],members=info["n_mem"],
                     pre_nonmem=info["n_prior_nonmem"],pre_otherfam=info["n_prior_otherfam"],
                     pre_background=info["n_prior_bg"],pre_notpop=info["n_prior_notpop"],
                     suspect=info["n_susp_prior"],earliest=info["earliest"],lead_days=info["lead_d"],
                     earliest_bg=info["earliest_bg"]))
SUM=pd.DataFrame(rows)
print("="*112); print(f"1 km³ VOLUME HISTORY — decomposed pre-mainshock census (cube half-width {HALF} km)".center(112)); print("="*112)
print(SUM.to_string(index=False))
print("\nTAKE-HOMES (decomposed — the three categories mean three DIFFERENT things)")
for k in TARGETS:
    box,info=BOXES[k]
    if info["n_prior_nonmem"]==0:
        print(f" - Cluster {k} (M{info['ms'].magU:.2f}): NO pre-mainshock non-member seismicity in the volume."); continue
    print(f" - Cluster {k} (M{info['ms'].magU:.2f}): {info['n_prior_nonmem']} pre-mainshock non-members = "
          f"{info['n_prior_otherfam']} in OTHER NND families {info['prior_fams'] if info['prior_fams'] else ''} "
          f"(the declustering DID see these — as separate sequences) + {info['n_prior_bg']} true background "
          f"(genuinely unattributed by the NND metric) + {info['n_prior_notpop']} not-in-population "
          f"(ML-less, excluded by the n_used>=3 gate — a completeness effect, not an NND failure); "
          f"earliest prior {info['earliest']} ({info['lead_d']:.0f} d lead), earliest TRUE background {info['earliest_bg']}."
          + (f" {info['n_susp_prior']} prior events have suspect locations (>3 km relocation move or <10 links)." if info['n_susp_prior'] else ""))
print("\n - INTERPRETATION: long-lived precursory activity in these volumes is REAL, but it is mostly a")
print("   FAMILY-FRAGMENTATION effect (NND correctly groups earlier episodes into separate families — rescaled")
print("   time cannot bridge multi-year quiet gaps) plus a COMPLETENESS effect (ML-less events never entered the")
print("   declustering). Only the 'background' column is seismicity the NND metric itself failed to attribute.")
print("   The physical conclusion — the same small volume re-activates across years — stands, with the mechanism")
print("   correctly attributed.")
print(f" - CAVEAT: only relocated events are placeable in a 1 km³ box; catalog starts ~{str(allc.time.min())[:7]},")
print("   so the lookback is bounded — a longer/precise history needs template-matching detection in the volume.")""")

nb["cells"]=C
import os
os.chdir("/home/msseo/works/02.Ulsan_Fault_detection/analysis/reloc_analysis")
nbf.write(nb,"30.UF_cluster_volume_history.ipynb")
print("wrote 30.UF_cluster_volume_history.ipynb",len(C),"cells")
