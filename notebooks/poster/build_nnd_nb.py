#!/usr/bin/env python
"""Generate notebooks/poster/01.NND_declustering.ipynb — background vs clustered seismicity.

Poster analysis 1: separate SPONTANEOUS (background) from CLUSTERED events in the established
KS/KG-era dt.cc-relocated catalog using nearest-neighbour distance (Zaliapin & Ben-Zion 2008/2013).

Everything numerical comes from the CANONICAL, cross-validated module
`kma_absolute_location.nnd` (16.kma_absolute_location) — nothing is reimplemented here. In
particular `nnd.decimal_year` is the only decimal-year source (a day-resolution reimplementation
once faked cluster extents by force-linking same-day events at eta=0).

Parameters follow the Ulsan-adopted values: Df=1.2 (data-driven, nb27 — not the generic Z&B 1.6),
b=1.0 fixed, 3-D metric, link_rmax=1 km, no Mc cut (eta depends only on the PARENT magnitude, so
small events are valid children).

    python notebooks/poster/build_nnd_nb.py       # (re)writes the unexecuted notebook

Kernel: base (pygmt + sklearn). Catalog-scale (~2.5k events) — seconds, safe alongside detections.
Outputs: figs/N1..N5 (PDF+PNG 600 dpi), figs/nnd_numbers.csv (citable), figs/nnd_events.csv
(event_idx + cluster label + background flag, for later poster notebooks).
"""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "01.NND_declustering.ipynb")
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Background vs clustered seismicity — nearest-neighbour declustering

Separates **spontaneous (background)** from **clustered (triggered)** events in the established
KS/KG-era dt.cc-relocated Ulsan-Fault catalog, using the Zaliapin & Ben-Zion nearest-neighbour
distance

$$\eta_{ij} = \tau_{ij}\,(r_{ij})^{D}\,10^{-b M_i}$$

with the rescaled coordinates $T = \tau\,10^{-qbM}$ and $R = r^{D}10^{-qbM}$, so $\eta = TR$.
Clustered pairs occupy the small-$\eta$ mode; background events the large-$\eta$ mode. The threshold
$\eta_0$ is the **valley between the two modes** of a 2-component Gaussian mixture on
$\log_{10}\eta$ — fitted, not chosen — and validated against a randomized-catalog null.

All computation uses the canonical `kma_absolute_location.nnd` module (cross-validated 1:1 against
Goebel's `clustering-analysis`). **Kernel: base.**""")

co(r'''# ============================ PARAMETERS ============================
REPO   = "/home/msseo/works/02.Ulsan_Fault_detection"
NNDPKG = "/home/msseo/works/16.kma_absolute_location"

# --- NND parameters (Ulsan-adopted; see the markdown note below each) ---
DF_UF   = 1.2      # fractal dimension: DATA-DRIVEN for these hypocentres (nb27 box-counting),
                   # not the generic Z&B 1.6. Sensitivity to this choice is shown in N2b.
B_NND   = 1.0      # fixed b (the catalog b drifts in time; fixing it keeps eta comparable)
METRIC  = "3d"     # depth is resolved in the dt.cc catalog
MMIN    = None     # NO Mc cut: eta depends only on the PARENT magnitude, so small events are
                   # valid children; cutting them would discard real cluster membership
LINKR   = 1.0      # km, family-linking cap (module default 10 km is whole-peninsula scale)
MERGE_D, MERGE_Z = 5.0, 5.0     # spatial_merge: co-located but temporally separate families

# --- catalog selection (user decision 2026-08-16) ---
# NND is driven by inter-event DISTANCE, so select on LOCATION precision, not magnitude quality:
#   * keep every dt.cc-resolved event (waveform-precise relative locations), and
#   * add back the dt.ct-only events with ML >= DTCT_ML_MIN, because large events dominate
#     triggering and must not be missing from the parent population (the catalog's largest,
#     the 2014-09-23 ML 3.93, is dt.ct-only: it found no waveform-similar neighbour).
# NUSED_MIN=1 deliberately: `n_used` is the ML-quality count, and magnitude enters eta only as a
# weak 10^(-q*b*M_parent) weight — cutting n_used<3 would discard 624 well-located cc events (35%
# of the catalog) for a magnitude-term technicality. Set NUSED_MIN=3 to reproduce the earlier run.
SELECTION    = "dtcc_plus_large_dtct"           # "dtcc_plus_large_dtct" | "all" | "dtcc_only"
DTCT_ML_MIN  = 2.0                              # ML floor for the dt.ct-only events added back
NUSED_MIN    = 1                                # ML-quality floor (1 = no cut; see note above)
UF_BOX  = (129.25, 129.55, 35.60, 35.90)        # selection box
MAP_BOX = (129.20, 129.58, 35.58, 35.92)        # display box (reloc moves events slightly outside)
GJ_T    = "2016-09-12T11:32:54"                 # Gyeongju Mw 5.5 (UTC), OUTSIDE the box ~10 km W
GJ_WINDOW_M = 6                                 # months of elevated response
YEAR_RANGE  = (2010, 2024)                      # full catalog window for the time axes (archive
                                                # ends 2024 day 305; no 2025+ data exists)

# --- smoothed spatial density (N7) — SAME construction as the established
# build_background_density_anim_nb.py: histogram2d on a fixed lon/lat grid, then gaussian_filter,
# rendered with Blues + PowerNorm(0.5) (sqrt scale: honest and keeps the low end legible).
# Smoothing width is set FROM THE DATA: formal HypoDD errors ~100 m (median ex/ey = 0.1 km),
# median inter-event spacing 20 m, region ~27 x 32 km. SP=0.004 deg (~0.4 km cells) with SIG=1.5
# cells gives an effective ~0.6 km kernel = 6x the location uncertainty, resolving the ~1 km
# fault-patch scale without rendering location noise as structure.
SP  = 0.004                                     # grid step (deg)
SIG = 1.5                                       # Gaussian sigma (cells) -> ~0.6 km
SIG_ALT = (0.75, 3.0)                           # sensitivity row: ~0.3 km and ~1.2 km
FAULTS_GMT = REPO + "/data/hypoinv/faults_lonlat.gmt"
COAST_GMT  = REPO + "/analysis/reloc_analysis/coastline_lonlat.gmt"
DENS_CMAP  = "hot_r"                            # hot, reversed: white (empty) -> dark red (dense),
                                                # so the background stays light and the overlaid
                                                # open circles remain readable on top
MAG_S0, MAG_K = 2.0, 9.0                        # marker area = MAG_S0 * 10**(MAG_K*... ) see below:
                                                # s = MAG_S0 * 2**(2*ML) -> area doubles per 0.5 ML,
                                                # i.e. radius ~ rupture length scaling

CAT   = REPO + "/analysis/local_magnitudes/catalog_ml_heo_ufonly_reloc.csv"
BLAST = REPO + "/analysis/local_magnitudes/blast_event_idx_deblast.csv"
MLC   = REPO + "/analysis/local_magnitudes/catalog_ml_heo_const.csv"
FIGS  = REPO + "/notebooks/poster/figs"

import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import gaussian_kde
import pygmt
sys.path.insert(0, NNDPKG); from kma_absolute_location import nnd   # CANONICAL module

try:
    fm.findfont("Helvetica", fallback_to_default=False); plt.rcParams["font.family"]="Helvetica"
except Exception:
    plt.rcParams["font.family"]="DejaVu Sans"
plt.rcParams.update({"figure.dpi":120,"savefig.dpi":600,"legend.framealpha":1.0,
                     "legend.facecolor":"white","axes.unicode_minus":False})
os.makedirs(FIGS, exist_ok=True)

def save(fig, name, pygmt_fig=False):
    for ext in ("pdf","png"):
        p=os.path.join(FIGS,f"{name}.{ext}")
        (fig.savefig(p) if pygmt_fig else fig.savefig(p, bbox_inches="tight"))
    print(f"  saved figs/{name}.pdf + .png")

print(f"NND parameters: Df={DF_UF}  b={B_NND}  metric={METRIC}  mmin={MMIN}  link_rmax={LINKR} km")''')

md(r"""## 1 — Catalog: de-blasted, quality-cut, in the NND schema

The input is the HypoDD-relocated catalog. **Every event carries a local magnitude** — `n_used` is
the ML-*quality* station count, not magnitude availability.

Selection (see the parameters cell): all **dt.cc-resolved** events (waveform-precise relative
locations) **plus** the **dt.ct-only events with ML $\geq$ 2.0**, since large events dominate
triggering and must be present in the parent population — the catalog's largest event
(2014-09-23, ML 3.93) is dt.ct-only. Blasts are removed. `t_year` comes from `nnd.decimal_year`
(exact year length, second precision) — never re-derived.""")

co(r'''rl = pd.read_csv(CAT)
blast = set(pd.read_csv(BLAST).event_idx.dropna().astype(int))
n0 = len(rl)
rl = rl[~rl.event_idx.isin(blast)].copy()                       # DE-BLAST
g  = rl[rl.n_used >= NUSED_MIN].dropna(subset=["lat","lon","depth","ml_ufcorr_reloc"]).copy()
# --- relocation-level selection (see the parameters cell) ---
n_pre = len(g)
if SELECTION == "dtcc_plus_large_dtct":
    keep = g.is_dtcc | (g.ml_ufcorr_reloc >= DTCT_ML_MIN)
elif SELECTION == "dtcc_only":
    keep = g.is_dtcc
else:
    keep = pd.Series(True, index=g.index)
n_cc  = int((g.is_dtcc & keep).sum())
n_ct  = int((~g.is_dtcc & keep).sum())
g = g[keep].copy()
print(f"selection '{SELECTION}': {n_pre} -> {len(g)}  "
      f"(dt.cc-resolved {n_cc}; dt.ct-only kept {n_ct} at ML>={DTCT_ML_MIN})")
g["event_time"] = pd.to_datetime(g.event_time, format="ISO8601", utc=True, errors="coerce")
g = g.dropna(subset=["event_time"]).sort_values("event_time").reset_index(drop=True)

# --- the canonical NND schema (rename + decimal year + event_id) ---
g["event_id"] = g.event_idx.astype(int).astype(str)
g["t_year"]   = g.event_time.dt.strftime("%Y%m%d%H%M%S").map(nnd.decimal_year)   # CANONICAL
g = g.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep",
                      "ml_ufcorr_reloc":"kma_mag"})
g = g.sort_values("t_year").reset_index(drop=True)

print(f"catalog {n0} -> de-blast {n0-len(rl)} removed -> n_used>={NUSED_MIN} -> {len(g)} events")
print(f"  NOTE: every event carries a local magnitude (ml_ufcorr_reloc); n_used is the ML-quality "
      f"station count, not magnitude availability")
print(f"span {g.event_time.min():%Y-%m-%d} .. {g.event_time.max():%Y-%m-%d}   "
      f"ML {g.kma_mag.min():.1f}..{g.kma_mag.max():.1f}")''')

md(r"""## 2 — Run the NND and fit $\eta_0$""")

co(r'''nd = nnd.compute_nnd(g, b=B_NND, D=DF_UF, mmin=MMIN, metric=METRIC)
e0, info = nnd.fit_eta0(nd.eta.values, method="gmm")
LE0 = float(np.log10(e0))
labels = nnd.build_families(nd, e0, g.event_id.values, link_rmax_km=LINKR)
merged = nnd.spatial_merge(g, labels, dmax_km=MERGE_D, dz_km=MERGE_Z)
g["Cluster"]        = g.event_id.map(labels).fillna(-1).astype(int)
g["Cluster_merged"] = g.event_id.map(merged).fillna(-1).astype(int)
g["background"]     = g.Cluster < 0

n_fam = int(g.Cluster.max())+1; n_clu = int((g.Cluster>=0).sum()); n_bg = int(g.background.sum())
print(f"log10(eta0) = {LE0:.2f}   (GMM means {np.round(info['means'],2)}, weights {np.round(info['weights'],2)})")
print(f"clustered {n_clu}/{len(g)} ({100*n_clu/len(g):.0f}%)   background {n_bg} ({100*n_bg/len(g):.0f}%)"
      f"   families {n_fam} (merged {int(g.Cluster_merged.max())+1})")

# --- VERIFICATION: the two background definitions must agree ---
bg_alt = ~g.event_id.isin(set(nd.loc[nd.eta < e0, "event_id"]))
assert int(bg_alt.sum()) == n_bg or abs(int(bg_alt.sum()) - n_bg) <= n_fam, \
    f"background mismatch: labels {n_bg} vs eta-threshold {int(bg_alt.sum())}"
assert info["means"][0] < LE0 < info["means"][1], "eta0 must lie between the two GMM modes"
print(f"  checks OK: eta0 between GMM modes; eta-threshold background {int(bg_alt.sum())} "
      f"(differs from label-based only by family roots, expected)")''')

md(r"""## N1 — Rescaled time–distance density, with $\eta_0$

The signature plot of the method. Two lobes: the lower-left cloud is **clustered** (short rescaled
time *and* distance), the upper-right is **background**. The dashed anti-diagonal is $\eta_0$; the
axes are equal so it renders at exactly 45°.""")

co(r'''def rt_panel(ax, nd, e0, label):
    lt=nd.logT.values; lr=nd.logR.values; ok=np.isfinite(lt)&np.isfinite(lr); lt,lr=lt[ok],lr[ok]
    le0=np.log10(e0); binx=biny=0.1
    Tlo,Thi,Rlo,Rhi=-8.0,2.0,-6.0,4.0      # equal 10-unit spans -> eta0 at 45 deg
    Tb=np.arange(Tlo,Thi+binx,binx); Rb=np.arange(Rlo,Rhi+biny,biny); XX,YY=np.meshgrid(Tb,Rb)
    ZZ=gaussian_kde(np.vstack([lt,lr]))(np.vstack([XX.ravel(),YY.ravel()])).reshape(XX.shape)*binx*biny*len(lt)
    pc=ax.pcolormesh(XX,YY,ZZ,cmap=plt.cm.RdYlGn_r,shading="auto")
    ax.plot([Tlo,Thi],-np.array([Tlo,Thi])+le0,"-",lw=2.5,color="w")
    ax.plot([Tlo,Thi],-np.array([Tlo,Thi])+le0,"--",lw=1.5,color="0.3",
            label=f"$\\eta_0$ (log$_{{10}}$={le0:.2f})")
    ax.set(xlabel="Rescaled time  log$_{10}$ T", ylabel="Rescaled distance  log$_{10}$ R",
           title=label, xlim=(Tlo,Thi), ylim=(Rlo,Rhi))
    ax.set_aspect("equal", adjustable="box"); ax.legend(loc="lower left", fontsize=8)
    return pc

fig,ax=plt.subplots(figsize=(7.4,6.8))
pc=rt_panel(ax,nd,e0,f"Ulsan Fault, dt.cc catalog  (D$_f$={DF_UF}, b={B_NND}, 3-D)")
cb=fig.colorbar(pc,ax=ax,fraction=0.046,pad=0.04); cb.set_label("Event pairs")
save(fig,"N1_rt_density"); plt.show()''')

md(r"""## N2 — The threshold is a real valley, not a chosen cut

Left: $\log_{10}\eta$ is **bimodal**; the GMM components and their crossing (= $\eta_0$) are drawn.
Right: the same distribution against a **randomized catalog** (uniform times, uniform locations,
shuffled magnitudes) which destroys all space–time correlation — its clustered mode disappears,
confirming the observed small-$\eta$ mode is physical.""")

co(r'''fig,axes=plt.subplots(1,2,figsize=(13,5))
le = np.log10(nd.eta.values); le = le[np.isfinite(le)]

ax=axes[0]
ax.hist(le,bins=45,density=True,color="0.85",ec="w",label="observed")
xs=np.linspace(le.min(),le.max(),500)
for j,(mu,sg,w,nm,c) in enumerate(zip(info["means"],info["sigmas"],info["weights"],
                                      ["clustered","background"],["#d62728","#1f77b4"])):
    ax.plot(xs, w/(sg*np.sqrt(2*np.pi))*np.exp(-0.5*((xs-mu)/sg)**2), color=c, lw=2,
            label=f"{nm} (w={w:.2f})")
ax.axvline(LE0,color="0.2",ls="--",lw=1.6,label=f"$\\eta_0$ = {LE0:.2f}")
ax.set(xlabel="log$_{10}$ $\\eta$", ylabel="Density", title="Bimodal NND + GMM decomposition")
leg=ax.legend(fontsize=8); leg.set_zorder(10)

ax=axes[1]
try:
    mc_est = nnd.estimate_mc(g.kma_mag.values)
    e0_null, le_null = nnd.randomized_eta0(g, mc=mc_est, b=B_NND, D=DF_UF, metric=METRIC, n_boot=3)
    ax.hist(le,bins=45,density=True,color="0.85",ec="w",label="observed")
    ax.hist(le_null[np.isfinite(le_null)],bins=45,density=True,histtype="step",color="#2ca02c",
            lw=2,label="randomized null")
    ax.axvline(LE0,color="0.2",ls="--",lw=1.6,label=f"observed $\\eta_0$ = {LE0:.2f}")
    ax.axvline(np.log10(e0_null),color="#2ca02c",ls=":",lw=1.6,
               label=f"null $\\eta_0$ = {np.log10(e0_null):.2f}")
    print(f"randomized null: log10 eta0 = {np.log10(e0_null):.2f} (observed {LE0:.2f}; "
          f"Mc used {mc_est:.2f})")
except Exception as exc:
    ax.text(.5,.5,f"randomized null unavailable:\n{exc}",ha="center",va="center",transform=ax.transAxes)
ax.set(xlabel="log$_{10}$ $\\eta$", ylabel="Density", title="Observed vs randomized catalog")
leg=ax.legend(fontsize=8); leg.set_zorder(10)
save(fig,"N2_eta0_threshold"); plt.show()''')

md(r"""## N3 — Where the two populations sit

Background (spontaneous) events versus clustered (triggered) events. The distinction the poster
rests on: whether persistent activity is distributed background or a few episodic sequences.""")

co(r'''bg = g[g.background]; cl = g[~g.background]
fig=pygmt.Figure()
for i,(lab,d,color) in enumerate([(f"Background / spontaneous (n={len(bg)})", bg, "#1f77b4"),
                                  (f"Clustered / triggered (n={len(cl)})",   cl, "#d62728")]):
    if i: fig.shift_origin(xshift="9.6c")
    fig.basemap(region=list(MAP_BOX), projection="M9c",
                frame=["af", ("WSen" if i==0 else "wSen")+f'+t"{lab}"'])
    fig.coast(shorelines="0.4p,gray30", land="gray98", water="azure1")
    lon0,lon1,la0,la1 = UF_BOX
    fig.plot(x=[lon0,lon1,lon1,lon0,lon0], y=[la0,la0,la1,la1,la0], pen="0.7p,gray45,2_2")
    fig.plot(x=d.svi_lon, y=d.svi_lat, style="c0.09c", fill=color, pen="0.1p,gray20")
    fig.basemap(map_scale="jBL+w10k+o0.4c/0.4c")
save(fig,"N3_background_vs_clustered",pygmt_fig=True); fig.show(width=1000)''')

md(r"""## N4 — Background is steady; clusters are episodic

Cumulative counts and annual rates for the two populations, with the 2016 Gyeongju mainshock marked.
This is the quantitative form of the poster's central claim about persistent intraplate seismicity.""")

co(r'''GJ = pd.Timestamp(GJ_T, tz="UTC")
fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,
                      gridspec_kw=dict(height_ratios=[1.25,1]))
ax=axes[0]
ax.plot(bg.event_time, np.arange(1,len(bg)+1), color="#1f77b4", lw=1.8, label=f"background (n={len(bg)})")
ax.plot(cl.event_time, np.arange(1,len(cl)+1), color="#d62728", lw=1.8, label=f"clustered (n={len(cl)})")
ax.axvline(GJ,color="0.25",lw=1.2); ax.text(GJ,ax.get_ylim()[1],"  Gyeongju Mw 5.5",fontsize=9,va="top")
ax.set_ylabel("Cumulative events"); ax.set_title("Background vs clustered seismicity through time")
leg=ax.legend(loc="upper left"); leg.set_zorder(10)

ax=axes[1]
# bars placed on the SAME datetime axis as the cumulative panel (sharex=True): using year numbers
# here would collide with the datetime scale and blow up the render.
yrs=np.arange(YEAR_RANGE[0], YEAR_RANGE[1]+1)
wb=bg.event_time.dt.year.value_counts().reindex(yrs,fill_value=0)
wc=cl.event_time.dt.year.value_counts().reindex(yrs,fill_value=0)
centres=[pd.Timestamp(f"{y}-07-01", tz="UTC") for y in yrs]
w=pd.Timedelta(days=150)
ax.bar([c-w/2 for c in centres], wb.values, width=w, color="#1f77b4", label="background")
ax.bar([c+w/2 for c in centres], wc.values, width=w, color="#d62728", label="clustered")
ax.axvline(GJ, color="0.25", lw=1.2)
ax.set(xlabel="Year", ylabel="Events / year")
leg=ax.legend(); leg.set_zorder(10)
# full catalog window on the shared (datetime) axis — 2010-2024, not just the data extent
ax.set_xlim(pd.Timestamp(f"{YEAR_RANGE[0]}-01-01", tz="UTC"),
            pd.Timestamp(f"{YEAR_RANGE[1]}-12-31", tz="UTC"))
save(fig,"N4_time_series"); plt.show()''')

md(r"""## N5 — Family sizes and the largest sequences

How the clustered population is organized: a few large families (aftershock sequences / swarms) and
many small ones. The largest families' footprints test the "persistent subregions" picture.""")

co(r'''sizes = g[g.Cluster>=0].Cluster.value_counts().sort_values(ascending=False)
fig,axes=plt.subplots(1,2,figsize=(13,5.2))
ax=axes[0]
ax.loglog(np.arange(1,len(sizes)+1), sizes.values, "o", ms=4, color="#d62728")
ax.set(xlabel="Family rank", ylabel="Family size (events)",
       title=f"Family-size distribution ({len(sizes)} families)")
ax.grid(alpha=.3, which="both")

ax=axes[1]
TOPN=5
top=sizes.head(TOPN).index.tolist()
ax.scatter(bg.svi_lon,bg.svi_lat,s=5,c="0.8",label="background")
for k,cid in enumerate(top):
    d=g[g.Cluster==cid]
    ax.scatter(d.svi_lon,d.svi_lat,s=14,label=f"family {cid} (n={len(d)})")
ax.set(xlabel="Longitude", ylabel="Latitude", title=f"{TOPN} largest families",
       xlim=(MAP_BOX[0],MAP_BOX[1]), ylim=(MAP_BOX[2],MAP_BOX[3]))
ax.set_aspect(1/np.cos(np.radians(35.75)))
leg=ax.legend(fontsize=7,loc="upper left"); leg.set_zorder(10)
save(fig,"N5_families"); plt.show()

print("largest families:", sizes.head(TOPN).to_dict())''')

md(r"""## N7 — Smoothed spatial density

Density of each population on a **shared colour scale**, so the two maps are directly comparable,
with **surface fault traces** and the coastline overlaid. Construction and styling follow the
established `build_background_density_anim_nb.py`: `histogram2d` on a 0.004° grid, `gaussian_filter`
smoothing, `Blues` with a **square-root (PowerNorm 0.5)** stretch so the low end stays legible
without exaggerating the peaks.

Smoothing width comes from the data: formal HypoDD errors are ~100 m and median event spacing 20 m,
so σ ≈ 0.7 km (≈7× the location uncertainty) resolves the ~1 km fault-patch scale without rendering
location noise as structure.

Individual epicentres are overlaid as **open circles scaled by magnitude** (area doubles per 0.5 ML,
so the radius follows rupture-dimension scaling) — the density shows where activity concentrates,
the circles show the events themselves and which of them are large.""")

co(r'''from scipy.ndimage import gaussian_filter
import matplotlib as mpl

def _load_segs(path):
    """GMT multi-segment lon/lat file -> list of (N,2) arrays (same loader as the density-animation nb)."""
    segs, cur = [], []
    if not os.path.exists(path):
        return segs
    for ln in open(path):
        if ln.startswith(">"):
            if len(cur) > 1: segs.append(np.array(cur))
            cur = []; continue
        p = ln.split()
        if len(p) >= 2:
            try: cur.append([float(p[0]), float(p[1])])
            except ValueError: pass
    if len(cur) > 1: segs.append(np.array(cur))
    return segs

FSEG = _load_segs(FAULTS_GMT); CSEG = _load_segs(COAST_GMT)
print(f"overlays: {len(FSEG)} fault segments, {len(CSEG)} coastline segments")

LAT0 = float(g.svi_lat.mean())
CELL_KM2 = (SP*111.0*np.cos(np.radians(LAT0))) * (SP*111.0)
XB = np.arange(MAP_BOX[0], MAP_BOX[1]+SP, SP); YB = np.arange(MAP_BOX[2], MAP_BOX[3]+SP, SP)
EXT = [XB[0], XB[-1], YB[0], YB[-1]]
ASP = 1.0/np.cos(np.radians(LAT0))

def dens(d, sig):
    """events / km^2, smoothed — histogram2d + gaussian_filter (house style)."""
    H,_,_ = np.histogram2d(d.svi_lon.values, d.svi_lat.values, bins=[XB, YB])
    return gaussian_filter(H, sig).T / CELL_KM2

def mag_size(ml):
    """Marker AREA scaled with magnitude: area doubles per 0.5 ML, so the circle RADIUS grows
    like 10^(0.5*ML) — the same scaling as rupture dimension. Clipped at ML 0 so the smallest
    events stay visible."""
    return MAG_S0 * 2.0**(2.0*np.clip(np.asarray(ml, float), 0.0, None))

def basemap(ax, title, quakes=None, ring="0.15"):
    for sgm in FSEG: ax.plot(sgm[:,0], sgm[:,1], color="0.45", lw=0.7, zorder=3)
    for sgm in CSEG: ax.plot(sgm[:,0], sgm[:,1], color="black", lw=0.9, zorder=4)
    ax.plot([UF_BOX[0],UF_BOX[1],UF_BOX[1],UF_BOX[0],UF_BOX[0]],
            [UF_BOX[2],UF_BOX[2],UF_BOX[3],UF_BOX[3],UF_BOX[2]], "--", lw=0.9, color="0.3", zorder=5)
    if quakes is not None and len(quakes):
        ax.scatter(quakes.svi_lon, quakes.svi_lat, s=mag_size(quakes.kma_mag),
                   facecolor="none", edgecolor=ring, lw=0.45, alpha=0.85, zorder=6)
    ax.set(xlim=(MAP_BOX[0],MAP_BOX[1]), ylim=(MAP_BOX[2],MAP_BOX[3]),
           xlabel="Longitude", ylabel="Latitude", title=title)

def mag_legend(ax, mags=(1.0, 2.0, 3.0), loc="lower left"):
    h=[plt.scatter([],[],s=mag_size(m),facecolor="none",edgecolor="0.15",lw=0.6,
                   label=f"ML {m:.0f}") for m in mags]
    leg=ax.legend(handles=h, loc=loc, labelspacing=1.15, borderpad=0.8, frameon=True,
                  fontsize=8, title="Magnitude", title_fontsize=8)
    leg.set_zorder(10); return leg

Zb = dens(bg, SIG); Zc = dens(cl, SIG)
VMAX = float(np.nanmax([Zb.max(), Zc.max()]))          # SHARED scale -> the two are comparable

fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.4))
for ax, (Z, lab, n, d_) in zip(axes, [(Zb, "Background / spontaneous", len(bg), bg),
                                      (Zc, "Clustered / triggered",   len(cl), cl)]):
    im = ax.imshow(Z, origin="lower", extent=EXT, cmap=DENS_CMAP,
                   norm=mpl.colors.PowerNorm(0.5, vmin=0, vmax=VMAX),
                   aspect=ASP, interpolation="bilinear", zorder=1)
    basemap(ax, f"{lab} (n={n})", quakes=d_)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("Events / km$^2$")
mag_legend(axes[0])
fig.suptitle(f"Smoothed seismicity density — Gaussian sigma {SIG*SP*111:.1f} km "
             f"(relocation uncertainty ~0.1 km)", y=0.98)
fig.tight_layout()
for ext in ("pdf","png"):
    fig.savefig(os.path.join(FIGS, f"N7_spatial_density.{ext}"), bbox_inches="tight", dpi=400)
print("  saved figs/N7_spatial_density.pdf + .png")
plt.show()
print(f"peak density: background {Zb.max():.1f}  clustered {Zc.max():.1f} events/km^2")''')

md(r"""### N7b — smoothing sensitivity

The same clustered field at three smoothing widths. Features that persist across all three are
real; the single ridge that appears at the widest setting is a smoothing artifact, not a structure.""")

co(r'''fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
for ax, sg in zip(axes, (SIG_ALT[0], SIG, SIG_ALT[1])):
    Z = dens(cl, sg)
    im = ax.imshow(Z, origin="lower", extent=EXT, cmap=DENS_CMAP,
                   norm=mpl.colors.PowerNorm(0.5, vmin=0, vmax=Z.max()),
                   aspect=ASP, interpolation="bilinear", zorder=1)
    basemap(ax, f"Clustered, $\\sigma$ = {sg*SP*111:.1f} km", quakes=cl)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("Events / km$^2$")
fig.tight_layout()
for ext in ("pdf","png"):
    fig.savefig(os.path.join(FIGS, f"N7b_density_sensitivity.{ext}"), bbox_inches="tight", dpi=400)
print("  saved figs/N7b_density_sensitivity.pdf + .png")
plt.show()''')

md(r"""## Summary — numbers for the poster""")

co(r'''GJ_END = GJ + pd.DateOffset(months=GJ_WINDOW_M)
def rate_per_wk(d, t0, t1):
    n=int(((d.event_time>=t0)&(d.event_time<t1)).sum()); wk=(t1-t0).days/7.0
    return n, n/wk if wk else np.nan

rows=[]
rows.append(("events analysed (de-blasted)", f"{len(g):,}"))
rows.append(("selection", f"{SELECTION} (dt.cc {n_cc:,} + dt.ct-only ML>={DTCT_ML_MIN}: {n_ct})"))
rows.append(("NND parameters", f"Df={DF_UF}, b={B_NND}, {METRIC}, link_rmax={LINKR} km, no Mc cut"))
rows.append(("log10(eta_0)  [GMM valley]", f"{LE0:.2f}"))
rows.append(("background (spontaneous)", f"{n_bg:,} ({100*n_bg/len(g):.0f}%)"))
rows.append(("clustered (triggered)", f"{n_clu:,} ({100*n_clu/len(g):.0f}%)"))
rows.append(("families / after spatial merge", f"{n_fam} / {int(g.Cluster_merged.max())+1}"))
rows.append(("largest family size", f"{int(sizes.iloc[0])} events"))
for lab,(t0,t1) in [("pre-Gyeongju (12 mo)", (GJ-pd.DateOffset(months=12), GJ)),
                    (f"Gyeongju response ({GJ_WINDOW_M} mo)", (GJ, GJ_END)),
                    ("post-response (12 mo)", (GJ_END, GJ_END+pd.DateOffset(months=12)))]:
    nb_,rb=rate_per_wk(bg,t0,t1); nc_,rc=rate_per_wk(cl,t0,t1)
    rows.append((f"{lab}: background / clustered per week", f"{rb:.2f} / {rc:.2f}"))
S=pd.DataFrame(rows,columns=["quantity","value"])
print(S.to_string(index=False))
S.to_csv(os.path.join(FIGS,"nnd_numbers.csv"),index=False)

# event-level split for downstream poster notebooks
g[["event_idx","event_time","svi_lon","svi_lat","svi_dep","kma_mag",
   "Cluster","Cluster_merged","background"]].to_csv(os.path.join(FIGS,"nnd_events.csv"),index=False)
print(f"\n-> figs/nnd_numbers.csv (cite these)  +  figs/nnd_events.csv ({len(g):,} rows, "
      f"reusable background/cluster labels)")''')

nb = nbf.v4.new_notebook(cells=cells,
                         metadata={"kernelspec":{"name":"python3","display_name":"Python 3 (base)",
                                                 "language":"python"}})
with open(NB, "w") as f:
    nbf.write(nb, f)
print(f"wrote {NB} ({len(cells)} cells, unexecuted)")
