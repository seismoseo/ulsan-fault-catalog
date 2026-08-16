#!/usr/bin/env python
"""Generate notebooks/poster/02.Kmeans_subregions.ipynb — spatial grouping + spontaneous fraction.

Poster analysis 2. Groups the SAME event set used by the NND notebook (01.NND_declustering) into
spatial sub-regions with K-means, then asks the question the poster actually needs:

    does each sub-region behave differently — some dominated by spontaneous (background)
    seismicity, others by triggered sequences?

Consistency by construction: the event set, its coordinates and its spontaneous/triggered labels are
READ FROM `figs/nnd_events.csv` (written by 01), so the two notebooks can never diverge. Re-run 01
first if the NND parameters change.

Clustering follows the established analysis/reloc_analysis/build_kmeans_clusters_dtcc_nb.py:
z-scored map-plane (lon,lat) K-means, K fixed with a silhouette-vs-K scan shown for context, and a
3-D (lon,lat,depth) labelling carried alongside for the depth-structure comparison.

    python notebooks/poster/build_kmeans_nb.py [--k 7]

Kernel: base. Seconds to run; safe alongside the detections.
"""
import argparse
import os
import nbformat as nbf

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--k", type=int, default=7, help="number of K-means clusters (default 7)")
_ap.add_argument("--events", default="nnd_events.csv",
                 help="NND event file in figs/ (use nnd_events_Df1p6.csv for the Df=1.6 variant)")
_A = _ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "02.Kmeans_subregions.ipynb")
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Spatial sub-regions and their spontaneous/triggered character

Groups the **same events used in the NND analysis** into spatial sub-regions with K-means, then
measures how the **spontaneous (background) fraction varies between them**.

The motivating question for the poster: is the Ulsan Fault volume homogeneous, or is it partitioned
into sub-regions with *distinct behaviour* — some sustaining steady spontaneous seismicity, others
producing episodic triggered sequences? That distinction is the difference between "one fault
operating at a steady rate" and "a network of patches at different stages of their own cycles".

**Inputs come from `figs/nnd_events.csv`** (written by `01.NND_declustering`), so the event set,
coordinates and spontaneous/triggered labels are identical by construction. **Kernel: base.**""")

co(r'''# ============================ PARAMETERS ============================
REPO = "/home/msseo/works/02.Ulsan_Fault_detection"
FIGS = REPO + "/notebooks/poster/figs"
EVENTS = FIGS + "/__EVENTS__"          # from 01.NND_declustering (same dataset, same labels)

K        = __K__          # number of spatial clusters (established value in the reloc_analysis nb)
K_SWEEP  = range(3, 11)   # silhouette scan, shown for context — K is FIXED, not chosen by silhouette
RANDOM_STATE = 0          # K-means is seeded: the labelling is reproducible

UF_BOX  = (129.25, 129.55, 35.60, 35.90)
MAP_BOX = (129.20, 129.58, 35.58, 35.92)
GJ_T    = "2016-09-12T11:32:54"
YEAR_RANGE = (2010, 2024)
FAULTS_GMT = REPO + "/data/hypoinv/faults_lonlat.gmt"
COAST_GMT  = REPO + "/analysis/reloc_analysis/coastline_lonlat.gmt"
TAB = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2",
       "#7f7f7f","#bcbd22","#17becf"]

import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

try:
    fm.findfont("Helvetica", fallback_to_default=False); plt.rcParams["font.family"]="Helvetica"
except Exception:
    plt.rcParams["font.family"]="DejaVu Sans"
plt.rcParams.update({"figure.dpi":120,"savefig.dpi":400,"legend.framealpha":1.0,
                     "legend.facecolor":"white","axes.unicode_minus":False})

def save(fig, name):
    for ext in ("pdf","png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight")
    print(f"  saved figs/{name}.pdf + .png")

def _load_segs(path):
    segs, cur = [], []
    if not os.path.exists(path): return segs
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
FSEG=_load_segs(FAULTS_GMT); CSEG=_load_segs(COAST_GMT)

d = pd.read_csv(EVENTS)
# parse explicitly: the column carries UTC offsets, which parse_dates leaves as object dtype
# (then `.dt` raises "Can only use .dt accessor with datetimelike values")
d["event_time"] = pd.to_datetime(d.event_time, format="ISO8601", utc=True, errors="coerce")
assert d.event_time.notna().all(), "unparsed event_time values"
LAT0 = float(d.svi_lat.mean()); ASP = 1.0/np.cos(np.radians(LAT0))
print(f"loaded {len(d):,} events from {os.path.basename(EVENTS)}")
print(f"  spontaneous {int(d.background.sum()):,} ({100*d.background.mean():.0f}%)   "
      f"triggered {int((~d.background).sum()):,}")
print(f"  NOTE labels come from the NND notebook — identical event set by construction")''')

md(r"""## 1 — K-means sub-regions

Primary clustering is in the **map plane** (lon, lat), z-scored so the two axes weigh equally. A
3-D labelling (lon, lat, depth) is carried alongside for comparison. K is **fixed**; the
silhouette-vs-K curve is shown for context, not to select K.""")

co(r'''XY  = d[["svi_lon","svi_lat"]].to_numpy()
XYZ = d[["svi_lon","svi_lat","svi_dep"]].to_numpy()
Xs2 = StandardScaler().fit_transform(XY)
Xs3 = StandardScaler().fit_transform(XYZ)

km2 = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(Xs2)
km3 = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(Xs3)
d["kc"]   = km2.labels_          # primary (map-plane)
d["kc3d"] = km3.labels_          # 3-D comparison

# relabel by descending size so cluster 0 is the largest (stable, readable legends)
order = d.kc.value_counts().index.tolist()
remap = {old:new for new,old in enumerate(order)}
d["kc"] = d.kc.map(remap)

sil = {k: silhouette_score(Xs2, KMeans(n_clusters=k, n_init=10,
                                       random_state=RANDOM_STATE).fit_predict(Xs2))
       for k in K_SWEEP}
ari = adjusted_rand_score(d.kc, d.kc3d)
print(f"K = {K}   silhouette(2D) = {sil[K]:.3f}   ARI(2D vs 3D labelling) = {ari:.3f}")
print("cluster sizes:", d.kc.value_counts().sort_index().to_dict())

fig, ax = plt.subplots(figsize=(6.4,3.8))
ax.plot(list(sil), list(sil.values()), "o-", color="0.3")
ax.plot([K],[sil[K]], "o", ms=11, mfc="none", mec="#d62728", mew=2, label=f"K = {K} (fixed)")
ax.set(xlabel="Number of clusters K", ylabel="Mean silhouette (2-D)",
       title="Silhouette vs K — context for the fixed K, not a selection criterion")
leg=ax.legend(); leg.set_zorder(10)
save(fig, "K0_silhouette"); plt.show()''')

md(r"""## K1 — Where the sub-regions are""")

co(r'''fig, ax = plt.subplots(figsize=(7.6,8.4))
for sgm in FSEG: ax.plot(sgm[:,0], sgm[:,1], color="0.5", lw=0.7, zorder=2)
for sgm in CSEG: ax.plot(sgm[:,0], sgm[:,1], color="black", lw=0.9, zorder=3)
for k in range(K):
    s = d[d.kc==k]
    ax.scatter(s.svi_lon, s.svi_lat, s=9, color=TAB[k%len(TAB)], alpha=0.85,
               label=f"C{k} (n={len(s)})", zorder=4)
    cx, cy = s.svi_lon.mean(), s.svi_lat.mean()
    ax.text(cx, cy, f"C{k}", fontsize=12, fontweight="bold", ha="center", va="center",
            zorder=6, bbox=dict(boxstyle="circle,pad=0.22", fc="white", ec="0.3", lw=0.8, alpha=.9))
ax.plot([UF_BOX[0],UF_BOX[1],UF_BOX[1],UF_BOX[0],UF_BOX[0]],
        [UF_BOX[2],UF_BOX[2],UF_BOX[3],UF_BOX[3],UF_BOX[2]], "--", lw=0.9, color="0.3", zorder=5)
ax.set(xlim=(MAP_BOX[0],MAP_BOX[1]), ylim=(MAP_BOX[2],MAP_BOX[3]), aspect=ASP,
       xlabel="Longitude", ylabel="Latitude", title=f"K-means spatial sub-regions (K = {K})")
leg=ax.legend(fontsize=8, loc="lower left", ncol=2); leg.set_zorder(10)
save(fig, "K1_subregion_map"); plt.show()''')

md(r"""## K2 — Spontaneous vs triggered by sub-region

**The core panel.** If every sub-region had the same spontaneous fraction, the volume would be
behaving homogeneously and the partition would carry no information. Differences between the bars
mean the sub-regions are in genuinely different states — chronic background activity in some,
episodic triggered sequences in others.

The error bars are binomial 95% intervals, so small clusters cannot masquerade as strong signals.""")

co(r'''from scipy.stats import beta as _beta, chi2_contingency

rows=[]
for k in range(K):
    s = d[d.kc==k]; n=len(s); nb=int(s.background.sum())
    p = nb/n
    lo = _beta.ppf(0.025, nb, n-nb+1) if nb else 0.0            # Clopper-Pearson 95%
    hi = _beta.ppf(0.975, nb+1, n-nb) if nb<n else 1.0
    rows.append(dict(cluster=k, n=n, spontaneous=nb, triggered=n-nb, frac=p, lo=lo, hi=hi,
                     lon=s.svi_lon.mean(), lat=s.svi_lat.mean(),
                     dep=s.svi_dep.median(), mlmax=s.kma_mag.max()))
T = pd.DataFrame(rows)
overall = d.background.mean()

# is the variation between sub-regions statistically real?
chi2, pval, dof, _ = chi2_contingency(T[["spontaneous","triggered"]].to_numpy())
print(f"overall spontaneous fraction {overall:.3f}")
print(f"chi-square test of homogeneity across the {K} sub-regions: chi2={chi2:.1f}, dof={dof}, p={pval:.2e}")
print("  -> " + ("sub-regions DIFFER significantly in spontaneous fraction"
                 if pval<0.05 else "no significant difference between sub-regions"))

fig, axes = plt.subplots(1, 2, figsize=(13.5,5.0), gridspec_kw=dict(width_ratios=[1.25,1]))
ax=axes[0]
Ts = T.sort_values("frac")
ax.bar(range(K), Ts.frac, color=[TAB[k%len(TAB)] for k in Ts.cluster], edgecolor="0.25")
ax.errorbar(range(K), Ts.frac, yerr=[Ts.frac-Ts.lo, Ts.hi-Ts.frac], fmt="none",
            ecolor="0.2", capsize=4, lw=1.2)
ax.axhline(overall, color="0.25", ls="--", lw=1.4, label=f"catalog mean {overall:.2f}")
ax.set_xticks(range(K)); ax.set_xticklabels([f"C{c}\n(n={n})" for c,n in zip(Ts.cluster,Ts.n)], fontsize=8)
ax.set(ylabel="Spontaneous fraction", ylim=(0,1),
       title="Spontaneous fraction by sub-region (95% binomial CI)")
leg=ax.legend(); leg.set_zorder(10)

ax=axes[1]
sc=ax.scatter(T.lon, T.lat, s=18*np.sqrt(T.n), c=T.frac, cmap="coolwarm_r", vmin=0, vmax=1,
              edgecolor="0.2", lw=0.8, zorder=4)
for sgm in FSEG: ax.plot(sgm[:,0], sgm[:,1], color="0.55", lw=0.6, zorder=2)
for _,r in T.iterrows(): ax.text(r.lon, r.lat, f"C{int(r.cluster)}", fontsize=8, ha="center",
                                 va="center", zorder=5)
ax.set(xlim=(MAP_BOX[0],MAP_BOX[1]), ylim=(MAP_BOX[2],MAP_BOX[3]), aspect=ASP,
       xlabel="Longitude", ylabel="Latitude", title="Sub-region centroids (size ~ n, colour = spontaneous fraction)")
fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03).set_label("Spontaneous fraction")
fig.tight_layout()
save(fig, "K2_spontaneous_fraction"); plt.show()

print(T[["cluster","n","spontaneous","triggered","frac","dep","mlmax"]]
      .rename(columns={"frac":"spont_frac","dep":"median_depth_km","mlmax":"ML_max"})
      .to_string(index=False, float_format=lambda v: f"{v:.2f}"))''')

md(r"""## K3 — Do the sub-regions switch on at different times?

Per-sub-region event rate through the catalog. A sub-region dominated by triggered activity should
show sharp bursts; a spontaneous-dominated one should be closer to steady.""")

co(r'''GJ = pd.Timestamp(GJ_T, tz="UTC")
yrs = np.arange(YEAR_RANGE[0], YEAR_RANGE[1]+1)
nrow = int(np.ceil(K/2))
fig, axes = plt.subplots(nrow, 2, figsize=(13, 2.1*nrow), sharex=True)
for k, ax in zip(range(K), axes.ravel()):
    s = d[d.kc==k]
    sp = s[s.background].event_time.dt.year.value_counts().reindex(yrs, fill_value=0)
    tr = s[~s.background].event_time.dt.year.value_counts().reindex(yrs, fill_value=0)
    ax.bar(yrs-0.2, sp.values, width=0.4, color="#1f77b4", label="spontaneous")
    ax.bar(yrs+0.2, tr.values, width=0.4, color="#d62728", label="triggered")
    ax.axvline(GJ.year + GJ.dayofyear/365.25, color="0.3", lw=1.0)
    ax.set_ylabel(f"C{k}", rotation=0, labelpad=18, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    if k==0:
        leg=ax.legend(fontsize=8, ncol=2); leg.set_zorder(10)
for ax in axes.ravel()[K:]: ax.axis("off")
axes.ravel()[min(K,len(axes.ravel()))-1].set_xlabel("Year")
fig.suptitle("Annual event counts by sub-region — spontaneous vs triggered "
             "(vertical line = 2016 Gyeongju)", y=1.0)
fig.tight_layout()
save(fig, "K3_subregion_timeseries"); plt.show()''')

md(r"""## K4 — Continuous cumulative counts by sub-region

Cumulative event count for each sub-region, plotted continuously in time. Slope *is* rate, so this
shows directly which sub-regions accumulate steadily (chronic) and which jump in steps (episodic).
The right panel splits each sub-region into its spontaneous and triggered components — a chronic
patch should track its spontaneous curve closely, an episodic one should show triggered staircases.""")

co(r'''fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharex=True)

ax = axes[0]
for k in range(K):
    s = d[d.kc==k].sort_values("event_time")
    ax.plot(s.event_time, np.arange(1, len(s)+1), lw=1.8, color=TAB[k%len(TAB)],
            label=f"C{k} (n={len(s)}, {100*s.background.mean():.0f}% spont.)")
ax.axvline(GJ, color="0.25", lw=1.2, ls="--")
ax.text(GJ, ax.get_ylim()[1]*0.98, " Gyeongju Mw 5.5", fontsize=9, va="top", color="0.25")
ax.set(xlabel="Year", ylabel="Cumulative events", title="Cumulative count by sub-region")
ax.set_xlim(pd.Timestamp(f"{YEAR_RANGE[0]}-01-01", tz="UTC"),
            pd.Timestamp(f"{YEAR_RANGE[1]}-12-31", tz="UTC"))
leg = ax.legend(fontsize=8, loc="upper left"); leg.set_zorder(10)

ax = axes[1]
for k in range(K):
    s = d[d.kc==k]
    sp = s[s.background].sort_values("event_time")
    tr = s[~s.background].sort_values("event_time")
    ax.plot(sp.event_time, np.arange(1,len(sp)+1), lw=1.6, color=TAB[k%len(TAB)])
    ax.plot(tr.event_time, np.arange(1,len(tr)+1), lw=1.6, color=TAB[k%len(TAB)], ls=":")
ax.axvline(GJ, color="0.25", lw=1.2, ls="--")
ax.plot([], [], color="0.35", lw=1.6, label="spontaneous (solid)")
ax.plot([], [], color="0.35", lw=1.6, ls=":", label="triggered (dotted)")
ax.set(xlabel="Year", ylabel="Cumulative events", title="Split into spontaneous vs triggered")
leg = ax.legend(fontsize=8, loc="upper left"); leg.set_zorder(10)
fig.tight_layout()
save(fig, "K4_cumulative"); plt.show()

# quantify "steady vs episodic": largest single-30-day jump as a fraction of the total
print(f"{'cluster':>8} {'n':>5} {'spont%':>7} {'max 30-day burst':>17} {'burst/total':>12}")
for k in range(K):
    s = d[d.kc==k].sort_values("event_time")
    t = s.event_time.values
    burst = max((np.sum((t >= t0) & (t < t0 + np.timedelta64(30,"D"))) for t0 in t), default=0)
    print(f"{k:>8} {len(s):>5} {100*s.background.mean():>6.0f}% {burst:>17} {burst/len(s):>11.2f}")''')

md(r"""## Summary — sub-region table for the poster""")

co(r'''T2 = T.copy()
T2["spont_pct"] = (100*T2.frac).round(0).astype(int)
T2["label"] = np.where(T2.frac > overall+0.10, "spontaneous-dominated",
              np.where(T2.frac < overall-0.10, "triggered-dominated", "mixed"))
out = T2[["cluster","n","spontaneous","triggered","spont_pct","label","lon","lat","dep","mlmax"]]
out.columns = ["cluster","n_events","n_spontaneous","n_triggered","spontaneous_%","character",
               "centroid_lon","centroid_lat","median_depth_km","ML_max"]
print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
out.to_csv(os.path.join(FIGS, "kmeans_subregions.csv"), index=False)
d[["event_idx","event_time","svi_lon","svi_lat","svi_dep","kma_mag","background","kc","kc3d"]] \
    .to_csv(os.path.join(FIGS, "kmeans_events.csv"), index=False)
print(f"\n-> figs/kmeans_subregions.csv (the poster table)  +  figs/kmeans_events.csv")
print(f"   chi-square homogeneity p = {pval:.2e}  "
      f"({'sub-regions differ' if pval<0.05 else 'no significant difference'})")''')

nb = nbf.v4.new_notebook(cells=cells,
                         metadata={"kernelspec":{"name":"python3","display_name":"Python 3 (base)",
                                                 "language":"python"}})
for c in nb.cells:
    if c.cell_type == "code":
        c.source = c.source.replace("__K__", str(_A.k)).replace("__EVENTS__", _A.events)
with open(NB, "w") as f:
    nbf.write(nb, f)
print(f"wrote {NB} ({len(cells)} cells, unexecuted)  K={_A.k}  events={_A.events}")
