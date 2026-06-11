#!/usr/bin/env python
"""Generate batch_summary.ipynb — an aggregate overview of ALL relocated 5-25 Hz multiplet families.

Reads the batch products (master_metrics.csv, batch_manifest.csv, failures.csv, master_map_relocated.png,
per-family thumbnails) written by batch_relocate.py + aggregate_results.py — no pipeline re-run; just
loads CSVs and plots. Sections: overview + regional map, collapse statistics, the repeater-vs-multiplet
view (dt.cc spread vs bootstrap error), size/depth/time relationships, the master table + failures, and
a top-N fault-frame thumbnail gallery.

Usage: python build_batch_summary_nb.py   (writes batch_summary.ipynb next to this file)
"""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md("""# Ulsan multiplet relocation — batch summary (5-25 Hz, all families)

Aggregate overview of every 5-25 Hz waveform-similarity multiplet family relocated with the **reuse**
scheme (PocketQuake HypoInverse + dt.cc HypoDD, kim2011). Built from the batch products
(`master_metrics.csv`, `batch_manifest.csv`, `failures.csv`, `master_map_relocated.png`) — no pipeline
is re-run here. Regenerate the data with `./run_all.sh`, this notebook with `build_batch_summary_nb.py`.""")

co("""import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib as mpl, matplotlib.font_manager as fm
from IPython.display import Image, display

# Helvetica for plot text, graceful fallback
_avail = {f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica", "Arial", "Nimbus Sans", "TeX Gyre Heros", "DejaVu Sans"):
    if _f in _avail:
        mpl.rcParams["font.family"] = _f; break
mpl.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})

HERE = os.getcwd()
M = pd.read_csv("master_metrics.csv")
MAN = pd.read_csv("batch_manifest.csv")
RELOC = M[M.status.isin(["done", "done_cached"])].copy()         # the dt.cc-relocated families
print(f"{len(M)} families | {len(RELOC)} relocated | {(M.status=='absolute_only').sum()} absolute-only | "
      f"{M.status.str.startswith('failed').sum()} failed | {int(M.n.sum())} events in families, "
      f"{int(M.n_relocated.sum())} relocated")""")

md("""## 1 · Overview — status and the relocated catalogue on the fault""")
co("""fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
vc = M.status.value_counts()
ax[0].bar(vc.index, vc.values, color="steelblue"); ax[0].set(title="Family outcome", ylabel="Families")
ax[0].tick_params(axis="x", rotation=20)
ax[1].hist(RELOC.n, bins=range(3, int(RELOC.n.max()) + 2), color="0.5", edgecolor="k", linewidth=0.4)
ax[1].set(title="Relocated family size", xlabel="Events per family", ylabel="Families")
fig.tight_layout(); plt.show()""")
md("""**Combined regional map** — all relocated families on the Ulsan Fault subregion (fault traces +
coastline), one colour per family; absolute-only families are faint grey.""")
co("""display(Image(filename="master_map_relocated.png"))""")

md("""## 2 · Collapse — how much each family tightened under dt.cc

`collapse_ratio = dt.cc horizontal spread / absolute horizontal spread`. Small = strong tightening
(repeaters co-locate). The before→after scatter shows the absolute scatter shrinking to the dt.cc patch.""")
co("""r = RELOC.dropna(subset=["abs_spread_horiz_m", "dtcc_spread_horiz_m"])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(RELOC.collapse_ratio.dropna(), bins=20, color="indianred", edgecolor="k", linewidth=0.4)
ax[0].axvline(RELOC.collapse_ratio.median(), color="k", ls="--", lw=1,
              label=f"median {RELOC.collapse_ratio.median():.2f}")
ax[0].set(title="Collapse ratio (dt.cc / absolute)", xlabel="Collapse ratio", ylabel="Families"); ax[0].legend()
ax[1].scatter(r.abs_spread_horiz_m, r.dtcc_spread_horiz_m, s=12 + r.n, c=r.n, cmap="viridis",
              edgecolor="k", linewidth=0.3, alpha=0.85)
lim = [1, max(r.abs_spread_horiz_m.max(), 10)]
ax[1].plot(lim, lim, "0.5", ls=":", label="1:1 (no change)")
ax[1].set(xscale="log", yscale="log", xlabel="Absolute spread (m)", ylabel="dt.cc spread (m)",
          title="Before vs after, horizontal"); ax[1].legend()
cb = fig.colorbar(ax[1].collections[0], ax=ax[1]); cb.set_label("Events per family")
fig.tight_layout(); plt.show()""")

md("""## 3 · Are the collapses real? — repeater vs multiplet

The key check: a tight dt.cc cluster only *means* something if its **bootstrap 95% error is smaller than
the cluster spread**. Families below the 1:1 line (spread > error) are well-resolved tight clusters
(candidate true repeaters); families above it collapsed to within their own uncertainty (marginal —
mostly the small families). Colour = family size.""")
co("""b = RELOC.dropna(subset=["dtcc_spread_horiz_m", "boot_horiz95_m"])
resolved = b[b.dtcc_spread_horiz_m > b.boot_horiz95_m]
fig, ax = plt.subplots(figsize=(6.5, 6))
sc = ax.scatter(b.boot_horiz95_m, b.dtcc_spread_horiz_m, s=14 + b.n, c=b.n, cmap="plasma",
                edgecolor="k", linewidth=0.3, alpha=0.85)
lim = [0.5, max(b.boot_horiz95_m.max(), b.dtcc_spread_horiz_m.max())]
ax.plot(lim, lim, "0.4", ls="--", lw=1.2, label="1:1  (spread = error)")
ax.set(xscale="log", yscale="log", xlabel="Bootstrap 95% horizontal half-width (m)",
       ylabel="dt.cc cluster horizontal spread (m)",
       title="Cluster spread vs location uncertainty")
cb = fig.colorbar(sc); cb.set_label("Events per family"); ax.legend(loc="upper left")
fig.tight_layout(); plt.show()
print(f"well-resolved (spread > bootstrap error): {len(resolved)} / {len(b)} families; "
      f"median size {resolved.n.median():.0f} vs {b[b.dtcc_spread_horiz_m <= b.boot_horiz95_m].n.median():.0f} "
      f"for the marginal ones")""")

md("""## 4 · Size relationships

Larger families relocate more completely and with smaller relative error — the basis for trusting them.""")
co("""fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].scatter(RELOC.n, RELOC.n_relocated, s=18, color="teal", edgecolor="k", linewidth=0.3, alpha=0.7)
ax[0].plot([3, RELOC.n.max()], [3, RELOC.n.max()], "0.5", ls=":")
ax[0].set(xlabel="Family size", ylabel="Events relocated", title="Completeness")
ax[1].scatter(RELOC.n, RELOC.collapse_ratio, s=18, color="indianred", edgecolor="k", linewidth=0.3, alpha=0.7)
ax[1].set(xlabel="Family size", ylabel="Collapse ratio", title="Tightening vs size")
ax[2].scatter(RELOC.n, RELOC.boot_horiz95_m, s=18, color="slateblue", edgecolor="k", linewidth=0.3, alpha=0.7)
ax[2].set(xlabel="Family size", ylabel="Bootstrap 95% horiz (m)", yscale="log", title="Uncertainty vs size")
fig.tight_layout(); plt.show()""")

md("""## 5 · Depth and timing

Family median depth, first-activity year, and median recurrence interval. The **2016-09-12 Gyeongju
M5.8** mainshock is marked — many families post-date it.""")
co("""fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].hist(RELOC.depth_med_km.dropna(), bins=20, color="0.5", edgecolor="k", linewidth=0.4)
ax[0].set(xlabel="Family median depth (km)", ylabel="Families", title="Depth distribution")
yr = pd.to_datetime(RELOC.t_first, errors="coerce").dt.year
ax[1].hist(yr.dropna(), bins=range(2010, 2026), color="cadetblue", edgecolor="k", linewidth=0.4)
ax[1].axvline(2016.7, color="crimson", ls="--", lw=1.2, label="2016 Gyeongju M5.8")
ax[1].set(xlabel="First-activity year", ylabel="Families", title="When families begin"); ax[1].legend()
rc = RELOC.recur_med_days.dropna()
ax[2].hist(rc[rc > 0], bins=np.logspace(0, np.log10(max(rc.max(), 10)), 20),
           color="darkorange", edgecolor="k", linewidth=0.4)
ax[2].set(xscale="log", xlabel="Median recurrence (days)", ylabel="Families", title="Recurrence intervals")
fig.tight_layout(); plt.show()""")

md("""## 6 · Master table and failures

`master_metrics.csv` — one row per family (sorted by size). `collapse_ratio` is the tightening;
**trust it only where `boot_horiz95_m` is small** (§3).""")
co("""cols = ["id", "n", "n_relocated", "status", "abs_spread_horiz_m", "dtcc_spread_horiz_m",
        "collapse_ratio", "boot_horiz95_m", "boot_depth95_m", "mean_cc", "depth_med_km", "recur_med_days"]
display(M.sort_values("n", ascending=False)[cols].head(25).reset_index(drop=True))""")
md("""**Families not fully relocated** (`failures.csv`) — tracked explicitly, none silently dropped.""")
co("""fpath = "failures.csv"
display(pd.read_csv(fpath) if os.path.exists(fpath) else pd.DataFrame({"note": ["no failures"]}))""")

md("""## 7 · Fault-frame thumbnails — the largest relocated families

The before/after PyGMT panel (`pygmt_reloc_map.make_map`) for the top families by size. Full per-family
detail (depth sections, source patches, GIF) is in `build_summary_nb.py` (run on demand).""")
co("""top = M[M.status.isin(["done", "done_cached"])].sort_values("n", ascending=False).head(6)
for fid in top.id:
    p = os.path.join(f"family{fid}", f"pygmt_reloc_f{fid}_reuse.png")
    if os.path.exists(p):
        print(f"family {fid}")
        display(Image(filename=p))
    else:
        print(f"family {fid}: thumbnail not generated (run aggregate_results.py --topn N)")""")

md("""## 8 · Fault-frame sections (SVD plane) — large families

The same SOTA fault-coordinate view used for the flagship family 738 (`viz.fault_sections`,
`frame_from="svd"`), for every large family (n ≥ 15): fault-plane map view + along-strike (A-A') and
across-strike (B-B', with the dip guide) depth sections + the along-dip view, coloured by origin time,
with 95% bootstrap error bars. The fault plane is the **SVD best-fit** of each relocated cloud, so each
family's strike/dip is read directly off the title — the basis for the fault-architecture analysis.""")
co("""import sys
sys.path.insert(0, "/home/msseo/works/15.PocketQuake")
sys.path.insert(0, "/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation")
from pipeline import config, viz

NMIN = 15                                                        # "large" families
big = M[(M.status.isin(["done", "done_cached"])) & (M.n >= NMIN)].sort_values("n", ascending=False)
print(f"{len(big)} families with n >= {NMIN}")
for fid, n in zip(big.id, big.n):
    print(f"================  family {fid}  (n={int(n)})  ================")
    try:
        viz.fault_sections(config.load_cluster(f"f{fid}_reuse"), velmodel="kim2011",
                           frame_from="svd", color_by="time", show_bootstrap=True); plt.show()
    except Exception as e:                                       # noqa: BLE001
        print(f"  family {fid} skipped: {type(e).__name__}: {e}")""")

md("""## 9 · Reading this

- **Overview (§1)**: of the 117 multiplet families, most relocate; a handful of 3-4-event families are
  absolute-only (HypoDD too few links) — all tracked in `failures.csv`.
- **Collapse (§2)**: dt.cc tightens families well below the absolute scatter — the precision gain.
- **Repeater vs multiplet (§3)**: the decisive plot. A small collapse ratio is only meaningful where the
  **bootstrap error is smaller than the cluster spread**. Well-resolved families (below the 1:1 line,
  mostly the larger ones) are candidate **true repeating earthquakes**; the rest are marginal multiplets.
  This partition — not the raw catalogue — is the science.
- **Trust rule**: rank/filter families by `boot_horiz95_m`, not by `collapse_ratio` alone.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = os.path.join(HERE, "batch_summary.ipynb")
nbf.write(nb, out); print("wrote", out, len(C), "cells")
