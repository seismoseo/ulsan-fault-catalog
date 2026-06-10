#!/usr/bin/env python
"""Generate compare_relocations.ipynb — compare the family-738 relocations:
  (0) Ulsan/PocketQuake absolute HypoInverse(kim2011)  vs
  (1) reuse-picks dt.cc relocation  vs
  (2) fresh-picks dt.cc relocation.
Shows the spatial collapse, map + depth sections, and the (1)-vs-(2) per-event offset (matched by
cuspid) — the robustness-to-pick-source headline. Reuses PocketQuake's sumio + viz.

Usage: python build_compare_nb.py   (writes compare_relocations.ipynb next to this file)
"""
import os
import nbformat as nbf

PQ = "/home/msseo/works/15.PocketQuake"
PIPE = os.path.join(PQ, "external", "korea-cluster-relocation")
HERE = os.path.dirname(os.path.abspath(__file__))

nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md("""# Family 738 relocation — reuse-picks vs fresh-picks (PocketQuake HypoInverse + HypoDD, kim2011)

The largest waveform-similarity multiplet (KG.HDB 5-25 Hz single-linkage CC>=0.9, 35 events,
2016-11-17 -> 2017-03-11) relocated two ways through the **same** PocketQuake Fortran
HypoInverse+HypoDD pipeline on the **same** Ulsan waveforms, **kim2011** — differing only in the picks:

- **(1) reuse** — Ulsan's existing PhaseNet+ picks (`f738_reuse`).
- **(2) fresh** — PocketQuake re-picked PhaseNet+ on the identical waveforms (`f738_fresh`).

Compared against the **absolute** HypoInverse(kim2011) catalog (before relative relocation).""")

co(f"""import os, sys
sys.path.insert(0, "{PQ}"); sys.path.insert(0, "{PIPE}")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from obspy.geodetics.base import gps2dist_azimuth
from pipeline import config, viz
from pipeline.core import sumio

RUNS = os.path.join("{PIPE}", "pipeline", "runs")
def sum_abs(slug):   return sumio.read_sum(os.path.join(RUNS, slug, "1.HypoInv", "kim2011", slug + ".sum"))
def reloc(slug):     return sumio.read_reloc(os.path.join(RUNS, slug, "2.HypoDD", "02.dt.cc", "hypoDD.reloc"))
A  = sum_abs("f738_reuse")     # absolute HypoInverse(kim2011) — same picks as (1)
R1 = reloc("f738_reuse")       # (1) reuse-picks dt.cc
R2 = reloc("f738_fresh")       # (2) fresh-picks dt.cc
print(f"abs {{len(A)}} | reuse-dtcc {{len(R1)}} | fresh-dtcc {{len(R2)}} events")""")

md("""## 1 · Spatial collapse — absolute vs dt.cc (RMS horizontal spread + depth scatter)""")
co("""def spread(d):
    clat, clon = d["lat"].mean(), d["lon"].mean()
    h = np.array([gps2dist_azimuth(clat, clon, la, lo)[0] for la, lo in zip(d["lat"], d["lon"])])
    return np.sqrt((h**2).mean()), d["depth"].std() * 1000.0   # m, m
rows = []
for name, d in [("absolute HypoInverse(kim2011)", A), ("(1) reuse-picks dt.cc", R1),
                ("(2) fresh-picks dt.cc", R2)]:
    h, z = spread(d); rows.append(dict(catalog=name, n=len(d), rms_horiz_m=round(h), depth_std_m=round(z)))
pd.DataFrame(rows)""")

md("""### Saved result tables
The relocated catalogs + the (1)-vs-(2) offsets are written to `family738/reloc_*.csv` (readable
CSVs — event_id, lat/lon/depth, relative x/y/z, link counts, and the **bootstrap 95% errors** once
`run.sh`'s bootstrap step has run). Raw PocketQuake outputs live at
`…/runs/f738_{reuse,fresh}/2.HypoDD/02.dt.cc/hypoDD.reloc`.""")
co("""import subprocess
subprocess.run([sys.executable, "save_results.py"], check=True)        # writes family738/reloc_*.csv
print("reloc_compare.csv (first 8 of 35):")
display(pd.read_csv("family738/reloc_compare.csv").head(8))
print("reloc_f738_reuse.csv (first 5):")
display(pd.read_csv("family738/reloc_f738_reuse.csv").head(5))""")

md("""## 2 · Map + depth — the three catalogs overlaid (shared extent)

A tight repeating multiplet should **collapse** from the absolute scatter to a compact dt.cc patch.""")
co("""fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=130)
def panel(a, x, y, ttl, xl, yl):
    for d, c, m, lab in [(A, "0.6", "o", "abs"), (R1, "crimson", "o", "(1) reuse"), (R2, "steelblue", "x", "(2) fresh")]:
        a.scatter(d[x], d[y], s=22, c=c, marker=m, alpha=0.8, label=lab, edgecolor="k" if m=="o" else None, lw=0.3)
    a.set_xlabel(xl); a.set_ylabel(yl); a.set_title(ttl); a.legend(fontsize=8)
panel(ax[0], "lon", "lat", "Map", "lon", "lat"); ax[0].set_aspect(1/np.cos(np.radians(A['lat'].mean())))
panel(ax[1], "lon", "depth", "Lon-depth", "lon", "depth (km)"); ax[1].invert_yaxis()
panel(ax[2], "lat", "depth", "Lat-depth", "lat", "depth (km)"); ax[2].invert_yaxis()
fig.suptitle("Family 738: absolute (grey) vs (1) reuse-picks dt.cc (red) vs (2) fresh-picks dt.cc (blue)")
fig.tight_layout(); plt.show()""")
md("""Publication-quality PocketQuake map of each dt.cc relocation (depth-coloured, bootstrap errors if cached):""")
co("""for slug in ("f738_reuse", "f738_fresh"):
    try:
        viz.map_catalog(config.load_cluster(slug), source="reloc"); plt.show()
    except Exception as e:
        print(slug, "map skipped:", type(e).__name__, e)""")

md("""## 3 · (1) reuse vs (2) fresh — per-event offset (matched by cuspid)

Same waveforms, same engine, same velocity model — so this offset isolates the effect of the **pick
instance** (existing Ulsan PhaseNet+ picks vs a fresh PhaseNet+ re-pick).""")
co("""m = R1.set_index("id").join(R2.set_index("id"), lsuffix="_1", rsuffix="_2", how="inner")
off = np.array([gps2dist_azimuth(a, b, c, e)[0] for a, b, c, e in zip(m.lat_1, m.lon_1, m.lat_2, m.lon_2)])
dz = (m.depth_1 - m.depth_2).abs() * 1000.0
print(f"matched {len(m)}/{len(R1)} | horiz offset median {np.median(off):.0f} m, max {off.max():.0f} m"
      f" | depth offset median {dz.median():.0f} m, max {dz.max():.0f} m")
fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
ax[0].hist(off, bins=15, color="teal", alpha=0.8); ax[0].set(xlabel="(1)-(2) horizontal offset (m)", ylabel="events",
          title="Pick-source sensitivity (horizontal)")
ax[1].hist(dz, bins=15, color="indianred", alpha=0.8); ax[1].set(xlabel="(1)-(2) depth offset (m)", title="(depth)")
fig.tight_layout(); plt.show()""")

md("""## 4 · Bootstrap 95% errors (data-resampling uncertainty)

PocketQuake's Fortran-hypoDD bootstrap (`hypodd.bootstrap_relocation`) resamples the differential-time
data and re-inverts `n=1000` times (cached — `run.sh` precomputes it, so this loads instantly). The
per-event 95% half-widths are the honest relative-location uncertainty, and put the (1)-vs-(2) offset
(§3) in context.""")
co("""from pipeline.core import hypodd
boot = {}
for slug in ("f738_reuse", "f738_fresh"):
    bb = hypodd.bootstrap_relocation(config.load_cluster(slug), branch="dtcc", n=1000, seed=0)  # cached
    boot[slug] = bb
    hw = np.hypot(bb["ex95"], bb["ey95"])
    print(f"{slug}: median 95% half-width  horiz {np.nanmedian(hw):.0f} m  vert {np.nanmedian(bb['ez95']):.0f} m"
          f"  (n_events {bb['ex95'].notna().sum()})")
print(f"\\n(1)-vs-(2) horizontal offset median was ~124 m — compare to the 95% half-widths above:"
      f" a larger offset than the error means the pick instance dominates that dimension.)" )""")

md("""### 4a · Under-constrained events + reuse-vs-fresh

An event is flagged **under-constrained** (and dropped from the §5 dt.cc views) when its bootstrap 95%
half-width exceeds **100 m horizontally** or **100 m vertically** (`viz.BOOT_DROP_{HORIZ,VERT}_KM=0.1`),
is relocated in < 60 % of replicas, or has no CI. Same waveforms + engine + kim2011 → the only
difference between the two runs is the **pick instance**, so the comparison below isolates pick
robustness. The poorly-resolved direction is consistently **N-S (ey95)** and **depth (ez95)** — the
E-W (ex95) is always tight — i.e. the azimuthal station geometry, not the waveforms, sets the limit.""")
co("""def under(slug):
    df = pd.read_csv(os.path.join(RUNS, slug, "2.HypoDD", "02.dt.cc", "bootstrap_errors.csv"), comment="#")
    df["horiz"] = np.hypot(df.ex95, df.ey95)
    rl = (reloc(slug)).set_index("id")
    df["eid"] = df.id.map(lambda i: rl.loc[i, "time"].strftime("%Y%m%d%H%M%S") if i in rl.index else str(i))
    return df, df[(df.horiz > 100) | (df.ez95 > 100)]
for slug in ("f738_reuse", "f738_fresh"):
    df, bad = under(slug)
    print(f"{slug}: {len(bad)} under-constrained / {len(df)}  | ez95 median {df.ez95.median():.0f} m, "
          f"mean {df.ez95.mean():.0f} m, max {df.ez95.max():.0f} m  | ey95 mean {df.ey95.mean():.0f} m")
    for r in bad.sort_values("ez95", ascending=False).itertuples():
        print(f"    {r.eid}   ex95={r.ex95:.0f}  ey95={r.ey95:.0f}  ez95={r.ez95:.0f} m")
print("\\nThe well-constrained core (median) is ~identical between runs; reuse has a much lighter TAIL\\n"
      "(1 vs 7 under-constrained) — the existing Ulsan PhaseNet+ picks give more *stable* relative\\n"
      "locations even where the fresh re-pick has more raw obs, so reuse is the more robust choice.")""")

md("""## 5 · Depth sections + fault-frame (SVD plane) — PocketQuake views

The standard PocketQuake cross-sections for each dt.cc relocation: `depth_sections` (lon-depth /
lat-depth) and `fault_sections` (2×2 in fault coordinates, the fault plane = the **SVD best-fit plane**
of the relocated cloud, `frame_from="svd"`), with the **bootstrap 95% error bars** overlaid.""")
co("""for slug in ("f738_reuse", "f738_fresh"):
    cfg = config.load_cluster(slug)
    print(f"================  {slug}  ================")
    try:
        viz.depth_sections(cfg, velmodel="kim2011", source="reloc"); plt.show()
    except Exception as e:
        print(slug, "depth_sections skipped:", type(e).__name__, e)
    try:
        viz.fault_sections(cfg, velmodel="kim2011", frame_from="svd", color_by="time",
                           show_bootstrap=True); plt.show()
    except Exception as e:
        print(slug, "fault_sections skipped:", type(e).__name__, e)""")

md("""## 5b · Same sections, **keeping** the under-constrained events

The §5 figures drop the bootstrap-flagged under-constrained events. Here we relax the three
`viz.BOOT_DROP_*` thresholds so **all 35 events** are plotted with their (larger) 95% error bars — the
dropped events appear as the points with big N-S / depth bars. (Note the SVD plane is now fit to all 35,
so the poorly-located events can tilt it slightly vs §5.) The constants are restored afterwards.""")
co("""_save = (viz.BOOT_DROP_HORIZ_KM, viz.BOOT_DROP_VERT_KM, viz.BOOT_DROP_MIN_NBOOT_FRAC)
viz.BOOT_DROP_HORIZ_KM = np.inf      # keep everything
viz.BOOT_DROP_VERT_KM = None
viz.BOOT_DROP_MIN_NBOOT_FRAC = 0.0
try:
    for slug in ("f738_reuse", "f738_fresh"):
        cfg = config.load_cluster(slug)
        print(f"================  {slug}  (ALL 35 events kept)  ================")
        viz.depth_sections(cfg, velmodel="kim2011", source="reloc"); plt.show()
        viz.fault_sections(cfg, velmodel="kim2011", frame_from="svd", color_by="time",
                           show_bootstrap=True); plt.show()
finally:
    viz.BOOT_DROP_HORIZ_KM, viz.BOOT_DROP_VERT_KM, viz.BOOT_DROP_MIN_NBOOT_FRAC = _save""")

md("""## 6 · Reading this

- **Collapse** (§1): the dt.cc relocation should tighten the multiplet far below the absolute scatter
  (repeaters are co-located) — that is the precision gain from waveform cross-correlation.
- **(1) ≈ (2)?** (§3): both runs share waveforms + engine + kim2011, so any offset is the **pick
  instance**. A small offset (within the bootstrap 95%) means the result is robust to re-picking; a
  larger offset flags pick-source sensitivity worth reporting.
- Everything here is reproduced by `run.sh` (no manual steps); scale to the other multiplets by
  re-running `make_catalog.py --family <id>` + the same staging/relocation commands.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = os.path.join(HERE, "compare_relocations.ipynb")
nbf.write(nb, out); print("wrote", out, len(C), "cells")
