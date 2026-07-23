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

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
BAND = sys.argv[1] if len(sys.argv) > 1 else "5-25"            # e.g. "5-25" (default) or "5-15"
BT = "" if BAND == "5-25" else "_b" + BAND.replace("-", "")    # filename/slug suffix per band
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(f"""# Ulsan multiplet relocation — batch summary ({BAND} Hz, all families)

Aggregate overview of every {BAND} Hz waveform-similarity multiplet family relocated with the **reuse**
scheme (PocketQuake HypoInverse + dt.cc HypoDD, kim2011). Built from the batch products
(`master_metrics{BT}.csv`, `batch_manifest{BT}.csv`, `failures{BT}.csv`, `master_map_relocated{BT}.png`)
— no pipeline is re-run. Regenerate via `./run_all.sh --band {BAND}` then `build_batch_summary_nb.py {BAND}`.""")

co(f'BAND, BT = "{BAND}", "{BT}"   # band selector — every file/slug reference below is suffixed by BT')
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
M = pd.read_csv(f"master_metrics{BT}.csv")
MAN = pd.read_csv(f"batch_manifest{BT}.csv")
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
coastline), one colour per family; absolute-only families are faint grey. **Before** = the same events
at their absolute HypoInverse (kim2011) locations; **after** = the dt.cc relocation — the families
collapse onto tight, fault-aligned patches.""")
co("""display(Image(filename=f"master_map_absolute{BT}.png"))    # before — absolute HypoInverse
display(Image(filename=f"master_map_relocated{BT}.png"))   # after  — dt.cc relocated""")

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
co("""fpath = f"failures{BT}.csv"
display(pd.read_csv(fpath) if os.path.exists(fpath) else pd.DataFrame({"note": ["no failures"]}))""")

md("""## 7 · Fault-frame thumbnails — the largest relocated families

The before/after PyGMT panel (`pygmt_reloc_map.make_map`) for the top families by size. Full per-family
detail (depth sections, source patches, GIF) is in `build_summary_nb.py` (run on demand).""")
co("""top = M[M.status.isin(["done", "done_cached"])].sort_values("n", ascending=False).head(6)
for fid in top.id:
    p = os.path.join(f"family{fid}{BT}", f"pygmt_reloc_f{fid}{BT}_reuse.png")
    if os.path.exists(p):
        print(f"family {fid}")
        display(Image(filename=p))
    else:
        print(f"family {fid}: thumbnail not generated (run aggregate_results.py --topn N)")""")

md("""## 8 · Fault-frame sections (SVD plane) + recurrence — large families (n ≥ 6)

The SOTA fault-coordinate view (`viz.fault_sections`, `frame_from="svd"`) for every large family
(n ≥ 6). The fault plane is the **true SVD best-fit of the cluster** — its orientation (strike/dip in
the title) is the principal plane of the relocated cloud, and the section is centred on the **cluster
centroid** (`center_on="centroid"`), through which the SVD plane passes, so it is **not tied to any
single event** (e.g. the mainshock). Panels: map view + along-strike (A-A') and across-strike (B-B',
with the dip guide) depth sections + along-dip view, coloured by origin time, with 95% bootstrap bars.
Markers are **rupture circles drawn to scale for a 10 MPa stress drop** (Eshelby circular-crack radius
from the local magnitude, ML used as Mw proxy) — same scaling as the §8b along-dip view.
Below each family: its **magnitude-vs-time** history (the 2016-09-12 Gyeongju M5.8 is marked).""")
co("""import sys
import matplotlib.dates as mdates
sys.path.insert(0, "/home/msseo/works/15.PocketQuake")
sys.path.insert(0, "/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation")
from pipeline import config, viz
from pipeline.core import sumio

def _srad(m, dsigma=10e6):                       # Eshelby circular-crack radius (m), ML as Mw proxy
    M0 = 10.0 ** (1.5 * np.asarray(m, float) + 9.1)            # seismic moment (N·m)
    return (7.0 * M0 / (16.0 * dsigma)) ** (1.0 / 3.0)

NMIN = 6                                                        # "large" families
big = M[(M.status.isin(["done", "done_cached"])) & (M.n >= NMIN)].sort_values("n", ascending=False)
print(f"{len(big)} families with n >= {NMIN}")
for fid, n in zip(big.id, big.n):
    cfg = config.load_cluster(f"f{fid}{BT}_reuse")
    print(f"================  family {fid}  (n={int(n)})  ================")
    try:
        viz.fault_sections(cfg, velmodel="kim2011", frame_from="svd", color_by="time",
                           center_on="centroid", show_bootstrap=True,
                           source_radius=_srad, source_label="10 MPa"); plt.show()   # true SVD, 10 MPa circles
    except Exception as e:                                       # noqa: BLE001
        print(f"  sections skipped: {type(e).__name__}: {e}"); continue
    try:                                                         # magnitude-vs-time recurrence below
        D = sumio.read_reloc(os.path.join(config.dtcc_dir(cfg), "hypoDD.reloc"))
        t = pd.to_datetime([str(x) for x in D.time])
        mw = np.asarray(viz._mag_for(cfg, D.id), float)
        fig, ax = plt.subplots(figsize=(11, 2.1))
        ax.vlines(t, 0, mw, color="0.75", lw=0.8, zorder=1)
        ax.scatter(t, mw, s=18 + 12 * np.nan_to_num(mw), c=mdates.date2num(t), cmap="coolwarm",
                   edgecolor="k", linewidth=0.3, zorder=3)
        ax.set(xlabel="Origin time", ylabel="Local magnitude", ylim=(0, None),
               title=f"Family {fid} — magnitude vs time ({len(D)} events)")
        fig.tight_layout(); plt.show()
    except Exception as e:                                       # noqa: BLE001
        print(f"  recurrence skipped: {type(e).__name__}: {e}")""")

md("""## 8b · Compiled fault-frame views — large families (n ≥ 6), same format as §8

The §8 individual SOTA fault-frame panels (`viz.fault_sections`), **compiled** across every large family
(n ≥ 6, i.e. the ones shown individually above) so the population is comparable at a glance. Identical
rendering: markers **coloured by origin time**, **95 % bootstrap** error bars, the **SVD best-fit plane**
frame, centred on each cluster centroid. Three figures —
(A) **across-strike depth sections** (B–B'; dashed = SVD dip line),
(B) **fault-plane map view** (solid = strike, dashed = across-strike),
(C) **fault-plane along-dip view** with **rupture circles drawn to scale for a 10 MPa stress drop**
(Eshelby circular-crack radius from the local magnitude, ML used as Mw proxy).
Panel titles: `f<id> n<N> strike/dip`. Below: **strike/dip statistics** over all families **except
poorly-planar fits** (`flat = S3/S2 > 0.5` dropped). Writes `fault_plane_fits{BT}.csv`.""")
co("""# SVD plane fit for ALL families (cheap; for the orientation statistics below)
from numpy.linalg import svd
def _fit_plane(fid):
    f = os.path.join(f"family{fid}{BT}", f"reloc_f{fid}{BT}_reuse.csv")
    if not os.path.exists(f): return None
    d = pd.read_csv(f)
    if len(d) < 4: return None
    Q = d[["x", "y", "z"]].to_numpy(float); Q = Q - Q.mean(0)
    U, S, Vt = svd(Q, full_matrices=False); nrm = Vt[2]
    if nrm[2] < 0: nrm = -nrm
    dip = np.degrees(np.arccos(abs(nrm[2]))); dipdir = np.degrees(np.arctan2(nrm[0], nrm[1])) % 360
    return dict(id=fid, n=len(Q), strike=(dipdir - 90) % 180, dip=dip, flat=(S[2] / S[1] if S[1] > 0 else 1.0))
PF_all = pd.DataFrame([r for r in (_fit_plane(i) for i in RELOC.id) if r])
PF_all = PF_all.merge(M[["id", "lat_c", "lon_c"]], on="id").sort_values("n", ascending=False).reset_index(drop=True)
PF_all.to_csv(f"fault_plane_fits{BT}.csv", index=False)
print(f"fitted {len(PF_all)} families (n>=4) | median strike {PF_all.strike.median():.0f}deg "
      f"dip {PF_all.dip.median():.0f}deg | bad-planar (flat>0.5): {(PF_all.flat>0.5).sum()}")""")
co("""# fault-frame projection per family (reuses the §8 viz internals -> identical rendering)
import sys, matplotlib.dates as mdates, matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
sys.path.insert(0, "/home/msseo/works/15.PocketQuake")
sys.path.insert(0, "/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation")
from pipeline import config as _cfgmod, viz as _viz
from pipeline.core import sumio as _sumio

def _srad(m, dsigma=10e6):                       # Eshelby circular-crack radius (m), ML as Mw proxy
    M0 = 10.0 ** (1.5 * np.asarray(m, float) + 9.1)            # seismic moment (N·m)
    return (7.0 * M0 / (16.0 * dsigma)) ** (1.0 / 3.0)

def _proj(fid):
    cfg = _cfgmod.load_cluster(f"f{fid}{BT}_reuse")
    reloc, branch = _viz._reloc_path(cfg); d = _sumio.read_reloc(reloc).reset_index(drop=True)
    d = d[~d.id.isin(_viz._boot_underconstrained(cfg, branch))].reset_index(drop=True)
    if len(d) < 3: return None
    us, ud = _viz._best_fit_plane(d.x, d.y, d.z)
    x0, y0, z0 = float(d.x.mean()), float(d.y.mean()), float(d.z.mean())
    rx = (d.x - x0).to_numpy(); ry = (d.y - y0).to_numpy(); th = np.deg2rad(90 - us)
    along = (rx*np.cos(th)+ry*np.sin(th))/1000; across = (-rx*np.sin(th)+ry*np.cos(th))/1000
    dep = (d.z.to_numpy()-z0)/1000
    # Orient each section axis so its + (right) end points to the SE quadrant (maximise E - N):
    # A'/B' = East/South end (right), A/B = West/North end (left). Apply to along + across.
    aE, aN = np.sin(np.deg2rad(us)), np.cos(np.deg2rad(us))          # +along direction in (E, N)
    s_al = 1.0 if (aE - aN) >= 0 else -1.0
    cE, cN = -np.cos(np.deg2rad(us)), np.sin(np.deg2rad(us))         # +across direction in (E, N)
    s_ac = 1.0 if (cE - cN) >= 0 else -1.0
    along *= s_al; across *= s_ac
    along_dip = -across*np.cos(np.deg2rad(ud)) + dep*np.sin(np.deg2rad(ud))
    dipslope = -np.tan(np.deg2rad(ud)) * s_ac                        # SVD dip line in the (across, depth) panel
    boot = _viz._load_bootstrap(cfg, branch); sig = {k: np.full(len(d), np.nan) for k in ["al","ac","dp","ad","e","n"]}
    if boot:
        ct, st, cd, sd = np.cos(th), np.sin(th), np.cos(np.deg2rad(ud)), np.sin(np.deg2rad(ud))
        V = dict(al=[ct,st,0], ac=[-st,ct,0], dp=[0,0,1.], ad=[st*cd,-ct*cd,sd], e=[1.,0,0], n=[0,1.,0])
        for i, e in enumerate(d.id.astype(int)):
            if e in boot:
                for k, v in V.items(): sig[k][i] = _viz._pct_hw(boot[e], np.array(v))/1000
    mag = _viz._mag_for(cfg, d.id); sz = _viz._mag_size(mag, smin=25, smax=1500)
    cv = np.array(mdates.date2num([t.datetime for t in d.time]))
    norm = mcolors.Normalize(vmin=cv.min(), vmax=cv.max() if cv.max() > cv.min() else cv.min()+1)
    rgba = plt.get_cmap("coolwarm")(norm(cv))
    return dict(fid=fid, along=along, across=across, dep=dep, along_dip=along_dip, dipslope=dipslope,
                rx=rx/1000, ry=ry/1000, sig=sig, boot=bool(boot), mag=mag, sz=sz, rgba=rgba, norm=norm,
                us=us, ud=ud, n=len(d))

BIG = M[(M.status.isin(["done","done_cached"])) & (M.n >= 6)].sort_values("n", ascending=False)
PJ = [p for p in (_proj(f) for f in BIG.id) if p]
print(f"{len(PJ)} families with n>=6 projected into fault frame")
_NC = 4; _NR = int(np.ceil(len(PJ)/_NC))

def _cbar(fig, ax, p):                                # individual per-panel origin-time colour bar
    sm = plt.cm.ScalarMappable(norm=p["norm"], cmap="coolwarm"); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    tk = np.linspace(p["norm"].vmin, p["norm"].vmax, 3); cb.set_ticks(tk)
    cb.set_ticklabels([mdates.num2date(t).strftime("%y-%m") for t in tk]); cb.ax.tick_params(labelsize=6)

def _ffgrid(kind, title):
    # constrained_layout keeps the per-panel colour bars + titles from overlapping; equal-aspect
    # square main axes (1:1) with limits chosen so nothing (incl. 10 MPa circles) is clipped.
    fig, axes = plt.subplots(_NR, _NC, figsize=(3.6*_NC, 3.3*_NR), dpi=120, constrained_layout=True)
    axf = np.atleast_1d(axes).flatten()
    for ax, p in zip(axf, PJ):
        if kind == "across":
            X, Y = p["across"], p["dep"]; lim = 1.12*max(np.nanmax(np.abs(X)), np.nanmax(np.abs(Y)), 1e-3)
            if p["boot"]: ax.errorbar(X, Y, xerr=p["sig"]["ac"], yerr=p["sig"]["dp"], fmt="none", ecolor="0.55", elinewidth=0.5, capsize=1, zorder=3)
            ax.scatter(X, Y, s=p["sz"]*0.45, facecolors="none", edgecolors=p["rgba"], linewidth=1.2, zorder=4)
            xx = np.linspace(-lim, lim, 30); ax.plot(xx, p["dipslope"]*xx, "k--", lw=0.9, zorder=1)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.invert_yaxis()
            ax.text(0.04, 0.06, "B", transform=ax.transAxes, fontsize=12, fontweight="bold")
            ax.text(0.90, 0.06, "B'", transform=ax.transAxes, fontsize=12, fontweight="bold")
            ax.set_xlabel("across-strike (km)  W/N→E/S", fontsize=7); ax.set_ylabel("depth (km)", fontsize=7)
        elif kind == "map":
            X, Y = p["rx"], p["ry"]; lim = 1.12*max(np.nanmax(np.abs(X)), np.nanmax(np.abs(Y)), 1e-3)
            su, du = np.sin(np.deg2rad(p["us"])), np.cos(np.deg2rad(p["us"]))
            if p["boot"]: ax.errorbar(X, Y, xerr=p["sig"]["e"], yerr=p["sig"]["n"], fmt="none", ecolor="0.55", elinewidth=0.5, capsize=1, zorder=3)
            ax.scatter(X, Y, s=p["sz"]*0.45, facecolors="none", edgecolors=p["rgba"], linewidth=1.2, zorder=4)
            ax.plot([-lim*su, lim*su], [-lim*du, lim*du], "0.35", lw=1.0); ax.plot([lim*du, -lim*du], [-lim*su, lim*su], "0.35", lw=1.0, ls="--")
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_xlabel("E (km)", fontsize=7); ax.set_ylabel("N (km)", fontsize=7)
        else:  # along-dip with 10 MPa source circles (to scale, fully inside)
            X, Y = p["along"], p["along_dip"]; rk = _srad(p["mag"])/1000.0
            lim = 1.12*max(np.nanmax(np.abs(X)+rk), np.nanmax(np.abs(Y)+rk), 1e-3)
            ax.add_collection(PatchCollection([Circle((X[i], Y[i]), rk[i]) for i in range(p["n"])], facecolors="none", edgecolors=p["rgba"], linewidths=1.2, zorder=4))
            if p["boot"]: ax.errorbar(X, Y, xerr=p["sig"]["al"], yerr=p["sig"]["ad"], fmt="none", ecolor="0.55", elinewidth=0.5, capsize=1, zorder=3)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.invert_yaxis()
            ax.text(0.04, 0.06, "A", transform=ax.transAxes, fontsize=12, fontweight="bold")
            ax.text(0.90, 0.06, "A'", transform=ax.transAxes, fontsize=12, fontweight="bold")
            ax.set_xlabel("along-strike (km)  W/N→E/S", fontsize=7); ax.set_ylabel("along-dip (km)", fontsize=7)
        ax.set_aspect("equal", "box"); ax.grid(True, ls=":", alpha=0.6); ax.set_facecolor("#FAFAFA"); ax.tick_params(labelsize=7)
        ax.set_title(f"f{p['fid']} n{p['n']} {p['us']:.0f}/{p['ud']:.0f}", fontsize=8); _cbar(fig, ax, p)
    for ax in axf[len(PJ):]: ax.axis("off")
    fig.suptitle(title, fontsize=13); plt.show()

_ffgrid("across", "Across-strike depth sections (B-B') — n>=6 families; B=W/N (left), B'=E/S (right); color=origin time (per panel), bars=95% bootstrap, dashed=SVD dip")""")
co("""_ffgrid("map", "Fault-plane map view — n>=6 families; solid=strike, dashed=across-strike; color=origin time, bars=95% bootstrap")""")
co("""_ffgrid("alongdip", "Fault-plane along-dip view — circles = rupture radius at 10 MPa stress drop (to scale); color=origin time")""")
co("""# strike/dip statistics over all families EXCEPT poorly-planar fits (flat = S3/S2 > 0.5 dropped)
GD = PF_all[PF_all.flat <= 0.5].copy()
print(f"statistics over {len(GD)} families (dropped {len(PF_all)-len(GD)} bad-planar with flat>0.5)")
fig = plt.figure(figsize=(13, 4))
ax1 = fig.add_subplot(1, 3, 1, projection="polar"); ax1.set_theta_zero_location("N"); ax1.set_theta_direction(-1)
ax1.hist(np.radians(np.concatenate([GD.strike, GD.strike+180])), bins=np.radians(np.arange(0,361,15)),
         color="steelblue", edgecolor="k", linewidth=0.4)
ax1.set_title(f"Strike rose (n={len(GD)}, planar)", pad=15)
ax2 = fig.add_subplot(1, 3, 2); ax2.hist(GD.dip, bins=np.arange(0,91,7.5), color="indianred", edgecolor="k", linewidth=0.4)
ax2.axvline(GD.dip.median(), color="k", ls="--", label=f"median {GD.dip.median():.0f}d"); ax2.legend()
ax2.set(xlabel="Dip (deg)", ylabel="Families", title="Dip distribution")
ax3 = fig.add_subplot(1, 3, 3)
sc = ax3.scatter(GD.strike, GD.dip, s=10+GD.n*2, c=GD.flat, cmap="RdYlGn_r", edgecolor="k", linewidths=0.3, vmin=0, vmax=0.5)
ax3.set(xlabel="Strike (deg)", ylabel="Dip (deg)", title="Strike vs dip (size proportional to n)", xlim=(0,180), ylim=(0,90))
fig.colorbar(sc, ax=ax3, label="planarity flat (0=best)")
fig.suptitle(f"Fault-plane orientation statistics — {BAND} Hz clusters, bad planar fits excluded (n={len(GD)})", fontsize=12)
fig.tight_layout(); plt.show()""")

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
out = os.path.join(HERE, f"batch_summary{BT}.ipynb")
nbf.write(nb, out); print("wrote", out, len(C), "cells")
