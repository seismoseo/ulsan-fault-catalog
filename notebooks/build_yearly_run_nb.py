#!/usr/bin/env python
"""Generate notebooks/00.Run_yearly_pipeline.ipynb — the per-year pipeline cockpit.

One notebook to run + check a whole year: detection -> association -> augment/phs -> locate -> QC ->
relocation, each stage followed by an intermediate-check cell (counts, distributions) and a PyGMT map,
so problems surface BEFORE the next (heavier) stage is launched. All parameters at the top; every stage
is idempotent/resumable, so re-running the notebook top-to-bottom is always safe.

    python notebooks/build_yearly_run_nb.py          # (re)writes the unexecuted notebook

Kernel: **base** (pyocto/pygmt live there). The detection cell shells out to the `eqnet` env; the
relocation cell shells out to run_pipeline (which handles the pq-gpu xcorr env internally).
"""
import nbformat as nbf

NB = "notebooks/00.Run_yearly_pipeline.ipynb"
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ---------------------------------------------------------------- title
md(r"""# Run one catalog year — pipeline cockpit

Run and check a full year through the 6-stage pipeline, one stage at a time:

**detection → association → augment + phs → locate → QC → relocation**

Each stage has a *run* cell, a *check* cell (counts, distributions), and a *map* cell (PyGMT), so you can
inspect intermediate results before launching the next (heavier) stage. Every stage is idempotent — the
notebook is safe to re-run top-to-bottom at any time.

**Kernel: `base`** (pyocto + pygmt). The detection cell shells out to the `eqnet` env; the relocation
cell shells out to `run_pipeline` (pq-gpu xcorr handled internally). Rough per-stage cost for a sparse
year (2010): detection hours (GPU; already done = skipped), association minutes, augment+phs+locate
minutes, relocation `--through hypoinverse` ~minutes / `--through dtcc` hours (GPU xcorr).""")

# ---------------------------------------------------------------- parameters
co(r'''# ================================ PARAMETERS (edit here only) ================================
YEAR          = 2010
MODEL         = "phasenet_plus"     # picker: phasenet_plus | original | stead | eqt
MIN_PROB      = 0.2                 # PhaseNet+ pick threshold; 0.2 = the validated benchmark setting
                                    # (config default is 0.3 — SeisBench pickers ignore this flag)
NETWORKS      = None                # None = all of KS,KG,GJ,NS; or e.g. "KS,KG" to restrict
VELMODEL      = "kim2011"           # HYPOINVERSE crustal model (kim2011 | kim1983)
STRICT_ASSOC  = False               # True -> config.ASSOC_GATE_STRICT (6/3/3/2)
ASSOC_WORKERS = 8                   # parallel daily association chunks
RELOC_THROUGH = "dtcc"              # "hypoinverse" = fast QC-only reloc preflight; "dtcc" = full (hours)
UF_BOX        = (129.25, 129.55, 35.60, 35.90)   # lon0, lon1, lat0, lat1 (the relocation subregion)

import os, sys, glob, subprocess
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# kernel sanity: association runs IN-KERNEL and needs pyocto -> this must be the base kernel
try:
    import pyocto  # noqa: F401
except ImportError:
    raise SystemExit("This notebook must run on the *base* kernel (pyocto missing here).")

import pygmt
from ufpipe import config, core, stations, relocate
from uflib import uf_cluster as uf

# Helvetica everywhere (graceful fallback), sentence-case labels, opaque legends
try:
    fm.findfont("Helvetica", fallback_to_default=False)
    plt.rcParams["font.family"] = "Helvetica"
except Exception:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams.update({"figure.dpi": 120, "legend.framealpha": 1.0,
                     "legend.facecolor": "white", "legend.edgecolor": "0.6"})

NET_COLORS = {"KS": "#1f77b4", "KG": "#ff7f0e", "GJ": "#2ca02c", "NS": "#d62728"}
networks = NETWORKS.split(",") if NETWORKS else None
RELOC_ROOT = relocate._reloc_root(MODEL, YEAR)          # outputs/reloc/reloc_<year>_uf[_<model>]
SUM_PATH = os.path.join(config.MODELS, MODEL, "HypoInv", VELMODEL, f"UF{YEAR}.sum")

# -------- disclosed parameters (nothing hidden; assoc values from config) --------
print(f"YEAR={YEAR}  MODEL={MODEL}  MIN_PROB={MIN_PROB}  NETWORKS={NETWORKS or 'KS,KG,GJ,NS'}")
print(f"VELMODEL={VELMODEL}  STRICT_ASSOC={STRICT_ASSOC}  ASSOC_WORKERS={ASSOC_WORKERS}  RELOC_THROUGH={RELOC_THROUGH}")
print(f"association: center={config.REGION_CENTER} +/-({config.ASSOC_LAT_PAD},{config.ASSOC_LON_PAD}) deg, "
      f"z={config.ASSOC_ZLIM} km, overlap={config.ASSOC_OVERLAP_S}s, tol={config.ASSOC_PICK_MATCH_TOL}s")
print(f"gate={'STRICT ' + str(config.ASSOC_GATE_STRICT) if STRICT_ASSOC else str(config.ASSOC_GATE)}  "
      f"velocity(assoc)={config.ASSOC_VELMODEL}")
print(f"QC gate (uf_cluster.QC): {uf.QC}")
print(f"detection thresholds: SeisBench P/S={config.P_THRESHOLD}/{config.S_THRESHOLD}, PN+ default={config.PNPLUS_MIN_PROB}")
print(f"reloc results dir -> {RELOC_ROOT}")''')

# ---------------------------------------------------------------- stage 0 station table
md(r"""## Stage 0 — station table for the year

Which stations exist this year, per network (metadata epoch + data actually on disk).""")
co(r'''ST = stations.build_year_table(YEAR, networks=networks)
print(f"{len(ST)} stations in {YEAR}: {ST.net.value_counts().to_dict()}")
ST.head(8)''')
co(r'''# map: the year's station set, colored by network (full-network view)
reg = [float(ST.lon.min()) - 0.3, float(ST.lon.max()) + 0.3,
       float(ST.lat.min()) - 0.25, float(ST.lat.max()) + 0.25]
fig = pygmt.Figure()
fig.basemap(region=reg, projection="M13c", frame=["af", "WSen+tStations " + str(YEAR)])
fig.coast(shorelines="0.4p,gray30", land="gray95", water="azure1", borders="1/0.3p,gray60")
for net, g in ST.groupby("net"):
    fig.plot(x=g.lon, y=g.lat, style="t0.28c", fill=NET_COLORS.get(net, "gray50"),
             pen="0.3p,black", label=net)
fig.plot(x=[UF_BOX[0], UF_BOX[1], UF_BOX[1], UF_BOX[0], UF_BOX[0]],
         y=[UF_BOX[2], UF_BOX[2], UF_BOX[3], UF_BOX[3], UF_BOX[2]], pen="0.8p,black,-")
fig.basemap(map_scale="jBL+w50k+o0.5c/0.5c")
fig.legend(position="JTR+jTR+o0.2c", box="+gwhite+p0.5p")
fig.show()''')

# ---------------------------------------------------------------- stage 1 detection
md(r"""## Stage 1 — detection (shells out to the `eqnet` env)

Idempotent: days whose picks CSV already exists are skipped, so this is safe to re-run (it only fills
gaps). A full un-detected year takes hours on GPU.""")
co(r'''cmd = ["conda", "run", "--no-capture-output", "-n", "eqnet",
       "python", "-m", "ufpipe.detection", "--model", MODEL, "--year", str(YEAR),
       "--min-prob", str(MIN_PROB)]
if NETWORKS:
    cmd += ["--networks", NETWORKS]
print("$", " ".join(cmd)); subprocess.run(cmd, check=True)''')
co(r'''# ---- check: coverage, per-network pick counts, threshold floor ----
files = sorted(glob.glob(os.path.join(config.picks_dir(MODEL, YEAR), f"picks_{YEAR}.*.csv")))
print(f"daily pick files: {len(files)} / {config.days_in_year(YEAR)} days")
P = core.load_picks(MODEL, YEAR)
P["net"] = P.station.str.split(".").str[0]
P["day"] = pd.to_datetime(P.peak_time).dt.floor("D")
print(f"total picks: {len(P):,}  |  stations: {P.station.nunique()}  |  by net: {P.net.value_counts().to_dict()}")
print(f"P/S split: {P.phase.value_counts().to_dict()}")
print(f"min probability in picks: {P.probability.min():.3f}  (should equal the threshold actually used)")

fig, ax = plt.subplots(figsize=(10, 3))
for net, g in P.groupby("net"):
    g.groupby("day").size().plot(ax=ax, lw=0.9, color=NET_COLORS.get(net, "gray"), label=net)
ax.set_xlabel("Time"); ax.set_ylabel("Picks per day"); ax.set_yscale("log")
ax.legend(ncol=4, loc="upper left"); plt.tight_layout(); plt.show()''')

# ---------------------------------------------------------------- stage 2 association
md(r"""## Stage 2 — association (in-kernel, daily-chunked PyOcto, kim2011)

Returns the events + assignments directly for immediate inspection.""")
co(r'''EV, ASG = core.run_association_year(MODEL, YEAR, strict=STRICT_ASSOC,
                                    networks=networks, workers=ASSOC_WORKERS)
print(f"events: {len(EV):,}   assigned picks: {len(ASG):,}")''')
co(r'''# ---- check: picks/event, timeline ----
EV2 = pd.read_csv(config.pyocto_events(MODEL, YEAR), parse_dates=["time"])
npk = EV2.picks
print(f"picks/event: median {npk.median():.0f}, p90 {npk.quantile(.9):.0f}, max {npk.max():.0f}")
print(f"depth: median {EV2.depth.median():.1f} km, in UF box: "
      f"{((EV2.longitude.between(UF_BOX[0], UF_BOX[1])) & (EV2.latitude.between(UF_BOX[2], UF_BOX[3]))).sum()}")
fig, axes = plt.subplots(1, 2, figsize=(11, 3))
axes[0].hist(npk, bins=np.arange(3.5, min(npk.max(), 60) + 1), color="#1f77b4")
axes[0].set_xlabel("Picks per event"); axes[0].set_ylabel("Events")
EV2.set_index("time").resample("D").size().cumsum().plot(ax=axes[1], color="#1f77b4")
axes[1].set_xlabel("Time"); axes[1].set_ylabel("Cumulative events")
plt.tight_layout(); plt.show()''')
co(r'''# map: associated epicenters colored by depth (association region), UF box dashed
c0 = config.REGION_CENTER
reg = [c0[1] - config.ASSOC_LON_PAD, c0[1] + config.ASSOC_LON_PAD,
       c0[0] - config.ASSOC_LAT_PAD, c0[0] + config.ASSOC_LAT_PAD]
fig = pygmt.Figure()
fig.basemap(region=reg, projection="M13c", frame=["af", f"WSen+tAssociated events {YEAR}"])
fig.coast(shorelines="0.4p,gray30", land="gray95", water="azure1")
pygmt.makecpt(cmap="viridis", series=[0, 25])
fig.plot(x=EV2.longitude, y=EV2.latitude, style="c0.12c", fill=EV2.depth, cmap=True, pen="0.1p,gray20")
fig.plot(x=[UF_BOX[0], UF_BOX[1], UF_BOX[1], UF_BOX[0], UF_BOX[0]],
         y=[UF_BOX[2], UF_BOX[2], UF_BOX[3], UF_BOX[3], UF_BOX[2]], pen="0.8p,black,-")
fig.colorbar(frame="af+lDepth (km)")
fig.basemap(map_scale="jBL+w20k+o0.5c/0.5c")
fig.show()''')

# ---------------------------------------------------------------- stage 3 augment + phs
md(r"""## Stage 3 — pick augmentation + PHS file

Augment rescans the daily picks for arrivals PyOcto missed (updates the assignment in place, with a
backup); `write_phs` then emits the HYPOINVERSE phase file from the augmented assignment.""")
co(r'''core.run_augment_year(MODEL, YEAR)
core.write_phs(MODEL, YEAR)''')

# ---------------------------------------------------------------- stage 4 locate + QC
md(r"""## Stage 4 — absolute location (HYPOINVERSE) and QC

QC gate = `uf_cluster.QC` (erh<5, erz<5, gap<270, num>5, rms<1.0) — the same gate the relocation
stage applies internally, shown here so you see what will survive *before* launching stage 5.""")
co(r'''core.run_hypoinverse_year(MODEL, YEAR, velmodel=VELMODEL)''')
co(r'''# ---- check: location quality + the QC gate ----
SM = uf.read_sum(SUM_PATH)
QCOK = uf.apply_qc(SM)
print(f"located: {len(SM):,}   QC pass: {len(QCOK):,} ({100 * len(QCOK) / max(len(SM), 1):.0f}%)")
for k, op in [("erh", "<"), ("erz", "<"), ("gap", "<"), ("num", ">"), ("rms", "<")]:
    thr = uf.QC[k]
    frac = (SM[k] < thr).mean() if op == "<" else (SM[k] > thr).mean()
    print(f"  {k} {op} {thr}: {100 * frac:.0f}% pass")
fig, axes = plt.subplots(1, 4, figsize=(13, 2.8))
for ax, colname, thr in zip(axes, ["rms", "erh", "erz", "gap"],
                            [uf.QC["rms"], uf.QC["erh"], uf.QC["erz"], uf.QC["gap"]]):
    ax.hist(SM[colname].dropna(), bins=40, color="#1f77b4")
    ax.axvline(thr, color="crimson", lw=1.2)
    ax.set_xlabel(colname.upper() if colname != "gap" else "Gap (deg)")
axes[0].set_ylabel("Events"); plt.tight_layout(); plt.show()''')
co(r'''# map: located events, QC pass vs fail — UF-subregion zoom (10 km scale bar)
fig = pygmt.Figure()
fig.basemap(region=list(UF_BOX), projection="M12c", frame=["af", f"WSen+tLocated events {YEAR} (UF box)"])
fig.coast(shorelines="0.4p,gray30", land="gray97", water="azure1")
fail = SM.loc[~SM.index.isin(uf.apply_qc(SM).index)] if len(SM) else SM
inb = lambda d: d[(d.lon.between(UF_BOX[0], UF_BOX[1])) & (d.lat.between(UF_BOX[2], UF_BOX[3]))]
smb, qcb = inb(SM), inb(QCOK)
if len(smb):
    fig.plot(x=smb.lon, y=smb.lat, style="c0.10c", fill="gray70", pen="0.1p,gray40", label="All located")
if len(qcb):
    fig.plot(x=qcb.lon, y=qcb.lat, style="c0.12c", fill="#1f77b4", pen="0.1p,black", label="QC pass")
fig.basemap(map_scale="jBL+w10k+o0.4c/0.4c")
fig.legend(position="JTR+jTR+o0.2c", box="+gwhite+p0.5p")
print(f"UF box: {len(smb)} located, {len(qcb)} QC-pass")
fig.show()''')

# ---------------------------------------------------------------- stage 5 relocation
md(r"""## Stage 5 — relocation (HypoDD dt.ct + dt.cc)

Self-fed from this year's association. `RELOC_THROUGH="hypoinverse"` stops at the QC'd absolute subset
(fast sanity pass); `"dtcc"` runs the full chain incl. GPU cross-correlation (**hours** for dense years).
Results land in `outputs/reloc/`.""")
co(r'''cmd = [sys.executable, "-m", "ufpipe.run_pipeline", "--model", MODEL, "--years", str(YEAR),
       "--stage-from", "relocate", "--through", RELOC_THROUGH]
if RELOC_THROUGH == "dtcc":
    cmd.append("--clean-cache")
print("$", " ".join(cmd)); subprocess.run(cmd, check=True)''')
co(r'''# ---- check: relocated catalog vs absolute (only after --through dtcc) ----
RELOC_COLS = ["id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez", "yr", "mo", "dy",
              "hr", "mi", "sc", "mag", "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid"]
frel = os.path.join(RELOC_ROOT, "results", "hypoDD.reloc.dtcc")
if not os.path.exists(frel):
    print(f"(no dt.cc result yet at {frel} — run stage 5 with RELOC_THROUGH='dtcc')")
else:
    RL = pd.read_csv(frel, sep=r"\s+", names=RELOC_COLS)
    print(f"dt.cc-relocated events: {len(RL):,}  (cc links: median NCCP {RL.nccp.median():.0f}, NCCS {RL.nccs.median():.0f})")
    from scipy.spatial import cKDTree
    def med_nnd_m(d):
        xy = np.c_[d.lon * 111.0 * np.cos(np.radians(d.lat.mean())), d.lat * 111.0, d.depth]
        t = cKDTree(xy); dd, _ = t.query(xy, k=2)
        return float(np.median(dd[:, 1]) * 1000.0)
    qcb = uf.apply_qc(uf.read_sum(SUM_PATH))
    qcb = qcb[(qcb.lon.between(UF_BOX[0], UF_BOX[1])) & (qcb.lat.between(UF_BOX[2], UF_BOX[3]))]
    if len(qcb) > 2 and len(RL) > 2:
        print(f"median nearest-neighbour distance: absolute {med_nnd_m(qcb):.0f} m -> dt.cc {med_nnd_m(RL):.0f} m")
    fig = pygmt.Figure()
    fig.basemap(region=list(UF_BOX), projection="M9c", frame=["af", "WSen+tAbsolute (QC)"])
    fig.coast(shorelines="0.4p,gray30", land="gray97", water="azure1")
    if len(qcb):
        fig.plot(x=qcb.lon, y=qcb.lat, style="c0.10c", fill="gray50", pen="0.1p,gray30")
    fig.basemap(map_scale="jBL+w10k+o0.4c/0.4c")
    fig.shift_origin(xshift="10c")
    fig.basemap(region=list(UF_BOX), projection="M9c", frame=["af", "wSen+tHypoDD dt.cc"])
    fig.coast(shorelines="0.4p,gray30", land="gray97", water="azure1")
    fig.plot(x=RL.lon, y=RL.lat, style="c0.10c", fill="#d62728", pen="0.1p,black")
    fig.basemap(map_scale="jBL+w10k+o0.4c/0.4c")
    fig.show()''')

# ---------------------------------------------------------------- summary
md(r"""## Summary — the year at a glance""")
co(r'''# comprehensive computed summary (each row reflects what exists on disk right now)
rows = []
files = glob.glob(os.path.join(config.picks_dir(MODEL, YEAR), f"picks_{YEAR}.*.csv"))
rows.append(("1 detection", f"{len(files)} daily files",
             f"{len(core.load_picks(MODEL, YEAR)):,} picks" if files else "-"))
try:
    ev = pd.read_csv(config.pyocto_events(MODEL, YEAR))
    rows.append(("2 association", f"{len(ev):,} events", f"{int(ev.picks.sum()):,} assigned picks"))
except FileNotFoundError:
    rows.append(("2 association", "-", "not run"))
phs = os.path.join(config.MODELS, MODEL, "HypoInv", "PHS", f"UF{YEAR}.phs")
rows.append(("3 augment+phs", "written" if os.path.exists(phs) else "-", ""))
if os.path.exists(SUM_PATH):
    sm = uf.read_sum(SUM_PATH); qc = uf.apply_qc(sm)
    rows.append(("4 locate", f"{len(sm):,} located", f"{len(qc):,} QC-pass ({100 * len(qc) / max(len(sm), 1):.0f}%)"))
else:
    rows.append(("4 locate", "-", "not run"))
frel = os.path.join(RELOC_ROOT, "results", "hypoDD.reloc.dtcc")
if os.path.exists(frel):
    n = sum(1 for _ in open(frel))
    rows.append(("5 relocate", f"{n:,} dt.cc events", RELOC_ROOT))
else:
    rows.append(("5 relocate", "-", f"pending -> {RELOC_ROOT}"))
summary = pd.DataFrame(rows, columns=["stage", "output", "detail"])
print(f"=== {MODEL} {YEAR} ===")
print(summary.to_string(index=False))
print("\nTake-homes: check (i) pick-count timeline for station dropouts, (ii) picks/event and the QC pass "
      "fraction before trusting locations, (iii) the min-probability floor equals the intended threshold, "
      "(iv) NND tightening absolute -> dt.cc as the relocation sanity metric.")''')

nb = nbf.v4.new_notebook(cells=cells,
                         metadata={"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"}})
nbf.write(nb, NB)
print(f"wrote {NB} ({len(cells)} cells, unexecuted)")
