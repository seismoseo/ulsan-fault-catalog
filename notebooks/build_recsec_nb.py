#!/usr/bin/env python
"""Generate notebooks/01.Record_sections_located_events.ipynb — record sections of N random located
events, each vertical (Z) trace plotted vs epicentral distance with its ASSOCIATED P/S pick times marked.

    python notebooks/build_recsec_nb.py            # (re)writes the unexecuted notebook

Kernel: base (obspy/pandas). Reads the pyocto events + assignment (which P/S picks belong to each event) and
the located .sum (to restrict to genuinely-located events), cuts Z waveforms from the continuous archive
around each origin, and overlays the picks. Parameters (YEAR/MODEL/N/window/filter/seed) at the top.
"""
import nbformat as nbf

NB = "notebooks/01.Record_sections_located_events.ipynb"
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
co = lambda s: C.append(nbf.v4.new_code_cell(s))

md(r"""# Record sections — random located events with P/S picks

For **N random located events**, plot each station's vertical (Z) waveform against epicentral distance,
with the event's **associated** P and S pick times marked (red = P, blue = S). A quick visual check that the
picks that PyOcto associated into each event actually land on real arrivals.

**Kernel: `base`.** Origin + associated picks come from the pyocto event/assignment tables (the source of
truth for which pick belongs to which event); events are restricted to those that HYPOINVERSE located.
Parameters at the top.""")

co(r'''# ================================ PARAMETERS ================================
YEAR      = 2010
MODEL     = "phasenet_plus"        # picker whose picks/associations to plot
VELMODEL  = "kim2011"
N_EVENTS  = 20                     # how many random located events
SEED      = 0                      # RNG seed (change for a different random draw)
PRE_S     = 5.0                    # seconds before origin to start each trace
POST_S    = 35.0                   # seconds after origin
BANDPASS  = (2.0, 20.0)            # display filter (Hz); None = raw
MAX_STA   = 12                     # cap traces per panel (nearest stations)
COLS      = 4                      # subplot columns (rows = ceil(N/COLS))

import os, glob, math, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import obspy
from obspy.geodetics import gps2dist_azimuth

import sys; sys.path.insert(0, "src")
from ufpipe import config, stations
from uflib import uf_cluster as uf

try:
    fm.findfont("Helvetica", fallback_to_default=False); plt.rcParams["font.family"] = "Helvetica"
except Exception:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams.update({"figure.dpi": 110, "legend.framealpha": 1.0,
                     "legend.facecolor": "white", "legend.edgecolor": "0.6"})

# station coords + archive (which waveform dir each station's data lives in)
ST = stations.build_year_table(YEAR)
COORD = {r.sta: (r.lat, r.lon) for r in ST.itertuples()}
ARCH = {r["sta"]: (r["archive"], r["band"]) for r in stations.discover_rows(YEAR)}
print(f"{len(COORD)} stations with coords for {YEAR}")''')

md(r"""## Select N random located events

An event is *located* if it appears in the HYPOINVERSE `.sum`. We match the pyocto events (which carry the
pick associations) to the located `.sum` by origin time, then draw `N_EVENTS` at random.""")
co(r'''EV  = pd.read_csv(config.pyocto_events(MODEL, YEAR), parse_dates=["time"])
ASG = pd.read_csv(config.pyocto_assign(MODEL, YEAR))
SM  = uf.read_sum(os.path.join(config.MODELS, MODEL, "HypoInv", VELMODEL, f"UF{YEAR}.sum"))
SM  = SM.dropna(subset=["time"]).copy()

# match pyocto events to located .sum rows by nearest origin time (<=2 s) -> keep only located events
ev_t = EV["time"].dt.tz_localize(None).values.astype("datetime64[ns]")
sm_t = pd.to_datetime(SM["time"]).values.astype("datetime64[ns]")
order = np.argsort(sm_t); sm_sorted = sm_t[order]
pos = np.searchsorted(sm_sorted, ev_t)
located = np.zeros(len(EV), dtype=bool)
for i, p in enumerate(pos):
    for q in (p - 1, p):
        if 0 <= q < len(sm_sorted) and abs((ev_t[i] - sm_sorted[q]) / np.timedelta64(1, "s")) <= 2.0:
            located[i] = True; break
EVL = EV[located].reset_index(drop=True)
print(f"{len(EVL)} located events (of {len(EV)} associated); drawing {N_EVENTS}")

rng = np.random.default_rng(SEED)
pick = EVL.iloc[rng.choice(len(EVL), size=min(N_EVENTS, len(EVL)), replace=False)].sort_values("time")
pick = pick.reset_index(drop=True)
pick[["idx", "time", "latitude", "longitude", "depth", "picks"]]''')

md(r"""## Cut waveforms + plot

Each panel = one event. Traces are the Z component of every station that has an associated pick, offset by
its epicentral distance; the associated **P (red)** and **S (blue)** pick times are drawn as short bars on
each trace. Time axis is seconds after origin.""")
co(r'''def day_glob(sta, net, jstr, band):
    arch = ARCH.get(sta, (config.KS_KG_DIR, "HH"))[0]
    return sorted(glob.glob(os.path.join(arch, sta, f"{band}Z.D", f"{net}.{sta}..{band}Z.D.{jstr}")))

def event_traces(row):
    """Return list of (dist_km, trace, p_rel, s_rel) for one event's associated stations (Z only)."""
    ot = pd.Timestamp(row["time"]).tz_localize(None)
    o_utc = obspy.UTCDateTime(ot.to_pydatetime())
    jstr = f"{ot.year}.{ot.dayofyear:03d}"
    picks = ASG[ASG["event_idx"] == int(row["idx"])].copy()
    picks["code"] = picks["station"].str.rstrip(".").str.split(".").str[1]
    picks["net"]  = picks["station"].str.split(".").str[0]
    out = []
    for code, g in picks.groupby("code"):
        if code not in COORD:
            continue
        net = g["net"].iloc[0]
        band = ARCH.get(code, (None, "HH"))[1]
        fs = day_glob(code, net, jstr, band)
        if not fs:
            continue
        try:
            tr = obspy.read(fs[0])[0]
            tr.trim(o_utc - PRE_S, o_utc + POST_S)
            if tr.stats.npts < 10:
                continue
            tr.detrend("demean")
            if BANDPASS:
                tr.filter("bandpass", freqmin=BANDPASS[0], freqmax=BANDPASS[1], corners=4, zerophase=True)
        except Exception:
            continue
        d_m, _, _ = gps2dist_azimuth(row["latitude"], row["longitude"], *COORD[code])
        p_rel = s_rel = None
        for _, pk in g.iterrows():
            rel = float(pk["time"]) - o_utc.timestamp        # pick seconds after origin
            if pk["phase"] == "P": p_rel = rel
            elif pk["phase"] == "S": s_rel = rel
        out.append((d_m / 1000.0, tr, p_rel, s_rel))
    out.sort(key=lambda x: x[0])
    return out[:MAX_STA]

rows_n = math.ceil(len(pick) / COLS)
fig, axes = plt.subplots(rows_n, COLS, figsize=(COLS * 3.6, rows_n * 2.6))
axes = np.atleast_1d(axes).ravel()
for ax in axes[len(pick):]:
    ax.axis("off")

for k, row in pick.iterrows():
    ax = axes[k]
    traces = event_traces(row)
    if not traces:
        ax.text(0.5, 0.5, "no waveforms", ha="center", va="center", transform=ax.transAxes); ax.axis("off"); continue
    dmax = max(d for d, *_ in traces) or 1.0
    off = 0.10 * (dmax if dmax > 0 else 1.0)               # vertical spacing in distance units
    for d, tr, p_rel, s_rel in traces:
        t = tr.times() - PRE_S                              # seconds after origin
        y = tr.data / (np.max(np.abs(tr.data)) or 1.0) * off * 0.9
        ax.plot(t, y + d, lw=0.4, color="0.25")
        if p_rel is not None:
            ax.plot([p_rel, p_rel], [d - off * 0.5, d + off * 0.5], color="crimson", lw=1.1)
        if s_rel is not None:
            ax.plot([s_rel, s_rel], [d - off * 0.5, d + off * 0.5], color="royalblue", lw=1.1)
    ot = pd.Timestamp(row["time"]).tz_localize(None)
    ax.set_title(f"{ot:%Y-%m-%d %H:%M:%S}  z={row['depth']:.0f} km  ({len(traces)} sta)", fontsize=8)
    ax.set_xlim(-PRE_S, POST_S)
    ax.set_xlabel("Time after origin (s)", fontsize=7)
    ax.set_ylabel("Epicentral distance (km)", fontsize=7)
    ax.tick_params(labelsize=6)

# one shared legend (P red, S blue)
from matplotlib.lines import Line2D
fig.legend([Line2D([0], [0], color="crimson", lw=1.5), Line2D([0], [0], color="royalblue", lw=1.5)],
           ["P pick", "S pick"], loc="upper right", ncol=2, framealpha=1.0, edgecolor="0.6")
fig.suptitle(f"Record sections — {len(pick)} random located events ({MODEL} {YEAR})", y=1.005, fontsize=11)
plt.tight_layout(); plt.show()''')

md(r"""## Notes

- Each trace is the **Z** component, amplitude-normalized, offset by its **epicentral** distance; red/blue
  bars are the P/S pick times **PyOcto associated with that event** (seconds after origin).
- Stations without an associated pick, without on-disk data for that day, or missing coordinates are skipped;
  the nearest `MAX_STA` are shown.
- Change `SEED` for a different random draw, or `N_EVENTS` / `COLS` for layout. A single event usually has
  more S than P near-source; moveout (P and S bars sloping to larger time with distance) is the sanity signal.""")

nb = nbf.v4.new_notebook(cells=C, metadata={"kernelspec": {"display_name": "Python 3",
                                                           "language": "python", "name": "python3"}})
nbf.write(nb, NB)
print(f"wrote {NB} ({len(C)} cells, unexecuted)")
