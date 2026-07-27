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
COMPONENT = "Z"                    # component to plot: "Z" | "N" | "E". NOTE S picks are made on the
                                   # horizontals, so a station whose Z has an outage can still carry a valid
                                   # S pick -- switch to "N"/"E" to see it (the skip note lists what exists).
N_AUG_MAX = 20                     # cap for the SECOND figure (all augmented events, up to this many)
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
co(r'''def day_glob(sta, net, jstr, band, comp=None):
    comp = comp or COMPONENT
    arch = ARCH.get(sta, (config.KS_KG_DIR, "HH"))[0]
    return sorted(glob.glob(os.path.join(arch, sta, f"{band}{comp}.D", f"{net}.{sta}..{band}{comp}.D.{jstr}")))

def comps_with_data(sta, net, jstr, band, o_utc):
    """Which components actually have samples in the plot window (for an honest skip message)."""
    ok = []
    for c in ("Z", "N", "E"):
        for f in day_glob(sta, net, jstr, band, c):
            try:
                st = obspy.read(f, headonly=True)
                if any(tr.stats.starttime <= o_utc + POST_S and tr.stats.endtime >= o_utc - PRE_S for tr in st):
                    ok.append(c)
            except Exception:
                pass
            break
    return ok

def event_traces(row):
    """(dist_km, trace, p_rel, s_rel, p_aug, s_aug) per associated station, + skip notes + n associated."""
    ot = pd.Timestamp(row["time"]).tz_localize(None)
    o_utc = obspy.UTCDateTime(ot.to_pydatetime())
    jstr = f"{ot.year}.{ot.dayofyear:03d}"
    picks = ASG[ASG["event_idx"] == int(row["idx"])].copy()
    picks["code"] = picks["station"].str.rstrip(".").str.split(".").str[1]
    picks["net"]  = picks["station"].str.split(".").str[0]
    out, skipped = [], []
    for code, g in picks.groupby("code"):
        if code not in COORD:
            skipped.append((code, "no coords")); continue
        net = g["net"].iloc[0]
        band = ARCH.get(code, (None, "HH"))[1]
        fs = day_glob(code, net, jstr, band)
        if not fs:
            skipped.append((code, "no waveform file")); continue
        try:
            # MERGE first: a fragmented station-day (e.g. KG.HDB 2010.018 = 25 traces) would otherwise be
            # represented by its first fragment only, which may not span the origin -> 0 samples after trim
            # and the station silently vanishes from the section.
            st = obspy.read(fs[0])
            if len(st) > 1:
                st.merge(method=1, fill_value=0)
            tr = st[0]
            tr.trim(o_utc - PRE_S, o_utc + POST_S)
            if tr.stats.npts < 10:
                have = comps_with_data(code, net, jstr, band, o_utc)
                skipped.append((code, f"{COMPONENT} has no data in window"
                                      + (f"; but {'/'.join(have)} does" if have else "")))
                continue
            tr.detrend("demean")
            if BANDPASS:
                tr.filter("bandpass", freqmin=BANDPASS[0], freqmax=BANDPASS[1], corners=4, zerophase=True)
        except Exception as e:
            skipped.append((code, f"read error: {type(e).__name__}")); continue
        d_m, _, _ = gps2dist_azimuth(row["latitude"], row["longitude"], *COORD[code])
        p_rel = s_rel = None; p_aug = s_aug = False
        for _, pk in g.iterrows():
            rel = float(pk["time"]) - o_utc.timestamp        # pick seconds after origin
            aug = ("source" in g.columns) and (pk.get("source", "pyocto") != "pyocto")
            if pk["phase"] == "P": p_rel, p_aug = rel, aug
            elif pk["phase"] == "S": s_rel, s_aug = rel, aug
        out.append((d_m / 1000.0, tr, p_rel, s_rel, p_aug, s_aug))
    out.sort(key=lambda x: x[0])
    return out[:MAX_STA], skipped, len(picks["code"].unique())


from matplotlib.lines import Line2D

def plot_sections(events, title):
    """Record-section grid: one panel per event. Augmented picks are dashed + green triangle."""
    if not len(events):
        print(f"({title}: no events)"); return None
    rows_n = math.ceil(len(events) / COLS)
    fig, axes = plt.subplots(rows_n, COLS, figsize=(COLS * 3.6, rows_n * 2.6), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(events):]:
        ax.axis("off")
    for k, (_, row) in enumerate(events.iterrows()):
        ax = axes[k]
        traces, skipped, n_assoc = event_traces(row)
        if skipped:
            print(f"  {pd.Timestamp(row['time']):%Y-%m-%d %H:%M:%S}: {len(traces)}/{n_assoc} stations plotted; "
                  f"skipped {skipped}")
        if not traces:
            ax.text(0.5, 0.5, "no waveforms", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off"); continue
        dmax = max(d for d, *_ in traces) or 1.0
        off = 0.10 * dmax
        for d, tr, p_rel, s_rel, p_aug, s_aug in traces:
            t = tr.times() - PRE_S
            y = tr.data / (np.max(np.abs(tr.data)) or 1.0) * off * 0.9
            ax.plot(t, y + d, lw=0.4, color="0.25")
            for rel, col, aug in ((p_rel, "crimson", p_aug), (s_rel, "royalblue", s_aug)):
                if rel is None:
                    continue
                if aug:
                    ax.plot([rel, rel], [d - off * 0.7, d + off * 0.7], color=col, lw=2.0, ls=(0, (2, 1)))
                    ax.plot([rel], [d + off * 0.85], marker="v", ms=4, color="darkgreen", mec="k", mew=0.3)
                else:
                    ax.plot([rel, rel], [d - off * 0.5, d + off * 0.5], color=col, lw=1.1)
        ot = pd.Timestamp(row["time"]).tz_localize(None)
        ax.set_title(f"{ot:%Y-%m-%d %H:%M:%S}  z={row['depth']:.0f} km  ({len(traces)}/{n_assoc} sta)", fontsize=8)
        ax.set_xlim(-PRE_S, POST_S)
        ax.set_xlabel("Time after origin (s)", fontsize=7)
        ax.set_ylabel("Epicentral distance (km)", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.legend([Line2D([0], [0], color="crimson", lw=1.5), Line2D([0], [0], color="royalblue", lw=1.5),
                Line2D([0], [0], color="darkgreen", marker="v", ls="none", ms=5)],
               ["P pick", "S pick", "added by augmentation"], loc="upper right", ncol=3,
               framealpha=1.0, edgecolor="0.6")
    fig.suptitle(title, y=1.005, fontsize=11)
    plt.tight_layout(); plt.show()
    return fig

# ---- FIGURE 1: N random located events ----
plot_sections(pick, f"Record sections — {len(pick)} random located events ({MODEL} {YEAR})")''')

md(r"""## Figure 2 — the augmented events (their own figure)

Augmented picks are rare, so a random draw usually misses them. This figure plots **every located event that
gained a pick** in the augment stage (up to `N_AUG_MAX`), with the added arrivals drawn as **thick dashed bars
with a green triangle** and the PyOcto ones as thin solid bars — i.e. before vs after augmentation, per event.""")
co(r'''if "source" not in ASG.columns:
    print("assignment has no 'source' column — run the augment stage first")
else:
    aug_ids = set(ASG.loc[ASG["source"] != "pyocto", "event_idx"].unique())
    AUG = EVL[EVL["idx"].isin(aug_ids)].sort_values("time").head(N_AUG_MAX).reset_index(drop=True)
    n_tot = len(aug_ids)
    print(f"{n_tot} events gained augmented picks; {len(EVL[EVL['idx'].isin(aug_ids)])} of them are located "
          f"(only located ones can be shown) -> plotting {len(AUG)}")
    for _, r in AUG.iterrows():
        a = ASG[(ASG.event_idx == int(r["idx"])) & (ASG.source != "pyocto")]
        print(f"  {pd.Timestamp(r['time']):%Y-%m-%d %H:%M:%S}: +{len(a)} "
              f"({', '.join(a.station.str.rstrip('.') + ' ' + a.phase)})")
    plot_sections(AUG, f"Record sections — {len(AUG)} AUGMENTED events ({MODEL} {YEAR})")''')

md(r"""## Augmentation — what the added picks are

Stage 3 (`augment`) exists because **PyOcto is a *streaming* associator**: it scans picks in time order and
*finalizes* an event as soon as the gate is met, so picks at **farther stations that arrive later in the
stream are never reconsidered** — and picks rejected against PyOcto's *initial* hypothesized hypocenter are
not re-offered after refinement moves it. Augmentation takes PyOcto's refined hypocenter as a **seed** and
rescans the day's raw picks for orphans that (i) are at a station within `radius_km` (100 km), (ii) match the
predicted arrival within `tolerance_s` (1.0 s), (iii) are phase-consistent (P→P, S→S), (iv) have probability
≥ `min_pick_probability` (0.3), and (v) are not already assigned. Safeguards against stealing picks between
close-in-time events: phase-strict matching, **best-match-wins** (smallest residual), and **drop-on-tie**
(if the best two residuals differ by < `tie_threshold_s` = 0.2 s, the pick is dropped for both).

Set `ONLY_AUGMENTED = True` in the parameters cell to plot only the events that gained picks; the added
arrivals are drawn as **thick dashed bars with a green triangle**, the PyOcto ones as thin solid bars.

```python
# how much was augmented this year:
A = pd.read_csv(config.pyocto_assign(MODEL, YEAR))
print(A["source"].value_counts())                       # pyocto vs augmented
print(A.loc[A.source!="pyocto", "event_idx"].nunique(), "events gained >=1 pick")
```

**Is it redundant if PyOcto were perfect?** In principle yes — but this is *architectural*, not a PyOcto bug:
a streaming associator cannot revisit finalized events. The size of the effect scales with **station
density**: on a sparse year it is negligible, on a dense one (GJ 2016, NS 2017+) it recovers far more.

## Notes

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
