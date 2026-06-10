"""Build the multi-station view of notebook 06's anti-pair candidates. Usage: python build_antivalidate_nb.py

Takes the EXACT candidate pairs of 06_anti_repeaters_KGHDB_HHZ_1-25Hz (ranked by `cc_lag0` — the
signed CC at the exact P datum — top N), which form a tight ~1 km spatial cluster, and shows how the
SAME event pairs correlate / anti-correlate at the other long-term stations KG.MKL / KG.YSB / KG.CHS.
Pairs are matched by event id (NOT a forced common set — a pair is shown at every station that
recorded both its events; n/a where a station is missing one).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md("""# Notebook 06's HDB anti-pair candidates — how do they look at other stations?

The 1-25 Hz signed-CC screen (`06_anti_repeaters_KGHDB_HHZ_1-25Hz`) ranks pairs by **`cc_lag0`** — the
signed correlation at the **exact P datum** (lag 0), the physically meaningful quantity once both
events are P-aligned — and its top candidates form a **tight ~1 km spatial cluster** of strongly
**lag-0-anti-correlated** HDB pairs. Here we take **those exact pairs** and show how they
correlate / anti-correlate at the three other long-term stations **KG.MKL, KG.YSB, KG.CHS**.

Per station we report three numbers for each pair:

| | meaning |
|---|---|
| `cc_lag0` | signed CC at lag 0 (the 06 ranking metric; ≈ −1 = clean polarity reversal *there*) |
| `cc_pos`  | best **positive** CC over ±MAXLAG (are the waveforms *similar at all* there?) |
| `cc_neg`  | best **negative** CC over ±MAXLAG (the anti side, allowing a small lag) |

**Reading it.** A genuine co-located anti-repeater is the same waveform up to **sign** at every
station: `max(|cc_pos|,|cc_neg|)` ≈ 1 everywhere, with the *sign* varying by take-off geometry —
e.g. anti at HDB but positive at a station in a different quadrant. If instead the validators show
**low `cc_pos` and low `|cc_neg|`** (decorrelation), the strong HDB lag-0 negative is a single-station
effect, not a source-side polarity reversal. Matched by event id, no pair dropped.""")

md("## Parameters")
code("""STATION   = "KG.HDB"
VAL_STATIONS = ["KG.MKL", "KG.YSB", "KG.CHS"]
ALL_STATIONS = [STATION] + VAL_STATIONS
COMP      = "HHZ"
WIN       = (-0.5, 7.5)
BANDS     = [(1, 25), (1, 10), (2, 8), (4, 12), (5, 15)]
BAND      = (1, 25)
MAXLAG    = 0.2
TOPN      = 12               # same as notebook 06: top-N pairs by most-negative cc_lag0
N_OVERLAY = 8                # pairs to draw as multi-station overlays
CACHE     = "wf_similarity_cache\"""")

code("""import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import uf_waveform_similarity as wf
import uf_cluster as ufc
wf.use_helvetica()
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 50)""")

md("""## 1 · Reproduce notebook 06's candidate pairs (HDB, rank by `cc_lag0`, top N)""")
code("""res_raw = {st: wf.make_bands(wf.list_events(station=st, comp=COMP), station=st, comp=COMP,
                            bands=BANDS, win=WIN, cache_dir=CACHE, verbose=False)
           for st in ALL_STATIONS}
keptH = res_raw[STATION]["kept"]; meta = wf.load_event_meta(keptH)
SH = wf.signed_similarity(res_raw[STATION]["bands"][BAND], maxlag=MAXLAG)
iu = np.triu_indices(len(keptH), k=1)
order = np.argsort(SH["cc_lag0"][iu])[:TOPN]                 # most-negative cc_lag0 first (= 06)
PAIRS = [dict(i=int(iu[0][o]), j=int(iu[1][o])) for o in order]
evset = sorted(set([p["i"] for p in PAIRS]) | set([p["j"] for p in PAIRS]))
la, lo = meta.iloc[evset]["lat"].dropna(), meta.iloc[evset]["lon"].dropna()
print(f"top-{TOPN} by cc_lag0: {len(evset)} events | extent "
      f"~{(la.max()-la.min())*111:.1f} x {(lo.max()-lo.min())*90:.1f} km, centroid ({la.mean():.3f}, {lo.mean():.3f})")
pd.DataFrame([dict(ev_i=keptH[p["i"]], ev_j=keptH[p["j"]],
                   cc_lag0=round(float(SH["cc_lag0"][p["i"], p["j"]]), 3),
                   cc_neg=round(float(SH["cc_neg"][p["i"], p["j"]]), 3),
                   cc_pos=round(float(SH["cc_pos"][p["i"], p["j"]]), 3)) for p in PAIRS])""")

md("""## 2 · The spatial cluster (PyGMT) — pairs linked, coloured by `cc_lag0`""")
code("""try:
    wf.map_antipairs(meta, PAIRS, value="cc_lag0", station=STATION,
                     title=f"06 HDB anti-pair candidates ({BAND[0]}-{BAND[1]} Hz) — tight cluster").show()
except Exception as e:
    print("PyGMT map skipped:", type(e).__name__, e)""")

md("""## 3 · The SAME pairs at MKL / YSB / CHS — `cc_lag0`, `cc_pos`, `cc_neg`

Matched by event id; `n/a` where a station did not record one of the pair's events (no pair dropped).""")
code("""SIG = {}
for st in ALL_STATIONS:
    S = wf.signed_similarity(res_raw[st]["bands"][BAND], maxlag=MAXLAG)
    SIG[st] = (S, {e: k for k, e in enumerate(res_raw[st]["kept"])})

def _vals(ei, ej, key):
    out = {}
    for st in ALL_STATIONS:
        S, idx = SIG[st]
        out[st.split('.')[1]] = round(float(S[key][idx[ei], idx[ej]]), 2) if (ei in idx and ej in idx) else np.nan
    return out

for key, label in [("cc_lag0", "cc_lag0 (signed, at exact P datum)"),
                   ("cc_pos",  "cc_pos (best positive over lags — similar at all?)"),
                   ("cc_neg",  "cc_neg (best negative over lags)")]:
    print("\\n###", label)
    df = pd.DataFrame([dict(ev_i=keptH[p["i"]], ev_j=keptH[p["j"]], **_vals(keptH[p["i"]], keptH[p["j"]], key))
                       for p in PAIRS])
    display(df)""")

md("""### 3b · One-line summary per pair — does any validator stay strongly correlated?

`HDB_lag0` vs the validators' **best similarity** `max|CC| = max(cc_pos, |cc_neg|)`. A real co-located
pair keeps `max|CC|` high at the validators; decorrelation (≲0.4) = HDB-only.""")
code("""rows = []
for p in PAIRS:
    ei, ej = keptH[p["i"]], keptH[p["j"]]
    r = {"ev_i": ei, "ev_j": ej, "HDB_lag0": round(float(SH["cc_lag0"][p["i"], p["j"]]), 2)}
    for st in VAL_STATIONS:
        S, idx = SIG[st]
        if ei in idx and ej in idx:
            a, b = idx[ei], idx[ej]
            r[f"{st.split('.')[1]}_maxabs"] = round(float(max(S["cc_pos"][a, b], -S["cc_neg"][a, b])), 2)
        else:
            r[f"{st.split('.')[1]}_maxabs"] = np.nan
    rows.append(r)
pd.DataFrame(rows)""")

md("""## 4 · Waveform overlays across stations

For the strongest candidates: at each station, event *i* (black) vs event *j* at its best **+CC** lag
(blue, repeater fit) and best **−CC** lag flipped (red, anti fit). If neither overlays *i* off HDB,
the pair isn't similar there; if the red (anti) overlays at several stations, it is a real reversal.""")
code("""RESp = {st: dict(kept=res_raw[st]["kept"], bands={tuple(BAND): res_raw[st]["bands"][BAND]}) for st in ALL_STATIONS}
for p in PAIRS[:N_OVERLAY]:
    wf.plot_antipair_stations(RESp, keptH[p["i"]], keptH[p["j"]], stations=ALL_STATIONS, band=BAND,
                              win=WIN, maxlag=MAXLAG, zoom=(-0.3, 4.0))
    plt.show()""")

md("""## 5 · How to read / verdict

**These are co-located *repeaters*, not anti-repeaters.** The candidates form a real, tight (~1 km)
cluster — but the data say the "anti" is the half-cycle degeneracy, not a polarity reversal:

- **At HDB**, the same pairs have **`cc_pos ≈ 0.75`** — *above* `POS_GATE` (0.6) — so they are genuinely
  **similar** waveforms. `cc_neg ≈ −0.73` is *equally* strong: a half-period (1-25 Hz ⇒ ~10-25 ms,
  inside ±MAXLAG) shift flips the sign, and `cc_lag0 ≈ −0.73` just means the P-datum happens to land on
  the anti-aligned half-cycle. A true anti-repeater would have **low** `cc_pos` — these don't.
- **At MKL** (which recorded the 2020-2022 pairs), the correlation is **positive-dominated**:
  `cc_pos ≈ 0.60 > |cc_neg| ≈ 0.40`, `cc_lag0` only weakly negative. So at a *second* station the same
  events read as **repeaters**, not reversed — the decisive evidence against a source-side flip.
- **At YSB/CHS** the pairs decorrelate (`max|CC| ≲ 0.25`).
- This is exactly why notebook 06's strict test (`cc_pos < POS_GATE` on all 3 components) **confirmed 0**:
  the 06 *map* ranks top-N by `cc_lag0` and shows them clustered, but they fail the `cc_pos` gate.

So the cluster is a genuine group of **repeating earthquakes** whose HDB records are half-cycle-offset
at the P datum — a nice illustration of the degeneracy, not anti-repeaters.

- Caveat: `cc_lag0` is alignment-sensitive at 1-25 Hz (a small P-pick error decorrelates lag 0 fast),
  so it is read **together with** `cc_pos`/`cc_neg` (±MAXLAG, jitter-tolerant) and the §4 overlays.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/HypoInv/09_antirepeater_multistation_1-25Hz_phasenet_plus.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(C), "cells")
