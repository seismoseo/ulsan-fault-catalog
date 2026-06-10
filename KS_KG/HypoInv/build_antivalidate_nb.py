"""Build the multi-station anti-repeater VALIDATION notebook (KG.HDB candidates checked at nearby
long-term stations, on a COMMON event set). Usage: python build_antivalidate_nb.py

Premise: a genuine anti-repeater (two co-located events with opposite first motion — mechanisms
straddling a nodal plane) must correlate STRONGLY (|CC|->1) at EVERY station, with the winning sign
set by each station's take-off geometry. A single-station (HDB-only) negative correlation that
DECORRELATES at other stations is an artifact (half-cycle-lag degeneracy on only-moderately-similar
waveforms), not a polarity reversal.

RIGOUR: every station is restricted to the **intersection** of event sets, so all four stations
analyse the *identical* events (no per-station population differences, no missing-event gaps).
Validators: KG.MKL / KG.YSB / KG.CHS — the other ~long-term, ~full-coverage stations.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md("""# Anti-repeater candidates at `KG.HDB` (1-25 Hz) — multi-station validation (common event set)

The 1-25 Hz signed-CC screen flags a small, spatially-compact set of HDB pairs with a strong
**negative** correlation (`cc_neg ≤ NEG_LOOSE`) that are **not** also positively correlated
(`cc_pos < POS_GATE`) — apparent **anti-repeaters**. We validate them at the three other long-term
stations **KG.MKL, KG.YSB, KG.CHS**.

**Same events at every station.** Each station is restricted to the **intersection** of its event set
with the others, so all four analyse the *identical* events (this fixes the earlier per-station
population mismatch — e.g. YSB previously missed several candidate events).

**The test.** A real anti-repeater is *co-located* with an *opposite* mechanism, so at **every**
station the two waveforms are identical up to **sign**: one of {repeater-fit, anti-fit} is near
|CC| = 1, the winning sign varying by station per its take-off angle. If the strong negative is
**HDB-only** and the pair **decorrelates** (both fits weak) at MKL/YSB/CHS, the HDB `cc_neg` is the
half-cycle-lag artifact of two only-moderately-similar waveforms (note HDB `cc_pos` ≈ 0.5 already),
*not* a polarity reversal.""")

md("## Parameters")
code("""STATION   = "KG.HDB"           # primary (screening) station
VAL_STATIONS = ["KG.MKL", "KG.YSB", "KG.CHS"]   # nearby long-term validators
ALL_STATIONS = [STATION] + VAL_STATIONS
COMP      = "HHZ"              # vertical (rotation-immune)
WIN       = (-0.5, 7.5)
BANDS     = [(1, 25), (1, 10), (2, 8), (4, 12), (5, 15)]
BAND      = (1, 25)            # stricter band (compact candidate set)
MAXLAG    = 0.2
NEG_LOOSE = -0.7              # looser than the -0.85 headline cut, to surface the candidates you saw
POS_GATE  = 0.6              # genuine anti must NOT also be positively correlated
VAL_OK    = 0.7              # a candidate is "validated" if max|CC| stays >= this at the validators
N_OVERLAY = 8                # candidate pairs to draw as multi-station overlays
CACHE     = "wf_similarity_cache\"""")

code("""import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import uf_waveform_similarity as wf
import uf_cluster as ufc
wf.use_helvetica()
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)""")

md("""## 1 · Build all four stations and restrict to the COMMON event set

`make_bands` per station (each P-aligned on its own pick), then intersect the `kept` lists so every
station's matrix is reindexed to the **same** events in the **same** order.""")
code("""res_raw = {st: wf.make_bands(wf.list_events(station=st, comp=COMP), station=st, comp=COMP,
                            bands=BANDS, win=WIN, cache_dir=CACHE, verbose=False)
           for st in ALL_STATIONS}
common = sorted(set.intersection(*[set(res_raw[st]["kept"]) for st in ALL_STATIONS]))
print("events per station:", {st: len(res_raw[st]["kept"]) for st in ALL_STATIONS})
print("COMMON to all four:", len(common))

def _reindex(res, events, band):
    idx = {e: i for i, e in enumerate(res["kept"])}
    return res["bands"][tuple(band)][[idx[e] for e in events]]

# per-station band matrix + pseudo-res (kept=common) for the overlay function; signed CC on common
RES = {st: dict(kept=common, bands={tuple(BAND): _reindex(res_raw[st], common, BAND)}) for st in ALL_STATIONS}
SIG = {st: wf.signed_similarity(RES[st]["bands"][tuple(BAND)], maxlag=MAXLAG) for st in ALL_STATIONS}
meta = wf.load_event_meta(common)
print("done — all stations on the identical", len(common), "events")""")

md("""## 2 · HDB anti-repeater candidates (1-25 Hz, common set)""")
code("""neg, pos = SIG[STATION]["cc_neg"], SIG[STATION]["cc_pos"]
iu = np.triu_indices(len(common), k=1)
mask = (neg[iu] <= NEG_LOOSE) & (pos[iu] < POS_GATE)
pairs = [dict(i=int(a), j=int(b), cc_neg=round(float(neg[a, b]), 3), cc_pos=round(float(pos[a, b]), 3))
         for a, b in zip(iu[0][mask], iu[1][mask])]
pairs.sort(key=lambda p: p["cc_neg"])
print(f"{len(pairs)} HDB candidate pairs (cc_neg<={NEG_LOOSE} & cc_pos<{POS_GATE}); "
      f"{len(set([p['i'] for p in pairs]) | set([p['j'] for p in pairs]))} events involved")
pd.DataFrame([dict(ev_i=common[p["i"]], ev_j=common[p["j"]], cc_neg=p["cc_neg"], cc_pos=p["cc_pos"])
              for p in pairs])""")

md("""## 3 · Where are they? — spatial clustering (PyGMT)

Each candidate pair = two epicentres joined by a line coloured by `cc_neg`, over the UF subregion.""")
code("""try:
    wf.map_antipairs(meta, pairs, value="cc_neg", station=STATION,
                     title=f"HDB anti-repeater candidates ({BAND[0]}-{BAND[1]} Hz, common set)").show()
except Exception as e:
    print("PyGMT map skipped:", type(e).__name__, e)""")

md("""## 4 · Do they reproduce at MKL / YSB / CHS? — multi-station signed CC (same events)

Per station, `max|CC| = max(best +CC, |best −CC|)` for each pair. **No NaN now** — every pair exists
at every station. **Validation = `max|CC|` stays ≥ `VAL_OK` at the validators.** Decorrelation
(`max|CC|` ~0.2-0.4 off HDB) = the HDB negative is a single-station artifact.""")
code("""rows = []
for p in pairs:
    a, b = p["i"], p["j"]
    row = {"ev_i": common[a], "ev_j": common[b]}
    for st in ALL_STATIONS:
        cp, cn = SIG[st]["cc_pos"], SIG[st]["cc_neg"]
        row[f"{st.split('.')[1]}"] = round(float(max(cp[a, b], -cn[a, b])), 2)
    rows.append(row)
tbl = pd.DataFrame(rows)
vcols = [s.split('.')[1] for s in VAL_STATIONS]
tbl["val_median"] = tbl[vcols].median(axis=1)
tbl["validated"] = tbl[vcols].min(axis=1) >= VAL_OK
print(f"max|CC| per station (common events).  validated = min over validators >= {VAL_OK}: "
      f"{int(tbl['validated'].sum())} / {len(tbl)} pairs")
tbl""")

md("""## 5 · Waveform overlays across stations — the visual test

For the strongest candidate pairs: at **each** station, event *i* (black) vs event *j* at its best
**+CC** lag (blue, repeater fit) and best **−CC** lag flipped (red, anti fit). All on the common
events, so every row has data. If neither curve overlays *i* off HDB, the pair isn't similar there.""")
code("""for p in pairs[:N_OVERLAY]:
    wf.plot_antipair_stations(RES, common[p["i"]], common[p["j"]], stations=ALL_STATIONS, band=BAND,
                              win=WIN, maxlag=MAXLAG, zoom=(-0.3, 4.0))
    plt.show()""")

md("""## 6 · How to read / verdict

- **Validated anti-repeater:** `max|CC|` high at HDB **and** all validators (§4 `validated=True`), and
  §5 shows one consistent fit (usually the red anti-fit) overlaying at every station.
- **HDB-only artifact:** `max|CC|` collapses to ~0.2-0.4 at the validators; §5 shows the waveforms
  disagreeing off HDB. The HDB `cc_neg ≈ −0.7` is the half-cycle image of two only-moderately-similar
  (`cc_pos ≈ 0.5`) waveforms — the degeneracy, not a polarity reversal.
- Now on a **common event set**, so the comparison is the *same* events at all four stations (the
  earlier per-station population/coverage mismatch is removed). Each station still aligns on its **own**
  P pick (fallback + xcorr if unpicked); the ±`MAXLAG` search + the §5 overlays guard against jitter.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/HypoInv/09_antirepeater_multistation_1-25Hz_phasenet_plus.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(C), "cells")
