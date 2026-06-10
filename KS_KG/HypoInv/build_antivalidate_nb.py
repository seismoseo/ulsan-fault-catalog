"""Build the multi-station anti-repeater VALIDATION notebook (KG.HDB candidates checked at nearby
long-term stations). Usage: python build_antivalidate_nb.py

Premise: a genuine anti-repeater (two co-located events with opposite first motion — mechanisms
straddling a nodal plane) must correlate STRONGLY (|CC|->1) at EVERY station, with the winning sign
set by each station's take-off geometry. A single-station (HDB-only) negative correlation that
DECORRELATES at other stations is an artifact (half-cycle-lag degeneracy on only-moderately-similar
waveforms), not a polarity reversal. We test the HDB 1-25 Hz candidates at KG.MKL / KG.YSB / KG.CHS,
the three other ~long-term, ~full-coverage stations in event_waveforms_ulsanfault.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md("""# Anti-repeater candidates at `KG.HDB` (1-25 Hz) — multi-station validation

The 1-25 Hz signed-CC screen flags a small, spatially-compact set of HDB pairs with a strong
**negative** correlation (`cc_neg ≤ NEG_LOOSE`) that are **not** also positively correlated
(`cc_pos < POS_GATE`) — apparent **anti-repeaters**. Before believing them we validate against the
three other long-term, ~full-coverage stations in the event archive — **KG.MKL, KG.YSB, KG.CHS**
(coverage ≈ 2769 / 2751 / 2754 of ~2770, like HDB's 2770).

**The test.** A real anti-repeater is *co-located* with an *opposite* mechanism, so at **every**
station the two waveforms are identical up to **sign**: one of {repeater-fit, anti-fit} should be
near |CC| = 1, with the winning sign varying by station per its take-off angle. If instead the strong
negative is **HDB-only** and the pair **decorrelates** (both fits weak) at MKL/YSB/CHS, the HDB
`cc_neg` is the half-cycle-lag artifact of two only-moderately-similar waveforms — *not* a polarity
reversal. (Recall `cc_pos` at HDB for these candidates is only ~0.5, already un-repeater-like.)""")

md("## Parameters")
code("""STATION   = "KG.HDB"           # primary (screening) station
VAL_STATIONS = ["KG.MKL", "KG.YSB", "KG.CHS"]   # nearby long-term validators
COMP      = "HHZ"              # vertical (rotation-immune)
WIN       = (-0.5, 7.5)
BANDS     = [(1, 25), (1, 10), (2, 8), (4, 12), (5, 15)]
BAND      = (1, 25)            # stricter band (compact candidate set)
MAXLAG    = 0.2
NEG_LOOSE = -0.7              # looser than the -0.85 headline cut, to surface the candidates you saw
POS_GATE  = 0.6              # genuine anti must NOT also be positively correlated
N_OVERLAY = 8                # candidate pairs to draw as multi-station overlays
CACHE     = "wf_similarity_cache"
ALL_STATIONS = [STATION] + VAL_STATIONS""")

code("""import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import uf_waveform_similarity as wf
import uf_cluster as ufc
wf.use_helvetica()
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)""")

md("""## 1 · HDB anti-repeater candidates (1-25 Hz)""")
code("""resH = wf.make_bands(wf.list_events(station=STATION, comp=COMP), station=STATION, comp=COMP,
                     bands=BANDS, win=WIN, cache_dir=CACHE, verbose=False)
keptH = resH["kept"]; meta = wf.load_event_meta(keptH)
Sg = wf.signed_similarity(resH["bands"][BAND], maxlag=MAXLAG)
neg, pos = Sg["cc_neg"], Sg["cc_pos"]
iu = np.triu_indices(len(keptH), k=1)
mask = (neg[iu] <= NEG_LOOSE) & (pos[iu] < POS_GATE)
pairs = [dict(i=int(a), j=int(b), cc_neg=round(float(neg[a, b]), 3), cc_pos=round(float(pos[a, b]), 3))
         for a, b in zip(iu[0][mask], iu[1][mask])]
pairs.sort(key=lambda p: p["cc_neg"])
print(f"{len(pairs)} HDB candidate pairs (cc_neg<={NEG_LOOSE} & cc_pos<{POS_GATE}); "
      f"{len(set([p['i'] for p in pairs]) | set([p['j'] for p in pairs]))} events involved")
pd.DataFrame([dict(ev_i=keptH[p["i"]], ev_j=keptH[p["j"]], **{k: p[k] for k in ("cc_neg","cc_pos")})
              for p in pairs])""")

md("""## 2 · Where are they? — spatial clustering (PyGMT)

Each candidate pair drawn as two epicentres joined by a line coloured by `cc_neg`, over the UF
subregion. If the candidates cluster in one narrow spot, that is the population to validate.""")
code("""try:
    wf.map_antipairs(meta, pairs, value="cc_neg", station=STATION,
                     title=f"HDB anti-repeater candidates ({BAND[0]}-{BAND[1]} Hz)").show()
except Exception as e:
    print("PyGMT map skipped:", type(e).__name__, e)""")

md("""## 3 · Do they reproduce at MKL / YSB / CHS? — multi-station signed CC

For each pair we report, per station, the best repeater fit (`+CC`) and anti fit (`−CC`), and
`max|CC| = max(+CC, |−CC|)`. **Validation = `max|CC|` stays high (≳0.7) at the other stations** (the
pair really is the same waveform up to sign everywhere). Decorrelation (`max|CC|` ~0.2-0.4 off HDB)
= the HDB negative is a single-station artifact.""")
code("""res_by_station = {STATION: resH}
for st in VAL_STATIONS:
    res_by_station[st] = wf.make_bands(wf.list_events(station=st, comp=COMP), station=st, comp=COMP,
                                       bands=BANDS, win=WIN, cache_dir=CACHE, verbose=False)
SIG = {}
for st, r in res_by_station.items():
    S = wf.signed_similarity(r["bands"][BAND], maxlag=MAXLAG)
    SIG[st] = (S["cc_pos"], S["cc_neg"], {e: k for k, e in enumerate(r["kept"])})

rows = []
for p in pairs:
    ei, ej = keptH[p["i"]], keptH[p["j"]]
    row = {"ev_i": ei, "ev_j": ej}
    for st in ALL_STATIONS:
        cp, cn, idx = SIG[st]
        if ei in idx and ej in idx:
            a, b = idx[ei], idx[ej]
            row[f"{st.split('.')[1]}_maxabs"] = round(float(max(cp[a, b], -cn[a, b])), 2)
        else:
            row[f"{st.split('.')[1]}_maxabs"] = np.nan
    rows.append(row)
tbl = pd.DataFrame(rows)
val_cols = [f"{s.split('.')[1]}_maxabs" for s in VAL_STATIONS]
tbl["val_median_maxabs"] = tbl[val_cols].median(axis=1)
print("max|CC| per station (HDB is the screen; validators should also be high if genuine):")
tbl""")

md("""## 4 · Waveform overlays across stations — the visual test

For the strongest candidate pairs: at **each** station, event *i* (black) vs event *j* aligned at its
best **+CC** lag (blue, repeater fit) and at its best **−CC** lag then flipped (red, anti fit). Read
each row: if one curve overlays *i* tightly the pair is real *there*; if **neither** does (both curves
disagree with black), the pair simply isn't similar at that station.""")
code("""TOP = pairs[:N_OVERLAY]
for p in TOP:
    ei, ej = keptH[p["i"]], keptH[p["j"]]
    wf.plot_antipair_stations(res_by_station, ei, ej, stations=ALL_STATIONS, band=BAND,
                              win=WIN, maxlag=MAXLAG, zoom=(-0.3, 4.0))
    plt.show()""")

md("""## 5 · How to read / verdict

- **Validated anti-repeater:** `max|CC|` high at HDB **and** MKL/YSB/CHS (§3), and in §4 a single
  consistent fit (mostly the red anti-fit) overlays at every station. Co-located events with an
  opposite mechanism.
- **HDB-only artifact (expected here):** `max|CC|` collapses to ~0.2-0.4 at the validators while only
  HDB is high; §4 shows the waveforms disagreeing off HDB. The HDB `cc_neg ≈ −0.7` is then the
  most-negative lag of two only-moderately-similar (`cc_pos ≈ 0.5`) waveforms — the half-cycle
  degeneracy, not a polarity reversal.
- Caveat: each station aligns on its **own** P pick (fallback + xcorr if unpicked); the ±`MAXLAG`
  search absorbs small jitter, and §4 lets you judge the overlays directly rather than trust one CC.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/HypoInv/09_antirepeater_multistation_1-25Hz_phasenet_plus.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(C), "cells")
