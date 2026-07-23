"""Build the anti-repeater (signed cross-correlation) notebook for KG.HDB, one per component.
Usage: python build_antirepeater_nb.py [HHZ|HHN|HHE]   (default HHZ).

Question: do "anti-repeating" pairs exist — events whose KG.HDB waveforms are near-perfect
POLARITY-REVERSED copies (cross-correlation ~ -1.0)? Reuses the P-aligned, L2-normalised band
matrices from uf_waveform_similarity.make_bands and the new signed_similarity / plot_antipair_gathers
/ map_antipairs. The central subtlety it is built to expose: for oscillatory regional waveforms a
half-period lag (< maxlag) turns a positive correlation into a negative one, so a negative CC is only
meaningful READ WITH the positive CC and confirmed across the three components.
"""
import sys
import nbformat as nbf

NB_COMP = sys.argv[1] if len(sys.argv) > 1 else "HHZ"
assert NB_COMP in ("HHZ", "HHN", "HHE"), NB_COMP

# optional analysis band "LO-HI" Hz (default 1-10). A stricter high-frequency band (e.g. 1-25)
# makes the half-cycle degeneracy WORSE (shorter periods → a half-period is an even smaller lag),
# so it is a harder test for genuine polarity reversals.
BAND_ARG = sys.argv[2] if len(sys.argv) > 2 else "1-10"
PRIMARY_BAND = tuple(int(x) for x in BAND_ARG.split("-"))
assert len(PRIMARY_BAND) == 2 and PRIMARY_BAND[0] < PRIMARY_BAND[1], BAND_ARG
IS_DEFAULT_BAND = PRIMARY_BAND == (1, 10)
BAND_TAG = f"{PRIMARY_BAND[0]}-{PRIMARY_BAND[1]}Hz"
BANDS_LIST = list(dict.fromkeys([PRIMARY_BAND, (1, 10), (2, 8), (4, 12), (5, 15)]))

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md(f"""# Anti-repeating events at `KG.HDB` ({NB_COMP}, {BAND_TAG}) — signed cross-correlation

**Question.** Do *anti-repeaters* exist — event pairs whose `KG.HDB.{NB_COMP}` waveforms are
near-perfect **polarity-reversed** copies (cross-correlation ≈ **−1.0**)? Physically that would mean
two events with otherwise identical source→path but **opposite first motion** at this station (e.g.
mechanisms straddling a nodal plane), i.e. the negative-CC twin of a repeater.

**Method.** Reuse the P-aligned, L2-normalised band matrices from the blast-screen pipeline
(`uf_waveform_similarity.make_bands`) and the **signed** correlation `wf.signed_similarity`, which keeps
four N×N matrices per band:

| matrix | meaning |
|---|---|
| `cc_pos` | max over lags (= the usual `similarity_matrix`; the repeater signal) |
| `cc_neg` | **min** over lags (the most-negative correlation) |
| `cc_ext` | signed extreme (`cc_neg` where `|cc_neg|>|cc_pos|` else `cc_pos`) |
| `cc_lag0` | signed CC at **lag 0** — correlation at the exact P datum, no lag freedom |

**The crux (built into every figure below).** For ~2–10 Hz regional waveforms a shift of half a period
(≈ 0.05–0.25 s) is **inside** the ±`MAXLAG` search and flips the sign. So `cc_neg ≈ −1` can simply be a
repeater offset by a half-cycle — *not* a polarity reversal. A genuine anti-repeater must therefore have
**`cc_neg ≤ NEG_THRESHOLD` AND `cc_pos` NOT also high** (you cannot make anti-phase signals positively
correlate within ±`MAXLAG`), and ideally flip **consistently on all three components**. This notebook
applies exactly that test.
""")

md("## Parameters")
code(f"""# --- parameters (edit + run top-to-bottom) ---------------------------------------------
STATION   = "KG.HDB"
COMP      = "{NB_COMP}"          # primary component for this notebook
COMPS     = ("HHZ", "HHN", "HHE")   # all three, for the cross-component confirmation
WIN       = (-0.5, 7.5)         # s relative to P (same short window as the blast screen)
BANDS     = {BANDS_LIST}   # Hz
PRIMARY   = {PRIMARY_BAND}            # analysis band (P alignment still uses REF_BAND 2-8 Hz internally)
MAXLAG    = 0.2                 # s, signed-CC lag search
NEG_THRESHOLD = -0.85          # a candidate anti-pair needs cc_neg / cc_lag0 at least this negative
POS_GATE      = 0.6            # ...and cc_pos BELOW this (else it is a half-cycle-offset repeater)
TOPN      = 12                 # candidates to gallery / map
CACHE     = "wf_similarity_cache"
""")

code("""import os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from uflib import uf_waveform_similarity as wf
from uflib import uf_cluster as ufc
wf.use_helvetica()
CACHE = wf.CACHE_DIR          # absolute cache dir (overrides the relative default; cwd-independent)
os.makedirs(CACHE, exist_ok=True)
pd.set_option("display.width", 180); pd.set_option("display.max_columns", 40)
print("station", STATION, COMP, "| window", WIN, "s | primary band", PRIMARY)""")

md("""## 1 · Load the P-aligned band matrices (all three components)

We need all three components: the primary `COMP` drives the ranking, the other two confirm whether a
sign flip is a real source-polarity reversal (consistent across Z/N/E) or a single-channel/half-cycle
artefact. `make_bands` reuses the on-disk `feat_*.npz` caches, so this is fast.""")
code("""def load_comp(comp):
    evs = wf.list_events(station=STATION, comp=comp)
    res = wf.make_bands(evs, station=STATION, comp=comp, bands=BANDS, win=WIN, cache_dir=CACHE, verbose=False)
    return res

RES  = {c: load_comp(c) for c in COMPS}
res  = RES[COMP]; kept = res["kept"]; meta = wf.load_event_meta(kept)
IDX  = {c: {e: i for i, e in enumerate(RES[c]["kept"])} for c in COMPS}
print({c: len(RES[c]["kept"]) for c in COMPS}, "events per component")
print("joined to catalog:", int(meta["joined"].sum()), "/", len(meta))""")

md("""## 2 · Signed cross-correlation per band (primary component)

`cc_pos` is provably identical to the blast-screen `similarity_matrix`; we additionally keep `cc_neg`,
`cc_ext`, `cc_lag0`. Cached as `signed_<tag>.npz`.""")
code("""def signed_cc(comp, band):
    tag = f"{STATION}_{comp}_w{WIN[0]}_{WIN[1]}_b{band[0]}-{band[1]}_lag{MAXLAG}_n{len(RES[comp]['kept'])}".replace(".", "p")
    f = os.path.join(CACHE, f"signed_{tag}.npz")
    if os.path.exists(f):
        z = np.load(f); return {k: z[k] for k in z.files}
    S = wf.signed_similarity(RES[comp]["bands"][band], maxlag=MAXLAG)
    np.savez_compressed(f, **S); return S

S = {b: signed_cc(COMP, b) for b in BANDS}
SP = S[PRIMARY]
print("signed CC ready for", COMP, "bands", BANDS)""")

md("""## 3 · Does an anti-correlated population exist? — distribution + the degeneracy

Left: distributions of `cc_pos` (repeater side) and `cc_neg` (anti side) over all pairs. Right: for the
strongly-negative pairs, **`cc_neg` vs `cc_pos`** — the decisive plot. If every point with `cc_neg ≤ −0.85`
*also* sits at high `cc_pos` (top-left), the negatives are **half-cycle-offset repeaters**, not reversals;
genuine anti-repeaters would fall in the **low-`cc_pos`** band (below `POS_GATE`).""")
code("""iu = np.triu_indices(len(kept), k=1)
neg, pos, lag0 = SP["cc_neg"][iu], SP["cc_pos"][iu], SP["cc_lag0"][iu]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), dpi=130)
ax[0].hist(pos, bins=80, color="steelblue", alpha=0.8, label="cc_pos (repeater)")
ax[0].hist(neg, bins=80, color="indianred", alpha=0.8, label="cc_neg (anti)")
ax[0].axvline(NEG_THRESHOLD, color="k", ls="--", lw=0.8)
ax[0].set(xlabel="cross-correlation", ylabel="pairs", title=f"{COMP} {PRIMARY} Hz — signed CC distribution")
ax[0].set_yscale("log"); ax[0].legend()

m = neg <= -0.70
sc = ax[1].scatter(neg[m], pos[m], s=6, c=np.abs(lag0[m]), cmap="viridis", vmin=0, vmax=1)
ax[1].axhline(POS_GATE, color="crimson", ls="--", lw=1.0)
ax[1].axvline(NEG_THRESHOLD, color="k", ls="--", lw=0.8)
ax[1].annotate("genuine anti-repeater zone\\n(cc_neg low, cc_pos low)", (-0.97, 0.30), fontsize=8, color="crimson")
ax[1].set(xlabel="cc_neg (most-negative over lags)", ylabel="cc_pos (most-positive over lags)",
          title="The half-cycle degeneracy")
fig.colorbar(sc, ax=ax[1], label="|cc_lag0|", shrink=0.85); fig.tight_layout()

print(f"pairs cc_neg<=-0.85: {(neg<=-0.85).sum()}  |  ALSO cc_pos<{POS_GATE}: {((neg<=-0.85)&(pos<POS_GATE)).sum()}"
      f"  |  cc_lag0<=-0.85: {(lag0<=-0.85).sum()}")""")

md("""## 4 · Ranked candidate anti-pairs + cross-component check

Rank by **`cc_lag0`** (anti-correlation at the exact P datum — the physically meaningful quantity once
events are P-aligned). For each candidate we list `cc_neg`/`cc_pos` (the degeneracy), the **same pair's
`cc_lag0` on all three components**, inter-event distance and time separation. `confirmed` requires the
flip on **all three** components (`cc_lag0 ≤ NEG_THRESHOLD`) **and** `cc_pos < POS_GATE` — the strict
definition of a true reversal.""")
code("""def pair_rows(topn):
    order = np.argsort(SP["cc_lag0"][iu])              # most-negative lag0 first
    rows, pairs = [], []
    for k in order[:topn]:
        i, j = int(iu[0][k]), int(iu[1][k])
        ea, eb = kept[i], kept[j]
        rec = dict(i=i, j=j, ev_i=ea, ev_j=eb,
                   cc_lag0=round(float(SP["cc_lag0"][i, j]), 3),
                   cc_neg=round(float(SP["cc_neg"][i, j]), 3),
                   cc_pos=round(float(SP["cc_pos"][i, j]), 3))
        # cross-component cc_lag0 / cc_pos on the SAME event-id pair
        ok_all, pos_ok_all = True, True
        for c in COMPS:
            ia, ib = IDX[c].get(ea), IDX[c].get(eb)
            if ia is None or ib is None:
                rec[f"lag0_{c[-1]}"] = np.nan; ok_all = False; continue
            Sc = signed_cc(c, PRIMARY)
            v = float(Sc["cc_lag0"][ia, ib]); rec[f"lag0_{c[-1]}"] = round(v, 3)
            ok_all &= (v <= NEG_THRESHOLD); pos_ok_all &= (float(Sc["cc_pos"][ia, ib]) < POS_GATE)
        # hypocentre context
        ri, rj = meta.iloc[i], meta.iloc[j]
        if ri["joined"] and rj["joined"]:
            dist = float(np.hypot((ri["lat"]-rj["lat"])*111.0,
                                  (ri["lon"]-rj["lon"])*111.0*np.cos(np.radians(ri["lat"]))))
            dt_days = abs((pd.to_datetime(ri["time"]) - pd.to_datetime(rj["time"])).total_seconds())/86400.0
            rec["dist_km"] = round(dist, 1); rec["dt_days"] = round(dt_days, 1)
        else:
            rec["dist_km"] = np.nan; rec["dt_days"] = np.nan
        rec["confirmed"] = bool(ok_all and pos_ok_all)
        rows.append(rec); pairs.append(rec)
    return pd.DataFrame(rows), pairs

tbl, PAIRS = pair_rows(TOPN)
print(f"CONFIRMED anti-repeaters (all 3 comps cc_lag0<= {NEG_THRESHOLD} AND cc_pos<{POS_GATE}): "
      f"{int(tbl['confirmed'].sum())} of top {TOPN}")
tbl[["ev_i","ev_j","cc_lag0","cc_neg","cc_pos","lag0_Z","lag0_N","lag0_E","dist_km","dt_days","confirmed"]]""")

md("""## 5 · Overlay gallery — are they mirror images?

Black = event *i*; **red = event *j* flipped (`−X[j]`)**; faint grey = event *j* un-flipped. If the red
curve lands on the black one, the pair is a true reversal. Watch the titles: a high `pos` next to a low
`lag0`/`neg` is the half-cycle signature (the un-flipped grey would *also* match at a shifted lag).""")
code("""wf.plot_antipair_gathers(res, PAIRS[:TOPN], band=PRIMARY, win=WIN);""")

md("""### 5b · Full-width detail — one wide row per pair

The grid above squeezes the whole window into a narrow panel; here each pair gets a **full-width row**
so individual wiggles are legible, plus a **zoom on the P→S window** (`zoom=(-0.3, 3.0)`) for the
finest detail. Black = event *i*, red = event *j* **flipped**. If the red lands exactly on the black
the pair is a true reversal; a visible ~half-cycle slip between them is the timing artefact.""")
code("""wf.plot_antipair_detail(res, PAIRS[:TOPN], band=PRIMARY, win=WIN);
wf.plot_antipair_detail(res, PAIRS[:min(8, TOPN)], band=PRIMARY, win=WIN, zoom=(-0.3, 3.0),
                        title="Anti-pair detail — zoom P→S");""")

md("""### 5c · Repeater vs anti-repeater — each aligned at its OWN best lag

For each pair, **left** slides event *j* to its best **positive**-CC lag and overlays it (blue) on
event *i* (black) — the *repeater* fit; **right** slides *j* to its best **negative**-CC lag and flips
it (red) — the *anti-repeater* fit. Because the candidates are selected for a strong negative CC **at
lag 0**, drawing the as-is overlay at lag 0 would unfairly make every pair look anti; aligning each
hypothesis at its **own maximum CC** is the honest, like-for-like test. The title reports each best
correlation + lag (ms) and flags **"repeater wins"** when the positive fit (typically at a ~half-period
lag) is at least as strong as the flipped fit. A **genuine anti-repeater is one where _anti wins_**
clearly — and on HHZ a few do.""")
code("""wf.plot_antipair_compare(res, PAIRS[:min(8, TOPN)], band=PRIMARY, win=WIN, maxlag=MAXLAG, zoom=(-0.3, 3.0));""")

md("""> **Caveat — KG.HDB horizontal sensor rotation.** The cross-component `confirmed` flag (§4) requires
> the flip on N and E as well, but **HDB's horizontal sensor orientation has changed over time**, so for
> pairs with a long inter-event time the N/E waveforms are not directly comparable until a
> **time-dependent orientation correction** is applied. The **vertical (HHZ) is immune to horizontal
> rotation** and is therefore the trustworthy basis for anti-repeater identification here — treat the
> N/E columns (and `confirmed`) as *provisional*, and combine Z/N/E only **after** the rotation
> correction. The present working criterion is a clean **as-is-mirror / flipped-match on HHZ**.""")

md("""## 6 · Where are the candidates? + distance / time separation

`map_antipairs` joins each candidate's two epicentres with a segment (short = co-located, a possible
same-patch reversal; long = coincidental). The scatter shows inter-event distance vs time gap, coloured
by `cc_lag0`.""")
code("""try:
    fig_map = wf.map_antipairs(meta, PAIRS[:TOPN], value="cc_lag0", station=STATION,
                               title=f"KG.HDB {COMP} anti-pair candidates")
    fig_map.show()
except Exception as e:
    print("PyGMT map skipped:", type(e).__name__, e)

d = tbl.dropna(subset=["dist_km", "dt_days"])
if len(d):
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=130)
    sc = ax.scatter(d["dist_km"], d["dt_days"], c=d["cc_lag0"], cmap="coolwarm_r", vmin=-1, vmax=0, s=40, edgecolor="k", lw=0.3)
    ax.set(xlabel="inter-event distance (km)", ylabel="time separation (days)",
           title="Candidate anti-pairs: distance vs time"); fig.colorbar(sc, ax=ax, label="cc_lag0")""")

md(f"""## 7 · Conclusion / how to read this

- **An anti-repeater requires `cc_neg ≤ {{NEG_THRESHOLD}}` AND `cc_pos < {{POS_GATE}}` AND a consistent
  flip on all three components.** Read §3 (right panel) and §4 (`confirmed`) together: if the negatives
  all carry a high `cc_pos`, and the flip does **not** reproduce on Z/N/E, there are **no genuine
  anti-repeaters** — the negatives are ordinary repeaters offset by ~half a period (the §5 overlays make
  this concrete: the *un-flipped* trace also matches, just at a shifted lag).
- **Why this is expected.** At a single station with ~2–10 Hz energy and a ±{{MAXLAG}} s alignment
  tolerance, a polarity reversal and a half-period time shift are **degenerate** — you cannot tell them
  apart from one channel. Only a broadband/impulsive pair (no half-period match within the lag window)
  *and* a consistent Z/N/E flip would break the degeneracy.
- **If `confirmed` is non-empty**, inspect those few pairs hard (mechanisms, picks) — they are the real
  candidates. **If empty**, the honest result is: *no anti-repeating events at KG.HDB on {NB_COMP}.*
- Cross-check the other components by running this notebook for `HHN` / `HHE` (`build_antirepeater_nb.py`).
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
_suffix = "" if IS_DEFAULT_BAND else f"_{BAND_TAG}"
out = f"/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv/repeaters/06_anti_repeaters_KGHDB_{NB_COMP}{_suffix}_phasenet_plus.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(C), "cells")
