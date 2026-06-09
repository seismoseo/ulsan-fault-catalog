"""Build the traditional repeating-earthquake notebook for KG.HDB, one per component.
Usage: python build_repeater_nb.py [HHZ|HHN|HHE] [LO-HI]   (default HHZ, 1-10 Hz).
  e.g.  python build_repeater_nb.py HHZ 1-25   # stricter high-frequency band, separate notebook.

Classic repeater analysis: events whose waveforms at a common station are near-identical
(CC >= CC_REPEAT) repeat slip on the same fault patch. We cluster on the POSITIVE max-lag CC
(reusing the cached cc_*.npy from the blast screen), then characterise each family by its
recurrence cadence, spatial compactness, and hour-of-day (tectonic vs residual quarry blast).
Magnitude is intentionally NOT used (the catalog ML is preliminary).
"""
import sys
import nbformat as nbf

NB_COMP = sys.argv[1] if len(sys.argv) > 1 else "HHZ"
assert NB_COMP in ("HHZ", "HHN", "HHE"), NB_COMP

# optional analysis band "LO-HI" Hz (default 1-10). A stricter high-frequency band (e.g. 1-25)
# demands the waveforms match into the high-frequency coda -> a tighter repeater criterion.
BAND_ARG = sys.argv[2] if len(sys.argv) > 2 else "1-10"
PRIMARY_BAND = tuple(int(x) for x in BAND_ARG.split("-"))
assert len(PRIMARY_BAND) == 2 and PRIMARY_BAND[0] < PRIMARY_BAND[1], BAND_ARG
IS_DEFAULT_BAND = PRIMARY_BAND == (1, 10)
BAND_TAG = f"{PRIMARY_BAND[0]}-{PRIMARY_BAND[1]}Hz"
# bands to build features/CC for: primary + standard reference bands (dedup, primary first)
BANDS_LIST = list(dict.fromkeys([PRIMARY_BAND, (1, 10), (2, 8), (4, 12), (5, 15)]))

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md(f"""# Repeating earthquakes at `KG.HDB` ({NB_COMP}, {BAND_TAG}) — waveform-similarity families

**Idea.** Events that rupture the **same fault patch** produce near-identical waveforms at a fixed
station. Clustering on the **positive** max-lag cross-correlation (CC ≥ `CC_REPEAT`) groups these
**repeating-earthquake families**; each is then characterised the classic way — **recurrence cadence**,
**spatial compactness**, and **hour-of-day** (a tight, daytime, hour-clustered family is a residual
quarry blast; a uniform-hour family is tectonic).

This reuses the same P-aligned band features and the cached `cc_*.npy` matrices as the blast screen
(`04_waveform_similarity_*`), but at a **higher CC threshold** and with **recurrence-interval**
diagnostics rather than a blast-removal lens. It is the positive-correlation counterpart of the
anti-repeater notebook (`06_anti_repeaters_*`).

> **Magnitude is intentionally excluded** — the catalog local magnitudes are preliminary. Repeaters
> are characterised here by *recurrence and similarity*, not size.
>
> **KG.HDB horizontal sensor rotation:** the vertical (HHZ) is immune to horizontal sensor
> re-orientation; on N/E a real repeater spanning a rotation epoch would **decorrelate and be missed**.
> So **HHZ is the robust basis**; treat HHN/HHE family counts as lower bounds until a time-dependent
> orientation correction is applied.
""")

md("## Parameters")
code(f"""# --- parameters (edit + run top-to-bottom) ---------------------------------------------
STATION    = "KG.HDB"
COMP       = "{NB_COMP}"          # HHZ vertical is the robust basis (rotation-immune)
WIN        = (-0.5, 7.5)        # s relative to P — short phase window
BANDS      = {BANDS_LIST}   # Hz
PRIMARY    = {PRIMARY_BAND}            # band for families / table / map
MAXLAG     = 0.2               # s, CC lag search
CC_REPEAT  = 0.90             # families merge while average CC >= this (repeaters are very similar)
LINKAGE    = "average"        # 'average' (UPGMA, conventional for CC distance); 'single' chains more
MIN_FAMILY = 2                # min members to call it a repeating family (a doublet counts)
CACHE      = "wf_similarity_cache"
"""
)

code("""import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import uf_waveform_similarity as wf
import uf_cluster as ufc
wf.use_helvetica()
os.makedirs(CACHE, exist_ok=True)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
print("station", STATION, COMP, "| window", WIN, "s | CC>=", CC_REPEAT, "| band", PRIMARY)""")

md("""## 1 · Load the P-aligned band features""")
code("""events = wf.list_events(station=STATION, comp=COMP)
res  = wf.make_bands(events, station=STATION, comp=COMP, bands=BANDS, win=WIN, cache_dir=CACHE, verbose=False)
kept, info = res["kept"], res["info"]
meta = wf.load_event_meta(kept)          # hypocentre + KST hour (NO magnitude — preliminary)
print(f"events with {STATION}.{COMP}: {len(kept)} | joined to catalog: {int(meta['joined'].sum())}")""")

md("""## 2 · Positive similarity matrix + repeater families

Max-lag normalised cross-correlation (cached `cc_*.npy`, shared with the blast screen), then
hierarchical clustering on `1 − CC`: families merge while their mean CC ≥ `CC_REPEAT`. Singletons (no
similar sibling) stay unclustered.""")
code("""def band_cc(band):
    tag = f"{STATION}_{COMP}_w{WIN[0]}_{WIN[1]}_b{band[0]}-{band[1]}_lag{MAXLAG}_n{len(kept)}".replace(".", "p")
    f = os.path.join(CACHE, f"cc_{tag}.npy")
    if os.path.exists(f):
        return np.load(f)
    cc = wf.similarity_matrix(res["bands"][band], maxlag=MAXLAG); np.save(f, cc); return cc

CC = band_cc(PRIMARY)
labels, Z, order = wf.ward_clusters(CC, threshold=1 - CC_REPEAT, method=LINKAGE)
nfam = sum(1 for c, n in zip(*np.unique(labels, return_counts=True)) if n >= MIN_FAMILY)
print(f"families (>= {MIN_FAMILY} members) at CC>={CC_REPEAT}: {nfam} | "
      f"events in families: {int(sum(n for c,n in zip(*np.unique(labels, return_counts=True)) if n>=MIN_FAMILY))}/{len(kept)}")""")

md("""## 3 · Clustered similarity heatmap + dendrogram

Reordered by the dendrogram so repeating families appear as bright diagonal blocks (each family
≥ `MIN_FAMILY` outlined in white).""")
code("""fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), dpi=130)
wf.plot_similarity(CC, order=order, ax=ax[0], title=f"{COMP} {PRIMARY} Hz CC (CC>={CC_REPEAT} families)")
wf.outline_clusters(ax[0], labels, order, min_size=MIN_FAMILY)
wf.plot_dendrogram(Z, color_threshold=1 - CC_REPEAT, ax=ax[1]); fig.tight_layout()""")

md("""## 4 · Repeating-earthquake family table

One row per family: repeat count `n`, intra-family `mean_cc`, centroid + `spread_km` (repeaters are
co-located), first/last time, `span_days`, **median recurrence interval**, and `daytime_frac` /
`rayleigh_p` (tectonic vs residual blast). Sorted by repeat count.

**`spread_km`** = the **median epicentral distance of the members from the family centroid** (km).
The centroid is the mean lat/lon of the *located* members; each member's distance uses the flat
111 km/deg approximation with a cos(lat) factor on longitude. So it measures how tightly the family
is co-located — a true repeating source is a few hundred metres or less; a large `spread_km` means the
waveform-similar events are *not* spatially clustered (often a hint the similarity is path/site, i.e.
a residual blast family, or that catalog location scatter dominates).

**Why some families show `NaN` for `recur_med_days`.** The recurrence interval is the median gap
between **consecutively-located** members. Only events that *joined the catalog* (`joined=True`, i.e.
the waveform event matched a blastclean hypocentre by origin time) carry a time/location; a family
needs **≥ 2 joined members** to have even one interval. So a family whose waveform cluster has ≥2
members but ≤1 of them is catalog-located gets `NaN` (you'll see `n_joined ≤ 1` on those rows). It is
a *catalog-join* gap, not a clustering failure — the waveforms still correlate; they just lack the two
located origin times needed to define an interval.""")
code("""rep = wf.repeater_table(meta, labels, CC, min_size=MIN_FAMILY)
print(f"{len(rep)} families; {int(rep['n'].sum())} events; "
      f"largest family n={int(rep['n'].max()) if len(rep) else 0}")
rep.head(30)""")

md("""## 5 · Family waveform gathers

Overlaid aligned traces (grey) + the family stack (bold) for the top families — near-identical
repeating waveforms are visually obvious.""")
code("""wf.plot_cluster_gathers(res["bands"][PRIMARY], labels, rep, win=WIN, max_clusters=8, max_traces=40);""")

md("""### 5b · Full-width family gathers (more visible, long axis)

One **full-width row per family** (largest first): member traces (grey) + the family **stack** (bold),
plus a **P→S zoom**. The long axis + tall rows make the repeating waveform shape legible (the grid
above is compact). Tune `top`, `width`, `row_h`, `zoom`.""")
code("""wf.plot_family_gathers(res, labels, rep, band=PRIMARY, win=WIN, top=8, width=15, row_h=1.9);
wf.plot_family_gathers(res, labels, rep, band=PRIMARY, win=WIN, top=8, width=15, zoom=(-0.3, 3.0),
                       title="Repeater family gathers — zoom P→S");""")

md("""### 5c · Per-cluster record sections — every event, time-ordered, S marked (ALL families)

For **every** family a separate full-width figure: all member waveforms stacked **top = oldest →
bottom = newest** (UTC on the right), **P-aligned at t=0** (blue dashed) with the **S arrival as a
short black bar** (PhaseNet+ pick). Traces are peak-normalised so the shape is clearly visible; a
genuine repeater shows the same wiggle repeating straight down the column.

**How the traces are aligned in time.** Each trace is cut on its **station P pick** (P → t=0); the few
events with no pick get `origin + median P travel-time`, then are cross-correlated onto the picked
stack (`align_fallback`). So registration is **pick-based**, *not* member-to-member cross-correlation
— honest (it shows the real pick jitter, ~a few samples) but not forced to line up. The per-pair lag
search in `similarity_matrix` absorbs that jitter when scoring CC. (A perfectly-aligned display would
additionally CC each trace to the family stack — a refinement we can add.) Set `top=N` to limit, or
`zoom=(-0.3, 3.0)` for the P→S detail.""")
code("""SP = wf.s_minus_p(kept, station=STATION)            # S-P per event (slow once; reused in 5d)
_ = wf.plot_family_sections(res, labels, rep, band=PRIMARY, win=WIN, sp=SP);""")

md("""### 5d · Same sections — 1 Hz high-pass only

The same per-family record sections but each trace is **only high-pass filtered at 1 Hz**
(`display_matrix` with the identical P-alignment) — minimal processing / broadband shape, to judge
similarity without the band-pass shaping.""")
code("""Xhp = wf.display_matrix(res, band=("highpass", 1.0), station=STATION, comp=COMP)   # 1 Hz HP, same alignment
_ = wf.plot_family_sections(res, labels, rep, win=WIN, X=Xhp, sp=SP, label="1 Hz highpass");""")

md("""## 6 · Recurrence timeline (all families)

Every family as a full-width time-lane (a marker at each member's origin time, largest family at
top). The **2016 Gyeongju mainshock** (ML 5.8, 2016-09-12 11:32 UTC) is the red dashed line, so you
can see which families activate around the sequence. For many families the per-row labels are
omitted — use §6b to inspect individuals. (The old recurrence-interval histogram is dropped: its
log-count y-axis visually exaggerated a few pairs.) Magnitude-free.""")
code("""wf.plot_repeater_sequences(meta, labels, rep, top=None);   # all families, single full-width axis""")

md("""### 6b · Every family as a separate recurrence plot

One **separate figure per family** (all of them, largest first): the members as a marker rake on
calendar time with a **cumulative-count staircase** (activity rate — bursts vs steady recurrence),
the **2016 Gyeongju mainshock** marked in red when in range, and `n` / `span` / median recurrence /
`spread_km` in each title. Set `top=N` to cap the count.""")
code("""_ = wf.plot_family_recurrence(meta, labels, rep);   # top=None -> ALL families, one figure each""")

md("""## 7 · Map of repeating-earthquake families

Joined events coloured by family (PyGMT). Repeaters should be spatially compact; known quarry
centroids (red ✗) flag residual-blast families.""")
code("""try:
    fig_map = wf.map_clusters(meta, labels, rep, station=STATION, title=f"KG.HDB {COMP} repeater families")
    fig_map.show()
except Exception as e:
    print("PyGMT map skipped:", type(e).__name__, e)""")

md("""## 8 · Enlarged UF-subregion map — families linked by lines

Close-up on the Ulsan-fault subregion with each family's events **linked by lines** to their centroid:
a **tight coloured star** = a genuinely co-located repeating family; a **long spoke** flags a member
with a location outlier (or a spurious cross-correlation between distant events) — cross-check those
against `spread_km` in the §4 table. Faint grey = events in no family; KG.HDB = yellow square; quarry
centroids = red ✗; fault traces + subregion box drawn. Pass `link="time"` for the time-ordered path.""")
code("""try:
    fig_uf = wf.map_cluster_links(meta, labels, rep, link="centroid",
                                  title=f"KG.HDB {COMP} repeater families (CC>={CC_REPEAT}) — UF subregion")
    fig_uf.show()
except Exception as e:
    print("PyGMT subregion map skipped:", type(e).__name__, e)""")

md("""### 8b · Subregion map — top 15 families only

The same UF-subregion close-up but highlighting **only the 15 largest repeating families** (by repeat
count); all other events sit behind as faint grey context. This declutters the map so the dominant
repeater sites stand out. Edit `TOP_MAP` to show more/fewer.""")
code("""TOP_MAP = 15
try:
    fig_top = wf.map_cluster_links(meta, labels, rep, top=TOP_MAP, link="centroid",
                                   title=f"KG.HDB {COMP} — top {TOP_MAP} repeater families (UF subregion)")
    fig_top.show()
except Exception as e:
    print("PyGMT top-families map skipped:", type(e).__name__, e)""")

md(f"""## 9 · How to read this

- **Repeating-earthquake family** = a tight (`mean_cc` ≥ {{CC_REPEAT}}), spatially compact
  (`spread_km` small) cluster that recurs over time. Read `recur_med_days` + the §6 timeline for the
  cadence.
- **Tectonic vs residual blast** = use `daytime_frac` / `rayleigh_p`: a family piled into daytime hours
  (small `rayleigh_p`) near a quarry centroid is a leftover blast; a uniform-hour family is tectonic.
- **Component caveat:** this is `{NB_COMP}`. **HHZ is the robust basis** (rotation-immune); on the
  horizontals a real repeater spanning a sensor-orientation change will decorrelate and be undercounted
  — fold in HHN/HHE only after a time-dependent orientation correction.
- **Magnitude is excluded** (preliminary ML). Add it only once the magnitudes are finalised.
- Tighten/loosen `CC_REPEAT` (0.90 is conservative; 0.95 = only the most identical) and compare bands.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
_suffix = "" if IS_DEFAULT_BAND else f"_{BAND_TAG}"
out = f"/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/HypoInv/07_repeaters_KGHDB_{NB_COMP}{_suffix}_phasenet_plus.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(C), "cells")
