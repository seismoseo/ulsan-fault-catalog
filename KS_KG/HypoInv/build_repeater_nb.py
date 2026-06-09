"""Build the traditional repeating-earthquake notebook for KG.HDB, one per component.
Usage: python build_repeater_nb.py [HHZ|HHN|HHE]   (default HHZ).

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

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md(f"""# Repeating earthquakes at `KG.HDB` ({NB_COMP}) — waveform-similarity families

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
BANDS      = [(1, 10), (2, 8), (4, 12), (5, 15)]   # Hz
PRIMARY    = (1, 10)            # band for families / table / map
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
`rayleigh_p` (tectonic vs residual blast). Sorted by repeat count.""")
code("""rep = wf.repeater_table(meta, labels, CC, min_size=MIN_FAMILY)
print(f"{len(rep)} families; {int(rep['n'].sum())} events; "
      f"largest family n={int(rep['n'].max()) if len(rep) else 0}")
rep.head(30)""")

md("""## 5 · Family waveform gathers

Overlaid aligned traces (grey) + the family stack (bold) for the top families — near-identical
repeating waveforms are visually obvious.""")
code("""evid = rep.rename(columns={"n": "n", "mean_cc": "mean_cc"})  # plot_cluster_gathers reads cluster/n/mean_cc
wf.plot_cluster_gathers(res["bands"][PRIMARY], labels, evid, win=WIN, max_clusters=8, max_traces=40);""")

md("""## 6 · Recurrence timeline + interval distribution

Top families as time-lanes (a marker at each member's origin time), and the pooled distribution of
inter-event (recurrence) intervals. Magnitude-free.""")
code("""wf.plot_repeater_sequences(meta, labels, rep, top=15);""")

md("""## 7 · Map of repeating-earthquake families

Joined events coloured by family (PyGMT). Repeaters should be spatially compact; known quarry
centroids (red ✗) flag residual-blast families.""")
code("""try:
    fig_map = wf.map_clusters(meta, labels, rep, station=STATION, title=f"KG.HDB {COMP} repeater families")
    fig_map.show()
except Exception as e:
    print("PyGMT map skipped:", type(e).__name__, e)""")

md(f"""## 8 · How to read this

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
out = f"/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/HypoInv/07_repeaters_KGHDB_{NB_COMP}_phasenet_plus.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(C), "cells")
