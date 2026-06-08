"""Build the per-cluster space-time notebook (nbformat), ready to nbconvert.
Usage: python build_seq_nb.py [HHZ|HHN|HHE]   (default HHZ). Component drives PARAMS + filename.

One composite per waveform family: chronological gather (left) + a FIXED-extent epicentre map
coloured by origin year (right top) + a cumulative-count-vs-year curve (right bottom). Reuses the
clustering of 04_waveform_similarity_hdb_<COMP>; warm caches make it fast."""
import sys
import nbformat as nbf

NB_COMP = sys.argv[1] if len(sys.argv) > 1 else "HHZ"
assert NB_COMP in ("HHZ", "HHN", "HHE"), NB_COMP

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md(f"""# Per-cluster space-time sequence — Ulsan Fault waveform families (**{NB_COMP}**)

Companion to `04_waveform_similarity_hdb_{NB_COMP}_phasenet_plus.ipynb`. For **every** waveform
family (≥ `MIN_SIZE`) at `KG.HDB`, one **composite** figure shows:

- **left** — the family's **chronological waveform gather** (every member, oldest at top, constant
  per-trace height, UTC origin times on the right);
- **right top** — a **fixed-extent** epicentre map (same region for every family) with the family's
  events coloured by **origin year**, all other events faint grey, faults + station + quarry ✗;
- **right bottom** — the **cumulative event count vs year** step curve.

Together they read the **time-cumulative sequence + location** of each family: a **compact spatial
pocket filling up across years** is the still-remaining quarry-blast signature; a spatially spread or
single-burst family is tectonic. Exploratory — no events removed.
""")

md("## Parameters")
code(f"""STATION    = "KG.HDB"
COMP       = "{NB_COMP}"
WIN        = (-0.5, 7.5)
BANDS      = [(1, 10), (2, 8), (4, 12), (5, 15)]
PRIMARY    = (1, 10)
MAXLAG     = 0.2
CC_THRESHOLD = 0.6
LINKAGE      = "average"
YEARS      = None            # FULL catalog
MIN_SIZE   = 4               # families with >= MIN_SIZE members get a composite
CACHE      = "wf_similarity_cache"
""")

code("""import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import uf_waveform_similarity as wf
import uf_cluster as ufc
wf.use_helvetica()
plt.rcParams["figure.max_open_warning"] = 0
os.makedirs(CACHE, exist_ok=True)
print("station", STATION, COMP, "| window", WIN, "s")""")

md("## 1 · Load, cluster (same pipeline as notebook 04)")
code("""events = wf.list_events(station=STATION, comp=COMP)
if YEARS is not None:
    yrs = {str(y) for y in YEARS}; events = [e for e in events if e[:4] in yrs]
res  = wf.make_bands(events, station=STATION, comp=COMP, bands=BANDS, win=WIN, cache_dir=CACHE)
kept, info = res["kept"], res["info"]
meta = wf.load_event_meta(kept)

def band_cc(band):
    tag = f"{STATION}_{COMP}_w{WIN[0]}_{WIN[1]}_b{band[0]}-{band[1]}_lag{MAXLAG}_n{len(kept)}".replace(".", "p")
    f = os.path.join(CACHE, f"cc_{tag}.npy")
    return np.load(f) if os.path.exists(f) else np.save(f, wf.similarity_matrix(res["bands"][band], maxlag=MAXLAG)) or np.load(f)

cc = band_cc(PRIMARY)
labels, Z, order = wf.ward_clusters(cc, threshold=1 - CC_THRESHOLD, method=LINKAGE)
fams = sorted([int(c) for c in np.unique(labels) if (labels == c).sum() >= MIN_SIZE],
              key=lambda c: -(labels == c).sum())
COLORS = wf.cluster_colors(fams)
Xhp = wf.display_matrix(res, band=("highpass", 1.0), station=STATION, comp=COMP, win=WIN)
reg = wf.spacetime_region(meta)
print(f"{len(kept)} events | {len(fams)} families >= {MIN_SIZE} | joined {meta['joined'].sum()} | "
      f"fixed map region {[round(x,3) for x in reg]}")""")

md("""## 2 · Space-time composite per family

One figure per family (size order). The map extent is **fixed** (`wf.spacetime_region`, the bounding
box of all joined events) so every family is spatially comparable; events are coloured by **origin
year** (shared 2010–2025 scale) and the cumulative curve shares that axis. **1 Hz highpass** traces
(same as notebook 04's gathers).""")
code("""wf.plot_clusters_spacetime(Xhp, labels, kept, meta, reg=reg, win=WIN, station=STATION,
                           comp=COMP, min_show=MIN_SIZE, colors=COLORS, order_by="size");""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = f"/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/HypoInv/05_cluster_spacetime_{NB_COMP}_phasenet_plus.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(C), "cells")
