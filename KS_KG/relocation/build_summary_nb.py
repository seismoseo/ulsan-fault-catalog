#!/usr/bin/env python
"""Generate summary_reuse.ipynb — a dedicated summary of the **reuse** family-738 relocation
(f738_reuse, kim2011), styled after PocketQuake's final results notebook *section 1 (locations)*,
**skipping the focal-mechanism / first-motion parts**. Reuses PocketQuake's `pipeline.viz` verbatim,
and adds the PyGMT before/after subregion map (`pygmt_reloc_map.py`).

Usage: python build_summary_nb.py   (writes summary_reuse.ipynb next to this file)
"""
import os
import nbformat as nbf

PQ = "/home/msseo/works/15.PocketQuake"
PIPE = os.path.join(PQ, "external", "korea-cluster-relocation")
HERE = os.path.dirname(os.path.abspath(__file__))

nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md("""# Family 738 — relocation summary (reuse picks, PocketQuake HypoInverse + HypoDD, kim2011)

The largest Ulsan waveform-similarity multiplet (35 events, 2016-11-17 → 2017-03-11), relocated with
the **reuse** strategy (Ulsan's existing PhaseNet+ picks → Fortran HypoInverse + dt.ct + dt.cc + HypoDD
at **kim2011**) — the strategy chosen as more robust (see `compare_relocations.ipynb`).

Styled after PocketQuake's results notebook *§1 Locations*; **focal-mechanism / first-motion sections
are intentionally omitted**. Everything is reproduced by `run.sh`.""")

co(f"""import os, sys
sys.path.insert(0, "{PQ}"); sys.path.insert(0, "{PIPE}")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display
from pipeline import config, viz

CLUSTER, VELMODEL = "f738_reuse", "kim2011"
cfg = config.load_cluster(CLUSTER)
# bootstrap drop thresholds (same defaults as the PocketQuake results notebook)
viz.BOOT_DROP_HORIZ_KM = 0.1
viz.BOOT_DROP_VERT_KM  = 0.1
print("cluster:", CLUSTER, "| velmodel:", VELMODEL)""")

md("""## 1. Locations

### Relocation counts — how many events survive each stage""")
co("""display(viz.relocation_counts(cfg, VELMODEL))""")

md("""### Final relocation table — locations + bootstrap 95% errors

`viz.location_table` is the headline deliverable: one row per event with the dt.cc location and the
bootstrap 95 % half-widths (ex/ey/ez). It also writes `<dt.cc dir>/final_locations.csv`.""")
co("""loc = viz.location_table(cfg)
print(f"{len(loc)} events; columns: {list(loc.columns)}")
display(loc.head(10))""")

md("""### Location uncertainty — bootstrap 95 %

Median / mean / max of the per-event 95 % half-widths. The well-constrained core is a few metres; the
tail is set by N-S / depth (azimuthal geometry). Events above the drop thresholds
(`viz._boot_underconstrained`) are flagged and excluded from the filtered dt.cc views below.""")
co("""b = os.path.join(config.dtcc_dir(cfg), "bootstrap_errors.csv")
bb = pd.read_csv(b, comment="#"); bb["horiz"] = np.hypot(bb.ex95, bb.ey95)
print(f"horizontal 95% half-width (m): median {bb.horiz.median():.0f}, mean {bb.horiz.mean():.0f}, max {bb.horiz.max():.0f}")
print(f"vertical   95% half-width (m): median {bb.ez95.median():.0f}, mean {bb.ez95.mean():.0f}, max {bb.ez95.max():.0f}")
drop = viz._boot_underconstrained(cfg, "dtcc")
print(f"under-constrained (dropped from filtered views): {len(drop)} / {len(bb)}")""")

md("""### Absolute catalog (before relocation) — map, depth sections, cumulative count""")
co("""viz.map_catalog(cfg, velmodel=VELMODEL, source="sum"); plt.show()
viz.depth_sections(cfg, velmodel=VELMODEL, source="sum"); plt.show()
viz.cumulative_events(cfg, velmodel=VELMODEL); plt.show()""")

md("""### dt.cc relocation — map + depth sections (bootstrap-filtered), and dt.ct vs dt.cc""")
co("""viz.map_catalog(cfg, velmodel=VELMODEL, source="reloc"); plt.show()
viz.depth_sections(cfg, velmodel=VELMODEL, source="reloc"); plt.show()
try:
    viz.compare_epicenters(cfg, velmodel=VELMODEL); plt.show()      # dt.ct vs dt.cc
except Exception as e:
    print("compare_epicenters skipped:", type(e).__name__, e)""")

md("""### Fault-frame sections (SVD best-fit plane) + summary view (all events)

`fault_sections(frame_from="svd")` fits the fault plane to the relocated cloud; the all-events
`map_catalog(include_all=True)` overlays the dt.cc-/bootstrap-dropped events at their absolute
positions (hollow), so nothing is silently hidden.""")
co("""viz.fault_sections(cfg, velmodel=VELMODEL, frame_from="svd", color_by="time", show_bootstrap=True); plt.show()
viz.map_catalog(cfg, velmodel=VELMODEL, source="reloc", include_all=True, show_errors=False); plt.show()""")

md("""### HypoDD link map — inter-event differential-time connectivity""")
co("""viz.link_maps(cfg, velmodel=VELMODEL); plt.show()""")

md("""## 2. PyGMT subregion map — exact locations before vs after relocation

Per the project convention, the spatial map uses **PyGMT** (`pygmt_reloc_map.py`): two panels on a
shared, square extent + shared depth colour scale — (a) absolute HypoInverse(kim2011), (b) dt.cc
HypoDD — circles coloured by depth, sized by KMA local magnitude. The scattered ~1 km cloud (with a
SW-shallow → NE-deep gradient, plus one southern absolute-location outlier) collapses to a compact
~200 m patch.""")
co("""import pygmt_reloc_map
png = pygmt_reloc_map.make_map(CLUSTER)
display(Image(filename=png))""")

md("""## 3. Reading this

- The multiplet **collapses** from the absolute HypoInverse scatter (≈530 m horizontal RMS, depth
  9.4–11.1 km) to a compact dt.cc patch (≈94 m horizontal, depth ≈10.5 km) — repeaters are co-located.
- The bootstrap 95 % relative half-widths are **a few metres** in the well-constrained core; only the
  N-S / depth tail is larger (station geometry, not the waveforms).
- The reuse strategy was chosen for its lighter under-constrained tail (1 vs 7) — see
  `compare_relocations.ipynb` for the reuse-vs-fresh comparison and the kept-vs-dropped sections.
- Reproduced end-to-end by `run.sh`; the tidy tables are in `family738/reloc_f738_reuse.csv` and the
  pipeline's `final_locations.csv`.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = os.path.join(HERE, "summary_reuse.ipynb")
nbf.write(nb, out); print("wrote", out, len(C), "cells")
