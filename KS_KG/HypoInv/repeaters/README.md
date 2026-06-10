# `repeaters/` — repeater & anti-repeater waveform-similarity analyses

KG.HDB-station (and multi-station) **repeating-earthquake** and **anti-repeater** analyses, split out
from the flat `HypoInv/` root on 2026-06-10. The shared modules (`uf_waveform_similarity.py`,
`uf_cluster.py`), the feature/CC cache (`wf_similarity_cache/`), the catalogs and `STA/` **stay at the
HypoInv root** — they are imported/read by non-repeater notebooks (e.g. the `04_waveform_similarity`
blast screen) too.

## Runs from anywhere
Unlike the old root notebooks, these are **cwd-independent**: each notebook's setup cell does
`sys.path.insert(0, "<HypoInv>")` (so the bare `import uf_waveform_similarity` resolves) and uses the
**absolute** `CACHE = wf.CACHE_DIR`. So you can `cd repeaters/ && jupyter nbconvert --execute …` and it
hits the existing cache at the root — no feature rebuild, no stray cache dir.

## Builders (tracked) and what they generate
| Builder | Generates | Notes |
|---|---|---|
| `build_repeater_nb.py [COMP] [BAND] [CC] [LINKAGE]` | `07_repeaters_KGHDB_<COMP>[_<band>][_<linkage>]_phasenet_plus.ipynb` | Classic repeating-earthquake families on positive max-lag CC; UPGMA default. |
| `build_antirepeater_nb.py [COMP] [BAND]` | `06_anti_repeaters_KGHDB_<COMP>[_<band>]_phasenet_plus.ipynb` | Signed-CC search for polarity-reversed pairs. **Null result** (no anti-repeaters). |
| `build_antivalidate_nb.py` | `09_antirepeater_multistation_1-25Hz_phasenet_plus.ipynb` | Multi-station check of the 06 candidates, fair across HH/HG/EL on native channels + a repeater control. |
| `build_multistation_nb.py [COMP]` | `10_multistation_repeaters_KGHDB_<COMP>_phasenet_plus.ipynb` | **Network confirmation** of HDB repeater families (5–15 Hz, CC≥0.9) across nearby stations, adaptive to the time-varying network. |

Generated `.ipynb` are gitignored; regenerate with the builders (run from anywhere).

## Key finding carried over
The anti-repeater investigation (06/09) showed a strong **single-station (HDB)** waveform signal can be
an artifact: a genuine co-located repeater reproduces at the nearby stations (HG/HH control: `cc_pos`
0.88–0.99), while the candidate "anti" pairs collapsed to 0.1–0.3 off HDB. That motivates the
multi-station **network confirmation** in `build_multistation_nb.py` (notebook 10).
