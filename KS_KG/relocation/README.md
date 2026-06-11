# `relocation/` — precise relocation of Ulsan multiplets (PocketQuake HypoInverse + HypoDD)

Relatively-relocate the waveform-similarity multiplets (the single-linkage CC≥0.9 families from
`HypoInv/repeaters/`) with the **PocketQuake** Fortran **HypoInverse + HypoDD** pipeline at the
**kim2011** velocity model. Start with the **largest cluster, family 738** (35 events,
2016-11-17 → 2017-03-11) as the de-risking gate before scaling to the rest.

**Design — reuse, don't re-invent.** Every scientific step (HypoInverse, ph2dt, dt.ct, dt.cc
cross-correlation, HypoDD, bootstrap) is PocketQuake's own pipeline. The scripts here only **prepare a
catalog** and **stage the existing Ulsan waveforms/picks** into a PocketQuake `stp_sac` cluster (the
Ulsan SAC are already in STP naming). Two runs are compared, both on the **same** waveforms, kim2011,
Fortran HypoInverse+HypoDD — differing **only in the picks**:

| run | picks | how |
|---|---|---|
| **(1) `f738_reuse`** | Ulsan's **existing** PhaseNet+ picks | gather preserves SAC `a`/`t0`; `--stage-from hypoinverse` (skip picking) |
| **(2) `f738_fresh`** | a **fresh** PhaseNet+ re-pick | `--stage-from stations` runs the picking stage on the identical waveforms |

## Files
- `make_catalog.py` — reproduce family 738 (cached 5-25 Hz cc → single-linkage) → `family738/catalog_kma.csv`
  (KMA/KST), `members.txt`, `scaffold_args.txt` (epicenter + region bounds).
- `scaffold_offline.py` — scaffold a PocketQuake `stp_cluster` **offline** (station tables written from
  the Ulsan roster instead of the STP network fetch). Reuses PocketQuake's `write_cluster_module` +
  `register_cluster`.
- `stage.py` — symlink the Ulsan SAC into `stp_download/SAC/{eid}/{HH,HG,EL}/`; with `--reuse-picks`,
  also convert the Ulsan `{eid}_picks.csv` → PocketQuake `picks/{eid}_picks.csv`.
- `run.sh` — the full reproducible sequence (catalog → scaffold → stage → relocate (1)+(2) → compare).
- `build_compare_nb.py` → `compare_relocations.ipynb` — absolute vs (1) vs (2): spatial collapse, map +
  depth sections, fault-frame SVD sections (kept vs under-constrained-dropped), bootstrap 95 %, and the
  **(1)-vs-(2) per-event offset** (pick-source robustness).
- `build_summary_nb.py` → `summary_reuse.ipynb` — **dedicated summary of the chosen reuse run**, styled
  after PocketQuake's results notebook §1 (locations), **no focal-mechanism parts**: relocation counts,
  `location_table` (writes `final_locations.csv`), bootstrap uncertainty, absolute + dt.cc maps/sections,
  fault-frame SVD, link map.
- `pygmt_reloc_map.py` — **PyGMT** before/after subregion map (`family738/pygmt_reloc_f738_reuse.png`):
  two panels on a shared square extent + depth colour scale, circles sized by KMA magnitude.

## Run it
```bash
./run.sh                      # or  PQ_PY=/path/to/python ./run.sh
```
PocketQuake run outputs land under `15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/f738_{reuse,fresh}/`
(`2.HypoDD/02.dt.cc/hypoDD.reloc`). Scaffolding registers `f738_reuse`/`f738_fresh` cluster modules in
that submodule (re-created by `run.sh`, so they need not be committed).

## Where the results are
- **Tidy tables** (`save_results.py`, written to `family738/`): `reloc_f738_reuse.csv`,
  `reloc_f738_fresh.csv` (event_id, time_utc, lat, lon, depth_km, relative x/y/z m, **ex95/ey95/ez95**
  bootstrap 95% half-widths once the bootstrap step has run, and P/S link counts), and
  `reloc_compare.csv` ((1)-vs-(2) per-event horizontal + depth offsets).
- **Raw PocketQuake outputs** under
  `15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/f738_{reuse,fresh}/`:
  `1.HypoInv/kim2011/*.sum` (absolute), `2.HypoDD/02.dt.cc/hypoDD.reloc` (dt.cc relocated),
  `2.HypoDD/02.dt.cc/bootstrap_errors.csv` + `bootstrap_samples.npz` (bootstrap).
- **Figures**: `compare_relocations.ipynb` (collapse overlay, PocketQuake map, depth + fault-frame SVD
  sections with bootstrap error bars, and the (1)-vs-(2) offset).

## Result (family 738)
- **Collapse**: absolute HypoInverse(kim2011) RMS horizontal spread **530 m → 94 m** with dt.cc (5.6×);
  depth std **410 m → 30 m** (14×). The multiplet tightens into a compact patch, as expected for a
  repeating family.
- **(1) ≈ (2)**: both dt.cc relocations collapse to ~95 m; the per-event offset between the reused and
  freshly-re-picked relocations is **~124 m median** (max ~260 m) horizontally — i.e. the pick instance
  moves events by about a cluster width. (§4 of the notebook can run the bootstrap 95% to contextualize.)

## Scale to ALL multiplets — the batch (`run_all.sh`)
The reuse scheme above is applied to **every 5-25 Hz family** (117 families ≥3 members) by:
```bash
./run_all.sh                     # all families, bootstrap n=200, then aggregate
./run_all.sh --no-bootstrap      # faster; --families 738,1218 or --limit 5 for subsets; --redo to force
```
- **`batch_relocate.py`** — loops the families largest-first, running the per-family reuse pipeline for
  each (slug `f<ID>_reuse`, dir `family<ID>`). **Robust**: a family HypoDD can't relocate (too few dt.cc
  links) is logged and falls back to **absolute-only** — the loop never aborts. **Resumable**: families
  already relocated are skipped. Writes **`batch_manifest.csv`** (one row/family: status, n_relocated,
  timing) + `batch.log`.
- **`aggregate_results.py`** — writes **`master_metrics.csv`** (one row per family: n, n_relocated,
  absolute vs dt.cc spread, `collapse_ratio`, bootstrap 95% medians, mean_cc, recurrence, centroid,
  status/stage_failed/note), **`failures.csv`** (families not fully relocated), **`master_map_relocated.png`**
  (all relocated families on the UF subregion + fault traces, coloured by family; absolute-only as grey),
  and top-N per-family fault-frame thumbnails.
- **`build_batch_summary_nb.py`** → **`batch_summary.ipynb`** — the aggregate overview notebook: status +
  combined map, collapse statistics, the **repeater-vs-multiplet** view (dt.cc spread vs bootstrap error —
  only families below the 1:1 line are well-resolved candidate repeaters), size/depth/time relationships,
  the master table + failures, and a top-N thumbnail gallery. Loads the CSVs only (no pipeline re-run).

Each family's tidy table lands in `family<ID>/reloc_f<ID>_reuse.csv`; PocketQuake run outputs under
`…/runs/f<ID>_reuse/`. The heavy per-family products (`build_summary_nb.py` summary notebook + GIF,
`build_compare_nb.py`) stay **on-demand** for scientifically interesting families — not generated in the batch.

**Reading the master table**: `collapse_ratio = dt.cc spread / absolute spread` (≈0.18 for the flagship
738) is the headline tightening; but cross-check `boot_horiz95_m` — a tiny family can collapse tight yet
have a large bootstrap error (e.g. a 3-event family at 28 m spread but ~370 m 95% half-width), meaning the
tightness is **not** well-constrained. Trust the collapse only where the bootstrap error is small.
