# Ulsan Fault Seismicity Catalog — project guide

Guidance for Claude Code working in this repository.

## Goal

Build a **long-term (2010–present) earthquake catalog for the Ulsan Fault region** using
AI phase pickers (SeisBench) on continuous waveforms from a station network that **grows over
time**. The aim is an automated-but-rigorous pipeline that can be re-run as new data and new
picker models become available.

## Pipeline

```
Detection (PhaseNet, SeisBench)  ->  Association (PyOcto)  ->  Absolute location (HYPOINVERSE)  ->  Relative relocation (HypoDD, future)
        picks CSV                       events + assignments        UF<year>.{sum,arc}                 (not yet implemented)
```

The automated implementation lives in **`KS_KG/models/pipeline/`** (shared module + thin CLIs).
See `docs/how-to-run.md` for commands and `docs/pipeline.md` for stage details.

## Directory map

```
KS_KG/
  continuous/            raw waveforms (NOT in git; ~3.8 TB)
  station_table/         station metadata (stations.csv, station_update.dat)
  velocity_model/        kim1983.csv (PyOcto layered model)
  detection_location/    per-year notebooks 01/02/03 — the "stead" REFERENCE run
  picks/ pyocto/ HypoInv/ existing stead outputs (data mostly NOT in git)
  models/                picker-model dimension (this project's working area)
    stead/               symlinks to the reference run (read-only view; not in git)
    original/            PhaseNet "original" run (copies of notebooks, switched)
    pipeline/            *** automated pipeline: config.py, core.py, *.py CLIs ***
    build_original_tree.py
NS/                      second network, ~3.3 TB, mostly post-2018/19 — DEFERRED
tuto_material/           tutorials / third-party (NOT in git)
docs/                    documentation
tools/                   git helpers (nbstrip.py)
```

## Two independent "model" dimensions — keep them straight

- **Picker model** (`--model`): `stead` / `original` (SeisBench PhaseNet) or `phasenet_plus` (EQNet
  PhaseNet+, in-process; needs a local EQNet clone at `config.EQNET_DIR`). Top of `models/`.
- **Velocity model** (`--velmodel`): the crustal model — `kim1983` vs `kim2011`. Used by PyOcto/HYPOINVERSE.

**Preprocessing**: PhaseNet-style pickers want **raw** (demeaned) data + their own normalization — no
bandpass. (The legacy stead/original detection notebooks applied a 1–40 Hz bandpass; the `pipeline/`
SeisBench path and the `phasenet_plus` backend feed minimally-processed data.)

## Conventions / rules

- **Do not edit the `stead` reference run** (`KS_KG/detection_location/**`, `KS_KG/{picks,pyocto,HypoInv}`).
  It is the baseline to compare against. `models/stead/` only symlinks it (the scripts refuse
  `--model stead` writes unless `--force`).
- **New work goes under `KS_KG/models/`.** The `original` notebooks and `pipeline/` code are editable.
- **Canonical pick id**: detection writes `station = "NET.STA"` (e.g. `KG.BBK`); association derives the
  network from it. Do not reintroduce the old hardcoded `["KS"]*N + ["KG"]*…` split.
- **Non-destructive scaffolding**: `models/build_original_tree.py` only ever writes under `models/`.
- Defaults (paths, thresholds, region) live in `KS_KG/models/pipeline/config.py` — change them there.

## How to run (quick)

```bash
cd KS_KG/models/pipeline
# SHARED 64-core box: ALWAYS pin cores with taskset + set OMP_NUM_THREADS.
OMP_NUM_THREADS=1  taskset -c 0-7  python run_pipeline.py --model original      --years 2010-2024
OMP_NUM_THREADS=16 taskset -c 8-23 python run_pipeline.py --model phasenet_plus --years 2010-2024
python detection.py --model original --year 2024 --days 1-5    # one stage / slice
```
Detection is idempotent (skips days whose picks already exist). Full details: `docs/how-to-run.md`.

### Performance & shared-server CPU (see `docs/performance-notes.md`)

This is a **shared 64-core server** — keep the footprint polite. Detection sizes its preprocessing
pool and `torch` threads from the process CPU **affinity** (`os.sched_getaffinity`), capped by
`config.MAX_CORES` (24), so launching under **`taskset -c <cores>`** auto-scopes the whole job
(and all worker threads) to that core budget. Without pinning, PhaseNet+ grabbed ~49 cores / 193
threads and starved everything. Inference is **GPU-preferred** (warns loudly, never silently falls
back to CPU). Preprocessing uses **one reused `forkserver` `ProcessPoolExecutor`** per year
(created before the model loads, so workers are lean) — not the old per-day pool that forked the
23 GB CUDA parent 5,475×.

## Environment

- miniforge Python; `seisbench`, `pyocto`, `obspy`, `torch` (+ CUDA GPU). See `requirements.txt` / `environment.yml`.
- **HYPOINVERSE** needs the external `hyp1.40` binary on `PATH` (not pip-installable).

## Version control

- GitHub: `seismoseo/ulsan-fault-catalog` (public). The repo tracks **code, docs, and small reference
  metadata only** (station tables, velocity model, `*.crh`, `HypoInv/STA/*.sta`).
- **Not tracked** (gitignored — verify with `git status` before committing): waveforms
  (`KS_KG/continuous/`, `NS/`); **all notebooks** (the `stead` per-year run, the generated
  `models/original/` run, the 424 MB `01.PhaseNet_detection_test.ipynb`); original run scripts
  (`HypoInv/UF*.sh`, `HypoInv/STA/hypoinverse/`); large outputs (`picks/`, `pyocto/`, HYPOINVERSE
  `*.prt/*.arc/*.sum`); `tuto_material/`.
- A git clean filter (`tools/nbstrip.py`, enabled once via `bash tools/setup-git-filters.sh`) strips
  notebook outputs **if** a notebook is ever intentionally added — kept as a safety net.

## Status & next steps (2026-05-26)

- **stead** catalog complete (2010–2024 located `kim2011/UF<year>.sum`). Summary in
  `KS_KG/HypoInv/catalog_summary.ipynb` (model-parameterized; writes `catalog_<model>_2010_2024.csv`;
  includes maps, depth sections, cumulative/rate, network growth, KMA comparison, and hour-of-day
  (KST) diurnal + spatial-variation analysis for blast vs tectonic discrimination).
- **Detection performance fixed** (was a ~35-day ETA): reused forkserver pool, lossless handling of
  fragmented station-days, GPU-preferred inference, polite `taskset` CPU budget. Root-cause writeup
  in `docs/performance-notes.md`. **original** + **phasenet_plus** 2010–2024 re-runs running pinned
  to ~24 cores (`models/<model>/run_2010_2024.log`).
- **PhaseNet+ inspection**: `core.annotate_phasenet_plus(year, day, station, t0, t1)` returns the
  per-sample P/S/noise probability, first-motion polarity, and single-station event-detection traces
  (the PhaseNet+ analogue of SeisBench `annotate`); used by the rebuilt
  `models/pipeline/notebooks/phasenet_plus_test.ipynb`.
- **Post-location analysis** (`KS_KG/HypoInv/uf_cluster.py` + notebooks `03_blast_decluster_hdbscan`,
  `04_subregion_seismicity`, `05_error_ellipses`; see `docs/analysis.md`): 3D HDBSCAN clustering with
  hour-of-day **quarry-blast discrimination** (writes a declustered catalog), an **east-of-fault subregion**
  long-term-seismicity study, and **95% HYPOINVERSE error ellipses** parsed from the `.prt` covariance.
  `uf_cluster.py` is tracked; the notebooks + their CSV/HTML outputs are gitignored.
- **Residual quarry blasts** survive cluster-level declustering as HDBSCAN **noise** (diffuse daytime shots).
  A second-stage **spatial daytime-fraction grid mask** (`grid_blast_stats`/`flag_blast_cells`/
  `decluster_spatial`/`decluster_full`; cell 0.02°, N≥10, daytime_frac>0.80, Rayleigh p<0.01) removes daytime
  events in flagged "quarry cells" → `catalog_*_blastclean.csv`. Empirically (stead): 22 cells, +302 events
  (295 from noise), 11,065→10,763, daytime frac 0.473→0.458, and **0 subregion events** (east-of-fault zone is
  blast-free). Residual blasts are reported **deep** (~9 km) but **avoid weekends** — `weekend_ratio` =
  (Sat/Sun fraction)÷(2/7) (1.0 = no preference, <1 = weekday-only/blast-like). It is **reported** in the
  cluster + grid tables and is an **optional gate** (`flag_blasts`/`flag_blast_cells`/`decluster_full`/notebook-03
  `WEEKEND_MAX`, default **off** — daytime + Rayleigh only); enable (e.g. <0.7) to also demand weekend-avoidance
  for the deep residual blasts that depth can't catch. Notebook 03 also **maps the final blast-clean catalog**
  (§9c: epicenter + cyclic hour-of-day) and includes a **grid-only-vs-two-step robustness check** (§9d, compared
  by catalog index; the two-step is kept — grid-only on the full catalog raises subregion false-positives).
  Notebook 04 defaults to the blast-clean catalog (`USE_BLASTCLEAN=True`) and adds **per-year subregion
  small-multiples** (`uf.annual_maps`, epicenters + density) for year-to-year spatial comparison.
- **#1 gap**: HYPOINVERSE `.sum` `MAG` column is empty → no magnitudes ⇒ no FMD/Mc/b-value yet. Top TODO:
  compute **Md** (coda duration via HYPOINVERSE) or **ML** (Wood–Anderson amplitudes + station corrections).
- Later: 3-picker comparison once re-runs finish; **HypoDD** relative relocation.

## Gotchas

- 2010 & 2013 detection notebooks originally pointed at an outdated `2014_sequence/continuous`
  path; the `models/original` copies are repointed to `KS_KG/continuous`.
- `detection_location/2022/picks/` (stead) has ~670 files (a likely double run) — sanity-check.
- Detection runs all stations into a single per-day GPU `classify()` call; preprocessing is
  parallelized with one reused `forkserver` `ProcessPoolExecutor` (see Performance note above).
- **YSB is fragmented in 2010** (~142 days stored as tens of thousands of ~5 s miniSEED records —
  real continuous data, but obspy `merge()` is ~O(n²) ⇒ ~100 s/day). It is processed **losslessly**
  (`config.MAX_SEGMENTS` only logs a warning; `HARD_MAX_SEGMENTS` is the sole skip for a corrupt
  file). The pre-fix run had silently *lost* YSB on those days. Fragmentation is YSB-/2010-specific.
- **HYPOINVERSE `.prt` errors are 1-σ.** ERH ≈ 1-σ horizontal semi-major and ERZ ≈ √var_Z (verified
  median ratios ≈ 1.0), so a 95% **joint** horizontal error ellipse needs `k = √χ²₂,₀.₉₅ = 2.448×` the
  1-σ axes (depth 95% = 1.96·σ_z). `.prt` covariance fields are fixed-width 8-char (slice, don't split —
  they glue when large); overflow prints `********`→NaN (junk events with 20–99 km errors, all
  QC-excluded so harmless); origin seconds can be negative; longitude carries an `E`/`W` letter. A few
  2023 events lack `.prt` covariance because the filtered `.sum` and `UF2023.prt` are from different runs
  (~0.4 s / ~1 km apart) — left unmatched (≈99.9% coverage). Parser/maps in `uf_cluster.py`.
- **matplotlib maps draw coastlines** via `uf_cluster.coast_mpl`/`coast_mpl_km` (cartopy 0.25 NaturalEarth
  10m, lon/lat or local-km frame) to match the PyGMT `fig.coast` maps. Only the coastline **line** is used —
  the 10m land/ocean polygons aren't cached and would trigger a download (helpers degrade gracefully if the
  cache is missing). Z-order on every map: coast 0.5 < faults 1 < noise 2 (small + translucent) < clusters /
  seismicity 3 < subregion box 4–5, so clusters always render above the noise background.
