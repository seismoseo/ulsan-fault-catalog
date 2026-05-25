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

- **Picker model** (`--model`): the PhaseNet weights — `stead` vs `original`. Top of `models/`.
- **Velocity model** (`--velmodel`): the crustal model — `kim1983` vs `kim2011`. Used by PyOcto/HYPOINVERSE.

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
python run_pipeline.py --model original --years 2010-2024      # full chain
python detection.py    --model original --year 2024 --days 1-5 # one stage / slice
```
Detection is idempotent (skips days whose picks already exist). Full details: `docs/how-to-run.md`.

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

## Gotchas

- 2010 & 2013 detection notebooks originally pointed at an outdated `2014_sequence/continuous`
  path; the `models/original` copies are repointed to `KS_KG/continuous`.
- `detection_location/2022/picks/` (stead) has ~670 files (a likely double run) — sanity-check.
- Detection runs all stations into a single per-day `classify()` call (efficient); preprocessing
  is parallelized with `ProcessPoolExecutor`.
