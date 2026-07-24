# Ulsan Fault Seismicity Catalog

An automated, reproducible pipeline to build a **long-term earthquake catalog (2010–present)**
for the Ulsan Fault region (SE Korea) from continuous seismic waveforms, using AI phase pickers
(SeisBench PhaseNet / EQNet PhaseNet+). The station network grows over time, and the workflow is
designed to be re-run as new data and new picker models become available.

```
 Detection              Association       Absolute location   QC        Relative relocation
 PhaseNet / PhaseNet+ ─► PyOcto ─────────► HYPOINVERSE ──────► filter ─► HypoDD dt.ct + dt.cc
 daily pick parquet      events+assign      located catalog              double-difference
```

## Status

| Stage | Tool | Status |
|-------|------|--------|
| Detection | SeisBench PhaseNet (`stead`, `original`), EQTransformer | ✅ automated |
| Detection | EQNet **PhaseNet+** (`phasenet_plus`) | ✅ automated (needs a local EQNet clone) |
| Association | PyOcto | ✅ automated |
| Absolute location | HYPOINVERSE (`hyp1.40`) | ✅ automated |
| Relative relocation | HypoDD (dt.ct + dt.cc) | ✅ done |
| Networks | KS/KG, GJ (Gyeongju temporary array), NS | ✅ KS/KG/GJ; NS integrating |

Two independent dimensions: **picker model** (`--model`: `stead` / `original` / `phasenet_plus` / `eqt`)
and **velocity model** (`--velmodel`: `kim1983` / `kim2011`). PhaseNet-style pickers run on **raw**
(demeaned) data with their own internal normalization — no bandpass.

## Repository layout

Restructured (2026-07) into a software-style tree — installable packages, code, data, and outputs
cleanly separated. The four waveform networks are parallel top-level directories.

```
pyproject.toml  environment.yml  requirements.txt  README.md  CLAUDE.md
src/
  uflib/          shared analysis library  (uf_cluster, uf_waveform_similarity, event_sac_export)
  ufpipe/         the detection→location pipeline  (was models/pipeline; renamed to avoid a name clash)
analysis/         non-installable analysis code + notebook builders
  relocation/  reloc_analysis/  local_magnitudes/  uf_subregion_hypodd/  repeaters/  hypoinv/
detection_test/   the 4-picker comparison pipeline (year-general; see src/ufpipe/reloc_driver/PIPELINE.md)
KS_KG/  GJ/  NS/  NS_100hz/     raw waveforms — station dirs at each root (~7 TB, NOT in git)
data/
  waveforms/      symlinks to the network dirs (browsable view; no data copied)
  metadata/       station tables, velocity model, StationXML, external catalogs
  hypoinv/        HYPOINVERSE control inputs (STA/*.sta, kim*/*.crh) + working data
outputs/          regenerable pipeline products (models/<picker>/{picks,pyocto,HypoInv}, …) — NOT in git
docs/             documentation  ·  notebooks/  archive/  papers/  tools/
```

> **Git tracks code, docs, and small reference metadata only.** Raw waveforms (`KS_KG/`, `GJ/`, `NS/`,
> `NS_100hz/`, ~7 TB), pipeline outputs (`outputs/`), generated notebooks, and large data
> (StationXML, per-station ML CSVs, HypoDD residuals) are gitignored — the pipeline runs against these
> local files. See [docs/directory-structure.md](docs/directory-structure.md).

## Install

On this server the pipeline uses a **two-env split**: detection (PhaseNet+/SeisBench, torch) runs in the
`eqnet` conda env (Py 3.9); association (PyOcto) + orchestration run in `base` (Py 3.12). Install the
packages editable in **both**:

```bash
conda run -n eqnet pip install -e . --no-deps
conda run -n base  pip install -e . --no-deps
```

On a fresh machine, `environment.yml` describes the dependency set for a single new env
(`conda env create -f environment.yml && conda activate uf-catalog`), then
`pip install torch --index-url https://download.pytorch.org/whl/cu128` to match your CUDA and `pip install -e .`.

`pip install -e .` makes `from uflib import uf_cluster` and `import ufpipe.config` work from any
directory — no more `sys.path.insert`.

## Quickstart

**Long-term catalog pipeline** (detection → association → location):

```bash
python -m ufpipe.run_pipeline --model original --years 2024          # one year
python -m ufpipe.run_pipeline --model original --years 2010-2024     # a range
```

**4-picker comparison / relocation** (year-general; any year whose inputs exist):

```bash
python src/ufpipe/reloc_driver/preflight_year.py --year 2016                 # readiness check
python src/ufpipe/reloc_driver/run_picker_reloc.py --picker original \
       --year 2016 --through dtcc --clean-cache                                    # relocate
```

Detection is idempotent (skips days whose picks already exist) and resumable.

## Documentation

- [docs/overview.md](docs/overview.md) — goal, scientific motivation, roadmap
- [docs/pipeline.md](docs/pipeline.md) — each stage: inputs, outputs, parameters
- [docs/how-to-run.md](docs/how-to-run.md) — step-by-step commands
- [docs/directory-structure.md](docs/directory-structure.md) — layout & what's in git
- [src/ufpipe/README.md](src/ufpipe/README.md) — pipeline quick reference
- [src/ufpipe/reloc_driver/PIPELINE.md](src/ufpipe/reloc_driver/PIPELINE.md) — the relocation
  pipeline: invariants, provenance checks, the KST/KMA convention

## License

Not yet chosen — treat as all-rights-reserved until a `LICENSE` is added.
