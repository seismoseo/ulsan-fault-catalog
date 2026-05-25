# Ulsan Fault Seismicity Catalog

An automated, reproducible pipeline to build a **long-term earthquake catalog (2010–present)**
for the Ulsan Fault region (SE Korea) from continuous seismic waveforms, using AI phase pickers
(SeisBench / PhaseNet). The station network grows over time, and the workflow is designed to be
re-run as new data and new picker models become available.

```
 Detection            Association          Absolute location      Relative relocation
 PhaseNet (SeisBench) ──► PyOcto ──────────► HYPOINVERSE ─────────► HypoDD  (planned)
 daily pick CSVs          events+assignments  located catalog        double-difference
```

## Status

| Stage | Tool | Status |
|-------|------|--------|
| Detection | SeisBench PhaseNet | ✅ automated (`stead` reference + `original` runs) |
| Association | PyOcto | ✅ automated |
| Absolute location | HYPOINVERSE (`hyp1.40`) | ✅ automated |
| Relative relocation | HypoDD | ⏳ planned |
| 2nd network (`NS`, post-2018/19) | — | ⏳ deferred |

The pipeline supports two independent dimensions: **picker model** (`stead` vs `original` PhaseNet
weights) and **velocity model** (`kim1983` vs `kim2011`).

## Quickstart

```bash
# 1. environment (see docs/how-to-run.md for the torch/CUDA detail)
conda env create -f environment.yml && conda activate ulsan
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA

# 2. one-time per clone: enable notebook output-stripping in git
bash tools/setup-git-filters.sh

# 3. run the full chain for one year (or a range)
cd KS_KG/models/pipeline
python run_pipeline.py --model original --years 2024
python run_pipeline.py --model original --years 2010-2024
```

Detection is idempotent — it skips days whose picks already exist, so runs resume safely.

## Repository layout (what's tracked)

```
CLAUDE.md  README.md  requirements.txt  environment.yml
docs/                  documentation (see below)
tools/                 git helpers (notebook output stripping)
KS_KG/
  models/
    pipeline/          ★ the automated pipeline (config.py, core.py, *.py CLIs)
    build_original_tree.py   regenerates the (local) models/original/ run
    README.md
  station_table/  velocity_model/  HypoInv/{*.crh, STA/*.sta}   reference metadata
```

> **Code, docs, and small reference metadata only.** The following live on the workstation and are
> **not** versioned (gitignored): continuous waveforms (~7 TB) and `NS/`; **all Jupyter notebooks**
> (the per-year `stead` reference run and the generated `models/original/` run); and large regenerable
> outputs (picks, association tables, HYPOINVERSE `.prt/.arc/.sum`). The pipeline runs against these
> local files; `build_original_tree.py` regenerates the `models/original/` notebooks on the workstation.
> See [docs/directory-structure.md](docs/directory-structure.md).

## Documentation

- [docs/overview.md](docs/overview.md) — goal, scientific motivation, plan & roadmap
- [docs/pipeline.md](docs/pipeline.md) — each stage: inputs, outputs, parameters, data flow
- [docs/how-to-run.md](docs/how-to-run.md) — step-by-step commands (env, stages, orchestrator, resume)
- [docs/directory-structure.md](docs/directory-structure.md) — layout, `models/` convention, what's in git
- [docs/development.md](docs/development.md) — git workflow, notebook filter, extending the pipeline
- [KS_KG/models/pipeline/README.md](KS_KG/models/pipeline/README.md) — pipeline quick reference
- [KS_KG/models/README.md](KS_KG/models/README.md) — the `stead`/`original` model split

## License

Not yet chosen — treat as all-rights-reserved until a `LICENSE` is added.
