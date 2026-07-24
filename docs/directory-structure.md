# Directory structure & conventions

Restructured (2026-07) into a software-style tree: installable packages under `src/`, non-installable
analysis code under `analysis/`, waveforms as parallel top-level network directories, and a clean
data / code / outputs separation.

```
02.Ulsan_Fault_detection/
├── pyproject.toml  environment.yml  requirements.txt  README.md  CLAUDE.md
├── src/                          ★ installable packages (pip install -e .)
│   ├── uflib/                    shared analysis library
│   │   ├── uf_cluster.py             spatial/temporal quarry-blast decluster + map helpers (QC, read_sum)
│   │   ├── uf_waveform_similarity.py waveform-feature blast screening (imports uf_cluster)
│   │   └── event_sac_export.py       event-idx-keyed SAC store writer (fully path-parameterized)
│   └── ufpipe/                   the 6-stage pipeline (detection→association→augment→phs→locate→relocate)
│       ├── config.py core.py stations.py detection.py association.py make_phs.py run_hypoinverse.py
│       ├── reloc_inputs.py relocate.py run_pipeline.py
│       └── reloc_driver/             the relocation orchestration (year_paths, run_picker_reloc, PIPELINE.md)
├── analysis/                     non-installable analysis code + notebook builders (import uflib/ufpipe)
│   ├── relocation/                  HypoDD relocation batch driver + family maps
│   ├── reloc_analysis/              cluster / NND / fractal-dimension notebooks
│   ├── local_magnitudes/            ML (Heo 2024 + Sheen 2018) pipeline + notebooks
│   ├── uf_subregion_hypodd/         whole-box dt.cc relocation + SVD volumes
│   ├── repeaters/                   repeating-earthquake + Vp/Vs notebooks
│   └── hypoinv/                     HYPOINVERSE-related analysis scripts + nb builders
├── detection_test/               FROZEN 2016 4-picker pilot archive (fully stale; nothing live)
│   ├── lib/                          DEPRECATED per-month feeder (see lib/DEPRECATED.md)
│   └── reloc_2016_uf*/              the pilot's frozen run dirs (manifests, results, study_guide)
├── KS_KG/  GJ/  NS/  NS_100hz/    ★ raw waveforms — station dirs at each root (parallel; ~7 TB; NOT in git)
├── data/
│   ├── waveforms/                    symlinks to the four network dirs (browsable view; no data copied)
│   ├── metadata/                     ★ single metadata home, organized BY KIND
│   │   ├── stations/                 ks_kg/  gj/  ns/  kigam/   (per-network station tables)
│   │   ├── responses/                master/ (148 MB StationXML, gitignored) + fetched/ + RESP.* text
│   │   ├── velocity/                 kim1983.csv
│   │   └── catalogs/                 ghbsn_heo/ (Heo et al.), USGS_M7_event_catalog.csv
│   └── hypoinv/                      HYPOINVERSE control inputs (STA/*.sta, kim*/*.crh) + working data
├── outputs/                      regenerable pipeline products — NOT in git
│   ├── models/<picker>/              picks / pyocto / HypoInv per picker model
│   └── reloc/reloc_<year>_uf[_<m>]/  stage-6 relocation working dirs + results/
├── docs/                         documentation (this folder) + docs/planning/ (design + gap-analysis notebooks)
├── notebooks/  archive/  papers/  tools/
```

## Two "model" dimensions

| Dimension | Flag | Values | Where it appears |
|-----------|------|--------|------------------|
| **Picker model** | `--model` / `--picker` | `stead`, `original`, `phasenet_plus`, `eqt` | pipeline output paths |
| **Velocity model** | `--velmodel` | `kim1983`, `kim2011` | PyOcto filename + `data/hypoinv/<velmodel>/` |

Orthogonal: any picker model can be located with any velocity model.

## What is tracked in git

The repo holds **code, docs, and small reference metadata only** — no waveforms, no large outputs.

**Tracked**
- code: `src/**` (uflib + ufpipe packages), `analysis/**/*.py`, `detection_test/**/*.py`, `tools/**`, `pyproject.toml`
- docs: `README.md`, `CLAUDE.md`, `docs/**`, package READMEs, `src/ufpipe/reloc_driver/PIPELINE.md`
- reference metadata: `data/metadata/stations/**` (all networks), `data/metadata/velocity/*`,
  `data/metadata/catalogs/*.csv`, small text responses `data/metadata/responses/**/RESP.*`,
  HYPOINVERSE control inputs `data/hypoinv/STA/*.sta`, `data/hypoinv/{kim1983,kim2011}/*.crh`

**Not tracked** (see [`.gitignore`](../.gitignore))
- waveforms: `KS_KG/`, `GJ/`, `NS/`, `NS_100hz/` (station dirs at each root, ~7 TB) + the `data/waveforms/` symlinks
- outputs: `outputs/`, `**/picks/`, `**/pyocto/`, HYPOINVERSE `*.prt/*.arc/*.sum`, `*.phs`
- large data: `data/hypoinv/event_waveforms_*/`, `data/metadata/responses/master/` (148 MB StationXML) +
  `responses/fetched/zips/`, per-station ML CSVs, HypoDD `*.res`, SVD volumes, `.gif`
- **generated notebooks** (`analysis/**/*.ipynb`, `detection_test/**/*.ipynb`, `data/hypoinv/**/*.ipynb`) — the
  builders (`build_*_nb.py`) are tracked; the notebooks they emit are not
- Jupyter checkpoints, Python caches, `*.egg-info/`

Waveforms live on the workstation; notebooks and data products are **regenerable** (notebooks via their
`build_*_nb.py` builders; catalogs via the pipeline), so the repository stays small and code-focused.

## Install & import

```bash
# two-env split: detection (PhaseNet+, torch) in `eqnet`; association (PyOcto) + rest in `base`.
conda run -n eqnet pip install -e . --no-deps
conda run -n base  pip install -e . --no-deps   # makes uflib + ufpipe importable from any directory
```

Then `from uflib import uf_cluster`, `import ufpipe.config`, or `python -m ufpipe.run_pipeline` work
anywhere — the former `sys.path.insert(".../KS_KG/HypoInv")` pattern is gone.
