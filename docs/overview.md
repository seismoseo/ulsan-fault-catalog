# Overview — goal, plan, and roadmap

## Goal

Construct a **long-term, internally consistent earthquake catalog for the Ulsan Fault region**
(southeastern Korea), spanning 2010 to the present, from continuous seismic waveforms. The catalog
is built with modern AI phase pickers so that small events are detected uniformly across the whole
period, even as the **number of recording stations increases over time**.

The broader aim is a pipeline that is **as automated as possible without losing rigor**: every stage
is scripted and parameterized, results are reproducible, and alternative choices (different picker
models, different velocity models) can be run side-by-side and compared.

## Why this design

- **AI picking (PhaseNet/SeisBench)** detects far more events than classic STA/LTA, and consistently,
  which matters for a multi-year catalog with a changing network.
- **Comparing picker models** (`stead` vs `original` PhaseNet weights) tests how sensitive the catalog
  is to the training data of the picker — hence the `models/<picker>/` split that keeps runs isolated.
- **Comparing velocity models** (`kim1983` vs `kim2011`) tests location sensitivity, independently of
  the picker.

## The four-stage pipeline

1. **Detection** — PhaseNet picks P/S arrivals on each station-day → daily pick CSVs.
2. **Association** — PyOcto groups picks into events with hypocentres → event + assignment tables.
3. **Absolute location** — HYPOINVERSE relocates each event in a 1-D crustal model → located catalog.
4. **Relative relocation** — HypoDD (double-difference) for high-precision relative locations *(planned)*.

Details in [pipeline.md](pipeline.md); commands in [how-to-run.md](how-to-run.md).

## Plan / roadmap

| Step | State | Notes |
|------|-------|-------|
| Detection automation (CLI + resume) | ✅ done | `models/pipeline/detection.py` |
| Association automation | ✅ done | `association.py`; robust per-station network from picks |
| HYPOINVERSE automation | ✅ done | `run_hypoinverse.py` wraps `hyp1.40` |
| Orchestrator (full chain, year ranges) | ✅ done | `run_pipeline.py` |
| `stead` reference run | ✅ exists | the original 2010–2024 notebooks |
| `original` PhaseNet run | ◐ in progress | scaffold + scripts ready; full years to be run |
| HypoDD relative relocation | ⏳ planned | `models/<model>/hypodd/` placeholder |
| `NS` network (post-2018/19) | ⏳ deferred | extend `--model`/path convention later |
| Magnitudes / completeness (Mc) | ⏳ future | not started |

## Scope of this repository

Code, documentation, and small metadata only. The ~7 TB of continuous waveforms and the large
regenerable data products (picks, association tables, HYPOINVERSE outputs) live on the workstation
and are **not** distributed here — see [directory-structure.md](directory-structure.md).
