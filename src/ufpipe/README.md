# `ufpipe` — the end-to-end Ulsan-Fault catalog pipeline

Installable package (`pip install -e .` from the repo root — needed in BOTH the `eqnet` env for
detection and `base` for association). Everything is keyed by a **picker model** (`--model`, default
`original`) and writes into `outputs/models/<model>/`. Six stages:

```
detection.py        picks             -> outputs/models/<model>/detection_location/<year>/picks/     (eqnet env)
association.py      PyOcto events     -> outputs/models/<model>/pyocto/pyocto_kim2011_<year>.csv (+ assignment)  (base env)
(augment)           orphan-pick rescan — updates the assignment in place (via run_pipeline; no standalone CLI)
make_phs.py         HYPO71 phase      -> outputs/models/<model>/HypoInv/PHS/UF<year>.phs
run_hypoinverse.py  located events    -> outputs/models/<model>/HypoInv/<velmodel>/UF<year>.{sum,prt,arc}
relocate.py         HypoDD dt.cc      -> detection_test/reloc_<year>_uf[_<model>]/results/  (self-fed via reloc_inputs.py)
run_pipeline.py     orchestrates all six over a year range
config.py / core.py / stations.py     shared defaults + stage functions + the KS/KG/GJ/NS station table
```

## Quick start

```bash
cd /home/msseo/works/02.Ulsan_Fault_detection

# one year, end to end (through HypoDD dt.cc)
python -m ufpipe.run_pipeline --model original --years 2024 --through dtcc

# the whole record
python -m ufpipe.run_pipeline --model original --years 2010-2024

# individual stages (detection in eqnet, association in base)
conda run -n eqnet python -m ufpipe.detection   --model original --year 2024 [--days 1-305] [--networks KS,KG,GJ,NS]
conda run -n base  python -m ufpipe.association --model original --year 2024 [--workers 8]
python -m ufpipe.make_phs        --model original --year 2024
python -m ufpipe.run_hypoinverse --model original --year 2024 --velmodel kim2011

# resume midway / relocation only
python -m ufpipe.run_pipeline --model original --years 2015,2016 --stage-from association
python -m ufpipe.run_pipeline --model original --years 2016 --stage-from relocate --through dtcc
```

Detection is idempotent: it **skips days whose picks CSV already exists** (use
`--no-skip-existing` to recompute). The orchestrator continues on per-year errors
and prints a summary at the end.

## From a notebook

```python
from ufpipe import core, config            # works from any directory (pip install -e .)
core.run_detection_year("original", 2024, days=range(1, 6))   # small test slice
events, assignments = core.run_association_year("original", 2024)
core.write_phs("original", 2024)
core.run_hypoinverse_year("original", 2024, velmodel="kim2011")
```

## Notes

- **Networks**: detection + association cover **KS/KG/GJ/NS** by default; the per-year multi-archive
  station table is built by `stations.py` (NS reads the `NS_100hz/` mirror). Restrict with `--networks`.
- **Association is daily-chunked** (±150 s window per day, in-day origins) — required for the dense NS
  array; velocity = kim2011 (`config.PYOCTO_VELMODEL`). Legacy pre-2026-07 whole-year artifacts on disk
  are named `pyocto_kim1983_*`.
- **Canonical pick station id**: detection writes `station = "NET.STA"` (e.g. `KG.BBK`),
  normalized from the picker `trace_id`. Association derives Network from this.
- **stead is protected**: `outputs/models/stead/*` symlinks the reference run; scripts refuse
  `--model stead` writes unless `--force`.
- **Two model dimensions**: picker model (`--model`: stead/original/phasenet_plus/eqt) vs velocity model
  (`--velmodel`: kim1983/kim2011). They are independent.
- Defaults (region, thresholds, paths) live in `config.py`; edit there, not in each script.
- HYPOINVERSE needs `hyp1.40` on `PATH`; `STA/` and the `*.crh` files are symlinked into
  `outputs/models/<model>/HypoInv/` (targets live in `data/hypoinv/`).
