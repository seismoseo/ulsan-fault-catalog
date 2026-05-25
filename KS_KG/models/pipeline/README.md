# `models/pipeline/` — automated KS_KG catalog pipeline

Parameterized scripts for the full chain, replacing the manual per-year notebooks
for production runs (notebooks stay for exploration). Everything is keyed by a
**picker model** (`--model`, default `original`) and writes into `models/<model>/`.

```
detection.py        picks            -> models/<model>/detection_location/<year>/picks/
association.py      PyOcto events    -> models/<model>/pyocto/pyocto_kim1983_<year>.csv (+ assignment)
make_phs.py         HYPO71 phase     -> models/<model>/HypoInv/PHS/UF<year>.phs
run_hypoinverse.py  located events   -> models/<model>/HypoInv/<velmodel>/UF<year>.{sum,prt,arc}
run_pipeline.py     orchestrates all four over a year range
config.py / core.py shared defaults + stage functions (single source of truth)
```

## Quick start

```bash
cd /home/msseo/works/02.Ulsan_Fault_detection/KS_KG/models/pipeline

# one year, end to end
python run_pipeline.py --model original --years 2024

# the whole record
python run_pipeline.py --model original --years 2010-2024

# individual stages
python detection.py       --model original --year 2024 [--days 1-305] [--workers 8] [--device cuda]
python association.py     --model original --year 2024
python make_phs.py        --model original --year 2024
python run_hypoinverse.py --model original --year 2024 --velmodel kim2011

# resume midway (picks already computed)
python run_pipeline.py --model original --years 2015,2016 --stage-from association
```

Detection is idempotent: it **skips days whose picks CSV already exists** (use
`--no-skip-existing` to recompute). The orchestrator continues on per-year errors
and prints a summary at the end.

## From a notebook

```python
import sys; sys.path.insert(0, "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/models/pipeline")
import core, config
core.run_detection_year("original", 2024, days=range(1, 6))   # small test slice
events, assignments = core.run_association_year("original", 2024)
core.write_phs("original", 2024)
core.run_hypoinverse_year("original", 2024, velmodel="kim2011")
```

## Notes

- **Canonical pick station id**: detection writes `station = "NET.STA"` (e.g. `KG.BBK`),
  normalized from the SeisBench `trace_id`. This fixes the per-year format drift in the
  old picks (`BBK` / `KG.BBK.00` / `KG.BBK.`). Association derives Network from this and
  no longer relies on the fragile hardcoded `["KS"]*N + ["KG"]*…` split.
- **stead is protected**: `models/stead/*` symlinks the existing reference run, so the
  scripts refuse `--model stead` unless `--force` is given.
- **Two model dimensions**: picker model (`--model`: stead/original) vs velocity model
  (`--velmodel`: kim1983/kim2011). They are independent.
- Defaults (region, thresholds, paths) live in `config.py`; edit there, not in each script.
- HYPOINVERSE needs `hyp1.40` on `PATH`; `STA/` and the `*.crh` files are symlinked into
  `models/<model>/HypoInv/` by `models/build_original_tree.py`.
