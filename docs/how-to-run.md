# How to run

> **Full reference:** [ufpipe_reference_manual.pdf](ufpipe_reference_manual.pdf) — all stages, parameters, and troubleshooting in one place.

## 0. Environment (once)

```bash
conda env create -f environment.yml
conda activate ulsan
# install the torch build matching your CUDA (GPU strongly recommended for detection):
pip install torch --index-url https://download.pytorch.org/whl/cu128
# install the uflib + ufpipe packages (editable) — makes them importable from any directory:
pip install -e .
```

Check it imports and sees the GPU:

```bash
python -c "import seisbench, pyocto, obspy, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import ufpipe.config as c, os; print('waveforms:', c.CONTINUOUS, os.path.isdir(c.CONTINUOUS))"
```

HYPOINVERSE: the external `hyp1.40` binary must be on your `PATH` (used by the location stage).

All commands below are run from the repository root with `python -m ufpipe.run_pipeline`.

## 1. Full chain (recommended)

```bash
# one year, end to end (detection → association → PHS → location)
python -m ufpipe.run_pipeline --model original --years 2024

# the whole record
python -m ufpipe.run_pipeline --model original --years 2010-2024

# a few specific years, kim1983 crustal model
python -m ufpipe.run_pipeline --model original --years 2015,2016 --velmodel kim1983

# resume from association (picks already computed)
python -m ufpipe.run_pipeline --model original --years 2015,2016 --stage-from association

# quick plumbing test on a 3-day slice
python -m ufpipe.run_pipeline --model original --years 2024 --days 1-3
```

The orchestrator runs years independently, continues past a failing year, and prints a
`PIPELINE SUMMARY` at the end.

## 2. Individual stages

```bash
python -m ufpipe.detection       --model original --year 2024 [--days 1-305] [--workers 8] [--device cuda]
python -m ufpipe.association     --model original --year 2024
python -m ufpipe.make_phs        --model original --year 2024
python -m ufpipe.run_hypoinverse --model original --year 2024 --velmodel kim2011
```

Common flags: `--model` (picker: `original`/`stead`/future), `--year`, `--force` (allow writing into
the protected `stead` tree). Detection extras: `--days A-B`, `--stations CODE1,CODE2`, `--device`,
`--workers`, `--no-skip-existing`.

**Resume / idempotency**: detection skips days whose `picks_<year>.<doy>.csv` already exists, so
re-running a year only fills the gaps. Use `--no-skip-existing` to force recomputation.

## 2b. PhaseNet+ (EQNet picker)

`--model phasenet_plus` swaps the detector for EQNet's PhaseNet+ (P/S picks + polarity + event
detection) while the rest of the chain is identical. Prerequisite: a local clone of
[AI4EPS/EQNet](https://github.com/AI4EPS/EQNet) with the bundled weights
(`docs/model_phasenet_plus/model_99.pth`); set `EQNET_DIR` in `src/ufpipe/config.py` (default
`/home/msseo/works/14.EQNet/EQNet`). No `wandb` needed (the backend imports `eqnet` in-process).

```bash
# detection only (raw input; --highpass 0 = no filter; --min-prob is the pick threshold)
python -m ufpipe.detection --model phasenet_plus --year 2024 [--days 1-305] [--min-prob 0.3] [--highpass 0]
# full chain
python -m ufpipe.run_pipeline --model phasenet_plus --years 2024 --velmodel kim2011
```

Outputs: canonical picks in `outputs/models/phasenet_plus/detection_location/<year>/picks/` (same schema as
stead/original) and the raw PhaseNet+ picks (with `phase_polarity`/`phase_amplitude`) + single-station
event detections in `outputs/models/phasenet_plus/phasenet_plus_raw/<year>/`.

> **Comparison note**: PhaseNet+ runs on **raw** data (model normalizes internally), whereas the
> existing stead/original picks were made on 1–40 Hz bandpassed data, and PhaseNet+'s default threshold
> (0.3) differs from the SeisBench runs (0.2). Keep these in mind when comparing catalogs.

## 3. From a Jupyter notebook

The same functions back the notebooks, so you can call them directly:

```python
from ufpipe import core, config

core.run_detection_year("original", 2024, days=range(1, 6))   # small slice
events, assignments = core.run_association_year("original", 2024)
core.write_phs("original", 2024)
core.run_hypoinverse_year("original", 2024, velmodel="kim2011")
```

## 4. Where the outputs go

```
outputs/models/original/detection_location/<year>/picks/picks_<year>.<doy>.csv
outputs/models/original/pyocto/pyocto_kim1983_<year>.csv  (+ _assignment_)
outputs/models/original/HypoInv/PHS/UF<year>.phs
outputs/models/original/HypoInv/<velmodel>/UF<year>.{sum,prt,arc}   # .sum = located catalog
```

Compare against the `stead` reference run via `outputs/models/stead/…` (symlinks to the original outputs).

## 5. Tips

- **Tuning**: detection thresholds (0.2) and PyOcto association params live in `src/ufpipe/config.py`; edit there.
  With low thresholds + `n_picks=4`, association can produce many events — adjust if needed.
- **Performance**: detection is GPU-bound for `classify` and CPU-bound for preprocessing; tune `--workers`.
- **stead is protected**: scripts refuse `--model stead` writes unless `--force` (it's the reference run).
