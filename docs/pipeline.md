# Pipeline stages

Each stage is a function in [`KS_KG/models/pipeline/core.py`](../KS_KG/models/pipeline/core.py),
exposed by a thin CLI. All defaults live in
[`config.py`](../KS_KG/models/pipeline/config.py). Outputs are written under
`KS_KG/models/<model>/` (default `--model original`).

```
continuous waveforms
   │  detection.py            PhaseNet
   ▼
models/<model>/detection_location/<year>/picks/picks_<year>.<doy>.csv   (station, phase, peak_time, probability)
   │  association.py          PyOcto
   ▼
models/<model>/pyocto/pyocto_kim1983_<year>.csv            (events: idx, time, x,y,z, picks, lat, lon, depth)
models/<model>/pyocto/pyocto_assignment_kim1983_<year>.csv (assignments: event_idx, station, phase, time, …)
   │  make_phs.py
   ▼
models/<model>/HypoInv/PHS/UF<year>.phs                    (HYPO71 fixed-width phase file)
   │  run_hypoinverse.py      hyp1.40
   ▼
models/<model>/HypoInv/<velmodel>/UF<year>.{sum,prt,arc}   (located catalog)
```

## 1. Detection — `detection.py`

- **Tool**: SeisBench `PhaseNet.from_pretrained(<model>)`, GPU if available.
- **Input**: `KS_KG/continuous/<STA>/<CHAN>/*.<year>.<doy>` (auto-discovers stations with data that year).
- **Per day**: all stations are preprocessed in parallel (`ProcessPoolExecutor`) and merged into one
  `Stream`, then a **single** `classify()` call runs on the whole day.
- **Preprocessing** (uniform across years): interpolate→100 Hz, `merge(method=1, fill_value=0)`,
  pad/trim 10 s, demean+taper, bandpass 1–40 Hz (corners=4, zerophase=False).
- **Thresholds**: `P_threshold = S_threshold = 0.2`.
- **Output**: `picks_<year>.<doy>.csv` with columns `station, phase, peak_time, probability`, where
  `station` is the **canonical `NET.STA`** (e.g. `KG.BBK`) normalized from the pick `trace_id`.
- **Resume**: days whose CSV already exists are skipped (`--no-skip-existing` to recompute).

## 2. Association — `association.py`

- **Tool**: PyOcto `OctoAssociator.from_area`.
- **Region/params** (in `config.REGION`): lat (34.5, 37.0), lon (128.5, 130.0), depth (0, 40) km,
  `time_before=300`, `n_picks=4`, `n_p_picks=2`, `n_s_picks=2`, `n_p_and_s_picks=1`.
- **Velocity model**: layered from `KS_KG/velocity_model/kim1983.csv`.
- **Stations**: coordinates from `KS_KG/station_table/station_update.dat`; the **network (KS/KG) is
  taken from the picks themselves** (not a hardcoded count).
- **Output**: `pyocto_kim1983_<year>.csv` (events) + `pyocto_assignment_kim1983_<year>.csv`
  (pick→event), plus `stations_<year>.csv` under the model's `station_table/`.

## 3. PHS file — `make_phs.py`

- Converts PyOcto events+assignments into a **HYPO71 fixed-width `.phs`** file (P→`HHZ`/`IP`,
  S→`HHN`/`ES`), one block per event. Layout ported verbatim from the original notebook.
- **Output**: `models/<model>/HypoInv/PHS/UF<year>.phs`.

## 4. Absolute location — `run_hypoinverse.py`

- **Tool**: external `hyp1.40` binary (must be on `PATH`).
- Generates the HYPOINVERSE control on the fly (from the `UF<year>.sh` template) parameterized by
  year + `--velmodel`; runs in `models/<model>/HypoInv/` where `STA/` and the `*.crh` crustal-model
  files are symlinked in.
- **Output**: `models/<model>/HypoInv/<velmodel>/UF<year>.{sum,prt,arc}` (`.sum` = located catalog).

## Orchestrator — `run_pipeline.py`

Runs stages 1→4 in order for one or more years (`--years 2010-2024`), with `--stage-from` to resume
mid-chain. Continues on per-year errors and prints a summary.

## Relative relocation — HypoDD *(planned)*

Will consume the HYPOINVERSE catalog + differential times into `models/<model>/hypodd/`. A reference
implementation exists at `/home/msseo/works/relocDD-py/`.
