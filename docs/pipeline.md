# Pipeline stages

Each stage is a function in [`src/ufpipe/core.py`](../src/ufpipe/core.py),
exposed by a thin CLI. All defaults live in
[`config.py`](../src/ufpipe/config.py). Outputs are written under
`outputs/models/<model>/` (default `--model original`).

```
continuous waveforms (KS_KG/ GJ/ NS_100hz/)
   │  detection.py            PhaseNet / PhaseNet+ (multi-network)
   ▼
models/<model>/detection_location/<year>/picks/picks_<year>.<doy>.csv   (station, phase, peak_time, probability)
   │  association.py          PyOcto, daily-chunked
   ▼
models/<model>/pyocto/pyocto_kim2011_<year>.csv            (events: idx, time, x,y,z, picks, lat, lon, depth)
models/<model>/pyocto/pyocto_assignment_kim2011_<year>.csv (assignments: event_idx, station, phase, time, …)
   │  (augment: orphan-pick rescan updates the assignment in place)
   ▼
models/<model>/HypoInv/PHS/UF<year>.phs                    (HYPO71 fixed-width phase file; make_phs.py)
   │  run_hypoinverse.py      hyp1.40
   ▼
models/<model>/HypoInv/<velmodel>/UF<year>.{sum,prt,arc}   (located catalog)
   │  relocate (stage 6)      reloc_inputs -> driver -> GPU xcorr -> HypoDD v2.1beta
   ▼
detection_test/reloc_<year>_uf[_<model>]/results/          (hypoDD.reloc.dtcc — dt.cc-relocated catalog)
```

## 1. Detection — `detection.py`

- **Tool**: SeisBench `PhaseNet.from_pretrained(<model>)`, GPU if available.
- **Input**: `<archive>/<STA>/<BAND>?.D/*.<year>.<doy>` across **KS/KG/GJ/NS** — each station's archive
  (`KS_KG/`, `GJ/`, `NS_100hz/`) and band come from the per-year multi-network table (`src/ufpipe/stations.py`),
  which keeps only stations that both have a metadata epoch overlapping the year and real data on disk. Mixed
  sampling rates within a station-day (GJ/NS native SAC: 100/200/1000 Hz) are unified before merge.
- **Per day**: all stations are preprocessed in parallel (`ProcessPoolExecutor`) and merged into one
  `Stream`, then a **single** `classify()` call runs on the whole day.
- **Preprocessing** (uniform across years): interpolate→100 Hz, `merge(method=1, fill_value=0)`,
  pad/trim 10 s, demean+taper, bandpass 1–40 Hz (corners=4, zerophase=False).
- **Thresholds**: `P_threshold = S_threshold = 0.2`.
- **Output**: `picks_<year>.<doy>.csv` with columns `station, phase, peak_time, probability`, where
  `station` is the **canonical `NET.STA`** (e.g. `KG.BBK`) normalized from the pick `trace_id`.
- **Resume**: days whose CSV already exists are skipped (`--no-skip-existing` to recompute).
- **PhaseNet+ backend** (`--model phasenet_plus`): routes to EQNet's PhaseNet+ (in-process import of
  `eqnet`; needs a local EQNet clone, `config.EQNET_DIR`). Per station-day it builds a comma-joined E/N/Z
  `data_list`, runs EQNet's `SeismicTraceIterableDataset` (raw demean + internal moving-norm; **no
  bandpass**) + `detect_peaks`/`extract_picks`, and writes the **same canonical pick schema** plus raw
  outputs (polarity, amplitude, single-station event detection) under `phasenet_plus_raw/`. Threshold
  `config.PNPLUS_MIN_PROB=0.3`, optional highpass `config.PNPLUS_HIGHPASS`.

## 2. Association — `association.py` (daily-chunked, all networks)

- **Tool**: PyOcto `OctoAssociator.from_area`, run **one calendar day at a time**.
- **Why daily-chunked**: a whole-year single-pass associate is intractable on the dense ~200-station NS array
  (>>1 h, 12 GB). Associating a ±`ASSOC_OVERLAP_S` (150 s) window per day and keeping only events whose origin
  is in-day (dedup) keeps each solve to seconds and is physically equivalent — local events are seconds long.
- **Region/gate** (`config`): area = `REGION_CENTER` (35.856, 129.224) ± (`ASSOC_LAT_PAD`=1.0, `ASSOC_LON_PAD`
  =1.2)°, depth (0, 30) km, `time_before=300`, gate `ASSOC_GATE` = {n_picks 4, n_p 2, n_s 2, n_ps 1}
  (`--strict` → `ASSOC_GATE_STRICT` = {6, 3, 3, 2}).
- **Velocity model**: kim2011 1-D (`config.KIM2011`), matching the reloc feeder.
- **Stations**: coordinates come from the **multi-network year table** (`src/ufpipe/stations.py`) covering
  **KS/KG/GJ/NS** — so GJ/NS picks associate (KS/KG-only `station_update.dat` is no longer the source).
- **Output** (schema unchanged): `pyocto_kim2011_<year>.csv` (events: `idx,time,x,y,z,picks,latitude,longitude,
  depth`) + `pyocto_assignment_kim2011_<year>.csv` (`event_idx,pick_idx,residual,station,phase,time`), plus
  `stations_<year>.csv` under the model's `station_table/`.

## 3. PHS file — `make_phs.py`

- Converts PyOcto events+assignments into a **HYPO71 fixed-width `.phs`** file (P→`HHZ`/`IP`,
  S→`HHN`/`ES`), one block per event. Layout ported verbatim from the original notebook.
- **Output**: `outputs/models/<model>/HypoInv/PHS/UF<year>.phs`.

## 4. Absolute location — `run_hypoinverse.py`

- **Tool**: external `hyp1.40` binary (must be on `PATH`).
- Generates the HYPOINVERSE control on the fly (from the `UF<year>.sh` template) parameterized by
  year + `--velmodel`; runs in `outputs/models/<model>/HypoInv/` where `STA/` and the `*.crh` crustal-model
  files are symlinked in.
- **Output**: `outputs/models/<model>/HypoInv/<velmodel>/UF<year>.{sum,prt,arc}` (`.sum` = located catalog).

## Orchestrator — `run_pipeline.py`

Runs all 6 stages (`detection → association → augment → phs → locate → relocate`) in order for one or
more years (`--years 2010-2024`), with `--stage-from` to resume mid-chain. Continues on per-year errors
and prints a summary.

## 6. Relative relocation — HypoDD dt.ct + dt.cc (implemented, self-fed)

`ufpipe.relocate` builds the reloc inputs (event-idx SAC store + pyocto tables + multi-network station
table) from ufpipe's own per-year association via `src/ufpipe/reloc_inputs.py`, then hands off to the
validated driver `detection_test/reloc_2016_uf/run_picker_reloc.py --skip-build` (scaffold → HYPOINVERSE →
QC → rereference → GPU xcorr → HypoDD v2.1beta; external 15.PocketQuake engine). Results are symlinked
under `detection_test/reloc_<year>_uf[_<model>]/results/`. See the reference manual §7.
