# 2016 Gyeongju / Ulsan-Fault ML-picker comparison — project guide

Guidance for Claude Code working in `Gyeongju_catalog/detection_test/`. This is a **controlled comparison
of four ML pickers** on year 2016, run through one identical relocation pipeline. Parent project guide:
`../../CLAUDE.md` (the long-term UF catalog). This sub-project shares that pipeline but isolates the
**picker as the only variable**.

## Goal

Systematically compare four ML pickers on 2016 UF-box seismicity — **detection → association → absolute
location → QC → HypoDD dt.ct/dt.cc** — with every stage after picking held identical:

| key | picker | weights |
|---|---|---|
| **phasenet_plus** (PN+) | PhaseNet+ (EQNet) | Buan-tuned build |
| **original** | PhaseNet-original | NCEDC (Zhu & Beroza 2019) |
| **stead** | PhaseNet-STEAD | STEAD-retrained |
| **eqt** | EQTransformer-STEAD | Mousavi 2020 |

**Consistent P=S=0.2 threshold on all four** (`gj_config.PICK_PROB`). Identical downstream: PyOcto 4/2/2
gate, HypoInverse kim2011, `uf_cluster` QC, HypoDD kim2011/ISTART=2/adaptive-damping, xcorr interp_hz=1000.

## Pipeline (9 stages; the picker is the only variable)

```
0 Detection      lib/run_seisbench_picker.py (--model) / run_pnplus_month.py   -> picks/*.parquet (prob>=0.2)
1 Association     lib/associate_daily.py (--picker --month)                     -> catalogs/catalog_*_pyocto.csv + assign
2 SAC store       reloc_2016_uf/build_sac_and_pyocto.py                         -> event_sac/<event_idx>/*.sac (NATIVE rate)
3 UF-box catalog  reloc_2016_uf/build_catalog_kma.py                            -> catalog_kma.csv (KST), members.txt
4 Absolute loc    pipeline hypoinverse (kim2011), FULL run                      -> 1.HypoInv/kim2011/<slug>.sum
5 QC (filter)     reloc_2016_uf/build_qc_catalog.py (uf_cluster.QC)             -> members_qc.txt, catalog_kma_qc.csv
6 Re-reference    pipeline rereference                                          -> restamp SAC origins from .sum
7 ph2dt           pipeline ph2dt                                                -> event.dat, dt.ct
8 Cross-corr      pipeline xcorr/dtcc (pq-gpu, interp_hz=1000)                  -> dt.cc_0.7_combined
9 HypoDD          reloc_2016_uf/run_hypodd_kim2011_istart2.py                   -> hypoDD.reloc
```

Driver `reloc_2016_uf/run_picker_reloc.py --picker <p> --through dtcc` chains stages 2–9.
PN+ reuses the finished `reloc_2016_uf/` (slug `uf_2016`); others use `reloc_2016_uf_<p>/` (slug `uf_2016_<p>`).

**Quality gates.** PyOcto: total picks≥4, P≥2, S≥2, ≥1 P+S station (permissive by design).
HypoInverse QC (`uf_cluster.QC`, the real gate): `erh<5 ∧ erz<5 ∧ gap<270 ∧ num>5 ∧ rms<1.0`.
"dt.cc-resolved" = event keeping ≥1 cross-correlation link (`nccp+nccs>0`), the highest-precision subset.

## ⚠️ CRITICAL INVARIANTS (violating these caused a silent origin-corruption bug — 2026-07)

Full write-up in `reloc_2016_uf/PIPELINE.md`; study PDF in `reloc_2016_uf/study_guide.pdf`.

1. **One HypoInverse solution, computed once, reused everywhere.** QC only *selects* a subset; it must
   NEVER re-run HypoInverse. The QC-subset relocation reuses the full-run `.sum`/`.arc`
   (`run_picker_reloc.py::inject_full_hypoinverse`). **The old bug re-ran HypoInverse on the QC subset.**
2. **Pick provenance is by `event_idx`, never by timestamp.** An event's picks live in
   `event_sac/<event_idx>/<event_idx>_picks.csv`. A second-resolution `event_id` key grabbed an adjacent
   event's raw picks → wrong picks → wrong origins. (Same class as the parent guide's pyocto-vs-timewindow note.)
3. **Origins flow into dt.cc, not just start locations.** `rereference` stamps SAC origins from a `.sum`;
   xcorr stores `dt = (t1+shift−ot1)−(t2−ot2)` — it SUBTRACTS the origins. A wrong `.sum` corrupts dt.cc
   *values*, so the fix must re-run rereference→xcorr, not just patch event.dat.
4. **`members.txt` row order defines the cuspid** (200000+row) tying `.sum`/`.arc`/`event.dat`/`dt.cc`.
   QC cuspid = 200000+qc_row; full cuspid = 200000+full_row (pipeline `sumio.read_sum` exposes `id`;
   `uf_cluster.read_sum` does NOT).
5. **catalog_kma.csv is fed in KST and round-trips to UTC.** The pipeline hard-codes `kst_offset_hours=9`
   (`origin_utc = catalog_KST − 9h`). `build_catalog_kma.py` adds +9h ONLY so the pipeline subtracts it
   back. `catalog_kma.csv` is a FORMAT label — **we do NOT use the KMA catalog**. Feeding UTC = 9h shift.

### The 2026-07 bug (fixed) and how to detect a recurrence
Symptom: `event.sel`/`event.dat` showed `ERZ>5 km` events that had passed QC — because those errors came
from a redundant QC HypoInverse RE-RUN, not the full run QC gated on. Fix: `fix_qc_rerun_bug.py` (repair)
+ `run_picker_reloc.py::inject_full_hypoinverse` (permanent). Backups at `1.HypoInv/kim2011.rerun_backup`.
**Recurrence check:** an event's `event.dat` location must match the full-run `.sum` for that event; a SAC
`(starttime−b)` must equal the full-run origin (a ~1s OR 9h mismatch = wrong `.sum` / broken KST round-trip).

## Current results (corrected, 2026-07-23)

Corrected 4-picker dt.cc comparison (consistent 0.2 threshold). **Headline: PN+ yields the most
cross-correlation-resolved events despite not picking the most — pick quantity ≠ located quality.**

| picker | picks | region ev | UF-box | QC | dt.cc | **cc-resolved** | rel-err (m) |
|---|---|---|---|---|---|---|---|
| **PN+** | 1.17M | 28,168 | 3,867 | 596 | 512 | **255** | 0.42 |
| PN-original | 1.79M | 31,762 | 4,117 | 574 | 447 | 189 | 0.50 |
| PN-STEAD | 350k | 6,472 | 398 | 188 | 172 | 103 | 0.36 |
| EQT-STEAD | 183k | 6,540 | 424 | 160 | 144 | 92 | 0.42 |

PN-original picks the MOST but yields FEWER cc-resolved than PN+ (noisy picking → association liability).
Its post-fix drop (212→189 cc-res) is the fix working: buggy origins (median 0.13s, 121 events >0.5s) let
inconsistent links survive HypoDD's WDCC residual cut at a wrong offset-driven location; correcting exposes
and rejects them (rel-err improved 0.72→0.50m). PN+ barely moved (origin errors only 0.07s median). 0
air-quakes for all four.

## Reproduce (run in order; all from detection_test/)

```
# per month (mm = 01..12); pickers <p> = original|stead|eqt (PN+ already done)
python lib/build_stations.py       --month 2016-<mm>
python lib/run_seisbench_picker.py --model  <p> --month 2016-<mm>      # PN+: run_pnplus_month.py
python lib/associate_daily.py      --picker <p> --month 2016-<mm>
# once per picker (chains stages 2-9; xcorr ~6h for dense pickers, in pq-gpu env)
python reloc_2016_uf/run_picker_reloc.py --picker <p> --through dtcc   # or --through hypoinverse (fast, to QC)
# refresh comparison
jupyter nbconvert --to notebook --execute --inplace reloc_2016_uf/10.Picker_comparison_2016.ipynb
```

Notebooks: `09.UF2016_reloccmp` (3-way reloc cmp), `10.Picker_comparison_2016` (funnel + maps + temporal +
station maps), `12.PNplus_dtcc_only_record_sections` (PN+-only events vs others, HypoInverse geometry).

## Year-generality (2010–2025) — DONE (orchestration `--year`, 2026-07)

The whole chain is now **year-general**. All year-dependent names go through `reloc_2016_uf/year_paths.py`;
`--year` (default 2016) resolves to the EXACT existing 2016 paths/slugs, so 2016 work is byte-untouched
(verified). Run any year whose stage 0-1 inputs exist:

```
python reloc_2016_uf/preflight_year.py --year <Y>                         # what inputs exist / what to run
python reloc_2016_uf/run_picker_reloc.py --picker <p> --year <Y> --through dtcc --clean-cache
python reloc_2016_uf/fix_qc_rerun_bug.py --year <Y> --apply               # repair driver, also --year
```

**Long-run flags (recommended for the 16-year sweep):**
- `--clean-cache` — after dt.cc completes, delete that run's `wf_interp_cache` (a derived xcorr speed-cache,
  ~tens of GB/picker; it hit ~90 GB for one dense picker-year). ESSENTIAL over 16 years or it accumulates TBs.
- `--link-only` — (re)publish result symlinks for a completed run without re-running anything.

**Results are symlinked back into the working dir** — every run auto-creates `reloc_<Y>_uf[_<p>]/results/` with
symlinks to the external `runs/<slug>/` outputs (`hypoDD.reloc.dtcc`, `hypoDD.reloc.dtct`, `HypoInv.full.sum`,
`HypoInv.qc.sum`, `dt.cc.02`, `run.dir`), so all results are reachable from under `detection_test/` even though
the heavy data lives in `15.PocketQuake/.../runs/`.

Naming: `reloc_<year>_uf[_<p>]/`, slug `uf_<year>[_<p>]`, `stations_<year>.csv`, `catalog_<p>_<year>_<mm>_pyocto.csv`.

**Robustness (already in the lower layers, no code change needed):**
- **Network change** (KS/KG → +NS 2017 → +GJ temp 2016): stations DISCOVERED from StationXML + on-disk
  `coverage` per month (`lib/build_stations.py --month`), not hard-listed.
- **Sampling-rate change** (100/200/1000 Hz): native-rate SAC store (`SAC_TARGET_HZ=None`) + interpolate at
  correlation time (the GJ mixed-rate fix).
- Detection/association are `--month YYYY-MM` parameterized (run on 2014/2016/2021 already).

**Only stage 0-1 inputs need generating per new year** (detection + association for all 12 months — the heavy
compute); the orchestration (stages 2-9) then runs unchanged. Real per-year effects (network-gap months →
smaller sets; sparse early years → larger gaps/fewer QC survivors; O(N²) xcorr in dense aftershock years) are
the pipeline HONESTLY reporting what each network resolves — not ad hoc tuning. Velocity model, QC thresholds,
4/2/2 gate are epoch-invariant so catalogs stay comparable across years.

## Key files

- `reloc_2016_uf/PIPELINE.md` — invariants, stage-by-stage, KST/KMA note, provenance checks
- `reloc_2016_uf/study_guide.{pdf,tex}` — self-study guide (pipeline + reproduction + comparison + year-generality)
- `reloc_2016_uf/run_picker_reloc.py` — the per-picker/per-year driver (stages 2–9; `--year`; `inject_full_hypoinverse`)
- `reloc_2016_uf/year_paths.py` — centralized year-dependent paths/slugs (`--year` back-compat with 2016); self-test on run
- `reloc_2016_uf/preflight_year.py` — reports a year's stage 0-1 input readiness + the exact commands to run
- `reloc_2016_uf/fix_qc_rerun_bug.py` — the repair driver (dry-run default; `--apply`; `--year`)
- `reloc_2016_uf/build_qc_catalog.py` — `uf_cluster` QC (drop-robust .sum↔members alignment)
- `lib/{build_stations,run_seisbench_picker,associate_daily}.py` — year/network/rate-general lower layers
- `gj_config.py` — single disclosed source for PICK_PROB, GATE, ZLIM, velocity, tolerances

## Conventions (inherited)

- USER RUNS ALL SCRIPTS — deliver transparent, params-at-top, dry-run-default scripts; verify scratch copies only.
- Compile .tex with **tectonic** in the `tex` conda env (not pdflatex).
- Sentence-case titles/labels; Helvetica for plot text; every analysis notebook ends with a computed summary.
- PyGMT for spatial maps (never SeismoStats matplotlib default); decimal degrees, scale bar, plain frame.
