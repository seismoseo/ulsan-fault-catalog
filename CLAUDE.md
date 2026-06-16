# Ulsan Fault Seismicity Catalog — project guide

Guidance for Claude Code working in this repository.

## Goal

Build a **long-term (2010–present) earthquake catalog for the Ulsan Fault region** using
AI phase pickers (SeisBench) on continuous waveforms from a station network that **grows over
time**. The aim is an automated-but-rigorous pipeline that can be re-run as new data and new
picker models become available.

## Pipeline

```
Detection (PhaseNet)  ->  Association (PyOcto, strict)  ->  Pick augmentation  ->  Absolute location (HYPOINVERSE)  ->  Local magnitudes (ML)  ->  Relative relocation (HypoDD, future)
   picks CSV               events + assignments              augmented assignments    UF<year>.{sum,arc}              Heo 2024 + Sheen 2018 ML       (not yet implemented)
```

**Stages** (from `models/pipeline/run_pipeline.py`): `detection` → `association` → `augment` → `phs` → `locate`.

**PyOcto assignment (after augmentation) is the source of truth for which (station, phase) tuples belong to each event.**
The legacy time-window pick dump in `HypoInv/event_waveforms_*/*_picks.csv` was a recipe for
mis-assignment when two events fall < 60 s apart at the same station (the earlier event's pick
"won" by chronological earliest, leaking into the later event's SAC headers). The production
SAC-header writer (`event_sac_export.export_event(..., pyocto_root=...)`, the audit notebook
(`local_magnitudes/04.Catalog_quality_audit.ipynb`), and any downstream that reads `SAC.a`/`.t0`
should consume `models/phasenet_plus/pyocto/pyocto_assignment_kim1983_{year}.csv`, NOT the
time-window CSV.

The automated implementation lives in **`KS_KG/models/pipeline/`** (shared module + thin CLIs).
See `docs/how-to-run.md` for commands and `docs/pipeline.md` for stage details.

## Directory map

```
KS_KG/
  continuous/            raw waveforms (NOT in git; ~3.8 TB)
  station_table/         station metadata (stations.csv, station_update.dat)
  velocity_model/        kim1983.csv (PyOcto layered model)
  detection_location/    per-year notebooks 01/02/03 — the "stead" REFERENCE run
  picks/ pyocto/ HypoInv/ existing stead outputs (data mostly NOT in git)
    HypoInv/uf_cluster.py              spatial/temporal blast decluster + map helpers
    HypoInv/uf_waveform_similarity.py  WAVEFORM-similarity blast screening (KG.HDB; 2026-06-07)
    HypoInv/04_waveform_similarity_hdb_phasenet_plus.ipynb   its controlled notebook
  models/                picker-model dimension (this project's working area)
    stead/               symlinks to the reference run (read-only view; not in git)
    original/            PhaseNet "original" run (copies of notebooks, switched)
    pipeline/            *** automated pipeline: config.py, core.py, *.py CLIs ***
    build_original_tree.py
NS/                      second network, ~3.3 TB, mostly post-2018/19 — DEFERRED
tuto_material/           tutorials / third-party (NOT in git)
docs/                    documentation
tools/                   git helpers (nbstrip.py)
```

## Two independent "model" dimensions — keep them straight

- **Picker model** (`--model`): `stead` / `original` (SeisBench PhaseNet) or `phasenet_plus` (EQNet
  PhaseNet+, in-process; needs a local EQNet clone at `config.EQNET_DIR`). Top of `models/`.
- **Velocity model** (`--velmodel`): the crustal model — `kim1983` vs `kim2011`. Used by PyOcto/HYPOINVERSE.

**Preprocessing**: PhaseNet-style pickers want **raw** (demeaned) data + their own normalization — no
bandpass. (The legacy stead/original detection notebooks applied a 1–40 Hz bandpass; the `pipeline/`
SeisBench path and the `phasenet_plus` backend feed minimally-processed data.)

## Conventions / rules

- **Do not edit the `stead` reference run** (`KS_KG/detection_location/**`, `KS_KG/{picks,pyocto,HypoInv}`).
  It is the baseline to compare against. `models/stead/` only symlinks it (the scripts refuse
  `--model stead` writes unless `--force`).
- **New work goes under `KS_KG/models/`.** The `original` notebooks and `pipeline/` code are editable.
- **Canonical pick id**: detection writes `station = "NET.STA"` (e.g. `KG.BBK`); association derives the
  network from it. Do not reintroduce the old hardcoded `["KS"]*N + ["KG"]*…` split.
- **Non-destructive scaffolding**: `models/build_original_tree.py` only ever writes under `models/`.
- Defaults (paths, thresholds, region) live in `KS_KG/models/pipeline/config.py` — change them there.

## How to run (quick)

```bash
cd KS_KG/models/pipeline
# SHARED 64-core box: ALWAYS pin cores with taskset + set OMP_NUM_THREADS.
OMP_NUM_THREADS=1  taskset -c 0-7  python run_pipeline.py --model original      --years 2010-2024
OMP_NUM_THREADS=16 taskset -c 8-23 python run_pipeline.py --model phasenet_plus --years 2010-2024
python detection.py --model original --year 2024 --days 1-5    # one stage / slice
```
Detection is idempotent (skips days whose picks already exist). Full details: `docs/how-to-run.md`.

### Performance & shared-server CPU (see `docs/performance-notes.md`)

This is a **shared 64-core server** — keep the footprint polite. Detection sizes its preprocessing
pool and `torch` threads from the process CPU **affinity** (`os.sched_getaffinity`), capped by
`config.MAX_CORES` (24), so launching under **`taskset -c <cores>`** auto-scopes the whole job
(and all worker threads) to that core budget. Without pinning, PhaseNet+ grabbed ~49 cores / 193
threads and starved everything. Inference is **GPU-preferred** (warns loudly, never silently falls
back to CPU). Preprocessing uses **one reused `forkserver` `ProcessPoolExecutor`** per year
(created before the model loads, so workers are lean) — not the old per-day pool that forked the
23 GB CUDA parent 5,475×.

**PyOcto association** is separately capped via `REGION_STRICT["n_threads"]=16` (default would
use all available cores). To live-restrict a running pipeline, `taskset --pid --cpu-list 0-15`
on every thread of the launcher process works for the in-flight years; the config change keeps
subsequent years polite from the start.

## Environment

- miniforge Python; `seisbench`, `pyocto`, `obspy`, `torch` (+ CUDA GPU). See `requirements.txt` / `environment.yml`.
- **HYPOINVERSE** needs the external `hyp1.40` binary on `PATH` (not pip-installable).

## Version control

- GitHub: `seismoseo/ulsan-fault-catalog` (public). The repo tracks **code, docs, and small reference
  metadata only** (station tables, velocity model, `*.crh`, `HypoInv/STA/*.sta`).
- **Not tracked** (gitignored — verify with `git status` before committing): waveforms
  (`KS_KG/continuous/`, `NS/`); **all notebooks** (the `stead` per-year run, the generated
  `models/original/` run, the 424 MB `01.PhaseNet_detection_test.ipynb`); original run scripts
  (`HypoInv/UF*.sh`, `HypoInv/STA/hypoinverse/`); large outputs (`picks/`, `pyocto/`, HYPOINVERSE
  `*.prt/*.arc/*.sum`); `tuto_material/`.
- A git clean filter (`tools/nbstrip.py`, enabled once via `bash tools/setup-git-filters.sh`) strips
  notebook outputs **if** a notebook is ever intentionally added — kept as a safety net.

## Waveform-similarity blast screening (KS_KG/HypoInv, 2026-06-07)

Second, **waveform-feature** pass to catch quarry blasts the spatial/temporal decluster
(`uf_cluster.py`) missed. Premise: blasts from one pit repeat the same source→path, so at a
fixed station they share near-identical waveforms; tectonic events don't (repeaters/aftershocks
correlate too but separate by hour-of-day + location).

- **Files** (beside `uf_cluster.py`): `KS_KG/HypoInv/uf_waveform_similarity.py` (module, same
  style; reuses `uf_cluster` KST/Rayleigh/maps/`SUBREGION`) + controlled notebook
  `KS_KG/HypoInv/04_waveform_similarity_hdb_phasenet_plus.ipynb` (PARAMS cell, run top-to-bottom).
- **Data**: `KS_KG/HypoInv/event_waveforms_ulsanfault/` = 2797 dirs `YYYYMMDDHHMMSS/` with
  `{ev}.{NET}.{STA}.{CHA}.sac` (100 Hz, 120 s; SAC `a`=P, `t0`=S, `o`=origin, `stla/stlo`) +
  `{ev}_picks.csv`. **KG.HDB.HHZ** covers 2771/2797; ~284 have the trace but no HDB pick.
  Join to `catalog_phasenet_plus_2010_2024_blastclean.csv` by `time`→dir name (2729 join) for
  hypocentres; known quarry centroids from `cluster_summary_phasenet_plus_2010_2024.csv`.
- **Method**: common station **KG.HDB / HHZ**; align on P — **two deterministic sources only**:
  `pick` (`{ev}_picks.csv`) else `fallback` (`origin + median P-traveltime`). (The SAC-`a`-header
  branch is defensive dead code: picks.csv + SAC a-marks come from the same PhaseNet run, so a P is
  in both or neither — `header` source = **0 events** on this data, verified; not random.) **Picked
  events keep P at t=0, only the ~284 fallbacks are xcorr-aligned to the picked stack**; bandpass
  + **SHORT P-window `[P-0.5,+7.5]s` (NEVER the 120 s record)** + L2-norm; bands 1-10/2-8/4-12/
  5-15 Hz; N×N **max-lag CC** similarity matrix (small `MAXLAG` since aligned) → **Ward** on
  (1-CC) → clustered heatmap + dendrogram + per-cluster gathers + PyGMT map + **blast-likeness
  evidence** (`mean_cc`, `spread_km`, `daytime_frac`, `rayleigh_p`, `peak_hour`).
- **Read it**: tight (high `mean_cc`) + **daytime-concentrated** + compact + non-uniform hour =
  blast candidate (`blast_like` flag); tight but **night/uniform** = tectonic repeater.
- **Notebook views (per family, all colour-consistent + a station-context PyGMT map)**: 4-band
  square similarity matrices; average-linkage dendrogram (cut at `CC_THRESHOLD`); per-event gathers
  in **filtered / raw / 1 Hz-highpass / hour-of-day(HSV)** flavours; per-cluster **stacks**; and a
  **per-event spectrogram gather** (`stft`, full-window, 0.5–40 Hz, contiguous strips, HSV hour tab
  per event). `event_hours()` derives KST hour from the dir name (every event; no catalog needed).
  §6 has both a `top=20` and a `top=None` (**all families**) subregion map; §7 adds
  `plot_blast_hour_histograms` — a per-`blast_like`-family **hour-of-day histogram grid** (KST,
  daytime shaded, `peak_hour`/`rayleigh_p` annotated). `plot_cluster_sections`: traces ordered
  **chronologically within each family** (top=earliest→bottom=latest), each trace right-annotated with
  its **event origin time in UTC** (`annotate_utc=True`, pinned just right of the axes). `cluster_colors`
  now gives the first 20 (size-ranked) families the qualitative `tab20` palette so the plotted top-N
  are visually **distinct** (old all-`hsv` made a top-N subset collapse to one hue band); §7 blast
  figures use a dedicated `BLAST_COLORS = cluster_colors(blast_ids)` palette. The grouped
  `plot_cluster_sections` caps output (`max_clusters`/`max_per_cluster`/`show_singletons`) so it shows
  only a few hundred of the ~2.7k events; to see them **all**: `plot_clusters_individually` = each
  family as its **own full-size chronological gather** (separate figures, **constant per-trace height**
  regardless of family size via `head_in`/`min_fig_h`, UTC origins legible) — the default §4 view;
  `plot_cluster_grid` = the compact subplot-grid alternative; `plot_all_chronological` = a literal
  single all-events time stack (tall, ~163 in for full catalog; best on a one-year `kept`).
  Similarity matrices (§2) outline each identified family with a white box (`outline_clusters`).
- **Per-cluster space-time notebook** `05_cluster_spacetime_{COMP}_phasenet_plus.ipynb` (built by
  `build_seq_nb.py`): one composite per family — chronological gather (left) + **fixed-extent**
  epicentre map coloured by origin year (right top) + cumulative-N(t) curve (right bottom)
  (`cluster_spacetime_fig`/`plot_clusters_spacetime`, `spacetime_region` for the shared extent).
  The inset map is **matplotlib** (`uf_cluster.coast_mpl`/`plot_faults_mpl`) — 99 PyGMT renders timed
  out (~63 s in matplotlib vs >3000 s); PyGMT stays for the §6 publication maps.
- **KG.HDB coverage**: of **2796** `event_waveforms_ulsanfault` dirs, **2770 have HHZ** (26 missing,
  0.93 %), **2773 have any KG.HDB component**, and only **23 (0.82 %) have no KG.HDB at all** — the
  events the screen cannot see at this station. ~99 % coverage is why KG.HDB is the common station.
- **Status — FULL PERIOD RUN DONE (2026-06-07), still exploratory / NO removal yet.** 2010→full:
  `YEARS=None`, 2770 events → **2716 kept** (2446 pick + 270 xcorr-fallback aligned; 2651 join
  blastclean). CC≥0.6 average linkage → **99 families ≥4** (+1159 singletons). Evidence flags
  **7 `blast_like` families = 66 still-remaining quarry-blast candidate events** (tight `mean_cc`
  0.69–0.80, **daytime_frac 0.6–1.0, peak 12–15 KST**, compact ≤3.7 km) — two pockets ~129.28°E
  (W) and ~129.40–43°E (near KG.HDB). Timing: make_bands 118 s (cached), CC 2 s, ward 0 s. Caches
  keyed by events-hash (`feat_…_n{N}_{md5}.npz`); gathers cap to top `MAX_CLUSTERS_PLOT` families.
- **Three-component notebooks** (same analysis, `COMP` swap via `build_wf_nb.py {HHZ|HHN|HHE}`):
  `04_waveform_similarity_hdb_{HHZ,HHN,HHE}_phasenet_plus.ipynb`. Horizontals carry the same
  P(`a`)/S(`t0`) headers + npts/timing, so it's a clean parameter swap (alignment is from the
  station pick, component-independent). **Per-component KG.HDB coverage HHZ 2770 / HHN 2772 / HHE
  2771** — the differences are **4 events with incomplete component files on disk** (dropout):
  `20100531170439` (HHE,HHN; no HHZ), `20131113192434` (HHN only), `20140104222332` (HHE,HHN; no
  HHZ), `20140911190328` (HHZ only) → HHN = 2770+3−1, HHE = 2770+2−1. ~0.1 %, no material effect;
  caches/notebooks are per-component (cache tag has comp + events-hash).
- **Repeater / anti-repeater notebooks (2026-06-09).** Same `uf_waveform_similarity.py` infra,
  positive- and negative-CC counterparts at KG.HDB.
  - `06_anti_repeaters_KGHDB_{HHZ,HHN,HHE}_phasenet_plus.ipynb` (`build_antirepeater_nb.py`): signed
    CC (`signed_similarity` → cc_pos/cc_neg/cc_ext/cc_lag0), hunts near-(−1) polarity-reversed pairs.
    **NULL result** — half-period degeneracy (every strong cc_neg also has high cc_pos) + cross-
    component inconsistency. `plot_antipair_compare` aligns each hypothesis at its **own** best lag
    (repeater-fit vs anti-fit), not lag 0 (the lag-0 bias was a real bug, fixed).
  - `07_repeaters_KGHDB_{COMP}[_1-25Hz][_single]_phasenet_plus.ipynb`
    (`build_repeater_nb.py [COMP] [BAND] [CC] [LINKAGE]`): classic repeating-earthquake families.
    `repeater_table` is **magnitude-free** (catalog ML is preliminary). `plot_family_sections`
    (per-family record sections, S bars, 1 Hz-HP variant), `plot_family_recurrence` (one fig/family
    + cumulative-N staircase), `plot_repeater_sequences` (full-width timeline, **2016 Gyeongju**
    mainshock marked; histogram removed), `map_cluster_links` (UF-subregion, `top=15`). Built for
    band ∈ {1-10, 1-25} Hz × linkage ∈ {**average** UPGMA, **single** = friends-of-friends chaining}.
  - `make_bands` now builds **missing bands incrementally** (cache key = events+window, NOT the band
    list — a newly-requested band like 1-25 Hz was silently absent before → KeyError/stale).

- **Rough de-blasted catalog — the removal step, DONE (2026-06-10).** `08_deblasted_catalog_KGHDB_
  HHZ_phasenet_plus.ipynb` (`build_deblast_nb.py`). **Blast events are severely mislocated**, so the
  flag is **location-free** — only **waveform similarity + daytime fraction** (NO depth, spread_km,
  rayleigh_p): `mean_cc ≥ 0.6` AND `daytime_frac == 1.0` over `DAY=(6,19)` KST (every member in
  working hours — families are small, one night event disqualifies). → **8 families / 59 events**
  `[71,803,824,837,838,869,1097,1175]`. `NATURAL_OVERRIDE=[1158]` keeps a **deep (~11.5 km) repeating
  natural** cluster out (also dropped by the cut); `BLAST_OVERRIDE` force-includes any obvious blast
  under 1.0. **Product is subregion-scoped**: blastclean 14803 (whole study area) → **UF subregion
  2798 = de-blasted 2741 + blast 57** → `catalog_phasenet_plus_2010_2024_deblasted_rough.csv`. Maps
  (`map_catalog_subregion`, `color_by="hour"|"depth"`, `draw_box`): original / de-blasted / blast over
  the **exact** subregion (no blue box), coloured by **hour-of-day** only (depth dropped — mislocated).
  §4 per-cluster waveforms in **two filters (1-10 Hz + 1 Hz high-pass), every member, no omission**;
  §5 per-family hour histograms. Run in the project default (`python` in HypoInv: obspy+scipy+pygmt).

## Status & next steps (2026-06-03)

- **PhaseNet+ strict-PyOcto + augmentation full re-run in flight** (years 2010–2024). Years
  complete through 2016 (16,166 events, **+1,453 picks augmented total**, 0 drop-on-tie
  events — safeguards working). Year 2017 in flight. ETA: ~5–6 h to finish.
- **After re-run**: rebuild `catalog_phasenet_plus_2010_2024.csv` + re-blast-clean (cat_dq) →
  re-inject SAC headers if hypocenters shifted → re-run Heo + Sheen bulk-ML
  (`local_magnitudes/02.Compute_ML_all_events.ipynb`) → re-execute Heo + Sheen summary
  notebooks (`03/06`).
- **Lab-meeting catalog (in summary notebooks)** uses the current strict-n_s=3 catalog (no
  augmentation) — augmentation is a quality improvement that lands cleanly after the re-run.
- **KMA comparison in notebooks**: per-event match (TIME_TOL_S=30, DIST_TOL_KM=10) added to
  `04_subregion_seismicity_phasenet_plus.ipynb` (subregion) and
  `catalog_summary_phasenet_plus.ipynb` (entire region, §5b + §7). Both use **`cat_dq`
  (blast-removed)** for apples-to-apples KMA comparison — entire-region: 5,638 KMA / 14,896
  PN+, 4,853 matched (86% of KMA), 785 KMA-only (mostly sub-1.5 ML detection floor + 16
  events ≥ M3 to investigate).
- **Hour-of-day plots in `04_subregion`**: per-year histograms + per-year spatial maps
  colored by hour-of-day with matplotlib `hsv` cyclic colormap (matches the PyGMT
  `uf.hour_map` cyclic). Also added the matching KMA per-year hour-of-day map.

## Earlier status (2026-05-26)

- **stead** catalog complete (2010–2024 located `kim2011/UF<year>.sum`). Summary in
  `KS_KG/HypoInv/catalog_summary.ipynb` (model-parameterized; writes `catalog_<model>_2010_2024.csv`;
  includes maps, depth sections, cumulative/rate, network growth, KMA comparison, and hour-of-day
  (KST) diurnal + spatial-variation analysis for blast vs tectonic discrimination).
- **Detection performance fixed** (was a ~35-day ETA): reused forkserver pool, lossless handling of
  fragmented station-days, GPU-preferred inference, polite `taskset` CPU budget. Root-cause writeup
  in `docs/performance-notes.md`. **original** + **phasenet_plus** 2010–2024 re-runs running pinned
  to ~24 cores (`models/<model>/run_2010_2024.log`).
- **PhaseNet+ inspection**: `core.annotate_phasenet_plus(year, day, station, t0, t1)` returns the
  per-sample P/S/noise probability, first-motion polarity, and single-station event-detection traces
  (the PhaseNet+ analogue of SeisBench `annotate`); used by the rebuilt
  `models/pipeline/notebooks/phasenet_plus_test.ipynb`.
- **Post-location analysis** (`KS_KG/HypoInv/uf_cluster.py` + notebooks `03_blast_decluster_hdbscan`,
  `04_subregion_seismicity`, `05_error_ellipses`; see `docs/analysis.md`): 3D HDBSCAN clustering with
  hour-of-day **quarry-blast discrimination** (writes a declustered catalog), an **east-of-fault subregion**
  long-term-seismicity study, and **95% HYPOINVERSE error ellipses** parsed from the `.prt` covariance.
  `uf_cluster.py` is tracked; the notebooks + their CSV/HTML outputs are gitignored.
- **Residual quarry blasts** survive cluster-level declustering as HDBSCAN **noise** (diffuse daytime shots).
  A second-stage **spatial daytime-fraction grid mask** (`grid_blast_stats`/`flag_blast_cells`/
  `decluster_spatial`/`decluster_full`; cell 0.02°, N≥10, daytime_frac>0.80, Rayleigh p<0.01) removes daytime
  events in flagged "quarry cells" → `catalog_*_blastclean.csv`. Empirically (stead): 22 cells, +302 events
  (295 from noise), 11,065→10,763, daytime frac 0.473→0.458, and **0 subregion events** (east-of-fault zone is
  blast-free). Residual blasts are reported **deep** (~9 km) but **avoid weekends** — `weekend_ratio` =
  (Sat/Sun fraction)÷(2/7) (1.0 = no preference, <1 = weekday-only/blast-like). It is **reported** in the
  cluster + grid tables and is an **optional gate** (`flag_blasts`/`flag_blast_cells`/`decluster_full`/notebook-03
  `WEEKEND_MAX`, default **off** — daytime + Rayleigh only); enable (e.g. <0.7) to also demand weekend-avoidance
  for the deep residual blasts that depth can't catch. Notebook 03 also **maps the final blast-clean catalog**
  (§9c: epicenter + cyclic hour-of-day) and includes a **grid-only-vs-two-step robustness check** (§9d, compared
  by catalog index; the two-step is kept — grid-only on the full catalog raises subregion false-positives).
  Notebook 04 defaults to the blast-clean catalog (`USE_BLASTCLEAN=True`) and adds a wide cumulative-count
  curve (§3b) + **per-year subregion small-multiples** (`uf.annual_maps`, depth-coloured epicenters + density
  normalised **per-year** so quiet years aren't washed out — colorbar = fraction of that year's peak; edge-only
  ticks). Single equal-aspect maps height-match their colorbar via `uf._match_cbar`.
- ~~**#1 gap**: no magnitudes~~ → **DONE**: local magnitudes computed (Wood–Anderson ML, Heo 2024 +
  Sheen 2018) with the `require_pick` detectability gate; FMD / Mc / b-value + temporal evolution in
  `local_magnitudes/` (see the Local-magnitudes section). Heo 2024 is the representative ML.
- Later: 3-picker comparison once re-runs finish; **HypoDD** relative relocation.

## Local magnitudes — Heo 2024 + Sheen 2018 (require_pick fix + strict recompute, 2026-06-16)

`KS_KG/local_magnitudes/ml_pipeline.py` deconvolves each event's response, simulates Wood-Anderson
(sensitivity 2080, Uhrhammer & Collins 1990), measures peak post-P amplitude with SNR ≥ 3, and
converts to ML via two attenuation laws.

**Detectability gate `require_pick=True` (the key bug fix).** Earlier ML had no gate, so a station
with **no P pick** fell back to a 20 s trace-start noise window → meaningless SNR → far unpicked KS/KG
stations (60–100 km) at the ambient/coda amplitude **floor** leaked into the event median. Their
amplitude is flat with distance, so the −logA₀(R) term over-corrected them up to station ML ≈ 1,
**saturating small-event ML and inflating b** (the same bug found + fixed in the 2024 Buan project).
`wood_anderson_amp_mm` / `per_station_ml` / `export_ml_catalog` now skip traces with no detected phase
(picks are written to all 3 components, so a picked station keeps its horizontals, an unpicked one drops).

**Strict per paper, NO station-correction term:**
- **Heo 2024** — **vertical only** (paper calibration), hypocentral R, 17 km ref, +2.0: `ml_heo2024`.
- **Sheen 2018** — all 3 components (geom-mean horizontals + Z), epicentral R, 100 km ref, +3.0: `ml_sheen2018`.
- Bulk driver: **`run_bulk_ml_both.py`** (both scales, strict) → `catalog_…_with_ml_heo.csv` + `…_with_ml_sheen.csv`.

**Heo 2024 is the representative ML** for this vertical-dominated dense micro-earthquake catalogue.

**Result (14,775 events with ML — the SAME set for both scales)**:

| Scale | median ML | Mc (MAXC) | _b_ ± SE | Gyeongju M5.8 |
|---|---|---|---|---|
| **Heo (Z-only)**  | **0.34** | **0.50** | **0.77** | 5.40 (Δ ≈ −0.4) |
| Sheen (3-comp)    | 1.14 | 1.30 | 1.31 | cross-check |

The fix de-saturated the FMD (Heo: median 0.71→0.34, Mc 0.80→0.50, **b 1.03→0.77**); large events are
pick-rich and unchanged. **Heo vs Sheen** (`07`): same events, related by **Sheen ≈ 0.96 + 0.59·Heo** —
the +0.8 level is the **component basis** (horizontal-dominated 3-comp median vs vertical-only), **not**
the distance law (formula ΔML(R) ≈ 0); the **0.59 slope** (Heo's steeper distance term) **sets the
b-value ratio** `b_Sheen = b_Heo / 0.59 ≈ 1.7×`.

The notebook chain:
- `02.Compute_ML_all_events.ipynb` — bulk pass (strict Heo Z-only config; superseded operationally by `run_bulk_ml_both.py`).
- `03.Magnitude_summary.ipynb` — **the Heo summary**: FMD / Mc / b per subregion, size-scaled PyGMT maps,
  and **§11 temporal completeness & b-value evolution** (sliding-window MAXC `Mc(t)` + Aki–Utsu `b(t)` ±
  Shi–Bolt SE, SeismoStats), for the **full catalog and the Ulsan-Fault subregion**.
- `04.Catalog_quality_audit.ipynb` — duplicates / pick consistency / mislocations (**uses PyOcto
  assignments + HypoInverse arc residuals**, NOT the time-window pick CSV — see Gotchas).
- `06.Magnitude_summary_sheen.ipynb` — Sheen 2018 cross-check (FMD, temporal Mc, Gyeongju benchmarks).
- `07.Heo_vs_Sheen_comparison.ipynb` — event-by-event Heo vs Sheen (why they differ; the b-value link).

(Removed in the strict cleanup: notebook `05.Magnitude_summary_corrected.ipynb` and the experimental
trial catalogs `…_heo_corrected / _deduped / _v3_heo_no_corrections / _NO_SNR / _SNR3_legacyfilt`.)

**Station corrections** (`ml_pipeline.estimate_station_corrections` / `apply_station_corrections`) are
**kept but dormant** — the strict published scales use no S term. They were dropped because the S_j is
calibrated against a network consensus that itself drifts across the 2016-09 KG densification, over-
correcting pre-2017 events.

## Catalog quality audit + SAC-export refactor (2026-06-02)

The audit (`local_magnitudes/04.Catalog_quality_audit.ipynb`) and the production SAC-export
(`HypoInv/event_sac_export.py` + `06.Export_event_waveforms_from_continuous.ipynb`) both used
to read the per-event `*_picks.csv` time-window dump. That CSV contains every PhaseNet+ pick
within ±30 s of origin — including the picks PyOcto associated to the NEIGHBOURING event for a
close-in pair. The 2015-11-13 11:04:24 / 11:04:33 pair was the smoking gun: both events' SAC
headers carried event A's BBK P pick because `earliest_per_station_phase` took the chronologically
earliest within the window. Both files are now PyOcto-driven:

- **`event_sac_export.py`** v2: `export_event(..., pyocto_root="...models/phasenet_plus/pyocto")`
  routes through `associate_picks_from_pyocto(event_idx)` → real (station, phase) sets PyOcto
  assigned to that event. Legacy `associate_picks` retained as fallback only with deprecation
  warning. Re-run with `skip_existing=False` to overwrite the buggy SAC headers in place
  (no extra disk).
- **`04.Catalog_quality_audit.ipynb` §2** v2: Jaccard now operates on the **PyOcto-assigned set**
  (the 2015-11-13 pair scores 0.39 → correctly classified `ok`, not `duplicate`).
  Mislocation is replaced by **HypoInverse arc residuals** (parsed from
  `models/phasenet_plus/HypoInv/kim2011/UF{year}.arc`): flagged if `max(|residual|) > 1.0 s`
  (10.2 % of catalog). Wide gap + low qual events get a separate `poorly_constrained` flag
  (27 % of catalog; that's "loose location" not "wrong location").

Confirmed by spot-check:
- 2015-11-13 11:04:24 / 11:04:33: both **ok** (max |residual| = 0.10 s)
- 2016-09-12 11:32:54 (M5.4 mainshock): **ok** (max |residual| = 0.42 s)
- 2017-11-15 06:09:49: **mislocated** (max |residual| = 4.86 s; picks do not fit a single source)

## PyOcto strict + post-PyOcto pick augmentation (2026-06-03)

The "parking lot" PyOcto tightening is now committed as **`config.REGION_STRICT`** and is
opt-in via `association.py --strict`. The strict set rejects the 2017-11-15-style chimera
associations (HypoInverse `max|residual|` up to 4.86 s) by demanding more genuine multi-
station P+S coverage, a tighter residual cap, and a finer octree.

```python
REGION_STRICT = dict(
    n_picks=6, n_p_picks=3, n_s_picks=3, n_p_and_s_picks=2,   # genuine coverage
    pick_match_tolerance=1.0,                                  # primary residual cap
    min_node_size=2.0,        # finer initial octree — fixes wrong-basin convergences
    min_node_size_location=0.5, refinement_iterations=8,       # finer refinement
    min_interevent_time=2.0,                                   # allow real doublets
    n_threads=16,                                              # polite on shared 64-core box
)
```

`min_node_size=2.0` is the load-bearing knob — it fixes the **2013-03-22 13:40:04** case
where PyOcto's default 10 km octree landed at a phantom 36.66°N hypocenter 110 km north
of the true location.

**Why a separate augmentation stage was needed.** PyOcto's streaming associator freezes
candidates at the `n_picks` threshold and does NOT re-scan picks after refinement. Once
PyOcto has 6 close-station picks, it stops looking — so FARTHER stations that fit the
refined hypocenter get orphaned. The tweak alone is insufficient; the augmentation stage
uses PyOcto's now-correct hypocenter to scan back through the daily picks and recover the
orphans (with strict safeguards so it never steals picks from a neighbour event).

**Augmentation module: `models/pipeline/pick_augmentation.py`**. For each PyOcto event, the
direct-ray travel time to every station within `radius_km=100 km` is computed from
`kim1983` (`velocity_at_depth`/`predict_arrival_offset`); any daily pick within
`tolerance_s=1.0 s` of the predicted arrival, on a station not already in the PyOcto set
for that event, is a candidate. `apply_safeguards` enforces: (1) phase-strict matching, (2)
best-match-wins across competing events, (3) drop-on-tie when two candidates are within
`tie_threshold_s=0.2 s` of each other. The output overwrites
`pyocto_assignment_kim1983_<year>.csv` so the downstream PHS/HypoInverse stages consume the
augmented set unchanged.

`run_pipeline.py` runs `augment` between `association` and `phs`. Validated on 2013-03-22
13:40:04: 10 picks / DMIN=42.8 / ERZ=3.9 → 14 picks / DMIN=2.1 / ERZ=0.5.

## HypoInverse QC (in `KS_KG/HypoInv/uf_cluster.py`)

`QC = dict(erh=5.0, erz=5.0, gap=270.0, num=5, rms=1.0)` — the `rms<1.0` cap was added in
the strict-PyOcto branch (chimera events had arc-residual RMS multi-second). Don't add a
per-event `max|residual|` filter on top of this — single-station pick outliers are OK
when the overall RMS is good; the remaining picks still carry the hypocenter (see
`feedback_ulsan_single_pick_outliers`).

## Gotchas

- 2010 & 2013 detection notebooks originally pointed at an outdated `2014_sequence/continuous`
  path; the `models/original` copies are repointed to `KS_KG/continuous`.
- `detection_location/2022/picks/` (stead) has ~670 files (a likely double run) — sanity-check.
- Detection runs all stations into a single per-day GPU `classify()` call; preprocessing is
  parallelized with one reused `forkserver` `ProcessPoolExecutor` (see Performance note above).
- **Each model's `HypoInv/STA` MUST be a symlink to the shared `KS_KG/HypoInv/STA`** (per-year
  `UF<year>_hyp.sta` — picker-independent station metadata). `build_original_tree.py` links it for
  `original`; **phasenet_plus was set up without it**, so HYPOINVERSE rejected every phase as `*** SKIP
  PHASE CARD WITH UNKNOWN STATION` and located ~0 events/year despite ~15k associated. Fixed:
  `core.ensure_sta(model)` (called by `run_hypoinverse_year`) auto-creates the symlink, and a low-located-
  count WARNING now flags such a mismatch instead of silently shipping a 1-event catalog. The symlink is an
  untracked filesystem artifact (like `original`'s).
- **Each model's `HypoInv/<velmodel>/` MUST contain the crustal model `<velmodel>_{p,s}.crh`.** The HYP
  control reads them at the relative path `<velmodel>/<velmodel>_p.crh` (cwd = `models/<model>/HypoInv`).
  For `stead` the `<velmodel>` dir is a symlink to the shared `KS_KG/HypoInv/<velmodel>` (which has the
  `.crh`); for a model with a REAL output dir (`original`/`phasenet_plus`) the picker-independent `.crh`
  must be copied in. **phasenet_plus was set up without them**, so hyp1.40 printed `*** ERROR - CRUST FILE
  DOES NOT EXIST` and **silently located every event on its built-in DEFAULT velocity model** → depths
  pinned at the `ZTR` trial (~10 km), multi-second RMS, diffuse epicenters (the picks/association/`.phs`
  were all correct — only the location step was broken). Fixed: `core.ensure_crh(model, velmodel)` (called
  by `run_hypoinverse_year`) copies the shared `.crh` in, and a **median-RMS > 1 s WARNING** now flags a
  wrong/missing crustal model instead of shipping a bad catalog. *(This invalidated the first
  phasenet_plus catalog; re-run after the fix.)*
- **YSB is fragmented in 2010** (~142 days stored as tens of thousands of ~5 s miniSEED records —
  real continuous data, but obspy `merge()` is ~O(n²) ⇒ ~100 s/day). It is processed **losslessly**
  (`config.MAX_SEGMENTS` only logs a warning; `HARD_MAX_SEGMENTS` is the sole skip for a corrupt
  file). The pre-fix run had silently *lost* YSB on those days. Fragmentation is YSB-/2010-specific.
- **HYPOINVERSE `.prt` errors are 1-σ.** ERH ≈ 1-σ horizontal semi-major and ERZ ≈ √var_Z (verified
  median ratios ≈ 1.0), so a 95% **joint** horizontal error ellipse needs `k = √χ²₂,₀.₉₅ = 2.448×` the
  1-σ axes (depth 95% = 1.96·σ_z). `.prt` covariance fields are fixed-width 8-char (slice, don't split —
  they glue when large); overflow prints `********`→NaN (junk events with 20–99 km errors, all
  QC-excluded so harmless); origin seconds can be negative; longitude carries an `E`/`W` letter. A few
  2023 events lack `.prt` covariance because the filtered `.sum` and `UF2023.prt` are from different runs
  (~0.4 s / ~1 km apart) — left unmatched (≈99.9% coverage). Parser/maps in `uf_cluster.py`.
- **matplotlib maps draw coastlines** via `uf_cluster.coast_mpl`/`coast_mpl_km` (cartopy 0.25 NaturalEarth
  10m, lon/lat or local-km frame) to match the PyGMT `fig.coast` maps. Only the coastline **line** is used —
  the 10m land/ocean polygons aren't cached and would trigger a download (helpers degrade gracefully if the
  cache is missing). Z-order on every map: coast 0.5 < faults 1 < noise 2 (small + translucent) < clusters /
  seismicity 3 < subregion box 4–5, so clusters always render above the noise background.
