# Ulsan Fault Seismicity Catalog — project guide

Guidance for Claude Code working in this repository.

## Goal

Build a **long-term (2010–present) earthquake catalog for the Ulsan Fault region** using
AI phase pickers (SeisBench) on continuous waveforms from a station network that **grows over
time**. The aim is an automated-but-rigorous pipeline that can be re-run as new data and new
picker models become available.

## Pipeline

```
Detection (PhaseNet)  ->  Association (PyOcto, strict)  ->  Pick augmentation  ->  Absolute location (HYPOINVERSE)  ->  Local magnitudes (ML)  ->  Relative relocation (HypoDD dt.cc) ✅
   picks CSV               events + assignments              augmented assignments    UF<year>.{sum,arc}              Heo 2024 + Sheen 2018 ML       uf_subregion dt.cc reloc (§ below)
```

**Stages** (from `src/ufpipe/run_pipeline.py`): `detection` → `association` → `augment` → `phs` → `locate` →
`relocate`. ufpipe is **end-to-end through HypoDD dt.cc and fully SELF-FED**: the 6th `relocate` stage
(`src/ufpipe/relocate.py`) builds its inputs (event-idx SAC store + per-year pyocto + multi-network station table)
from ufpipe's OWN detection+association via `src/ufpipe/reloc_inputs.py`, then hands off to the downstream driver
`detection_test/reloc_2016_uf/run_picker_reloc.py --skip-build` (scaffold → HYPOINVERSE → QC → GPU xcorr → HypoDD
v2.1beta; the external 15.PocketQuake engine). Run: `python -m ufpipe.run_pipeline --model <p> --years <Y>
--stage-from relocate --through dtcc`.

**Networks — ufpipe detection + association cover KS/KG/GJ/NS** (2026-07). The per-year multi-network station
table is built by `src/ufpipe/stations.py` (KS/KG from StationXML, NS from GHBSN with `N003a→N003`, GJ from the
temporary-array list), keeping stations with a metadata epoch overlapping the year AND data on disk. Each station
carries an `archive` (KS_KG/ · GJ/ · NS_100hz/ mirror for speed); detection globs
`archive/<sta>/<band>?.D/*.{Y}.{doy}`. Coverage timeline is automatic: 2010 = KS/KG only, 2016 adds GJ, 2017+
adds the dense NS array. **Association is daily-chunked** (`config.ASSOC_*`, kim2011, ±150 s/day, in-day origins)
— required because a whole-year single-pass associate is intractable on the ~200-station NS array. Restrict with
`--networks`. GOTCHA: KG.MKL (and other) 2010 station-days have **corrupt Steim2 records** → obspy
`InternalMSEEDWarning` ("Data integrity check for Steim2 failed"); the reader collapses these to one summary line
per station-day and reads anyway (data returned, checksum unverified). **`detection_test/lib` is now DEPRECATED**
(see `detection_test/lib/DEPRECATED.md`): ufpipe's own detection+association feed the relocate stage — the old
per-month lib feeder + `build_sac_and_pyocto.py` are no longer used. Only the downstream reloc driver
(`reloc_2016_uf/run_picker_reloc.py`, run with `--skip-build`) + the PocketQuake engine remain live.

**PyOcto assignment (after augmentation) is the source of truth for which (station, phase) tuples belong to each event.**
The legacy time-window pick dump in `HypoInv/event_waveforms_*/*_picks.csv` was a recipe for
mis-assignment when two events fall < 60 s apart at the same station (the earlier event's pick
"won" by chronological earliest, leaking into the later event's SAC headers). The production
SAC-header writer (`event_sac_export.export_event(..., pyocto_root=...)`, the audit notebook
(`local_magnitudes/04.Catalog_quality_audit.ipynb`), and any downstream that reads `SAC.a`/`.t0`
should consume `models/phasenet_plus/pyocto/pyocto_assignment_<vm>_{year}.csv`, NOT the
time-window CSV. (`<vm>` = `config.PYOCTO_VELMODEL`: **kim2011** for current daily-chunked runs;
legacy pre-2026-07 whole-year artifacts on disk are named `kim1983`.)

The automated implementation lives in **`src/ufpipe/`** (shared module + thin CLIs).
See `docs/how-to-run.md` for commands and `docs/pipeline.md` for stage details.

## Relative relocation — HypoDD dt.cc + MAXDATA recompile (2026-06-25)

Whole-box dt.cc relocation of the ~2,600 UF-subregion events is **DONE**. Outputs in
`…/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/uf_subregion_reuse/2.HypoDD/02.dt.cc/`:
`hypoDD.loc_backup` (HypoInverse absolute), `hypoDD.reloc_dtct_only` (dt.ct), `hypoDD.reloc` (dt.cc≥0.7).
Comparison notebook: `KS_KG/HypoInv/21.UF_relocation_dtcc_comparison.ipynb`.

**Result:** dt.cc tightens *local* clustering hard — median nearest-neighbour distance
254 m (absolute) → 147 m (dt.ct) → **51 m (dt.cc≥0.7)** — and recovers absolute structure that dt.ct
artificially shrank. 78% of events get cc links (median NCCP 6 / NCCS 14). The all-pairs CC xcorr is the
slow part (~days incl. GPU stalls) but clearly worth it.

**Two gotchas that cost real time (KEY):**
1. **Empty dt.cc combine → silent dt.ct-only reloc.** The framework's combine step (`xcorr._filter_combine`,
   filter CC≥0.7 → `dt.cc_0.7_combined`) iterates a `pairs` list that the kill/resume churn left EMPTY,
   so it wrote a **0-byte** file and hypoDD relocated on `dt.ct` ALONE without error. **Always check
   `dt.cc_0.7_combined` is non-zero** before trusting a dt.cc reloc; regenerate from the per-pair
   `dt.cc_P/`, `dt.cc_S/` dirs (keep `line[23:34]` ≥ 0.7) if empty.
2. **HypoDD `MAXDATA` (Fortran compile-time array).** Stock binary had `MAXDATA=3,000,000` in
   `include/hypoDD.inc`; full all-pairs CC≥0.7 (7.3 M cc + 2.6 M ct = 9.9 M dt's) overflows it
   ("STOP >>> Increase MAXDATA"). Recompiled Waldhauser source (`/home/msseo/Downloads/HypoDD`) with
   `MAXDATA=15,000,000`. The >2 GB static arrays need the **large code model**:
   `make FFLAGS="-O -I../../include -std=legacy -mcmodel=large -fno-pie"
   CFLAGS="-O -I../../include -mcmodel=large -fno-pie" LDFLAGS="-no-pie -mcmodel=large"` (also strip the
   macOS `LDFLAGS` line in `src/hypoDD/Makefile`). Installed: `/home/msseo/bin/hypoDD`
   (old 3 M binary kept as `hypoDD_maxdata3M_backup`).

## 2016 Gyeongju 4-picker comparison (`detection_test/`, 2026-07)

Controlled comparison of four ML pickers (PhaseNet+, PhaseNet-original, PhaseNet-STEAD, EQT-STEAD) on year
2016 through one identical pipeline (detection → PyOcto → HypoInverse → QC → HypoDD dt.cc), consistent
P=S=0.2 threshold, picker the only variable. **See `detection_test/CLAUDE.md` for the full
sub-project guide, `reloc_2016_uf/PIPELINE.md` for invariants, and `reloc_2016_uf/study_guide.pdf`.**
Headline: PN+ yields the most cross-correlation-resolved events (255) despite not picking the most —
pick quantity ≠ located quality. **CRITICAL bug fixed (2026-07):** the QC subset used to re-run HypoInverse
(redundant) which mis-staged picks by timestamp → wrong origins → corrupted rereference/dt.cc (same class as
the pyocto-vs-timewindow rule above, but it also poisons dt.cc because origins are SUBTRACTED there). Fixed by
reusing the full-run HypoInverse (`inject_full_hypoinverse`); backups at `1.HypoInv/kim2011.rerun_backup`.

## Cluster analysis + per-volume relocation (Phase 2–3, `analysis/reloc_analysis/` + `uf_subregion_hypodd/`, 2026-07)

Downstream of the whole-box dt.cc reloc: NND declustering, cluster deep-dives, and per-volume HypoDD
relocation of the two largest-event volumes. Builders emit notebooks (run in base): nb26 R-T density, nb27
fractal dimension (Df=1.2), nb28 top-10 NND family deep-dive, nb29 completeness-augmented NND, nb30/31/32
1 km³ **volume-history** cubes (decompose in-cube non-members into other-family / background / not-in-pop),
nb33 **per-volume relocation + bootstrap** of the top-2 volumes (`build_cluster_svd_nb.py`), nb34 the SAME for
the **next 8 largest-event clusters** (ranks 3–10; `build_cluster_svd_next8_nb.py`).

- **NND**: always feed `t_year` via the canonical `kma_absolute_location.nnd.decimal_year` (exact year length).
  A hand-rolled `(doy-1)/365.25` day-resolution formula force-linked same-day pairs (η=0) and faked km cluster
  extents — fixed; don't re-derive it. Df=1.2 (full-pop GP dimension; dt.cc-only 1.16). link_rmax_km=1.0 for UF.
- **Per-volume runner** (`uf_subregion_hypodd/run_svd_volumes.py`, stages select|extract|run|primary|boot|analyze):
  pulls every relocated event in a 1 km³ cube (`svd_volumes/{m389,m373}/`), subsets dt.cc/dt.ct by cuspid
  (streaming filter), kim2011. `stage_run` does the SVD (ISOLV=1) diagnostic → `hypoDD.reloc.svd`; `stage_primary`
  writes the **reported** solution. `svd_volumes_catseed/` = a catalog-seeded backup.
- **KEY solver finding — absolute depth is a DD NULL SPACE; report LSQR-CND on the whole-box seed, NOT SVD.**
  Differential times pin *relative* positions tightly but barely constrain the *absolute centroid depth*.
  Undamped **SVD** has no anchor, so it slides the whole cloud down that null direction to a **seed-dependent**
  depth (m373: whole-box 11.5 → 9.5 km from a whole-box seed, but 10.4 km from a catalog seed — proves it's the
  solver, not the data; m389 barely moves, 13.77→13.75, so it's volume-specific). **Light-damped LSQR (CND 40-80,
  `stage_primary`)** is softly anchored to the seed and holds the **physical** whole-box centroid at *every* damping
  level, gives the **same relative structure** as SVD (≈10 m median, identical thickness), and drops no events. So
  the reported solver is LSQR-CND; SVD is kept only as the nb33 §1 depth-drift diagnostic. **LSQR is also ~500×
  faster** than SVD (m389 3 s vs 26 min; dense O(params²)).
- **Damping = PocketQuake-style PER-SET adaptive (`_adaptive_damp`, CND 40–80 per weighting set).** stage_primary
  and the bootstrap tune EACH of the 7 weighting sets' DAMP via a feedback loop (`new = damp·(CND/60)^0.5`, ≤12
  attempts) so every set's worst-iteration CND lands in 40–80 — vendored from `pipeline/core/hypodd.py:_exec_hypodd`.
  **The earlier single-global-DAMP scan (floor 15, final-iteration CND only) was WRONG**: it over-damped small
  clusters (c105 CND 8, c41 18) and under-damped nothing here, and even m389 in nb33 had 3 of 7 cc-sets at CND
  83–94. Fix records `primary_damps` (7-tuple) + `primary_cnd_per_set` + `primary_cnd_inband` in meta; the bootstrap
  REUSES `primary_damps` so replicas match the reported estimator. Iteration→set mapping is by ORDER (i//NITER),
  not weighting signature (hypoDD rewrites echoed −9→−999, breaking signature matching). A uniform DAMP does NOT
  hold CND constant across iterations — it drifts (c1 44–67) — which is why per-set adaptive is needed.
- **Bootstrap = global resampling, LSQR-CND, seeded on the reported solution, n=200** (`--boot-solver lsqr
  --boot-resample global`, now the defaults). `ez95` is **RELATIVE precision** (event-to-event) — honest for
  that (stable per-set-CND-40-80 estimator, global resample captures pair selection, ~0 failed replicas). It is **NOT** the
  absolute-depth uncertainty: every replica is seeded on one solution so the centroid is fixed by construction; the
  true absolute-depth uncertainty is the **~1-2 km** solver spread from §1, uncaptured by any such bootstrap. (The
  earlier "SVD bootstrap 34 m is the honest depth error" reading was wrong — that scatter was the null-space wander,
  not relative precision.) within-pair resampling omits pair-selection variance and runs ~30% smaller.
- **next 8 clusters (ranks 3–10 by mainshock M): `run_svd_volumes_next8.py` + nb34.** The companion runner imports
  `run_svd_volumes.py`, points `rsv.BASE` at `svd_volumes_next8/`, and reuses every stage; only `stage_select8`
  differs (`fammax.iloc[2:10]`, vol names `c<clusterid>`, soft nb31 count-check). Volumes c11/c4/c40/c41/c1/c105/
  c8/c95 (n 5–86). Findings: SVD-depth wander recurs (c4 11.4→7.1 km, c95 9.9→7.7 km); c1/c8 tightly constrained
  (ez95 ~3–7 m); tiny c105 (5 ev) / c41 (12 ev) weak (ez95 80–240 m) — disclose. All near-0 failed replicas.
- **PCA plane fit (nb33 §3b + nb34 §4b, error-gated).** `pca_plane` = unconditional PCA (strike/dip from PC3
  normal; length/width/thickness = full extent along PC1/PC2/PC3), events gated to **max 95% bootstrap half-width
  < 100 m**; 95% CIs by re-fitting the same event set on every replica (strike branch-centred). m373 fit 3 ways:
  all / before / after the Nov-2023 mainshock. KEY: **before-MS (C6 swarm) strike ~70°/dip ~83° vs after-MS
  strike ~161°/dip ~53° differ**, so the "all" strike is unstable (CI ~72–214°) — the split is necessary. Shape
  gate printed with every fit: top-2 stay **BLOB** (indicative only); in nb34 **c4 & c40 resolve as PLANES**
  (c40 strike ~50°/dip ~85°) once properly damped, c11/c95/c105 linear, c1/c8 blob. Two views: (1) plane rectangle
  on the geographic E-N/E-Z/N-Z sections; (2) **clear fault-frame read-out — map view = strike (solid line+arrow),
  across-strike depth section = dip** (matches `analysis/relocation` fig_fault_strikedip/alongdip). `analysis/relocation`
  already does SVD plane fits + fault_sections viz via PocketQuake — reference it for style.
- **PRODUCTION = kim2011, ISTART=2, per-set adaptive damping (2026-07-07 FINAL). ISTART=1 was the contraction
  bug.** The adaptive relocation looked spatially CONTRACTED vs absolute (E-N RMS 2.6 vs 3.5 km, footprint 12 vs
  15 km) — the cause was **ISTART=1**, which initialises every event at the cluster CENTROID (collapsed), so the
  inversion must spread them and damping resists → contraction. NOT ph2dt (MAXSEP=10 km is generous; the DAMP=600
  run with the same ph2dt kept the extent). **`ISTART=2` (start from the catalog/absolute event.dat locations)
  fixes it**: dt.cc E-N RMS **3.45 vs absolute 3.49**, dt.ct 3.20 vs 3.22 — extent matches in all 3 dims, cc-links
  retained (dt.cc 2116, dt.ct 5/5 in band), all 7 dt.cc sets CN 60-68. `make_inp(...,istart=2)` sets the ISTART
  column; `_adaptive_damp(...,istart=2)`. Production dt.cc = per-set DAMP [978,1079,1044,751,671,614,589]; dt.ct =
  [473,569,506,460,243] (already in band at ISTART=2). Backups: dt.cc `hypoDD.reloc.istart1.bak` + `.damp600.bak`;
  dt.ct `.istart1.bak`. **Per-volume runner also set to ISTART=2** (stage_primary + bootstrap). nb23
  `23.UF_istart2_dimension_check.ipynb` documents the fix (maps: absolute / ISTART=1 contracted / ISTART=2
  restored). **Lesson: never inspect `hypoDD.reloc` mid-run** — `_exec_hypodd_once` deletes it before regenerating,
  so a mid-run read shows empty; monitor via a completion sentinel (result.json), not intermediate files, and use
  the right pgrep pattern (`python3 run.py`, not the dir path).
- **[superseded by the ISTART=2 bullet above] Earlier ISTART=1 adaptive relocation (2026-07-07).**
  History: DAMP=600 (deliberate, `run_kim2011_dtcc.py:53-58`) gave final CND ~58 in band + 2157 cc-resolved; the
  high CND 121-149 is the EARLY ct-weighted transient (sets 0-2), not the converged CN (max-per-set penalizes the
  ct setup phase — my earlier "1/7 in band" alarm was that measurement error). The author had rejected PocketQuake
  adaptive because it drove DAMP DOWN to 6-8 (lost cc links → 487). **The user asked for per-set adaptive CN-40-80
  anyway, and it WORKS here because the fix is to RAISE the out-of-band ct sets, not lower them:** `_adaptive_damp`
  converged to per-set DAMP **[1049,1149,1111,800,682,632,611]** (ct raised ~600→~1050, cc kept ~600-800), giving
  **all 7 sets CN 61-72 AND retaining 2140/2157 cc-resolved (99.2%)**. Swapped into production (backup
  `hypoDD.reloc.damp600.bak`); positions shift ~1 km absolute (relative structure 15-58 m, unchanged); the
  catalog CSV was regenerated and **ML is invariant (median Δ -0.001)**. NND family labels relabeled (same physical
  clusters, same mainshock IDs): c11→c12, c40→c38, c41→c39, c105→c101, c95→c93; nb34 + `run_svd_volumes_next8.py`
  read `svd_volumes_next8/volumes.txt` for the current set. **`make_inp`/`_template_*` now match weighting rows
  STRUCTURALLY (10 numeric tokens in the data-weighting section), NOT by DAMP=="600"** — else they break on the
  new adaptive-DAMP template. `_max_cnd_per_set` groups iterations by weighting-signature change (handles the
  whole-box 6-ct + 4-cc structure). To reproduce/adjust the whole-box damping, update `run_kim2011_dtcc.py` (it
  still writes uniform 600; the current production reloc came from `_adaptive_damp`, not that script).
- **dt.ct-only run (03b.dt.ct_kim2011) ALSO made adaptive + HypoInv notebooks re-run (2026-07-07).** dt.ct was
  already per-set tuned (5 sets, DAMP [473,569,506,460,243], 4/5 in band — set 0 marginally high CN 84); adaptive →
  [561,622,543,497,259], all 5 CN 61-65, ~563 m shift. Applied to production (backup `hypoDD.reloc.prev.bak`).
  nb21 (HypoInv/dt.ct/dt.cc 3-way, now both cc+ct adaptive — dt.ct→dt.cc 822 m H / 141 m V) re-run; nb22 (velocity)
  reads the kim2011 `hypoDD.reloc.damp600.bak` to stay MATCHED-DAMP600 vs generic-600 (else velocity confounds with
  the adaptive ~1 km shift; noted in-notebook). The 7-set `_adaptive_damp` is dt.cc-specific; the dt.ct 5-set one-off
  = scratchpad `run_dtct_adaptive.py` (generalize to N sets if reused). dt.ct inputs unchanged by the dt.cc reloc.
- **event.dat seeding bug (fixed)**: `open(p,"w").write(seed(read(p)))` truncates before read → empty event.dat
  → hypoDD reads 0 events, "0 s" run, and a stale `hypoDD.reloc` masks the failure. Fix = read-then-write +
  assert non-empty + delete any stale reloc before each run.
- **KG.HDB waveform-shape break, 2015-05-21 (found via CC in nb33 §8)**: the M3.89-volume CC matrix shows a
  permanent similarity step at the *documented* KG.HDB sensor break 2015-05-21 (pre-vs-post CC 0.33 vs within
  0.67, does not recover). It is **only at KG.HDB** (multi-station contrast ~0.2 vs ~0.02 elsewhere) → the
  instrument, not the sources. DISTINCT from the ~2014-12 HDB *gain* drift (amplitude only; invisible to
  amplitude-normalised CC — see the Local-magnitudes epoch term). General test for such transitions: recompute
  CC at the next-closest stations with coverage in both eras — instrument artefacts are station-local.

## Directory map

Software (src/), data (data/), outputs (outputs/), and per-network raw waveforms (KS_KG/ GJ/ NS/ NS_100hz/)
are separated. Restructured 2026-07; metadata consolidated by kind under data/metadata/ (2026-07).

```
src/            installable packages — ufpipe/ (the 6-stage pipeline) + uflib/ (pip install -e .)
KS_KG/          KS+KG raw waveforms (station dirs at root; NOT in git; ~3.8 TB) — pure waveform dir
GJ/  NS/  NS_100hz/   the other network waveform dirs (pure waveforms; NOT in git)
data/
  metadata/     *** single home for all metadata, organized BY KIND ***
    stations/     ks_kg/  gj/  ns/  kigam/     — per-network station tables
    responses/    master/ (148 MB KS_KG StationXML, gitignored) + fetched/ + small RESP.* text (tracked)
    velocity/     kim1983.csv (PyOcto layered model)
    catalogs/     ghbsn_heo/ (Heo et al.), USGS_M7_event_catalog.csv
  waveforms/    symlinks to the network dirs
  hypoinv/      HYPOINVERSE station files (STA/), PHS/, per-year catalogs + location notebooks
outputs/
  models/       picker-model dimension: stead/ (reference), original/, phasenet_plus/ — detection→location
analysis/       local_magnitudes/ (ML + responses consumers), uf_subregion_hypodd/, reloc_analysis/, ...
detection_test/ picker-comparison pilot + reloc driver (reloc_2016_uf/, --skip-build); lib/ DEPRECATED
docs/           documentation (+ ufpipe_reference_manual.pdf)
tools/          git helpers (nbstrip.py)
```

## Two independent "model" dimensions — keep them straight

- **Picker model** (`--model`): `stead` / `original` (SeisBench PhaseNet) or `phasenet_plus` (EQNet
  PhaseNet+, in-process; needs a local EQNet clone at `config.EQNET_DIR`). Top of `models/`.
- **Velocity model** (`--velmodel`): the crustal model — `kim1983` vs `kim2011`. Used by PyOcto/HYPOINVERSE.

**Preprocessing**: PhaseNet-style pickers want **raw** (demeaned) data + their own normalization — no
bandpass. (The legacy stead/original detection notebooks applied a 1–40 Hz bandpass; the `pipeline/`
SeisBench path and the `phasenet_plus` backend feed minimally-processed data.)

## Conventions / rules

- **Do not edit the `stead` reference run** (`outputs/models/stead/**`). It is the baseline to compare
  against; `outputs/models/stead/` symlinks it (the scripts refuse `--model stead` writes unless `--force`).
- **New pipeline code lives in `src/ufpipe/`** (installed via `pip install -e .`). The `original` outputs are editable.
- **Canonical pick id**: detection writes `station = "NET.STA"` (e.g. `KG.BBK`); association derives the
  network from it. Do not reintroduce the old hardcoded `["KS"]*N + ["KG"]*…` split.
- **Non-destructive scaffolding**: `models/build_original_tree.py` only ever writes under `models/`.
- Defaults (paths, thresholds, region) live in `src/ufpipe/config.py` — change them there.

## How to run (quick)

```bash
# SHARED 64-core box: ALWAYS pin cores with taskset + set OMP_NUM_THREADS.
OMP_NUM_THREADS=1  taskset -c 0-7  python -m ufpipe.run_pipeline --model original      --years 2010-2024
OMP_NUM_THREADS=16 taskset -c 8-23 python -m ufpipe.run_pipeline --model phasenet_plus --years 2010-2024
python -m ufpipe.detection --model original --year 2024 --days 1-5    # one stage / slice
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

- miniforge Python, **two-env split**: detection (PhaseNet+/SeisBench, torch) in `eqnet` (Py 3.9);
  association (PyOcto) + orchestration in `base` (Py 3.12). `pip install -e .` in BOTH (requires-python
  >=3.9). GPU xcorr (relocate) shells out to `pq-gpu`. See `environment.yml` header + `requirements.txt`.
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

- **Files** (beside `uf_cluster.py`): `src/uflib/uf_waveform_similarity.py` (module, same
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
  `outputs/models/pipeline/notebooks/phasenet_plus_test.ipynb`.
- **Post-location analysis** (`src/uflib/uf_cluster.py` + notebooks `03_blast_decluster_hdbscan`,
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

`analysis/local_magnitudes/ml_pipeline.py` deconvolves each event's response, simulates Wood-Anderson
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

**Augmentation module: `src/ufpipe/pick_augmentation.py`**. For each PyOcto event, the
direct-ray travel time to every station within `radius_km=100 km` is computed from
`kim1983` (`velocity_at_depth`/`predict_arrival_offset`); any daily pick within
`tolerance_s=1.0 s` of the predicted arrival, on a station not already in the PyOcto set
for that event, is a candidate. `apply_safeguards` enforces: (1) phase-strict matching, (2)
best-match-wins across competing events, (3) drop-on-tie when two candidates are within
`tie_threshold_s=0.2 s` of each other. The output overwrites
`pyocto_assignment_kim2011_<year>.csv` (the `config.PYOCTO_VELMODEL` filename; augmentation's internal
travel-time layers are still Kim-1983 — that's the physics table, not the filename) so the downstream
PHS/HypoInverse stages consume the augmented set unchanged.

`ufpipe/run_pipeline.py` runs `augment` between `association` and `phs`. Validated on 2013-03-22
13:40:04: 10 picks / DMIN=42.8 / ERZ=3.9 → 14 picks / DMIN=2.1 / ERZ=0.5.

## HypoInverse QC (in `src/uflib/uf_cluster.py`)

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
