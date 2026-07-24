# 2016 Ulsan-Fault relocation pipeline — full walkthrough & invariants

> **DEPRECATION NOTE (2026-07).** Stages 0–2 below (lib per-month detection/association +
> `build_sac_and_pyocto.py`) are superseded: the ufpipe `relocate` stage now builds those inputs from ufpipe's
> own per-year detection+association (`src/ufpipe/reloc_inputs.py`) and invokes this driver with
> `--skip-build` (see `detection_test/lib/DEPRECATED.md`). Stages 3+ (scaffold → HYPOINVERSE → QC → **inject full-run
> HYPOINVERSE** → rereference → xcorr → HypoDD) and all **Invariants** in this document remain fully in force —
> they describe the live driver. This document is also the historical record of the 2016 4-picker pilot.

This document explains, stage by stage, how a picker's picks become a dt.cc-relocated catalog, **where the
origin/location information comes from at each step**, and the invariants that must hold. It exists because a
subtle bug (a redundant HYPOINVERSE re-run that mis-staged picks by a timestamp key) silently corrupted dt.cc for
~18–21% of events on the dense pickers. Read the **Invariants** first — they are the rules that, if kept, prevent
that class of mistake.

---

## Invariants (the rules that prevent silent corruption)

1. **One HYPOINVERSE solution, computed once, reused everywhere.**
   Absolute origins/locations are produced exactly once (the *full-run* HYPOINVERSE, Stage 4) and **reused** by
   every downstream stage — never re-computed for a subset. QC (Stage 5) *selects* a subset; it must not trigger a
   re-location. The QC-subset relocation injects the full-run `.sum`/`.arc` (subset + renumbered), it does not
   re-run HYPOINVERSE. *(This is the invariant the original bug violated.)*

2. **Pick provenance is by `event_idx`, never by timestamp.**
   The correct, PyOcto-associated picks for an event live in `event_sac/<event_idx>/<event_idx>_picks.csv`
   (prob = 1.0). Any stage that re-derives an event's picks from a **time window** can pick up an *adjacent*
   event's raw picks — the exact mechanism of the bug (two events sharing a second-resolution `event_id` /
   overlapping window). Always key picks by `event_idx`.

3. **Origins flow into dt.cc — not just into starting locations.**
   `rereference` (Stage 6) stamps each SAC's origin from a `.sum`; `xcorr` (Stage 8) then measures
   `dt = (t1 + shift − ot1) − (t2 − ot2)`, i.e. it **subtracts those origins**. A wrong `.sum` origin therefore
   corrupts the dt.cc *values*, not merely the `event.dat` seeds. The `.sum` used by `rereference` must be the
   authoritative full-run solution.

4. **`members.txt` row order defines the cuspid (200000 + row) that ties `.sum`, `.arc`, `event.dat`, and `dt.cc`
   together.** Never reorder members without regenerating all four consistently. QC cuspid = 200000 + qc_row;
   full cuspid = 200000 + full_row; the mapping between them is `members_qc[event_idx] → members[event_idx]`.

5. **The catalog is fed to the pipeline in KST and round-trips to UTC. Do not "fix" this to UTC.** See the boxed
   note below — feeding UTC directly would shift every event by 9 h.

---

## ⚠️ Why the catalog is `catalog_kma.csv` and why it is in KST (a common confusion)

**We do not use the KMA catalog.** The relocation engine is the pre-existing `korea-cluster-relocation` pipeline,
which was originally written to relocate **KMA-catalog** clusters (Kimcheon, Jangsung, …) from a shared KMA SAC
archive. We feed *our own* PhaseNet+ → PyOcto catalog into that pipeline **unchanged**, so we must produce a file
in its "KMA catalog" *schema* — hence the inherited name `catalog_kma.csv` and the `_kma` suffixes. It is a
**format label, not a data source.**

The pipeline hard-codes a KST convention (`kst_offset_hours = 9`):

```
pipeline/core/waveforms.py :  origin_utc = catalog_time(KST) − 9 h
                              event_id   = strftime(origin_utc)          # UTC-second timestamp key
```

So `build_catalog_kma.py` deliberately converts our PyOcto **UTC** origins **into KST** (`kst = utc + 9h`) *only
because the pipeline immediately subtracts 9 h again* to recover the UTC origin:

```
  our PyOcto UTC origin  --(+9h in build_catalog_kma)-->  catalog_kma.csv (KST)
                         --(−9h in pipeline load_catalog)-->  origin_utc  (back to the true UTC origin)
```

It is a **deliberate no-op round-trip** that satisfies the pipeline's KST-input contract. **If you fed UTC
directly, the pipeline would subtract 9 h and place every event 9 h in the past.** Leave the round-trip in place.

---

## Stage-by-stage (the picker is the only variable)

| # | Stage | Script / pipeline stage | In → Out | Origin/location source |
|---|-------|------------------------|----------|------------------------|
| 0 | **Detection** | `lib/run_seisbench_picker.py`, `run_pnplus_month.py` | continuous mSEED → `picks/picks_<picker>_2016_<mm>.parquet` | — (raw picks, prob ≥ 0.2, P=S) |
| 1 | **Association** | `lib/associate_daily.py` (PyOcto, `gj_config.py`) | picks → `catalogs/catalog_<picker>_2016_<mm>_pyocto.csv` + `assign_*.parquet` | **PyOcto** (rough). Gate 4/2/2, ≥1 P+S. Binds picks↔event. |
| 2 | **SAC store** | `build_sac_and_pyocto.py` → `event_sac_export.py` | assoc + wf → `event_sac/<event_idx>/*.sac` + `<event_idx>_picks.csv` (assoc picks, prob 1.0) + `pyocto/pyocto_kim2011_2016.csv` | PyOcto (in SAC headers). **`<event_idx>_picks.csv` = the one true pick set.** |
| 3 | **UF-box catalog** | `build_catalog_kma.py` | region pyocto → `catalog_kma.csv` (KST), `members.txt`, `members_event_idx.csv` | Defines cuspid = 200000 + members-row. UTC→KST round-trip (see box). |
| 4 | **Absolute location (FULL)** | pipeline `hypoinverse` (kim2011) | assoc-pick `.arc` → `1.HypoInv/kim2011/<slug>.{sum,arc}` | **FULL-run HYPOINVERSE = the authoritative absolute solution.** |
| 5 | **QC (filter only)** | `build_qc_catalog.py` (`uf_cluster.QC`) | full `.sum` → `members_qc.txt`, `catalog_kma_qc.csv` | **No re-location.** Keeps erh<5 ∧ erz<5 ∧ gap<270 ∧ num>5 ∧ rms<1.0. |
| 6 | **Re-reference** | pipeline `rereference` | `.sum` origins → restamps SAC `nz*` + `a`/`t0` in `waveforms_100km/` | **Must read the full-run `.sum`.** ⚠️ origins flow into dt.cc. |
| 7 | **ph2dt** | pipeline `ph2dt` | `.arc` → `ncsn2pha` → `.pha` → `event.dat`, `event.sel`, `dt.ct` | Same `.arc` as Stage 4/6 → `event.dat` seeds + `dt.ct`. |
| 8 | **Cross-correlation dt.cc** | pipeline `xcorr`/`dtcc` (pq-gpu, interp_hz=1000) | re-ref SACs → `dt.cc_0.7_combined` | `dt = (t1+shift−ot1) − (t2−ot2)`; **subtracts the Stage-6 origins.** |
| 9 | **HypoDD** | `run_hypodd_kim2011_istart2.py` (ISTART=2, adaptive CND 40–80) | event.dat + dt.ct + dt.cc + station.dat + hypoDD.inp → `hypoDD.reloc` | Relative relocation; event.dat = starts, dt.ct/dt.cc = data. |

Driver: `run_picker_reloc.py --picker <p> --year <Y> --through {hypoinverse,dtcc}` chains Stages 2–9.
`phasenet_plus` reuses the finished `reloc_<Y>_uf/` (slug `uf_<Y>`); other pickers use `reloc_<Y>_uf_<p>/`.

---

## Year-generality (2010–2025) — the orchestration is `--year` parameterized

Stages 0–1 (detection, association) were always `--month YYYY-MM` general; Stages 2–9 (this orchestration) are
now `--year` general too. **All year-dependent names go through `year_paths.py`**; `--year` defaults to 2016 and
resolves to the exact existing 2016 paths/slugs, so 2016 work is byte-unchanged.

```
python preflight_year.py --year <Y>                                   # what stage 0-1 inputs exist / what to run
python run_picker_reloc.py --picker <p> --year <Y> --through dtcc     # relocate (any year with inputs)
python fix_qc_rerun_bug.py --year <Y> --apply                        # repair driver (also --year)
```

Naming: `reloc_<Y>_uf[_<p>]/`, slug `uf_<Y>[_<p>]`, `stations_<Y>.csv`, `catalog_<p>_<Y>_<mm>_pyocto.csv`.

**Why it survives network/rate change without ad hoc tuning** (all in the lower layers, unchanged):
- **Network change** (KS/KG → +NS 2017 → +GJ temp 2016): stations are *discovered* from StationXML + on-disk
  `coverage` each month (`lib/build_stations.py`), never hard-listed — the deployed network for that month is
  what enters the run.
- **Sampling-rate change** (100/200/1000 Hz): the SAC store keeps each station's *native* rate
  (`SAC_TARGET_HZ=None`) and the dt.cc xcorr interpolates to a common `interp_hz` only at correlation time.
- Velocity model (kim2011), the QC thresholds, and the 4/2/2 gate are **epoch-invariant by choice**, so catalogs
  stay directly comparable across years.

**What a new year still needs:** only its Stage 0–1 *inputs* generated (detection + association for the 12
months — the heavy compute). The orchestration then runs unchanged. Real per-year effects — network-gap months
give smaller sets, sparse early years (2010–2012, KS/KG only) have larger azimuthal gaps so fewer events survive
QC, and the O(N²) xcorr cost grows in dense aftershock years — are the pipeline *honestly reporting what that
year's network resolves*, not tuning. `preflight_year.py` reports exactly which inputs are present vs missing.

---

## The bug (2026-07) and its fix — a worked example of violating the invariants

**Root cause.** Stage 5→9 for the QC subset scaffolded a *separate* cluster (`uf_2016[_p]_qc`) and re-ran
HYPOINVERSE + ph2dt on it (**Stage 4 + 7 a second time — violates Invariant 1**). That re-run re-staged picks by a
second-resolution timestamp key instead of reading `<event_idx>_picks.csv` (**violates Invariant 2**), so for
dense catalogs an adjacent event's raw picks (prob < 1) overwrote the correct associated picks. HYPOINVERSE then
mis-located those events (origins up to ~1 s off), which poisoned:

- `rereference` → wrong SAC origins → wrong **dt.cc values** (**Invariant 3**),
- `ph2dt` → wrong `event.dat` locations **and** wrong `dt.ct`.

**Symptom that exposed it:** `event.sel`/`event.dat` showed events with ERZ > 5 km that had passed QC — because
those errors came from the *re-run* HYPOINVERSE, not the full run QC gated on. Blast radius (events whose re-run
origin drifted): median |Δorigin| ≈ 0.07 s; PN+ 106, original 119, stead 8, eqt 6 fell below QC on the re-run;
hundreds more had smaller (still real) dt.cc contamination.

**Fix (`fix_qc_rerun_bug.py`, and permanently in `run_picker_reloc.py::inject_full_hypoinverse`).** Replace the QC
cluster's `1.HypoInv/kim2011/{.sum,.arc}` with the **full-run** solution subset to the QC members and renumbered to
QC cuspids (200000 + qc_row), then re-run `rereference → ph2dt → xcorr → dtcc` (now against correct origins) and
the adaptive HypoDD. One HYPOINVERSE solution, reused — Invariant 1 restored. The corrupted re-run is backed up to
`1.HypoInv/kim2011.rerun_backup`.

**What was NOT affected:** raw event SACs in `event_sac/` (they carry the PyOcto location and are read by the
record-section notebooks nb11/nb12); the QC *selection* itself (596/574/188/160 events unchanged); the pick lag
measurements in dt.cc (only the origin bookkeeping was wrong).

---

## Quick provenance checks (run these if a result looks off)

- **Which HYPOINVERSE does a QC event carry?** Compare `event.dat` location vs the full-run `.sum` row for that
  event (via `members_qc → members → cuspid 200000+full_row`). They must match. If `event.dat` disagrees with the
  QC-gated `.sum`, a re-run has crept back in.
- **Is a SAC origin correct?** In `waveforms_100km/<ts>/…sac`, `(starttime − b)` must equal the full-run `.sum`
  origin for that event. A ~1 s (or 9 h!) mismatch means a wrong `.sum` (or a broken KST round-trip).
- **KST round-trip intact?** `catalog_kma.csv` time − 9 h must equal the PyOcto UTC origin in
  `members_event_idx.csv`. If they're equal *without* the −9 h, someone "fixed" the round-trip and every event is
  9 h off.
