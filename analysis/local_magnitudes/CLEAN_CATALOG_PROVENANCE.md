# Clean-catalog derivation — provenance & reproduction (Heo ML, 2026-06-25)

How `..._with_ml_heo_clean.csv` and `..._with_ml_heo_homogenised_clean.csv` are produced from the
recomputed magnitudes. Every step is deterministic (no randomness) → bit-reproducible.

## Pipeline

```
catalog_phasenet_plus_2010_2024_blastclean.csv         (HypoInverse locations + PhaseNet+ picks)
        │  run_heo_recompute.py   (Heo2024, HYPOCENTRAL dist, snr_pp>=2.0, TauP P for missing picks)
        ▼
..._with_ml_heo.csv (event ML)  +  ..._per_station_ml_heo.csv (per-station, carries p_source pick/taup)
        │  rebuild_derived_heo.py qc      (location-coherence QC; ML-independent)
        ▼
..._with_ml_heo_scored.csv (all events + mv_* + coherence_fail)
..._with_ml_heo_clean.csv  (coherence_fail==False)         ← removes edge-mislocations
..._with_ml_heo_coherence_audit.csv (the removed events)
        │  09.Station_corrections_ML.ipynb  (median-polish station terms → ml_homogenised)
        ▼  catalog_ml_heo_station_homogenised.csv
        │  rebuild_derived_heo.py homog     (clean catalog, magnitude <- ml_homogenised)
        ▼
..._with_ml_heo_homogenised_clean.csv
```

## Reproduce

```bash
cd KS_KG/local_magnitudes
ML_WORKERS=8 python run_heo_recompute.py        # foundational ML (~90 min, region-wide 14,803 ev)
python rebuild_derived_heo.py qc                # -> scored + clean + coherence_audit
jupyter nbconvert --execute --inplace 09.Station_corrections_ML.ipynb
python rebuild_derived_heo.py homog             # -> homogenised_clean
```

## Location-coherence QC — exact spec (`HypoInv/qc_location_coherence.py`)

Per event: take the PhaseNet+ **detection** P picks within `[origin, origin+25 s]`, earliest P per
station, probability ≥ 0.3; robust (Theil–Sen) fit of arrival-time vs station distance. A station
"fits" if its residual < 1.0 s. `mv_inlier` = fraction fitting. **Flag (`coherence_fail`)** iff the
event has ≥ `MIN_P=5` P picks **and** `mv_inlier < INLIER_MIN=0.5` (fewer than half the stations lie on
one moveout → the hypocentre is inconsistent with how the wavefield actually arrived).

- inputs: picks `KS_KG/detection_location/{year}/picks/picks_{year}.{doy}.csv` (13–56 stations by year);
  stations `KS_KG/station_table/stations_{year}.csv` (covers 2015–2024; pre-2015 untestable → unflagged).
- params: `MIN_P=5, INLIER_MIN=0.5, WIN_S=25 s, P_MIN=0.3, INLIER_TOL=1.0 s`.

## Result (2026-06-25)

- 14,803 events → **5,537 testable** (≥5 P picks) → **133 flagged** → **clean = 14,670**.
- homogenised_clean = 13,813 (clean events that also have a station-homogenised ML).
- **Verification** — the QC catches the known leaked events, e.g. the **2017-11-15 05:56 Pohang
  M2.8** mislocated into the box (`mv_inlier = 0.00` → flagged), plus the 2016 Gyeongju events leaking
  in at 129.19°E and far-west (128.6–128.8°E) events. 13 flagged events are M≥2 — all genuinely
  misplaced.

## Two bug fixes applied to `qc_location_coherence.py` (made it work on the current picks)

1. **tz mismatch** — picks `peak_time` parse as tz-aware UTC, catalog `time` as naive → comparison
   crashed. Both now normalized to `tz_localize(None)` (naive UTC).
2. **station-key mismatch** — picks use `NET.STA.LOC` (`"KG.BBK."`, trailing dot) but the station table
   keys are `NET.STA` (`"KG.BBK"`). Picks station is now normalized to `NET.STA` → 100% match
   (6255/6255 on a dense day). Without this every event failed the ≥5-pick test (0 flagged).

## Note on the old "510"

Earlier notebook text said this QC removed **510** edge-mislocations. That figure is **not reproducible**
from the data in the repo: the current densest pick archive (`detection_location`, ≤13 stations in 2016)
with 100%-correct station matching yields **133**. The 510 came from a run whose inputs/params are not
recoverable (most likely a denser legacy pick set or a more aggressive threshold). The current 133-set is
verifiably correct (catches every known mislocation), so it supersedes the 510. Notebook text updated
accordingly.

## n_used>=3 magnitude-statistics filter (2026-06-25)
- catalogs '_clean' and '_homogenised_clean' now carry: ml_all (event-median ML, ALL events n_used>=1,
  for location/HypoDD merge) and magnitude (= ml_all only where n_used>=3, else NaN -> magnitude
  statistics auto-exclude single/double-station events via their existing dropna).
- Rationale: 1-2 station ML is unreliable; keep the events (location) but exclude from FMD/b/Mc.
- UF box: 2589 with ML -> 1712 with n_used>=3.
- Dead-trace floor (DEAD_TRACE_FLOOR_MM=1e-8) removed 179 non-physical readings (67 events corrected,
  incl. the BUS3 ML=-8.8 artifact -> now NaN).
