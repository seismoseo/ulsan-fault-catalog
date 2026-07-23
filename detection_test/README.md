# detection_test/ — picker × associator benchmark → SETTLED PRODUCTION CONFIG

## ★ PRODUCTION PIPELINE (settled 2026-07-19): PhaseNet+ → PyOcto (kim2011)

Confirmed on the DENSE 2021-09 month (241 stations = KS22 + KG21 + NS198; run `04.Evaluation_2021_09.ipynb`,
record sections `07.PN_vs_original_record_sections_2021_09.ipynb`):

- **Recall**: UF-relocated **43/43 (perfect)**, KMA **23/24** (the 1 formal miss = a ~6 s origin-time offset on
  2021-09-28, event detected & located 1.36 km from KMA epicentre → effectively 24/24). PhaseNet+ beats
  PhaseNet-original (23/24 + 38/43 at strict gate); its real edge is the smallest UF events.
- **Timing (measured, clean, 1 GPU)**: PhaseNet+ detection **74 min** + PyOcto association 66 min = **141 min /
  dense month**. Full 2010–2026 run ≈ **~5.5–7 days** (detection 3-shard, association overlapped on CPU) — inside
  the user's 1-week limit.

Two decisions that define the pipeline:
1. **CONSISTENCY** (user): full network, ONE fixed permissive association gate **4/2/2 at every epoch**; the dense
   era's false coincidences (~10k events/month) are removed DOWNSTREAM by the network-independent physical
   **location-QC** (HYPOINVERSE RMS / gap / ERH / ERZ / min-stations) — the gate is NEVER tuned per network
   (16/8/8 was a mistake; it breaks temporal consistency). Association MUST be daily-chunked (`associate_daily.py`).
2. **SPEED** — PhaseNet+ was never fundamentally slow: the Buan wrapper left inputs on the CPU and paid a slow
   transfer on every forward. Fix in `run_pnplus_month.py`: `pin_memory` + `meta["data"].cuda(non_blocking=True)`
   → **5.6 → 1.0 s/station-day (~5×), bit-identical picks; PN+ now FASTER than PhaseNet-original.** Detection reads
   the one-time 100 Hz mirror `NS_100hz/` (`predecimate_ns.py`, DETECTION-ONLY; original 200 Hz `NS/` untouched for
   relocation/xcorr). Pickers checkpoint per station (crash-safe).

Runners (all `--month`): `build_stations.py`, `run_pnplus_month.py` (`--predecimated --doy-start/--doy-end` shards),
`run_seisbench_picker.py` (`--predecimated/--native`), `associate_daily.py` (`--workers`; but PyOcto is already
multi-threaded so day-pools oversubscribe — overlap association with detection instead), `predecimate_ns.py`.
Orchestration: `run_confirm_2021_09.sh`.

---

# detection_test/ — one-month picker × associator benchmark (2014-09) — COMPLETE (v2, refined)

Locked decisions: month 2014-09 (M3.89 UF sequence) · **kim2011 for all association/location** · EQT = SeisBench
"original" (STEAD-trained Mousavi) · HARPA = Buan recipe (P6.0/S3.5, seeded) · 12-station early network.

REFINEMENTS applied (v2, per user):
  * FAIR thresholds: PhaseNet original & stead now at **0.2** (== PhaseNet+); EQT keeps its two-stage 0.3/0.1/0.1.
  * Association params matched to the validated **Ulsan/Buan PyOcto config**: n_picks=4, n_p=2, n_s=2,
    **n_p_and_s_picks=1** (omitting it defaults to 3 -> silently raises the gate to 6/3/3, as in v1),
    velocity tolerance 1.0, pick_match_tolerance 1.5. HARPA gate identical (4/2/2).

RESULT MATRIX (truths: 24 KMA regional + 57 stage-1 dt.cc-relocated M3.89 events; match ±5 s/≤30 km; tierA=≥4P&≥2S):

  picker         assoc   events  KMA    UF     tierA  runtime
  phasenet_plus  pyocto  508     23/24  57/57  165    10 s     <- BEST (perfect UF recall, high per-event quality)
  original       pyocto  714     23/24  55/57  198    9 s      (most events, but 42% are small <=4-pick)
  phasenet_plus  harpa   214     23/24  55/57  146    124 s
  original       harpa   222     23/24  48/57  154    223 s
  eqt            pyocto  242     21/24  49/57  111    9 s
  eqt            harpa   132     22/24  36/57  111    54 s
  stead          pyocto  187     23/24  35/57  78     5 s
  stead          harpa   99      21/24  32/57  72     31 s

KEY FINDINGS:
  * BEST = PhaseNet+ × PyOcto(kim2011): 57/57 relocated events, 23/24 KMA, best per-event quality, ~10 s.
  * picks/event PyOcto≈HARPA on matched events (median 9-10, ±0.2) -> the Buan "HARPA associates far more picks"
    is a station-DENSITY effect (Buan ~150 sta), NOT a HARPA property; confirmed on this 12-sta network.
  * HARPA seed repeatability 88-91% at 0.2 (was 95-96% at 0.3 -> more marginal picks = more SGLD variability).
  * PyOcto residuals < HARPA on the mainshock (RMS 0.093 vs 0.130 s) — kim2011 layered beats HARPA homogeneous vel.
  * Native-200-Hz (nb 05): no recall gain at matched false rate; PhaseNet+ ALSO does clean native via dataset
    sampling_rate=200 (identical to PhaseNet model.sampling_rate=200); ~5 ms pick agreement -> ADOPT DECIMATION.

Notebooks (all re-runnable, run in this dir): 01 readiness · 04 evaluation · 05 native-rate · 06 record sections.
Runners: lib/run_seisbench_picker.py · lib/run_pnplus_month.py · lib/run_association.py (+ run_native_*).
Quirks handled in code: KMA code rename BUS->BUS2 (2014 headers carry OLD code), HARPA needs RangeIndex station_df
/ returns UTCDateTime + uppercase P-S, pyocto wants float-second picks + explicit n_p_and_s_picks, pyarrow needed.

## nb06 record sections (v4)
Two comparison axes, same two-panel style (dotted grey = available raw picks; colored = associated, blue P/red S):
  * ASSOCIATOR axis (PICKER fixed = phasenet_plus): 5 magnitude-spread events (M3.5/2.2/1.4/0.8/0.3, PyOcto|HARPA
    agree) + 2 PyOcto-ONLY events (M0.1, M0.2) where HARPA associated NO event though 8-14 raw picks were
    available — the picture of HARPA's small-event conservatism. Change PICKER to 'original' for its 4 cases.
  * PICKER axis (associator fixed = PyOcto+kim2011): 2 PN+-ONLY events (2014-09-19 18:29, 2014-09-21 21:34;
    neither in the KMA catalog) present in the PN+xPyOcto catalog but absent from PN-original x PyOcto. Left =
    PhaseNet+ forms the event; right = PhaseNet-original picked nothing (09-19) or only 4 picks < gate (09-21)
    -> no event. The mechanism behind PN+'s 57/57 vs original's 55/57 UF recall. figures/rec_pncompare_*.png.
