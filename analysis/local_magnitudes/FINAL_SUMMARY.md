# Final ML reanalysis — summary (2026-06-25)

Recompute of the local-magnitude catalog with **all corrections**, and what it changes. Numbers below
are direct maximum-curvature Mc + Aki-b (quick, robust); the **bootstrap-CI authority is nb03 / nb10 /
nb11** once the re-run batch finishes (running).

## Corrections applied (the full chain)

| step | what | effect |
|---|---|---|
| **Hypocentral distance** | Sheen S-window + Heo Eq.3 use √(epi²+depth²), was epicentral | near-station ML no longer under-estimated; **UF median ML +0.39** |
| **TauP theoretical-P** | kim2011 P for stations PhaseNet+ missed | recovers measurable events (e.g. 2019-02-15: 0→3 stations) |
| **snr_pp ≥ 2** | peak/peak SNR gate (vetted in nb20) | drops noise-dominated readings |
| **Dead-trace floor** | reject WA peak/noise < 1e-8 mm | removed 179 non-physical readings; fixed the BUS3 **ML −8.8** artifact + 67 contaminated events (up to +0.4 ML) |
| **n_used ≥ 3 (stats only)** | event ML reliable only with ≥3 stations | events kept for **location** (`ml_all`), excluded from **magnitude statistics** (`magnitude`) |
| **Static station corr** (nb09) | median-polish, one term/station | homogenised ML |
| **Epoch station corr** (nb15/17) | time-dependent terms (HDB/MKL drift) | confirms drift; small net effect (median robust) |

## Catalog counts

- region-wide: 14,803 events → **13,938 with ML** → coherence-clean **14,670** (133 edge-mislocations removed; see CLEAN_CATALOG_PROVENANCE.md — note this is 133, *not* the old unreproducible 510).
- **Ulsan-Fault box: 2,781 located → 2,589 with ML → 1,712 with n_used ≥ 3** (the magnitude-statistics sample). The other 877 (1–2 station) are kept for location via `ml_all`.

## Headline: UF-box FMD (n_used ≥ 3)

**FINAL authoritative values — CAPPED (≤60 km, temporally-stationary) catalog, n_used ≥ 3:**

| catalog | source | Mc | b |
|---|---|---|---|
| **CAPPED raw** (hypo + TauP + ≤60 km) | nb03 | **0.80** | **1.15 ± 0.06** |
| **CAPPED homogenised** | nb10 | **0.80** | **1.15 ± 0.06** |

Raw and homogenised now **agree (both 1.15)** — the distance cap removed the network artifact, so the
earlier raw-vs-static spread is gone. *(Pre-cap / uncapped values, now superseded: raw 0.91@Mc0.60,
static 1.26@Mc0.80 — those carried the network-distance bias.)*

> ⚠ **b is strongly Mc-dependent here** — UF b = **0.91 at Mc 0.60** but **1.26 at Mc 0.80**. nb03 (raw)
> and nb10 (static) pick *different* KS-Mc, so the raw→static jump is partly the higher Mc, **not** purely
> the station correction. Don't read a clean "raw→static→epoch" progression off a single number; the
> honest statement is **UF b ≈ 0.9–1.3 depending on Mc**, with the bootstrap CIs above.
> (My earlier quick maximum-curvature estimate used Mc 0.8 throughout and over-stated the raw b as 1.14 —
> the notebook KS-Mc values here supersede it.)

**Two things changed; one didn't:**
1. **Mc rose 0.3 → 0.6–0.8.** Driven by the +0.4 ML scale shift (hypocentral fix) **and** the n_used≥3 reliability filter. **Restate any absolute-magnitude / Mc statement and any KMA cross-comparison with this offset.**
2. **The epoch correction barely moves the bulk b** (whole-catalog 1.03 → 1.05) — it matters for individual station-dominated events, not the FMD as a whole.
3. **The size distribution is not reshaped pathologically** — b stays in a physical ~0.9–1.3 band; no conclusion reverses.

## Per-year UF b (n_used ≥ 3): raw → static → epoch

*(quick maximum-curvature estimates — **nb13 has the authoritative bootstrap per-year FMD**; trend, not exact numbers)*

| year | raw | static | epoch | note |
|---|---|---|---|---|
| 2014 | 0.81 | 0.85 | **0.87** | stays low after *all* corrections |
| 2015 | 0.90 | 0.95 | **1.02** | normalises to ~1.0 |
| 2016 | 0.95 | 1.13 | 1.16 | |
| 2017 | 1.35 | 1.39 | 1.35 | |

**The 2014 low-b (~0.87) is REAL — the 2014-09-23 M3.88 mainshock + aftershock sequence (the largest UF
event), not the HDB amplitude artifact.** The epoch correction (which fixes HDB/MKL drift) does NOT raise
it, because event ML is a **median over many stations** and is robust to one station (HDB) collapsing.
This reframes the "2014–2015 distortion": it's a real sequence + year-to-year Mc/network variation, not a
magnitude artifact.

## Station-correction findings (nb15)

- **Time-invariance is genuinely violated** — significant epoch steps in HDB, MKL, YSB, GUWB, POHB, DUC
  (HDB/MKL up to ~0.5–0.6 ML; HDB steps at 2015-03-20 +0.11, 2019-03-12 −0.43). So the static one-term
  model is incomplete — your instinct was right.
- **But the whole-catalog b barely moves** (static 1.03 → epoch 1.04), and the *naive* epoch-split
  over-corrects HDB/MKL → the **conservative** epoch is used. Net: epoch correction matters for
  individual station-dominated events, not for the bulk FMD.

## What to carry into interpretation

1. **Magnitudes/Mc are on a new scale (+~0.4 UF, Mc 0.3→0.8 with the reliability cut).** Re-state
   absolute-magnitude claims; redo any KMA-ML comparison with the offset.
2. **UF b ≈ 1.1–1.3 (corrected)** vs old ~1.0 — slightly higher; check the bootstrap CI (nb11) before
   locking a number.
3. **2014 low-b is a real sequence**, not an instrument artifact — safe to interpret tectonically.
4. **Relocation (nb21/22) is independent** (location, not ML): dt.cc collapses NN 254→43 m; velocity-model
   systematic ~114 m ≫ LSQR formal error.

## Caveats

- Mc/b above are quick maximum-curvature + Aki; **nb03/10/11/12 (re-running now) give the bootstrap CIs**
  and KS-Mc — treat those as authoritative.
- b is sensitive to the Mc choice at this N; the ~1.0→1.3 trend is indicative, confirm with EMR (nb12).

---

# Temporal-stationarity fix — post-2019 magnitude inflation (2026-06-25, late)

**Problem (user-flagged):** median ML drifted UP after ~2019 (network densification), inflating the
M≥0.9 rate — a spurious step exactly where the network changed.

**Diagnosis (exact):** it is a **network-geometry artifact**, not real seismicity:
- The attenuation formula is correct — our `−log A₀` = Heo Eq 3 *exactly*, and **core (≤2015) stations
  have a flat residual to 100+ km** (matching Heo Fig 5c).
- The **new (post-2016) stations are far and read systematically high**, with a bias that **grows with
  source distance within a single station** (e.g. HCH.HHZ: +0.15 near → +0.74 at >100 km) — a
  path-attenuation effect (their low-attenuation paths are over-corrected by the regional-average −log A₀).
- It is **not** a constant site/path term, so a one-term-per-station correction cannot remove it (verified:
  global *and* core-anchored corrections leave the drift; the bias is distance-dependent, not constant).
- As far stations grow from ~5% (≤2016) to 31% (2024) of readings, they pull the event-median ML up.

**Fix:** **distance cap `max_dist_km = 60` km** (added to `ml_pipeline`; applied by re-aggregating the
event ML from ≤60 km readings only) + the station correction re-estimated on the capped data.
- Median-ML slope (UF, 2017–24): **−0.0246/yr (old) → −0.0015/yr (flat)**.
- **M≥0.9 rate post/pre-2019: → 0.89** (the spurious inflation is gone).
- ≤60 km chosen over ≤40 km because ≤40 km destroys the early UF era (keeps only 192/534 early events;
  the sparse 2010–2016 network sat 40–60 km from the cluster). ≤60 km loses just 19 early UF events.
- Catalogs now carry `ml_all` = UNCAPPED event median (for location/HypoDD, all stations) and
  `magnitude` = capped ≤60 km median, n_used≥3 (for statistics). Uncapped per-station backed up to
  `..._per_station_ml_heo_uncapped.csv`.

**Caveat:** this supersedes the b/Mc numbers earlier in this file — those were on the *uncapped*
catalog. The re-run downstream notebooks (nb03/10/11/12/13) now reflect the capped, temporally-stationary
magnitudes; read those for the final b/Mc.

---

# Constant-reference-network catalog — the distance cap superseded (2026-06-26)

**Why the cap was not enough.** The ≤60 km cap removed the post-2019 inflation but **censored the
genuine post-2019 completeness gain**: requiring n_used≥3 within 60 km drops the smallest recent events,
forcing the magnitude floor artificially *flat*. A flat floor is itself an artifact — the densifying
network *should* reach lower. Inflation and completeness are entangled in distance (the same far readings
that over-correct are what small events need), so **no per-event distance cut can separate them**.

**The right fix — a fixed reference network** (the textbook approach for secular magnitude studies).
Measure every event with the SAME station set across 2010–2024, so the scale cannot drift. Exactly **5
station-channels span the full period** and sit ≤50 km from the box (the "persistent anchors"):
`KG.MKL.HHZ KG.HDB.HHZ KG.YSB.HHZ KG.CGD.ELZ KG.CHS.HHZ`. Epoch drift handled by the same
median-polish/changepoint machinery (build_constant_network_ml.py), using the **documented sensor breaks** (`responses/sensor_breaks_master.json`: HDB 4, YSB 6,
CHS/MKL 1) **plus the HDB 2014–2015 sensor-failure window as its own epoch** (offset ≈ −1.94 ML; see nb24
record sections) — halves the HDB residual (|res| 0.136→0.073 ML). Correcting vs excluding that window
gives identical UF b/slope/N (HDB is 1 of 5; the window is 7 readings).

- Catalog: `catalog_ml_heo_const.csv` (event_idx, n_const, ml_const_inv, **ml_const** = epoch-corrected).
- Notebook: **`23.UF_constant_network_ML.ipynb`** (generator `build_const_network_nb.py`).

**Result (UF box, n_const ≥ 3, ≈890 events):**

| diagnostic | uncapped | ≤60 km cap | **constant-net** |
|---|---|---|---|
| annual-median slope 2017–24 | +0.018 (inflated) | −0.0015 (flat-censored) | **−0.005 (stationary)** |
| rate ratio M≥1.0 / M≥1.5 (post/pre) | 4.42 / 3.45 | 2.42 / 1.15 | **1.73 / 1.67 (flat across thr)** |
| **UF b (Mc 0.80)** | ~0.9–1.3 | 1.15 | **1.07 ± 0.06** |

The constant-network rate ratio is **~1.7 flat across all thresholds** — the signature of a stationary
scale with a real, scale-consistent rate difference (true productive peak = 2014–2018, not post-2019).

**Two quantities, kept separate (the core reframing):**
1. **Detection completeness Mc(t)** *does* step down at **2016 and 2019** — visible in the full-network
   5th-percentile floor (`+0.20 ≤2013 → 0.00 2014–18 → −0.27 2019+`). MAXC Mc misses the 2nd step
   (saturates ~0.5, confounded by the 2016 aftershock flood); the percentile floor shows it.
2. **Magnitude-scale homogeneity** must be stationary → the constant network delivers it.

**Status / recommendation:** built as a parallel secular catalog + verification notebook (user deferred
promotion). For secular UF b/rate work, quote **b ≈ 1.07 (Mc 0.80)** on the constant-network catalog;
keep the full catalog (`ml_all`) for location/HypoDD. The ≤60 km cap and its downstream b=1.15 are *not
wrong for the inflation alone* but are inferior to the constant network and should be retired if/when the
constant-network catalog is promoted to primary.
