# Post-location analysis — clustering, blast discrimination, subregion, error ellipses

After HYPOINVERSE produces the located catalog (`KS_KG/HypoInv/catalog_<model>_2010_2024.csv`,
written by `catalog_summary.ipynb`), a set of analyses operate on that catalog. The shared,
**tracked** logic lives in `KS_KG/HypoInv/uf_cluster.py` (spatial/temporal blast decluster + map
helpers) and `KS_KG/HypoInv/uf_waveform_similarity.py` (a second, waveform-feature blast screen);
the notebooks that drive them are exploratory and **gitignored** (like all `KS_KG/HypoInv/*.ipynb`
and `*.csv`):

| Notebook | Purpose |
|---|---|
| `03_blast_decluster_hdbscan.ipynb` | 3D HDBSCAN clustering + hour-of-day quarry-blast discrimination → declustered catalog |
| `04_subregion_seismicity.ipynb`    | East-of-fault ("Ulsan Fault zone") subcatalog + long-term spatiotemporal seismicity |
| `05_error_ellipses.ipynb`          | 95% location-error ellipses parsed from the HYPOINVERSE `.prt` covariance |
| `catalog_model_comparison.ipynb`   | stead vs phasenet_plus catalog consistency (counts, maps, depth, time, per-event quality) |
| `04_waveform_similarity_hdb_{HHZ,HHN,HHE}_phasenet_plus.ipynb` | Inter-event waveform-similarity screen for still-remaining quarry blasts at `KG.HDB` (one notebook per component; built by `build_wf_nb.py`) |
| `05_cluster_spacetime_{HHZ,…}_phasenet_plus.ipynb` | Per-family space-time composites — chronological gather + fixed-extent year-coloured epicentre map + cumulative-N(t) curve (built by `build_seq_nb.py`) |

All three are **PARAMS-driven** (edit the first cell, re-run) and `model`-parameterized
(`stead`/`original`/`phasenet_plus`). They `import uf_cluster as uf` (the module sits beside them in
`KS_KG/HypoInv/`).

## `uf_cluster.py` API

**Catalog loading + QC** (the model-agnostic source of truth; used by `catalog_summary` and the
comparison notebook)
- `read_sum(path)` → tidy frame (time, lat, lon, depth, num, gap, rms, erh, erz, qual) from one
  HYPOINVERSE `.sum`. **Robust**: every numeric field is `pd.to_numeric(errors="coerce")`, so a PhaseNet+
  overflow row (`********` in e.g. `SEC`/`ERH`) becomes NaN instead of crashing the whole year (the old
  inline loader did a bare `pd.to_timedelta(df['SEC'])` and raised on PhaseNet+ 2018).
- `QC = dict(erh=5.0, erz=5.0, gap=270.0, num=5)` + `apply_qc(df, qc=QC)` → the **confirmed legacy filter**
  `(erh<5) & (erz<5) & (gap<270) & (num>5)` (strict `<`; `num>5` ≡ ≥6 picks) that produced stead's
  `UF{year}_filtered.sum`. NaN rows fail every gate and drop out.
- `load_catalog(sum_dir, years=range(2010,2025), prefix="UF", filtered=True)` → merge all `UF{y}.sum`
  under `sum_dir` (+ `year` column), optionally QC-filtered. **Config-free** (caller passes `sum_dir =
  config.velmodel_dir(model, velmodel)`); filters in-pandas from the raw `.sum`, so **no precomputed
  `_filtered.sum` is needed** and every picker model is filtered identically (apples-to-apples).
  (`_filtered.sum` is now legacy/optional; the per-year `03.Draw_*` notebooks that wrote it are superseded.)

**Coordinates / clustering**
- `to_cartesian_km(df, epsg="EPSG:32652")` → `(df+[x_km,y_km,z_km], transformer)` (Korea ≈ UTM 52N).
- `get_xyz_weighted(df, depth_weight=1.0)` → `(n,3)` array for clustering (depth-weight knob).
- `run_hdbscan_3d(X, min_cluster_size=30, min_samples=30, …)` → labels (`-1`=noise), via `sklearn.cluster.HDBSCAN`.

**Hour-of-day / blast discrimination**
- `add_kst_columns(df, kst=9)` → adds `hour` (0–23), `hour_kst` (continuous), `dow`.
- `rayleigh_test(hours)` → `{n,R,z,p,peak_hour}` circular non-uniformity (p clamped [0,1]).
- `cluster_blast_stats(df)` → per-cluster table: n, centroid, median_depth, `daytime_frac`, `rayleigh_R/p`, `peak_hour`, `weekend_ratio`.
- `flag_blasts(summary, day_frac_min=0.75, alpha=0.01, peak_in_day=(6,18), …, weekend_max=None)` → adds
  `is_blast`; `weekend_max` (None = off) optionally also requires `weekend_ratio < weekend_max`.
- `decluster(df, summary, keep_noise=True)` → events not in flagged-blast clusters.

`weekend_ratio` = (Sat/Sun event fraction) ÷ (2/7): **1.0** = no weekday/weekend preference (tectonic), **<1**
= avoids weekends (blast-like, ~0.5–0.6 for the residual quarry shots), **>1** = weekend-preferring. Reported
in the cluster + grid tables; an **optional** blast signal via `weekend_max` (default off — daytime + Rayleigh
only), useful for the deep residual blasts where depth doesn't help.

**Spatial residual-blast mask** (catches quarry shots left as noise) — `grid_blast_stats(df, cell_deg=0.02)`
(per-cell hour-of-day stats), `flag_blast_cells(grid, n_min=10, day_frac_min=0.80, alpha=0.01, weekend_max=None)`
(→ `is_quarry_cell`), `decluster_spatial(df, grid)` (drop daytime events in quarry cells), `decluster_full(df,
summary, …, weekend_max=None)` (cluster-level then spatial), `blast_grid_map(df, …)` (gridded daytime-fraction
+ flagged cells).

**Maps** (PyGMT unless noted) — `plot_faults`/`plot_faults_mpl`, `coast_mpl`/`coast_mpl_km` (cartopy 10m
coastline for matplotlib maps), `epicenter_map`, `hour_map` (cyclic cmap on `hour_kst`), `map_by_cluster`
(matplotlib), `annual_maps(df, reg, kind="scatter"|"density", …, density_norm="per_year")` (per-year
small-multiples — depth-coloured epicenters with a shared scale, or density where `density_norm` is
`per_year` (each panel ÷ its own annual max, colorbar = fraction of peak) / `shared` / `shared_log`; edge-only
tick labels), `error_ellipse_map(…, erh_max=None)`/`error_section` (matplotlib). Single equal-aspect maps use
`_match_cbar` so the colorbar height tracks the map.

**`.prt` error ellipses** — `parse_prt`, `load_prt_errors`, `attach_prt_errors`, `error_ellipse`,
`error_ellipse_map`, `error_section` (see below).

Constants: `REGION=[128.5,130.0,35.3,36.5]`, `SUBREGION=[129.25,129.55,35.6,35.9]`, `KST=9`,
`SHALLOW_KM=2.0`, `UTM52N`, `FAULT_TRACE`.

## 1 — 3D HDBSCAN clustering + quarry-blast discrimination (`03`)

Convert lat/lon/depth → Cartesian km, cluster in 3D with HDBSCAN, then flag clusters whose hour-of-day
distribution is anthropogenic. **A cluster is a blast if (3-signal AND):** daytime fraction (06–18 KST)
> `DAY_FRAC_MIN` (0.75), Rayleigh p < `ALPHA` (0.01, statistically non-uniform), and the diurnal peak
falls in daytime. Tectonic clusters are deeper and ~uniform/night-leaning — correctly *not* flagged.
Outputs a **declustered catalog** (blast clusters removed, noise kept as background) + a per-cluster
summary. Empirically (stead, mcs=30): 36 clusters / 24 blasts; 16,771 → ~11,065 events; daytime
fraction 0.64 → 0.47. The before/after epicenter **and cyclic hour-of-day maps** confirm the removal.

**Spatial residual-blast mask (§9b).** Cluster-level declustering misses diffuse quarry blasts that HDBSCAN
labels **noise** (sparse daytime shots — still obvious on the hour-of-day map). Since a quarry is a fixed
location, grid the region (`CELL_DEG=0.02°`), flag **quarry cells** (`n≥10`, daytime_frac>0.80, Rayleigh
p<0.01), and drop the **daytime** events there (clustered or noise) → `catalog_*_blastclean.csv`.
Empirically: 22 quarry cells, +302 events removed (**295 = 98% from noise**), 11,065 → 10,763, daytime
fraction 0.473 → 0.458, and **0 events removed from the subregion** (the east-of-fault zone is blast-free).
The flag is daytime-fraction + Rayleigh only by default (weekend_ratio reported, not gating — but available
as the optional `WEEKEND_MAX` knob); it does **not** require shallow depth — residual blasts are reported
deep (~9 km median) but avoid weekends (ratio ~0.56).

**Final maps + robustness (§9c/§9d).** §9c maps the **fully-filtered blast-clean catalog** (epicenters +
cyclic hour-of-day) — the downstream product, after both filters (ALL 16,771 → declustered 11,065 →
blast-clean 10,763). §9d is a **grid-only-vs-two-step robustness check**: it reruns the spatial quarry-cell
mask on the *full* catalog and compares removals by catalog index. The two approaches agree on the bulk
(Jaccard ≈ 0.87) and the two-step removes a bit more (dense quarries' night/edge events the daytime-only cell
mask keeps); but grid-only on the full catalog lights up many more cells and trims ~49 subregion events,
whereas the two-step's spatial step removes **0** from the subregion — so the **two-step is kept**.

## 2 — East-of-fault subregion seismicity (`04`)

Mask the catalog to the `SUBREGION` box and study long-term patterns on **both** the full and the
fully-filtered (blast-clean, `USE_BLASTCLEAN=True`) catalog (side-by-side): cumulative + annual/monthly rate,
depth cross-sections, hour-of-day (histograms + cyclic maps), along-strike PCA migration, inter-event times,
spatial density, a wide **cumulative-count** curve (§3b, for spotting bursts/rate changes by eye), and
**per-year small-multiples** (`annual_maps`: depth-coloured epicenters + density where each panel is
normalised to its own annual max) for comparing how the spatial distribution evolves year-to-year. Result: only ~3% of
subregion events are blasts (vs 34% region-wide) — the east-of-fault zone is essentially clean tectonic
seismicity (blasts cluster elsewhere); the choice of declustered vs blast-clean input does not change it
(0 subregion events removed by the spatial mask). *Magnitude-based stats (FMD/Mc/b-value) are deferred —
the `.sum` magnitude column is empty (#1 gap).*

## 3 — 95% error ellipses from the HYPOINVERSE `.prt` (`05`)

HYPOINVERSE `.prt` (print) files contain, per located event, a 4×4 covariance matrix (OT/LAT/LON/Z, in
**km²**) and an `ERROR ELLIPSE` line (3 principal axes SERR/AZ/DIP). `parse_prt` extracts the horizontal
covariance `cov_ee`(var_LON, E), `cov_nn`(var_LAT, N), `cov_en`(cov LAT,LON), `cov_zz`(var_Z);
`attach_prt_errors` joins it onto the catalog by rounded time (100 ms) + nearest lat/lon (≈99.9% match).
`error_ellipse_map` draws each event's confidence ellipse in a **true-shape local E–N km frame**
(events + faults + stations transformed together).

**Confidence (verified against the `.sum`):**
- `ERH ≈ 1-σ` horizontal semi-major, `ERZ ≈ √var_Z` (median ratios ≈ 1.0). So **ERH/ERZ are 1-σ**
  (~68% for a single coordinate; this is the answer to "is ERH ~65% CI?" — essentially 1-σ, not a 2-D
  joint confidence).
- A **95% joint 2-D horizontal ellipse** = `k·(1-σ axes)` with `k = √(χ²₂,₀.₉₅) = 2.448`. (A "65% joint"
  ellipse → `k = 1.449`.) Depth error bars use the 1-D normal `k = 1.96` on `σ_z`.
- HYPOINVERSE defines `ERH = max_i(SERR_i·cos DIP_i)` and `ERZ = max_i(SERR_i·sin DIP_i)` (largest
  axis projections); the 2-D **marginal** map ellipse (the proper thing to plot) is ≥ ERH, so the
  median 1-σ-semi-major/ERH ratio is slightly > 1 (≈1.08).

**`.prt` parsing edge cases** (handled in `parse_prt`): covariance fields are fixed-width 8-char and
glue together when large (slice, don't `split`); overflow prints `********` → NaN; origin seconds can be
negative (roll back a minute); longitude carries an `E`/`W` letter; ~18 "overflow" events/yr are junk
locations (20–99 km errors) that all fail the catalog QC (erh/erz ≤ 5 km), so they never enter the
analysis. A handful of 2023 catalog events lack covariance because the filtered `.sum` and `UF2023.prt`
on disk are from slightly different runs (~0.4 s / ~1 km apart) — left unmatched rather than mis-paired.

> `.prt` files are large and **gitignored** (`KS_KG/HypoInv/**/*.prt`); `05` needs them locally at
> `KS_KG/HypoInv/<velmodel>/UF<year>.prt`.

## 4 — Picker-model comparison: stead vs phasenet_plus (`catalog_model_comparison`)

Loads each model's catalog via `uf.load_catalog` (same robust parse + QC) and compares counts, epicenter
maps, depth, time, and per-event quality (`num`/`gap`/`erh`/`erz`, QASR), to check the phasenet_plus run is
consistent with the stead reference.

**Caught a critical bug first.** The initial phasenet_plus HYPOINVERSE run had **no crustal model**: the
`kim2011_{p,s}.crh` files were missing from `models/phasenet_plus/HypoInv/kim2011/`, so hyp1.40 printed
`*** ERROR - CRUST FILE DOES NOT EXIST` and **silently located every event on its built-in default velocity
model** → depths pinned at the `ZTR` trial (~10 km), median **RMS ≈ 3.5 s** (vs stead's 0.08 s), diffuse
epicenters. Detection, picks, association and the `.phs` were all correct — only the location step was
broken. Diagnosis trail: pick times for a shared event matched stead to ~0.05 s; the `.phs` cards matched;
but the `.arc` residuals were multi-second → wrong travel-time predictions → wrong velocity model. Fixed by
`core.ensure_crh(model, velmodel)` (copies the shared `.crh` into the model's velmodel dir, mirroring
`ensure_sta` for the station file) + a **median-RMS > 1 s** warning in `run_hypoinverse_year`; all 15 years
were then re-located. **Post-fix:** filtered median RMS ≈ **0.08 s for both** models; phasenet_plus yields
**~26k filtered vs stead ~17k** (~1.7×) with the same spatial/depth/temporal structure. The unfiltered
phasenet_plus `.sum` still carries the expected marginal-association tail (few-pick, one-sided), removed by QC.

**Caveat — the ~1.7× is NOT a controlled sensitivity measurement.** The two pickers run with **different
detection settings** (`config.py`): stead = SeisBench PhaseNet, `P/S_threshold=0.2`, input **bandpassed
1–40 Hz** (`preprocess_station`); phasenet_plus = EQNet, `min_prob=0.3` (single), input **raw / no filter**
(`PNPLUS_HIGHPASS=0.0`). The two models' probability scales are **not comparable** (a "0.3" in EQNet ≠ a
"0.2" in SeisBench), and the preprocessing differs. Empirically (2020 sample), phasenet_plus emits ~**6× more
picks/day** than stead *despite* its higher nominal threshold, with a different P/S mix (≈44 % P vs stead's
≈23 % — i.e. ~12× more P picks, which is what drives the extra associable events); phasenet_plus's pick
*count* only matches stead's when thresholded near ~0.6. So the larger phasenet_plus catalog reflects **each
picker under its own settings + preprocessing**, not a like-for-like sensitivity test. A fair comparison
would harmonise preprocessing and pick a matched operating point (equal false-alarm rate, or count-matched),
then re-associate/locate — a parameter study, not run here.

## 5 — Waveform-similarity blast screening (`04_waveform_similarity_hdb_*`)

A **second, waveform-feature** pass that catches quarry blasts the spatial/temporal decluster
(§1) misses. Premise: blasts from one pit repeat the same source→path, so at a fixed common station
(`KG.HDB`, ~99 % coverage of the working set) they produce **near-identical waveforms**; tectonic
events do not (genuine repeaters/aftershocks correlate too, but separate out by hour-of-day and
location in the evidence table). It operates on `event_waveforms_ulsanfault/` (per-event SAC, P/S
picks) — **not** the located catalog — so it is an independent line of evidence. Exploratory only:
it surfaces candidate blast families; **it does not remove events**.

**Pipeline** (`uf_waveform_similarity.py`, same KST/Rayleigh/map helpers as `uf_cluster.py`):

1. **Align on P** — two deterministic sources only: a station `pick` (`{ev}_picks.csv`) else a
   synthetic `fallback` (`origin + median P-traveltime`), the fallbacks xcorr-refined to the picked
   stack; picked events keep P at *t*=0.
2. **Window + filter** — a **short P-aligned window `[P−0.5, +7.5] s`** (never the raw 120 s) +
   L2-normalise; several bands (1–10 / 2–8 / 4–12 / 5–15 Hz; PRIMARY 1–10).
3. **Similarity** — N×N **max-lag normalised cross-correlation** per band (small `MAXLAG`, alignment
   already refined).
4. **Cluster** — `linkage(method='average')` on `1−CC`, cut at a **correlation threshold**
   (`fcluster(Z, 1−CC_THRESHOLD, 'distance')`) → data-driven family count; events that never reach
   `CC_THRESHOLD` stay singletons (the non-repeating background).
5. **Evidence + figures** — `cluster_evidence` (intra-cluster `mean_cc`, `spread_km`, `daytime_frac`,
   `rayleigh_p`, `peak_hour`); clustered heatmaps, dendrogram, per-event/per-family **waveform
   gathers** (`plot_cluster_sections` / `plot_cluster_grid`) and **spectrogram gathers**, PyGMT
   cluster + hour-of-day maps, and per-family hour histograms.

**Reading it.** Tight (high `mean_cc`) **and** daytime-concentrated (`daytime_frac` high,
`rayleigh_p` small) **and** spatially compact (`spread_km`) **and** non-uniform hour = the
**`blast_like`** flag (still-remaining quarry blast); tight but night/uniform = tectonic repeater.

**Full-period result (HHZ, CC≥0.6 average linkage, 2716 of 2770 events):** 99 families ≥4 (+1159
singletons); **7 `blast_like` families = 66 candidate events** (tight `mean_cc` 0.69–0.80, daytime,
compact ≤3.7 km), in two pockets ~129.28°E and ~129.40–43°E. The analysis is replicated per
component (`build_wf_nb.py {HHZ|HHN|HHE}`); `cross_component_blast.py` intersects the candidate
**events** across components — **20 events flag on all three** (the robust set), 51 on ≥2.

## Reproducing

```bash
cd KS_KG/HypoInv
# politely on the shared box:
taskset -c 0-7 jupyter nbconvert --to notebook --execute --inplace 03_blast_decluster_hdbscan.ipynb
taskset -c 0-7 jupyter nbconvert --to notebook --execute --inplace 04_subregion_seismicity.ipynb   # needs 03's declustered CSV
taskset -c 0-7 jupyter nbconvert --to notebook --execute --inplace 05_error_ellipses.ipynb          # needs local .prt
# waveform-similarity blast screen (build the per-component notebook, then execute):
python build_wf_nb.py HHZ && jupyter nbconvert --to notebook --execute --inplace 04_waveform_similarity_hdb_HHZ_phasenet_plus.ipynb
python cross_component_blast.py            # cross-component candidate-event intersection (warm caches)
python build_seq_nb.py HHZ && jupyter nbconvert --to notebook --execute --inplace 05_cluster_spacetime_HHZ_phasenet_plus.ipynb  # per-family space-time
```
Outputs (`catalog_*_declustered.csv`, `cluster_summary_*.csv`, `subcatalog_*`, `cluster3d_*.html`,
`wf_similarity_cache/`) land in `KS_KG/HypoInv/` and are gitignored.
