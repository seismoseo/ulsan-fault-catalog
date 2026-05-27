# Post-location analysis — clustering, blast discrimination, subregion, error ellipses

After HYPOINVERSE produces the located catalog (`KS_KG/HypoInv/catalog_<model>_2010_2024.csv`,
written by `catalog_summary.ipynb`), a set of analyses operate on that catalog. The shared,
**tracked** logic lives in `KS_KG/HypoInv/uf_cluster.py`; the notebooks that drive it are
exploratory and **gitignored** (like all `KS_KG/HypoInv/*.ipynb` and `*.csv`):

| Notebook | Purpose |
|---|---|
| `03_blast_decluster_hdbscan.ipynb` | 3D HDBSCAN clustering + hour-of-day quarry-blast discrimination → declustered catalog |
| `04_subregion_seismicity.ipynb`    | East-of-fault ("Ulsan Fault zone") subcatalog + long-term spatiotemporal seismicity |
| `05_error_ellipses.ipynb`          | 95% location-error ellipses parsed from the HYPOINVERSE `.prt` covariance |

All three are **PARAMS-driven** (edit the first cell, re-run) and `model`-parameterized
(`stead`/`original`/`phasenet_plus`). They `import uf_cluster as uf` (the module sits beside them in
`KS_KG/HypoInv/`).

## `uf_cluster.py` API

**Coordinates / clustering**
- `to_cartesian_km(df, epsg="EPSG:32652")` → `(df+[x_km,y_km,z_km], transformer)` (Korea ≈ UTM 52N).
- `get_xyz_weighted(df, depth_weight=1.0)` → `(n,3)` array for clustering (depth-weight knob).
- `run_hdbscan_3d(X, min_cluster_size=30, min_samples=30, …)` → labels (`-1`=noise), via `sklearn.cluster.HDBSCAN`.

**Hour-of-day / blast discrimination**
- `add_kst_columns(df, kst=9)` → adds `hour` (0–23), `hour_kst` (continuous), `dow`.
- `rayleigh_test(hours)` → `{n,R,z,p,peak_hour}` circular non-uniformity (p clamped [0,1]).
- `cluster_blast_stats(df)` → per-cluster table: n, centroid, median_depth, `daytime_frac`, `rayleigh_R/p`, `peak_hour`, `weekend_ratio`.
- `flag_blasts(summary, day_frac_min=0.75, alpha=0.01, peak_in_day=(6,18), …)` → adds `is_blast`.
- `decluster(df, summary, keep_noise=True)` → events not in flagged-blast clusters.

**Spatial residual-blast mask** (catches quarry shots left as noise) — `grid_blast_stats(df, cell_deg=0.02)`
(per-cell hour-of-day stats), `flag_blast_cells(grid, n_min=10, day_frac_min=0.80, alpha=0.01)` (→
`is_quarry_cell`), `decluster_spatial(df, grid)` (drop daytime events in quarry cells), `decluster_full(df,
summary, …)` (cluster-level then spatial), `blast_grid_map(df, …)` (gridded daytime-fraction + flagged cells).

**Maps** (PyGMT unless noted) — `plot_faults`/`plot_faults_mpl`, `coast_mpl`/`coast_mpl_km` (cartopy 10m
coastline for matplotlib maps), `epicenter_map`, `hour_map` (cyclic cmap on `hour_kst`), `map_by_cluster`
(matplotlib), `error_ellipse_map(…, erh_max=None)`/`error_section` (matplotlib).

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
The flag is daytime-fraction + Rayleigh only (weekend_ratio reported, not gating); it does **not** require
shallow depth — residual blasts are reported deep (~9 km median) but avoid weekends (ratio ~0.56).

## 2 — East-of-fault subregion seismicity (`04`)

Mask the catalog to the `SUBREGION` box and study long-term patterns on **both** the full and the
declustered catalog (side-by-side): cumulative + annual/monthly rate, depth cross-sections, hour-of-day
(histograms + cyclic maps), along-strike PCA migration, inter-event times, spatial density. Result: only
~3% of subregion events are blasts (vs 34% region-wide) — the east-of-fault zone is essentially clean
tectonic seismicity (blasts cluster elsewhere). *Magnitude-based stats (FMD/Mc/b-value) are deferred —
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

## Reproducing

```bash
cd KS_KG/HypoInv
# politely on the shared box:
taskset -c 0-7 jupyter nbconvert --to notebook --execute --inplace 03_blast_decluster_hdbscan.ipynb
taskset -c 0-7 jupyter nbconvert --to notebook --execute --inplace 04_subregion_seismicity.ipynb   # needs 03's declustered CSV
taskset -c 0-7 jupyter nbconvert --to notebook --execute --inplace 05_error_ellipses.ipynb          # needs local .prt
```
Outputs (`catalog_*_declustered.csv`, `cluster_summary_*.csv`, `subcatalog_*`, `cluster3d_*.html`) land
in `KS_KG/HypoInv/` and are gitignored.
