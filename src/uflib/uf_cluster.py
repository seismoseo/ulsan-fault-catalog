"""
Ulsan-fault catalog: 3D clustering + quarry-blast discrimination + map helpers.

Shared, importable logic for the analysis notebooks `03_blast_decluster_hdbscan.ipynb`
(HDBSCAN 3D clustering -> per-cluster hour-of-day blast test -> declustering) and
`04_subregion_seismicity.ipynb` (east-of-fault subcatalog + long-term seismicity).

Design: a cluster of earthquakes that fires only during daytime working hours (and avoids
weekends), is shallow, and is spatially compact is almost certainly anthropogenic quarry/
mine **blasts**; genuine tectonic seismicity is ~uniform over 24 h. We cluster in 3D
(lat/lon/depth -> local Cartesian km), then flag clusters whose hour-of-day distribution is
both daytime-concentrated (high daytime fraction) and statistically non-uniform (Rayleigh
test), with the peak in daytime — a 3-signal AND that is robust to a bimodal day/night case.

Reads the filtered catalog CSV directly (no pipeline `config` dependency). The map helpers
are faithful ports of the ones inside `catalog_summary.ipynb`, parameterized so the fault
trace / station table / subregion box are explicit arguments.

Empirical anchor (stead 2010-2024 filtered): daytime(06-17 KST) fraction 0.64 overall, but
0.96 for shallow (<2 km) vs 0.64 deep -> DAY_FRAC_MIN~0.75 separates blast-like from tectonic.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------- defaults
FAULT_TRACE = "/home/msseo/from_PAGO/21.230822_SRC_Workshop/map-fig2/Map2/ss.txt"
DEFAULT_STA = ("/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/models/stead/"
               "HypoInv/STA/UF2024.sta")
REGION    = [128.5, 130.0, 35.3, 36.5]      # full study area [lon0, lon1, lat0, lat1]
SUBREGION = [129.25, 129.55, 35.6, 35.9]    # Ulsan Fault zone (east-of-fault box)
KST = 9                                     # local time = UTC + 9 h
SHALLOW_KM = 2.0
UTM52N = "EPSG:32652"                        # Korea ~ UTM zone 52N (metres)


# --------------------------------------------------------------- catalog loading
# HYPOINVERSE .sum quality filter — the confirmed legacy criterion that produced stead's
# UF{year}_filtered.sum (from the per-year 03.Draw_HypInv_Epicenters_*.ipynb notebooks):
#     (ERH < 5) & (ERZ < 5) & (GAP < 270) & (NUM > 5)
# i.e. strict `<` on the error/gap and num>5 (>= 6 picks). Applied identically to every picker
# model (stead / original / phasenet_plus) so the catalogs are directly comparable, and computed
# in-pandas from the raw .sum so no precomputed _filtered.sum is needed.
QC = dict(erh=5.0, erz=5.0, gap=270.0, num=5)

_SUM_COLMAP = [("LAT", "lat"), ("LON", "lon"), ("DEPTH", "depth"), ("NUM", "num"),
               ("GAP", "gap"), ("RMS", "rms"), ("ERH", "erh"), ("ERZ", "erz"), ("QASR", "qual")]


def _col(df, name):
    """A single column by (stripped) name, taking the first if duplicated — space-padded HYPOINVERSE
    headers can strip to the same name."""
    s = df[name]
    return s.iloc[:, 0] if getattr(s, "ndim", 1) > 1 else s


def read_sum(path):
    """Read one HYPOINVERSE `.sum` (comma-separated, space-padded headers) into a tidy frame:
    time, lat, lon, depth, num, gap, rms, erh, erz, qual.

    Robust to the overflow rows PhaseNet+ can emit (`********` in a numeric field): every numeric
    column is coerced (`errors="coerce"`, bad -> NaN), so one junk row survives as NaN (then dropped
    by `apply_qc`) instead of crashing the whole year — the inline `catalog_summary` loader did a bare
    `pd.to_timedelta(df['SEC'])` and raised on PhaseNet+ 2018. `qual` keeps the QASR string (1st char
    is the A/B/C/D quality)."""
    df = pd.read_csv(path, sep=",")
    df.columns = df.columns.str.strip()
    datecol = [c for c in df.columns if c.startswith("DATE")][0]
    sec = pd.to_numeric(_col(df, "SEC"), errors="coerce")            # coerce overflow / str -> NaN
    out = pd.DataFrame()
    out["time"] = (pd.to_datetime(df[datecol], format="%Y/%m/%d %H:%M", errors="coerce")
                   + pd.to_timedelta(sec, unit="s"))
    for src, dst in _SUM_COLMAP:
        s = _col(df, src)
        out[dst] = s.astype(str).str.strip() if dst == "qual" else pd.to_numeric(s, errors="coerce")
    return out


def apply_qc(df, qc=QC):
    """Filter a catalog by the legacy quality criterion (`QC`): erh<5, erz<5, gap<270, num>5 (strict
    `<`; num>5 == >= 6 picks). NaN rows (overflow / unlocated errors) fail every comparison and drop
    out, which is what we want. Returns a re-indexed copy."""
    m = ((df["erh"] < qc["erh"]) & (df["erz"] < qc["erz"])
         & (df["gap"] < qc["gap"]) & (df["num"] > qc["num"]))
    return df[m].reset_index(drop=True)


def load_catalog(sum_dir, years=range(2010, 2025), prefix="UF", filtered=True, qc=QC):
    """Merge all `{prefix}{year}.sum` under `sum_dir` into one catalog (adds a `year` column),
    optionally applying the QC filter (`apply_qc`). Config-free: the caller resolves `sum_dir`
    (e.g. `config.velmodel_dir(model, velmodel)`). Reads the unfiltered `.sum` and filters in-pandas
    — no precomputed `_filtered.sum` needed, and every model is filtered identically. Returns a
    DataFrame (empty if no files found)."""
    frames = []
    for y in years:
        p = os.path.join(sum_dir, f"{prefix}{y}.sum")
        if os.path.exists(p):
            d = read_sum(p)
            d["year"] = int(y)
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    cat = pd.concat(frames, ignore_index=True)
    return apply_qc(cat, qc) if filtered else cat


# --------------------------------------------------------- coordinate transform
def to_cartesian_km(df, epsg=UTM52N):
    """Add local Cartesian km columns (x_km=E, y_km=N, z_km=depth) via pyproj.

    WGS84 (EPSG:4326) -> `epsg` (UTM, metres) for the horizontal; depth is already km.
    Returns (df_copy, transformer). Mirrors the Ridgecrest `to_utm` idiom but in km."""
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    x, y = t.transform(df["lon"].values, df["lat"].values)
    out = df.copy()
    out["x_km"] = np.asarray(x) / 1000.0
    out["y_km"] = np.asarray(y) / 1000.0
    out["z_km"] = out["depth"].astype(float)
    return out, t


def get_xyz_weighted(df, depth_weight=1.0):
    """(n,3) array [x_km, y_km, z_km*depth_weight] for clustering.

    `depth_weight` scales the vertical axis BEFORE clustering: 1.0 = equal km (Ridgecrest-
    style); <1 loosens depth coupling (useful when ERZ smears planar structures); >1 tightens."""
    return np.column_stack([df["x_km"].values, df["y_km"].values,
                            df["z_km"].values * float(depth_weight)])


# ------------------------------------------------------------------- clustering
def run_hdbscan_3d(X, min_cluster_size=30, min_samples=30,
                   cluster_selection_epsilon=0.0, method="eom", verbose=True):
    """sklearn HDBSCAN on the (n,3) Cartesian-km cloud. Returns integer labels (-1=noise)."""
    from sklearn.cluster import HDBSCAN
    labels = HDBSCAN(
        min_cluster_size=int(min_cluster_size), min_samples=int(min_samples),
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        cluster_selection_method=method, metric="euclidean", copy=True,
    ).fit_predict(X)
    if verbose:
        ids, counts = np.unique(labels[labels != -1], return_counts=True)
        order = np.argsort(counts)[::-1]
        print(f"HDBSCAN: {len(ids)} clusters, {(labels == -1).sum()} noise / {len(labels)} events; "
              f"sizes (top 10): {[int(c) for c in counts[order][:10]]}")
    return labels


# --------------------------------------------------------------- time + circular
def add_kst_columns(df, kst=KST):
    """Add local-time columns: `hour` (int 0-23), `hour_kst` (continuous), `dow` (0=Mon)."""
    t = pd.to_datetime(df["time"], utc=True) + pd.Timedelta(hours=kst)
    df = df.copy()
    df["hour"] = t.dt.hour
    df["hour_kst"] = t.dt.hour + t.dt.minute / 60.0
    df["dow"] = t.dt.dayofweek
    return df


def rayleigh_test(hours):
    """Rayleigh test for non-uniformity of a circular (hour-of-day) sample.

    angle = hour/24 * 2pi; R = |mean resultant|; z = n R^2;
    p ~ exp(-z)*(1 + (2z - z^2)/(4n))  (Zar small-sample correction).
    Returns dict(n, R, z, p, peak_hour). Small p + high R => concentrated (blast-like);
    p ~ 1, R ~ 0 => uniform (tectonic)."""
    h = np.asarray(hours, dtype=float)
    n = h.size
    if n == 0:
        return dict(n=0, R=np.nan, z=np.nan, p=np.nan, peak_hour=np.nan)
    theta = h / 24.0 * 2.0 * np.pi
    C, S = np.cos(theta).mean(), np.sin(theta).mean()
    R = float(np.hypot(C, S))
    z = n * R * R
    # Zar small-sample correction; clamp to [0,1] (the polynomial overshoots negative for
    # large z, where p is effectively 0 anyway).
    p = float(np.clip(np.exp(-z) * (1.0 + (2.0 * z - z * z) / (4.0 * n)), 0.0, 1.0))
    peak_hour = float((np.arctan2(S, C) % (2.0 * np.pi)) / (2.0 * np.pi) * 24.0)
    return dict(n=n, R=R, z=float(z), p=p, peak_hour=peak_hour)


# ------------------------------------------------------- per-cluster statistics
def cluster_blast_stats(df, label_col="cluster", kst=KST, day=(6, 17)):
    """Per-cluster (incl. -1 noise) hour-of-day + depth + weekend statistics.

    Columns: cluster, n, lat_centroid, lon_centroid, depth_centroid, median_depth,
    daytime_frac (06-18 KST), rayleigh_R, rayleigh_p, peak_hour, weekend_ratio
    (fraction on Sat/Sun normalised by 2/7; <1 = avoids weekends)."""
    if "hour" not in df.columns:
        df = add_kst_columns(df, kst)
    rows = []
    for cid, g in df.groupby(label_col):
        rt = rayleigh_test(g["hour"].values)
        rows.append(dict(
            cluster=int(cid), n=int(len(g)),
            lat_centroid=g["lat"].mean(), lon_centroid=g["lon"].mean(),
            depth_centroid=g["depth"].mean(), median_depth=g["depth"].median(),
            daytime_frac=float(g["hour"].between(day[0], day[1]).mean()),
            rayleigh_R=rt["R"], rayleigh_p=rt["p"], peak_hour=rt["peak_hour"],
            weekend_ratio=float((g["dow"] >= 5).mean() / (2.0 / 7.0)),
        ))
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def flag_blasts(summary, day_frac_min=0.75, alpha=0.01, peak_in_day=(6, 18),
                require_shallow=False, shallow_med_km=5.0, weekend_max=None):
    """Add `is_blast`: a non-noise cluster that is daytime-concentrated (daytime_frac >
    day_frac_min), statistically non-uniform (rayleigh_p < alpha), and peaks in daytime.
    Optionally also require a shallow median depth (`require_shallow`) and/or weekend
    avoidance (`weekend_max`: require weekend_ratio < weekend_max; None = off) as
    corroborating anthropogenic signals."""
    s = summary.copy()
    is_blast = (
        (s["cluster"] != -1)
        & (s["daytime_frac"] > day_frac_min)
        & (s["rayleigh_p"] < alpha)
        & (s["peak_hour"] >= peak_in_day[0]) & (s["peak_hour"] < peak_in_day[1])
    )
    if require_shallow:
        is_blast &= s["median_depth"] < shallow_med_km
    if weekend_max is not None:
        is_blast &= s["weekend_ratio"] < weekend_max
    s["is_blast"] = is_blast
    return s


def decluster(df, summary, label_col="cluster", keep_noise=True):
    """Return events NOT in any flagged-blast cluster. Noise (-1) is kept as background
    seismicity by default (a flagged blast is a coherent daytime cluster, not diffuse noise)."""
    blast_ids = set(summary.loc[summary.get("is_blast", False), "cluster"].astype(int))
    mask = ~df[label_col].isin(blast_ids)
    if not keep_noise:
        mask &= df[label_col] != -1
    return df[mask].copy()


# ----------------------------------------- spatial residual-blast (quarry-cell) mask
# Cluster-level declustering misses quarry blasts that HDBSCAN leaves as NOISE (diffuse
# daytime shots that never form a dense cluster). A quarry is a FIXED LOCATION, so we grid
# the region, find cells whose hour-of-day is daytime-concentrated + non-uniform ("quarry
# cells"), and drop the daytime events there (clustered or noise). NOTE: residual blasts are
# reported deep (median ~9 km) but avoid weekends — so the flag does NOT require shallow depth.
def _cell_index(df, cell_deg, reg):
    gi = np.floor((df["lon"] - reg[0]) / cell_deg).astype(int)
    gj = np.floor((df["lat"] - reg[2]) / cell_deg).astype(int)
    return gi, gj


def grid_blast_stats(df, cell_deg=0.02, reg=REGION, kst=KST, day=(6, 17), label_col="cluster"):
    """Per spatial-cell hour-of-day stats over `reg` (lon/lat grid, `cell_deg`°). Columns:
    gi, gj, lon_c, lat_c, n, n_noise, daytime_frac (06-18 KST), rayleigh_p, peak_hour,
    median_depth, weekend_ratio. `df` needs hour/dow (add_kst_columns)."""
    if "hour" not in df.columns:
        df = add_kst_columns(df, kst)
    d = df[df["lon"].between(reg[0], reg[1]) & df["lat"].between(reg[2], reg[3])].copy()
    d["gi"], d["gj"] = _cell_index(d, cell_deg, reg)
    rows = []
    for (gi, gj), g in d.groupby(["gi", "gj"]):
        rt = rayleigh_test(g["hour"].values)
        rows.append(dict(
            gi=int(gi), gj=int(gj),
            lon_c=reg[0] + (gi + 0.5) * cell_deg, lat_c=reg[2] + (gj + 0.5) * cell_deg,
            n=int(len(g)), n_noise=int((g[label_col] == -1).sum()) if label_col in g else 0,
            daytime_frac=float(g["hour"].between(day[0], day[1]).mean()),
            rayleigh_p=rt["p"], peak_hour=rt["peak_hour"],
            median_depth=float(g["depth"].median()),
            weekend_ratio=float((g["dow"] >= 5).mean() / (2.0 / 7.0)),
        ))
    return pd.DataFrame(rows)


def flag_blast_cells(grid, n_min=10, day_frac_min=0.80, alpha=0.01, peak_in_day=(6, 18),
                     weekend_max=None):
    """Add `is_quarry_cell` = n>=n_min & daytime_frac>day_frac_min & rayleigh_p<alpha & peak
    in daytime. `weekend_ratio` is reported in `grid` for transparency and does NOT gate
    unless `weekend_max` is set (then also require weekend_ratio < weekend_max)."""
    g = grid.copy()
    g["is_quarry_cell"] = (
        (g["n"] >= n_min) & (g["daytime_frac"] > day_frac_min) & (g["rayleigh_p"] < alpha)
        & (g["peak_hour"] >= peak_in_day[0]) & (g["peak_hour"] < peak_in_day[1])
    )
    if weekend_max is not None:
        g["is_quarry_cell"] &= g["weekend_ratio"] < weekend_max
    return g


def decluster_spatial(df, grid, cell_deg=0.02, reg=REGION, kst=KST, daytime=(6, 17)):
    """Drop DAYTIME (06-18 KST) events that fall in a flagged quarry cell; night events in
    those cells are kept. Returns the cleaned df."""
    if "hour" not in df.columns:
        df = add_kst_columns(df, kst)
    quarry = set(zip(grid.loc[grid["is_quarry_cell"], "gi"],
                     grid.loc[grid["is_quarry_cell"], "gj"]))
    gi, gj = _cell_index(df, cell_deg, reg)
    in_quarry = pd.Series(list(zip(gi, gj)), index=df.index).isin(quarry)
    drop = in_quarry & df["hour"].between(daytime[0], daytime[1])
    return df[~drop].copy()


def decluster_full(df, summary, cell_deg=0.02, reg=REGION, kst=KST, daytime=(6, 17),
                   n_min=10, day_frac_min=0.80, alpha=0.01, keep_noise=True,
                   weekend_max=None):
    """Cluster-level decluster THEN the spatial quarry-cell mask. Returns (clean_df, grid).
    `weekend_max` (None = off) is forwarded to `flag_blast_cells`."""
    d = decluster(df, summary, keep_noise=keep_noise)
    grid = flag_blast_cells(grid_blast_stats(d, cell_deg, reg, kst),
                            n_min, day_frac_min, alpha, weekend_max=weekend_max)
    return decluster_spatial(d, grid, cell_deg, reg, kst, daytime), grid


def blast_grid_map(df, reg=REGION, cell_deg=0.02, kst=KST, n_min=10, day_frac_min=0.80,
                   alpha=0.01, subregion=SUBREGION, fault_trace=FAULT_TRACE, ax=None,
                   weekend_max=None):
    """matplotlib pcolormesh of per-cell daytime fraction; flagged quarry cells outlined red,
    cells with n<n_min greyed; faults + subregion box + coastline overlaid."""
    import matplotlib.pyplot as plt
    grid = flag_blast_cells(grid_blast_stats(df, cell_deg, reg, kst),
                            n_min, day_frac_min, alpha, weekend_max=weekend_max)
    ni = int(np.ceil((reg[1] - reg[0]) / cell_deg)); nj = int(np.ceil((reg[3] - reg[2]) / cell_deg))
    Z = np.full((nj, ni), np.nan)
    for r in grid.itertuples():
        if 0 <= r.gj < nj and 0 <= r.gi < ni and r.n >= n_min:
            Z[r.gj, r.gi] = r.daytime_frac
    xe = reg[0] + np.arange(ni + 1) * cell_deg
    ye = reg[2] + np.arange(nj + 1) * cell_deg
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8), dpi=120)
    pm = ax.pcolormesh(xe, ye, Z, cmap="coolwarm", vmin=0, vmax=1, zorder=1)
    _match_cbar(pm, ax, "daytime fraction (06–18 KST)")
    coast_mpl(ax, reg, color="0.35", zorder=2.5)
    plot_faults_mpl(ax, fault_trace, color="k", lw=0.7, zorder=2)
    for r in grid[grid["is_quarry_cell"]].itertuples():     # outline quarry cells
        ax.add_patch(plt.Rectangle((reg[0] + r.gi * cell_deg, reg[2] + r.gj * cell_deg),
                                   cell_deg, cell_deg, fill=False, edgecolor="red", lw=1.3, zorder=3))
    if subregion is not None:
        bl, ba = _subregion_box(subregion); ax.plot(bl, ba, "b-", lw=1.5, zorder=4)
    ax.set(xlim=reg[:2], ylim=reg[2:], xlabel="longitude", ylabel="latitude",
           title=f"daytime-fraction grid (red = {int(grid.is_quarry_cell.sum())} quarry cells)")
    ax.set_aspect("equal", "box")
    return ax.figure


# --------------------------------------------------------------- map utilities
def read_fault_segments(fault_trace=FAULT_TRACE):
    """GMT multi-segment fault file -> list of (n,2) [lat, lon] arrays (segments split by '>')."""
    if not os.path.exists(fault_trace):
        return []
    segs, seg = [], []
    for ln in open(fault_trace):
        if ln.startswith(">"):
            if seg:
                segs.append(np.array(seg)); seg = []
        else:
            p = ln.split()
            if len(p) >= 2:
                try:
                    seg.append([float(p[0]), float(p[1])])     # [lat, lon]
                except ValueError:
                    pass
    if seg:
        segs.append(np.array(seg))
    return segs


def load_stations(path=DEFAULT_STA):
    try:
        return pd.read_csv(path, sep=",",
                           names=["Networkcode", "Latitude", "Longitude", "Elevation", "Weight"])
    except Exception:
        return pd.DataFrame(columns=["Latitude", "Longitude"])


STA = load_stations()                       # loaded once; maps fall back to this


def plot_faults(fig, fault_trace=FAULT_TRACE):
    """PyGMT: overlay fault traces (port of catalog_summary `_plot_faults`)."""
    for i, s in enumerate(read_fault_segments(fault_trace)):
        fig.plot(x=s[:, 1], y=s[:, 0], pen="1p,black",
                 label="Fault trace" if i == 0 else None)


def plot_faults_mpl(ax, fault_trace=FAULT_TRACE, **kw):
    """matplotlib: overlay fault traces."""
    kw = dict(dict(color="0.3", lw=0.5, zorder=1), **kw)
    for s in read_fault_segments(fault_trace):
        ax.plot(s[:, 1], s[:, 0], **kw)


# ---- coastlines for matplotlib maps (cartopy 10m; matches the PyGMT fig.coast maps) ----
_COAST_CACHE: dict = {}


def _coast_segments(reg, pad=0.1):
    """10m coastline (cartopy NaturalEarth) clipped to `reg` (±pad°) as a list of (n,2)
    [lon,lat] arrays. Cached per region. Returns [] gracefully if cartopy/the cached
    shapefile is unavailable (so maps still render). Only the coastLINE is used — the 10m
    land/ocean polygons are not cached and would trigger a download."""
    key = tuple(np.round(reg, 3))
    if key in _COAST_CACHE:
        return _COAST_CACHE[key]
    segs = []
    try:
        import cartopy.feature as cf
        from shapely.geometry import box
        bb = box(reg[0] - pad, reg[2] - pad, reg[1] + pad, reg[3] + pad)
        for g in cf.NaturalEarthFeature("physical", "coastline", "10m").geometries():
            if not g.intersects(bb):
                continue
            clip = g.intersection(bb)
            for p in getattr(clip, "geoms", [clip]):
                xy = np.asarray(p.coords)
                if len(xy) >= 2:
                    segs.append(xy)
    except Exception as exc:                                # noqa: BLE001
        print(f"[coast] coastline unavailable ({exc}); skipping")
    _COAST_CACHE[key] = segs
    return segs


def coast_mpl(ax, reg, color="0.5", lw=0.6, zorder=0.5, **kw):
    """Draw the 10m coastline on a lon/lat matplotlib Axes (beneath the data)."""
    for s in _coast_segments(reg):
        ax.plot(s[:, 0], s[:, 1], color=color, lw=lw, zorder=zorder, **kw)


def coast_mpl_km(ax, reg, transformer, color="0.5", lw=0.6, zorder=0.5, **kw):
    """Draw the 10m coastline on a local E–N km Axes, transforming lon/lat via `transformer`
    (the pyproj Transformer returned by to_cartesian_km)."""
    for s in _coast_segments(reg):
        cx, cy = transformer.transform(s[:, 0], s[:, 1])
        ax.plot(np.asarray(cx) / 1000, np.asarray(cy) / 1000, color=color, lw=lw,
                zorder=zorder, **kw)


def _subregion_box(subregion):
    bl = [subregion[0], subregion[1], subregion[1], subregion[0], subregion[0]]
    ba = [subregion[2], subregion[2], subregion[3], subregion[3], subregion[2]]
    return bl, ba


def _match_cbar(mappable, ax, label, size="4.5%", pad=0.1):
    """Attach a colorbar whose height tracks `ax` (via make_axes_locatable). For equal-aspect
    maps a default colorbar over/undershoots the map height; this keeps them matched."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes("right", size=size, pad=pad)
    return ax.figure.colorbar(mappable, cax=cax, label=label)


def epicenter_map(df, reg, title, dmax=50.0, subregion=SUBREGION, sta=None,
                  fault_trace=FAULT_TRACE):
    """PyGMT epicenter map coloured by depth (port of catalog_summary `epicenter_map`)."""
    import pygmt as pmt
    sta = STA if sta is None else sta
    fig = pmt.Figure()
    pmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")
    fig.basemap(region=reg, projection="M15c", frame=["a", f"+t{title}"])
    fig.coast(land="white", water="lightblue", shorelines=True)
    pmt.makecpt(cmap="viridis", series=[0.0, dmax], reverse=True)
    plot_faults(fig, fault_trace)
    fig.plot(x=df["lon"], y=df["lat"], fill=df["depth"], cmap=True, style="c0.10c", pen="0.1p,black")
    if len(sta):
        fig.plot(x=sta.Longitude, y=sta.Latitude, style="i0.4c", fill="red", pen="1p,black", label="Stations")
    if subregion is not None:
        bl, ba = _subregion_box(subregion)
        fig.plot(x=bl, y=ba, pen="1.5p,blue,solid")
    fig.colorbar(frame=["x+lDepth (km)"])
    return fig


def hour_map(df, reg, title, subregion=SUBREGION, sta=None, fault_trace=FAULT_TRACE,
             hour_col="hour_kst"):
    """PyGMT map coloured by hour-of-day with a cyclic colormap (port of `hour_map`)."""
    import pygmt as pmt
    sta = STA if sta is None else sta
    fig = pmt.Figure()
    pmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")
    fig.basemap(region=reg, projection="M15c", frame=["a", f"+t{title}"])
    fig.coast(land="white", water="lightblue", shorelines=True)
    pmt.makecpt(cmap="cyclic", series=[0, 24, 1], continuous=True)
    plot_faults(fig, fault_trace)
    fig.plot(x=df["lon"], y=df["lat"], fill=df[hour_col], cmap=True, style="c0.10c", pen="0.1p,black")
    if len(sta):
        fig.plot(x=sta.Longitude, y=sta.Latitude, style="i0.4c", fill="black", pen="0.5p,white")
    if subregion is not None:
        bl, ba = _subregion_box(subregion)
        fig.plot(x=bl, y=ba, pen="1.5p,blue,solid")
    fig.colorbar(frame=["a6", "x+lHour of day (KST)"])
    return fig


def map_by_cluster(df, reg, title, label_col="cluster", subregion=SUBREGION,
                   fault_trace=FAULT_TRACE, ax=None, noise_color="0.8", s=6):
    """Fast matplotlib epicenter map coloured by cluster id (noise = light gray)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7), dpi=120)
    coast_mpl(ax, reg)
    plot_faults_mpl(ax, fault_trace)
    noise = df[df[label_col] == -1]
    clus = df[df[label_col] != -1]
    ax.scatter(noise["lon"], noise["lat"], s=s * 0.5, c=noise_color, lw=0, alpha=0.4,
               label=f"noise ({len(noise)})", zorder=2)
    if len(clus):
        ax.scatter(clus["lon"], clus["lat"], s=s, c=clus[label_col].values, cmap="tab20",
                   lw=0, zorder=3)
    if subregion is not None:
        bl, ba = _subregion_box(subregion)
        ax.plot(bl, ba, "b-", lw=1.5, zorder=4)
    ax.set(xlim=reg[:2], ylim=reg[2:], xlabel="longitude", ylabel="latitude", title=title)
    ax.set_aspect("equal", "box")
    return ax.figure


def _panel_decor(ax, reg, subregion, title, show_x=True, show_y=True):
    """Shared per-panel framing for small-multiple maps (extent, box, aspect, small fonts).
    Tick labels are shown only on edge panels (`show_x`/`show_y`) to avoid overlap; both axes
    are limited to ~3 ticks."""
    from matplotlib.ticker import MaxNLocator
    if subregion is not None:
        bl, ba = _subregion_box(subregion)
        ax.plot(bl, ba, "b-", lw=1.0, zorder=4)
    ax.set(xlim=reg[:2], ylim=reg[2:])
    ax.set_aspect("equal", "box")
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=7)
    if not show_x:
        ax.set_xticklabels([])
    if not show_y:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=8)


def annual_maps(df, reg, kind="scatter", ncols=5, color_by="depth", bins=30, years=None,
                subregion=None, fault_trace=FAULT_TRACE, share_clim=True, cmap=None,
                dmax=None, panel=2.6, density_norm="per_year"):
    """Small-multiple maps, one panel per calendar year, over `reg` — for comparing the
    temporal variation of the spatial distribution.

    `kind="scatter"`: epicenters coloured by `color_by` (e.g. depth), with a shared colour
    norm across years (`share_clim`). `kind="density"`: per-year 2-D histogram (bins×bins
    over `reg`); `density_norm` sets the colour scale:
      - `"per_year"` (default): each panel normalised to its OWN year's max → colorbar is the
        *fraction of that year's peak* (0–1), so every year's spatial pattern is equally
        visible (absolute counts NOT comparable across years).
      - `"shared"`: one shared `vmax` (absolute counts) across all years.
      - `"shared_log"`: shared log-scaled counts (comparable, lifts quiet years).
    Every panel draws the coastline + fault traces (+ optional `subregion` box). Tick labels
    appear only on the left column + bottom row (no overlap). Years with no events render as
    empty framed axes; spare axes are hidden. Returns the figure."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    d = df.copy()
    d["_yr"] = pd.to_datetime(d["time"], utc=True).dt.year
    if years is None:
        years = list(range(int(d["_yr"].min()), int(d["_yr"].max()) + 1)) if len(d) else []
    nrow = int(np.ceil(len(years) / ncols)) if years else 1
    fig, axes = plt.subplots(nrow, ncols, figsize=(panel * ncols, panel * nrow + 0.4),
                             dpi=120, squeeze=False)
    axes = axes.ravel()
    rng = [reg[:2], reg[2:]]
    n_last_row = len(years) - ncols      # panels at index >= this are on the bottom content row

    def _edges(i):
        return dict(show_x=(i >= n_last_row), show_y=(i % ncols == 0))

    if kind == "scatter":
        cm = plt.get_cmap(cmap or "viridis_r")
        vals = d[color_by].values
        norm = None
        if share_clim and len(vals):
            vmax = dmax if dmax is not None else np.nanpercentile(vals, 98)
            norm = mpl.colors.Normalize(vmin=np.nanpercentile(vals, 2), vmax=vmax)
        sc = None
        for i, (ax, yr) in enumerate(zip(axes, years)):
            coast_mpl(ax, reg)
            plot_faults_mpl(ax, fault_trace)
            g = d[d["_yr"] == yr]
            if len(g):
                sc = ax.scatter(g["lon"], g["lat"], c=g[color_by], cmap=cm, norm=norm,
                                s=7, lw=0, zorder=3)
            _panel_decor(ax, reg, subregion, f"{yr} (n={len(g)})", **_edges(i))
        if sc is not None:
            fig.colorbar(sc, ax=axes.tolist(), label=color_by, shrink=0.6)
    elif kind == "density":
        cm = cmap or "hot_r"
        if density_norm == "shared":           # one absolute vmax across years
            vmax = dmax
            if vmax is None:
                vmax = max((np.histogram2d(d[d._yr == y].lon, d[d._yr == y].lat, bins=bins,
                                           range=rng)[0].max() for y in years if (d._yr == y).any()),
                           default=0) or None
            norm = None
        elif density_norm == "shared_log":
            norm = mpl.colors.LogNorm(vmin=1, vmax=dmax)
        im = None
        for i, (ax, yr) in enumerate(zip(axes, years)):
            g = d[d["_yr"] == yr]
            if len(g):
                H, xe, ye = np.histogram2d(g["lon"], g["lat"], bins=bins, range=rng)
                if density_norm == "per_year":
                    Z = H / H.max() if H.max() > 0 else H
                    im = ax.pcolormesh(xe, ye, Z.T, cmap=cm, vmin=0, vmax=1, zorder=1)
                elif density_norm == "shared":
                    im = ax.pcolormesh(xe, ye, H.T, cmap=cm, vmin=0, vmax=vmax, zorder=1)
                else:                           # shared_log
                    im = ax.pcolormesh(xe, ye, np.where(H > 0, H, np.nan).T, cmap=cm,
                                       norm=norm, zorder=1)
            coast_mpl(ax, reg, color="0.25", zorder=2.5)
            plot_faults_mpl(ax, fault_trace, color="cyan", lw=0.4, alpha=0.6, zorder=2)
            _panel_decor(ax, reg, subregion, f"{yr} (n={len(g)})", **_edges(i))
        if im is not None:
            lab = {"per_year": "fraction of annual max", "shared": "events / cell",
                   "shared_log": "events / cell (log)"}[density_norm]
            fig.colorbar(im, ax=axes.tolist(), label=lab, shrink=0.6)
    else:
        raise ValueError("kind must be 'scatter' or 'density'")

    for ax in axes[len(years):]:
        ax.axis("off")
    return fig


# ============================================================================
# HYPOINVERSE .prt error-ellipse parsing + 95% confidence ellipse mapping
#
# Each located event's .prt block has a 4x4 covariance (OT,LAT,LON,Z; km^2), an
# `ERROR ELLIPSE` line (3 principal axes SERR/AZ/DIP), and a summary line. VERIFIED:
# ERH ~= 1-sigma horizontal semi-major, ERZ ~= sqrt(var_Z) (median ratios ~1.0), so
# ERH/ERZ are 1-sigma (~68% per coord). A 95% JOINT 2-D horizontal ellipse scales the
# 1-sigma axes by k = sqrt(chi2.ppf(0.95, 2)) = 2.448 ("65%" -> 1.449; 1-D depth 95% ->
# 1.96 sigma_z). Covariance maps: cov_ee=var_LON(E), cov_nn=var_LAT(N), cov_en=cov(LAT,LON),
# cov_zz=var_Z. .prt is gitignored (regenerate/keep locally).
# ============================================================================
_HDR_RE = re.compile(r"^\s*\d{1,2}\s+[A-Z]{3}\s+\d{4},.*ID NO\.\s*(\d+)")
_COVROW_RE = re.compile(r"^\s*(OT|LAT|LON|Z)\s+\(")
_ELL_RE = re.compile(r"<\s*([\d.]+)\s+(\d+)\s+(\d+)\s*>")
_SUM_RE = re.compile(
    r"^\s*(\d{4})-(\d{2})-(\d{2})\s+(\d{2})(\d{2})\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+\.\d+)"
    r"\s+(\d+)([EW])\s*(\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)")


def _cov_fields(line):
    """The 2nd (...) group of a covariance row, sliced into 4 fixed-width 8-char floats
    (handles glued large values and '********' overflow -> NaN)."""
    groups = re.findall(r"\(([^)]*)\)", line)
    if len(groups) < 2:
        return [np.nan] * 4
    c = groups[1]
    out = []
    for i in range(4):
        s = c[i * 8:(i + 1) * 8].strip()
        out.append(float(s) if s and "*" not in s else np.nan)
    return out


def parse_prt(prt_path):
    """Parse a HYPOINVERSE .prt into a per-event DataFrame with location + covariance.

    Columns: id, time(UTC), lat, lon, depth, rms, erh, erz, cov_ee(var_LON), cov_nn(var_LAT),
    cov_en(cov LAT,LON), cov_zz(var_Z), cov_ez(LON,Z), cov_nz(LAT,Z), and ellipse axes
    serr1..3/az1..3/dip1..3. Only events that reach a summary line are emitted (the ~21
    'CANT SOLVE' headers per year self-exclude)."""
    lines = open(prt_path, errors="replace").read().splitlines()
    rows, cur, last_id = [], None, None
    for ln in lines:
        h = _HDR_RE.match(ln)
        if h:
            last_id = int(h.group(1))
            continue
        if "EIGENVALUES" in ln:
            cur = dict(id=last_id, cov={}, axes=[])
            continue
        if cur is not None and not cur.get("cov_done") and _COVROW_RE.match(ln):
            lab = _COVROW_RE.match(ln).group(1)
            cur["cov"][lab] = _cov_fields(ln)
            if lab == "Z":
                cur["cov_done"] = True
            continue
        if cur is not None and "ERROR ELLIPSE" in ln:
            cur["axes"] = _ELL_RE.findall(ln)        # list of (serr, az, dip) strings
            continue
        m = _SUM_RE.match(ln)
        if m and cur is not None:
            (yr, mo, da, hh, mm, ss, latd, latm, lond, ew, lonm,
             dep, rms, erh, erz) = m.groups()
            import datetime as _dt
            t = (_dt.datetime(int(yr), int(mo), int(da), int(hh), int(mm), 0,
                              tzinfo=_dt.timezone.utc) + _dt.timedelta(seconds=float(ss)))
            lat = int(latd) + float(latm) / 60.0
            lon = (int(lond) + float(lonm) / 60.0) * (1.0 if ew == "E" else -1.0)
            cov = cur["cov"]
            def _g(lab, j):
                v = cov.get(lab)
                return v[j] if v is not None else np.nan
            ax = cur["axes"]
            rec = dict(id=cur["id"], time=pd.Timestamp(t), lat=lat, lon=lon,
                       depth=float(dep), rms=float(rms), erh=float(erh), erz=float(erz),
                       cov_nn=_g("LAT", 1), cov_ee=_g("LON", 2), cov_en=_g("LAT", 2),
                       cov_zz=_g("Z", 3), cov_nz=_g("LAT", 3), cov_ez=_g("LON", 3))
            for i in range(3):
                if i < len(ax):
                    rec[f"serr{i+1}"], rec[f"az{i+1}"], rec[f"dip{i+1}"] = map(float, ax[i])
            rows.append(rec)
            cur = None
    return pd.DataFrame(rows)


def load_prt_errors(prt_dir, years, prefix="UF"):
    """Concatenate parse_prt over UF<year>.prt in prt_dir (warns + skips missing years)."""
    frames = []
    for y in years:
        p = os.path.join(prt_dir, f"{prefix}{y}.prt")
        if os.path.exists(p):
            d = parse_prt(p); d["year"] = y; frames.append(d)
        else:
            print(f"[load_prt_errors] missing (gitignored?): {p}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def attach_prt_errors(cat, prt_df, max_deg=0.02):
    """Left-join .prt covariance onto the catalog by rounded time (100 ms) + nearest
    lat/lon (guards same-0.1s different-event collisions). Returns cat + cov_* columns;
    prints the match fraction."""
    cov_cols = ["cov_ee", "cov_nn", "cov_en", "cov_zz", "id", "erh", "erz"]
    c = cat.copy().reset_index(drop=True)
    c["_ci"] = np.arange(len(c))
    c["_tk"] = pd.to_datetime(c["time"], utc=True).dt.round("100ms")
    p = prt_df.copy()
    p["_tk"] = pd.to_datetime(p["time"], utc=True).dt.round("100ms")
    p = p.rename(columns={"lat": "lat_p", "lon": "lon_p", "erh": "erh_p", "erz": "erz_p"})
    m = c.merge(p[["_tk", "lat_p", "lon_p", "cov_ee", "cov_nn", "cov_en", "cov_zz",
                   "id", "erh_p", "erz_p"]], on="_tk", how="left")
    bad = (m["lat_p"].notna() & (((m["lat"] - m["lat_p"]).abs() > max_deg) |
                                 ((m["lon"] - m["lon_p"]).abs() > max_deg)))
    m.loc[bad, ["cov_ee", "cov_nn", "cov_en", "cov_zz"]] = np.nan
    m["_d2"] = (m["lat"] - m["lat_p"]) ** 2 + (m["lon"] - m["lon_p"]) ** 2
    m = (m.sort_values("_d2", na_position="last").drop_duplicates("_ci", keep="first")
           .sort_values("_ci"))
    out = cat.copy()
    for col in ["cov_ee", "cov_nn", "cov_en", "cov_zz", "id", "erh_p", "erz_p"]:
        out[col] = m.set_index("_ci")[col].reindex(np.arange(len(cat))).values
    frac = out["cov_ee"].notna().mean()
    print(f"[attach_prt_errors] matched covariance for {int(out['cov_ee'].notna().sum())}/"
          f"{len(out)} events ({frac:.1%})")
    return out


def error_ellipse(cov_ee, cov_en, cov_nn, confidence=0.95):
    """Horizontal error ellipse from the 2x2 covariance [[ee,en],[en,nn]] (km^2).
    Returns (semi_major_km, semi_minor_km, angle_deg_CCW_from_E) at the given JOINT 2-D
    confidence (k = sqrt(chi2.ppf(confidence, 2))). NaN-safe."""
    if any(np.isnan(v) for v in (cov_ee, cov_en, cov_nn)):
        return (np.nan, np.nan, np.nan)
    from scipy.stats import chi2
    tr = cov_ee + cov_nn
    d = np.hypot((cov_ee - cov_nn) / 2.0, cov_en)
    l1, l2 = tr / 2.0 + d, max(tr / 2.0 - d, 0.0)
    k = np.sqrt(chi2.ppf(confidence, 2))
    ang = 0.5 * np.degrees(np.arctan2(2.0 * cov_en, cov_ee - cov_nn))
    return k * np.sqrt(l1), k * np.sqrt(l2), ang


def error_ellipse_map(df, reg, title, confidence=0.95, color_by="erh", max_events=None,
                      erh_max=None, sta=None, fault_trace=FAULT_TRACE, subregion=SUBREGION, lw=0.6):
    """Per-event horizontal confidence ellipses on a true-shape local E-N km frame
    (events/faults/stations transformed via to_cartesian_km). `color_by` in df (erh/depth).
    `erh_max` (km): keep only well-located events with ERH <= erh_max."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.patches import Ellipse
    from matplotlib.collections import PatchCollection
    d = df.dropna(subset=["cov_ee", "cov_nn", "cov_en"]).copy()
    d = d[d.lon.between(reg[0], reg[1]) & d.lat.between(reg[2], reg[3])]
    if erh_max is not None:
        d = d[d["erh"] <= erh_max]
    if max_events and len(d) > max_events:
        d = d.sample(max_events, random_state=0)
    d, T = to_cartesian_km(d)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    coast_mpl_km(ax, reg, T)
    for seg in read_fault_segments(fault_trace):
        fx, fy = T.transform(seg[:, 1], seg[:, 0])
        ax.plot(np.asarray(fx) / 1000, np.asarray(fy) / 1000, color="0.45", lw=0.5, zorder=1)
    patches, vals = [], []
    for r in d.itertuples():
        a, b, ang = error_ellipse(r.cov_ee, r.cov_en, r.cov_nn, confidence)
        if np.isnan(a):
            continue
        patches.append(Ellipse((r.x_km, r.y_km), 2 * a, 2 * b, angle=ang))
        vals.append(getattr(r, color_by))
    vals = np.asarray(vals, float)
    norm = mpl.colors.Normalize(vmin=np.nanpercentile(vals, 5), vmax=np.nanpercentile(vals, 95))
    cmap = plt.get_cmap("magma_r" if color_by == "erh" else "viridis_r")
    ax.add_collection(PatchCollection(patches, facecolors="none",
                                      edgecolors=cmap(norm(vals)), linewidths=lw, alpha=0.7,
                                      zorder=3))
    sta = STA if sta is None else sta
    if len(sta):
        sx, sy = T.transform(sta.Longitude.values, sta.Latitude.values)
        ax.plot(np.asarray(sx) / 1000, np.asarray(sy) / 1000, "k^", ms=5, zorder=4)
    if subregion is not None:
        bl, ba = _subregion_box(subregion)
        bx, by = T.transform(bl, ba)
        ax.plot(np.asarray(bx) / 1000, np.asarray(by) / 1000, "b-", lw=1.2, zorder=5)
    cx, cy = T.transform([reg[0], reg[1]], [reg[2], reg[3]])
    ax.set_xlim(cx[0] / 1000, cx[1] / 1000); ax.set_ylim(cy[0] / 1000, cy[1] / 1000)
    ax.set_aspect("equal")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    _match_cbar(sm, ax, color_by)
    q = f", ERH≤{erh_max} km" if erh_max is not None else ""
    ax.set(xlabel="E (km)", ylabel="N (km)",
           title=f"{title} — {int(confidence*100)}% error ellipses{q} (n={len(patches)})")
    fig.tight_layout()
    return fig


def error_section(df, axis="lon", confidence=0.95):
    """(lon|lat) vs depth with 1-D vertical error bars +/- k*sigma_z (k=norm 95%=1.96)."""
    import matplotlib.pyplot as plt
    from scipy.stats import norm as _norm
    d = df.dropna(subset=["cov_zz"]).copy()
    k = _norm.ppf((1.0 + confidence) / 2.0)
    sigz = np.sqrt(d["cov_zz"].clip(lower=0))
    fig, ax = plt.subplots(figsize=(11, 4), dpi=120)
    ax.errorbar(d[axis], d["depth"], yerr=k * sigz, fmt="o", ms=2, lw=0,
                elinewidth=0.4, ecolor="0.5", mfc="k", mec="k", alpha=0.6)
    ax.invert_yaxis()
    ax.set(xlabel=axis, ylabel="depth (km)",
           title=f"depth with {int(confidence*100)}% vertical CI (±{k:.2f}·σ_z), n={len(d)}")
    fig.tight_layout()
    return fig
