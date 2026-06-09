"""
Ulsan-fault catalog: inter-event WAVEFORM-similarity clustering to flag still-remaining
quarry blasts that the spatial/temporal decluster (`uf_cluster.py`) missed.

Idea: blasts from one quarry repeat the same source + path, so at a fixed common station
(default KG.HDB, ~99 % coverage of the 2797 event_waveforms_ulsanfault events) they produce
near-identical waveforms; tectonic events do not (genuine repeaters/aftershocks also
correlate, but separate out by location/time in the evidence table). We:

  1. load the station's HHZ trace per event, align on P (station pick -> synthetic fallback,
     then refine by cross-correlation to a stack so picked & unpicked share one datum),
  2. bandpass + cut a SHORT phase window (never the raw 120 s) + L2-normalise,
  3. build an N x N max-lag cross-correlation similarity matrix (per band),
  4. Ward-cluster on (1 - CC),
  5. show clustered heatmap / dendrogram / per-cluster waveform gathers,
  6. map clusters (PyGMT) + a per-cluster blast-likeness evidence table (intra-cluster CC,
     spatial compactness, daytime fraction / hour-of-day reusing uf_cluster).

Exploratory first: this module produces the matrices/figures/evidence; it does NOT remove
events. Reuses uf_cluster for KST/Rayleigh/maps and joins the blastclean catalog for
hypocentres. SAC: 100 Hz, 120 s, headers carry a(P)/t0(S)/o(origin)/stla/stlo.
"""
from __future__ import annotations

import os
from glob import glob

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read

import uf_cluster as ufc

# --------------------------------------------------------------------- defaults
HYPO_DIR   = "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/HypoInv"
WF_ROOT    = os.path.join(HYPO_DIR, "event_waveforms_ulsanfault")
BLASTCLEAN = os.path.join(HYPO_DIR, "catalog_phasenet_plus_2010_2024_blastclean.csv")
CLUSTER_SUMMARY = os.path.join(HYPO_DIR, "cluster_summary_phasenet_plus_2010_2024.csv")
STA_DIR    = os.path.join(HYPO_DIR, "STA")            # per-year HYPOINVERSE station tables UF{yyyy}.sta
CACHE_DIR  = os.path.join(HYPO_DIR, "wf_similarity_cache")

STATION = "KG.HDB"          # NET.STA common station (default)
COMP    = "HHZ"             # vertical
SR      = 100.0            # Hz (all traces)
# Analysis defaults — a SHORT P-aligned window (NOT the 120 s record) is essential: the raw
# trace is mostly pre-event noise + long coda that dilutes the correlation.
DEFAULT_WIN   = (-0.5, 7.5)                       # s relative to P (P -> S -> early coda)
DEFAULT_BANDS = [(1, 10), (2, 8), (4, 12), (5, 15)]  # Hz; blasts vs quakes differ by band
REF_BAND      = (2, 8)                            # band used to refine alignment
DEFAULT_MAXLAG = 0.2                              # s; CC lag search (small: alignment refined)
REFINE_MAXSHIFT = 2.0                             # s; alignment-refine search half-width
HELVETICA = "/home/msseo/Downloads/Helvetica/helvetica.ttf"


# --------------------------------------------------------------- plotting font
def use_helvetica():
    """Register Helvetica for matplotlib text if the font file is present (graceful no-op
    otherwise) — matches the user's plot-style preference. Call once before plotting."""
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm
        if os.path.isfile(HELVETICA):
            fm.fontManager.addfont(HELVETICA)
            name = fm.FontProperties(fname=HELVETICA).get_name()
            mpl.rcParams["font.family"] = [name, "DejaVu Sans"]
    except Exception:                                   # noqa: BLE001 — never break plotting
        pass


# --------------------------------------------------------------- event listing
def list_events(wf_root=WF_ROOT, station=STATION, comp=COMP):
    """Sorted event ids (YYYYMMDDHHMMSS dir names) that have the station's `comp` SAC trace."""
    net, sta = station.split(".")
    out = []
    for d in sorted(glob(os.path.join(wf_root, "2" + "[0-9]" * 13))):
        ev = os.path.basename(d)
        if os.path.exists(os.path.join(d, f"{ev}.{net}.{sta}.{comp}.sac")):
            out.append(ev)
    return out


def _sac_path(ev, station=STATION, comp=COMP, wf_root=WF_ROOT):
    net, sta = station.split(".")
    return os.path.join(wf_root, ev, f"{ev}.{net}.{sta}.{comp}.sac")


def origin_time(ev):
    """UTCDateTime origin from the event-dir name (YYYYMMDDHHMMSS, second precision)."""
    return UTCDateTime(f"{ev[0:4]}-{ev[4:6]}-{ev[6:8]}T{ev[8:10]}:{ev[10:12]}:{ev[12:14]}")


def event_hours(events, kst=ufc.KST):
    """Hour-of-day (KST, continuous hh.h) for each event straight from its origin time (the dir
    name) — NO catalog join, so EVERY event has one (catalog `hour_kst` is NaN for events not in
    the blastclean catalog). Returns an array aligned to `events`."""
    out = []
    for ev in events:
        t = origin_time(ev) + kst * 3600.0                   # UTC -> KST
        out.append(t.hour + t.minute / 60.0)
    return np.asarray(out, dtype=float)


def event_decimal_years(events):
    """Decimal year (e.g. 2018.42) of each event's origin time from the dir name — a continuous
    time axis for colouring / cumulative curves. Aligned to `events`."""
    out = []
    for ev in events:
        y = int(ev[:4])
        y0 = UTCDateTime(f"{y}-01-01"); y1 = UTCDateTime(f"{y + 1}-01-01")
        out.append(y + (origin_time(ev) - y0) / (y1 - y0))
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------- P-time per event
def pick_time(ev, station=STATION, phase="P", wf_root=WF_ROOT, comp=COMP):
    """Absolute P (or S) time for `station` at event `ev`, or None.

    Source order: the event's `{ev}_picks.csv` (`peak_time`, matched on the `station`
    prefix, e.g. 'KG.HDB') -> the SAC header mark (a=P, t0=S) on the `comp` trace.

    NOTE on the SAC-header branch: on this dataset the `picks.csv` rows and the SAC a/t0 marks
    were written from the SAME PhaseNet run, so a pick is present in BOTH or NEITHER — the header
    read is a defensive fallback that is never actually reached for P (verified 0 events). It is
    kept only for robustness if a future event ever ships SAC marks without a picks.csv. Either
    way the result is fully deterministic (no randomness). Returns a UTCDateTime."""
    net, sta = station.split(".")
    csv = os.path.join(wf_root, ev, f"{ev}_picks.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        m = (df["station"].astype(str).str.startswith(f"{net}.{sta}")) & (df["phase"] == phase)
        if m.any():
            # round to µs: peak_time carries ns that obspy would warn about discarding;
            # 1 µs is far below the 10 ms sample interval, so alignment is unaffected.
            return UTCDateTime(pd.Timestamp(df.loc[m, "peak_time"].iloc[0]).round("us").to_pydatetime())
    # SAC header fallback (a = P, t0 = S; both relative to the trace reference o=origin).
    # Read the SAME `comp` being processed, not a hard-coded channel, so the datum is consistent.
    try:
        tr = read(_sac_path(ev, station, comp, wf_root))[0]
        mark = {"P": "a", "S": "t0"}[phase]
        v = tr.stats.sac.get(mark, -12345.0)
        if v != -12345.0:
            return tr.stats.starttime - tr.stats.sac.b + v
    except Exception:                                   # noqa: BLE001
        pass
    return None


def median_p_traveltime(events, station=STATION, wf_root=WF_ROOT):
    """Median (P_pick - origin) seconds over events that DO have a station P-pick — the
    fallback datum for the ~10 % of events with a trace but no pick at this station."""
    tt = []
    for ev in events:
        p = pick_time(ev, station, "P", wf_root)
        if p is not None:
            tt.append(p - origin_time(ev))
    return float(np.median(tt)) if tt else 3.0


# --------------------------------------------------------------- feature build
def _proc(tr, band, sr=SR):
    """Process a copy of `tr`: demean + linear detrend (always), resample to `sr` if needed, and
    apply a zero-phase filter when `band` is given. `band` may be:
      None              -> minimal preprocessing (no taper, no filter): the raw-data view
      (lo, hi)          -> bandpass
      ('highpass', f)   -> highpass at f Hz (also 'lowpass')."""
    t = tr.copy()
    t.detrend("demean"); t.detrend("linear")
    if abs(t.stats.sampling_rate - sr) > 1e-6:
        t.resample(sr)
    if band is not None:
        t.taper(0.05)
        if isinstance(band[0], str):                         # ('highpass'|'lowpass', freq)
            t.filter(band[0], freq=band[1], corners=4, zerophase=True)
        else:
            t.filter("bandpass", freqmin=band[0], freqmax=band[1], corners=4, zerophase=True)
    return t


def _cut(tr, p_time, win, sr=SR):
    """Slice a P-relative window [p+win0, p+win1] -> 1-D float array of fixed length, or None
    if the window runs off the trace. Length = round((win1-win0)*sr)."""
    n = int(round((win[1] - win[0]) * sr))
    start = p_time + win[0]
    i0 = int(round((start - tr.stats.starttime) * sr))
    if i0 < 0 or i0 + n > tr.stats.npts:
        return None
    seg = tr.data[i0:i0 + n].astype(np.float64)
    return seg if seg.size == n else None


def _l2(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def build_features(events, station=STATION, comp=COMP, band=REF_BAND, win=DEFAULT_WIN,
                   wf_root=WF_ROOT, sr=SR, pad=REFINE_MAXSHIFT, verbose=True):
    """Load + P-align + bandpass + window + L2-normalise one trace per event.

    P datum per event, in exactly two deterministic sources:
      `pick`     -> the station's P in `{ev}_picks.csv` (or, defensively, a SAC a-mark; see
                    `pick_time` — the header path is unreached on this dataset).
      `fallback` -> no station pick exists, so `origin_time + median(P pick - origin)`
                    (`median_p_traveltime`), refined later by xcorr in `align_fallback`.
    Each window is cut with `pad` s extra on both sides (room for the later xcorr refine).

    Returns (X, kept, info):
      X     : (n, nsamp_padded) float32, L2-normalised, P at offset `-win[0]+pad` s.
      kept  : list[str] event ids in X-row order.
      info  : DataFrame[event, p_source, p_rel] (p_source in {pick, fallback} — two-valued).
    The padded window is trimmed to `win` later by `finalize` after alignment refine."""
    med_tt = median_p_traveltime(events, station, wf_root)
    win_pad = (win[0] - pad, win[1] + pad)
    rows, kept, meta = [], [], []
    n_pick = n_fb = n_skip = 0
    for ev in events:
        try:
            tr = read(_sac_path(ev, station, comp, wf_root))[0]
        except Exception:                               # noqa: BLE001
            n_skip += 1; continue
        if np.allclose(tr.data, tr.data[0]):            # dead/flat trace
            n_skip += 1; continue
        # P datum: station pick if present, else synthetic fallback (deterministic; no header label)
        p = pick_time(ev, station, "P", wf_root, comp)
        if p is not None:
            src = "pick"
        else:
            p = origin_time(ev) + med_tt; src = "fallback"
        seg = _cut(_proc(tr, band, sr), p, win_pad, sr)
        if seg is None:
            n_skip += 1; continue
        rows.append(_l2(seg)); kept.append(ev)
        meta.append(dict(event=ev, p_source=src, p_rel=float(p - origin_time(ev))))
        n_pick += src == "pick"; n_fb += src == "fallback"
    X = np.asarray(rows, dtype=np.float32)
    if verbose:
        print(f"[build_features] {station}.{comp} band{band}: kept {len(kept)}/{len(events)} "
              f"(pick {n_pick}, fallback {n_fb}; skipped {n_skip})")
    return X, kept, pd.DataFrame(meta)


def _best_lag(xi, stack, L):
    """Integer lag in [-L, L] maximising the (overlap) dot-product of row `xi` with `stack`."""
    m = xi.size
    best, blag = -np.inf, 0
    for lag in range(-L, L + 1):
        a, b = (xi[lag:], stack[:m - lag]) if lag >= 0 else (xi[:lag], stack[-lag:])
        c = float(np.dot(a, b))
        if c > best:
            best, blag = c, lag
    return blag


def align_fallback(Xpad, info, sr=SR, maxshift=REFINE_MAXSHIFT, ref_sources=("pick", "header"),
                   iters=2):
    """Bring the unpicked (`p_source == 'fallback'`) events onto the picked datum.

    PICKED events keep their pick alignment (P stays at t=0 — accurate gathers, honest axis);
    only fallback events are cross-correlated to a stack of the well-aligned picked/header
    events and shifted by the best lag in [-maxshift, maxshift]. This is more robust than a
    single GLOBAL stack (which can mis-align distinct waveform families); small residual
    pick jitter is absorbed by the per-pair lag search in `similarity_matrix`.

    Returns integer SAMPLE shifts (0 for picked/header events)."""
    src = info["p_source"].values
    ref = np.isin(src, ref_sources)
    shifts = np.zeros(Xpad.shape[0], dtype=int)
    if not ref.any() or ref.all():
        return shifts                                    # nothing to anchor to, or nothing to move
    L = int(round(maxshift * sr))
    stack = _l2(Xpad[ref].mean(axis=0))
    for _ in range(iters):
        for i in np.where(~ref)[0]:
            shifts[i] = _best_lag(Xpad[i], stack, L)
        # refine the stack with the now-aligned fallbacks folded in
        stack = _l2(_apply_shifts(Xpad, shifts).mean(axis=0))
    return shifts


def _apply_shifts(X, shifts):
    """Shift each row by its integer sample lag (zero-fill the vacated edge), same length."""
    out = np.zeros_like(X)
    for i, s in enumerate(shifts):
        if s == 0:
            out[i] = X[i]
        elif s > 0:
            out[i, :-s] = X[i, s:]
        else:
            out[i, -s:] = X[i, :s]
    return _row_l2(out)


def _row_l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def finalize(X_pad, shifts, win, pad=REFINE_MAXSHIFT, sr=SR):
    """Apply alignment shifts to the padded features, then crop the `pad` margins to the exact
    analysis window `win` and re-L2-normalise. Returns (n, nsamp) float32."""
    Xs = _apply_shifts(X_pad, shifts)
    p = int(round(pad * sr))
    Xc = Xs[:, p:Xs.shape[1] - p] if p > 0 else Xs
    return _row_l2(Xc)


def feature_matrix(events, station=STATION, comp=COMP, band=REF_BAND, win=DEFAULT_WIN,
                   refine=True, wf_root=WF_ROOT, sr=SR, verbose=True):
    """End-to-end per-band feature builder: build_features (padded) -> refine_shifts (on
    REF_BAND) -> finalize. Returns (X, kept, info). Pass a shared `shifts` via refine=False +
    a precomputed alignment to keep bands consistently aligned (see make_bands)."""
    Xp, kept, info = build_features(events, station, comp, band, win, wf_root, sr, verbose=verbose)
    shifts = align_fallback(Xp, info, sr) if refine else np.zeros(len(kept), dtype=int)
    return finalize(Xp, shifts, win, sr=sr), kept, info, shifts


def make_bands(events, station=STATION, comp=COMP, bands=None, win=DEFAULT_WIN,
               wf_root=WF_ROOT, sr=SR, cache_dir=CACHE_DIR, verbose=True):
    """Build aligned feature matrices for every band with ONE shared alignment (computed on
    REF_BAND so all bands use the same P datum). Caches to `cache_dir/feat_<station>_<win>.npz`.
    Returns dict: {'kept': [...], 'info': DataFrame, 'bands': {band: X}, 'shifts': array}."""
    import hashlib
    bands = bands or DEFAULT_BANDS
    os.makedirs(cache_dir, exist_ok=True)
    # tag includes an events-hash so different event sets (e.g. 2010 vs full catalog) never
    # collide on the same cache file.
    h = hashlib.md5(",".join(events).encode()).hexdigest()[:8]
    tag = f"{station}_{comp}_w{win[0]}_{win[1]}_n{len(events)}_{h}".replace(".", "p")
    cache = os.path.join(cache_dir, f"feat_{tag}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if verbose:
            print(f"[make_bands] loaded cache {cache}")
        return dict(kept=list(z["kept"]), info=pd.DataFrame(z["info"].tolist()),
                    shifts=z["shifts"], bands={tuple(b): z[f"b{i}"]
                                               for i, b in enumerate(z["band_list"])})
    # shared alignment from REF_BAND (only the unpicked fallbacks are moved onto the pick datum)
    Xp, kept, info = build_features(events, station, comp, REF_BAND, win, wf_root, sr, verbose=verbose)
    shifts = align_fallback(Xp, info, sr)
    out = {"kept": kept, "info": info, "shifts": shifts, "bands": {}}
    for b in bands:
        Xb, kb, _ = build_features(events, station, comp, b, win, wf_root, sr, verbose=False)
        # kb should equal kept (same skip logic); align by event just in case
        if kb != kept:
            idx = {e: i for i, e in enumerate(kb)}
            Xb = np.asarray([Xb[idx[e]] for e in kept], dtype=np.float32)
        out["bands"][tuple(b)] = finalize(Xb, shifts, win, sr=sr)
        if verbose:
            print(f"[make_bands] band {b}: {out['bands'][tuple(b)].shape}")
    np.savez_compressed(cache, kept=np.array(kept), info=np.array(info.to_dict("records")),
                        shifts=shifts, band_list=np.array(bands),
                        **{f"b{i}": out["bands"][tuple(b)] for i, b in enumerate(bands)})
    if verbose:
        print(f"[make_bands] cached -> {cache}")
    return out


def display_matrix(res, band=None, station=STATION, comp=COMP, win=DEFAULT_WIN, wf_root=WF_ROOT,
                   sr=SR, pad=REFINE_MAXSHIFT):
    """Aligned display matrix for the SAME events + the SAME P-alignment as `make_bands` output
    `res`, processed with `band` (None = raw demean/detrend; (lo,hi) = bandpass; ('highpass',f) =
    highpass). Clustering is UNCHANGED — this only re-renders a chosen filter for the already-
    identified clusters. Reuses the exact per-event P datum from `res['info'].p_rel`, so rows line
    up 1:1 with `res['kept']`/labels. Each trace L2-normalised for display."""
    info = res["info"].set_index("event")
    n_pad = int(round((win[1] - win[0] + 2 * pad) * sr))
    rows = []
    for ev in res["kept"]:
        seg = None
        try:
            tr = read(_sac_path(ev, station, comp, wf_root))[0]
            p = origin_time(ev) + float(info.loc[ev, "p_rel"])
            seg = _cut(_proc(tr, band, sr), p, (win[0] - pad, win[1] + pad), sr)
        except Exception:                                    # noqa: BLE001
            pass
        rows.append(_l2(seg) if seg is not None else np.zeros(n_pad))
    return finalize(np.asarray(rows, dtype=np.float32), res["shifts"], win, sr=sr)


def raw_matrix(res, **kw):
    """Raw-data view (band=None) — minimal preprocessing. Thin wrapper over `display_matrix`."""
    return display_matrix(res, band=None, **kw)


# --------------------------------------------------------------- similarity
def similarity_matrix(X, maxlag=DEFAULT_MAXLAG, sr=SR):
    """N x N max-lag normalised cross-correlation. Rows are L2-normalised; for each integer
    lag in [-L, L] (L = maxlag*sr) compute the (renormalised) overlap dot-product matrix and
    keep the element-wise max. CC in [-1, 1] (clip to [0,1] for distance). Symmetric, diag 1.

    Cost ~ (2L+1) matmuls of (n x m)(m x n); keep maxlag small (alignment is refined)."""
    n, m = X.shape
    L = int(round(maxlag * sr))
    best = X @ X.T                                        # lag 0
    for lag in range(1, L + 1):
        A = _row_l2(X[:, lag:]); B = _row_l2(X[:, :m - lag])
        c = A @ B.T
        best = np.maximum(best, c)                        # row shifted +lag vs col
        best = np.maximum(best, c.T)                      # and the symmetric -lag
    np.fill_diagonal(best, 1.0)
    return best.astype(np.float32)


def signed_similarity(X, maxlag=DEFAULT_MAXLAG, sr=SR, return_lags=False):
    """Signed max-lag cross-correlation — like `similarity_matrix` but KEEPS the sign, to expose
    anti-correlated ("anti-repeater") pairs that the max-|CC| similarity throws away.

    For each integer lag in [-L, L] (L = maxlag*sr) build the renormalised overlap dot-product
    matrix and track BOTH extrema over lags. Returns a dict of N x N float32 matrices:

      cc_pos : MAX over lags  — **identical to `similarity_matrix(X)`** (the positive half).
      cc_neg : MIN over lags  — the most-negative correlation (the anti-correlation signal).
      cc_ext : signed extreme — cc_neg where |cc_neg| > |cc_pos| else cc_pos.
      cc_lag0: signed CC at lag 0 (X @ X.T) — correlation at the EXACT P datum, no lag freedom.

    The crux: within +/-maxlag a half-cycle shift can FAKE cc_neg ~ -1 for an otherwise positively
    correlated pair, so cc_neg is only meaningful READ WITH cc_pos. A true anti-repeater has
    `cc_neg <= -0.85 AND cc_pos` not also high (you cannot make anti-phase signals positively
    correlate within +/-maxlag); `cc_lag0 <= -0.85` is the cleanest evidence (anti-correlated at the
    same datum used for repeaters). Diagonals: cc_pos/cc_ext/cc_lag0 -> 1; cc_neg -> 1 (self has no
    anti-twin). Symmetric. `return_lags=True` adds int lag_pos/lag_neg (signed sample lag). Same cost
    as `similarity_matrix` (one extra running extremum)."""
    n, m = X.shape
    L = int(round(maxlag * sr))
    c0 = (X @ X.T).astype(np.float32)                    # lag 0 (signed)
    pos = c0.copy(); neg = c0.copy()
    if return_lags:
        lag_pos = np.zeros((n, n), dtype=np.int16)
        lag_neg = np.zeros((n, n), dtype=np.int16)
    for lag in range(1, L + 1):
        A = _row_l2(X[:, lag:]); B = _row_l2(X[:, :m - lag])
        c = A @ B.T
        for Mat, sgn in ((c, +1), (c.T, -1)):            # +lag (row leads) and symmetric -lag
            if return_lags:
                gp = Mat > pos; pos = np.where(gp, Mat, pos); lag_pos = np.where(gp, sgn * lag, lag_pos)
                gn = Mat < neg; neg = np.where(gn, Mat, neg); lag_neg = np.where(gn, sgn * lag, lag_neg)
            else:
                pos = np.maximum(pos, Mat); neg = np.minimum(neg, Mat)
    cc_ext = np.where(np.abs(neg) > np.abs(pos), neg, pos)
    for M in (pos, cc_ext, c0):
        np.fill_diagonal(M, 1.0)
    np.fill_diagonal(neg, 1.0)                            # self has no anti-twin
    out = dict(cc_pos=pos.astype(np.float32), cc_neg=neg.astype(np.float32),
               cc_ext=cc_ext.astype(np.float32), cc_lag0=c0.astype(np.float32))
    if return_lags:
        out["lag_pos"] = lag_pos.astype(np.int16); out["lag_neg"] = lag_neg.astype(np.int16)
    return out


# --------------------------------------------------------------- clustering
def ward_clusters(cc, threshold=None, n_clusters=None, method="ward"):
    """Hierarchical clustering on distance = 1 - CC.

    NOTE: Ward assumes Euclidean distances; 1-CC is not strictly metric, so also try
    method='average' (UPGMA), which is the conventional choice for correlation distance, as a
    cross-check. Returns (labels, Z, order) where `order` is the dendrogram leaf order (for the
    clustered heatmap). Provide EITHER threshold (cophenetic dist) OR n_clusters."""
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
    from scipy.spatial.distance import squareform
    D = 1.0 - np.clip(cc, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)                                   # enforce symmetry for squareform
    Z = linkage(squareform(D, checks=False), method=method)
    if n_clusters is not None:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    else:
        labels = fcluster(Z, t=(0.4 if threshold is None else threshold), criterion="distance")
    return labels, Z, leaves_list(Z)


# --------------------------------------------------------------- catalog join
def load_event_meta(events, catalog=BLASTCLEAN, wf_root=WF_ROOT, kst=ufc.KST):
    """Join events to the blastclean catalog (hypocentre + KST hour/dow) by origin time.

    Returns a DataFrame indexed to `events` order: event, time, lat, lon, depth, hour,
    hour_kst, dow, cluster(spatial), joined(bool). Unjoined events keep NaN coords."""
    cat = pd.read_csv(catalog)
    cat["time"] = pd.to_datetime(cat["time"], utc=True)
    cat = ufc.add_kst_columns(cat, kst)
    cat["evid"] = cat["time"].dt.strftime("%Y%m%d%H%M%S")
    cat = cat.drop_duplicates("evid", keep="first")
    base = pd.DataFrame({"event": list(events)})
    out = base.merge(cat, left_on="event", right_on="evid", how="left")
    out["joined"] = out["lat"].notna()
    return out


# --------------------------------------------------------------- evidence table
def cluster_evidence(meta, labels, cc, min_size=5, day=(6, 17)):
    """Per-waveform-cluster blast-likeness evidence (clusters with >= min_size members).

    Columns: cluster, n, mean_cc (intra-cluster off-diagonal mean CC: high = tight repeating
    family), lat_c, lon_c, depth_med, spread_km (median epicentral dist to centroid),
    daytime_frac (06-18 KST), rayleigh_p, peak_hour, weekend_ratio, n_joined.
    A tight (high mean_cc), spatially compact, daytime-concentrated cluster = strong
    still-remaining quarry-blast candidate. Sorted by mean_cc desc."""
    m = meta.copy().reset_index(drop=True)
    m["wf_cluster"] = labels
    rows = []
    for cid, g in m.groupby("wf_cluster"):
        if len(g) < min_size:
            continue
        idx = g.index.values
        sub = cc[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), k=1)
        mean_cc = float(sub[iu].mean()) if len(iu[0]) else np.nan
        gj = g[g["joined"]]
        if len(gj):
            latc, lonc = gj["lat"].mean(), gj["lon"].mean()
            spread = float(np.median(np.hypot((gj["lat"] - latc) * 111.0,
                                              (gj["lon"] - lonc) * 111.0 * np.cos(np.radians(latc)))))
            rt = ufc.rayleigh_test(gj["hour"].values)
            rows.append(dict(
                cluster=int(cid), n=int(len(g)), mean_cc=round(mean_cc, 3),
                lat_c=round(latc, 4), lon_c=round(lonc, 4),
                depth_med=round(float(gj["depth"].median()), 2), spread_km=round(spread, 2),
                daytime_frac=round(float(gj["hour"].between(day[0], day[1]).mean()), 2),
                rayleigh_p=round(rt["p"], 4), peak_hour=round(rt["peak_hour"], 1),
                weekend_ratio=round(float((gj["dow"] >= 5).mean() / (2.0 / 7.0)), 2),
                n_joined=int(len(gj))))
        else:
            rows.append(dict(cluster=int(cid), n=int(len(g)), mean_cc=round(mean_cc, 3),
                             lat_c=np.nan, lon_c=np.nan, depth_med=np.nan, spread_km=np.nan,
                             daytime_frac=np.nan, rayleigh_p=np.nan, peak_hour=np.nan,
                             weekend_ratio=np.nan, n_joined=0))
    return pd.DataFrame(rows).sort_values("mean_cc", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------- plots
def plot_similarity(cc, order=None, ax=None, title="Waveform similarity (max-lag CC)"):
    """Heatmap of the CC matrix, optionally reordered by dendrogram leaf order `order`."""
    import matplotlib.pyplot as plt
    M = cc if order is None else cc[np.ix_(order, order)]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6), dpi=130)
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=1, aspect="equal", interpolation="nearest")
    ax.set(title=title, xlabel="Event (clustered order)", ylabel="Event (clustered order)")
    ax.figure.colorbar(im, ax=ax, label="Cross-correlation", shrink=0.8)
    return ax.figure


def outline_clusters(ax, labels, order, min_size=2, color="white", lw=0.8):
    """Outline each identified cluster as a box on a dendrogram-ORDERED similarity matrix.

    `labels` are the flat-cluster ids; `order` is the leaf order the matrix was reordered by
    (`leaves_list(Z)`). A flat cluster (cut from the linkage) is a **contiguous run** in leaf order,
    so each cluster ≥ `min_size` becomes one square drawn from its first to last reordered index.
    Call right after `ax.imshow(CC[np.ix_(order, order)], …)`."""
    import matplotlib.patches as mpatches
    lab = np.asarray(labels)[np.asarray(order)]
    n = len(lab); i = 0
    while i < n:
        j = i
        while j + 1 < n and lab[j + 1] == lab[i]:
            j += 1
        if j - i + 1 >= min_size:
            ax.add_patch(mpatches.Rectangle((i - 0.5, i - 0.5), j - i + 1, j - i + 1,
                                            fill=False, edgecolor=color, lw=lw, zorder=3))
        i = j + 1


def plot_dendrogram(Z, color_threshold=None, ax=None, title="Ward dendrogram (1 − CC)"):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4), dpi=130)
    dendrogram(Z, no_labels=True, color_threshold=color_threshold, ax=ax)
    ax.set(title=title, ylabel="Distance (1 − CC)")
    return ax.figure


def plot_cluster_gathers(X, labels, evidence, sr=SR, win=DEFAULT_WIN, max_clusters=8,
                         max_traces=60, colors=None):
    """For the top clusters in `evidence` (by mean_cc): overlaid aligned traces + the cluster
    stack (bold), one panel each — repeating families are visually obvious. Returns the fig."""
    import matplotlib.pyplot as plt
    cids = evidence["cluster"].head(max_clusters).tolist()
    cols = colors or cluster_colors(cids)
    ncol = min(4, len(cids)) or 1
    nrow = int(np.ceil(len(cids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), dpi=130, squeeze=False)
    t = np.arange(X.shape[1]) / sr + win[0]
    for ax, cid in zip(axes.ravel(), cids):
        idx = np.where(labels == cid)[0]
        ev = evidence[evidence["cluster"] == cid].iloc[0]
        col = cols.get(int(cid), "crimson")
        sel = idx[:max_traces]
        for i in sel:
            ax.plot(t, X[i], color="0.65", lw=0.3, alpha=0.5)
        ax.plot(t, _l2(X[idx].mean(axis=0)), color=col, lw=1.4)
        ax.axvline(0, color="b", lw=0.6, ls="--")            # P
        ax.set_title(f"cl {cid}: n={ev['n']}, cc={ev['mean_cc']}", fontsize=8, color=col)
        ax.set_xlabel("Time from P (s)", fontsize=7); ax.tick_params(labelsize=6)
        ax.set_yticks([])
    for ax in axes.ravel()[len(cids):]:
        ax.axis("off")
    fig.suptitle("Cluster waveform gathers (grey = members, red = stack)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_antipair_gathers(res, pairs, band=REF_BAND, sr=SR, win=DEFAULT_WIN, ncol=3, title=None):
    """Overlay gallery for candidate ANTI-correlated pairs. Each panel shows, on the P-aligned
    `band` window: event *i* (black), event *j* un-flipped (faint grey), and event *j* **flipped**
    (`-X[j]`, red). A true polarity reversal makes the red curve coincide with the black one. The
    title carries the three diagnostics so the half-cycle degeneracy is visible at a glance:
    `lag0` (signed CC at the P datum), `neg` (most-negative over lags), `pos` (most-positive over
    lags — if this is ALSO high, the pair is an ordinary repeater offset by ~half a period, not a
    reversal). `pairs` = list of dicts with at least i, j (+ optional cc_lag0/cc_neg/cc_pos)."""
    import matplotlib.pyplot as plt
    X = res["bands"][tuple(band)]; kept = res["kept"]
    t = np.arange(X.shape[1]) / sr + win[0]
    npr = len(pairs); ncol = min(ncol, npr) or 1; nrow = int(np.ceil(npr / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 2.4 * nrow), dpi=130, squeeze=False)
    for ax, p in zip(axes.ravel(), pairs):
        i, j = p["i"], p["j"]
        ax.plot(t, _l2(X[j]), color="0.75", lw=0.5)                       # j (un-flipped)
        ax.plot(t, _l2(X[i]), color="k", lw=0.9)                          # i
        ax.plot(t, -_l2(X[j]), color="crimson", lw=0.9)                   # j flipped
        ax.axvline(0, color="b", lw=0.5, ls="--")                        # P datum
        lab = "lag0={:.2f} neg={:.2f} pos={:.2f}".format(
            p.get("cc_lag0", np.nan), p.get("cc_neg", np.nan), p.get("cc_pos", np.nan))
        ax.set_title("{}\n× {}\n{}".format(kept[i], kept[j], lab), fontsize=6.5)
        ax.set_xlabel("Time from P (s)", fontsize=7); ax.set_yticks([]); ax.tick_params(labelsize=6)
    for ax in axes.ravel()[npr:]:
        ax.axis("off")
    fig.suptitle(title or "Anti-pair overlays ({}-{} Hz): black = event i, red = event j FLIPPED "
                 "(coincide ⇒ true reversal)".format(band[0], band[1]), fontsize=9)
    fig.tight_layout()
    return fig


def plot_antipair_detail(res, pairs, band=REF_BAND, sr=SR, win=DEFAULT_WIN, zoom=None,
                         width=14.0, row_h=1.8, title=None):
    """**One full-width row per pair** so the wiggle match is legible (the grid `plot_antipair_gathers`
    squeezes the whole window into a narrow panel). Each row overlays, on the P-aligned `band`:
    event *i* (black), event *j* **flipped** (`-X[j]`, red), event *j* un-flipped (faint grey).
    `zoom=(t0, t1)` restricts the time axis for fine detail (e.g. `zoom=(-0.3, 3.0)` around P→S).
    `pairs` = list of dicts with i, j (+ optional cc_lag0/cc_neg/cc_pos). Returns the fig."""
    import matplotlib.pyplot as plt
    X = res["bands"][tuple(band)]; kept = res["kept"]
    t = np.arange(X.shape[1]) / sr + win[0]
    n = len(pairs)
    fig, axes = plt.subplots(n, 1, figsize=(width, row_h * n), dpi=130, squeeze=False)
    for k, (ax, p) in enumerate(zip(axes.ravel(), pairs)):
        i, j = p["i"], p["j"]
        ax.plot(t, _l2(X[j]), color="0.78", lw=0.6)                     # j (un-flipped)
        ax.plot(t, _l2(X[i]), color="k", lw=1.0)                        # i
        ax.plot(t, -_l2(X[j]), color="crimson", lw=1.0)                 # j flipped
        ax.axvline(0, color="b", lw=0.6, ls="--")                      # P datum
        if zoom is not None:
            ax.set_xlim(*zoom)
        ax.margins(x=0)
        lab = "lag0={:.2f}  neg={:.2f}  pos={:.2f}".format(
            p.get("cc_lag0", np.nan), p.get("cc_neg", np.nan), p.get("cc_pos", np.nan))
        ax.set_title("{} × {}    {}".format(kept[i], kept[j], lab), fontsize=8, loc="left")
        ax.set_yticks([]); ax.tick_params(labelsize=7)
        if k < n - 1:
            ax.set_xticklabels([])
    axes.ravel()[-1].set_xlabel("Time from P (s)", fontsize=9)
    fig.suptitle(title or "Anti-pair detail ({}-{} Hz): black = event i, red = event j FLIPPED".format(
        band[0], band[1]), fontsize=10)
    fig.tight_layout()
    return fig


def plot_antipair_compare(res, pairs, band=REF_BAND, sr=SR, win=DEFAULT_WIN, zoom=None,
                          width=14.0, row_h=1.8, title=None):
    """**Two overlays per pair, side by side**, at the SAME P-aligned (lag-0) datum, so you can see
    directly whether a pair is an anti-repeater or an ordinary repeater:

      LEFT  — event i (black) vs event j **as-is** (blue): they overlay for a *repeater*, mirror for
              an *anti-repeater*. (At lag 0 the as-is correlation is `cc_lag0`.)
      RIGHT — event i (black) vs event j **flipped** (`-X[j]`, red): they overlay for an *anti-repeater*.

    Caveat baked into the title: `pos` is the best POSITIVE correlation over *all* lags — if it is high,
    the as-is traces also match after a small (≈ half-period) shift, i.e. a half-cycle-offset repeater
    rather than a true reversal; a genuine anti-repeater has a clean flipped match AND a *low* `pos`.
    `zoom=(t0,t1)` for detail; `pairs` = dicts with i, j (+ optional cc_lag0/cc_neg/cc_pos)."""
    import matplotlib.pyplot as plt
    X = res["bands"][tuple(band)]; kept = res["kept"]
    t = np.arange(X.shape[1]) / sr + win[0]
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(width, row_h * n), dpi=130, squeeze=False, sharex=True)
    for r, p in enumerate(pairs):
        i, j = p["i"], p["j"]; a0, a1 = axes[r]
        a0.plot(t, _l2(X[i]), color="k", lw=1.0); a0.plot(t, _l2(X[j]), color="royalblue", lw=1.0)
        a1.plot(t, _l2(X[i]), color="k", lw=1.0); a1.plot(t, -_l2(X[j]), color="crimson", lw=1.0)
        for a in (a0, a1):
            a.axvline(0, color="b", lw=0.5, ls="--"); a.set_yticks([]); a.tick_params(labelsize=7); a.margins(x=0)
            if zoom is not None:
                a.set_xlim(*zoom)
        a0.set_title("{} × {}   as-is (blue)   lag0={:.2f}".format(
            kept[i], kept[j], p.get("cc_lag0", np.nan)), fontsize=7, loc="left")
        a1.set_title("flipped (red)   neg={:.2f}   pos={:.2f}".format(
            p.get("cc_neg", np.nan), p.get("cc_pos", np.nan)), fontsize=7, loc="left")
    axes[-1, 0].set_xlabel("Time from P (s)", fontsize=9); axes[-1, 1].set_xlabel("Time from P (s)", fontsize=9)
    fig.suptitle(title or "Each pair — LEFT: event j as-is (blue) · RIGHT: event j flipped (red) · "
                 "black = event i   [{}-{} Hz]".format(band[0], band[1]), fontsize=10)
    fig.tight_layout()
    return fig


def s_minus_p(kept, station=STATION, wf_root=WF_ROOT):
    """S-P seconds per event (NaN if the station's S or P pick is missing) — for the gather's
    S annotation. P is at t=0 in the aligned window, so S plots at this value."""
    out = []
    for ev in kept:
        p = pick_time(ev, station, "P", wf_root)
        s = pick_time(ev, station, "S", wf_root)
        out.append((s - p) if (p is not None and s is not None) else np.nan)
    return np.asarray(out, dtype=float)


def cluster_colors(keep_ids):
    """Map each cluster id to a DISTINCT colour. The first 20 ids (the families most likely to be
    plotted — `keep_ids` is passed in size order) get the qualitative `tab20` palette so adjacent
    families are easy to tell apart; any remainder gets spread over `hsv`. (The old code put ALL
    ids on `hsv`, so a plotted top-N subset fell in one narrow hue band and looked identical.)"""
    import matplotlib.pyplot as plt
    keep_ids = list(keep_ids); n = len(keep_ids)
    tab = plt.get_cmap("tab20"); hsv = plt.get_cmap("hsv")
    cols = [tab(i) if i < 20 else hsv((i - 20) / max(n - 20, 1)) for i in range(n)]
    return {int(c): cols[i] for i, c in enumerate(keep_ids)}


def plot_cluster_sections(X, labels, kept, sr=SR, win=DEFAULT_WIN, station=STATION, comp=COMP,
                          wf_root=WF_ROOT, min_show=3, max_per_cluster=60, order_by="size",
                          sp=None, row_h=0.16, fig_w=9, colors=None, show_singletons=True,
                          singleton_color="0.55", max_singletons=None, title=None,
                          trace_values=None, value_cmap="hsv", value_range=None, value_label="",
                          max_clusters=None, clusters=None, annotate_utc=True,
                          min_fig_h=3.0, head_in=0.0):
    """Record-section of EVERY trace (not stacked), P-aligned at t=0, GROUPED by cluster, with
    each cluster drawn in a DISTINCT colour.

    One wiggle per event; blue dashed line = P (t=0); short black bars = S arrivals where picked
    (`s_minus_p`). Multi-event families (>= `min_show` members) are stacked top-to-bottom,
    ordered by `order_by` ('size' or 'meancc'); within a family, traces are sorted by correlation
    to the family stack (most coherent on top), capped at `max_per_cluster`.

    `show_singletons=True` appends EVERY remaining event (clusters smaller than `min_show`, i.e.
    singletons / tiny groups) as a grey "unclustered" block at the bottom — so no waveform is
    omitted (`max_singletons` caps that block for huge catalogs; None = all). Pass `colors`
    (cluster_id -> colour) to stay consistent with the other figures.

    `trace_values` (array aligned to `kept`, e.g. hour-of-day) overrides the wiggle colour with
    `value_cmap`(`value_range`) and adds a colorbar — the grouping/labels stay cluster-coloured, so
    you read family on the left and the per-event value (e.g. KST hour, cyclic 'hsv') on the trace.
    A blast family then shows one colour band (single time-of-day); a tectonic one is mixed.

    `annotate_utc` (default True) prints each trace's **event origin time in UTC** (`YYYY-MM-DD
    HH:MM:SS`, from the event-dir name) in the right margin, so every wiggle is timestamped."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    labels = np.asarray(labels); kept = list(kept)
    if sp is None:
        sp = s_minus_p(kept, station, wf_root)
    t = np.arange(X.shape[1]) / sr + win[0]
    ids, counts = np.unique(labels, return_counts=True)
    keep = ids[counts >= min_show]
    if order_by == "meancc":
        def _key(c):
            idx = np.where(labels == c)[0]; st = _l2(X[idx].mean(0))
            return -float((X[idx] @ st).mean())
        keep = sorted(keep, key=_key)
    else:
        keep = keep[np.argsort(-counts[np.isin(ids, keep)])]
    if clusters is not None:                                  # explicit subset (e.g. blast families)
        keep = [int(c) for c in clusters if (labels == c).any()]
    elif max_clusters is not None:
        keep = list(keep)[:max_clusters]
    colors = colors or cluster_colors(keep)
    rows, clabels = [], []
    y, gap = 0, 1.6
    for c in keep:
        idx = np.where(labels == c)[0]
        st = _l2(X[idx].mean(0)); ccs = X[idx] @ st         # mean CC over ALL members (label)
        clabels.append((int(c), int(counts[ids == c][0]), y, float(ccs.mean()), colors[int(c)]))
        # order traces CHRONOLOGICALLY within the family (kept is time-sorted, so the row index is
        # itself chronological): top = earliest, bottom = most recent.
        idx = np.sort(idx)[:max_per_cluster]
        for i in idx:
            rows.append((i, y, colors[int(c)])); y += 1
        y += gap
    if show_singletons:
        rest = np.where(~np.isin(labels, keep))[0]                # everything below min_show
        if max_singletons is not None:
            rest = rest[:max_singletons]
        if len(rest):
            clabels.append((-1, len(rest), y, np.nan, singleton_color))    # 'unclustered' header
            for i in rest:                                        # event/time order (kept is sorted)
                rows.append((i, y, singleton_color)); y += 1
            y += gap
    smap = None
    if trace_values is not None:
        tv = np.asarray(trace_values, dtype=float)
        vr = value_range or (np.nanmin(tv), np.nanmax(tv))
        cmap = plt.get_cmap(value_cmap); norm = mpl.colors.Normalize(*vr)
    # height = row_h per trace + a fixed header (title/xlabel/ticks); `head_in` keeps the per-trace
    # height CONSTANT across families of different sizes (else a small family hits `min_fig_h` and
    # its traces get stretched). `min_fig_h` is just a floor against a degenerate tiny figure.
    fig, ax = plt.subplots(figsize=(fig_w, max(min_fig_h, row_h * y + head_in)), dpi=120)
    ytrans = ax.get_yaxis_transform()                        # x in axes-fraction, y in data coords
    for i, yy, col in rows:
        w = X[i] / (np.max(np.abs(X[i])) + 1e-9) * 0.45      # per-trace display normalise
        if trace_values is not None:
            col = cmap(norm(tv[i])) if np.isfinite(tv[i]) else "0.7"
        ax.plot(t, yy - w, color=col, lw=0.45)
        if np.isfinite(sp[i]) and win[0] <= sp[i] <= win[1]:
            ax.plot([sp[i], sp[i]], [yy - 0.45, yy + 0.45], color="k", lw=0.8)   # S
        if annotate_utc:                                     # event origin time (UTC) from dir name,
            e = kept[i]                                       # pinned just right of the axes (no margin)
            ax.text(1.005, yy, f"{e[0:4]}-{e[4:6]}-{e[6:8]} {e[8:10]}:{e[10:12]}:{e[12:14]}",
                    transform=ytrans, fontsize=4.2, va="center", ha="left", color="0.25",
                    clip_on=False)
    ax.axvline(0, color="b", lw=0.9, ls="--")                # P
    for c, n, yy, mcc, col in clabels:
        lbl = f"unclustered\nn={n}" if c == -1 else f"cl {c}\nn={n}\ncc={mcc:.2f}"
        ax.text(win[0] - 0.08, yy - 0.5, lbl, ha="right", va="top", fontsize=6.5,
                color=col, weight="bold")
        ax.axhline(yy - gap / 2, color="0.85", lw=0.5)
    ax.set_xlim(win[0] - 0.9, win[1]); ax.set_ylim(y - gap, -gap)
    ax.set_yticks([]); ax.set_xlabel("Time from P (s)")
    ax.set_title(title or (f"{station} {comp} — every trace, P-aligned, coloured by cluster "
                           f"(grey = unclustered; blue dashed = P, black bar = S)"))
    if trace_values is not None:
        smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); smap.set_array([])
        cb = fig.colorbar(smap, ax=ax, fraction=0.03, pad=0.16 if annotate_utc else 0.02)
        cb.set_label(value_label)
    fig.tight_layout()
    if annotate_utc:                                         # reserve room for the UTC column
        fig.subplots_adjust(right=0.80 if trace_values is None else 0.74)
    return fig


def plot_all_chronological(X, labels, kept, sr=SR, win=DEFAULT_WIN, station=STATION, comp=COMP,
                           wf_root=WF_ROOT, min_show=3, colors=None, singleton_color="0.6",
                           sp=None, row_h=0.06, fig_w=9, title=None, annotate_utc=True,
                           mark_years=True, lw=0.4):
    """Plot **EVERY** event's trace in one global CHRONOLOGICAL stack (top = oldest → bottom =
    newest), P-aligned at *t*=0, coloured by its waveform family.

    Unlike `plot_cluster_sections` this applies **no caps** — all `len(kept)` traces are drawn — and
    does **not** group by cluster, so you read pure time order. A multi-event family (>= `min_show`
    members) keeps its `colors` colour; singletons / tiny groups are `singleton_color` grey, so
    repeating families still stand out as same-colour bands recurring through time. Year boundaries
    are ruled + labelled on the left; each trace's UTC origin is on the right (`annotate_utc`).
    `kept` must be in time order (it is — `list_events` returns sorted event-dir names). Uses a
    `LineCollection` so thousands of traces render fast. NOTE: for the full catalog this figure is
    very tall (~`row_h`×N inches); lower `row_h` or pass a one-year `kept` for a shorter plot."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    labels = np.asarray(labels); kept = list(kept)
    if sp is None:
        sp = s_minus_p(kept, station, wf_root)
    t = np.arange(X.shape[1]) / sr + win[0]
    ids, counts = np.unique(labels, return_counts=True)
    fam = set(int(c) for c in ids[counts >= min_show])
    fam_sorted = sorted(fam, key=lambda c: -int(counts[ids == c][0]))
    cols = colors or cluster_colors(fam_sorted)
    n = len(kept)
    segs, segcols, smarks = [], [], []
    for y in range(n):                                       # kept is chronological -> y is time
        c = int(labels[y])
        col = cols.get(c, singleton_color) if c in fam else singleton_color
        w = X[y] / (np.max(np.abs(X[y])) + 1e-9) * 0.45
        segs.append(np.column_stack([t, y - w])); segcols.append(col)
        if np.isfinite(sp[y]) and win[0] <= sp[y] <= win[1]:
            smarks.append([[sp[y], y - 0.45], [sp[y], y + 0.45]])
    fig, ax = plt.subplots(figsize=(fig_w, max(3.0, row_h * n)), dpi=110)
    ax.add_collection(LineCollection(segs, colors=segcols, linewidths=lw))
    if smarks:
        ax.add_collection(LineCollection(smarks, colors="k", linewidths=0.5))
    ax.axvline(0, color="b", lw=0.9, ls="--")                # P
    ytrans = ax.get_yaxis_transform()
    if annotate_utc:
        for y in range(n):
            e = kept[y]
            ax.text(1.005, y, f"{e[0:4]}-{e[4:6]}-{e[6:8]} {e[8:10]}:{e[10:12]}:{e[12:14]}",
                    transform=ytrans, fontsize=3.4, va="center", ha="left", color="0.3",
                    clip_on=False)
    if mark_years:                                           # year rule + label on the left
        prev = None
        for y in range(n):
            yr = kept[y][:4]
            if yr != prev:
                ax.axhline(y - 0.5, color="0.8", lw=0.5)
                ax.text(win[0] - 0.18, y, yr, ha="right", va="center", fontsize=7.5,
                        weight="bold", color="0.2")
                prev = yr
    ax.set_xlim(win[0] - 0.9, win[1]); ax.set_ylim(n, -1)    # oldest at top
    ax.set_yticks([]); ax.set_xlabel("Time from P (s)")
    ax.set_title(title or (f"{station} {comp} — ALL {n} events, chronological (top=oldest → "
                           f"bottom=newest); colour = family, grey = singleton"))
    fig.tight_layout()
    if annotate_utc:
        fig.subplots_adjust(right=0.80)
    return fig


def plot_cluster_grid(X, labels, kept, sr=SR, win=DEFAULT_WIN, station=STATION, comp=COMP,
                      wf_root=WF_ROOT, min_show=4, colors=None, ncol=6, max_per_cluster=None,
                      order_by="size", title=None, sp=None, panel_h=1.6, panel_w=2.2):
    """Small-multiples: **one panel per waveform family** (>= `min_show` members), each panel
    stacking that family's member traces **chronologically** (oldest at top), P-aligned at *t*=0, in
    the family colour — **every family member is drawn** (no cap unless `max_per_cluster`), so nothing
    is omitted within families. The sane alternative to one giant all-events stack: ~100 compact
    panels you can scan, instead of a 100+-inch scroll.

    Singletons / sub-`min_show` groups are NOT panelled (they don't repeat — they're the non-blast
    background); their count is shown in the suptitle. Panels are ordered by `order_by` ('size' or
    'meancc'). Blue dashed = P, black bar = S; each panel titled `cl <id>: n, yrs`."""
    import matplotlib.pyplot as plt
    labels = np.asarray(labels); kept = list(kept)
    if sp is None:
        sp = s_minus_p(kept, station, wf_root)
    t = np.arange(X.shape[1]) / sr + win[0]
    ids, counts = np.unique(labels, return_counts=True)
    fams = ids[counts >= min_show]
    if order_by == "meancc":
        def _key(c):
            idx = np.where(labels == c)[0]; st = _l2(X[idx].mean(0)); return -float((X[idx] @ st).mean())
        fams = sorted(fams, key=_key)
    else:
        fams = list(fams[np.argsort(-counts[np.isin(ids, fams)])])
    n_sing = int((counts < min_show).sum())
    cols = colors or cluster_colors(fams)
    nfam = len(fams)
    if not nfam:
        return None
    ncol = min(ncol, nfam)
    nrow = int(np.ceil(nfam / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel_w * ncol, panel_h * nrow), dpi=120,
                             squeeze=False, sharex=True)
    for ax, c in zip(axes.ravel(), fams):
        idx = np.where(labels == c)[0]
        st = _l2(X[idx].mean(0)); mcc = float((X[idx] @ st).mean())
        idx = np.sort(idx)                                   # chronological (kept is time-sorted)
        if max_per_cluster:
            idx = idx[:max_per_cluster]
        col = cols.get(int(c), "crimson")
        for k, i in enumerate(idx):
            w = X[i] / (np.max(np.abs(X[i])) + 1e-9) * 0.45
            ax.plot(t, k - w, color=col, lw=0.4)
            if np.isfinite(sp[i]) and win[0] <= sp[i] <= win[1]:
                ax.plot([sp[i], sp[i]], [k - 0.45, k + 0.45], color="k", lw=0.5)
        ax.axvline(0, color="b", lw=0.7, ls="--")
        yrs = f"{kept[idx[0]][:4]}–{kept[idx[-1]][:4]}"
        ax.set_title(f"cl {int(c)}: n={len(np.where(labels == c)[0])}, cc={mcc:.2f}\n{yrs}",
                     fontsize=6.5, color=col)
        ax.set_xlim(win[0] - 0.3, win[1]); ax.set_ylim(len(idx), -1)
        ax.set_yticks([]); ax.tick_params(labelsize=6)
    for ax in axes.ravel()[nfam:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Time from P (s)", fontsize=7)
    fig.suptitle(title or (f"{station} {comp} — every family as its own chronological gather "
                           f"({nfam} families ≥{min_show}; {n_sing} singletons not shown)"),
                 fontsize=10, y=0.998)
    fig.tight_layout()
    return fig


def plot_clusters_individually(X, labels, kept, sr=SR, win=DEFAULT_WIN, station=STATION, comp=COMP,
                               wf_root=WF_ROOT, min_show=4, colors=None, order_by="size",
                               max_per_cluster=None, annotate_utc=True, row_h=0.16, fig_w=9,
                               sp=None, show=True):
    """Render **each family as its OWN full-size chronological gather** — one separate figure per
    cluster, NOT a subplot grid. Every trace keeps the **same height** (`row_h`) regardless of family
    size, so a panel's height grows with its member count and the UTC origin times stay legible —
    matching the look of the per-event gathers.

    Each figure is `plot_cluster_sections` restricted to one cluster (all members, chronological,
    `colors`-coloured, P/S marks, UTC on the right). Families are taken in `order_by` ('size' or
    'meancc') order; singletons are skipped. With `show=True` (default) each figure is displayed
    inline and closed (no 'too many open figures' warning, low memory); with `show=False` the list
    of `(cluster_id, fig)` is returned instead (for testing / saving). `sp` is computed once and
    reused across panels."""
    import matplotlib.pyplot as plt
    labels = np.asarray(labels)
    ids, counts = np.unique(labels, return_counts=True)
    fams = ids[counts >= min_show]
    if order_by == "meancc":
        def _key(c):
            idx = np.where(labels == c)[0]; st = _l2(X[idx].mean(0)); return -float((X[idx] @ st).mean())
        fams = sorted(fams, key=_key)
    else:
        fams = list(fams[np.argsort(-counts[np.isin(ids, fams)])])
    cols = colors or cluster_colors(list(fams))
    if sp is None:
        sp = s_minus_p(kept, station, wf_root)
    try:
        from IPython.display import display
    except Exception:                                       # noqa: BLE001
        display = None
    out = []
    for c in fams:
        n = int(counts[ids == c][0])
        f = plot_cluster_sections(X, labels, kept, sr=sr, win=win, station=station, comp=comp,
                                  wf_root=wf_root, clusters=[int(c)], colors=cols,
                                  show_singletons=False, max_per_cluster=max_per_cluster,
                                  annotate_utc=annotate_utc, row_h=row_h, fig_w=fig_w, sp=sp,
                                  min_fig_h=1.2, head_in=0.92,  # constant per-trace height across families
                                  title=f"{station} {comp} — cluster {int(c)} (n={n}), chronological")
        if show and display is not None:
            display(f); plt.close(f)
        else:
            out.append((int(c), f))
    return out


def spacetime_region(meta, pad=0.03):
    """Fixed map region [W, E, S, N] enclosing ALL joined events (+`pad`°) — the SAME extent for
    every family's map so the per-cluster space-time panels are spatially comparable."""
    m = meta[meta["joined"]] if "joined" in meta else meta.dropna(subset=["lat"])
    if not len(m):
        return [ufc.SUBREGION[0] - 0.1, ufc.SUBREGION[1] + 0.1,
                ufc.SUBREGION[2] - 0.1, ufc.SUBREGION[3] + 0.1]
    return [float(m["lon"].min()) - pad, float(m["lon"].max()) + pad,
            float(m["lat"].min()) - pad, float(m["lat"].max()) + pad]


_COAST_GMT_CACHE: dict = {}


def coast_segments_gmt(reg, res="f"):
    """Coastline polylines for `reg` [W,E,S,N] dumped from GMT (`gmt coast -M`) — no cartopy needed
    (GMT/GSHHG ships with PyGMT). Returns a list of (n,2) [lon,lat] arrays; cached per (reg,res), so
    the shared fixed extent costs ONE `gmt` call for all families. [] on any failure (maps still
    render without a shoreline)."""
    import subprocess
    key = (round(reg[0], 3), round(reg[1], 3), round(reg[2], 3), round(reg[3], 3), res)
    if key in _COAST_GMT_CACHE:
        return _COAST_GMT_CACHE[key]
    segs, seg = [], []
    try:
        out = subprocess.run(
            ["gmt", "coast", f"-R{reg[0]}/{reg[1]}/{reg[2]}/{reg[3]}", f"-D{res}", "-M", "-W"],
            capture_output=True, text=True, timeout=60)
        for ln in out.stdout.splitlines():
            if ln.startswith(">"):
                if len(seg) >= 2:
                    segs.append(np.asarray(seg, dtype=float))
                seg = []
            else:
                p = ln.split()
                if len(p) >= 2:
                    try:
                        seg.append([float(p[0]), float(p[1])])      # lon, lat
                    except ValueError:
                        pass
        if len(seg) >= 2:
            segs.append(np.asarray(seg, dtype=float))
    except Exception:                                       # noqa: BLE001
        pass
    _COAST_GMT_CACHE[key] = segs
    return segs


def draw_coast(ax, reg, res="f", color="#3a6ea5", lw=0.6, zorder=0.5):
    """Draw the GMT shoreline (`coast_segments_gmt`) on a lon/lat matplotlib Axes."""
    for s in coast_segments_gmt(reg, res):
        ax.plot(s[:, 0], s[:, 1], color=color, lw=lw, zorder=zorder)


def cluster_spacetime_fig(cid, X, labels, kept, meta, reg, colors=None, win=DEFAULT_WIN, sr=SR,
                          station=STATION, comp=COMP, wf_root=WF_ROOT, sp=None,
                          faults=ufc.FAULT_TRACE, summary_csv=CLUSTER_SUMMARY,
                          year_range=(2010, 2025), row_h=0.16, map_cmap="viridis"):
    """Composite **space-time** panel for ONE family: the chronological waveform gather (left, every
    member, constant per-trace height, UTC origins) + a **fixed-extent** epicentre map (right top,
    this family's events coloured by **origin year**, all other events faint grey context) + a
    **cumulative-count vs year** step curve (right bottom). Together: where the family is and how it
    accumulates through time — a tight pocket filling up across years is the quarry-blast signature.

    The map is drawn in **matplotlib** (GMT shoreline via `draw_coast`, faults via
    `uf_cluster.plot_faults_mpl`) so rendering ~100 of these is fast — PyGMT (used for the publication
    maps in notebook 04 §6) is too slow per-call at this count. `reg` (use `spacetime_region(meta)`)
    is shared across families, so the shoreline costs one `gmt` call total. Returns the figure."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    labels = np.asarray(labels)
    idx = np.sort(np.where(labels == cid)[0])                # chronological (kept is time-sorted)
    n = len(idx)
    evs = [kept[i] for i in idx]
    yrs = event_decimal_years(evs)
    m = meta.reset_index(drop=True)
    sub = m.iloc[idx]
    jn = sub["joined"].values if "joined" in sub else np.isfinite(sub["lat"].values)
    flon, flat, ftime = sub["lon"].values[jn], sub["lat"].values[jn], yrs[jn]
    allj = m[m["joined"]] if "joined" in m else m.dropna(subset=["lat"])
    col = (colors or cluster_colors([cid])).get(int(cid), "crimson")
    if sp is None:
        sp = s_minus_p(kept, station, wf_root)
    # ---- composite figure --------------------------------------------------------------------
    figh = max(3.2, row_h * (n + 1.6) + 0.92)
    F = plt.figure(figsize=(13, figh), dpi=120)
    gs = GridSpec(2, 2, width_ratios=[1.5, 1.0], height_ratios=[3, 1], figure=F,
                  wspace=0.18, hspace=0.30)
    axg = F.add_subplot(gs[:, 0]); axm = F.add_subplot(gs[0, 1]); axc = F.add_subplot(gs[1, 1])
    t = np.arange(X.shape[1]) / sr + win[0]
    ytr = axg.get_yaxis_transform()
    for k, i in enumerate(idx):
        w = X[i] / (np.max(np.abs(X[i])) + 1e-9) * 0.45
        axg.plot(t, k - w, color=col, lw=0.45)
        if np.isfinite(sp[i]) and win[0] <= sp[i] <= win[1]:
            axg.plot([sp[i], sp[i]], [k - 0.45, k + 0.45], color="k", lw=0.6)
        e = kept[i]
        axg.text(1.005, k, f"{e[0:4]}-{e[4:6]}-{e[6:8]} {e[8:10]}:{e[10:12]}:{e[12:14]}",
                 transform=ytr, fontsize=4.2, va="center", ha="left", color="0.3", clip_on=False)
    axg.axvline(0, color="b", lw=0.9, ls="--")
    axg.set_xlim(win[0] - 0.9, win[1]); axg.set_ylim(n, -1); axg.set_yticks([])
    axg.set_xlabel("Time from P (s)")
    axg.set_title(f"{station} {comp} — cluster {int(cid)} (n={n}), chronological", color=col)
    # ---- fixed-extent matplotlib epicentre map (fast; GMT shoreline + uf_cluster faults) -----
    draw_coast(axm, reg); ufc.plot_faults_mpl(axm, faults, color="0.4", lw=0.5)
    axm.scatter(allj["lon"], allj["lat"], s=2, c="0.85", lw=0, zorder=1)            # context
    if len(flon):
        scm = axm.scatter(flon, flat, c=ftime, cmap=map_cmap, vmin=year_range[0],
                          vmax=year_range[1], s=26, edgecolor="k", linewidth=0.3, zorder=3)
        ufc._match_cbar(scm, axm, "Origin year")
    if summary_csv and os.path.exists(summary_csv):
        cs = pd.read_csv(summary_csv); q = cs[cs.get("is_blast", False) == True]
        if len(q):
            axm.scatter(q["lon_centroid"], q["lat_centroid"], marker="x", c="red", s=38,
                        linewidth=1.4, zorder=2)
    try:
        tr = read(_sac_path(kept[0], station))[0]
        axm.scatter([tr.stats.sac.stlo], [tr.stats.sac.stla], marker="s", c="yellow",
                    edgecolor="k", s=55, zorder=4)
    except Exception:                                       # noqa: BLE001
        pass
    axm.set_xlim(reg[0], reg[1]); axm.set_ylim(reg[2], reg[3])
    axm.set_aspect(1.0 / np.cos(np.radians(0.5 * (reg[2] + reg[3]))))               # geographic
    axm.tick_params(labelsize=6); axm.set_xlabel("lon", fontsize=7); axm.set_ylabel("lat", fontsize=7)
    ys = np.sort(yrs)
    axc.step(ys, np.arange(1, n + 1), where="post", color=col, lw=1.4)
    axc.scatter(ys, np.arange(1, n + 1), s=8, color=col, zorder=3)
    axc.set_xlim(*year_range); axc.set_ylim(0, n + 1)
    axc.set_xlabel("Year"); axc.set_ylabel("cumulative N"); axc.grid(alpha=0.3)
    axc.set_title("time-cumulative", fontsize=8)
    F.subplots_adjust(right=0.84)
    return F


def plot_clusters_spacetime(X, labels, kept, meta, reg=None, sr=SR, win=DEFAULT_WIN, station=STATION,
                            comp=COMP, wf_root=WF_ROOT, min_show=4, colors=None, order_by="size",
                            sp=None, show=True, **kw):
    """Render `cluster_spacetime_fig` for EVERY family (≥ `min_show`), one composite per cluster, in
    `order_by` order. `reg` defaults to `spacetime_region(meta)` (fixed extent shared by all). With
    `show=True` each figure is displayed inline and closed; else a list of `(cid, fig)` is returned.
    `sp` is computed once and reused."""
    import matplotlib.pyplot as plt
    labels = np.asarray(labels)
    ids, counts = np.unique(labels, return_counts=True)
    fams = ids[counts >= min_show]
    if order_by == "meancc":
        def _key(c):
            ix = np.where(labels == c)[0]; st = _l2(X[ix].mean(0)); return -float((X[ix] @ st).mean())
        fams = sorted(fams, key=_key)
    else:
        fams = list(fams[np.argsort(-counts[np.isin(ids, fams)])])
    cols = colors or cluster_colors(list(fams))
    if reg is None:
        reg = spacetime_region(meta)
    if sp is None:
        sp = s_minus_p(kept, station, wf_root)
    try:
        from IPython.display import display
    except Exception:                                       # noqa: BLE001
        display = None
    out = []
    for c in fams:
        f = cluster_spacetime_fig(int(c), X, labels, kept, meta, reg, colors=cols, win=win, sr=sr,
                                  station=station, comp=comp, wf_root=wf_root, sp=sp, **kw)
        if show and display is not None:
            display(f); plt.close(f)
        else:
            out.append((int(c), f))
    return out


def used_stations(events, sta_dir=STA_DIR, wf_root=WF_ROOT):
    """Stations that actually recorded `events`, with coordinates, for the corresponding year(s).

    `used` = the set of NET.STA appearing in the events' SAC filenames; coordinates come from the
    per-year `UF{year}.sta` tables for the years spanned by `events`. Returns a DataFrame with
    columns [station, lat, lon]."""
    years = sorted({e[:4] for e in events})
    frames = []
    for y in years:
        p = os.path.join(sta_dir, f"UF{y}.sta")
        if os.path.exists(p):
            d = ufc.load_stations(p)
            frames.append(d.rename(columns={"Networkcode": "station", "Latitude": "lat",
                                            "Longitude": "lon"})[["station", "lat", "lon"]])
    coords = (pd.concat(frames).drop_duplicates("station") if frames else
              pd.DataFrame(columns=["station", "lat", "lon"]))
    used = set()
    for ev in events:
        for f in glob(os.path.join(wf_root, ev, f"{ev}.*.sac")):
            p = os.path.basename(f).split(".")               # ev . NET . STA . CHA . sac
            if len(p) >= 4:
                used.add(f"{p[1]}.{p[2]}")
    return coords[coords["station"].isin(used)].reset_index(drop=True)


def _gmt_rgb(c):
    """matplotlib colour -> GMT 'R/G/B' (0-255) string."""
    import matplotlib.colors as mcolors
    r, g, b, _ = mcolors.to_rgba(c)
    return f"{int(r * 255)}/{int(g * 255)}/{int(b * 255)}"


def plot_cluster_spectrograms(Xsig, labels, kept, sr=SR, win=DEFAULT_WIN, station=STATION,
                              comp=COMP, fmin=0.5, fmax=40.0, nperseg=200, noverlap=184,
                              cmap="magma", min_show=3, max_per_cluster=60, order_by="size",
                              colors=None, show_singletons=True, max_singletons=None, sp=None,
                              strip_h=1.0, gap=0.0, row_h=0.26, fig_w=9, title=None,
                              hours=None, hour_cmap="hsv", hour_range=(0, 24), max_clusters=None):
    """Stacked per-event SPECTROGRAM gather: **common x = time from P**, each event a spectrogram
    strip (**y = `fmin`..`fmax` Hz**, colour = relative power), **grouped by cluster** — the
    time–frequency twin of `plot_cluster_sections`.

    `Xsig` is an aligned, windowed signal matrix whose rows match `kept` (use `raw_matrix(res)` for
    true spectral content). Each strip is dB power, normalised per event so its spectral *shape* is
    visible (not absolute level). Same family ordering / labels / grey 'unclustered' block; blue
    dashed = P, cyan tick = S. Compare spectral character across families — blasts often differ
    from quakes (e.g. richer low-frequency / Rg, spectral scalloping)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from scipy.signal import stft
    labels = np.asarray(labels); kept = list(kept)
    if sp is None:
        sp = s_minus_p(kept, station, WF_ROOT)
    hcmap = plt.get_cmap(hour_cmap); hnorm = mpl.colors.Normalize(*hour_range)
    htab_x = win[1] + 0.12                                    # right-margin hour annotation column
    htab_w = 0.30
    ids, counts = np.unique(labels, return_counts=True)
    keep = ids[counts >= min_show]
    if order_by == "meancc":
        def _key(c):
            idx = np.where(labels == c)[0]; st = _l2(Xsig[idx].mean(0))
            return -float((Xsig[idx] @ st).mean())
        keep = sorted(keep, key=_key)
    else:
        keep = keep[np.argsort(-counts[np.isin(ids, keep)])]
    if max_clusters is not None:
        keep = list(keep)[:max_clusters]
    colors = colors or cluster_colors(keep)
    # ordered blocks (family -> ... -> unclustered)
    ordered = []
    for c in keep:
        idx = np.where(labels == c)[0]                       # chronological within family (kept sorted)
        ordered.append((int(c), list(np.sort(idx)[:max_per_cluster]), colors[int(c)]))
    if show_singletons:
        rest = np.where(~np.isin(labels, keep))[0]
        if max_singletons is not None:
            rest = rest[:max_singletons]
        if len(rest):
            ordered.append((-1, list(rest), "0.4"))
    nrow = sum(len(p) for _, p, _ in ordered)
    total = nrow * (strip_h + gap)
    fig, ax = plt.subplots(figsize=(fig_w, max(3.0, row_h * total)), dpi=120)
    ytop, im, clabels = total, None, []
    for cid, idxs, col in ordered:
        clabels.append((cid, len(idxs) if cid == -1 else int(counts[ids == cid][0]), ytop, col))
        for i in idxs:
            ybot = ytop - strip_h
            sig = Xsig[i]
            npg = min(nperseg, len(sig))
            # stft with zero-boundary padding so the time-frequency image covers the FULL window
            # (scipy.spectrogram centres its windows and would leave ~nperseg/2 s blank at each edge)
            f, tt, Z = stft(sig, fs=sr, window="hann", nperseg=npg,
                            noverlap=min(noverlap, npg - 1), boundary="zeros", padded=True)
            m = (f >= fmin) & (f <= fmax)
            Sdb = 10 * np.log10(np.abs(Z[m]) ** 2 + 1e-12)
            Sn = (Sdb - Sdb.min()) / (Sdb.max() - Sdb.min() + 1e-9)
            im = ax.imshow(Sn, extent=[win[0], win[1], ybot, ybot + strip_h],   # full window
                           origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1,
                           interpolation="bilinear", zorder=1)
            if np.isfinite(sp[i]) and win[0] <= sp[i] <= win[1]:
                ax.plot([sp[i], sp[i]], [ybot, ybot + strip_h], color="cyan", lw=0.5, zorder=3)
            if hours is not None and np.isfinite(hours[i]):  # hour-of-day annotation (right)
                ax.add_patch(mpl.patches.Rectangle((htab_x, ybot), htab_w, strip_h,
                             facecolor=hcmap(hnorm(hours[i])), edgecolor="none", zorder=3))
                ax.text(htab_x + htab_w + 0.06, ybot + strip_h / 2, f"{hours[i]:.0f}",
                        fontsize=4.5, va="center", ha="left", zorder=4)
            ytop -= (strip_h + gap)
    ax.axvline(0, color="b", lw=0.9, ls="--", zorder=4)      # P
    for cid, n, yy, col in clabels:
        lbl = f"unclustered\nn={n}" if cid == -1 else f"cl {cid}\nn={n}"
        ax.text(win[0] - 0.08, yy - strip_h, lbl, ha="right", va="top", fontsize=6.5,
                color=col, weight="bold")
        ax.axhline(yy, color="w", lw=1.2, zorder=3)          # cluster boundary (contiguous strips)
    if hours is not None:
        ax.text(htab_x + htab_w / 2, total + 0.4 * strip_h, "hr", fontsize=6, ha="center",
                va="bottom")
    ax.set_xlim(win[0] - 0.9, htab_x + htab_w + 0.6 if hours is not None else win[1])
    ax.set_ylim(0, total)
    ax.set_yticks([]); ax.set_xlabel("Time from P (s)")
    ax.set_title(title or (f"{station} {comp} — per-event spectrogram ({fmin}–{fmax} Hz), "
                           f"grouped by cluster (blue dashed = P, cyan = S)"))
    if im is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("Relative power (dB, per event)")
    fig.tight_layout()
    return fig


def map_clusters(meta, labels, evidence, reg=None, title="Waveform clusters",
                 station=STATION, top=None, subregion=ufc.SUBREGION,
                 fault_trace=ufc.FAULT_TRACE, summary_csv=CLUSTER_SUMMARY, colors=None,
                 show_used_stations=True, station_pad=0.08):
    """PyGMT map of joined events coloured by waveform cluster (only `evidence` clusters; the
    rest grey). The common station (`station`) is a yellow square; all OTHER stations that
    recorded these events that year (`used_stations`) are grey squares; known quarry centroids
    (cluster_summary `is_blast`) are red x's. When `reg=None` the frame auto-expands to include
    the events + used stations (+ `station_pad`°). `colors` matches the waveform-gather colours."""
    import pygmt as pmt
    m = meta.copy().reset_index(drop=True)
    m["wf_cluster"] = labels
    m = m[m["joined"]]
    sta_used = used_stations(list(meta["event"])) if show_used_stations else \
        pd.DataFrame(columns=["station", "lat", "lon"])
    if reg is None:
        lons, lats = list(m["lon"]), list(m["lat"])
        if len(sta_used):
            lons += list(sta_used["lon"]); lats += list(sta_used["lat"])
        reg = ([min(lons) - station_pad, max(lons) + station_pad,
                min(lats) - station_pad, max(lats) + station_pad] if lons else
               [ufc.SUBREGION[0] - 0.1, ufc.SUBREGION[1] + 0.1,
                ufc.SUBREGION[2] - 0.1, ufc.SUBREGION[3] + 0.1])
    cids = evidence["cluster"].tolist()
    if top is not None:
        cids = evidence["cluster"].head(top).tolist()
    colors = colors or cluster_colors(cids)
    fig = pmt.Figure()
    pmt.config(FORMAT_GEO_MAP="ddd.xx", MAP_FRAME_TYPE="plain")
    fig.basemap(region=reg, projection="M14c", frame=["af", f"+t{title}"])
    fig.coast(land="white", water="lightblue", shorelines=True)
    ufc.plot_faults(fig, fault_trace)
    bg = m[~m["wf_cluster"].isin(cids)]
    fig.plot(x=bg["lon"], y=bg["lat"], style="c0.10c", fill="gray80", pen="0.1p,gray50")
    for cid in cids:
        g = m[m["wf_cluster"] == cid]
        fig.plot(x=g["lon"], y=g["lat"], fill=_gmt_rgb(colors[int(cid)]), style="c0.16c",
                 pen="0.3p,black")
    # known quarry centroids (prior spatial/temporal method) for reference
    if summary_csv and os.path.exists(summary_csv):
        cs = pd.read_csv(summary_csv)
        q = cs[cs.get("is_blast", False) == True]
        if len(q):
            fig.plot(x=q["lon_centroid"], y=q["lat_centroid"], style="x0.5c", pen="2p,red")
    # other stations used that year (grey squares); the station of interest stays yellow
    if len(sta_used):
        other = sta_used[sta_used["station"] != station]
        fig.plot(x=other["lon"], y=other["lat"], style="s0.30c", fill="gray70", pen="0.6p,gray25")
    try:
        tr = read(_sac_path(list(meta["event"])[0], station))[0]
        fig.plot(x=[tr.stats.sac.stlo], y=[tr.stats.sac.stla], style="s0.45c",
                 fill="yellow", pen="1.2p,black")
    except Exception:                                   # noqa: BLE001
        pass
    if subregion is not None:
        bl, ba = ufc._subregion_box(subregion)
        fig.plot(x=bl, y=ba, pen="1.5p,blue")
    return fig


def map_antipairs(meta, pairs, value="cc_neg", station=STATION, reg=None,
                  subregion=ufc.SUBREGION, fault_trace=ufc.FAULT_TRACE, pad=0.08,
                  title="Anti-correlated pairs"):
    """PyGMT map of candidate anti-pairs: each pair = two joined epicentres connected by a straight
    segment, so **co-located** pairs (short segment — a possible same-patch reversal) vs **distant**
    pairs (long segment — coincidental) are obvious by eye. Endpoints are red circles; the common
    `station` is a yellow square; the Ulsan-fault trace + `subregion` box are drawn for context.
    `pairs` = list of dicts with i, j (indices into `meta`) and the `value` key for reference. Pairs
    with an unjoined endpoint (no hypocentre) are skipped. Returns the PyGMT Figure."""
    import pygmt as pmt
    m = meta.copy().reset_index(drop=True)
    segs, lons, lats = [], [], []
    for p in pairs:
        ri, rj = m.iloc[p["i"]], m.iloc[p["j"]]
        if not (ri["joined"] and rj["joined"]):
            continue
        segs.append((ri["lon"], ri["lat"], rj["lon"], rj["lat"]))
        lons += [ri["lon"], rj["lon"]]; lats += [ri["lat"], rj["lat"]]
    fig = pmt.Figure()
    if reg is None:
        reg = ([min(lons) - pad, max(lons) + pad, min(lats) - pad, max(lats) + pad] if lons else
               [subregion[0] - 0.1, subregion[1] + 0.1, subregion[2] - 0.1, subregion[3] + 0.1])
    pmt.config(FORMAT_GEO_MAP="ddd.xx", MAP_FRAME_TYPE="plain")
    fig.basemap(region=reg, projection="M14c", frame=["af", f"+t{title}"])
    fig.coast(land="white", water="lightblue", shorelines=True)
    ufc.plot_faults(fig, fault_trace)
    for lo1, la1, lo2, la2 in segs:
        fig.plot(x=[lo1, lo2], y=[la1, la2], pen="1p,gray40")
    if segs:
        ex = np.array(segs)
        fig.plot(x=np.r_[ex[:, 0], ex[:, 2]], y=np.r_[ex[:, 1], ex[:, 3]],
                 style="c0.16c", fill="red", pen="0.3p,black")
    try:
        tr = read(_sac_path(list(meta["event"])[0], station))[0]
        fig.plot(x=[tr.stats.sac.stlo], y=[tr.stats.sac.stla], style="s0.45c",
                 fill="yellow", pen="1.2p,black")
    except Exception:                                       # noqa: BLE001
        pass
    if subregion is not None:
        bl, ba = ufc._subregion_box(subregion)
        fig.plot(x=bl, y=ba, pen="1.5p,blue")
    return fig


def map_blast_hours(meta, labels, evidence, station=STATION, reg=None, hour_cmap="cyclic",
                    subregion=ufc.SUBREGION, fault_trace=ufc.FAULT_TRACE,
                    summary_csv=CLUSTER_SUMMARY, show_used_stations=True, pad=0.03):
    """PyGMT subregion close-up that HIGHLIGHTS the blast-candidate events, coloured by
    hour-of-day (KST, cyclic cpt — same convention as `uf_cluster.hour_map`).

    Blast families = `evidence.loc[evidence['blast_like'], 'cluster']`; their joined events are
    drawn as larger circles filled by `event_hours()`; all other joined events sit behind as faint
    grey context. Zooms to `ufc.SUBREGION` (+`pad`) by default. KG.HDB is yellow, other used
    stations grey, known quarry centroids red x's. A blast cluster should read as one daytime
    colour band sitting in a compact pocket."""
    import pygmt as pmt
    m = meta.copy().reset_index(drop=True)
    m["wf_cluster"] = labels
    m = m[m["joined"]]
    blast_ids = set(int(c) for c in evidence.loc[evidence.get("blast_like", False), "cluster"])
    is_blast = m["wf_cluster"].isin(blast_ids)
    if reg is None:
        reg = [subregion[0] - pad, subregion[1] + pad, subregion[2] - pad, subregion[3] + pad]
    fig = pmt.Figure()
    pmt.config(FORMAT_GEO_MAP="ddd.xx", MAP_FRAME_TYPE="plain")
    fig.basemap(region=reg, projection="M14c",
                frame=["af", f"+t{station} blast candidates — hour of day (KST)"])
    fig.coast(land="white", water="lightblue", shorelines=True)
    ufc.plot_faults(fig, fault_trace)
    bg = m[~is_blast]
    fig.plot(x=bg["lon"], y=bg["lat"], style="c0.08c", fill="gray85", pen=None)
    # blast events coloured by hour-of-day with a cyclic colormap (matches uf.hour_map)
    b = m[is_blast]
    if len(b):
        pmt.makecpt(cmap=hour_cmap, series=[0, 24, 1], continuous=True)
        fig.plot(x=b["lon"], y=b["lat"], fill=event_hours(list(b["event"])), cmap=True,
                 style="c0.22c", pen="0.4p,black")
        fig.colorbar(frame=["a6", "x+lHour of day (KST)"])
    if summary_csv and os.path.exists(summary_csv):
        cs = pd.read_csv(summary_csv)
        q = cs[cs.get("is_blast", False) == True]
        if len(q):
            fig.plot(x=q["lon_centroid"], y=q["lat_centroid"], style="x0.5c", pen="2p,red")
    if show_used_stations:
        su = used_stations(list(meta["event"]))
        other = su[su["station"] != station]
        if len(other):
            fig.plot(x=other["lon"], y=other["lat"], style="s0.30c", fill="gray70", pen="0.6p,gray25")
    try:
        tr = read(_sac_path(list(meta["event"])[0], station))[0]
        fig.plot(x=[tr.stats.sac.stlo], y=[tr.stats.sac.stla], style="s0.45c",
                 fill="yellow", pen="1.2p,black")
    except Exception:                                       # noqa: BLE001
        pass
    if subregion is not None:
        bl, ba = ufc._subregion_box(subregion)
        fig.plot(x=bl, y=ba, pen="1.5p,blue")
    return fig


def plot_blast_hour_histograms(meta, labels, evidence, station=STATION, colors=None,
                               day=(6, 17), ncol=4, kst=ufc.KST, only_blast=True):
    """Per-cluster hour-of-day (KST) histograms as a subplot grid — the temporal signature behind
    the `blast_like` flag, one panel per family.

    Clusters = the `blast_like` rows of `evidence` (mean_cc order) when `only_blast` (default); set
    `only_blast=False` to show every evidence family. Each panel is the hourly histogram of its
    members' origin hours (`event_hours`, KST — every event, no catalog join), filled in the
    family colour, with the **daytime window `day` shaded** and `n / peak_hour / rayleigh_p`
    annotated. A blast family piles into one daytime bar (small `rayleigh_p`); a tectonic repeater
    spreads across 24 h. Returns the figure (None if no qualifying cluster)."""
    import matplotlib.pyplot as plt
    labels = np.asarray(labels)
    m = meta.copy().reset_index(drop=True)
    sel = evidence[evidence["blast_like"]] if (only_blast and "blast_like" in evidence) else evidence
    cids = [int(c) for c in sel["cluster"].tolist()]
    if not cids:
        return None
    cols = colors or cluster_colors(cids)
    info = sel.set_index("cluster")
    ncol = min(ncol, len(cids)) or 1
    nrow = int(np.ceil(len(cids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.2 * nrow), dpi=130,
                             squeeze=False, sharex=True)
    edges = np.arange(0, 25, 1)
    for ax, cid in zip(axes.ravel(), cids):
        evs = m.loc[labels == cid, "event"].tolist()
        h = event_hours(evs, kst=kst)
        col = cols.get(int(cid), "crimson")
        ax.axvspan(day[0], day[1], color="0.9", zorder=0)          # daytime band
        ax.hist(h, bins=edges, color=col, edgecolor="white", linewidth=0.3, zorder=2)
        r = info.loc[cid]
        ax.set_title(f"cl {cid}: n={int(r['n'])}", fontsize=8, color=col)
        ax.text(0.97, 0.92, f"peak {r['peak_hour']:.0f}h\np={r['rayleigh_p']:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
        ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24]); ax.tick_params(labelsize=6.5)
    for ax in axes.ravel()[len(cids):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Hour of day (KST)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("events", fontsize=7)
    fig.suptitle(f"{station} blast-candidate families — hour-of-day (KST); shaded = daytime "
                 f"{day[0]}–{day[1]} h", fontsize=10)
    fig.tight_layout()
    return fig
