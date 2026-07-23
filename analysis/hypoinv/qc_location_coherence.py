#!/usr/bin/env python
"""P-moveout location-coherence QC for the PhaseNet+ -> PyOcto(strict) -> HypoInverse(kim2011)
catalog. Flags network-edge events the PhaseNet+ run mislocates *into* the study region.

WHY THIS EXISTS
---------------
HypoInverse's own quality fields (rms, erh, erz, qual) are a *within-model* goodness-of-fit
on the *subset of picks HypoInverse kept*, with full freedom in depth + origin-time. For a
network-edge event with few associated picks and a wide gap, the solution can dive deep and
absorb timing scatter -- small RMS for a *wrong* location. Example: 2017-11-15 05:56 (a real
Pohang M3 aftershock) is reported at 35.84N / 21.5 km, rms=0.10 s, gap=159 -- "looks fine",
but is ~30 km south of the true Pohang location. rms/erh/erz are BLIND to this.

The trap is self-consistency: the reported location *is* the least-squares fit to the picks
PyOcto associated, so a moveout test against *those* picks also looks fine (05:56: rms 0.43 s
against the 6 associated picks). What exposes the mislocation is the moveout against ALL
**PhaseNet+ detection picks** -- including the close stations association dropped. For 05:56
the nearest station KG.HDB (13 km) is picked 3 s *after* MKL/HAK at 14-18 km, and CHS at
44 km arrives *before* HDB -- physically impossible for a single source at 35.84. So we use
the SOTA ML picks (NOT STA/LTA, which is too noisy on small events) at every station.

This is PN+-self-consistent: it uses only the PhaseNet+ detection-pick store + the catalog
hypocentre. No cross-picker (stead/original) comparison.

METRIC (per event)
------------------
From the PhaseNet+ detection picks in [origin, origin+WIN_S], earliest P per station,
probability >= P_MIN, vs the catalog hypocentre, robust (Theil-Sen) t-vs-distance fit:
  mv_nP     : number of stations with a usable P pick
  mv_app    : apparent velocity [km/s]
  mv_rms    : RMS residual about the line [s]               (diagnostic; outlier-sensitive)
  mv_rho    : Spearman rank corr. of distance vs P-time     (diagnostic)
  mv_inlier : fraction of stations whose P fits the line within INLIER_TOL seconds
  coherence_fail = (mv_nP >= MIN_P) and (mv_inlier < INLIER_MIN)

We reject on mv_inlier, NOT raw rms, on purpose: AI pickers emit a fraction of false /
mis-phased picks, so raw rms (outlier-sensitive) wrongly flags well-located events with a
couple of bad picks. The inlier fraction is robust -- a few bad picks leave the MAJORITY of
stations fitting one moveout, so it stays high; only a genuinely WRONG location knocks the
bulk of stations off the line.

Validation (detection picks): mislocated 05:56 inlier=0.00, 09:58 inlier=0.00, 2016-09-12
inlier=0.36; 25 random good events inlier 0.79-1.00 (incl. ones with rms up to 1.6 from a
few bad picks). Clean gap at INLIER_MIN=0.5. Events with < MIN_P P picks are left unflagged.

USAGE
-----
    python qc_location_coherence.py \
        --catalog ../local_magnitudes/catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo.csv \
        --picks-root ../models/phasenet_plus/detection_location \
        --stations-dir ../station_table \
        --out-scored scored.csv --out-clean clean.csv --out-audit audit.csv

Importable: score_catalog(cat_df, picks_root, stations_dir, ...) -> cat_df + mv_* columns.
"""
import argparse
import os

import numpy as np
import pandas as pd
from obspy.geodetics import gps2dist_azimuth
from scipy.stats import spearmanr, theilslopes

MIN_P = 5          # need >= 5 stations with a P pick to assess moveout
INLIER_TOL = 1.0   # s, a station P "fits" the moveout if |residual| < this
INLIER_MIN = 0.5   # reject if fewer than this fraction of stations fit (robust to bad picks)
WIN_S = 25.0       # s after origin to collect first-arrival P picks (P out to ~150 km)
P_MIN = 0.3        # minimum PhaseNet+ pick probability


def load_stations(stations_dir, prefix, year):
    """Per-year station table -> {NET.CODE: (lat, lon)}; year clamped to available range."""
    years = sorted(int(f[len(prefix):len(prefix) + 4])
                   for f in os.listdir(stations_dir)
                   if f.startswith(prefix) and f.endswith(".csv"))
    y = min(max(year, years[0]), years[-1])
    st = pd.read_csv(os.path.join(stations_dir, f"{prefix}{y}.csv"))
    return {f"{r.Network}.{r.Code}": (r.Latitude, r.Longitude) for r in st.itertuples()}


def _load_day_picks(picks_root, year, doy, _cache={}):
    """All PhaseNet+ detection picks for one day (cached); empty frame if missing."""
    key = (year, doy)
    if key not in _cache:
        f = os.path.join(picks_root, str(year), "picks", f"picks_{year}.{doy:03d}.csv")
        try:
            p = pd.read_csv(f)
            p["t"] = pd.to_datetime(p["peak_time"], utc=True).dt.tz_localize(None)  # tz-naive UTC
            # picks carry NET.STA.LOC ("KG.BBK."); station table keys are NET.STA -> normalize to match
            p["station"] = p["station"].astype(str).str.split(".").str[:2].str.join(".")
        except (FileNotFoundError, OSError):
            p = pd.DataFrame(columns=["station", "phase", "peak_time", "probability", "t"])
        if len(_cache) > 4:           # keep the cache small (catalog iterated in time order)
            _cache.clear()
        _cache[key] = p
    return _cache[key]


def moveout_coherence(day_picks, ev_lat, ev_lon, t0, sc):
    """(nP, v_app, rms, rho, inlier) from PhaseNet+ detection P picks vs the hypocentre;
    None if fewer than MIN_P stations with a usable P pick."""
    w = day_picks[(day_picks.phase == "P") & (day_picks.probability >= P_MIN) &
                  (day_picks.t >= t0) & (day_picks.t <= t0 + pd.Timedelta(seconds=WIN_S))]
    if len(w) == 0:
        return None
    w = w.sort_values("t").groupby("station", as_index=False).first()   # earliest P per station
    d, t = [], []
    for r in w.itertuples():
        if r.station in sc:
            d.append(gps2dist_azimuth(ev_lat, ev_lon, *sc[r.station])[0] / 1000.0)
            t.append(r.t.timestamp())
    if len(d) < MIN_P:
        return None
    d = np.asarray(d, float)
    t = np.asarray(t, float) - min(t)
    sl, ic, _, _ = theilslopes(t, d)          # s per km, robust to outlier picks
    res = t - (ic + sl * d)
    v = 1.0 / sl if sl > 1e-6 else 99.0
    rms = float(np.sqrt(np.mean(res ** 2)))
    inlier = float(np.mean(np.abs(res) < INLIER_TOL))
    rho, _ = spearmanr(d, t)
    return len(d), float(v), rms, float(rho), inlier


def score_catalog(cat, picks_root, stations_dir, stations_prefix="stations_",
                  inlier_min=INLIER_MIN):
    """Append mv_nP / mv_app / mv_rms / mv_rho / mv_inlier / coherence_fail to a copy of
    `cat` (needs columns: time, lat, lon). Iterates in time order so the day-pick cache hits."""
    cat = cat.copy()
    cat["time"] = pd.to_datetime(cat["time"], utc=True).dt.tz_localize(None)  # tz-naive UTC (match picks)
    order = cat["time"].argsort().to_numpy()
    n = len(cat)
    mv_nP = np.full(n, np.nan); mv_app = np.full(n, np.nan)
    mv_rms = np.full(n, np.nan); mv_rho = np.full(n, np.nan); mv_inlier = np.full(n, np.nan)
    sc_cache = {}

    for i in order:
        row = cat.iloc[i]
        t0 = row.time
        year = int(t0.year)
        if year not in sc_cache:
            sc_cache[year] = load_stations(stations_dir, stations_prefix, year)
        day = _load_day_picks(picks_root, year, int(t0.dayofyear))
        if len(day) == 0:
            continue
        res = moveout_coherence(day, row.lat, row.lon, t0, sc_cache[year])
        if res:
            mv_nP[i], mv_app[i], mv_rms[i], mv_rho[i], mv_inlier[i] = res

    cat["mv_nP"] = mv_nP; cat["mv_app"] = mv_app
    cat["mv_rms"] = mv_rms; cat["mv_rho"] = mv_rho; cat["mv_inlier"] = mv_inlier
    cat["coherence_fail"] = ((cat.mv_nP >= MIN_P) & (cat.mv_inlier < inlier_min)).fillna(False)
    return cat


def _report(cat):
    tested = int(cat.mv_rms.notna().sum())
    fail = int(cat.coherence_fail.sum())
    print(f"scored {tested}/{len(cat)} events (>= {MIN_P} P picks)")
    print(f"coherence_fail: {fail} ({100*fail/len(cat):.2f}% of catalog) "
          f"[mv_inlier < {INLIER_MIN}]")
    print("  by year:", cat[cat.coherence_fail].time.dt.year.value_counts().sort_index().to_dict())
    if "magnitude" in cat.columns:
        print(f"  flagged M>=2: {int((cat.coherence_fail&(cat.magnitude>=2)).sum())}, "
              f"M>=3: {int((cat.coherence_fail&(cat.magnitude>=3)).sum())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--picks-root", required=True,
                    help="PhaseNet+ detection_location root (contains {year}/picks/picks_{year}.{doy}.csv)")
    ap.add_argument("--stations-dir", required=True)
    ap.add_argument("--stations-prefix", default="stations_")
    ap.add_argument("--inlier-min", type=float, default=INLIER_MIN)
    ap.add_argument("--out-scored", default=None)
    ap.add_argument("--out-clean", default=None)
    ap.add_argument("--out-audit", default=None)
    a = ap.parse_args()

    cat = pd.read_csv(a.catalog)
    scored = score_catalog(cat, a.picks_root, a.stations_dir,
                           stations_prefix=a.stations_prefix, inlier_min=a.inlier_min)
    _report(scored)
    if a.out_scored:
        scored.to_csv(a.out_scored, index=False)
    if a.out_clean:
        scored[~scored.coherence_fail].drop(columns=["coherence_fail"]).to_csv(a.out_clean, index=False)
        print(f"  clean -> {a.out_clean} ({int((~scored.coherence_fail).sum())} events)")
    if a.out_audit:
        scored[scored.coherence_fail].to_csv(a.out_audit, index=False)
        print(f"  audit -> {a.out_audit} ({int(scored.coherence_fail.sum())} events)")


if __name__ == "__main__":
    main()
