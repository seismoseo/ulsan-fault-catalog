"""In-situ Vp/Vs (Lin & Shearer 2007; SOTA per Huang et al. 2025) for the KG.HDB repeating-earthquake
families of the Ulsan fault.

Repeating earthquakes are the IDEAL Lin & Shearer target: near-identical waveforms ⇒ the P and S
differential travel times between two family members are resolvable to sub-millisecond precision, and
δt_S/δt_P = Vp/Vs exactly for each station (P and S share the ray path). We:

  1. reproduce the KG.HDB families (5–25 Hz, single-linkage, CC ≥ 0.9, n ≥ 5) from the cached HDB
     cross-correlation matrix (uf_waveform_similarity);
  2. for each family, re-cross-correlate the P (native vertical) and S (best horizontal) windows of every
     event pair at every nearby station, with PARABOLIC SUB-SAMPLE refinement of the CC peak — this is
     Huang et al.'s consistent-window idea, essential here because the δt are a few ms (≪ the 10 ms
     sample);
  3. demean δt_P and δt_S per pair (removes the origin-time term) and fit the robust IRLS slope = Vp/Vs
     with the project-16 estimator (kma_absolute_location.vpvs): bootstrap-MAD error, robust-scale RMSE
     gate (Xu et al. 2026), Poisson's ratio.

NOTE — geometry: the catalog hypocentres are absolute HYPOINVERSE locations whose ~km error swamps the
true (tens-of-m) repeater separation, so we DO NOT use the inter-event distance gate or the d/V physical
cap (both need reliable relative locations). Coherence + the robust RMSE gate are the quality arbiters.
δt are cached to disk per family (the cross-correlation is the expensive step)."""
from __future__ import annotations

import os
import sys
import pickle
import itertools
import warnings

import numpy as np

HYPO = "/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv"
sys.path.insert(0, "/home/msseo/works/16.kma_absolute_location")
from obspy import read                                          # noqa: E402
from obspy.signal.cross_correlation import correlate            # noqa: E402
from uflib import uf_waveform_similarity as wf                             # noqa: E402
from kma_absolute_location import vpvs                          # noqa: E402

# --------------------------------------------------------------------- parameters
STATION, COMP = "KG.HDB", "HHZ"
WIN, MAXLAG = (-0.5, 7.5), 0.2          # HDB clustering window / lag (matches the cached CC matrix)
BAND = (5, 25)                          # HDB family-clustering band (user's choice)
CC_REPEAT, MIN_FAMILY = 0.9, 5          # single-linkage family threshold / min members
XB = (5, 25)                            # bandpass for the δt cross-correlation
SR = 100.0
CCXC_MIN = 0.80                         # per-measurement CC floor for a kept δt
MAXLAG_XC = 0.5                         # s; δt CC lag search half-width
P_WIN, S_WIN = (0.3, 0.8), (0.3, 1.2)   # (pre, post) s around the pick for P (vertical) / S (horizontal)
MAX_KM = 40.0                           # nearby-station radius around the family centroid
MIN_STATIONS = 3                        # common P&S stations required per event pair
MIN_PAIRS = 10                          # qualifying pairs to report a Vp/Vs (else under_determined)
N_BOOT = 1000

CACHE = wf.CACHE_DIR
DT_CACHE = os.path.join(CACHE, "vpvs_dt")
os.makedirs(DT_CACHE, exist_ok=True)
RESULTS_CSV = os.path.join(HYPO, "repeaters", "uf_vpvs_results.csv")


# --------------------------------------------------------------------- families
def families():
    """Reproduce the KG.HDB repeater families from the cached 5–25 Hz CC matrix.
    Returns (rep_table, ev_of_family: {fam: [event,...]}, meta_indexed_by_event)."""
    events = wf.list_events(station=STATION, comp=COMP)
    res = wf.make_bands(events, station=STATION, comp=COMP, bands=[BAND], win=WIN, verbose=False)
    kept = res["kept"]
    meta = wf.load_event_meta(kept)
    tag = (f"{STATION}_{COMP}_w{WIN[0]}_{WIN[1]}_b{BAND[0]}-{BAND[1]}_lag{MAXLAG}_n{len(kept)}"
           .replace(".", "p"))
    CC = np.load(os.path.join(CACHE, f"cc_{tag}.npy"))
    labels, _Z, _order = wf.ward_clusters(CC, threshold=1 - CC_REPEAT, method="single")
    rep = wf.repeater_table(meta, labels, CC, min_size=MIN_FAMILY)
    m2 = meta.assign(fam=labels)
    ev_of = {int(f): list(m2.query("fam == @f")["event"]) for f in rep.cluster.astype(int)}
    return rep, ev_of, meta.set_index("event")


# --------------------------------------------------------------------- δt by cross-correlation
def _trace(ev, sta, ch):
    p = os.path.join(wf.WF_ROOT, ev, f"{ev}.{sta}.{ch}.sac")
    return read(p)[0] if os.path.exists(p) else None


def _win(tr, pick, pre, post):
    if tr is None or pick is None:
        return None
    t = tr.copy().detrend("demean").taper(0.05).filter(
        "bandpass", freqmin=XB[0], freqmax=XB[1], corners=4, zerophase=True)
    s = t.slice(pick - pre, pick + post)
    if s.stats.npts < int((pre + post) * SR * 0.8):
        return None
    x = s.data.astype(float)
    x -= x.mean()
    n = np.linalg.norm(x)
    return x / n if n > 0 else None


def _dt(ev_i, ev_j, sta, ch, phase, pre, post):
    """Differential arrival time (i − j) at a station for a phase, refined by cross-correlation with
    PARABOLIC SUB-SAMPLE interpolation. δt = (pick_i − pick_j) + cc_lag. Returns (δt, cc) or None."""
    pi = wf.pick_time(ev_i, sta, phase)
    pj = wf.pick_time(ev_j, sta, phase)
    if pi is None or pj is None:
        return None
    xi = _win(_trace(ev_i, sta, ch), pi, pre, post)
    xj = _win(_trace(ev_j, sta, ch), pj, pre, post)
    if xi is None or xj is None or len(xi) != len(xj):
        return None
    cc = correlate(xi, xj, int(MAXLAG_XC * SR))
    k = int(np.argmax(cc))
    val = float(cc[k])
    if 0 < k < len(cc) - 1:                                     # parabolic peak refinement
        a, b, c = cc[k - 1], cc[k], cc[k + 1]
        denom = a - 2 * b + c
        dsub = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        dsub = 0.0
    shift = (k - len(cc) // 2) + dsub
    return float((pi - pj) + shift / SR), val


def _horiz(ch):
    return [ch[:-1] + "E", ch[:-1] + "N"]


def _az_gap(clat, clon, coords_list):
    """Largest azimuthal gap (deg) of the contributing stations as seen from the family centroid.
    > 180° ⇒ one-sided coverage: the Vp/Vs averages over a narrow ray-direction cone (directional),
    not the full azimuth — critical when interpreting a family's value. Returns 360 if < 2 stations."""
    import math
    if len(coords_list) < 2:
        return 360.0
    kx = math.cos(math.radians(clat))
    az = sorted((math.degrees(math.atan2((lo - clon) * kx, la - clat)) % 360.0) for la, lo in coords_list)
    gaps = [az[i + 1] - az[i] for i in range(len(az) - 1)]
    gaps.append(360.0 - az[-1] + az[0])                        # wrap-around gap
    return max(gaps)


def build_family_dt(fam, evs, meta_idx, force=False):
    """Build (and disk-cache) the per-pair, per-station P and S differential times for one family.
    Returns (P, S) dicts: {(ev_i,ev_j): {station: (δt, cc)}}."""
    cf = os.path.join(DT_CACHE, f"fam{fam}_b{XB[0]}-{XB[1]}_cc{CCXC_MIN}.pkl")
    if os.path.exists(cf) and not force:
        with open(cf, "rb") as fh:
            return pickle.load(fh)
    cen = (meta_idx.loc[evs].lat.mean(), meta_idx.loc[evs].lon.mean())
    ns = wf.nearby_stations(evs, cen, max_km=MAX_KM)
    chan = dict(zip(ns.station, ns.channel))
    P, S = {}, {}
    for i, j in itertools.combinations(range(len(evs)), 2):
        ei, ej = evs[i], evs[j]
        kp, ks = {}, {}
        for sta, chv in chan.items():
            rp = _dt(ei, ej, sta, chv, "P", *P_WIN)
            if rp and rp[1] >= CCXC_MIN:
                kp[sta] = rp
            best = None
            for chh in _horiz(chv):
                rs = _dt(ei, ej, sta, chh, "S", *S_WIN)
                if rs and (best is None or rs[1] > best[1]):
                    best = rs
            if best and best[1] >= CCXC_MIN:
                ks[sta] = best
        if kp:
            P[(ei, ej)] = kp
        if ks:
            S[(ei, ej)] = ks
    with open(cf, "wb") as fh:
        pickle.dump((P, S, dict(zip(ns.station, ns.dist_km))), fh)
    return P, S, dict(zip(ns.station, ns.dist_km))


# --------------------------------------------------------------------- per-family Vp/Vs
def run_family(fam, evs, meta_idx, force=False):
    """Vp/Vs for one repeater family (reuses the project-16 robust estimator + gates).
    Returns a result dict (no inter-event distance gate / physical cap — see module docstring)."""
    P, S, _dist = build_family_dt(fam, evs, meta_idx, force=force)
    m = vpvs.build_measurements(P, S, None, cc_min=CCXC_MIN, min_stations=MIN_STATIONS, dist_min_km=0.0)
    dep = meta_idx.loc[evs].depth.astype(float)
    clat = round(float(meta_idx.loc[evs].lat.mean()), 4)
    clon = round(float(meta_idx.loc[evs].lon.mean()), 4)
    sta_used = m.get("stations", set())
    sc = {r.station: (r.lat, r.lon) for r in wf.used_stations(evs).itertuples()}  # contributing-station coords
    az_gap = _az_gap(clat, clon, [sc[s] for s in sta_used if s in sc])
    rec = dict(family=int(fam), n_events=len(evs), n_pairs=m["n_pairs"], n_obs=int(len(m["dtp"])),
               n_sta=len(sta_used), az_gap=round(az_gap, 0), vpvs=np.nan, mad=np.nan, poisson=np.nan,
               corr=np.nan, rmse=np.nan, out_frac=np.nan, vpvs_ols=np.nan,
               lat=clat, lon=clon, dep_med=round(float(dep.median()), 2),
               t_first=min(evs)[:8], t_last=max(evs)[:8], status="", note="")
    if m["n_pairs"] < MIN_PAIRS:
        rec["status"] = "under_determined"
        rec["note"] = f"{m['n_pairs']} qualifying pairs < {MIN_PAIRS}"
        return rec
    v, mad, _ = vpvs.bootstrap_vpvs(m, n_boot=N_BOOT)
    inl, _rs = vpvs.robust_inliers(m, v)
    resid = m["dts"] - v * m["dtp"]
    rmse = float(np.sqrt(np.mean(resid[inl] ** 2))) if inl.sum() >= 3 else np.nan
    corr = float(np.corrcoef(m["dtp"][inl], m["dts"][inl])[0, 1]) if inl.sum() >= 3 else np.nan
    rec.update(vpvs=round(v, 4), mad=round(mad, 4),
               poisson=round(vpvs.poisson_ratio(v), 4) if v > vpvs.VPVS_FLOOR else np.nan,
               corr=round(corr, 3), rmse=round(rmse, 4), out_frac=round(float((~inl).mean()), 3),
               vpvs_ols=round(float(np.sum(m["dtp"] * m["dts"]) / np.sum(m["dtp"] ** 2)), 4))
    if v <= vpvs.VPVS_FLOOR:
        rec["status"], rec["note"] = "unphysical", f"Vp/Vs={v:.2f} ≤ 1"
    elif rmse > vpvs.RMSE_MAX:
        rec["status"], rec["note"] = "high_rmse", f"robust RMSE={rmse:.3f}s > {vpvs.RMSE_MAX}s"
    else:
        rec["status"] = "ok"
        notes = []
        if corr < 0.9:
            notes.append(f"moderate coherence (corr {corr:.2f})")
        if az_gap > 180:
            notes.append(f"azimuthal gap {az_gap:.0f}°: one-sided coverage, directional (narrow ray cone)")
        if v < vpvs.SQRT2:
            notes.append("Vp/Vs < √2: negative Poisson (physical: dry cracks / quartz-rich rock)")
        rec["note"] = "; ".join(notes)
    return rec


def run_all(out_csv=RESULTS_CSV, force=False, verbose=True):
    """Vp/Vs for every KG.HDB repeater family → uf_vpvs_results.csv."""
    import pandas as pd
    warnings.filterwarnings("ignore")
    rep, ev_of, meta_idx = families()
    rows = []
    for fam in rep.sort_values("n", ascending=False).cluster.astype(int):
        rec = run_family(fam, ev_of[fam], meta_idx, force=force)
        rows.append(rec)
        if verbose:
            v = rec["vpvs"]
            print(f"  family {fam:>4d}: {rec['status']:<16s} "
                  + (f"Vp/Vs={v:.3f}±{rec['mad']:.3f} corr={rec['corr']:.2f} "
                     f"({rec['n_pairs']}pairs, {rec['dep_med']:.1f}km)" if v == v else rec["note"]),
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    n_ok = int((df.status == "ok").sum())
    if verbose:
        print(f"\n[uf_vpvs] {n_ok}/{len(df)} families with Vp/Vs; wrote {out_csv}")
    return df


if __name__ == "__main__":
    run_all(force="--force" in sys.argv)
