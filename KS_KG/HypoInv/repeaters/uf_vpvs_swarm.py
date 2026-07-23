"""In-situ Vp/Vs of the 2021 Ulsan-fault swarm using the FULL spatial swarm (not just the CC>=0.9
repeater subset). Defining the swarm spatially (DBSCAN, à la NND declustering) keeps the more-separated
members, which carry the LONG-baseline event pairs — and therefore the large lever arm τ (Liu 2023) that
the tight repeater family lacks.

Fast: each SAC is read ONCE (trace cache); per band we precompute windowed/filtered/normalised P (native
vertical) and S (best horizontal) arrays per (event, station), then the pair loop is pure in-memory
cross-correlation with parabolic sub-sample refinement. δt cached to disk.

Reuses kma_absolute_location.vpvs for the robust estimator + bootstrap, and uf_waveform_similarity for the
event metadata / picks / station geometry.
"""
from __future__ import annotations
import os, sys, glob, math, itertools, pickle, warnings
warnings.filterwarnings("ignore")
HYPO = "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/HypoInv"
sys.path.insert(0, os.path.join(HYPO, "repeaters"))   # sibling uf_vpvs (not uflib)
sys.path.insert(0, "/home/msseo/works/16.kma_absolute_location")
import numpy as np
from obspy import read
from obspy.signal.cross_correlation import correlate
from uflib import uf_waveform_similarity as wf
from kma_absolute_location import vpvs

SR = 100.0
SWARM_CEN = (35.807, 129.436)          # 2021 swarm centroid (35.81N, 129.44E, ~14.5 km)
RADIUS_KM = 2.5                        # spatial selection radius around the centroid
YEARS = ("2021", "2022")              # the sequence runs Jun-2021 → 2022
DBSCAN_EPS, DBSCAN_MIN = 1.2, 6        # spatial cluster (km) to drop stragglers
CCXC_MIN, MAXLAG_XC, MIN_ST = 0.75, 0.6, 5
P_WIN, S_WIN = (0.3, 0.8), (0.3, 1.4)
MAX_KM = 60.0                          # nearby-station radius
TAU_MIN_S = 0.05                       # Liu (2023) lever-arm floor for the "robust τ" subset
CACHE = wf.CACHE_DIR


# --------------------------------------------------------------- swarm membership
def swarm_events():
    """Full 2021 swarm = waveform-available events within RADIUS_KM of the centroid in YEARS, kept by a
    spatial DBSCAN (drops isolated stragglers). Returns (event_list, meta_indexed_by_event)."""
    from sklearn.cluster import DBSCAN
    events = wf.list_events()
    meta = wf.load_event_meta(events).set_index("event")
    clat = SWARM_CEN[0]
    cand = [e for e in events if e[:4] in YEARS and e in meta.index
            and math.hypot((meta.loc[e].lat - SWARM_CEN[0]) * 111.195,
                           (meta.loc[e].lon - SWARM_CEN[1]) * 111.195 * math.cos(math.radians(clat))) < RADIUS_KM]
    if len(cand) < DBSCAN_MIN:
        return cand, meta
    X = np.array([[(meta.loc[e].lon - SWARM_CEN[1]) * 111.195 * math.cos(math.radians(clat)),
                   (meta.loc[e].lat - SWARM_CEN[0]) * 111.195, float(meta.loc[e].depth)] for e in cand])
    lab = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN).fit(X).labels_
    main = np.bincount(lab[lab >= 0]).argmax() if (lab >= 0).any() else -1
    return [e for e, l in zip(cand, lab) if l == main], meta


# --------------------------------------------------------------- trace cache + δt
def _orient(cha):
    return cha[-1]


def load_traces(evs):
    """Read each needed SAC once. TR[(ev,sta)] = {"V":(tr,band), "H":[(tr,band),...], "P":t, "S":t}.
    Native vertical for P; both horizontals kept for S."""
    TR = {}
    for ev in evs:
        for f in glob.glob(os.path.join(wf.WF_ROOT, ev, f"{ev}.*.sac")):
            parts = os.path.basename(f).split(".")            # ev . NET . STA . CHA . sac
            if len(parts) < 5:
                continue
            net, sta, cha = parts[1], parts[2], parts[3]
            o = _orient(cha)
            d = TR.setdefault((ev, f"{net}.{sta}"), {"H": []})
            tr = read(f)[0]
            if o == "Z":
                d["V"] = tr
            elif o in ("E", "N"):
                d["H"].append(tr)
        # picks (P, S) from picks.csv via uf_waveform_similarity
    for (ev, sta) in list(TR):
        TR[(ev, sta)]["P"] = wf.pick_time(ev, sta, "P")
        TR[(ev, sta)]["S"] = wf.pick_time(ev, sta, "S")
    return TR


def _win(tr, t, pre, post, band):
    if tr is None or t is None:
        return None
    x = tr.copy().detrend("demean").taper(0.05).filter("bandpass", freqmin=band[0], freqmax=band[1],
                                                        corners=4, zerophase=True).slice(t - pre, t + post)
    if x.stats.npts < int((pre + post) * SR * 0.8):
        return None
    a = x.data.astype(float); a -= a.mean(); n = np.linalg.norm(a)
    return a / n if n > 0 else None


def _xc(xi, xj):
    if xi is None or xj is None or len(xi) != len(xj):
        return None
    cc = correlate(xi, xj, int(MAXLAG_XC * SR)); k = int(np.argmax(cc)); val = float(cc[k])
    if 0 < k < len(cc) - 1:
        a, b, c = cc[k - 1], cc[k], cc[k + 1]; den = a - 2 * b + c
        dsub = 0.5 * (a - c) / den if den != 0 else 0.0
    else:
        dsub = 0.0
    return (k - len(cc) // 2 + dsub) / SR, val


def build_dt(evs, band, force=False):
    """Per-pair, per-station P & S differential times for the full swarm at a bandpass. Disk-cached.
    Returns (P, S) dicts: {(ev_i,ev_j): {station: (δt, cc)}}."""
    cf = os.path.join(CACHE, f"vpvs_swarm2021_b{band[0]}-{band[1]}_n{len(evs)}.pkl")
    if os.path.exists(cf) and not force:
        with open(cf, "rb") as fh:
            return pickle.load(fh)
    TR = load_traces(evs)
    # precompute windows once per (ev,sta)
    WP, WS = {}, {}
    for (ev, sta), d in TR.items():
        if "V" in d and d.get("P") is not None:
            w = _win(d["V"], d["P"], *P_WIN, band)
            if w is not None:
                WP[(ev, sta)] = (w, d["P"])
        if d.get("S") is not None and d["H"]:
            best = None
            for tr in d["H"]:
                w = _win(tr, d["S"], *S_WIN, band)
                if w is not None:
                    best = (w, d["S"]); break          # first available horizontal
            if best:
                WS[(ev, sta)] = best
    P, S = {}, {}
    for ei, ej in itertools.combinations(evs, 2):
        kp, ks = {}, {}
        stas = {s for (e, s) in WP if e == ei} | {s for (e, s) in WS if e == ei}
        for sta in stas:
            if (ei, sta) in WP and (ej, sta) in WP:
                r = _xc(WP[(ei, sta)][0], WP[(ej, sta)][0])
                if r and r[1] >= CCXC_MIN:
                    kp[sta] = ((WP[(ei, sta)][1] - WP[(ej, sta)][1]) + r[0], r[1])
            if (ei, sta) in WS and (ej, sta) in WS:
                r = _xc(WS[(ei, sta)][0], WS[(ej, sta)][0])
                if r and r[1] >= CCXC_MIN:
                    ks[sta] = ((WS[(ei, sta)][1] - WS[(ej, sta)][1]) + r[0], r[1])
        if kp:
            P[(ei, ej)] = kp
        if ks:
            S[(ei, ej)] = ks
    with open(cf, "wb") as fh:
        pickle.dump((P, S), fh)
    return P, S


# --------------------------------------------------------------- helpers
def pair_tau(P, S):
    """Per-pair δtP range (s) over the common P&S stations — Liu's lever arm, one per kept pair."""
    out = {}
    for k in set(P) & set(S):
        cm = [s for s in P[k] if s in S[k] and P[k][s][1] >= CCXC_MIN and S[k][s][1] >= CCXC_MIN]
        if len(cm) >= MIN_ST:
            dp = np.array([P[k][s][0] for s in cm]); out[k] = dp.max() - dp.min()
    return out


RMS_MAX_PAIR = 0.005     # Liu (2023) per-pair linearity gate: residual RMS ceiling (s)


def per_pair_gate(P, S, rms_max=RMS_MAX_PAIR):
    """Liu (2023) linearity step: keep an event pair only if its OWN demeaned δtP-δtS form a good line —
    per-pair robust slope in [0.5, 3] and residual RMS < rms_max. This removes the decohered far pairs
    (waveforms too dissimilar / shared-ray-path assumption broken) at the PAIR level, so the pooled fit is
    a single clean measurement. Returns (Pk, Sk) restricted to the surviving pairs."""
    Pk, Sk = {}, {}
    for k in set(P) & set(S):
        cm = [s for s in P[k] if s in S[k] and P[k][s][1] >= CCXC_MIN and S[k][s][1] >= CCXC_MIN]
        if len(cm) < MIN_ST:
            continue
        dp = np.array([P[k][s][0] for s in cm]); ds = np.array([S[k][s][0] for s in cm])
        dp = dp - dp.mean(); ds = ds - ds.mean()
        if np.sum(dp * dp) < 1e-12:
            continue
        sp = vpvs.irls_slope(dp, ds, np.ones_like(dp))
        rms = float(np.sqrt(np.mean((ds - sp * dp) ** 2)))
        if 0.5 <= sp <= 3.0 and rms < rms_max:
            Pk[k] = P[k]; Sk[k] = S[k]
    return Pk, Sk


def tls_slope(x, y):
    x = x - x.mean(); y = y - y.mean(); vp, vs, c = np.var(x), np.var(y), np.mean(x * y)
    return float((vs - vp + np.sqrt((vs - vp) ** 2 + 4 * c ** 2)) / (2 * c))


def az_gap(evs, meta, stations):
    sc = {r.station: (r.lat, r.lon) for r in wf.used_stations(evs).itertuples()}
    import uf_vpvs as uv
    return uv._az_gap(meta.loc[evs].lat.mean(), meta.loc[evs].lon.mean(),
                      [sc[s] for s in stations if s in sc])


if __name__ == "__main__":
    evs, meta = swarm_events()
    print(f"full 2021 swarm: {len(evs)} events, {sorted({e[:6] for e in evs})}")
    for band in [(5, 15), (3, 15)]:
        P, S = build_dt(evs, band, force="--force" in sys.argv)
        m = vpvs.build_measurements(P, S, None, cc_min=CCXC_MIN, min_stations=MIN_ST, dist_min_km=0.0)
        if m["n_pairs"] < 5:
            print(f"  {band}: too few pairs ({m['n_pairs']})"); continue
        v, mad, _ = vpvs.bootstrap_vpvs(m, n_boot=1000)
        inl, _ = vpvs.robust_inliers(m, v)
        corr = float(np.corrcoef(m["dtp"][inl], m["dts"][inl])[0, 1])
        tau = pair_tau(P, S); tv = np.array(list(tau.values()))
        print(f"  {band[0]}-{band[1]}Hz: Vp/Vs={v:.3f}±{mad:.3f} corr={corr:.3f} pairs={m['n_pairs']} obs={len(m['dtp'])} "
              f"nsta={len(m.get('stations',set()))} | τ med {np.median(tv)*1000:.0f}/max {tv.max()*1000:.0f}ms "
              f"(>50ms: {int((tv>TAU_MIN_S).sum())}/{len(tv)})")
    print("SWARMMODDONE")
