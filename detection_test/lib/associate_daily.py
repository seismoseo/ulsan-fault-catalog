#!/usr/bin/env python
"""Daily-chunked PyOcto association — required for the DENSE network.

On the 241-station array with a 0.2 pick threshold, a month is ~1.9M picks; PyOcto's whole-month octree
association becomes intractable (>>1 h, 12 GB). Associating one calendar day at a time (~60k picks/day, close
to the 2014 whole-month volume) keeps each solve to seconds and is physically equivalent: local events are
seconds long, so a +/-OVERLAP window around each day captures every event's picks, and we keep only events whose
ORIGIN falls inside the day (dedup). This is the association mode the full 2010-2025 run will use.

    python associate_daily.py --picker original --month 2021-09
    python associate_daily.py --picker original --suffix _native --month 2021-09

Output: catalogs/catalog_<picker>_<ptag>_pyocto.csv  (time,lat,lon,depth,n_picks,n_p,n_s)  + assign parquet + json
"""
import os, json, time, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import gj_config as C                          # ALL parameters live in gj_config.py (single disclosed source)
GYEONGJU = C.REGION_CENTER; ZLIM = C.ZLIM
KIM2011 = pd.DataFrame(C.KIM2011)
GATE = C.GATE                                  # fixed permissive gate for ALL epochs; false events removed
OVERLAP = pd.Timedelta(seconds=C.ASSOC_OVERLAP_S)   # DOWNSTREAM by physical location-QC (never tuned per-network)


def _assoc_day(day_ts, w, stations, vpath, gate, overlap_s):
    """Associate ONE day's pick window with PyOcto; return (catalog_df with _ei, assign_df) — day-local
    event_idx. Top-level + self-contained so it runs in a multiprocessing worker (each builds its own
    associator; from_area setup is cheap vs the per-day associate)."""
    import pyocto, pandas as pd
    day = pd.Timestamp(day_ts); d1 = day + pd.Timedelta(days=1)
    vm = pyocto.VelocityModel1D(vpath, tolerance=C.VEL_TOLERANCE)
    assoc = pyocto.OctoAssociator.from_area(
        lat=(GYEONGJU[0] - C.ASSOC_LAT_PAD, GYEONGJU[0] + C.ASSOC_LAT_PAD),
        lon=(GYEONGJU[1] - C.ASSOC_LON_PAD, GYEONGJU[1] + C.ASSOC_LON_PAD),
        zlim=ZLIM, time_before=C.TIME_BEFORE, velocity_model=vm,
        n_picks=gate["n_picks"], n_p_picks=gate["n_p"], n_s_picks=gate["n_s"],
        n_p_and_s_picks=gate["n_ps"], pick_match_tolerance=C.PICK_MATCH_TOLERANCE)
    st_x = assoc.transform_stations(stations)
    picks = pd.DataFrame({"station": w.net + "." + w.sta, "phase": w.phase.str.upper(),
                          "time": w.time.astype("int64") / 1e9, "probability": w.prob})
    ev, asg = assoc.associate(picks, st_x)
    if not len(ev):
        return None, None
    ev = assoc.transform_events(ev); ev["otime"] = pd.to_datetime(ev.time, unit="s")
    keep = ev[(ev.otime >= day) & (ev.otime < d1)]                      # dedup overlap -> origin in-day
    if not len(keep):
        return None, None
    cnt = asg.groupby("event_idx").pick_idx.count().rename("n_picks")
    npn = asg[asg.phase == "P"].groupby("event_idx").size().rename("n_p")
    nsn = asg[asg.phase == "S"].groupby("event_idx").size().rename("n_s")
    keep = keep.join(cnt, on="idx").join(npn, on="idx").join(nsn, on="idx").fillna(0)
    cat = pd.DataFrame({"time": keep.otime, "lat": keep.latitude, "lon": keep.longitude, "depth": keep.depth,
                        "n_picks": keep.n_picks.astype(int), "n_p": keep.n_p.astype(int),
                        "n_s": keep.n_s.astype(int), "_ei": keep.idx.astype(int)})
    asg_k = asg[asg.event_idx.isin(set(keep.idx))].copy()
    return cat, asg_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picker", required=True); ap.add_argument("--month", default="2021-09")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes over days (independent chunks)")
    ap.add_argument("--min-coverage", type=float, default=C.MIN_COVERAGE, help="min station local coverage (0 = use all)")
    a = ap.parse_args()
    import pyocto
    Y, MO = int(a.month[:4]), int(a.month[5:7]); stag = f"{Y}_{MO:02d}"; ptag = stag + a.suffix
    T0 = pd.Timestamp(f"{a.month}-01"); T1 = T0 + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)

    S = pd.read_csv(os.path.join(HERE, "cache", f"stations_{stag}.csv"))
    S = S[(S.coverage > 0) & (S.coverage >= a.min_coverage)].copy()
    P = pd.read_parquet(os.path.join(HERE, "picks", f"picks_{a.picker}_{ptag}.parquet"))
    P["time"] = pd.to_datetime(P.time, format="ISO8601", utc=True).dt.tz_localize(None)
    known = set(S.sta)
    alias = {c: c + "2" for c in P.sta.unique() if c not in known and (c + "2") in known}
    if alias: P["sta"] = P.sta.replace(alias)
    P = P.sort_values("time").reset_index(drop=True)
    print(f"[{a.picker}|{ptag}] {len(P)} picks, {P.sta.nunique()} stations -> daily PyOcto", flush=True)

    stations = pd.DataFrame({"id": S.net + "." + S.sta, "latitude": S.lat,
                             "longitude": S.lon, "elevation": S.elev})
    vpath = os.path.join(HERE, "cache", "kim2011_pyocto.dat")
    if not os.path.exists(vpath):
        pyocto.VelocityModel1D.create_model(KIM2011, 1.0, 130.0, 35.0, vpath)

    # slice picks per day (+/- OVERLAP); days are INDEPENDENT -> associate sequentially or across a process pool
    tasks = []
    day = T0
    while day < T1:
        d1 = day + pd.Timedelta(days=1)
        w = P[(P.time >= day - OVERLAP) & (P.time < d1 + OVERLAP)]
        if len(w) >= GATE["n_picks"]:
            tasks.append((day, w, stations, vpath, GATE, OVERLAP.seconds))
        day = d1
    t0 = time.time()
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            results = pool.starmap(_assoc_day, tasks)
    else:
        results = [_assoc_day(*t) for t in tasks]
    cats, assigns = [], []; off = 0                       # re-offset day-local event_idx into a global index
    for cat, asg in results:
        if cat is None:
            continue
        cat = cat.copy(); asg = asg.copy()
        cat["_ei"] += off; asg["event_idx"] += off
        off = int(asg.event_idx.max()) + 1
        cats.append(cat.rename(columns={"_ei": "event_idx"})); assigns.append(asg)   # keep the event<->assign link
    print(f"  {len(tasks)} days x {a.workers} workers -> {sum(len(c) for c in cats)} events  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    cat = pd.concat(cats, ignore_index=True).sort_values("time") if cats else \
        pd.DataFrame(columns=["time", "lat", "lon", "depth", "n_picks", "n_p", "n_s"])
    os.makedirs(os.path.join(HERE, "catalogs"), exist_ok=True)
    cp = os.path.join(HERE, "catalogs", f"catalog_{a.picker}_{ptag}_pyocto.csv")
    cat.to_csv(cp, index=False)
    if assigns:
        pd.concat(assigns, ignore_index=True).to_parquet(
            os.path.join(HERE, "catalogs", f"assign_{a.picker}_{ptag}_pyocto.parquet"), index=False)
    with open(os.path.join(HERE, "catalogs", f"catalog_{a.picker}_{ptag}_pyocto.json"), "w") as fh:
        json.dump(dict(velocity="kim2011 1-D", gate=GATE, mode="daily-chunked", overlap_s=OVERLAP.seconds,
                       n_picks_in=len(P), n_events=len(cat), runtime_s=round(time.time() - t0, 1)), fh, indent=1)
    print(f"[{a.picker}|{ptag}] {len(cat)} events -> {cp}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
