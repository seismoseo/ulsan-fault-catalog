#!/usr/bin/env python
"""Associate one picker's month of picks with PyOcto (kim2011 1-D) and/or HARPA.

USAGE (env: base):
    python run_association.py --picker original --assoc pyocto
    python run_association.py --picker original --assoc harpa [--seed 0]
    python run_association.py --all                         # every picks_*.parquet x both associators

Inputs : picks/picks_<picker>.parquet   (net, sta, phase, time, prob)
         cache/stations_2014_09.csv
Outputs: catalogs/catalog_<picker>_<assoc>.csv        (one row per event: time, lat, lon, depth, n_p, n_s, ...)
         catalogs/assign_<picker>_<assoc>.parquet     (pick -> event assignments)
         catalogs/catalog_<picker>_<assoc>.json       (config + runtime record)

Fairness: identical station set; identical quality gate at the associator level (>=4 picks with >=2P & >=2S,
the Buan HARPA baseline); kim2011 velocities for PyOcto; HARPA homogeneous vel P6.0/S3.5 (Buan-proven; its
optimal-transport loss tolerates the 1-D-vs-homogeneous difference via max_time_residual).
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GYEONGJU = (35.856, 129.224)
ZLIM = (0.0, 30.0)
KIM2011 = pd.DataFrame({          # depth-to-top (km), vp, vs — kim2011 (P) + kim2011 (S)
    "depth": [0.00, 7.29, 20.70, 31.30],
    "vp":    [5.63, 6.17, 6.58, 7.77],
    "vs":    [3.40, 3.60, 3.70, 4.45],
})
GATE = dict(n_picks=4, n_p=2, n_s=2, n_ps=1)   # n_ps=1 => stations-with-both-P&S>=1 (Buan value; omitting it defaults to 3 and silently raises the gate)


def load_inputs(picker, stag, ptag):
    S = pd.read_csv(os.path.join(HERE, "cache", f"stations_{stag}.csv"))
    S = S[S.coverage >= 0.8].copy()                    # HH/EL/HG all allowed (one band per station)
    P = pd.read_parquet(os.path.join(HERE, "picks", f"picks_{picker}_{ptag}.parquet"))
    P["time"] = pd.to_datetime(P.time, format="ISO8601", utc=True).dt.tz_localize(None)
    # KMA code renames (BUS->BUS2 etc): 2014 miniSEED headers carry the OLD code while the
    # station table (StationXML epochs) uses the successor -> normalize picks to table codes.
    known = set(S.sta)
    alias = {c: c + "2" for c in P.sta.unique() if c not in known and (c + "2") in known}
    if alias:
        print(f"  station-code aliases applied: {alias}")
        P["sta"] = P.sta.replace(alias)
    P = P.sort_values("time").reset_index(drop=True)
    return S, P


def run_pyocto(picker, stag, ptag):
    if os.path.exists(os.path.join(HERE, "catalogs", f"catalog_{picker}_{ptag}_pyocto.csv")):
        print(f"[{picker} x pyocto | {ptag}] exists, skip"); return
    import pyocto
    S, P = load_inputs(picker, stag, ptag)
    stations = pd.DataFrame({"id": S.net + "." + S.sta, "latitude": S.lat,
                             "longitude": S.lon, "elevation": S.elev})
    vpath = os.path.join(HERE, "cache", "kim2011_pyocto.dat")
    if not os.path.exists(vpath):
        pyocto.VelocityModel1D.create_model(KIM2011, 1.0, 130.0, 35.0, vpath)
    vm = pyocto.VelocityModel1D(vpath, tolerance=1.0)          # Buan/Ulsan value (was 2.0)
    assoc = pyocto.OctoAssociator.from_area(
        lat=(GYEONGJU[0] - 1.0, GYEONGJU[0] + 1.0), lon=(GYEONGJU[1] - 1.2, GYEONGJU[1] + 1.2),
        zlim=ZLIM, time_before=300.0, velocity_model=vm,
        n_picks=GATE["n_picks"], n_p_picks=GATE["n_p"], n_s_picks=GATE["n_s"],
        n_p_and_s_picks=GATE["n_ps"], pick_match_tolerance=1.5)   # Ulsan/Buan config: keeps the gate at 4/2/2
    picks = pd.DataFrame({"station": P.net + "." + P.sta, "phase": P.phase.str.upper(),
                          "time": P.time.astype("int64") / 1e9,        # pyocto: float UNIX seconds
                          "probability": P.prob})
    st_x = assoc.transform_stations(stations)
    t0 = time.time()
    events, assignments = assoc.associate(picks, st_x)
    rt = time.time() - t0
    if len(events):
        events = assoc.transform_events(events)
        cnt = assignments.groupby("event_idx").pick_idx.count().rename("n_picks")
        npn = assignments[assignments.phase == "P"].groupby("event_idx").size().rename("n_p")
        nsn = assignments[assignments.phase == "S"].groupby("event_idx").size().rename("n_s")
        events = events.join(cnt, on="idx").join(npn, on="idx").join(nsn, on="idx").fillna(0)
        cat = pd.DataFrame({"time": pd.to_datetime(events.time, unit="s"),
                            "lat": events.latitude, "lon": events.longitude, "depth": events.depth,
                            "n_picks": events.n_picks.astype(int),
                            "n_p": events.n_p.astype(int), "n_s": events.n_s.astype(int)})
    else:
        cat = pd.DataFrame(columns=["time", "lat", "lon", "depth", "n_picks", "n_p", "n_s"])
        assignments = pd.DataFrame()
    _save(picker, ptag, "pyocto", cat, assignments,
          dict(velocity="kim2011 1-D", gate=GATE, zlim=ZLIM, time_before=300.0, runtime_s=round(rt, 1)))


def run_harpa(picker, stag, ptag, seed=0):
    atag = "harpa" + ("" if seed == 0 else f"_seed{seed}")
    if os.path.exists(os.path.join(HERE, "catalogs", f"catalog_{picker}_{ptag}_{atag}.csv")):
        print(f"[{picker} x {atag} | {ptag}] exists, skip"); return
    sys.path.insert(0, "/home/msseo/works/phase_association")
    import torch
    from pyproj import Proj
    from harpa import association
    S, P = load_inputs(picker, stag, ptag)
    S = S.reset_index(drop=True)                 # HARPA indexes station_df POSITIONALLY (needs RangeIndex)
    proj = Proj(f"+proj=sterea +lon_0={GYEONGJU[1]} +lat_0={GYEONGJU[0]} +units=km")
    sx, sy = proj(S.lon.values, S.lat.values)
    station_df = pd.DataFrame({"id": S.net + "." + S.sta + ".", "x(km)": sx, "y(km)": sy,
                               "z(km)": -S.elev.values / 1000.0})
    pick_df = pd.DataFrame({"id": P.net + "." + P.sta + ".", "timestamp": P.time,
                            "prob": P.prob, "type": P.phase.str.lower()})
    config = {
        "x(km)": [float(sx.min()) - 15, float(sx.max()) + 15],
        "y(km)": [float(sy.min()) - 15, float(sy.max()) + 15],
        "z(km)": list(ZLIM),
        "vel": {"P": 6.0, "S": 3.5},                        # Buan-proven homogeneous velocity
        "min_peak_pre_event": GATE["n_picks"],
        "min_peak_pre_event_p": GATE["n_p"], "min_peak_pre_event_s": GATE["n_s"],
        "P_phase": True, "S_phase": True,
        "ncpu": 16,
    }
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    pick_df_out, catalog_df = association(pick_df, station_df, config, verbose=0)
    rt = time.time() - t0
    if len(catalog_df):
        lon, lat = proj(catalog_df["x(km)"].values, catalog_df["y(km)"].values, inverse=True)
        base = pd.DataFrame({"time": pd.to_datetime([str(t) for t in catalog_df["time"]]), "lat": lat, "lon": lon,
                             "depth": catalog_df["z(km)"].values,
                             "event_index": pd.to_numeric(catalog_df["event_index"]).astype("int64")})
        ass = pick_df_out.copy()
        ass["event_index"] = pd.to_numeric(ass.event_index, errors="coerce").fillna(-1).astype("int64")
        ass = ass[ass.event_index >= 0]
        ass["type"] = ass.type.str.upper()                 # HARPA returns P/S uppercase
        npn = ass[ass.type == "P"].groupby("event_index").size().rename("n_p")
        nsn = ass[ass.type == "S"].groupby("event_index").size().rename("n_s")
        cat = base.join(npn, on="event_index").join(nsn, on="event_index").fillna(0)
        cat["n_p"] = cat.n_p.astype(int); cat["n_s"] = cat.n_s.astype(int)
        cat["n_picks"] = cat.n_p + cat.n_s
        cat = cat.drop(columns=["event_index"])
        assignments = ass
    else:
        cat = pd.DataFrame(columns=["time", "lat", "lon", "depth", "n_picks", "n_p", "n_s"])
        assignments = pd.DataFrame()
    _save(picker, ptag, atag, cat, assignments,
          dict(config={k: (v if not isinstance(v, dict) else v) for k, v in config.items()},
               seed=seed, runtime_s=round(rt, 1)))


def _save(picker, tag, assoc, cat, assignments, meta):
    os.makedirs(os.path.join(HERE, "catalogs"), exist_ok=True)
    cp = os.path.join(HERE, "catalogs", f"catalog_{picker}_{tag}_{assoc}.csv")
    cat.sort_values("time").to_csv(cp, index=False)
    if len(assignments):
        assignments.to_parquet(os.path.join(HERE, "catalogs", f"assign_{picker}_{tag}_{assoc}.parquet"), index=False)
    meta["n_events"] = int(len(cat))
    with open(os.path.join(HERE, "catalogs", f"catalog_{picker}_{tag}_{assoc}.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    print(f"[{picker} x {assoc} | {tag}] {len(cat)} events -> {cp}  ({meta.get('runtime_s','?')}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picker"); ap.add_argument("--assoc", choices=["pyocto", "harpa"])
    ap.add_argument("--month", default="2021-09", help="YYYY-MM")
    ap.add_argument("--suffix", default="", help="picks/output tag suffix, e.g. _native (stations use base tag)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    Y, MO = int(a.month[:4]), int(a.month[5:7]); stag = f"{Y}_{MO:02d}"; ptag = stag + a.suffix
    if a.all:
        import glob as g
        suf = f"_{ptag}.parquet"
        pickers = sorted(os.path.basename(f)[6:-len(suf)] for f in
                         g.glob(os.path.join(HERE, "picks", f"picks_*{suf}")))
        for p in pickers:
            run_pyocto(p, stag, ptag); run_harpa(p, stag, ptag, seed=0)
    else:
        if a.assoc == "pyocto": run_pyocto(a.picker, stag, ptag)
        else: run_harpa(a.picker, stag, ptag, seed=a.seed)


if __name__ == "__main__":
    main()
