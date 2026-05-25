"""
Core pipeline stage functions for the KS_KG seismicity catalog.

Single source of truth, ported faithfully from the per-year notebooks but
parameterized by picker model + year and writing into the models/<model>/ tree.
Importable from notebooks; driven by the thin CLIs (detection.py, ...).

Stages:  detection -> association -> PHS -> HYPOINVERSE
"""
import os
import sys
import glob
import concurrent.futures

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ============================================================ detection (1)
def discover_stations(base_dir, year):
    """Station dirs under base_dir that hold data for `year`."""
    out = []
    for d in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, d)
        if os.path.isdir(p) and glob.glob(f"{p}/*/*.{year}.*"):
            out.append(d)
    return out


def preprocess_station(code, jday, base_dir):
    """Load + preprocess one station-day. Runs in a worker process.

    Ported verbatim from 01.Run_multi-station_detection: interpolate to 100 Hz,
    merge, pad-trim, demean/taper, bandpass, trim back. Returns a Stream or None.
    """
    from obspy import read
    import config as cfg
    try:
        stream = read(f"{base_dir}/{code}/*/*.{jday}")
        if not stream:
            return None
        stream.interpolate(sampling_rate=cfg.SAMPLING_RATE)
        stream.merge(**cfg.MERGE)
        t_start = max(tr.stats.starttime for tr in stream)
        t_end = min(tr.stats.endtime for tr in stream)
        stream.trim(starttime=t_start - cfg.PAD_TIME, endtime=t_end + cfg.PAD_TIME,
                    pad=True, fill_value=0)
        stream.detrend("demean")
        stream.taper(max_percentage=None, max_length=cfg.PAD_TIME)
        stream_filt = stream.filter("bandpass", **cfg.BANDPASS).detrend("demean")
        stream_filt.trim(starttime=t_start, endtime=t_end)
        return stream_filt
    except Exception as e:
        print(f"  ! preprocess failed [{code}] {jday}: {e}")
        return None


def canonical_station(trace_id):
    """Normalize a SeisBench pick trace_id to canonical 'NET.STA'.

    Handles the inconsistent forms seen across years ('KG.BBK.', 'KG.BBK.00',
    'BBK') so downstream association is uniform.
    """
    parts = str(trace_id).split(".")
    if len(parts) >= 2 and parts[1]:
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


def detect_day(pn_model, stations, jday, base_dir, workers=None):
    """Preprocess all stations in parallel, then a single classify() for the day."""
    from obspy import Stream
    daily = Stream()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(preprocess_station, c, jday, base_dir) for c in stations]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res:
                daily += res
    if not daily:
        return None
    outputs = pn_model.classify(daily, P_threshold=config.P_THRESHOLD, S_threshold=config.S_THRESHOLD)
    rows = [{"station": canonical_station(p.trace_id), "phase": p.phase,
             "peak_time": str(p.peak_time), "probability": p.peak_value}
            for p in outputs.picks]
    return pd.DataFrame(rows, columns=["station", "phase", "peak_time", "probability"])


def run_detection_year(model, year, days=None, stations=None, skip_existing=True,
                       device=None, workers=None, force=False):
    """Run PhaseNet(model) detection for a year; write daily picks CSVs."""
    import torch
    import seisbench.models as sbm

    config.assert_writable(model, force)
    base_dir = config.CONTINUOUS
    out_dir = config.picks_dir(model, year)
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[detection] model={model} year={year} device={device}")

    pn_model = sbm.PhaseNet.from_pretrained(model)
    pn_model.to(torch.device(device))

    if stations is None:
        stations = discover_stations(base_dir, year)
    print(f"[detection] {len(stations)} stations with {year} data")

    if days is None:
        days = range(1, config.days_in_year(year) + 1)

    n_written = 0
    for day in days:
        jday = f"{year}.{day:03d}"
        out = os.path.join(out_dir, f"picks_{jday}.csv")
        if skip_existing and os.path.exists(out):
            continue
        df = detect_day(pn_model, stations, jday, base_dir, workers=workers)
        if df is None or df.empty:
            continue
        df.to_csv(out, index=False)
        n_written += 1
        print(f"  {jday}: {len(df)} picks -> {os.path.relpath(out, config.MODELS)}")
    print(f"[detection] done: {n_written} daily file(s) written into {os.path.relpath(out_dir, config.MODELS)}")
    return out_dir


# ============================================================ association (2)
def load_picks(model, year):
    files = sorted(glob.glob(os.path.join(config.picks_dir(model, year), f"picks_{year}.*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No picks for model={model} year={year} in {config.picks_dir(model, year)} "
            f"(run detection first).")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def build_station_table(picks_df):
    """Station table for used stations: coords from station_update.dat (by Code),
    Network taken from the picks themselves (replaces the old hardcoded KS/KG split)."""
    parts = picks_df["station"].str.split(".", expand=True)
    nets, codes = parts[0], parts[1]
    code_net = pd.Series(nets.values, index=codes.values)
    code_net = code_net[~code_net.index.duplicated(keep="first")].to_dict()

    used = sorted(set(codes.dropna()))
    allsta = pd.read_csv(config.STATION_UPDATE, sep=r"\s+",
                         names=["Code", "Latitude", "Longitude", "Elevation"])
    st = allsta[allsta["Code"].isin(used)].reset_index(drop=True)
    st["Network"] = st["Code"].map(code_net)
    missing = st["Network"].isna()
    if missing.any():
        print(f"  ! {int(missing.sum())} station(s) missing a network; dropping: "
              f"{st.loc[missing, 'Code'].tolist()}")
        st = st[~missing].reset_index(drop=True)
    return st


def _station_pyocto(st):
    s = st.copy()
    s["id"] = s["Network"] + "." + s["Code"] + "."
    s = s.rename(columns={"Longitude": "longitude", "Latitude": "latitude", "Elevation": "elevation"})
    return s[["id", "longitude", "latitude", "elevation"]].copy()


def run_association_year(model, year, force=False):
    """Associate picks into events with PyOcto; write events + assignments + station table."""
    import datetime as dt
    import tempfile
    import pyocto

    config.assert_writable(model, force)
    picks_df = load_picks(model, year).copy()
    parts = picks_df["station"].str.split(".", expand=True)
    picks_df["net"], picks_df["code"] = parts[0], parts[1]

    st = build_station_table(picks_df)
    station_pyocto = _station_pyocto(st)

    # side-output: stations_<year>.csv into the model tree (not the shared dir)
    os.makedirs(config.station_table_dir(model), exist_ok=True)
    fs = station_pyocto.copy()
    fs[["Network", "Code"]] = fs["id"].str.split(".", expand=True)[[0, 1]]
    fs[["Network", "Code", "latitude", "longitude", "elevation"]] \
        .rename(columns=str.capitalize) \
        .to_csv(config.stations_year_csv(model, year), index=False)

    # picks for PyOcto: station id must match station 'id' ("NET.CODE.")
    pk = picks_df.rename(columns={"peak_time": "time"}).copy()
    pk["time"] = pd.to_datetime(pk["time"]).dt.tz_localize(None)
    pk["station"] = pk["net"] + "." + pk["code"] + "."
    picks_pyocto = pk[["station", "phase", "time"]].copy()

    layers = pd.read_csv(config.VELOCITY_CSV)
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, "vel_model")
        pyocto.VelocityModel1D.create_model(layers, 1.0, 100, 100, mp)
        velocity_model = pyocto.VelocityModel1D(mp, tolerance=1.0)
        associator = pyocto.OctoAssociator.from_area(velocity_model=velocity_model, **config.REGION)
        associator.transform_stations(station_pyocto)
        picks_pyocto["time"] = picks_pyocto["time"].apply(lambda x: x.timestamp())
        events, assignments = associator.associate(picks_pyocto, station_pyocto)
        associator.transform_events(events)

    if len(events):
        events["time"] = events["time"].apply(dt.datetime.fromtimestamp, tz=dt.timezone.utc)
    os.makedirs(config.pyocto_dir(model), exist_ok=True)
    events.to_csv(config.pyocto_events(model, year), index=False)
    assignments.to_csv(config.pyocto_assign(model, year), index=False)
    n_pk = int(events["picks"].sum()) if len(events) else 0
    print(f"[association] {model} {year}: events={len(events)} picks={n_pk} "
          f"-> {os.path.relpath(config.pyocto_events(model, year), config.MODELS)}")
    return events, assignments


# ============================================================ PHS file (3)
def _deg2min(angle):
    return int(angle), int(100 * 60 * (angle - int(angle)))


def write_phs(model, year, force=False):
    """Write the HYPO71-format .phs file from PyOcto events+assignments.

    Ported verbatim (fixed-width layout) from 04.Make_input_PHS_file_for_HypoInv.
    """
    from obspy import UTCDateTime as utc
    from datetime import datetime, timezone

    config.assert_writable(model, force)
    ev = pd.read_csv(config.pyocto_events(model, year))
    pk = pd.read_csv(config.pyocto_assign(model, year))
    os.makedirs(config.phs_dir(model), exist_ok=True)
    out = config.phs_file(model, year)

    with open(out, "w") as f:
        idn = 0
        for i in range(len(ev)):
            eid = ev["idx"][i]
            ot = utc(str(ev["time"][i]).split("+")[0])
            cat = f"{ot.year}{ot.month:02d}{ot.day:02d}{ot.hour:02d}{ot.minute:02d}{ot.second:02d}"
            ms = str(int(ot.microsecond / 1000)).zfill(2)
            la_d, la_m = _deg2min(ev["latitude"][i])
            lo_d, lo_m = _deg2min(ev["longitude"][i])
            f.write(f"{cat}{ms}{la_d}N{str(la_m).zfill(4)}{lo_d}E{str(lo_m).zfill(4)}\n")

            ept = pk[pk["event_idx"] == eid].reset_index(drop=True)
            for j in range(len(ept)):
                pt = utc(datetime.fromtimestamp(ept["time"][j], tz=timezone.utc))
                net = ept["station"][j].split(".")[0]
                sta = ept["station"][j].split(".")[1]
                phase = ept["phase"][j]
                if phase == "P":
                    f.write(sta.ljust(5)); f.write(net.ljust(4))
                    f.write(config.PHASE_CHANNELS["P"][:3].ljust(4))
                    f.write("IP".ljust(3)); f.write("0")
                    f.write(str(pt.year)); f.write(str(pt.month).zfill(2)); f.write(str(pt.day).zfill(2))
                    f.write(str(pt.hour).zfill(2)); f.write(str(pt.minute).zfill(2).ljust(3))
                    f.write(str(pt.second).zfill(2)); f.write(str(pt.microsecond).zfill(6)[:2])
                    f.write("\n")
                elif phase == "S":
                    f.write(sta.ljust(5)); f.write(net.ljust(4))
                    f.write(config.PHASE_CHANNELS["S"].ljust(4)); f.write("    ")
                    f.write(str(pt.year)); f.write(str(pt.month).zfill(2)); f.write(str(pt.day).zfill(2))
                    f.write(str(pt.hour).zfill(2)); f.write(str(pt.minute).zfill(2).ljust(15))
                    f.write(str(pt.second).zfill(2)); f.write(str(pt.microsecond).zfill(6)[:2])
                    f.write("ES".ljust(3)); f.write("1"); f.write("\n")

            f.write(" " * 66 + "20" + f"{idn}".zfill(4) + "\n")
            idn += 1

    print(f"[phs] {model} {year}: {len(ev)} events -> {os.path.relpath(out, config.MODELS)}")
    return out


# ============================================================ HYPOINVERSE (4)
# Control template, faithfully reproduced from KS_KG/HypoInv/UF<year>.sh.
# __REGION__ -> 'UF<year>', __MODEL__ -> velocity model (kim1983 / kim2011).
HYP_TEMPLATE = """
REP T T
CON 50
MIN 4
ZTR 10 F
DIS 4 25 1 3
RMS 4 .12 2 4

* OUTPUT FORMAT
ERF T
TOP F
LST 2 0 1
KPR 3
H71 4 1 3

* STATION DATA
STA 'STA/__REGION___hyp.sta'

* CRUSTAL MODEL
CRH 1 '__MODEL__/__MODEL___p.crh'
CRH 2 '__MODEL__/__MODEL___s.crh'
SAL 1 2

* PHASE FILE
PHS 'PHS/__REGION__.phs'
FIL

PRT '__MODEL__/__REGION__.prt'
SUM '__MODEL__/__REGION__.sum'
ARC '__MODEL__/__REGION__.arc'

LOC

STO
"""


def run_hypoinverse_year(model, year, velmodel=None, force=False):
    """Run hyp1.40 for one year/velocity-model; outputs land in models/<model>/HypoInv/<velmodel>/."""
    import shutil
    import subprocess

    config.assert_writable(model, force)
    velmodel = velmodel or config.DEFAULT_VELMODEL
    hd = config.hyp_dir(model)
    region = f"UF{year}"

    phs = config.phs_file(model, year)
    if not os.path.exists(phs):
        raise FileNotFoundError(f"PHS missing: {phs} (run make_phs first)")
    if shutil.which("hyp1.40") is None:
        raise RuntimeError("hyp1.40 not found on PATH")
    os.makedirs(config.velmodel_dir(model, velmodel), exist_ok=True)

    control = HYP_TEMPLATE.replace("__REGION__", region).replace("__MODEL__", velmodel)
    print(f"[hypoinverse] {model} {year} velmodel={velmodel} (cwd={os.path.relpath(hd, config.MODELS)})")
    proc = subprocess.run(["hyp1.40"], input=control, text=True, cwd=hd, capture_output=True)

    sumf = os.path.join(config.velmodel_dir(model, velmodel), f"{region}.sum")
    if not os.path.exists(sumf):
        print("---- hyp1.40 stdout (tail) ----\n", proc.stdout[-2000:])
        print("---- hyp1.40 stderr (tail) ----\n", proc.stderr[-2000:])
        raise RuntimeError(f"HYPOINVERSE produced no summary file: {sumf}")
    n = sum(1 for _ in open(sumf))
    print(f"[hypoinverse] {n} located events -> {os.path.relpath(sumf, config.MODELS)}")
    return sumf
