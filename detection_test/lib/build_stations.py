#!/usr/bin/env python
"""Build the station table + truth catalogs for a benchmark month (parameterized).

    python build_stations.py --month 2014-09     # KS/KG early network (NS not yet deployed)
    python build_stations.py --month 2021-09     # KS/KG + dense NS local array

Writes (relative to detection_test/):
    cache/stations_<YYYY_MM>.csv   net,sta,lat,lon,elev,t0,t1,accel_only,bands,dist_km,band,archive,days_local,coverage
    cache/truth_kma_<YYYY_MM>.csv
    cache/truth_uf_relocated_<YYYY_MM>.csv

Station sources (disclosed):
  * KS + KG: epochs/coords/bands from KS_KG_metadata_1.0.2.xml (seismometer-window rule, same as 2014);
             waveform archive = KS_KG/<STA>/<BAND>?.D/
  * NS      : coords/epochs from GHBSN_station_list_240220_modified_code.csv (dense local Gyeongju array,
             deployed 2017+); band = HH (200 Hz, decimated at pick time); archive = NS/<STA>/HH?.D/
Region = 100 km around the Gyeongju reference point (same as the KS/KG stage-1 work). A station is kept if its
metadata epoch overlaps the month; `coverage` = fraction of the month's days actually on local disk.
"""
import os, glob, argparse
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict

ROOT = "/home/msseo/works/02.Ulsan_Fault_detection"
HERE = f"{ROOT}/detection_test"
_META = f"{ROOT}/data/metadata"                 # single metadata home (by kind): stations/, responses/, velocity/, catalogs/
P_XML = f"{_META}/responses/master/KS_KG_metadata_1.0.2.xml"
ARCH_KSKG = f"{ROOT}/KS_KG"
ARCH_NS = f"{ROOT}/NS"
P_NS = f"{_META}/stations/ns/20231227/GHBSN_station_list_240220_modified_code.csv"
ARCH_GJ = f"{ROOT}/GJ"
P_GJ = f"{_META}/stations/gj/gj_temporary_station_list.csv"
P_KMA = "/home/msseo/works/16.kma_absolute_location/runs/kma_batch/results_final.csv"
P_UF = f"{ROOT}/analysis/local_magnitudes/catalog_ml_heo_ufonly_reloc.csv"
import gj_config as C                        # single disclosed source of all parameters
REGION_CENTER = C.REGION_CENTER; RMAX_KM = C.RMAX_KM
BANDS = C.BANDS              # allowed picking bands, in priority order (velocity HH/EL first, accel HG last)
NSX = "{http://www.fdsn.org/xml/station/1}"


def hav_km(a1, o1, a2, o2):
    x = (np.sin(np.radians(a2 - a1) / 2)**2 + np.cos(np.radians(a1)) * np.cos(np.radians(a2)) * np.sin(np.radians(o2 - o1) / 2)**2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(x))


def _ts(x):
    try: return pd.Timestamp(x).tz_localize(None)
    except Exception: return None


def load_kskg(path):
    """KS+KG stations with coords, epoch window (seismometer channels preferred), band set."""
    A = defaultdict(lambda: {"lat": None, "lon": None, "elev": 0.0, "seis": [], "all": [], "bands": set()}); net = None
    for ev, el in ET.iterparse(path, events=("start", "end")):
        t = el.tag.split("}")[-1]
        if ev == "start" and t == "Network": net = el.get("code")
        elif ev == "end" and t == "Station":
            d = A[(net, el.get("code"))]; la = el.findtext(NSX + "Latitude")
            if la:
                d["lat"] = float(la); d["lon"] = float(el.findtext(NSX + "Longitude"))
                e = el.findtext(NSX + "Elevation"); d["elev"] = float(e) if e else 0.0
            for c in el.findall(NSX + "Channel"):
                cc = c.get("code") or ""; d["bands"].add(cc[:2])
                iv = (c.get("startDate"), c.get("endDate")); d["all"].append(iv)
                if len(cc) > 1 and cc[1] in ("H", "L"): d["seis"].append(iv)  # velocity seismometer only (H/L gain)
            el.clear()
    rows = []
    for (net, code), d in A.items():
        if d["lat"] is None: continue
        use = d["seis"] or d["all"]
        st = [x for x in (_ts(a) for a, b in use if a) if x is not None]
        en = [x for x in (_ts(b) for a, b in use if b) if x is not None]
        me = max(en) if en else None
        rows.append(dict(net=net, sta=code, lat=d["lat"], lon=d["lon"], elev=d["elev"],
                         t0=min(st) if st else pd.Timestamp("2000-01-01"),
                         t1=(pd.Timestamp("2030-01-01") if (me is None or me.year >= 2098) else me),
                         accel_only=(len(d["seis"]) == 0), bands="/".join(sorted(d["bands"])),
                         archive=ARCH_KSKG))
    return pd.DataFrame(rows)


def load_ns(path):
    """NS dense local array: base code (N003a->N003), coords, epoch, band=HH, archive=NS."""
    d = pd.read_csv(path)
    d["sta"] = d.station.str.replace(r"[a-z]$", "", regex=True)
    d["t0"] = pd.to_datetime(d.starttime, errors="coerce"); d["t1"] = pd.to_datetime(d.endtime, errors="coerce")
    d = d.dropna(subset=["stla", "stlo"])
    # collapse epoch variants to one row per base code (widest epoch window, mean coords)
    g = d.groupby("sta").agg(lat=("stla", "mean"), lon=("stlo", "mean"), elev=("stel", "mean"),
                             t0=("t0", "min"), t1=("t1", "max")).reset_index()
    g["net"] = "NS"; g["accel_only"] = False; g["bands"] = "HH"; g["archive"] = ARCH_NS
    return g[["net", "sta", "lat", "lon", "elev", "t0", "t1", "accel_only", "bands", "archive"]]


def load_gj(path):
    """GJ dense temporary arrays (2016-2017 Gyeongju deployment): coords from list, band=HH, archive=GJ.
    No epoch info in the list -> wide window; the coverage scan decides actual presence per month."""
    if not os.path.exists(path): return pd.DataFrame()
    d = pd.read_csv(path)
    g = pd.DataFrame(dict(net="GJ", sta=d.Code, lat=d.Latitude, lon=d.Longitude,
                          elev=d.get("Elevation", 0.0), t0=pd.Timestamp("2000-01-01"),
                          t1=pd.Timestamp("2030-01-01"), accel_only=False, bands="HH", archive=ARCH_GJ))
    return g.drop_duplicates("sta")


def pick_band(bands):
    for cand in BANDS:                       # HH > EL > HG
        if cand in bands.split("/"): return cand
    return None


def days_local(row, Y, DOY0, DOY1):
    days = set()
    bdir = os.path.join(row.archive, row.sta)
    if os.path.isdir(bdir):
        for ch in os.listdir(bdir):
            if not ch.endswith(".D") or not ch.startswith(row.band): continue
            try: entries = os.scandir(os.path.join(bdir, ch))
            except OSError: continue
            for f in entries:
                p = f.name.split(".")
                if len(p) >= 7 and p[-2] == str(Y):
                    try:
                        if DOY0 <= int(p[-1]) <= DOY1: days.add(int(p[-1]))
                    except ValueError: pass
    return len(days)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2021-09", help="YYYY-MM")
    a = ap.parse_args()
    Y, MO = int(a.month[:4]), int(a.month[5:7])
    tag = f"{Y}_{MO:02d}"
    T0 = pd.Timestamp(f"{a.month}-01"); T1 = T0 + pd.offsets.MonthEnd(0)
    DOY0, DOY1 = T0.dayofyear, T1.dayofyear; NDAYS = DOY1 - DOY0 + 1
    print(f"[{a.month}] doy {DOY0}-{DOY1} ({NDAYS} d) | region {RMAX_KM:.0f} km around {REGION_CENTER}")

    S = pd.concat([load_kskg(P_XML), load_ns(P_NS), load_gj(P_GJ)], ignore_index=True)
    S["dist_km"] = hav_km(REGION_CENTER[0], REGION_CENTER[1], S.lat, S.lon)
    S = S[(S.dist_km <= RMAX_KM) & (S.t0 <= T1) & (S.t1 >= T0)].copy()
    S["band"] = S.bands.map(pick_band)                      # HH > EL > HG; None if none of those -> dropped
    S = S[S.band.notna()].sort_values("dist_km").reset_index(drop=True)
    S["days_local"] = [days_local(r, Y, DOY0, DOY1) for _, r in S.iterrows()]
    S["coverage"] = S.days_local / NDAYS

    os.makedirs(f"{HERE}/cache", exist_ok=True)
    S.to_csv(f"{HERE}/cache/stations_{tag}.csv", index=False)
    USE = S[S.coverage >= 0.8]                              # HH/EL/HG all allowed (one band per station)
    for net in ("KS", "KG", "NS", "GJ"):
        n_all = int((S.net == net).sum()); n_use = int((USE.net == net).sum())
        if n_all: print(f"  {net}: {n_all} in region / {n_use} usable (>=80% local coverage) "
                        f"| bands {dict(USE[USE.net==net].band.value_counts())}")
    print(f"  PICKING SET: {len(USE)} stations "
          f"(dist {USE.dist_km.min():.1f}-{USE.dist_km.max():.1f} km) -> cache/stations_{tag}.csv")

    # truth catalogs
    kma = pd.read_csv(P_KMA)
    kma["time"] = pd.to_datetime(kma.event_id.astype(str).str[:14], format="%Y%m%d%H%M%S")
    kma["dist_km"] = hav_km(REGION_CENTER[0], REGION_CENTER[1], kma.kma_lat, kma.kma_lon)
    TR_KMA = kma[(kma.time >= T0) & (kma.time <= T1 + pd.Timedelta(days=1)) & (kma.dist_km <= RMAX_KM)].copy()
    uf = pd.read_csv(P_UF)
    uf["time"] = pd.to_datetime(uf.event_time, format="ISO8601", utc=True, errors="coerce").dt.tz_localize(None)
    TR_UF = uf[(uf.time >= T0) & (uf.time <= T1 + pd.Timedelta(days=1))].copy()
    TR_KMA.to_csv(f"{HERE}/cache/truth_kma_{tag}.csv", index=False)
    TR_UF.to_csv(f"{HERE}/cache/truth_uf_relocated_{tag}.csv", index=False)
    print(f"  truth: {len(TR_KMA)} KMA (M {TR_KMA.kma_mag.min():.1f}-{TR_KMA.kma_mag.max():.1f}) "
          f"+ {len(TR_UF)} UF-relocated events")


if __name__ == "__main__":
    main()
