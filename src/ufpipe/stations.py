"""Multi-network station table (KS / KG / GJ / NS) for detection + association.

ufpipe covers four networks; each has its own waveform archive and metadata source. This module
builds ONE per-year table telling detection which stations exist that year, where each records, and
which waveform directory + channel band to read. Ported faithfully from the validated
``detection_test/lib/build_stations.py`` (same coord/epoch/band logic), keyed by YEAR instead of month
and driven entirely by ``config`` paths (no hard-coded locations).

Station sources (disclosed):
  * KS + KG : coords/epochs/bands from ``config.STATION_XML`` (seismometer-window rule); archive KS_KG/.
  * NS      : coords/epochs from ``config.NS_STATION_CSV`` (GHBSN; base-code N003a->N003 collapse),
              band HH, native 200 Hz; archive NS/ (or the 100 Hz mirror NS_100hz/ at detection time).
  * GJ      : coords from ``config.GJ_STATION_CSV`` (2016-2017 temporary arrays); band HH; archive GJ/.

A station is kept for a year if its metadata epoch overlaps the year AND it has >=1 local day on disk.
The result carries columns:  net, sta, lat, lon, elev, t0, t1, band, archive.
"""
import os
import sys
import glob
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

_NSX = "{http://www.fdsn.org/xml/station/1}"


def _ts(x):
    try:
        return pd.Timestamp(x).tz_localize(None)
    except Exception:
        return None


def _pick_band(bands):
    """One channel band per station, in priority order (velocity HH/EL before accel HG)."""
    have = set(str(bands).split("/"))
    for cand in config.BANDS:
        if cand in have:
            return cand
    return None


# ---------------------------------------------------------------- per-network loaders
def load_kskg(path=None):
    """KS + KG from the StationXML: coords, epoch window (velocity seismometer channels preferred),
    band set. Archive = config.KS_KG_DIR. Ported from build_stations.load_kskg."""
    path = path or config.STATION_XML
    A = defaultdict(lambda: {"lat": None, "lon": None, "elev": 0.0, "seis": [], "all": [], "bands": set()})
    net = None
    for ev, el in ET.iterparse(path, events=("start", "end")):
        t = el.tag.split("}")[-1]
        if ev == "start" and t == "Network":
            net = el.get("code")
        elif ev == "end" and t == "Station":
            d = A[(net, el.get("code"))]
            la = el.findtext(_NSX + "Latitude")
            if la:
                d["lat"] = float(la)
                d["lon"] = float(el.findtext(_NSX + "Longitude"))
                e = el.findtext(_NSX + "Elevation")
                d["elev"] = float(e) if e else 0.0
            for c in el.findall(_NSX + "Channel"):
                cc = c.get("code") or ""
                d["bands"].add(cc[:2])
                iv = (c.get("startDate"), c.get("endDate"))
                d["all"].append(iv)
                if len(cc) > 1 and cc[1] in ("H", "L"):     # velocity seismometer only (H/L gain)
                    d["seis"].append(iv)
            el.clear()
    rows = []
    for (net, code), d in A.items():
        if d["lat"] is None:
            continue
        use = d["seis"] or d["all"]
        st = [x for x in (_ts(a) for a, b in use if a) if x is not None]
        en = [x for x in (_ts(b) for a, b in use if b) if x is not None]
        me = max(en) if en else None
        rows.append(dict(net=net, sta=code, lat=d["lat"], lon=d["lon"], elev=d["elev"],
                         t0=min(st) if st else pd.Timestamp("2000-01-01"),
                         t1=(pd.Timestamp("2030-01-01") if (me is None or me.year >= 2098) else me),
                         bands="/".join(sorted(d["bands"])), archive=config.KS_KG_DIR))
    return pd.DataFrame(rows)


def load_ns(path=None):
    """NS dense local array: base code (N003a->N003), coords, epoch, band HH, archive NS/.
    Ported from build_stations.load_ns."""
    path = path or config.NS_STATION_CSV
    if not os.path.exists(path):
        return pd.DataFrame()
    d = pd.read_csv(path)
    d["sta"] = d.station.str.replace(r"[a-z]$", "", regex=True)
    d["t0"] = pd.to_datetime(d.starttime, errors="coerce")
    d["t1"] = pd.to_datetime(d.endtime, errors="coerce")
    d = d.dropna(subset=["stla", "stlo"])
    g = d.groupby("sta").agg(lat=("stla", "mean"), lon=("stlo", "mean"), elev=("stel", "mean"),
                             t0=("t0", "min"), t1=("t1", "max")).reset_index()
    g["net"] = "NS"
    g["bands"] = "HH"
    g["archive"] = config.NS_DIR
    return g[["net", "sta", "lat", "lon", "elev", "t0", "t1", "bands", "archive"]]


def load_gj(path=None):
    """GJ 2016-2017 temporary arrays: coords from list, band HH, archive GJ/. No epoch info in the
    list -> wide window; the on-disk day scan decides actual presence. Ported from build_stations.load_gj."""
    path = path or config.GJ_STATION_CSV
    if not os.path.exists(path):
        return pd.DataFrame()
    d = pd.read_csv(path)
    g = pd.DataFrame(dict(net="GJ", sta=d.Code, lat=d.Latitude, lon=d.Longitude,
                          elev=d.get("Elevation", 0.0),
                          t0=pd.Timestamp("2000-01-01"), t1=pd.Timestamp("2030-01-01"),
                          bands="HH", archive=config.GJ_DIR))
    return g.drop_duplicates("sta")


# ---------------------------------------------------------------- on-disk day scan
def _days_local(archive, sta, band, year):
    """Number of days in `year` that station has on local disk under archive/<sta>/<band>?.D/."""
    n = 0
    bdir = os.path.join(archive, sta)
    if not os.path.isdir(bdir):
        return 0
    for ch in os.listdir(bdir):
        if not ch.endswith(".D") or not ch.startswith(band):
            continue
        try:
            entries = os.scandir(os.path.join(bdir, ch))
        except OSError:
            continue
        for f in entries:
            p = f.name.split(".")
            if len(p) >= 7 and p[-2] == str(year):
                n += 1
        if n:
            break     # one channel dir with year-data is enough to confirm presence
    return n


# ---------------------------------------------------------------- the per-year table
def build_year_table(year, networks=None, use_cache=True):
    """Return the multi-archive station table for `year`, restricted to `networks`
    (default config.DETECT_NETWORKS). Columns: net, sta, lat, lon, elev, t0, t1, band, archive.

    A station is kept if its metadata epoch overlaps the year AND it has >=1 local day on disk.
    Cached to config.STATION_TABLE_CACHE/stations_<year>.csv (per full-network build)."""
    networks = tuple(networks) if networks else config.DETECT_NETWORKS
    cache = os.path.join(config.STATION_TABLE_CACHE, f"stations_{year}.csv")
    if use_cache and os.path.exists(cache):
        S = pd.read_csv(cache, parse_dates=["t0", "t1"])
    else:
        S = pd.concat([load_kskg(), load_ns(), load_gj()], ignore_index=True)
        S = S[S.lat.notna()].copy()
        y0 = pd.Timestamp(f"{year}-01-01")
        y1 = pd.Timestamp(f"{year}-12-31 23:59:59")
        S = S[(S.t0 <= y1) & (S.t1 >= y0)].copy()               # metadata epoch overlaps the year
        S["band"] = S.bands.map(_pick_band)
        S = S[S.band.notna()].copy()
        S["days_local"] = [_days_local(r.archive, r.sta, r.band, year) for _, r in S.iterrows()]
        S = S[S.days_local > 0].reset_index(drop=True)          # has real data on disk this year
        os.makedirs(config.STATION_TABLE_CACHE, exist_ok=True)
        S.to_csv(cache, index=False)
    return S[S.net.isin(networks)].reset_index(drop=True)


def discover_rows(year, networks=None, stations=None, use_ns_100hz=None):
    """Detection-facing station rows for `year`: list of dicts (net, sta, band, archive).

    - restricts to `networks` (default all four) and optionally an explicit `stations` list;
    - for NS, swaps archive -> NS_100hz mirror when config.USE_NS_100HZ and the station is mirrored
      (falls back to the native 200 Hz NS/ archive otherwise — read there is anti-alias decimated)."""
    use_ns_100hz = config.USE_NS_100HZ if use_ns_100hz is None else use_ns_100hz
    S = build_year_table(year, networks=networks)
    if stations:
        want = set(stations)
        S = S[S.sta.isin(want)]
    rows = []
    for _, r in S.iterrows():
        archive = r.archive
        if r.net == "NS" and use_ns_100hz and os.path.isdir(os.path.join(config.NS_100HZ_DIR, r.sta)):
            archive = config.NS_100HZ_DIR
        rows.append(dict(net=r.net, sta=r.sta, band=r.band, archive=archive))
    return rows
