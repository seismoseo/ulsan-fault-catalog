"""Export per-event SAC waveforms from the Ulsan-Fault continuous miniSEED archive.

For each row of the PhaseNet+ blast-free catalog this module cuts a per-station 3-component
window around the HYPOINVERSE origin time, writes a SAC file per (station, channel), and
populates SAC headers with the catalog hypocentre + PhaseNet+ P/S pick times. Output layout
is **flat per-event** so the dataset is browseable by event_id:

    <out_root>/<event_id>/<event_id>.<NET>.<STA>.<CHAN>.sac     # one per station-channel
    <out_root>/<event_id>/<event_id>_picks.csv                  # full PhaseNet+ pick rows
    <out_root>/export_summary.csv                               # one row per event

Designed to be driven from `06.Export_event_waveforms_from_continuous.ipynb`: each function
is small enough to run + inspect in a single notebook cell. The bulk orchestrator
`export_catalog()` is idempotent (skips events whose directory already has ≥1 SAC) so the
notebook's bulk cell can be re-run safely.

Picks are written to the **SAC `a` (P) and `t0` (S)** headers — the same convention that
[korea-cluster-relocation/pipeline/core/xcorr.py](
    https://github.com/seismoseo/korea-cluster-relocation/blob/main/pipeline/core/xcorr.py)
reads. So the output tree is plug-compatible with that framework's `stp_cluster()` config:
point `stp_sac_root` at `<out_root>` and adjust `stp_sac_glob` to drop the per-sensor
subdirectory (these SACs live flat per event).
"""
from __future__ import annotations

import functools
import glob
import os
from dataclasses import dataclass

import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth


# --- defaults ---------------------------------------------------------------------------
# Window around origin (matches the 2014-sequence prototype at 05.Cut_event_*.ipynb).
PRE_S_DEFAULT = 30.0
POST_S_DEFAULT = 90.0
# Pick-association window around origin (LEGACY): the old `associate_picks` time-window
# heuristic that bundled every raw PhaseNet+ pick within [-5, +60] s of origin. We now
# route per-event picks through PyOcto's actual association tables instead; this constant
# remains only for the deprecated `associate_picks` fallback used when no PyOcto table is
# provided.
ASSOC_WINDOW_DEFAULT = (-5.0, 60.0)
# PyOcto → catalog-row time-match tolerance. PyOcto's origin time differs from
# HypoInverse's refined origin by typically < 0.5 s; 5 s is generous and still avoids
# cross-matching distinct events (real intra-cluster spacing is rarely < 30 s).
PYOCTO_TIME_TOL_S = 5.0
# Velocity model the PyOcto association was run under. Files live at
#   <pyocto_root>/pyocto_kim1983_<year>.csv             (event table)
#   <pyocto_root>/pyocto_assignment_kim1983_<year>.csv  (per-pick assignment)
PYOCTO_VELMODEL_DEFAULT = "kim1983"
TARGET_HZ = 100.0


# --- catalog ---------------------------------------------------------------------------
def event_id_from_time(t) -> str:
    """UTC origin → ``YYYYMMDDHHMMSS`` (14 chars). Sub-second times truncate to whole
    seconds, matching the eq-cycle's event_id convention. Pass a ``pd.Timestamp`` or
    ``obspy.UTCDateTime``."""
    if isinstance(t, UTCDateTime):
        t = t.datetime
    elif isinstance(t, pd.Timestamp):
        t = t.to_pydatetime()
    return t.strftime("%Y%m%d%H%M%S")


def load_catalog(path) -> pd.DataFrame:
    """Read the blast-clean catalog CSV. Sorts by origin time and adds:

      - ``event_id``  — UTC ``YYYYMMDDHHMMSS`` (14 chars). When two or more catalog rows
        share the same whole-UTC-second (sub-second doublets), the FIRST one keeps the
        plain id and the rest get a single-letter suffix appended (``b``, ``c``, …) so
        each row's event_id is unique.  Example::

            2022-07-25 19:30:05.170  →  20220725193005
            2022-07-25 19:30:05.230  →  20220725193005b

        15-char ids still fit SAC's 16-char ``KEVNM`` slot.
      - ``origin_utc`` — ``obspy.UTCDateTime`` for the row.
    """
    cat = pd.read_csv(path, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    base = cat["time"].apply(event_id_from_time)
    # cumcount within each base id: 0 = first encounter (no suffix), 1 → "b", 2 → "c", …
    n_within = base.groupby(base).cumcount()
    cat["event_id"] = [
        b if k == 0 else f"{b}{chr(ord('a') + k)}"
        for b, k in zip(base, n_within)
    ]
    cat["origin_utc"] = cat["time"].apply(lambda t: UTCDateTime(t.to_pydatetime()))
    return cat


# --- stations --------------------------------------------------------------------------
def load_stations_for_year(station_table_root, year) -> pd.DataFrame:
    """Per-year station table (`stations_<YYYY>.csv`) → DataFrame with columns
    ``Network, Code, Latitude, Longitude, Elevation``.

    The PhaseNet+ pipeline emits one of these per year — they're the stations active
    in that year. Falls back to the closest earlier year if the requested year is
    missing (e.g. for a 2025 event when only 2024 is on disk)."""
    target = os.path.join(station_table_root, f"stations_{year}.csv")
    if os.path.exists(target):
        return pd.read_csv(target)
    # fallback: nearest earlier year
    candidates = sorted(glob.glob(os.path.join(station_table_root, "stations_*.csv")))
    years = [int(os.path.basename(p).split("_")[1].split(".")[0]) for p in candidates]
    earlier = [(y, p) for y, p in zip(years, candidates) if y <= year]
    if not earlier:
        raise FileNotFoundError(f"no stations_<YYYY>.csv with year ≤ {year} under {station_table_root}")
    _, path = earlier[-1]
    return pd.read_csv(path)


# --- picks -----------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def _load_picks_csv_cached(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["peak_time"] = pd.to_datetime(df["peak_time"], utc=True)
    # split "NET.CODE" → Network, Code (the canonical eq-cycle id form)
    nc = df["station"].str.split(".", n=1, expand=True)
    df["Network"], df["Code"] = nc[0], nc[1]
    return df


def load_picks_for_day(picks_root, year, julday) -> pd.DataFrame:
    """Read one PhaseNet+ daily pick CSV.

    Path: ``<picks_root>/<YYYY>/picks/picks_<YYYY>.<DDD>.csv`` (DDD = 3-digit julday).
    Columns: ``station, phase, peak_time, probability`` from disk, plus ``Network`` +
    ``Code`` split out by this loader. Empty DataFrame if the file is missing — picks
    are unavailable for some days (no PhaseNet+ run for that day)."""
    fn = os.path.join(picks_root, str(year), "picks", f"picks_{year}.{julday:03d}.csv")
    if not os.path.exists(fn):
        return pd.DataFrame(columns=["station", "phase", "peak_time", "probability",
                                     "Network", "Code"])
    return _load_picks_csv_cached(fn).copy()


def associate_picks(picks_day, origin_utc, window=ASSOC_WINDOW_DEFAULT) -> pd.DataFrame:
    """**DEPRECATED** time-window heuristic. Filters ``picks_day`` to picks within
    ``[origin + window[0], origin + window[1]]`` and returns them verbatim. Retained for
    fallback only — callers should prefer ``associate_picks_from_pyocto`` which uses
    PyOcto's actual association decisions instead of a fresh time window.

    The bug this caused, documented in `04.Catalog_quality_audit.ipynb` (2015-11-13
    case study): for a close-in pair of events the window catches BOTH events' picks at
    each station, and `earliest_per_station_phase` then promotes the chronologically
    earlier event's picks into the later event's SAC headers. Use only when you genuinely
    have no PyOcto run to consult."""
    if len(picks_day) == 0:
        return picks_day
    t_lo = pd.Timestamp(origin_utc.datetime, tz="UTC") + pd.Timedelta(seconds=window[0])
    t_hi = pd.Timestamp(origin_utc.datetime, tz="UTC") + pd.Timedelta(seconds=window[1])
    m = (picks_day["peak_time"] >= t_lo) & (picks_day["peak_time"] <= t_hi)
    return picks_day.loc[m].reset_index(drop=True)


def earliest_per_station_phase(matched):
    """Reduce a picks DataFrame to one (P, S) per (Network, Code).

    For `associate_picks` output (legacy): chooses the **chronologically earliest**
    P and S in the window — the source of the cross-event header bug. For
    `associate_picks_from_pyocto` output, PyOcto only emits one P + one S per
    (event, station) so the "earliest" reducer is a no-op identity.

    Returns ``{(Network, Code): {"P": (peak_time, probability), "S": (...)}}``. Stations
    without a P or S simply omit that key."""
    if len(matched) == 0:
        return {}
    matched = matched.sort_values("peak_time")
    out = {}
    for (net, code, ph), g in matched.groupby(["Network", "Code", "phase"]):
        r = g.iloc[0]
        out.setdefault((net, code), {})[ph] = (r["peak_time"], float(r["probability"]))
    return out


# --- PyOcto-association picks (the correct path) ----------------------------------------
@functools.lru_cache(maxsize=32)
def _load_pyocto_year_cached(pyocto_root: str, year: int, velmodel: str):
    """Cached per-year (events, assignment) PyOcto tables. Returns ``(None, None)`` if
    the files don't exist for that year — caller must fall back gracefully."""
    evt_path = os.path.join(pyocto_root, f"pyocto_{velmodel}_{year}.csv")
    asg_path = os.path.join(pyocto_root, f"pyocto_assignment_{velmodel}_{year}.csv")
    if not (os.path.exists(evt_path) and os.path.exists(asg_path)):
        return None, None
    evt = pd.read_csv(evt_path)
    evt["time"] = pd.to_datetime(evt["time"], utc=True)
    asg = pd.read_csv(asg_path)
    # Split the canonical "NET.CODE." form into Network + Code (mirrors the picks loader).
    parts = asg["station"].str.rstrip(".").str.split(".", expand=True)
    asg["Network"] = parts[0]
    asg["Code"] = parts[1]
    # Pick time → pandas Timestamp (UTC). PyOcto stores unix-epoch seconds.
    asg["peak_time"] = pd.to_datetime(asg["time"], unit="s", utc=True)
    return evt, asg


def load_pyocto_year(pyocto_root, year, velmodel=PYOCTO_VELMODEL_DEFAULT):
    """Public wrapper around the cached loader."""
    return _load_pyocto_year_cached(pyocto_root, int(year), str(velmodel))


def match_pyocto_event_idx(pyocto_events, origin_utc, tol_s=PYOCTO_TIME_TOL_S):
    """Return the ``idx`` of the PyOcto event whose origin time is closest to
    ``origin_utc`` and within ``tol_s`` seconds. Returns ``None`` if no PyOcto event is
    within tolerance (caller should log + skip)."""
    if pyocto_events is None or len(pyocto_events) == 0:
        return None
    t = pd.Timestamp(origin_utc.datetime, tz="UTC")
    dt = (pyocto_events["time"] - t).abs()
    best = dt.idxmin()
    if dt.loc[best].total_seconds() > tol_s:
        return None
    return int(pyocto_events.loc[best, "idx"])


def associate_picks_from_pyocto(pyocto_assignment, pyocto_event_idx) -> pd.DataFrame:
    """Return the (Network, Code, phase, peak_time, probability) rows PyOcto associated
    with ``pyocto_event_idx``. Same column schema as ``associate_picks`` so it's a
    drop-in replacement upstream of ``earliest_per_station_phase``.

    PyOcto assignment carries a per-pick ``residual`` (s) but no PhaseNet+ probability.
    A constant `probability=1.0` is emitted so the downstream pipeline (sidecar CSV,
    earliest reducer) has the expected column."""
    if pyocto_assignment is None or len(pyocto_assignment) == 0:
        return pd.DataFrame(columns=["Network", "Code", "phase", "peak_time",
                                     "probability", "residual", "station"])
    m = pyocto_assignment["event_idx"] == pyocto_event_idx
    sub = pyocto_assignment.loc[m, ["Network", "Code", "phase", "peak_time",
                                    "residual", "station"]].copy()
    sub["probability"] = 1.0
    return sub.reset_index(drop=True)


# --- continuous archive ----------------------------------------------------------------
def _continuous_glob_for_station(continuous_root, network, code, year, julday):
    """Glob the daily miniSEED files for one (station, day) across all channels.

    Layout: ``<continuous_root>/<CODE>/<CHA>.D/<NET>.<CODE>..<CHA>.D.<YYYY>.<DDD>``.
    Returns a list (possibly empty if the station has no data for that day)."""
    pat = os.path.join(continuous_root, code, "*.D",
                       f"{network}.{code}..*.D.{year}.{julday:03d}")
    return sorted(glob.glob(pat))


def read_continuous_for_event(continuous_root, network, code, origin_utc,
                              pre_s=PRE_S_DEFAULT, post_s=POST_S_DEFAULT,
                              target_hz=TARGET_HZ):
    """Return a 3-component ``Stream`` for one station, **read only for the window**
    [origin − pre_s, origin + post_s]. ObsPy's miniSEED reader supports `starttime` /
    `endtime` kwargs and reads only the matching records from disk — orders of
    magnitude faster than reading the whole 24-hour day-file. If the event straddles
    UTC midnight, both day-files are read with the same time bounds. Empty ``Stream``
    if no day-file is on disk for any channel.

    ``target_hz`` (default ``TARGET_HZ`` = 100): asserts each trace's sampling_rate equals
    it — the framework's KS/KG convention; we don't silently resample. Pass ``None`` to
    keep each station at its **native** sampling rate (the multi-rate GJ/NS archives): the
    SAC store is then a durable data product preserving the full band for later spectral /
    stress-drop analysis, and the dt.cc xcorr interpolates to a common `interp_hz` at
    correlation time anyway, so relocation is unaffected."""
    o = origin_utc
    t_start = o - pre_s
    t_end = o + post_s
    # Determine which (year, julday) pairs we need
    needed = [(t_start.year, t_start.julday)]
    if (t_end.year, t_end.julday) != needed[0]:
        needed.append((t_end.year, t_end.julday))

    files = []
    for y, jd in needed:
        files.extend(_continuous_glob_for_station(continuous_root, network, code, y, jd))
    if not files:
        return obspy.Stream()
    st = obspy.Stream()
    for f in files:
        try:
            # The miniSEED reader only loads records that intersect the time window
            st.extend(obspy.read(f, starttime=t_start, endtime=t_end))
        except Exception:                        # noqa: BLE001
            continue
    if not len(st):
        return st
    # Merge each channel across day boundary, filling gaps with 0 (SAC format does NOT
    # support masked arrays, and the alternative — `merge(fill_value=None)` — produces
    # exactly that, breaking the downstream `tr.write("SAC")` call. Filling gaps is the
    # right behaviour for a typed-out window: a brief telemetry dropout becomes a few
    # zeros in the SAC, easy to spot in a plot and harmless for picking + xcorr.
    try:
        st.merge(method=1, fill_value=0)
    except Exception:                            # noqa: BLE001
        pass
    # sample-rate guard (skipped when target_hz is None -> keep native multi-rate archives)
    if target_hz is not None:
        for tr in st:
            if abs(tr.stats.sampling_rate - target_hz) > 0.1:
                raise ValueError(f"{tr.id}: sampling_rate {tr.stats.sampling_rate} Hz "
                                 f"≠ expected {target_hz} Hz")
    # Trim exactly to the window (read may have over-fetched whole records)
    out = st.slice(starttime=t_start, endtime=t_end)
    # Defensive: if any trace still carries a masked array (e.g. merge failed), unmask it
    for tr in out:
        import numpy as _np
        if _np.ma.isMaskedArray(tr.data):
            tr.data = tr.data.filled(0).astype(_np.float32)
    return out


# --- SAC writer ------------------------------------------------------------------------
@dataclass
class EventMeta:
    """The minimum a SAC writer needs to populate event-level headers."""
    event_id: str
    origin_utc: UTCDateTime
    lat: float
    lon: float
    depth_km: float


@dataclass
class StationMeta:
    network: str
    code: str
    lat: float
    lon: float
    elev_m: float | None


def _populate_sac_header(tr, ev: EventMeta, sta: StationMeta,
                         p_rel_s=None, s_rel_s=None,
                         p_label="P", s_label="S"):
    """Write the full event-SAC header into ``tr.stats.sac`` in place."""
    s = tr.stats.sac if "sac" in tr.stats else tr.stats.setdefault("sac", obspy.core.AttribDict())
    o = ev.origin_utc
    # origin reference
    s.nzyear, s.nzjday = o.year, o.julday
    s.nzhour, s.nzmin, s.nzsec = o.hour, o.minute, o.second
    s.nzmsec = int(round(o.microsecond / 1000))
    s.o = 0.0
    # hypocentre
    s.evla, s.evlo, s.evdp = float(ev.lat), float(ev.lon), float(ev.depth_km)
    # station
    s.stla, s.stlo = float(sta.lat), float(sta.lon)
    if sta.elev_m is not None and np.isfinite(sta.elev_m):
        s.stel = float(sta.elev_m)
    # geometry
    dist_m, az, baz = gps2dist_azimuth(ev.lat, ev.lon, sta.lat, sta.lon)
    s.dist, s.az, s.baz = dist_m / 1000.0, float(az), float(baz)
    # identifiers
    s.kevnm = ev.event_id[:16]                  # SAC kevnm is 16 chars wide
    s.kstnm = sta.code[:8]
    s.knetwk = sta.network[:8]
    s.kcmpnm = tr.stats.channel[:8]
    # picks (relative to origin)
    if p_rel_s is not None and np.isfinite(p_rel_s):
        s.a = float(p_rel_s); s.ka = p_label
    if s_rel_s is not None and np.isfinite(s_rel_s):
        s.t0 = float(s_rel_s); s.kt0 = s_label


def write_event_sac(stream, out_dir, *, event_meta, station_meta,
                    p_rel_s=None, s_rel_s=None):
    """Write each trace in ``stream`` as ``<event_id>.<NET>.<STA>.<CHAN>.sac`` into
    ``out_dir``, with the full event-SAC header populated. Returns the list of written
    paths."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for tr in stream:
        _populate_sac_header(tr, event_meta, station_meta,
                             p_rel_s=p_rel_s, s_rel_s=s_rel_s)
        fn = (f"{event_meta.event_id}.{station_meta.network}.{station_meta.code}"
              f".{tr.stats.channel}.sac")
        path = os.path.join(out_dir, fn)
        tr.write(path, format="SAC")
        written.append(path)
    return written


# --- single-event orchestrator ----------------------------------------------------------
def export_event(catalog_row, picks_root, continuous_root, stations, out_root,
                 pre_s=PRE_S_DEFAULT, post_s=POST_S_DEFAULT,
                 assoc_window=ASSOC_WINDOW_DEFAULT,
                 *, pyocto_root=None, pyocto_velmodel=PYOCTO_VELMODEL_DEFAULT,
                 pyocto_tol_s=PYOCTO_TIME_TOL_S, target_hz=TARGET_HZ) -> dict:
    """Cut + write all SACs for one catalog event, plus a sidecar picks CSV.

    `catalog_row` must carry `event_id`, `origin_utc`, `lat`, `lon`, `depth`.
    `stations` is the per-year DataFrame from ``load_stations_for_year``.

    **Pick source**: when ``pyocto_root`` is set (the **default for v2**, passed by the
    notebook), the SAC `a`/`t0` headers are populated from PyOcto's per-event assignment
    (``<pyocto_root>/pyocto_assignment_<velmodel>_<year>.csv``) — one P + one S per
    station as PyOcto associated them. The sidecar CSV gets those same rows. When
    ``pyocto_root`` is `None`, we fall back to the legacy time-window
    `associate_picks` — preserved only for backwards reproducibility; not recommended.

    Returns a summary dict (event_id, n_sac, n_stations_with_p, n_stations_with_s,
    n_stations_tried, status, pick_source). Status is 'ok', 'no_picks_day',
    'no_streams', or 'no_pyocto_match' (only when ``pyocto_root`` is set)."""
    eid = catalog_row["event_id"]
    o = catalog_row["origin_utc"]
    event_meta = EventMeta(eid, o, float(catalog_row["lat"]),
                           float(catalog_row["lon"]), float(catalog_row["depth"]))

    out_dir = os.path.join(out_root, eid)
    # ---- picks: PyOcto-assignment (preferred) or legacy time-window (fallback) ----
    pick_source = "legacy_time_window"
    if pyocto_root is not None:
        evt, asg = load_pyocto_year(pyocto_root, o.year, velmodel=pyocto_velmodel)
        if evt is None:
            return dict(event_id=eid, n_sac=0, n_stations_with_p=0,
                        n_stations_with_s=0, n_stations_tried=0,
                        n_station_errors=0,
                        status=f"no_pyocto_tables_for_year_{o.year}",
                        pick_source="pyocto")
        ev_idx = match_pyocto_event_idx(evt, o, tol_s=pyocto_tol_s)
        if ev_idx is None:
            return dict(event_id=eid, n_sac=0, n_stations_with_p=0,
                        n_stations_with_s=0, n_stations_tried=0,
                        n_station_errors=0,
                        status="no_pyocto_match",
                        pick_source="pyocto")
        matched = associate_picks_from_pyocto(asg, ev_idx)
        pick_source = "pyocto"
    else:
        picks_day = load_picks_for_day(picks_root, o.year, o.julday)
        # if the event straddles midnight, include the previous day's picks too (rare)
        if (o - pre_s).julday != o.julday:
            prev = load_picks_for_day(picks_root, (o - pre_s).year, (o - pre_s).julday)
            picks_day = pd.concat([prev, picks_day], ignore_index=True)
        matched = associate_picks(picks_day, o, window=assoc_window)
    earliest = earliest_per_station_phase(matched)

    n_sac = 0
    n_with_p = n_with_s = 0
    n_tried = 0
    n_station_errors = 0
    for _, sr in stations.iterrows():
        n_tried += 1
        sm = StationMeta(sr["Network"], sr["Code"],
                         float(sr["Latitude"]), float(sr["Longitude"]),
                         float(sr["Elevation"]) if "Elevation" in sr else None)
        try:
            st = read_continuous_for_event(continuous_root, sm.network, sm.code, o,
                                           pre_s=pre_s, post_s=post_s, target_hz=target_hz)
            if not len(st):
                continue
            # earliest P/S for this station, expressed relative to origin
            picks_st = earliest.get((sm.network, sm.code), {})
            p_rel = None; s_rel = None
            if "P" in picks_st:
                pt, _ = picks_st["P"]
                p_rel = (UTCDateTime(pt.to_pydatetime()) - o)
                n_with_p += 1
            if "S" in picks_st:
                stt, _ = picks_st["S"]
                s_rel = (UTCDateTime(stt.to_pydatetime()) - o)
                n_with_s += 1
            paths = write_event_sac(st, out_dir, event_meta=event_meta, station_meta=sm,
                                    p_rel_s=p_rel, s_rel_s=s_rel)
            n_sac += len(paths)
        except Exception:                            # noqa: BLE001
            # one bad station shouldn't kill the whole event — log a count and move on
            n_station_errors += 1
            continue

    # sidecar picks CSV (PyOcto-assigned set when pyocto_root given; legacy window
    # output otherwise). Writing the same filename either way keeps downstream readers
    # transparent; the `pick_source` field in the summary tells you which scheme produced
    # the data on disk.
    if n_sac and len(matched):
        matched.to_csv(os.path.join(out_dir, f"{eid}_picks.csv"), index=False)

    if n_sac:
        status = "ok"
    elif pick_source == "pyocto":
        status = "no_streams" if len(matched) else "no_picks_assigned"
    else:
        status = "no_streams" if len(picks_day) else "no_picks_day"
    return dict(event_id=eid, n_sac=n_sac, n_stations_with_p=n_with_p,
                n_stations_with_s=n_with_s, n_stations_tried=n_tried,
                n_station_errors=n_station_errors, status=status,
                pick_source=pick_source)


# --- bulk orchestrator ------------------------------------------------------------------
def export_catalog(catalog_df, picks_root, continuous_root, station_table_root, out_root,
                   *, pre_s=PRE_S_DEFAULT, post_s=POST_S_DEFAULT,
                   assoc_window=ASSOC_WINDOW_DEFAULT,
                   pyocto_root=None, pyocto_velmodel=PYOCTO_VELMODEL_DEFAULT,
                   pyocto_tol_s=PYOCTO_TIME_TOL_S, target_hz=TARGET_HZ,
                   skip_existing=True, max_events=None, progress=True) -> pd.DataFrame:
    """Run ``export_event`` over every row of ``catalog_df`` and return a per-event
    summary DataFrame.

    `pyocto_root` (recommended): when set, every event's SAC headers come from the
    PyOcto association tables in that directory (one P + one S per associated station).
    Files expected per year: ``pyocto_<velmodel>_<year>.csv`` (event table) and
    ``pyocto_assignment_<velmodel>_<year>.csv`` (pick-to-event map). When ``None``, the
    legacy `associate_picks` time-window fallback is used (NOT recommended — see the
    docstring on `associate_picks` for the bug it caused).

    `skip_existing=True` (default): if ``<out_root>/<event_id>/`` already contains a SAC
    file, the row is skipped and recorded with status='skip_existing'. Re-running is
    therefore idempotent.

    `max_events`: limit (head); useful for a notebook dry-run."""
    os.makedirs(out_root, exist_ok=True)
    rows = catalog_df.head(max_events).to_dict("records") if max_events else \
        catalog_df.to_dict("records")
    if progress:
        try:
            from tqdm.auto import tqdm
            rows_iter = tqdm(rows, desc="events")
        except ImportError:
            rows_iter = rows
    else:
        rows_iter = rows

    # cache station tables by year (avoid re-reading per event)
    stations_for_year = {}

    out_rows = []
    for r in rows_iter:
        eid = r["event_id"]
        out_dir = os.path.join(out_root, eid)
        if skip_existing and os.path.isdir(out_dir) and any(
                f.endswith(".sac") for f in os.listdir(out_dir)):
            out_rows.append(dict(event_id=eid, n_sac=-1, n_stations_with_p=-1,
                                 n_stations_with_s=-1, n_stations_tried=-1,
                                 status="skip_existing"))
            continue
        yr = r["origin_utc"].year if isinstance(r["origin_utc"], UTCDateTime) \
            else pd.Timestamp(r["origin_utc"]).year
        if yr not in stations_for_year:
            stations_for_year[yr] = load_stations_for_year(station_table_root, yr)
        try:
            summary = export_event(r, picks_root, continuous_root,
                                   stations_for_year[yr], out_root,
                                   pre_s=pre_s, post_s=post_s, assoc_window=assoc_window,
                                   pyocto_root=pyocto_root,
                                   pyocto_velmodel=pyocto_velmodel,
                                   pyocto_tol_s=pyocto_tol_s, target_hz=target_hz)
        except Exception as exc:                # noqa: BLE001
            summary = dict(event_id=eid, n_sac=0, n_stations_with_p=0,
                           n_stations_with_s=0, n_stations_tried=0,
                           status=f"error: {type(exc).__name__}: {exc}",
                           pick_source="error")
        out_rows.append(summary)

    summary = pd.DataFrame(out_rows)
    if len(summary):
        summary.to_csv(os.path.join(out_root, "export_summary.csv"), index=False)
    return summary


# --- subregion filter + relocate -----------------------------------------------------
def select_subregion(catalog, region_bounds):
    """Return the rows of ``catalog`` whose epicentre falls within
    ``region_bounds = (lon0, lon1, lat0, lat1)``."""
    lon0, lon1, lat0, lat1 = region_bounds
    return catalog[catalog.lon.between(lon0, lon1)
                   & catalog.lat.between(lat0, lat1)].copy()


def move_events_in_region(catalog, src_root, dst_root, region_bounds,
                          *, action="move", dry_run=True) -> pd.DataFrame:
    """Filter ``catalog`` to ``region_bounds = (lon0, lon1, lat0, lat1)`` and **relocate**
    each matching ``<src_root>/<event_id>/`` directory to ``<dst_root>/<event_id>/``.

    `action` ∈ {"move", "copy", "symlink"}:
      - ``"move"`` (default): `shutil.move`, the source dir is gone afterwards. Use this
        when you want to shard the export by region without using extra disk.
      - ``"copy"``: `shutil.copytree`, keeps both copies (doubles disk usage).
      - ``"symlink"``: makes `<dst_root>/<event_id>` a symlink to the source — zero extra
        disk, but the link breaks if the source is later deleted.

    `dry_run=True` (default): reports what *would* be done without touching disk. Look at
    the returned DataFrame then re-call with `dry_run=False` to actually move.

    Returns one row per matching event: ``event_id, lon, lat, src_exists, dst_exists,
    action, status``."""
    import shutil
    sub = select_subregion(catalog, region_bounds)
    os.makedirs(dst_root, exist_ok=True) if not dry_run else None
    rows = []
    for r in sub.itertuples():
        src = os.path.join(src_root, r.event_id)
        dst = os.path.join(dst_root, r.event_id)
        src_exists = os.path.isdir(src)
        dst_exists = os.path.exists(dst)
        if not src_exists:
            status = "src_missing"
        elif dst_exists:
            status = "dst_exists"
        elif dry_run:
            status = f"would_{action}"
        else:
            try:
                if action == "move":
                    shutil.move(src, dst)
                elif action == "copy":
                    shutil.copytree(src, dst)
                elif action == "symlink":
                    os.symlink(os.path.abspath(src), dst)
                else:
                    raise ValueError(f"unknown action: {action}")
                status = "ok"
            except Exception as exc:            # noqa: BLE001
                status = f"error: {type(exc).__name__}: {exc}"
        rows.append(dict(event_id=r.event_id, lon=r.lon, lat=r.lat,
                         src_exists=src_exists, dst_exists=dst_exists,
                         action=action, status=status))
    return pd.DataFrame(rows)
