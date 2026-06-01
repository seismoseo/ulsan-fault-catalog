"""Local-magnitude pipeline for the Ulsan-Fault PhaseNet+ blast-clean catalog.

End-to-end chain that takes per-event SAC files (output of
[06.Export_event_waveforms_from_continuous.ipynb](../HypoInv/06.Export_event_waveforms_from_continuous.ipynb)),
removes instrument response, simulates a Wood-Anderson seismograph, picks the peak
amplitude, and applies a region-specific attenuation formula to yield ML per station-
channel. The notebook then aggregates (median across station-channels) to a single ML
per event.

Layout:
  responses/master/   — copy of the master StationXML covering 52 of 59 stations.
  responses/fetched/  — per-station StationXML for the 7 missing stations
                        (KS.BAEA, KS.DAJA, KS.GJAA, KS.HYDA, KS.NARA, KS.SRGA, KS.UICA),
                        produced by `Claude/fetch_responses.py`.

The attenuation function is **isolated as a single hot-swap point** (`ml_sheen2018`).
To switch to Heo et al. 2024 (or any other formula), add `ml_heo2024(...)` below and
change `ATT_FN = ml_sheen2018` to `ATT_FN = ml_heo2024` at the top of the notebook —
no other code changes.

Self-contained: nothing under `5.P_wave_moment_tensor_inversion/` or
`13.Local_magnitudes/` is read or imported. The master StationXML lives in
`responses/master/` (copied once at setup).
"""
from __future__ import annotations

import glob
import os
import warnings

import numpy as np
import obspy
import pandas as pd
from obspy.geodetics.base import gps2dist_azimuth


# Standard Wood-Anderson response parameters (Hutton & Boore 1987 / ObsPy convention).
WOOD_ANDERSON_PAZ = {
    "sensitivity": 2080.0,         # V·s/m at peak
    "zeros": [0j, 0j],             # double zero at origin
    "poles": [-6.283 + 4.7124j, -6.283 - 4.7124j],
    "gain": 1.0,                   # the sensitivity field carries the gain
}


# --- inventory loaders -----------------------------------------------------
def load_combined_inventory(master_dir, fetched_dir=None) -> obspy.Inventory:
    """Merge the master StationXML(s) and (optionally) the NECIS-fetched RESP files
    into a single `obspy.Inventory`. Both formats are auto-detected by obspy.

    Layout (matching what `Claude/fetch_responses.py` writes):
        master_dir/*.xml                                  ← master StationXML(s)
        fetched_dir/extracted/RESP.<NET>.<STA>..<CHAN>   ← SEED RESP files from NECIS

    For backward compatibility, also accepts:
        fetched_dir/*.xml                                 ← per-station StationXMLs

    The master is loaded first; the fetched files are appended after, so where both
    have the same (network, station) the master's entry wins."""
    inv = obspy.read_inventory(os.path.join(master_dir, "*.xml"))
    if fetched_dir and os.path.isdir(fetched_dir):
        # Priority: RESP files under extracted/, then any .xml at the root.
        # SEED RESP filenames begin with "RESP." and have no extension.
        targets = []
        ext_dir = os.path.join(fetched_dir, "extracted")
        if os.path.isdir(ext_dir):
            targets.extend(sorted(glob.glob(os.path.join(ext_dir, "RESP.*"))))
        targets.extend(sorted(glob.glob(os.path.join(fetched_dir, "*.xml"))))
        for f in targets:
            try:
                inv += obspy.read_inventory(f)
            except Exception as exc:                  # noqa: BLE001
                warnings.warn(f"failed to load {f}: {exc}", RuntimeWarning)
    return inv


def report_coverage(stations_needed, inventory) -> pd.DataFrame:
    """For each ``(network, station)`` in `stations_needed`, report whether the inventory
    has a matching entry. `stations_needed` can be an iterable of ``"NET.STA"`` strings
    or a list of ``(network, station)`` tuples.

    Returns a DataFrame with columns ``network, station, covered, n_channels``. Use this
    at the top of a notebook to spot missing stations before kicking off the chain."""
    pairs = []
    for s in stations_needed:
        if isinstance(s, str):
            net, sta = s.split(".", 1)
        else:
            net, sta = s
        pairs.append((net, sta))
    have = {(n.code, st.code): len(st) for n in inventory for st in n}
    rows = []
    for net, sta in pairs:
        n_ch = have.get((net, sta), 0)
        rows.append(dict(network=net, station=sta, covered=n_ch > 0, n_channels=n_ch))
    return pd.DataFrame(rows)


# --- response removal + Wood-Anderson sim ----------------------------------
def remove_response_to_disp(stream, inventory, *,
                            pre_filt=(0.001, 0.005, 50, 100),
                            water_level=60.0,
                            taper_percent=0.05) -> obspy.Stream:
    """Detrend(demean) → cosine taper → ``remove_response(output='DISP', water_level=…)``
    → detrend(demean) → taper. Returns a NEW stream (original untouched). Skips traces
    whose ``(network, station, channel)`` is not in the inventory (with a single
    `warnings.warn` per missing key — same warning is not repeated for the 3 components
    of the same station).

    `pre_filt` is the cosine-tapered bandpass that brackets the response deconvolution;
    `water_level` (dB) stabilises the deconvolution near response zeros. Defaults match
    the user's prior workflow and Sheen et al. 2018 conventions."""
    out = obspy.Stream()
    warned = set()
    for tr in stream:
        key = (tr.stats.network, tr.stats.station, tr.stats.channel)
        try:
            inventory.get_response(f"{key[0]}.{key[1]}..{key[2]}", tr.stats.starttime)
        except Exception:                             # noqa: BLE001
            short = (key[0], key[1])
            if short not in warned:
                warnings.warn(f"no response for {key[0]}.{key[1]} — skipping all "
                              f"channels of this station", RuntimeWarning)
                warned.add(short)
            continue
        work = tr.copy().detrend("demean").taper(taper_percent)
        try:
            work.remove_response(inventory=inventory, output="DISP",
                                 pre_filt=list(pre_filt), water_level=water_level,
                                 plot=False)
        except Exception as exc:                      # noqa: BLE001
            warnings.warn(f"remove_response failed for {tr.id}: {exc}", RuntimeWarning)
            continue
        work.detrend("demean").taper(taper_percent)
        out += work
    return out


def wood_anderson_amp_mm(stream_disp) -> pd.DataFrame:
    """Simulate the standard Wood-Anderson seismograph from a displacement stream
    (output of `remove_response_to_disp`) and return peak amplitude in millimetres.

    Returns one row per trace: ``network, station, channel, peak_mm, peak_time``."""
    rows = []
    for tr in stream_disp:
        wa = tr.copy().simulate(paz_simulate=WOOD_ANDERSON_PAZ)
        # paz_simulate gives output in metres; convert to mm.
        a = np.abs(wa.data)
        if not len(a):
            continue
        i = int(np.argmax(a))
        rows.append(dict(network=tr.stats.network, station=tr.stats.station,
                         channel=tr.stats.channel,
                         peak_mm=float(a[i]) * 1000.0,
                         peak_time=wa.stats.starttime + i * wa.stats.delta))
    return pd.DataFrame(rows)


# --- ATTENUATION (swap-in point for Heo et al. 2024) -----------------------
def ml_sheen2018(amp_mm, distance_km, component):
    """South Korea local-magnitude attenuation per Sheen, Kang & Rhie (2018), BSSA
    108(5A), 2748–2755. `amp_mm` is the peak Wood-Anderson amplitude in millimetres;
    `distance_km` is the hypocentral distance in km; `component` is one of
    ``'Z'`` / ``'N'`` / ``'E'`` (vertical vs horizontal coefficients).

    Returns ML (float). Vectorised over array inputs.

    Formula (one per orientation):
        Z   :  ML = log10(A) + 0.5107·log10(R/100) + 0.001699·(R-100) + 3
        N,E :  ML = log10(A) + 0.5869·log10(R/100) + 0.001680·(R-100) + 3
    """
    amp_mm = np.asarray(amp_mm, dtype=float)
    R = np.asarray(distance_km, dtype=float)
    if component == "Z":
        return (np.log10(amp_mm) + 0.5107 * np.log10(R / 100.0)
                + 0.001699 * (R - 100.0) + 3.0)
    elif component in ("N", "E", "1", "2"):
        return (np.log10(amp_mm) + 0.5869 * np.log10(R / 100.0)
                + 0.001680 * (R - 100.0) + 3.0)
    else:
        raise ValueError(f"unknown component {component!r} (expected Z/N/E/1/2)")


def ml_heo2024(amp_mm, distance_km, component=None):
    """Southeastern Korean Peninsula local-magnitude attenuation per **Heo et al.
    (2024)**, Geosciences Journal 28(1), "New insights on seismic activity in the
    southeastern Korean Peninsula from the Gyeongju Hi-density Broadband Seismic
    Network (GHBSN)" (https://doi.org/10.1007/s12303-024-0003-7).

    Calibrated on **2,860 micro-earthquakes** detected by GHBSN's 200-station
    array around the 2016 Gyeongju epicenter. Uses the Hutton-Boore (1987) form
    with a 17 km reference distance. The reported coefficients (their Eq. 3):

        −log10 A0 = 1.450676 · log10(R/17) − 0.000661 · (R − 17) + 2.0

    so

        ML = log10(A_mm) + 1.450676 · log10(R/17) − 0.000661 · (R − 17) + 2.0

    `amp_mm` is the peak Wood-Anderson amplitude in millimetres; `distance_km`
    is the hypocentral distance (km). Heo et al. calibrated on the **vertical
    component only**; `component` is accepted for signature compatibility with
    `ml_sheen2018` but does not affect the result. The notebook's per-station
    loop should be filtered to Z channels when using this formula for
    maximum-fidelity reproduction. Station corrections (their S term per
    GHBSN site) are not applied here — set them to zero for events outside
    the GHBSN, or extend this function with a per-station table if you have one.

    Vectorised over array inputs. Returns ML (float)."""
    amp_mm = np.asarray(amp_mm, dtype=float)
    R = np.asarray(distance_km, dtype=float)
    return (np.log10(amp_mm)
            + 1.450676 * np.log10(R / 17.0)
            - 0.000661 * (R - 17.0)
            + 2.0)


# To swap from Sheen 2018 → Heo 2024 in the notebook, change one line:
#   ATT_FN = ml_heo2024     # was ml_sheen2018
# No other code changes. The §5 benchmark cell re-runs with the new formula and
# produces a fresh residual column for direct comparison.


# --- per-station ML aggregator ---------------------------------------------
def per_station_ml(event_dir, inventory, *, attenuation_fn=ml_sheen2018,
                   pre_filt=(0.001, 0.005, 50, 100), water_level=60.0) -> pd.DataFrame:
    """For one event directory (per the export tree), compute ML at every station-channel.

    Reads all `*.sac` files in `event_dir` (expected layout: one per station-channel,
    written by the prior export notebook). Distance comes from the SAC `dist` header
    (km, populated at export). Returns a per-trace DataFrame:
        network, station, channel, dist_km, peak_mm, peak_time, ML.

    Tip: the calling notebook then aggregates with `df.groupby('station').ML.median()`
    to get one ML per station, and `df.ML.median()` for the event-level ML."""
    sacs = sorted(glob.glob(os.path.join(event_dir, "*.sac")))
    if not sacs:
        return pd.DataFrame()
    st = obspy.Stream()
    dist_for = {}
    for f in sacs:
        tr = obspy.read(f)[0]
        st += tr
        sac = tr.stats.sac
        if "dist" in sac:
            dist_for[(tr.stats.network, tr.stats.station, tr.stats.channel)] = float(sac.dist)
    disp = remove_response_to_disp(st, inventory, pre_filt=pre_filt, water_level=water_level)
    if not len(disp):
        return pd.DataFrame()
    amps = wood_anderson_amp_mm(disp)
    if not len(amps):
        return pd.DataFrame()
    amps["dist_km"] = [dist_for.get((r.network, r.station, r.channel), np.nan)
                       for r in amps.itertuples()]
    # Use only the channel orientation code (last character) for attenuation_fn
    amps["ML"] = [attenuation_fn(r.peak_mm, r.dist_km, r.channel[-1])
                  if r.peak_mm > 0 and np.isfinite(r.dist_km) else np.nan
                  for r in amps.itertuples()]
    return amps[["network", "station", "channel", "dist_km", "peak_mm", "peak_time", "ML"]]


def aggregate_ml(per_station_df) -> dict:
    """Aggregate per-station MLs into a single event ML. Drops NaNs, takes the median
    across all available station-channels. Returns a dict with ``ml_median``,
    ``ml_mean``, ``ml_std``, ``n_used``."""
    if not len(per_station_df) or per_station_df["ML"].notna().sum() == 0:
        return dict(ml_median=np.nan, ml_mean=np.nan, ml_std=np.nan, n_used=0)
    s = per_station_df["ML"].dropna()
    return dict(ml_median=float(s.median()), ml_mean=float(s.mean()),
                ml_std=float(s.std()), n_used=int(len(s)))
