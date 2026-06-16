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
                            pre_filt=(1.0, 2.0, 20.0, 22.0),
                            water_level=60.0,
                            taper_percent=0.05) -> obspy.Stream:
    """Detrend(demean + linear) → cosine taper → ``remove_response(output='DISP',
    water_level=…)`` (which applies `pre_filt` as a cosine-taper bandpass on the
    way to deconvolution) → detrend(demean) → taper. Returns a NEW stream
    (original untouched). Skips traces whose ``(network, station, channel)`` is
    not in the inventory (single warning per missing station).

    Preprocessing matches **Heo et al. (2024) §4.2** exactly:
      * demean + detrend(linear) before response removal,
      * cosine-tapered bandpass at **2–22 Hz** (flat 2–20 Hz, cosine ramps
        1→2 Hz and 20→22 Hz) — Heo specifies "bandpass filter at 2–20 Hz",
      * water_level=60 dB to stabilise the deconvolution,
      * the WA convolution is the next step (in `wood_anderson_amp_mm`).

    The legacy default was `pre_filt=(0.001, 0.005, 50, 100)` (0.005–50 Hz
    cosine), too broad for ML on micro-earthquakes — included long-period
    noise that gets amplified by the deconvolution at the response zeros."""
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
        work = tr.copy().detrend("demean").detrend("linear").taper(taper_percent)
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


def wood_anderson_amp_mm(stream_disp, *,
                         require_pick=True,
                         noise_pre_pad=1.0,
                         noise_fallback_s=20.0,
                         signal_min_after_p=0.5) -> pd.DataFrame:
    """Simulate the standard Wood-Anderson seismograph from a displacement stream
    (output of `remove_response_to_disp`) and return peak amplitude in mm
    **together with a pre-event noise RMS and the resulting SNR**.

    Windowing:
      * **noise window** : from trace start to ``(P_pick − noise_pre_pad) s``.
      * **signal window**: from ``(P_pick + signal_min_after_p) s`` to the end
        of the trace. The peak Wood-Anderson amplitude is the max of |WA| inside
        this window — restricting the search to post-P avoids picking pre-event
        noise spikes as the "peak" (the v1.0.0 behaviour, which inflated small-
        event MLs and biased the b-value upward).

    ``require_pick`` (default True, the CORRECT behaviour) skips any trace that has
    no P-pick header (`SAC.a`). A station with no pick is one where the phase was
    not detected (signal not above noise), so it must not contribute: without a pick
    the noise window has no real anchor and falls back to the trace start, the SNR
    test is meaningless, and unpicked far stations sitting at the ambient/coda
    amplitude FLOOR leak in. Their amplitude is flat with distance, so the distance
    term over-corrects them up to station ML ~1 and — being the majority for small
    events — saturates small-event ML / inflates Mc (the Buan bug, commit fd7800a).
    Pass ``require_pick=False`` ONLY to reproduce the buggy all-station result for a
    diagnostic; the ``noise_fallback_s`` window is used only on that path.

    Returns one row per trace with:
        network, station, channel, peak_mm, peak_time, noise_mm, snr.

    `snr` is the dimensionless ratio ``peak_mm / noise_mm`` (uses the noise RMS
    converted to mm in the same WA-simulated trace). Use ``snr_threshold`` in
    the higher-level `per_station_ml` to drop sub-threshold rows before
    aggregation."""
    rows = []
    for tr in stream_disp:
        # Detectability gate: the phase must have been picked HERE (SAC.a set), else
        # the SNR test below has no real pre-P anchor — skip it (see require_pick).
        sac = getattr(tr.stats, "sac", {}) or {}
        p_rel = None
        if "a" in sac and float(sac["a"]) > -1e4:                     # SAC -12345 sentinel
            p_rel = float(sac["a"])
        if require_pick and p_rel is None:
            continue
        wa = tr.copy().simulate(paz_simulate=WOOD_ANDERSON_PAZ)
        data_mm = np.asarray(wa.data, dtype=float) * 1000.0          # m → mm
        if not len(data_mm):
            continue
        sr = float(wa.stats.sampling_rate)
        # Noise window (sample indices) — pre-P, with a small pad
        if p_rel is not None and p_rel > noise_pre_pad + 0.5:
            i_noise_end = int((p_rel - noise_pre_pad) * sr)
        else:
            i_noise_end = int(noise_fallback_s * sr)
        i_noise_end = max(min(i_noise_end, len(data_mm)), 4)
        noise_window = data_mm[:i_noise_end]
        noise_mm = float(np.sqrt(np.mean(noise_window ** 2))) if len(noise_window) else np.nan
        # Signal window — post-P only when we have a pick, else from end-of-noise
        if p_rel is not None:
            i_sig_start = max(int((p_rel + signal_min_after_p) * sr), i_noise_end)
        else:
            i_sig_start = i_noise_end
        if i_sig_start >= len(data_mm) - 1:
            continue
        signal_window = data_mm[i_sig_start:]
        i_local = int(np.argmax(np.abs(signal_window)))
        peak_mm = float(np.abs(signal_window[i_local]))
        peak_idx = i_sig_start + i_local
        snr = peak_mm / noise_mm if noise_mm and noise_mm > 0 else np.nan
        rows.append(dict(network=tr.stats.network, station=tr.stats.station,
                         channel=tr.stats.channel,
                         peak_mm=peak_mm,
                         peak_time=wa.stats.starttime + peak_idx * wa.stats.delta,
                         noise_mm=noise_mm,
                         snr=snr))
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
def _auto_pre_filt(attenuation_fn):
    """Auto-resolve the response-removal pre_filt bandpass for the chosen attenuation
    formula. Each tuple matches the original calibration paper's preprocessing:

      * Heo 2024 §4.2 — strict 2–20 Hz cosine bandpass (micro-event spectrum).
      * Sheen 2018 §Data and Preliminary Analysis (p. 2749) — "filtered with a
        0.5–10 Hz, six-pole Butterworth band-pass filter to suppress microseismic
        noise." We use a cosine-tapered (0.3, 0.5, 10.0, 12.0) pre_filt in the
        response-removal step to approximate Sheen's 0.5–10 Hz passband; the cosine
        taper rolloff at the band edges is gentler than a 6-pole Butterworth but
        the in-band response is flat to within ~0.5 dB, which is well below the
        Wood-Anderson + amplitude-measurement noise floor.

    Returns a 4-tuple suitable for ``ObsPy.remove_response(pre_filt=…)``."""
    if attenuation_fn is ml_heo2024:
        return (1.0, 2.0, 20.0, 22.0)     # Heo 2024 §4.2: 2–20 Hz cosine bandpass
    if attenuation_fn is ml_sheen2018:
        return (0.3, 0.5, 10.0, 12.0)     # Sheen 2018: 0.5–10 Hz (paper §Data p.2749)
    # Unknown formula: default to Heo's tight band (safe for micro-events)
    return (1.0, 2.0, 20.0, 22.0)


def per_station_ml(event_dir, inventory, *, attenuation_fn=ml_sheen2018,
                   pre_filt=None, water_level=60.0,
                   snr_threshold=3.0, z_only=None, require_pick=True) -> pd.DataFrame:
    """For one event directory (per the export tree), compute ML at every station-channel.

    Reads all `*.sac` files in `event_dir` (expected layout: one per station-channel,
    written by the prior export notebook). Distance comes from the SAC `dist` header
    (km, populated at export).

    SNR filtering: each trace's peak Wood-Anderson amplitude is divided by the
    pre-P-pick RMS noise on the same trace (post-deconvolution); rows with
    ``snr < snr_threshold`` get ``ML = NaN`` and are dropped from the event-level
    aggregate (`aggregate_ml`). Default ``snr_threshold = 3.0`` follows Sheen et
    al. 2018 / Heo et al. 2024 convention. Set to 0 to keep all stations
    (legacy v1.0.0 behaviour — which inflates ML for noise-dominated traces).

    Returns one row per station-channel:
        network, station, channel, dist_km, peak_mm, noise_mm, snr, peak_time, ML.

    Tip: the calling notebook then aggregates with `df.groupby('station').ML.median()`
    to get one ML per station, and `df.ML.median()` for the event-level ML."""
    # Heo 2024 calibrated on vertical-component peaks only; honour that automatically
    # whenever the attenuation_fn is `ml_heo2024` (user can pass z_only=False to opt out).
    if z_only is None:
        z_only = (attenuation_fn is ml_heo2024)
    # Heo 2024 vs Sheen 2018 need different bandpasses; auto-resolve if not specified.
    if pre_filt is None:
        pre_filt = _auto_pre_filt(attenuation_fn)
    sacs = sorted(glob.glob(os.path.join(event_dir, "*.sac")))
    if not sacs:
        return pd.DataFrame()
    st = obspy.Stream()
    dist_for = {}
    for f in sacs:
        tr = obspy.read(f)[0]
        # Honour z_only EARLY — skip non-vertical traces before response removal so we
        # don't waste deconvolution work on traces we'll throw away.
        if z_only and not tr.stats.channel.endswith("Z"):
            continue
        st += tr
        sac = tr.stats.sac
        if "dist" in sac:
            dist_for[(tr.stats.network, tr.stats.station, tr.stats.channel)] = float(sac.dist)
    disp = remove_response_to_disp(st, inventory, pre_filt=pre_filt, water_level=water_level)
    if not len(disp):
        return pd.DataFrame()
    amps = wood_anderson_amp_mm(disp, require_pick=require_pick)
    if not len(amps):
        return pd.DataFrame()
    amps["dist_km"] = [dist_for.get((r.network, r.station, r.channel), np.nan)
                       for r in amps.itertuples()]
    # Use only the channel orientation code (last character) for attenuation_fn.
    # Skip ML when SNR is below the threshold (returns NaN; aggregate_ml drops NaNs).
    def _ml_one(r):
        if r.peak_mm <= 0 or not np.isfinite(r.dist_km):
            return np.nan
        if snr_threshold > 0 and (not np.isfinite(r.snr) or r.snr < snr_threshold):
            return np.nan
        return attenuation_fn(r.peak_mm, r.dist_km, r.channel[-1])
    amps["ML"] = [_ml_one(r) for r in amps.itertuples()]
    return amps[["network", "station", "channel", "dist_km",
                 "peak_mm", "noise_mm", "snr", "peak_time", "ML"]]


def aggregate_ml(per_station_df) -> dict:
    """Aggregate per-station MLs into a single event ML. Drops NaNs (SNR-filtered
    rows from `per_station_ml`), takes the median across all surviving station-
    channels. Returns ``ml_median``, ``ml_mean``, ``ml_std``, ``n_used``,
    plus ``n_total`` (rows before SNR filtering) and ``snr_median`` so callers
    can see how many stations were dropped vs how many made it into the median."""
    if not len(per_station_df):
        return dict(ml_median=np.nan, ml_mean=np.nan, ml_std=np.nan,
                    n_used=0, n_total=0, snr_median=np.nan)
    n_total = int(len(per_station_df))
    snr_med = float(per_station_df["snr"].dropna().median()) \
              if "snr" in per_station_df.columns and per_station_df["snr"].notna().any() \
              else float("nan")
    s = per_station_df["ML"].dropna()
    if not len(s):
        return dict(ml_median=np.nan, ml_mean=np.nan, ml_std=np.nan,
                    n_used=0, n_total=n_total, snr_median=snr_med)
    return dict(ml_median=float(s.median()), ml_mean=float(s.mean()),
                ml_std=float(s.std()), n_used=int(len(s)),
                n_total=n_total, snr_median=snr_med)


# --- bulk-catalog ML computation ------------------------------------------
def _event_dir_for(time_str, event_roots) -> str | None:
    """Locate the event SAC directory matching ``time_str`` (the catalog ``time`` column,
    'YYYY-MM-DD HH:MM:SS.ms'). Tries each root in ``event_roots`` in order; returns the
    first existing directory or None. The export convention names directories as
    YYYYMMDDHHMMSS (no separators, no sub-second precision)."""
    import re
    # 2010-01-10 00:14:59.580 → 20100110001459
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})", str(time_str))
    if not m:
        return None
    stem = "".join(m.groups())
    for root in event_roots:
        d = os.path.join(root, stem)
        if os.path.isdir(d):
            return d
    return None


def export_ml_catalog(catalog_df, event_roots, inventory, *,
                      attenuation_fn=None, restrict_to_z=None,
                      pre_filt=None,
                      snr_threshold=3.0, require_pick=True,
                      workers=1, skip_existing=True, out_path=None,
                      per_station_csv_path=None,
                      progress=True) -> pd.DataFrame:
    """Compute event-level ML for every row of ``catalog_df`` and return the catalog
    with four new columns:

      * ``magnitude``       — median ML across station-channels (SeismoStats convention).
      * ``magnitude_std``   — scatter (std) of the per-station MLs.
      * ``n_used``          — number of station-channels that contributed.
      * ``mag_status``      — ``"ok"`` | ``"no_dir"`` | ``"no_picks"`` | ``"error:<msg>"``.

    Why ``magnitude`` (not ``ml``)?  So the augmented CSV loads directly into
    ``seismostats.Catalog`` whose magnitude-aware methods (``estimate_b``,
    ``estimate_mc_maxc``, ``plot_fmd``, …) expect that column name.

    Parameters
    ----------
    catalog_df : pd.DataFrame
        The HypoInverse catalog. Must have a ``time`` column (string or UTCDateTime-
        formattable). Other columns are passed through.
    event_roots : Sequence[str]
        Ordered list of root directories that hold per-event SAC trees, e.g.
        ``("event_waveforms_blastclean", "event_waveforms_ulsanfault")``. The first
        root that contains a matching ``YYYYMMDDHHMMSS/`` dir wins.
    inventory : obspy.Inventory
        Merged inventory from ``load_combined_inventory()``.
    attenuation_fn : callable, default ``ml_heo2024``
        Distance-attenuation function ``(amp_mm, dist_km, component) -> ML``. The
        notebook defaults to ``ml_heo2024`` (Southeastern Korea, GHBSN-calibrated).
    restrict_to_z : bool | None, default None (auto)
        If True, drop non-vertical channels before aggregating — matches Heo et
        al. (2024) methodology. Default ``None`` auto-resolves: **True when
        attenuation_fn is ml_heo2024** (Heo calibrated on Z-only), False for
        Sheen 2018 (calibrated on all 3 components). Pass an explicit bool to
        override the auto-resolution.
    workers : int, default 1
        ThreadPool worker count for the per-event loop. obspy I/O + numpy releases
        the GIL so threads parallelise well; processes would re-load the inventory
        per worker (50+ MB) and hurt total memory. Set to ``os.cpu_count() // 2``
        for bulk runs.
    skip_existing : bool, default True
        If ``out_path`` exists and already has a ``magnitude`` column, restart from
        the first NaN row. Makes interrupted runs resumable.
    out_path : str | None
        If set, the augmented DataFrame is written here after EVERY chunk of 200
        events so a Ctrl-C / SIGKILL leaves a partial-but-valid CSV behind.
    progress : bool, default True
        Show a tqdm bar.

    Returns
    -------
    pd.DataFrame
        Copy of ``catalog_df`` with the 4 magnitude columns appended.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if attenuation_fn is None:
        attenuation_fn = ml_heo2024
    # Auto-resolve Z-only based on the attenuation formula (Heo 2024 → Z-only).
    if restrict_to_z is None:
        restrict_to_z = (attenuation_fn is ml_heo2024)
    # Auto-resolve pre_filt bandpass for the chosen attenuation formula.
    if pre_filt is None:
        pre_filt = _auto_pre_filt(attenuation_fn)

    out = catalog_df.copy()
    for col, default in (("magnitude", np.nan), ("magnitude_std", np.nan),
                         ("n_used", 0), ("n_total", 0),
                         ("snr_median", np.nan), ("mag_status", "")):
        if col not in out.columns:
            out[col] = default

    # Resume from a partial file if asked.
    if skip_existing and out_path and os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        if "magnitude" in prev.columns and len(prev) == len(out):
            out["magnitude"]     = prev["magnitude"].to_numpy()
            out["magnitude_std"] = prev["magnitude_std"].to_numpy() if "magnitude_std" in prev else np.nan
            out["n_used"]        = prev["n_used"].to_numpy()        if "n_used"        in prev else 0
            out["mag_status"]    = prev["mag_status"].fillna("").astype(str).to_numpy() \
                                      if "mag_status" in prev else ""

    todo_idx = [i for i in out.index if not str(out.at[i, "mag_status"])]
    if progress:
        try:
            from tqdm.auto import tqdm
            todo_iter = tqdm(todo_idx, desc="events", unit="evt")
        except ImportError:
            todo_iter = todo_idx
    else:
        todo_iter = todo_idx

    # Per-station rows accumulator (one row per event-station-channel) — flushed
    # periodically to per_station_csv_path when set, so a kill leaves a valid CSV.
    # Kept thread-safe via a lock; workers append, the flusher reads.
    from threading import Lock
    _ps_rows: list[dict] = []
    _ps_lock = Lock()

    def _compute_one(i):
        ev_dir = _event_dir_for(out.at[i, "time"], event_roots)
        if ev_dir is None:
            return i, np.nan, np.nan, 0, 0, np.nan, "no_dir"
        try:
            per_sta = per_station_ml(ev_dir, inventory,
                                     attenuation_fn=attenuation_fn,
                                     pre_filt=pre_filt,
                                     snr_threshold=snr_threshold,
                                     z_only=restrict_to_z,
                                     require_pick=require_pick)
            # Persist per-station rows when requested. The ML column is the post-SNR
            # value (NaN when SNR < threshold); downstream consumers can compute the
            # correction from any non-NaN subset.
            if per_station_csv_path is not None and len(per_sta):
                ev_time = str(out.at[i, "time"])
                rows = per_sta.assign(event_idx=int(i), event_time=ev_time).to_dict("records")
                with _ps_lock:
                    _ps_rows.extend(rows)
            agg = aggregate_ml(per_sta)
            if agg["n_used"] == 0:
                status = "no_picks" if agg["n_total"] == 0 else "low_snr"
                return (i, np.nan, np.nan, 0, agg["n_total"], agg["snr_median"], status)
            return (i, agg["ml_median"], agg["ml_std"], agg["n_used"],
                    agg["n_total"], agg["snr_median"], "ok")
        except Exception as exc:                          # noqa: BLE001
            return i, np.nan, np.nan, 0, 0, np.nan, f"error:{type(exc).__name__}"

    def _flush():
        if out_path:
            out.to_csv(out_path, index=False)
        # Stream the per-station accumulator to the side CSV (append-mode after
        # the first flush so we don't re-write 1M+ rows every chunk).
        if per_station_csv_path is not None:
            with _ps_lock:
                if _ps_rows:
                    first = not os.path.exists(per_station_csv_path)
                    pd.DataFrame(_ps_rows).to_csv(per_station_csv_path,
                                                  mode="a" if not first else "w",
                                                  header=first, index=False)
                    _ps_rows.clear()

    def _apply(idx, ml, std, n, n_tot, snr_med, st):
        out.at[idx, "magnitude"]     = ml
        out.at[idx, "magnitude_std"] = std
        out.at[idx, "n_used"]        = n
        out.at[idx, "n_total"]       = n_tot
        out.at[idx, "snr_median"]    = snr_med
        out.at[idx, "mag_status"]    = st

    if workers <= 1:
        for i in todo_iter:
            _apply(*_compute_one(i))
            if (i + 1) % 200 == 0:
                _flush()
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_compute_one, i): i for i in todo_idx}
            done = 0
            for fut in as_completed(futs):
                _apply(*fut.result())
                done += 1
                if progress and hasattr(todo_iter, "update"):
                    todo_iter.update(1)
                if done % 200 == 0:
                    _flush()
    _flush()
    if hasattr(todo_iter, "close"):
        todo_iter.close()
    return out


# --- station-correction helpers (Heo 2024 §4.2 S term, estimated from this catalog) ----
def estimate_station_corrections(per_station_csv: str, *, min_events: int = 30,
                                 use_z_only: bool | None = None) -> pd.DataFrame:
    """Estimate per-station magnitude corrections **S_j** from a per-station ML CSV
    produced by ``export_ml_catalog(..., per_station_csv_path=...)``.

    Method (matches Heo et al. 2024 § 4.2): for each event ``i`` and station-channel
    ``j``, compute the residual ``r_{i,j} = ML_{i,j} − median_j(ML_{i,j})`` (the
    deviation of the station's ML from the event-median). The station correction is
    then ``S_j = median_i(r_{i,j})`` — the systematic bias of station j across all
    events it observed. Stations with fewer than ``min_events`` contributing events
    get ``S_j = 0`` (insufficient stats; keep neutral so we don't introduce noise).

    Parameters
    ----------
    per_station_csv : str
        Path to the per-station ML CSV written by ``export_ml_catalog``. Must have
        columns ``event_idx, network, station, channel, ML``.
    min_events : int, default 30
        Stations with fewer than this many event-contributions get S_j = 0.
    use_z_only : bool | None
        Filter to Z-only channels (``channel.endswith('Z')``) before estimation.
        ``None`` (default) keeps whatever channels the CSV already has — appropriate
        when the upstream ``per_station_ml`` already filtered (Heo's ``z_only=True``).

    Returns
    -------
    pd.DataFrame
        Columns: ``station_key`` (NET.STA.CHAN), ``S_j``, ``n_events`` — one row per
        unique station-channel that contributed to any event. Sorted by ``S_j``.
    """
    psm = pd.read_csv(per_station_csv)
    if "ML" not in psm.columns:
        raise ValueError(f"per_station_csv {per_station_csv} missing 'ML' column")
    psm = psm.dropna(subset=["ML"]).copy()
    if use_z_only:
        psm = psm[psm["channel"].str.endswith("Z")]
    # Event-median ML (over all stations contributing to event i)
    ev_med = psm.groupby("event_idx")["ML"].transform("median")
    psm["residual"] = psm["ML"] - ev_med
    # Station-key: NET.STA.CHAN to disambiguate same-name stations across networks
    psm["station_key"] = (psm["network"].astype(str) + "." +
                          psm["station"].astype(str) + "." +
                          psm["channel"].astype(str))
    out = (psm.groupby("station_key")
              .agg(S_j=("residual", "median"), n_events=("residual", "size"))
              .reset_index())
    # Stations with too few observations → neutral correction (don't add noise)
    out.loc[out["n_events"] < min_events, "S_j"] = 0.0
    return out.sort_values("S_j").reset_index(drop=True)


def apply_station_corrections(per_station_csv: str,
                              station_corrections: pd.DataFrame) -> pd.DataFrame:
    """Re-aggregate event MLs after subtracting per-station S_j corrections.

    Produces a per-event DataFrame with ``magnitude_corr``, ``magnitude_std_corr``,
    ``n_used_corr`` columns. Combine with the original catalog by ``event_idx``.

    Parameters
    ----------
    per_station_csv : str
        Same CSV as for ``estimate_station_corrections``.
    station_corrections : pd.DataFrame
        Output of ``estimate_station_corrections`` — must have ``station_key`` and
        ``S_j`` columns. Missing stations get S_j = 0.

    Returns
    -------
    pd.DataFrame
        Per-event aggregates with the corrected magnitudes. Index = event_idx.
    """
    psm = pd.read_csv(per_station_csv).dropna(subset=["ML"]).copy()
    psm["station_key"] = (psm["network"].astype(str) + "." +
                          psm["station"].astype(str) + "." +
                          psm["channel"].astype(str))
    corr_map = dict(zip(station_corrections["station_key"],
                        station_corrections["S_j"]))
    psm["S_j"] = psm["station_key"].map(corr_map).fillna(0.0)
    psm["ML_corr"] = psm["ML"] - psm["S_j"]
    # Per-event aggregation
    grp = psm.groupby("event_idx")["ML_corr"]
    return pd.DataFrame({
        "magnitude_corr": grp.median(),
        "magnitude_std_corr": grp.std(),
        "n_used_corr": grp.size(),
    })
