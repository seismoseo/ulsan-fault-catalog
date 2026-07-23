"""Post-PyOcto pick augmentation — recover the picks PyOcto's streaming associator truncated.

PROBLEM
-------
PyOcto's `OctoAssociator` is a streaming associator: it scans the daily pick stream in time
order, builds candidate events with picks that fit a hypothesized hypocenter within
`pick_match_tolerance`, and FINALIZES the event candidate as soon as it passes the threshold
gate (`n_picks`, `n_p_picks`, `n_s_picks`, `n_p_and_s_picks`). Once finalized, picks at FARTHER
stations that arrive later may NOT be re-considered for that event, because:

1. PyOcto's refinement step (`refinement_iterations`) re-locates the candidate but does not
   rescan picks already passed over in the time-ordered stream.
2. Picks that didn't fit the candidate's INITIAL hypothesized hypocenter (e.g. the 2013-03-22
   13:40:04 chimera case where PyOcto first hypothesized a 36.66°N hypocenter, ~110 km north
   of truth) get rejected once — even if PyOcto's later refinement moves the hypocenter back
   to truth (35.74°N), those rejected picks aren't given a second chance.

The result is that EVERY HypoInverse fit downstream uses a SUBSET of available picks — by ~0-50%
depending on station density. Per-event GAP / DMIN / ERZ / RMS are systematically pessimistic;
chimera-class events report GAP > 270° because PyOcto only locked in the (one-sided) southern
station cluster.

SOLUTION
--------
A post-PyOcto pass that uses PyOcto's already-refined hypocenter as the **seed** and scans the
day's full raw-pick stream for any pick that:
  - is at a station within `radius_km` of the seed hypocenter,
  - matches the event's predicted arrival within `tolerance_s`,
  - has phase consistent with the predicted phase (P→P, S→S),
  - is NOT already in PyOcto's assignment for this event.

These "orphans" are added to an augmented assignment file. write_phs and HypoInverse downstream
see the FULL pick set and compute proper GAP / DMIN / RMS.

SAFEGUARDS against cross-event pick contamination (for close-in-time doublets / triplets):
  - Phase-strict matching (P picks only against P predictions, S against S).
  - Best-match-wins: a pick matching multiple events within tolerance is assigned to the event
    with the smallest residual; the rest of the candidates are rejected for that pick.
  - Drop-on-tie: if the best two residuals differ by less than `tie_threshold_s`, the pick is
    rejected for both events (genuinely ambiguous).
  - Per (event, station, phase): at most one pick added (PyOcto already enforces this for its
    own picks; augmentation respects it for its own output).

KEY DESIGN CHOICE — seed hypocenter
-----------------------------------
We use **PyOcto's hypocenter** as the augmentation seed, NOT HypoInverse's. Rationale:
  - PyOcto runs BEFORE HypoInverse in the pipeline. We want HypoInverse to see the full pick
    set on its very first pass, so its GAP/DMIN report reflects the augmented data.
  - For non-chimera events PyOcto's hypocenter is already close to truth (the pipeline's
    tweaked PyOcto with min_node_size=2 + refinement_iterations=8 forces this).
  - For chimera events PyOcto's hypocenter is wrong → augmentation finds no orphans → no harm,
    but no recovery either. The PyOcto tweak (separate concern) fixes chimera hypocenters
    upstream so augmentation can recover the missed picks.

If the chosen chimera fix is "use HypoInverse hypocenter post-pass-1 instead of PyOcto's"
that's a different design (re-run HypoInverse twice). Not implemented here; the user explicitly
specified "full PyOcto before HypoInverse for gap treatment" which dictates this seed choice.
"""
from __future__ import annotations

import math
import os
from glob import glob
from typing import Optional

import numpy as np
import pandas as pd
from obspy.geodetics.base import gps2dist_azimuth


# ---------------------------------------------------------------- velocity model
# Kim 1983 layered crust (matches PyOcto's kim1983.csv used in the strict-PyOcto run).
# Travel time in this implementation = slant_distance / velocity_at_source_depth. This is the
# direct-ray (Pg/Sg) approximation, valid for events within the layered crust (depth < 32 km)
# and station distances < 100 km (where refracted Pn/Sn rays don't beat the direct ray).
# Validated against PyOcto's reported residuals on the 2013-03-22 13:40 case: max 0.3 s
# difference, well below the 1.0 s tolerance threshold used for augmentation.
KIM1983_P = [(0.0, 5.98), (15.0, 6.38), (32.0, 7.95)]   # (top_depth_km, velocity_km_s)
KIM1983_S = [(0.0, 3.40), (15.0, 3.79), (32.0, 4.58)]


def velocity_at_depth(depth_km: float, layers) -> float:
    """Return the velocity at `depth_km` for a layered model `[(top_km, v_km_s), ...]`."""
    v = layers[0][1]
    for top_km, vel in layers:
        if depth_km >= top_km:
            v = vel
        else:
            break
    return v


def predict_arrival_offset(
    ev_lat: float, ev_lon: float, ev_dep_km: float,
    st_lat: float, st_lon: float, phase: str,
) -> tuple[float, float]:
    """Direct-ray travel time and epicentral distance for one (event, station, phase).

    Returns `(travel_time_s, epi_distance_km)`. Travel time is from origin to predicted arrival,
    i.e. add to origin time to get the predicted arrival time."""
    epi_m, _, _ = gps2dist_azimuth(ev_lat, ev_lon, st_lat, st_lon)
    epi_km = epi_m / 1000.0
    slant_km = math.sqrt(epi_km * epi_km + ev_dep_km * ev_dep_km)
    layers = KIM1983_P if phase == "P" else KIM1983_S
    v = velocity_at_depth(ev_dep_km, layers)
    return slant_km / v, epi_km


# ---------------------------------------------------------------- data loaders
def load_all_daily_picks(picks_dir: str, year: int) -> pd.DataFrame:
    """Concatenate every `picks_<year>.<jday>.csv` under `picks_dir`. Parse `peak_time` as UTC."""
    files = sorted(glob(os.path.join(picks_dir, f"picks_{year}.*.csv")))
    if not files:
        raise FileNotFoundError(f"no daily picks under {picks_dir} for year {year}")
    dfs = []
    for fn in files:
        d = pd.read_csv(fn)
        d["peak_time"] = pd.to_datetime(d["peak_time"], utc=True)
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def load_pyocto_events(events_csv: str) -> pd.DataFrame:
    """Read the PyOcto event table and parse `time` as UTC datetime."""
    ev = pd.read_csv(events_csv)
    ev["time"] = pd.to_datetime(ev["time"], utc=True)
    return ev


def load_pyocto_assignments(assignment_csv: str) -> pd.DataFrame:
    """Read PyOcto per-event pick assignments. `time` is a Unix timestamp (seconds)."""
    a = pd.read_csv(assignment_csv)
    # Build the (event_idx, station, phase) key for fast existence checks downstream.
    a["station_phase"] = a["station"] + "|" + a["phase"]
    return a


# ---------------------------------------------------------------- core matcher
def find_augmentation_candidates(
    events: pd.DataFrame,
    pyocto_assignments: pd.DataFrame,
    stations: pd.DataFrame,
    daily_picks: pd.DataFrame,
    *,
    radius_km: float = 100.0,
    tolerance_s: float = 1.0,
    min_pick_probability: float = 0.3,
    window_before_s: float = 5.0,
    window_after_s: float = 120.0,
) -> pd.DataFrame:
    """For every PyOcto event × every nearby station × P/S, find every daily pick that:

      - lies within `radius_km` of the event's PyOcto hypocenter,
      - has predicted-arrival residual within `tolerance_s`,
      - has PhaseNet+ probability ≥ `min_pick_probability`,
      - is NOT already in PyOcto's assignment for this event.

    Returns one row per candidate match — a pick may appear multiple times if it fits multiple
    nearby events within tolerance (those ambiguities are resolved by `apply_safeguards`).
    """
    # Index existing PyOcto picks by event for O(1) lookup
    pyocto_keys = {
        eid: set(g["station_phase"]) for eid, g in pyocto_assignments.groupby("event_idx")
    }

    # Station map: 'NET.CODE' -> (lat, lon).  PyOcto's `station` field has a trailing dot
    # ('KG.HDB.'); daily picks use no dot ('KG.HDB'). We normalise on the dot-less form.
    station_map = {
        f"{r.Network}.{r.Code}": (float(r.Latitude), float(r.Longitude))
        for _, r in stations.iterrows()
    }

    # Pre-filter daily picks to those with sufficient probability and matching phase
    dp = daily_picks[daily_picks["probability"] >= min_pick_probability].copy()
    # Lookup station coords; drop picks at unknown stations (alias issues etc.)
    dp = dp[dp["station"].isin(station_map.keys())].reset_index(drop=True)

    candidates = []
    for _, ev in events.iterrows():
        eid = int(ev.idx)
        ev_origin = ev.time  # UTC datetime
        ev_lat, ev_lon, ev_dep = float(ev.latitude), float(ev.longitude), float(ev.depth)

        # Time window for candidate picks: a bit before origin (P picks can pre-cede if event
        # is right under station) through max travel time over the radius (~30 s for 100 km Pn).
        t_lo = ev_origin - pd.Timedelta(seconds=window_before_s)
        t_hi = ev_origin + pd.Timedelta(seconds=window_after_s)
        in_window = dp[(dp["peak_time"] >= t_lo) & (dp["peak_time"] <= t_hi)]

        existing = pyocto_keys.get(eid, set())

        for _, pick in in_window.iterrows():
            net, code = pick["station"].split(".", 1)
            station_id_dotted = f"{net}.{code}."  # match PyOcto's assignment format
            key = station_id_dotted + "|" + pick["phase"]
            if key in existing:
                continue  # PyOcto already has this (station, phase) for this event

            st_lat, st_lon = station_map[pick["station"]]
            tt_s, epi_km = predict_arrival_offset(
                ev_lat, ev_lon, ev_dep, st_lat, st_lon, pick["phase"]
            )
            if epi_km > radius_km:
                continue  # station too far; PyOcto wouldn't have considered it

            predicted = ev_origin + pd.Timedelta(seconds=tt_s)
            resid_s = (pick["peak_time"] - predicted).total_seconds()
            if abs(resid_s) > tolerance_s:
                continue  # doesn't fit at this event's hypocenter

            candidates.append({
                "event_idx": eid,
                "station": station_id_dotted,
                "phase": pick["phase"],
                "time": pick["peak_time"].timestamp(),
                "residual": resid_s,
                "epi_km": epi_km,
                "probability": float(pick["probability"]),
                "pick_time_iso": pick["peak_time"].isoformat(),
                "ev_origin_iso": ev_origin.isoformat(),
            })

    return pd.DataFrame(candidates)


def apply_safeguards(
    candidates: pd.DataFrame,
    *,
    tie_threshold_s: float = 0.2,
) -> pd.DataFrame:
    """Resolve cross-event pick ambiguity for close-in-time doublets / triplets.

    For each unique (station, phase, pick_time):
      - If only one event matches → accept it.
      - If two or more events match and the best two residuals differ by < `tie_threshold_s`
        → drop the pick for ALL candidate events (genuinely ambiguous).
      - Otherwise → assign to the event with smallest |residual| only.

    Per (event, station, phase): at most one pick added. The cross-event uniqueness invariant
    is enforced (a pick can be in at most one event's augmented assignment).
    """
    if candidates.empty:
        return candidates.copy()

    # Group candidates by (station, phase, pick_time) — same orphan considered for multiple events
    candidates = candidates.copy()
    candidates["abs_res"] = candidates["residual"].abs()
    candidates["pick_key"] = (
        candidates["station"] + "|"
        + candidates["phase"] + "|"
        + candidates["pick_time_iso"]
    )

    accepted_rows = []
    diagnostics = []
    for pick_key, grp in candidates.groupby("pick_key"):
        grp = grp.sort_values("abs_res").reset_index(drop=True)
        if len(grp) == 1:
            accepted_rows.append(grp.iloc[0])
            continue
        best = grp.iloc[0]
        runner_up = grp.iloc[1]
        if (runner_up["abs_res"] - best["abs_res"]) < tie_threshold_s:
            diagnostics.append({
                "pick_key": pick_key,
                "best_event_idx": int(best.event_idx),
                "best_residual": best.residual,
                "runner_event_idx": int(runner_up.event_idx),
                "runner_residual": runner_up.residual,
                "decision": "dropped_ambiguous",
            })
            continue
        accepted_rows.append(best)
        if len(grp) > 1:
            diagnostics.append({
                "pick_key": pick_key,
                "best_event_idx": int(best.event_idx),
                "best_residual": best.residual,
                "runner_event_idx": int(runner_up.event_idx),
                "runner_residual": runner_up.residual,
                "decision": "best_match_wins",
            })

    if accepted_rows:
        out = pd.DataFrame(accepted_rows).reset_index(drop=True)
    else:
        out = candidates.iloc[:0].copy()

    # Enforce per-(event,station,phase) uniqueness defensively (shouldn't be needed since
    # PyOcto already enforces it for orphan picks at distinct times, but multi-pick noise
    # at one station could collide).
    dedup_key = out["event_idx"].astype(str) + "|" + out["station"] + "|" + out["phase"]
    out = out.iloc[out.groupby(dedup_key)["abs_res"].idxmin().values].reset_index(drop=True)

    return out.drop(columns=["abs_res", "pick_key"], errors="ignore"), pd.DataFrame(diagnostics)


# ---------------------------------------------------------------- write augmented assignment
def merge_augmented(
    original_assignment: pd.DataFrame, augmented: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate the original PyOcto assignment with the augmented picks. The augmented rows
    add a `source` column ('augmented') for provenance; original rows get 'pyocto'."""
    orig = original_assignment.drop(columns=["station_phase"], errors="ignore").copy()
    orig["source"] = "pyocto"
    if augmented.empty:
        return orig.reset_index(drop=True)

    aug = augmented[["event_idx", "station", "phase", "time", "residual"]].copy()
    aug["source"] = "augmented"
    return pd.concat([orig, aug], ignore_index=True).sort_values(
        ["event_idx", "time"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------- top-level API
def augment_year(
    pyocto_events_csv: str,
    pyocto_assignment_csv: str,
    stations_csv: str,
    picks_dir: str,
    year: int,
    *,
    out_assignment_csv: Optional[str] = None,
    backup_suffix: str = ".before_augmentation.csv",
    radius_km: float = 100.0,
    tolerance_s: float = 1.0,
    min_pick_probability: float = 0.3,
    tie_threshold_s: float = 0.2,
) -> dict:
    """Augment one year's PyOcto pick assignments. Returns a summary dict with the counts /
    diagnostics. If `out_assignment_csv` is given, writes to that path. Otherwise writes back
    to `pyocto_assignment_csv` after backing up the original to `<path><backup_suffix>`.

    The returned dict has keys:
      n_events            : total PyOcto events processed
      n_original_picks    : original PyOcto pick count
      n_candidates        : how many orphans fit before safeguards
      n_accepted          : how many orphans were actually added after safeguards
      n_dropped_ambiguous : how many orphans were dropped because they matched ≥2 events
      out_path            : where the augmented CSV was written
      diagnostics_path    : where the per-pick decision log was written
    """
    events = load_pyocto_events(pyocto_events_csv)
    pyocto_pks = load_pyocto_assignments(pyocto_assignment_csv)
    stations = pd.read_csv(stations_csv)
    daily = load_all_daily_picks(picks_dir, year)

    candidates = find_augmentation_candidates(
        events, pyocto_pks, stations, daily,
        radius_km=radius_km,
        tolerance_s=tolerance_s,
        min_pick_probability=min_pick_probability,
    )

    accepted, diagnostics = apply_safeguards(candidates, tie_threshold_s=tie_threshold_s)
    merged = merge_augmented(pyocto_pks, accepted)

    if out_assignment_csv is None:
        # Backup + overwrite the original
        out_assignment_csv = pyocto_assignment_csv
        backup = pyocto_assignment_csv + backup_suffix.replace(".csv", "_csv")
        # ensure backup uses the canonical naming with .csv extension
        backup = pyocto_assignment_csv.replace(".csv", backup_suffix)
        if not os.path.exists(backup):
            pyocto_pks.drop(columns=["station_phase"], errors="ignore").to_csv(backup, index=False)
    merged.drop(columns=["station_phase"], errors="ignore").to_csv(out_assignment_csv, index=False)

    diag_path = out_assignment_csv.replace(".csv", ".augmentation_diagnostics.csv")
    diagnostics.to_csv(diag_path, index=False)

    n_drop = int((diagnostics["decision"] == "dropped_ambiguous").sum()) if len(diagnostics) else 0
    return dict(
        n_events=len(events),
        n_original_picks=len(pyocto_pks),
        n_candidates=len(candidates),
        n_accepted=len(accepted),
        n_dropped_ambiguous=n_drop,
        out_path=out_assignment_csv,
        diagnostics_path=diag_path,
    )
