#!/usr/bin/env python
"""UF-only-corrected magnitude catalog.

A candidate strategy distinct from the constant network: keep the FULL evolving network (all stations,
all UF events — best completeness), but fit the station + epoch correction terms using **only the
Ulsan-Fault-box events**. The UF box is small (~30 km), so each station's distance to UF events spans a
narrow range; a constant per-station term therefore absorbs most of that station's (nearly-constant)
distance bias for UF paths — something a region-wide correction cannot do (there a station sees 10-200 km).

Result (see nb25): this *reduces* the post-2019 network-geometry drift (slope +0.018 -> +0.011 ML/yr,
b 0.90 -> 1.03) but does not fully flatten it (residual within-box distance spread + post-2019-only
stations). It keeps all ~2497 UF events at Mc ~0.6 (vs the constant network's ~890 at Mc ~0.8).

Output: catalog_ml_heo_ufonly.csv (event_idx, event_time, year, n_used, ml_ufraw, ml_ufcorr).
Run in `base`, cwd = local_magnitudes.
"""
import warnings; warnings.filterwarnings("ignore")
import json, numpy as np, pandas as pd

UF = [129.25, 129.55, 35.60, 35.90]; SNR_PP_MIN = 2.0; MIN_EPOCH_N = 50
CLEAN = "catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo_homogenised_clean.csv"
PS    = "catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo_uncapped.csv"
CACHE = "/home/msseo/works/02.Ulsan_Fault_detection/data/metadata/responses/sensor_breaks_master.json"

clean = pd.read_csv(CLEAN); clean["time"] = pd.to_datetime(clean.time, utc=True, errors="coerce")
clean = clean.dropna(subset=["time"])
ufev = clean[(clean.lon.between(UF[0], UF[1])) & (clean.lat.between(UF[2], UF[3]))]
ufkey = set(np.round(ufev.time.astype("int64") / 1e9).astype(int))

ps = pd.read_csv(PS)
ps = ps[(ps.snr_pp >= SNR_PP_MIN) & ps.ML.notna() & ps.dist_km.notna()].copy()
ps["t"] = pd.to_datetime(ps.event_time, utc=True, errors="coerce"); ps = ps.dropna(subset=["t"])
ps["sc"] = ps.network + "." + ps.station + "." + ps.channel
ps["uf"] = np.round(ps.t.astype("int64") / 1e9).astype(int).isin(ufkey)
d = ps[ps.uf].copy(); d["year"] = d.t.dt.year
print(f"UF readings: {len(d):,}  events: {d.event_idx.nunique():,}  stations: {d.sc.nunique()}  {d.year.min()}-{d.year.max()}")

# documented sensor-shape breaks (all stations)
breaks = {k: [pd.Timestamp(x).date() for x in v] for k, v in json.load(open(CACHE)).items()}

# HDB sensor-FAILURE window: its END (2015-05-21) is already a documented break; its ONSET is DATA-DRIVEN
# (first month where HDB's monthly-median station residual < -1.0 ML, before 2015-06). DISCLOSED parameter:
# failure threshold = -1.0 ML. Added as an extra HDB break so the failure window becomes its own epoch
# (term ~ -2 ML) and those corrupted readings are corrected, matching nb23. (Onset is the only data-driven
# break; all others are documented StationXML pole/zero changes.)
_res = d.ML.values - d.groupby("event_idx").ML.transform("median").values
_hm = pd.Series(_res, index=d.t)[d.sc.values == "KG.HDB.HHZ"].groupby(pd.Grouper(freq="ME")).median()
_f = _hm[(_hm < -1.0) & (_hm.index < pd.Timestamp("2015-06", tz="UTC"))]
FAIL_ON = (pd.Timestamp(_f.index.min()).replace(day=1).date() if len(_f) else pd.Timestamp("2014-12-01").date())
breaks["KG.HDB.HHZ"] = sorted(set(breaks.get("KG.HDB.HHZ", []) + [FAIL_ON]))
print(f"HDB failure-window onset (data-driven, monthly resid < -1.0): {FAIL_ON} -> own epoch; end 2015-05-21 (documented)")

def median_polish(df, col, n=80, tol=1e-4):
    w = df[col].value_counts(); mu = df.groupby("event_idx").ML.median(); S = pd.Series(0.0, index=w.index)
    for _ in range(n):
        Sn = pd.Series(df.ML.values - mu.reindex(df.event_idx).values, index=df[col]).groupby(level=0).median()
        Sn -= np.average(Sn.reindex(w.index), weights=w.values)
        mun = pd.Series(df.ML.values - Sn.reindex(df[col]).values, index=df.event_idx).groupby(level=0).median()
        if float(np.nanmax(np.abs(mun.reindex(mu.index).values - mu.values))) < tol: mu, S = mun, Sn; break
        mu, S = mun, Sn
    return mu, S

# SIMPLE epoch correction: every documented-break epoch (sc@e) gets its own median-polish term.
# NO MIN_EPOCH_N merge, NO threshold, NO per-station tuning. Only input = StationXML sensor breaks.
# Consequence (disclosed, not hidden): epochs with very few readings overfit those readings; reported below.
def era_unit(row):
    s = row.sc
    return s if s not in breaks else f"{s}@e{sum(row.t.date() >= b for b in breaks[s])}"
d["unit"] = d.apply(era_unit, axis=1)
_uc = d.unit.value_counts()
print(f"units {d.unit.nunique()} (NO merge) | epochs with <10 readings: {int((_uc<10).sum())} "
      f"({int(_uc[_uc<10].sum())} readings — each overfit by its own term)")

mu_raw = d.groupby("event_idx").ML.median()                       # full network, NO correction
mu_corr, S = median_polish(d, "unit")                             # full network, UF-only epoch+station correction
nused = d.groupby("event_idx").sc.nunique()
print(f"epoch units: {d.unit.nunique()}   |  station-term std {S.std():.3f}")

ev = pd.DataFrame({
    "event_time": d.groupby("event_idx").t.first(),
    "year": d.groupby("event_idx").year.first(),
    "n_used": nused,
    "ml_ufraw": mu_raw,
    "ml_ufcorr": mu_corr.reindex(mu_raw.index),
}).dropna(subset=["ml_ufcorr"])
ev.index.name = "event_idx"
ev.to_csv("catalog_ml_heo_ufonly.csv")
print(f"wrote catalog_ml_heo_ufonly.csv ({len(ev):,} events; n_used>=3: {(ev.n_used>=3).sum():,})")
