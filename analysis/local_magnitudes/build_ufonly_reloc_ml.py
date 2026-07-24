#!/usr/bin/env python
"""UF-only-corrected magnitude catalog AT THE HypoDD dt.cc-RELOCATED (kim2011) HYPOCENTRES.

Companion to build_ufonly_ml.py (which uses absolute HypoInverse locations). The ONLY thing that changes is
the hypocentral DISTANCE per station-reading: we recompute it from each event's refined dt.cc location
(kim2011 velocity) and the station coordinates, then re-evaluate the Heo et al. (2024) ML and re-run the
identical median-polish station/epoch correction. Amplitudes (peak_mm) are unchanged — only locations move,
so ML moves only through the distance-attenuation term.

Distance convention (IDENTICAL to ml_pipeline.hypocentral_km): R = sqrt(epicentral_km^2 + depth_km^2), with
epicentral = great-circle station->epicentre (gps2dist_azimuth) and depth = focal depth (km); station
elevation NOT added (matches the original SAC export). ML = log10(peak_mm) + 1.450676*log10(R/17)
- 0.000661*(R-17) + 2.0 (Heo et al. 2024, vertical component). SNR gate snr_pp>=2.0 (unchanged).

Correction machinery (IDENTICAL to build_ufonly_ml.py): every documented sensor-break epoch (sc@e) gets its
own median-polish term (NO MIN_EPOCH_N merge, NO ad-hoc cut); plus the data-driven HDB sensor-FAILURE onset
(first month with HDB monthly residual < -1.0 ML before 2015-06) as its own epoch. Gauge = obs-weighted
mean(S)=0.

Scope: ONLY events present in the relocated catalog. event_idx is recovered by the EXACT hypoDD
id->ts->event_idx map (cuspid = 200000 + sorted waveforms_100km "20*" dir index; dir name = pipeline
timestamp ts; members_event_idx.csv maps ts -> master event_idx) — NO time-window matching, drift-free by
construction. Same-second doublets (one waveform dir per second) resolve to the smallest event_idx (the
member stage.py staged first); the larger twin gets no separate relocation (disclosed at runtime).
Non-relocated events get no refined ML and are out of scope here.

Output: catalog_ml_heo_ufonly_reloc.csv with one row per relocated event:
  event_idx, event_time, lat, lon, depth, x, y, z, ex, ey, ez, nccp, nccs, nctp, ncts, cid,
  is_dtcc, n_used, ml_ufraw_reloc, ml_ufcorr_reloc, ml_ufcorr_old, dml
Run in `base`, cwd = local_magnitudes.
"""
import warnings; warnings.filterwarnings("ignore")
import os, json, glob, numpy as np, pandas as pd
from obspy.geodetics.base import gps2dist_azimuth
import sys; sys.path.insert(0, ".")
from ml_pipeline import ml_heo2024   # exact Heo 2024 formula (vertical component)

SNR_PP_MIN = 2.0
PS    = "catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo_uncapped.csv"
UFCAT = "catalog_ml_heo_ufonly.csv"
CACHE = "/home/msseo/works/02.Ulsan_Fault_detection/data/metadata/responses/sensor_breaks_master.json"
RUN   = ("/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/"
         "runs/uf_subregion_reuse")
RELOC = os.environ.get("UF_RELOC",                        # kim2011 dt.cc primary product (override for QC)
                       f"{RUN}/2.HypoDD/03.dt.cc_kim2011/hypoDD.reloc")
WF100 = f"{RUN}/waveforms_100km"                          # hypoDD cuspid = 200000 + sorted "20*" dir index
MEIDX = ("/home/msseo/works/02.Ulsan_Fault_detection/analysis/uf_subregion_hypodd/"
         "uf_subregion/members_event_idx.csv")            # ts (floor-sec) -> master event_idx
STATB = ("/home/msseo/works/02.Ulsan_Fault_detection/outputs/models/phasenet_plus/"
         "station_table/stations_*.csv")   # was ../models/... pre-restructure (cwd-dependent)

# ---- 1. relocated catalog (kim2011 dt.cc) -------------------------------------------------------------
COLS = ["id","lat","lon","depth","x","y","z","ex","ey","ez","yr","mo","dy","hr","mi","sc","mag",
        "nccp","nccs","nctp","ncts","rcc","rct","cid"]
r = pd.read_csv(RELOC, sep=r"\s+", header=None, names=COLS)
_sc = r.sc.clip(0, 59.999)
r["time"] = pd.to_datetime(dict(year=r.yr, month=r.mo, day=r.dy, hour=r.hr, minute=r.mi,
                                second=_sc.astype(int), microsecond=((_sc-_sc.astype(int))*1e6).astype(int)),
                           utc=True, errors="coerce")
r["is_dtcc"] = (r.nccp + r.nccs) > 0
r = r.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
print(f"relocated (kim2011 dt.cc): {len(r):,} events | dt.cc-resolved (ncc>0): {int(r.is_dtcc.sum()):,} | "
      f"dt.ct-only: {int((~r.is_dtcc).sum()):,} | depth {r.depth.min():.1f}-{r.depth.max():.1f} km")

# ---- 2. recover event_idx via the EXACT id->ts->event_idx map (NO time matching) ---------------------
# hypoDD cuspid = CUSPID_OFFSET(200000) + position in sorted(waveforms_100km/"20*"); each dir name IS the
# pipeline timestamp ts; members_event_idx.csv maps ts (floor-second of the single current catalog) -> the
# master event_idx. Drift-free by construction: catalog, staged dir and reloc id all come from one source.
_dirs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(WF100, "20*")))
id2ts = {200000 + i: ts for i, ts in enumerate(_dirs)}
mei = pd.read_csv(MEIDX).sort_values("event_idx")        # members order = event_idx-ascending
mei["ts"] = pd.to_datetime(mei.time, utc=True, format="ISO8601").dt.floor("s").dt.strftime("%Y%m%d%H%M%S")
# same-second doublets collapse to one waveform dir -> one relocation; stage.py fills each dir from the
# FIRST-staged (smallest event_idx) member, so the relocation belongs to the smallest event_idx; the larger
# twin gets no separate relocation. Assign deterministically (setdefault over ascending event_idx) + disclose.
ts2eidx = {}
for ei, ts in zip(mei.event_idx.astype(int), mei.ts):
    ts2eidx.setdefault(ts, ei)
_dups = mei[mei.ts.duplicated(keep=False)]
drop_twins = sorted(set(_dups.event_idx.astype(int)) - set(ts2eidx.values()))
if drop_twins:
    print(f"same-second doublets: {_dups.ts.nunique()} timestamp(s), {len(_dups)} members -> "
          f"{_dups.ts.nunique()} relocation(s); dropped twins (no separate reloc): {drop_twins}")
r["event_idx"] = r.id.map(id2ts).map(ts2eidx)
_unmapped = int(r.event_idx.isna().sum())
rm = r.dropna(subset=["event_idx"]).copy(); rm["event_idx"] = rm["event_idx"].astype(int)
assert rm.event_idx.is_unique, "event_idx not unique after exact map"
uf = pd.read_csv(UFCAT)[["event_idx", "ml_ufcorr"]]      # old absolute-location UF ML, exact event_idx join
rm = rm.merge(uf, on="event_idx", how="left")
print(f"exact id->ts->event_idx: {len(rm):,}/{len(r):,} reloc events carry event_idx (unmapped {_unmapped}) | "
      f"dt.cc-resolved {int(rm.is_dtcc.sum()):,} | with old SOTA ML {int(rm.ml_ufcorr.notna().sum()):,}")
loc = rm.set_index("event_idx")[["lat","lon","depth"]]   # refined hypocentre per event

# ---- 3. station coordinates (union of per-year tables; (Network,Code) -> lat/lon) --------------------
st = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(STATB))]).drop_duplicates(["Network","Code"])
STLA = {(n, c): la for n, c, la in zip(st.Network, st.Code, st.Latitude)}
STLO = {(n, c): lo for n, c, lo in zip(st.Network, st.Code, st.Longitude)}

# ---- 4. per-station readings -> recompute distance + ML at refined location --------------------------
ps = pd.read_csv(PS)
ps = ps[(ps.snr_pp >= SNR_PP_MIN) & ps.peak_mm.notna() & (ps.peak_mm > 0)].copy()
ps["t"] = pd.to_datetime(ps.event_time, utc=True, errors="coerce"); ps = ps.dropna(subset=["t"])
ps = ps[ps.event_idx.isin(loc.index)].copy()                       # only relocated events
ps["sc"] = ps.network + "." + ps.station + "." + ps.channel
ps["nlat"] = loc.lat.reindex(ps.event_idx).values
ps["nlon"] = loc.lon.reindex(ps.event_idx).values
ps["ndep"] = loc.depth.reindex(ps.event_idx).values
ps["stla"] = [STLA.get((n, s), np.nan) for n, s in zip(ps.network, ps.station)]
ps["stlo"] = [STLO.get((n, s), np.nan) for n, s in zip(ps.network, ps.station)]
_nost = ps.stla.isna()
if _nost.any():
    print(f"WARNING: {int(_nost.sum())} readings on {ps.loc[_nost,'sc'].nunique()} station(s) lack coords "
          f"-> dropped: {sorted(ps.loc[_nost,'station'].unique())}")
ps = ps.dropna(subset=["stla","stlo","nlat","nlon","ndep"]).copy()
# recompute hypocentral distance (km): epicentral great-circle + focal depth (matches hypocentral_km)
epi = np.array([gps2dist_azimuth(la, lo, sla, slo)[0] / 1000.0
                for la, lo, sla, slo in zip(ps.nlat, ps.nlon, ps.stla, ps.stlo)])
ps["dist_km_reloc"] = np.hypot(epi, ps.ndep.values)
ps["ML"] = ml_heo2024(ps.peak_mm.values, ps.dist_km_reloc.values)   # recomputed per-station ML (Z-formula)
ps = ps[np.isfinite(ps.ML)].copy()
print(f"per-station readings (relocated events, snr_pp>=2): {len(ps):,} | events {ps.event_idx.nunique():,} "
      f"| stations {ps.sc.nunique()}")

# ---- 5. epoch units (documented breaks + data-driven HDB failure onset) — IDENTICAL to build_ufonly_ml
breaks = {k: [pd.Timestamp(x).date() for x in v] for k, v in json.load(open(CACHE)).items()}
_res = ps.ML.values - ps.groupby("event_idx").ML.transform("median").values
_hm = pd.Series(_res, index=ps.t)[ps.sc.values == "KG.HDB.HHZ"].groupby(pd.Grouper(freq="ME")).median()
_f = _hm[(_hm < -1.0) & (_hm.index < pd.Timestamp("2015-06", tz="UTC"))]
FAIL_ON = (pd.Timestamp(_f.index.min()).replace(day=1).date() if len(_f) else pd.Timestamp("2014-12-01").date())
breaks["KG.HDB.HHZ"] = sorted(set(breaks.get("KG.HDB.HHZ", []) + [FAIL_ON]))
print(f"HDB failure-window onset (data-driven, monthly resid < -1.0): {FAIL_ON} -> own epoch")

def era_unit(row):
    s = row.sc
    return s if s not in breaks else f"{s}@e{sum(row.t.date() >= b for b in breaks[s])}"
ps["unit"] = ps.apply(era_unit, axis=1)

def median_polish(df, col, n=80, tol=1e-4):
    w = df[col].value_counts(); mu = df.groupby("event_idx").ML.median(); S = pd.Series(0.0, index=w.index)
    for _ in range(n):
        Sn = pd.Series(df.ML.values - mu.reindex(df.event_idx).values, index=df[col]).groupby(level=0).median()
        Sn -= np.average(Sn.reindex(w.index), weights=w.values)
        mun = pd.Series(df.ML.values - Sn.reindex(df[col]).values, index=df.event_idx).groupby(level=0).median()
        if float(np.nanmax(np.abs(mun.reindex(mu.index).values - mu.values))) < tol: mu, S = mun, Sn; break
        mu, S = mun, Sn
    return mu, S

mu_raw = ps.groupby("event_idx").ML.median()           # refined-location, NO correction
mu_corr, S = median_polish(ps, "unit")                 # refined-location, UF-only epoch+station correction
nused = ps.groupby("event_idx").sc.nunique()
print(f"epoch units: {ps.unit.nunique()} | station-term std {S.std():.3f}")

# ---- 6. assemble output -------------------------------------------------------------------------------
relfields = rm.set_index("event_idx")[["time","lat","lon","depth","x","y","z","ex","ey","ez",
                                        "nccp","nccs","nctp","ncts","cid","is_dtcc","ml_ufcorr"]]
ev = pd.DataFrame({
    "event_time": relfields.time,
    "lat": relfields.lat, "lon": relfields.lon, "depth": relfields.depth,
    "x": relfields.x, "y": relfields.y, "z": relfields.z, "ex": relfields.ex, "ey": relfields.ey, "ez": relfields.ez,
    "nccp": relfields.nccp, "nccs": relfields.nccs, "nctp": relfields.nctp, "ncts": relfields.ncts,
    "cid": relfields.cid, "is_dtcc": relfields.is_dtcc,
    "n_used": nused.reindex(relfields.index),
    "ml_ufraw_reloc": mu_raw.reindex(relfields.index),
    "ml_ufcorr_reloc": mu_corr.reindex(relfields.index),
    "ml_ufcorr_old": relfields.ml_ufcorr,
}).dropna(subset=["ml_ufcorr_reloc"])
ev["dml"] = ev.ml_ufcorr_reloc - ev.ml_ufcorr_old
ev.index.name = "event_idx"
ev.to_csv("catalog_ml_heo_ufonly_reloc.csv")
nuse3 = ev.n_used >= 3
print(f"wrote catalog_ml_heo_ufonly_reloc.csv ({len(ev):,} events; n_used>=3: {int(nuse3.sum()):,}; "
      f"dt.cc & n_used>=3: {int((ev.is_dtcc & nuse3).sum()):,})")
print(f"dml (reloc-old): median {ev.dml.median():+.3f}, IQR [{ev.dml.quantile(.25):+.3f},{ev.dml.quantile(.75):+.3f}], "
      f"std {ev.dml.std():.3f}")
print(f"dt.cc & n_used>=3 & ml_ufcorr_reloc>=1.2: {int((ev.is_dtcc & nuse3 & (ev.ml_ufcorr_reloc>=1.2)).sum())} "
      f"(was 144 on the full absolute-location catalog; 112 on old-ML dt.cc subset)")
