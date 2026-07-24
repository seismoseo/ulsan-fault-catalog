#!/usr/bin/env python
"""Constant-reference-network ML catalog.

Motivation (the temporal-stationarity fix done RIGHT). The evolving KS/KG network introduces a
*network-geometry* artifact into the magnitude SCALE: stations added after 2016/2019 sit far from the
Ulsan-Fault cluster and over-correct (read systematically high), inflating event ML post-2019. A
distance cap removes that inflation but ALSO censors the genuine post-2019 completeness gain (it drops
the smallest events), flattening the magnitude floor — which is itself wrong.

The correct fix for any *secular* magnitude study is a FIXED reference network: measure every event with
the SAME set of stations across the whole period, so the magnitude scale cannot drift with the network.
Exactly 5 station-channels operate over the full 2010-2024 span and sit within ~50 km of the box (the
"persistent anchors"):

    KG.MKL.HHZ (16 km)  KG.HDB.HHZ (24 km)  KG.YSB.HHZ (38 km)  KG.CGD.ELZ (39 km)  KG.CHS.HHZ (49 km)

Epoch-dependent station drift is handled with the **same canonical recipe as nb17/nb18**:
  * documented sensor-shape breaks from responses/sensor_breaks_master.json (HDB 4, YSB 6, CHS 1, MKL 1);
  * the HDB sensor-FAILURE window (~2014-11 .. 2015-05-21) EXCLUDED entirely (residual < -1 ML there);
  * median polish on epoch units (sc@epoch), epochs with < MIN_EPOCH_N readings merged back.

Outputs (event-level, keyed by event_idx / event_time):
    catalog_ml_heo_const.csv : n_const, ml_const_inv (single offset/station), ml_const (epoch-corrected)
A magnitude is "reliable" when n_const >= 3 (>=3 of the 5 anchors pass threshold for that event).

Run in `base`, cwd = local_magnitudes.
"""
import warnings; warnings.filterwarnings("ignore")
import os, json, numpy as np, pandas as pd

ANCHORS = ["KG.MKL.HHZ", "KG.HDB.HHZ", "KG.YSB.HHZ", "KG.CGD.ELZ", "KG.CHS.HHZ"]
PS   = "catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo_uncapped.csv"  # anchors all <50km: cap irrelevant
CACHE = "/home/msseo/works/02.Ulsan_Fault_detection/data/metadata/responses/sensor_breaks_master.json"      # the canonical, previously-defined epochs (nb17/nb18)
SNR_PP_MIN = 2.0; MIN_EPOCH_N = 50

# ----- load, restrict to the constant network, apply current quality gate -----
d = pd.read_csv(PS)
d["sc"] = d.network + "." + d.station + "." + d.channel
d = d[d.sc.isin(ANCHORS) & (d.snr_pp >= SNR_PP_MIN) & d.ML.notna()].copy()
d["t"] = pd.to_datetime(d.event_time, utc=True, errors="coerce"); d = d.dropna(subset=["t"])
d["year"] = d.t.dt.year
print(f"{len(d):,} anchor readings | {d.sc.nunique()}/{len(ANCHORS)} anchors | {d.year.min()}-{d.year.max()}")

# ----- the previously-defined epoch breaks (documented sensor-shape changes) -----
breaks_str = json.load(open(CACHE))
breaks = {k: [pd.Timestamp(x).date() for x in v] for k, v in breaks_str.items() if k in ANCHORS}
print("documented epoch breaks (from sensor_breaks_master.json):")
for s in ANCHORS: print(f"  {s:14} {breaks.get(s, [])}")

# ----- median polish (identical to nb18 `mp`) -----
def mp(df, col, n=60, tol=1e-4):
    w = df[col].value_counts(); mu = df.groupby("event_idx").ML.median(); S = pd.Series(0.0, index=w.index)
    for _ in range(n):
        Sn = pd.Series(df.ML.values - mu.reindex(df.event_idx).values, index=df[col]).groupby(level=0).median()
        Sn -= np.average(Sn.reindex(w.index), weights=w.values)
        mun = pd.Series(df.ML.values - Sn.reindex(df[col]).values, index=df.event_idx).groupby(level=0).median()
        if float(np.nanmax(np.abs(mun.reindex(mu.index).values - mu.values))) < tol: mu, S = mun, Sn; break
        mu, S = mun, Sn
    return mu, S

mu0, S0 = mp(d, "sc"); d["res"] = d.ML - d.event_idx.map(mu0).values   # single-offset baseline + raw station residual
print("\nsingle-offset anchor terms:", {s.split('.')[1]: round(float(S0.get(s, np.nan)), 2) for s in ANCHORS})

# ----- HDB sensor-FAILURE window: treat as its OWN epoch and correct it (per user; cf. nb18 which excluded) -----
_hm = d[d.sc == "KG.HDB.HHZ"].set_index("t").res.groupby(pd.Grouper(freq="ME")).median()
_f = _hm[(_hm < -1.0) & (_hm.index < pd.Timestamp("2015-06", tz="UTC"))]
FAIL_ON = pd.Timestamp(_f.index.min()).replace(day=1) if len(_f) else pd.Timestamp("2014-12-01", tz="UTC")
FAIL_OFF = pd.Timestamp("2015-05-21", tz="UTC")        # = a documented HDB break (sensor swap / recovery)
fail_mask = lambda df: (df.sc == "KG.HDB.HHZ") & (df.t >= FAIL_ON) & (df.t < FAIL_OFF)
# add the failure-window ONSET as an extra HDB break so [FAIL_ON, FAIL_OFF) is its own epoch
breaks["KG.HDB.HHZ"] = sorted(set(breaks.get("KG.HDB.HHZ", []) + [FAIL_ON.date()]))
print(f"HDB failure window: {FAIL_ON.date()} .. {FAIL_OFF.date()} ({int(fail_mask(d).sum())} HDB readings -> own epoch, CORRECTED)")

def era_unit(row):
    s = row.sc
    if s not in breaks: return s
    return f"{s}@e{sum(row.t.date() >= bd for bd in breaks[s])}"
# the failure-window unit (protected from the <MIN_EPOCH_N merge so it keeps its own offset despite few readings)
FAIL_UNIT = f"KG.HDB.HHZ@e{sum(pd.Timestamp('2015-02-01').date() >= bd for bd in breaks['KG.HDB.HHZ'])}"

# ----- epoch-split refit on the documented breaks (+ failure epoch), ALL readings kept -----
u = d.apply(era_unit, axis=1)
uc = u.value_counts(); small = set(uc[uc < MIN_EPOCH_N].index) - {FAIL_UNIT}
d["unit"] = u.where(~u.isin(small), d.sc)
mu1, S1 = mp(d, "unit")
print(f"\nepoch units ({d.unit.nunique()}): "
      + ", ".join(f"{k.split('.')[1]}:{int((d.unit==k).sum())}" for k in sorted(d.unit.unique())))
print(f"  failure epoch {FAIL_UNIT}: offset S = {S1.get(FAIL_UNIT, float('nan')):+.2f} ML  (n={int((d.unit==FAIL_UNIT).sum())})")
dd = d   # all readings kept now (no exclusion)

# ----- residual diagnostics: does the epoch split flatten HDB/YSB? -----
dd["res_inv"]   = dd.ML - dd.event_idx.map(mu0).values - dd.sc.map(S0).values     # after single offset
dd["res_epoch"] = dd.ML - dd.event_idx.map(mu1).values - dd.unit.map(S1).values   # after epoch split
for s in ("KG.HDB.HHZ", "KG.YSB.HHZ"):
    sub = dd[dd.sc == s]
    a = sub.res_inv.abs().median(); b = sub.res_epoch.abs().median()
    print(f"  {s}: median |residual|  single {a:.3f} -> epoch {b:.3f} ML")

# ----- assemble event-level catalog -----
ncon = dd.groupby("event_idx").sc.nunique()
ev = pd.DataFrame({
    "event_time": d.groupby("event_idx").t.first(),
    "year": d.groupby("event_idx").year.first(),
    "n_const": ncon,
    "ml_const_inv": mu0,
    "ml_const": mu1.reindex(mu0.index),
}).dropna(subset=["ml_const"])
ev["n_const"] = ev["n_const"].fillna(0).astype(int)
ev.index.name = "event_idx"
ev.to_csv("catalog_ml_heo_const.csv")
print(f"\nwrote catalog_ml_heo_const.csv ({len(ev):,} events; {(ev.n_const>=3).sum():,} with n_const>=3)")
