#!/usr/bin/env python
"""Native-200-Hz vs decimated-100-Hz picking benchmark on real NS (GHBSN) event windows.

USAGE (env: base):
    python run_native_rate.py            # SeisBench PhaseNet-original, T1 + T2, writes results + window files

Design (pre-registered in 03.Detection_test_design.md):
  events    : KMA events within 60 km of Gyeongju, M >= 2.0, inside the NS local-data span, largest N_EVENTS
  stations  : nearest N_STA NS stations with HH day files for the event day
  window    : origin-70 s .. origin+110 s  (>=60 s pre-P noise for the false-pick rate)
  T1        : proper decimation to 100 Hz (zero-phase lowpass 40 Hz, decimate x2)  -> classify
  T2        : native 200 Hz (model.sampling_rate = 200; model judges in SAMPLES)   -> classify
  reference : kim2011 flat-layer first-arrival predictions (matching window +/-1.5 s)

Outputs:
  cache/native_rate/windows_manifest.csv           event x station window list
  cache/native_rate/win_T1/*.mseed                 100 Hz windows   (also PhaseNet+ driver input)
  cache/native_rate/win_T2lie/*.mseed              200 Hz windows with header fs=100 (metadata lie, for EQNet)
  cache/native_rate/picks_pn_original_T{1,2}.csv   SeisBench picks per treatment
"""
import os, sys, glob, time, argparse
import numpy as np
import pandas as pd
import obspy

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSROOT = "/home/msseo/works/02.Ulsan_Fault_detection/NS"
P_KMA = "/home/msseo/works/16.kma_absolute_location/runs/kma_batch/results_final.csv"
GYEONGJU = (35.856, 129.224)
N_EVENTS, N_STA = 40, 12
R_EVT_KM, MIN_MAG = 60.0, 2.0
SPAN = (pd.Timestamp("2018-01-01"), pd.Timestamp("2023-11-30"))
PRE, POST = 70.0, 110.0
THRESH = dict(P_threshold=0.3, S_threshold=0.3)

KIM2011 = dict(h=[7.29, 13.41, 10.60, 1e9],      # layer thicknesses (last = halfspace)
               vp=[5.63, 6.17, 6.58, 7.77], vs=[3.40, 3.60, 3.70, 4.45])


def first_arrival(dist_km, z_km, v):
    """Flat-layer first arrival (min of direct wave and head waves), source at depth z, receiver at surface.

    Head wave along the top of layer n:  T = x/v_n + sum_i c_i * d_i * sqrt(1/v_i^2 - 1/v_n^2)
    where the up-leg crosses every layer 0..n-1 fully and the down-leg crosses only the part of the
    source layer below the source plus the full layers between source layer and the refractor.
    """
    h, tops = KIM2011["h"], np.cumsum([0.0] + KIM2011["h"][:-1])
    isrc = int(np.searchsorted(tops, z_km, side="right") - 1)
    tt = [np.hypot(dist_km, z_km) / v[isrc]]                 # direct wave, source-layer velocity
    for n in range(isrc + 1, len(v)):
        vn = v[n]
        if vn <= max(v[:n]):                                  # no head wave unless refractor is fastest so far
            continue
        t = dist_km / vn
        for i in range(n):
            eta = np.sqrt(1.0 / v[i] ** 2 - 1.0 / vn ** 2)
            up = h[i]                                          # up-leg: full layer i
            if i < isrc:   down = 0.0                          # down-leg starts at the source
            elif i == isrc: down = (tops[i] + h[i]) - z_km     # remainder of the source layer
            else:           down = h[i]
            t += (up + down) * eta
        tt.append(t)
    return float(min(tt))


def predict_ps(dist_km, z_km):
    return (first_arrival(dist_km, z_km, KIM2011["vp"]),
            first_arrival(dist_km, z_km, KIM2011["vs"]))


def hav_km(a1, o1, a2, o2):
    x = (np.sin(np.radians(a2 - a1) / 2) ** 2 +
         np.cos(np.radians(a1)) * np.cos(np.radians(a2)) * np.sin(np.radians(o2 - o1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(x))


def ns_stations():
    ns = pd.read_csv("/home/msseo/works/02.Ulsan_Fault_detection/data/metadata/GHBSN_metadata/20240715/GHBSN_info_ver202312_modified.csv")
    ns = ns.rename(columns={"station": "sta", "stla": "lat", "stlo": "lon"}).drop_duplicates("sta")
    return ns[~ns.sta.astype(str).str.startswith(("U", "Y"))]


def day_files(sta, t):
    doy = t.dayofyear
    return sorted(glob.glob(os.path.join(NSROOT, sta, "HH?.D", f"*.{t.year}.{doy:03d}")))


def main():
    out = os.path.join(HERE, "cache", "native_rate")
    for d in ("win_T1", "win_T2lie"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    kma = pd.read_csv(P_KMA)
    kma["time"] = pd.to_datetime(kma.event_id.astype(str).str[:14], format="%Y%m%d%H%M%S")
    kma["dist_km"] = hav_km(*GYEONGJU, kma.kma_lat, kma.kma_lon)
    ev = kma[(kma.time >= SPAN[0]) & (kma.time <= SPAN[1]) & (kma.dist_km <= R_EVT_KM)
             & (kma.kma_mag >= MIN_MAG)].sort_values("kma_mag", ascending=False).head(N_EVENTS)
    print(f"events: {len(ev)} (M {ev.kma_mag.min():.1f}..{ev.kma_mag.max():.1f}, {ev.time.min().date()}..{ev.time.max().date()})")

    NS = ns_stations()
    import torch, seisbench.models as sbm
    pn = sbm.PhaseNet.from_pretrained("original")
    pn.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    man, picksT1, picksT2 = [], [], []
    t_start = time.time()
    for _, e in ev.iterrows():
        NS2 = NS.copy(); NS2["edist"] = hav_km(e.kma_lat, e.kma_lon, NS2.lat, NS2.lon)
        got = 0
        for _, s in NS2.sort_values("edist").iterrows():
            if got >= N_STA: break
            fs = day_files(s.sta, e.time)
            if len(fs) < 3: continue
            try:
                t0 = obspy.UTCDateTime(e.time) - PRE
                st = obspy.Stream()
                for f in fs: st += obspy.read(f, format="MSEED", starttime=t0, endtime=t0 + PRE + POST)
                st.merge(fill_value=0)
                st.trim(t0, t0 + PRE + POST)
                if not len(st) or min(len(tr.data) for tr in st) < 200 * (PRE + POST) * 0.9: continue
                for tr in st: tr.detrend("demean")
            except Exception:
                continue
            got += 1
            key = f"{e.event_id}_{s.sta}"
            tp, ts = predict_ps(s.edist, max(e.kma_dep, 1.0) if pd.notna(e.kma_dep) else 10.0)
            man.append(dict(event_id=e.event_id, sta=s.sta, edist_km=s.edist, mag=e.kma_mag,
                            origin=str(e.time), win_start=str(t0), pred_p=tp, pred_s=ts))
            # ---- T1: proper decimate -> 100 Hz ----
            st1 = st.copy()
            for tr in st1:
                tr.filter("lowpass", freq=40.0, corners=4, zerophase=True)
                tr.decimate(2, no_filter=True)
            st1.write(os.path.join(out, "win_T1", key + ".mseed"), format="MSEED")
            pn.sampling_rate = 100
            for p in pn.classify(st1, **THRESH).picks:
                picksT1.append(dict(key=key, sta=s.sta, phase=p.phase,
                                    t_rel=float(p.peak_time - t0), prob=float(p.peak_value)))
            # ---- T2: native 200 Hz ----
            pn.sampling_rate = 200
            for p in pn.classify(st, **THRESH).picks:
                picksT2.append(dict(key=key, sta=s.sta, phase=p.phase,
                                    t_rel=float(p.peak_time - t0), prob=float(p.peak_value)))
            # ---- T2' window for the PhaseNet+ driver: 200 Hz samples, header says 100 Hz ----
            st2 = st.copy()
            for tr in st2: tr.stats.sampling_rate = 100.0
            st2.write(os.path.join(out, "win_T2lie", key + ".mseed"), format="MSEED")
        print(f"  {e.event_id} M{e.kma_mag}: {got} stations  [{time.time()-t_start:.0f}s]")

    pd.DataFrame(man).to_csv(os.path.join(out, "windows_manifest.csv"), index=False)
    pd.DataFrame(picksT1).to_csv(os.path.join(out, "picks_pn_original_T1.csv"), index=False)
    pd.DataFrame(picksT2).to_csv(os.path.join(out, "picks_pn_original_T2.csv"), index=False)
    print(f"windows {len(man)} | T1 picks {len(picksT1)} | T2 picks {len(picksT2)} | {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
