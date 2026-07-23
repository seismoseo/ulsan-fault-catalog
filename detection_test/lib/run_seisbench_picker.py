#!/usr/bin/env python
"""Month picking with a SeisBench picker (parameterized by --month).

USAGE (env: base, GPU auto):
    python run_seisbench_picker.py --model original --month 2021-09    # PhaseNet original (NCEDC-trained)
    python run_seisbench_picker.py --model stead    --month 2021-09    # PhaseNet STEAD-retrained
    python run_seisbench_picker.py --model eqt      --month 2021-09    # EQTransformer STEAD-trained (Mousavi)

Reads the station table written by build_stations.py (cache/stations_<YYYY_MM>.csv), picks every usable
station-day of the month, and writes ONE standardized parquet:
    picks/picks_<model>_<YYYY_MM>.parquet   columns: net, sta, phase, time (UTC), prob, picker
plus a sidecar json recording the exact thresholds used.

Data handling (disclosed):
  * each station's day files come from its own archive (the `archive` column: KS_KG, NS/, or GJ/);
  * bands are HH/EL/HG (chosen per station by build_stations.py, priority HH>EL>HG);
  * if a trace's sampling rate exceeds 100 Hz (e.g. NS/GJ at 200 Hz) it is lowpass-filtered (0.4x target,
    zero phase) and decimated by an integer factor BEFORE picking (proper anti-alias; the Buan lesson);
  * traces are demeaned; no other preprocessing (PhaseNet-style pickers expect raw data).
"""
import os, sys, glob, json, time, argparse
import numpy as np
import pandas as pd
import obspy

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # detection_test/
ARCH_DEFAULT = "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG"
import gj_config as GJC                     # single disclosed source of parameters
TARGET_FS = GJC.TARGET_FS

_THR = float(GJC.PICK_PROB)   # single disclosed threshold, SAME for P and S on every picker (=0.2, matches PN+)
MODELS = {   # SeisBench pretrained name + explicit classify thresholds (recorded in sidecar); consistent _THR
    "original":  ("PhaseNet",      "original", dict(P_threshold=_THR, S_threshold=_THR)),   # PhaseNet NCEDC-trained
    "stead":     ("PhaseNet",      "stead",    dict(P_threshold=_THR, S_threshold=_THR)),   # PhaseNet STEAD-retrained
    "eqt":       ("EQTransformer",  "stead",   dict(detection_threshold=_THR, P_threshold=_THR, S_threshold=_THR)),
}   # 'eqt' = EQTransformer STEAD (Mousavi 2020); detection_threshold also set to _THR so P/S gating dominates


def day_stream(archive, sta, band, Y, doy, decimate=True):
    """Read + prep one station-day (all components of the chosen band from that station's archive).

    decimate=True  -> anti-alias downsample anything >100 Hz to 100 Hz (proper 100-Hz input; the baseline).
    decimate=False -> NATIVE: leave the native rate untouched (200 Hz fed as-is). The caller must then set
                      model.sampling_rate to the stream's native rate so SeisBench does NOT resample -> the
                      model consumes the raw samples ('fool it into seeing 200 Hz as a longer 100-Hz trace')."""
    st = obspy.Stream()
    for f in sorted(glob.glob(os.path.join(archive, sta, f"{band}?.D", f"*.{Y}.{doy:03d}"))):
        try:
            st += obspy.read(f, format="MSEED")
        except Exception as e:
            print(f"    read error {os.path.basename(f)}: {e}")
    if not len(st):
        return None
    for tr in st:                                            # decimate BEFORE merge so a channel whose sub-traces
        tr.detrend("demean")                                 # have mixed rates (a mid-day rate change) unifies
        fs = tr.stats.sampling_rate
        if decimate and fs > TARGET_FS:                      # proper anti-alias decimation
            fac = int(round(fs / TARGET_FS))
            tr.filter("lowpass", freq=GJC.ANTIALIAS_FRAC * TARGET_FS, corners=GJC.ANTIALIAS_CORNERS, zerophase=True)
            tr.decimate(fac, no_filter=True)
    try:
        st.merge(fill_value=0)
    except Exception:                                        # still-differing rates -> force a common rate
        for tr in st:
            if tr.stats.sampling_rate != TARGET_FS:
                tr.resample(TARGET_FS)
        st.merge(fill_value=0)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--month", default="2021-09", help="YYYY-MM")
    ap.add_argument("--native", action="store_true",
                    help="feed native sampling rate WITHOUT decimation (200 Hz fed as-is); "
                         "for the decimate-vs-native performance test")
    ap.add_argument("--predecimated", action="store_true",
                    help="read 200 Hz (NS) stations from the pre-decimated 100 Hz mirror NS_100hz/ instead "
                         "(detection-only speed-up; identical result, ~10x faster prep). Original 200 Hz "
                         "archive is left untouched for relocation/cross-correlation.")
    ap.add_argument("--min-coverage", type=float, default=GJC.MIN_COVERAGE,
                    help="min local coverage to include a station (0 = use ALL stations with any data)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    Y, MO = int(a.month[:4]), int(a.month[5:7]); tag = f"{Y}_{MO:02d}"
    otag = f"{tag}_native" if a.native else tag          # cache uses base tag; outputs carry _native
    T0 = pd.Timestamp(f"{a.month}-01"); T1 = T0 + pd.offsets.MonthEnd(0)
    DOYS = range(T0.dayofyear, T1.dayofyear + 1)

    import torch
    import seisbench.models as sbm
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cls_name, weights, thresholds = MODELS[a.model]
    model = getattr(sbm, cls_name).from_pretrained(weights)
    model.to(torch.device(device))
    print(f"[{a.model}|{tag}] {cls_name}.from_pretrained('{weights}') on {device} | thresholds {thresholds}")

    S = pd.read_csv(os.path.join(HERE, "cache", f"stations_{tag}.csv"))
    S = S[(S.coverage > 0) & (S.coverage >= a.min_coverage)].reset_index(drop=True)
    if "archive" not in S.columns:
        S["archive"] = ARCH_DEFAULT                          # back-compat with pre-archive caches (2014)
    if a.predecimated:                                       # read NS 200 Hz stations from the 100 Hz mirror
        S["archive"] = S.archive.apply(lambda p: p[:-3] + "/NS_100hz" if p.endswith("/NS") else p)
    print(f"[{a.model}|{otag}] {len(S)} stations x {len(list(DOYS))} days | predecimated={a.predecimated}")

    default_sr = float(model.sampling_rate)                   # 100 Hz (the trained rate)
    ckpt_dir = os.path.join(HERE, "picks", f"_ckpt_{a.model}_{otag}")   # station-level checkpoints (crash-safe + resume)
    os.makedirs(ckpt_dir, exist_ok=True)
    COLS = ["net", "sta", "phase", "time", "prob"]
    t0 = time.time()
    for _, r in S.iterrows():
        fpart = os.path.join(ckpt_dir, f"{r.net}.{r.sta}.parquet")
        if os.path.exists(fpart):                              # already done in a previous (crashed) run -> resume
            continue
        srows = []
        for doy in DOYS:
            try:
                st = day_stream(r.archive, r.sta, r.band, Y, doy, decimate=not a.native)
                if st is None:
                    continue
                model.sampling_rate = float(round(st[0].stats.sampling_rate)) if a.native else default_sr
                try:
                    out = model.classify(st, **thresholds)
                except TypeError:                              # kwarg-name drift guard
                    out = model.classify(st)
                for p in out.picks:
                    srows.append(dict(net=r.net, sta=r.sta, phase=p.phase,
                                      time=str(p.peak_time), prob=float(p.peak_value)))
            except Exception as e:                             # one bad station-day must not kill the run
                print(f"    skip {r.net}.{r.sta} doy{doy}: {e}", flush=True)
        pd.DataFrame(srows, columns=COLS).to_parquet(fpart, index=False)   # checkpoint (written even if empty)
        print(f"  {r.net}.{r.sta}: {len(srows)} picks   [{time.time()-t0:.0f}s]", flush=True)

    parts = [pd.read_parquet(f) for f in glob.glob(os.path.join(ckpt_dir, "*.parquet"))]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)
    df["picker"] = a.model
    os.makedirs(os.path.join(HERE, "picks"), exist_ok=True)
    out_path = os.path.join(HERE, "picks", f"picks_{a.model}_{otag}.parquet")
    df.to_parquet(out_path, index=False)
    with open(os.path.join(HERE, "picks", f"picks_{a.model}_{otag}.json"), "w") as fh:
        json.dump(dict(model=cls_name, weights=weights, thresholds=thresholds, month=a.month,
                       native=a.native, device=device, n_stations=len(S), n_picks=len(df),
                       runtime_s=round(time.time() - t0, 1)), fh, indent=1)
    print(f"[{a.model}|{otag}] wrote {out_path}  ({len(df)} picks, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
