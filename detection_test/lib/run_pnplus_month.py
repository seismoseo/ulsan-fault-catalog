#!/usr/bin/env python
"""Month picking with PhaseNet+ (EQNet), parameterized by --month.

USAGE (env: eqnet, GPU auto):
    conda run -n eqnet python run_pnplus_month.py --month 2021-09

Reuses the PROVEN Buan pipeline pieces IN-PROCESS (no edits there):
  _load_pnplus / _make_antialias_dataset_cls / _pnplus_infer from
  20.2024_Buan_EQ_DL_detection/pipeline/core.py — only the file discovery and output
  paths are ours. min_prob=0.2 and highpass follow the Buan v2 settings. The anti-alias
  dataset resamples every trace to config.SAMPLING_RATE=100 Hz, so NS/GJ 200 Hz is handled.

Each station's day files come from its own `archive` column (KS_KG, NS/, or GJ/).

Output: picks/picks_phasenet_plus_<YYYY_MM>.parquet  (net, sta, phase, time, prob, picker)
        picks/pnplus_raw_<YYYY_MM>/picks_<Y>.DDD.csv  (raw picks incl. polarity/amplitude)
"""
import os, sys, glob, json, time, tempfile, argparse
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # detection_test/
ARCH_DEFAULT = "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG"
BUAN = "/home/msseo/works/20.2024_Buan_EQ_DL_detection/pipeline"
import gj_config as GJC                                # ALL our params from gj_config.py (distinct from Buan `config`)
MIN_PROB = GJC.PICK_PROB                               # PhaseNet+ pick threshold (single disclosed source)

sys.path.insert(0, BUAN)
import core, config                                    # Buan pipeline (read-only reuse)

import torch
import torch.utils.data


def day_lines(stations_df, Y, doy):
    """One data_list line per station = comma-joined component files (from that station's archive)."""
    lines, stas = [], []
    for _, r in stations_df.iterrows():
        comps = sorted(glob.glob(os.path.join(r.archive, r.sta, f"{r.band}?.D", f"*.{Y}.{doy:03d}")))
        if comps:
            lines.append(",".join(comps)); stas.append(r.sta)
    return lines, stas


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--month", default="2021-09", help="YYYY-MM")
    ap.add_argument("--predecimated", action="store_true",
                    help="read NS 200 Hz stations from the pre-decimated 100 Hz mirror (detection speed-up)")
    ap.add_argument("--doy-start", type=int, default=0, help="restrict to a day-of-year sub-range (sharding)")
    ap.add_argument("--doy-end", type=int, default=0)
    ap.add_argument("--min-coverage", type=float, default=GJC.MIN_COVERAGE,
                    help="min local coverage to include a station (0 = use ALL stations with any data)")
    a = ap.parse_args()
    Y, MO = int(a.month[:4]), int(a.month[5:7]); tag = f"{Y}_{MO:02d}"
    T0 = pd.Timestamp(f"{a.month}-01"); T1 = T0 + pd.offsets.MonthEnd(0)
    d0, d1 = T0.dayofyear, T1.dayofyear
    shard = a.doy_start > 0 or a.doy_end > 0             # PN+ is CPU-bound; shard days across parallel instances
    if shard: d0, d1 = (a.doy_start or d0), (a.doy_end or d1)
    MONTH_DOY = range(d0, d1 + 1)
    otag = f"{tag}_d{d0}-{d1}" if shard else tag         # shards write partial parquets; final assembly run (no
    #                                                       shard flags) re-reads all cached raw csvs -> canonical
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[phasenet_plus|{tag}] device={device} min_prob={MIN_PROB} highpass={config.PNPLUS_HIGHPASS}")
    net = core._load_pnplus(device)
    DatasetCls = core._make_antialias_dataset_cls()

    S = pd.read_csv(os.path.join(HERE, "cache", f"stations_{tag}.csv"))
    S = S[(S.coverage > 0) & (S.coverage >= a.min_coverage)].copy()
    if "archive" not in S.columns:
        S["archive"] = ARCH_DEFAULT                          # back-compat with pre-archive caches
    if a.predecimated:                                       # NS 200 Hz -> 100 Hz mirror (detection only)
        S["archive"] = S.archive.apply(lambda p: p[:-3] + "/NS_100hz" if p.endswith("/NS") else p)
    raw_dir = os.path.join(HERE, "picks", f"pnplus_raw_{tag}"); os.makedirs(raw_dir, exist_ok=True)
    print(f"[phasenet_plus|{tag}] {len(S)} stations x {len(list(MONTH_DOY))} days")

    rows = []
    for doy in MONTH_DOY:
        raw_csv = os.path.join(raw_dir, f"picks_{Y}.{doy:03d}.csv")
        if os.path.exists(raw_csv):                          # day-level resume
            raw = pd.read_csv(raw_csv)
            for _, p in raw.iterrows():
                parts = str(p["station_id"]).strip().strip(",").split(".")
                rows.append(dict(net=parts[0], sta=parts[1], phase=p["phase_type"],
                                 time=str(p["phase_time"]), prob=float(p["phase_score"])))
            print(f"  doy {doy}: cached ({len(raw)} picks)")
            continue
        lines, _ = day_lines(S, Y, doy)
        if not lines:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
            tf.write("\n".join(lines)); list_path = tf.name
        try:
            dataset = DatasetCls(data_path="", data_list=list_path, format="mseed",
                                 dataset="seismic_trace", training=False,
                                 sampling_rate=config.SAMPLING_RATE,
                                 highpass_filter=config.PNPLUS_HIGHPASS,
                                 cut_patch=False, nt=config.PNPLUS_NT)
            dataset.cut_patch = True
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8,
                                                 collate_fn=None, drop_last=False, pin_memory=(device == "cuda"))
            picks_all = []
            with torch.inference_mode():
                for meta in loader:
                    if device == "cuda":                 # 4.4x speed-up: the Buan wrapper leaves input on CPU and
                        meta["data"] = meta["data"].cuda(non_blocking=True)   # eats a slow transfer every forward
                    r = core._pnplus_infer(net, meta, MIN_PROB)
                    for p in r["picks"]:
                        picks_all.extend(p)
        finally:
            os.unlink(list_path)
        if picks_all:
            raw = pd.DataFrame(picks_all)
            raw.to_csv(os.path.join(raw_dir, f"picks_{Y}.{doy:03d}.csv"), index=False)
            for _, p in raw.iterrows():
                sid = str(p["station_id"]).strip().strip(",")           # e.g. KS.HDB.. or KG.HDB..HH
                parts = sid.split(".")
                rows.append(dict(net=parts[0], sta=parts[1], phase=p["phase_type"],
                                 time=str(p["phase_time"]), prob=float(p["phase_score"])))
        print(f"  doy {doy}: {len(picks_all)} picks  [{time.time()-t0:.0f}s]")

    df = pd.DataFrame(rows); df["picker"] = "phasenet_plus"
    out_path = os.path.join(HERE, "picks", f"picks_phasenet_plus_{otag}.parquet")
    df.to_parquet(out_path, index=False)
    with open(os.path.join(HERE, "picks", f"picks_phasenet_plus_{otag}.json"), "w") as fh:
        json.dump(dict(model="EQNet phasenet_plus (Buan weights model_99.pth)", min_prob=MIN_PROB,
                       month=a.month, highpass=config.PNPLUS_HIGHPASS, device=device,
                       n_stations=len(S), n_picks=len(df), runtime_s=round(time.time() - t0, 1)), fh, indent=1)
    print(f"[phasenet_plus|{otag}] wrote {out_path}  ({len(df)} picks, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
