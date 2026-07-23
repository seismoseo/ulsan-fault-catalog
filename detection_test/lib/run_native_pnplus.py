#!/usr/bin/env python
"""PhaseNet+ (EQNet) leg of the native-rate benchmark: pick the T1 and T2' window files.

USAGE (env: eqnet, after run_native_rate.py has written the windows):
    conda run -n eqnet python run_native_pnplus.py

Treatments (window files prepared by run_native_rate.py):
  win_T1/    100 Hz, properly decimated               -> pick as-is
  win_T2lie/ 200 Hz samples with header fs=100        -> EQNet sees "100 Hz", judges in samples;
             pick times must be mapped t_true = t_win0 + (t_pick - t_win0)/2 (done in the 05 notebook)

Outputs: cache/native_rate/picks_pnplus_T1.csv, picks_pnplus_T2lie.csv
         (key, sta, phase, t_rel_raw [s in the FILE timebase], prob)
"""
import os, sys, glob, json, time, tempfile
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "cache", "native_rate")
BUAN = "/home/msseo/works/20.2024_Buan_EQ_DL_detection/pipeline"
MIN_PROB = 0.2

sys.path.insert(0, BUAN)
import core, config
import torch, torch.utils.data


def run_treatment(net, DatasetCls, tag):
    files = sorted(glob.glob(os.path.join(OUT, f"win_{tag}", "*.mseed")))
    print(f"[{tag}] {len(files)} window files")
    rows, t0 = [], time.time()
    # one data_list line per window file (each holds all 3 components)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write("\n".join(files)); list_path = tf.name
    try:
        dataset = DatasetCls(data_path="", data_list=list_path, format="mseed",
                             dataset="seismic_trace", training=False,
                             sampling_rate=config.SAMPLING_RATE,
                             highpass_filter=config.PNPLUS_HIGHPASS,
                             cut_patch=False, nt=config.PNPLUS_NT)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0,
                                             collate_fn=None, drop_last=False)
        with torch.inference_mode():
            for meta in loader:
                r = core._pnplus_infer(net, meta, MIN_PROB)
                fname = meta["file_name"][0] if isinstance(meta["file_name"], (list, tuple)) else meta["file_name"]
                key = os.path.basename(str(fname)).replace(".mseed", "")
                t_begin = pd.Timestamp(meta["begin_time"][0] if isinstance(meta["begin_time"], (list, tuple))
                                       else meta["begin_time"])
                for plist in r["picks"]:
                    for p in plist:
                        t_rel = (pd.Timestamp(p["phase_time"]) - t_begin).total_seconds()
                        rows.append(dict(key=key, sta=key.split("_")[-1], phase=p["phase_type"].upper(),
                                         t_rel_raw=t_rel, prob=float(p["phase_score"])))
    finally:
        os.unlink(list_path)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"picks_pnplus_{tag}.csv"), index=False)
    print(f"[{tag}] {len(df)} picks  ({time.time()-t0:.0f}s)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = core._load_pnplus(device)
    DatasetCls = core._make_antialias_dataset_cls()
    for tag in ("T1", "T2lie"):
        run_treatment(net, DatasetCls, tag)


if __name__ == "__main__":
    main()
