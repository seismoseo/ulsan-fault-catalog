#!/usr/bin/env python
"""Whole-box dt.cc relocation of the QC'd 596-event UF 2016 subset (uf_2016_qc cluster).

The full 3867-event run established: native-rate SAC -> HypoInverse(kim2011) all 3867 -> uf_cluster QC
(erh<5,erz<5,gap<270,num>5,rms<1.0) -> 596 well-located events. This driver relocates ONLY those 596
(catalog_kma_qc.csv) with dt.cc:
  scaffold(uf_2016_qc, GJ-incl) -> stage(reuse picks, event_idx SAC) -> stations..ph2dt (HypoInverse kim2011
  + dt.ct) -> rereference..dtcc (rereference to HypoInverse origins -> all-pairs GPU xcorr -> ASSEMBLE 02.dt.cc).
The pipeline's own dtct/dtcc HypoDD (ISTART=1/DAMP=8/generic) air-quakes on this set, but (a) xcorr does NOT need
the dtct reloc (rereference uses the HypoInverse solution) and (b) run_dtcc assembles 02.dt.cc BEFORE its HypoDD
attempt — so the dt.cc inputs are produced regardless. We then run the VALIDATED HypoDD ourselves:
  run_hypodd_kim2011_istart2.py on 02.dt.cc  (kim2011 velocity + ISTART=2 + adaptive LSQR damping, CND 40-80).

Usage:  python run_uf2016_qc.py            # scaffold..xcorr + final kim2011/ISTART=2 dt.cc reloc
        python run_uf2016_qc.py --skip-setup   # resume: only rereference..dtcc + final HypoDD
"""
import argparse, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
PQ = "/home/msseo/works/15.PocketQuake"
PIPE = os.path.join(PQ, "external", "korea-cluster-relocation")
RELOC = "/home/msseo/works/02.Ulsan_Fault_detection/analysis/relocation"
HYPO = "/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv"
RUNS = os.path.join(PIPE, "pipeline", "runs")
SLUG = "uf_2016_qc"
ENV = {**os.environ, "PYTHONPATH": f"{PQ}:{PIPE}:{HYPO}"}
PY = sys.executable


def run(cmd, cwd, conda_env=None):
    if conda_env:
        cmd = ["conda", "run", "-n", conda_env, "python3"] + cmd[1:]
    print(f"\n$ (cwd={cwd}) {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=ENV, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-setup", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    if not a.skip_setup:
        run([PY, os.path.join(HERE, "scaffold_2016.py"), "--slug", SLUG,
             "--catalog", os.path.join(HERE, "catalog_kma_qc.csv")], PIPE)
        run([PY, os.path.join(RELOC, "stage.py"), SLUG, "--reuse-picks",
             "--members", os.path.join(HERE, "members_qc.txt"),
             "--wf-root", os.path.join(HERE, "event_sac"),
             "--catalog", os.path.join(HERE, "members_event_idx_qc.csv")], RELOC)
        run([PY, "-m", "pipeline.cli.run_pipeline", "--cluster", SLUG,
             "--stage-from", "stations", "--through", "ph2dt", "--velmodels", "kim2011",
             "--arc-velmodel", "kim2011"], PIPE)
    # rereference (HypoInverse origins) -> all-pairs GPU xcorr -> assemble 02.dt.cc. dtcc HypoDD may air-quake;
    # we ignore its reloc and run our own below. GPU xcorr needs the pq-gpu env.
    try:
        run([PY, "-m", "pipeline.cli.run_pipeline", "--cluster", SLUG, "--stage-from", "rereference",
             "--through", "dtcc", "--velmodels", "kim2011", "--arc-velmodel", "kim2011"], PIPE, conda_env="pq-gpu")
    except subprocess.CalledProcessError as e:
        print(f"\n[expected] pipeline dtcc HypoDD failed ({e.returncode}); 02.dt.cc inputs are assembled -> "
              f"running our kim2011/ISTART=2/adaptive HypoDD next.", flush=True)
    dtcc = os.path.join(RUNS, SLUG, "2.HypoDD", "02.dt.cc")
    for f in ("event.dat", "dt.ct", "station.dat", "hypoDD.inp"):
        assert os.path.exists(os.path.join(dtcc, f)), f"02.dt.cc/{f} missing — xcorr/assembly did not complete"
    run([PY, os.path.join(HERE, "run_hypodd_kim2011_istart2.py"), dtcc], HERE)
    reloc = os.path.join(dtcc, "hypoDD.reloc")
    print(f"\n=== uf_2016_qc dt.cc done in {time.perf_counter()-t0:.0f}s ===")
    print(f"reloc: {reloc} exists={os.path.exists(reloc)}"
          + (f"  events={sum(1 for _ in open(reloc))}" if os.path.exists(reloc) else ""))


if __name__ == "__main__":
    main()
