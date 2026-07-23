#!/usr/bin/env python
"""Whole-subregion HypoDD relocation of the Ulsan-Fault box as ONE cluster (all-pairs dt.cc).

Runs the validated PocketQuake / korea-cluster-relocation pipeline for a single cluster
`uf_subregion_reuse` whose members are every UF-box event (clean Heo catalog) — see make_uf_catalog.py.
Reuses the existing event waveforms + PhaseNet+ picks (no re-download, no re-pick). All-pairs dt.cc:
the framework cross-correlates every event pair at all shared stations and HypoDD keeps a link only
where CC>=0.7 at >=OBSCC stations, so the multi-station CC arbitrates (immune to the pairing/mislocation
problem). Isolated in uf_subregion_hypodd/ so it never touches the per-family relocation/ run.

Stages (mirrors relocation/batch_relocate.run_family, single cluster, NO per-step timeout):
  scaffold -> stage(reuse picks) -> gather -> hypoinverse -> ph2dt -> dt.ct -> [rereference -> dt.cc -> dtcc]

Usage:
  python run_uf_hypodd.py --through dtct      # validate: HypoInverse + ph2dt + dt.ct (fast)
  python run_uf_hypodd.py --through dtcc      # full: + all-pairs GPU dt.cc + HypoDD  (long; use pq-gpu)
  python run_uf_hypodd.py --through dtcc --skip-scaffold   # resume after scaffold/stage/gather done
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PQ = "/home/msseo/works/15.PocketQuake"
PIPE = os.path.join(PQ, "external", "korea-cluster-relocation")
HYPO = "/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv"
RELOC = "/home/msseo/works/02.Ulsan_Fault_detection/analysis/relocation"   # scaffold_offline.py / stage.py
RUNS = os.path.join(PIPE, "pipeline", "runs")
SLUG = "uf_subregion_reuse"
SUB = os.path.join(HERE, "uf_subregion")
ENV = {**os.environ, "PYTHONPATH": f"{PQ}:{PIPE}:{HYPO}"}
PY = sys.executable


def run(cmd, cwd):
    print(f"\n$ (cwd={cwd}) {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=ENV, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--through", default="dtct", choices=["dtct", "dtcc"])
    ap.add_argument("--skip-scaffold", action="store_true",
                    help="skip scaffold/stage/gather (already done) — go straight to location stages")
    a = ap.parse_args()

    epi, rb = open(os.path.join(SUB, "scaffold_args.txt")).read().split()[1::2]
    t0 = time.perf_counter()

    if not a.skip_scaffold:
        run([PY, os.path.join(RELOC, "scaffold_offline.py"), SLUG,
             "--catalog", os.path.join(SUB, "catalog_kma.csv"),
             "--epicenter", epi, "--region-bounds", rb], RELOC)
        run([PY, os.path.join(RELOC, "stage.py"), SLUG, "--reuse-picks",
             "--members", os.path.join(SUB, "members.txt")], RELOC)
        run([PY, "-m", "pipeline.cli.run_pipeline", "--cluster", SLUG,
             "--stage-from", "stations", "--through", "waveforms"], PIPE)

    run([PY, "-m", "pipeline.cli.run_pipeline", "--cluster", SLUG,
         "--stage-from", "hypoinverse", "--through", a.through, "--arc-velmodel", "kim2011"], PIPE)

    reloc = os.path.join(RUNS, SLUG, "2.HypoDD",
                         "02.dt.cc" if a.through == "dtcc" else "01.dt.ct", "hypoDD.reloc")
    print(f"\n=== through {a.through} in {time.perf_counter()-t0:.0f}s ===")
    print(f"reloc: {reloc}  exists={os.path.exists(reloc)}")
    if os.path.exists(reloc):
        print(f"  lines: {sum(1 for _ in open(reloc))}")


if __name__ == "__main__":
    main()
