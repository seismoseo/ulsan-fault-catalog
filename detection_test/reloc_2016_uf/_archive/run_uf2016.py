#!/usr/bin/env python
"""2016 UF whole-box HypoDD relocation driver (clone of analysis/uf_subregion_hypodd/run_uf_hypodd.py for uf_2016).
Chain: build catalog_kma -> GJ-inclusive scaffold -> stage(reuse our PhaseNet+ picks, event_idx SAC store) ->
       HypoInverse(kim2011) -> ph2dt -> dt.ct  [-> re-reference -> all-pairs dt.cc -> HypoDD].
  python run_uf2016.py --through dtct     # validate: HypoInverse + ph2dt + dt.ct (fast, base env)
  python run_uf2016.py --through dtcc     # full: + all-pairs GPU dt.cc + HypoDD (long; pq-gpu for the dt.cc stage)
  python run_uf2016.py --through dtcc --skip-scaffold   # resume after scaffold/stage/gather
"""
import argparse, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
PQ = "/home/msseo/works/15.PocketQuake"
PIPE = os.path.join(PQ, "external", "korea-cluster-relocation")
HYPO = "/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv"
RELOC = "/home/msseo/works/02.Ulsan_Fault_detection/analysis/relocation"   # stage.py
RUNS = os.path.join(PIPE, "pipeline", "runs")
SLUG = "uf_2016"
ENV = {**os.environ, "PYTHONPATH": f"{PQ}:{PIPE}:{HYPO}"}
PY = sys.executable


def run(cmd, cwd, conda_env=None):
    if conda_env:
        cmd = ["conda", "run", "-n", conda_env, "python3"] + cmd[1:]
    print(f"\n$ (cwd={cwd}) {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=ENV, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--through", default="dtct", choices=["dtct", "dtcc"])
    ap.add_argument("--skip-scaffold", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    if not a.skip_scaffold:
        run([PY, os.path.join(HERE, "build_catalog_kma.py")], HERE)                       # catalog_kma + members
        run([PY, os.path.join(HERE, "scaffold_2016.py")], PIPE)                           # GJ-inclusive scaffold
        run([PY, os.path.join(RELOC, "stage.py"), SLUG, "--reuse-picks",
             "--members", os.path.join(HERE, "members.txt"),
             "--wf-root", os.path.join(HERE, "event_sac"),
             "--catalog", os.path.join(HERE, "members_event_idx.csv")], RELOC)            # bridge event_idx SAC + reuse picks
        run([PY, "-m", "pipeline.cli.run_pipeline", "--cluster", SLUG,
             "--stage-from", "stations", "--through", "waveforms"], PIPE)
    # HypoInverse -> ph2dt -> dt.ct [-> dt.cc]. dt.cc needs the GPU xcorr env (pq-gpu).
    env = "pq-gpu" if a.through == "dtcc" else None
    run([PY, "-m", "pipeline.cli.run_pipeline", "--cluster", SLUG, "--stage-from", "hypoinverse",
         "--through", a.through, "--velmodels", "kim2011", "--arc-velmodel", "kim2011"], PIPE, conda_env=env)
    reloc = os.path.join(RUNS, SLUG, "2.HypoDD", "02.dt.cc" if a.through == "dtcc" else "01.dt.ct", "hypoDD.reloc")
    print(f"\n=== through {a.through} in {time.perf_counter()-t0:.0f}s ===")
    print(f"reloc: {reloc}  exists={os.path.exists(reloc)}")
    if os.path.exists(reloc):
        print(f"  relocated events: {sum(1 for _ in open(reloc))}")


if __name__ == "__main__":
    main()
