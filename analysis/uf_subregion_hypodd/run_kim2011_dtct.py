#!/usr/bin/env python
"""Build the **kim2011-velocity dt.ct-ONLY** HypoDD run (03b.dt.ct_kim2011) — the catalog-differential-time
counterpart of run_kim2011_dtcc.py, for the nb21 three-way comparison (HypoInverse vs dt.ct vs dt.cc).

WHY: nb21 (build_reloccmp_nb.py) compares absolute / dt.ct / dt.cc locations. To make it apples-to-apples with
the PRIMARY kim2011 dt.cc catalog (03.dt.cc_kim2011), the dt.ct leg must use the SAME kim2011 velocity AND the
SAME current event set. The old 02b.dt.ct run is (a) generic-velocity and (b) built on the pre-rebuild Jun-25
event.dat (stale). This run fixes both: kim2011 velocity, current 02.dt.cc/event.dat (2776-set).

WHAT IT DOES (only the velocity block differs from 02b.dt.ct's setup):
  - hypoDD.inp = 02b.dt.ct/hypoDD.inp with ONLY the 1D-model block swapped to kim2011 (dt.ct weighting/DAMP/
    ISOLV=2 kept verbatim — isolates velocity exactly as run_kim2011_dtcc does for the dt.cc leg).
  - inputs: symlink dt.ct -> ../02.dt.cc/dt.ct (current); copy event.dat + station.dat from 02.dt.cc (current).
  - run hypoDD -> 03b.dt.ct_kim2011/hypoDD.reloc + hypoDD.loc (absolute starts, model-independent).
"""
import os, shutil, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_kim2011_dtcc import swap_velocity  # reuse the exact 1D-model swap

RUN = ("/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/"
       "runs/uf_subregion_reuse/2.HypoDD")
GEN = os.path.join(RUN, "02.dt.cc")           # current inputs (event.dat/station.dat/dt.ct)
SRC = os.path.join(RUN, "02b.dt.ct")          # generic dt.ct-only run (source of the dt.ct-tuned hypoDD.inp)
OUT = os.path.join(RUN, "03b.dt.ct_kim2011")  # this script's output


def main():
    for f in ["hypoDD.inp"]:
        if not os.path.exists(os.path.join(SRC, f)):
            sys.exit(f"{SRC}/{f} missing — need the generic 02b.dt.ct run as the inp template.")
    for f in ["event.dat", "station.dat", "dt.ct"]:
        if not os.path.exists(os.path.join(GEN, f)):
            sys.exit(f"{GEN}/{f} missing — run the pipeline --through dtcc first.")

    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.startswith("hypoDD.") and f != "hypoDD.inp":
            os.remove(os.path.join(OUT, f))

    # hypoDD.inp = generic dt.ct inp with kim2011 velocity (everything else identical: dt.ct weighting, ISOLV=2)
    inp = swap_velocity(open(os.path.join(SRC, "hypoDD.inp")).read())
    open(os.path.join(OUT, "hypoDD.inp"), "w").write(inp)

    for f, link in [("dt.ct", True), ("event.dat", False), ("station.dat", False)]:
        dst = os.path.join(OUT, f)
        if os.path.lexists(dst):
            os.remove(dst)
        if link:
            os.symlink(os.path.join("..", "02.dt.cc", f), dst)
        else:
            shutil.copyfile(os.path.join(GEN, f), dst)

    print("running hypoDD (kim2011 velocity, dt.ct only) ...", flush=True)
    proc = subprocess.run(["hypoDD", "hypoDD.inp"], cwd=OUT, capture_output=True, text=True, errors="replace")
    open(os.path.join(OUT, "hypoDD.stdout"), "w").write(proc.stdout)
    reloc = os.path.join(OUT, "hypoDD.reloc")
    if proc.returncode != 0 or not os.path.exists(reloc):
        print(proc.stdout[-1500:]); print(proc.stderr[-1500:], file=sys.stderr)
        sys.exit(f"hypoDD failed (rc={proc.returncode}); reloc exists={os.path.exists(reloc)}")
    rids = [int(ln.split()[0]) for ln in open(reloc) if ln.split()]
    print(f"\nkim2011 dt.ct reloc: {len(rids)} events -> {reloc}")


if __name__ == "__main__":
    main()
