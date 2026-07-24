#!/usr/bin/env python
"""Relocate a prepared HypoDD directory with the VALIDATED UF whole-box config: kim2011 velocity + ISTART=2
(start from catalog HYPOINVERSE locations, not the centroid) + ISOLV=2 (LSQR) + per-set ADAPTIVE damping
(condition number driven into 40-80). This is the method the prior UF work used (run_kim2011_dtcc.swap_velocity
+ run_generic_istart2_adaptive.adaptive + run_svd_volumes engine); the korea-cluster pipeline's DEFAULT dtct
is only a fixed-DAMP=8/ISTART=1 regression baseline that diverges (negative-depth airquakes -> empty reloc) on
this 3856-event GJ-densified cluster.

Reuses, unchanged:
  - run_kim2011_dtcc.swap_velocity  -> swap the 1D-model block to kim2011 (Kim et al. 2011 SE-Korea)
  - run_svd_volumes._exec_hypodd_once / _max_cnd_per_set / _isnum / HYPODD (~/bin/hypoDD, recompiled MAXDATA=15M)

Operates on the pipeline-produced dir (its hypoDD.inp already has the right input filenames, weighting schedule,
DIST, OBSCC/OBSCT). Only ISTART/ISOLV, the 1D model, and DAMP are changed. Backs up any existing reloc.

    python run_hypodd_kim2011_istart2.py <hypoDD_dir>          # e.g. .../2.HypoDD/01.dt.ct
"""
import argparse, os, re, shutil, sys

UF_HD = "/home/msseo/works/02.Ulsan_Fault_detection/analysis/uf_subregion_hypodd"
sys.path.insert(0, UF_HD)
import run_svd_volumes as rsv                     # engine: _exec_hypodd_once, _max_cnd_per_set, _isnum, HYPODD
from run_kim2011_dtcc import swap_velocity        # kim2011 1D-model swap (verbatim)


def make_inp(template, damps, istart=2, isolv=2):
    """template (a hypoDD.inp string) -> kim2011 velocity + ISTART/ISOLV set + the 7 weighting-row DAMP replaced."""
    txt = swap_velocity(template)                                            # 1D model -> kim2011
    txt = re.sub(r"(\*--- solution control: ISTART ISOLV NSET\n\s*)\d+\s+\d+(\s+\d+)",
                 rf"\g<1>{istart}       {isolv}\g<2>", txt)                  # ISTART/ISOLV
    lines, wi, insec = txt.splitlines(keepends=True), 0, False
    for i, ln in enumerate(lines):
        if "data weighting" in ln:
            insec = True; continue
        if "1D model" in ln:
            insec = False
        t = ln.split()
        if insec and len(t) == 10 and all(rsv._isnum(x) for x in t):
            dmp = damps[wi] if wi < len(damps) else damps[-1]
            lines[i] = "    " + "  ".join(t[:-1] + [str(dmp)]) + "\n"; wi += 1
    assert wi >= 1, "found no weighting rows to set DAMP on"   # 5 rows for dt.ct, 7 for dt.cc
    return "".join(lines)


def adaptive(d, template, cnd_range=(40.0, 80.0), max_attempts=12, damp0=60):
    """Per-set adaptive LSQR damping targeting CND in cnd_range (vendored from run_generic_istart2_adaptive)."""
    lo, hi = cnd_range; mid = (lo + hi) / 2.0
    damps, best, hist = [int(damp0)] * 7, None, []
    for _ in range(max_attempts):
        txt = make_inp(template, tuple(damps))            # build BEFORE opening for write (never truncate template)
        open(os.path.join(d, "hypoDD.inp"), "w").write(txt)
        try:
            rsv._exec_hypodd_once(d)                       # may raise: air-quakes / no reloc = UNDER-damped
        except Exception:
            hist.append((list(damps), "CRASH -> raise DAMP", None))
            damps = [int(min(2000, x * 1.6)) for x in damps]   # too little damping -> increase and retry
            continue
        cnds = rsv._max_cnd_per_set(os.path.join(d, "hypoDD.log"))
        if not cnds:
            break
        score = max(max(0.0, c - hi) + max(0.0, lo - c) for c in cnds.values())
        hist.append((list(damps), {k: round(v, 1) for k, v in sorted(cnds.items())}, round(score, 1)))
        if best is None or score < best[0]:
            best = (score, list(damps), dict(cnds))
        if score <= 0.0:
            break
        for i, c in cnds.items():                          # floor at 10 so a step can't under-damp into air-quakes
            damps[i] = int(min(2000, max(10, round(damps[i] * (c / mid) ** 0.5))))
    bdamps = best[1] if best else [int(min(2000, damp0 * 8))] * 7   # no stable attempt -> heavy fixed damping
    txt = make_inp(template, tuple(bdamps))
    open(os.path.join(d, "hypoDD.inp"), "w").write(txt)
    rsv._exec_hypodd_once(d)
    with open(os.path.join(d, "damping_calibration.txt"), "w") as f:
        f.write(f"kim2011 ISTART=2 per-set adaptive LSQR damping, target CND {lo:.0f}-{hi:.0f}\n")
        for a, (dm, cn, sc) in enumerate(hist):
            f.write(f"  {a}: {dm} -> {cn}  ({sc})\n")
        f.write(f"chosen: {list(bdamps)}\n")
    return tuple(bdamps), (best[2] if best else {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="HypoDD run dir with hypoDD.inp + event.dat + dt.ct[/dt.cc] + station.dat")
    ap.add_argument("--damp0", type=int, default=60)
    a = ap.parse_args()
    d = os.path.abspath(a.dir)
    inp = os.path.join(d, "hypoDD.inp")
    if not os.path.exists(inp):
        sys.exit(f"{inp} missing — run the pipeline through this stage first (it writes the input template).")
    template = open(inp).read()
    # back up the pipeline-default inp + any prior (crashed/empty) reloc
    if not os.path.exists(inp + ".pipeline_default"):
        shutil.copy2(inp, inp + ".pipeline_default")
    rl = os.path.join(d, "hypoDD.reloc")
    if os.path.exists(rl) and os.path.getsize(rl) and not os.path.exists(rl + ".pipeline_default"):
        shutil.copy2(rl, rl + ".pipeline_default")
    print(f"kim2011 + ISTART=2 + adaptive damping on {d}", flush=True)
    bd, cnd = adaptive(d, template, damp0=a.damp0)
    n = sum(1 for _ in open(rl)) if os.path.exists(rl) else 0
    print(f"\nchosen DAMP per set: {list(bd)}")
    print(f"per-set CND: {{{', '.join(f'{k}: {v:.1f}' for k, v in sorted(cnd.items()))}}}")
    print(f"relocated events: {n} -> {rl}")
    print("DONE")


if __name__ == "__main__":
    main()
