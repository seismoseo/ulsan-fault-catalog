#!/usr/bin/env python
"""Re-relocate the GENERIC-velocity whole-box dt.cc leg (02.dt.cc) with ISTART=2 + per-set adaptive damping,
so nb22 compares FINAL(generic) vs FINAL(kim2011) with only the velocity model differing. Vendors the
adaptive-damping loop from run_svd_volumes.py but mutates the GENERIC hypoDD.inp template (not RUN03's kim2011
one), so the generic 3-layer velocity model is preserved. Backs up the old ISTART=1/DAMP=600 reloc."""
import os, re, shutil, importlib.util
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("rsv", os.path.join(_HERE, "run_svd_volumes.py"))
rsv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rsv)

GEN = ("/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/"
       "uf_subregion_reuse/2.HypoDD/02.dt.cc")
TEMPLATE = open(os.path.join(GEN, "hypoDD.inp")).read()   # generic 3-layer model + generic input file names

def make_inp_generic(damps, istart=2, isolv=2):
    """Set ISTART/ISOLV and replace the 7 weighting-row DAMP values in the GENERIC template (structural match)."""
    txt = re.sub(r"(\*--- solution control: ISTART ISOLV NSET\n\s*)\d+\s+\d+(\s+\d+)", rf"\g<1>{istart}  {isolv}\g<2>", TEMPLATE)
    lines, wi, insec = txt.splitlines(keepends=True), 0, False
    for i, ln in enumerate(lines):
        if "data weighting" in ln: insec = True; continue
        if "1D model" in ln: insec = False
        t = ln.split()
        if insec and len(t) == 10 and all(rsv._isnum(x) for x in t):
            dmp = damps[wi] if wi < len(damps) else damps[-1]
            lines[i] = "    " + "  ".join(t[:-1] + [str(dmp)]) + "\n"; wi += 1
    assert wi == 7, f"expected 7 weighting rows, mutated {wi}"
    return "".join(lines)

def adaptive(d, cnd_range=(40.0, 80.0), max_attempts=12, damp0=60):
    lo, hi = cnd_range; mid = (lo + hi) / 2.0
    damps = [int(damp0)] * 7; best, hist = None, []
    for _ in range(max_attempts):
        open(os.path.join(d, "hypoDD.inp"), "w").write(make_inp_generic(tuple(damps)))
        rsv._exec_hypodd_once(d)
        cnds = rsv._max_cnd_per_set(os.path.join(d, "hypoDD.log"))
        if not cnds: break
        score = max(max(0.0, c - hi) + max(0.0, lo - c) for c in cnds.values())
        hist.append((list(damps), {k: round(v, 1) for k, v in sorted(cnds.items())}, round(score, 1)))
        if best is None or score < best[0]: best = (score, list(damps), dict(cnds))
        if score <= 0.0: break
        for i, c in cnds.items():
            damps[i] = int(min(2000, max(1, round(damps[i] * (c / mid) ** 0.5))))
    bdamps = best[1] if best else damps
    open(os.path.join(d, "hypoDD.inp"), "w").write(make_inp_generic(tuple(bdamps)))
    rsv._exec_hypodd_once(d)
    with open(os.path.join(d, "damping_calibration.txt"), "w") as f:
        f.write(f"generic ISTART=2 per-set adaptive LSQR damping, target CND {lo:.0f}-{hi:.0f}\n")
        for a, (dm, cn, sc) in enumerate(hist): f.write(f"  {a}: {dm} -> {cn}  ({sc})\n")
        f.write(f"chosen: {list(bdamps)}\n")
    return tuple(bdamps), (best[2] if best else {})

if __name__ == "__main__":
    rl = os.path.join(GEN, "hypoDD.reloc")
    bak = rl + ".istart1_damp600.bak"
    if os.path.exists(rl) and not os.path.exists(bak):
        shutil.copy2(rl, bak); print(f"backed up old ISTART=1/DAMP600 reloc -> {os.path.basename(bak)}")
    bd, cnd = adaptive(GEN)
    n = sum(1 for _ in open(rl))
    print("generic ISTART=2 adaptive DAMP:", list(bd))
    print("per-set CND:", {k: round(v, 1) for k, v in sorted(cnd.items())})
    print("relocated events:", n)
    print("DONE")
