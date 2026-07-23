#!/usr/bin/env python
"""Reproduce the **kim2011-velocity** dt.cc HypoDD relocation (03.dt.cc_kim2011) on the CURRENT event_idx
data, from the fresh all-pairs cross-correlation the pipeline writes to 02.dt.cc.

WHY THIS EXISTS
---------------
`run_uf_hypodd.py --through dtcc` runs the pipeline's *default* dt.cc variant, which writes
`2.HypoDD/02.dt.cc/hypoDD.reloc` using a GENERIC 3-layer P-model (TOP 0/15/32, VEL 5.98/6.38/7.95). The
`--arc-velmodel kim2011` flag only sets the HYPOINVERSE starting locations, NOT the HypoDD velocity. The
analysis (build_ufonly_reloc_ml.py / nb22 / nb26 / Zhigang) uses the **kim2011** 4-layer model
(TOP 0/7.29/20.7/31.3, VEL 5.63/6.17/6.58/7.77; Kim et al. 2011) — consistent with the kim2011 absolute
HYPOINVERSE locations — and reads `2.HypoDD/03.dt.cc_kim2011/hypoDD.reloc`. That folder was historically a
one-off MANUAL hypoDD run; this script makes it a reproducible step.

WHAT IT DOES (drift-free; only the P-velocity block differs from 02.dt.cc)
  1. require fresh 02.dt.cc inputs (event.dat, dt.ct, station.dat, dt.cc_0.7_combined) + its generated
     hypoDD.inp (whose DAMP/DIST were ADAPTED by the pipeline to the current event set).
  2. build 03.dt.cc_kim2011/hypoDD.inp = the fresh 02.dt.cc/hypoDD.inp with ONLY the 1D-model block swapped
     to kim2011 (everything else — weighting schedule, DAMP, ISOLV, OBSCC — identical, so the comparison
     isolates the velocity model exactly as nb22 documents).
  3. symlink dt.ct + dt.cc_0.7_combined -> ../02.dt.cc (auto-fresh); copy event.dat + station.dat.
  4. run hypoDD -> 03.dt.cc_kim2011/hypoDD.reloc (Fortran stdout decoded errors="replace").

The hypoDD event cuspid (= 200000 + sorted waveforms_100km "20*" index) is preserved from 02.dt.cc/event.dat,
so the downstream EXACT id->ts->event_idx map is unchanged.

Usage:  python run_kim2011_dtcc.py            # after run_uf_hypodd.py --through dtcc has finished
        python run_kim2011_dtcc.py --check    # only validate that 02.dt.cc inputs are fresh, do not run
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

RUN = ("/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/"
       "runs/uf_subregion_reuse/2.HypoDD")
GEN = os.path.join(RUN, "02.dt.cc")              # pipeline default variant (generic velocity) — source of inputs
KIM = os.path.join(RUN, "03.dt.cc_kim2011")      # kim2011 product (this script's output)

# kim2011 1-D P model (Kim et al. 2011 SE-Korea); single Vp/Vs ratio per hypoDD 1.x (matches the absolute run)
KIM2011_MODEL = (
    "*--- 1D model: NLAY RATIO  (kim2011, Kim et al. 2011 SE-Korea; single Vp/Vs ratio per hypoDD 1.x)\n"
    "   4     1.73\n"
    "* TOP\n"
    "0.0  7.29  20.7  31.3\n"
    "* VEL\n"
    "5.63  6.17  6.58  7.77\n"
)


# PROVEN per-iteration DAMP schedule (the manual Jun-2025 kim2011 run that yielded the trusted
# 2,079 cc-resolved catalog). We must NOT inherit the pipeline's adaptively-tuned DAMP from
# 02.dt.cc/hypoDD.inp: for this full-link set the cond-number heuristic drove it to ~6-8, which is
# badly under-damped → LSQR lets events drift → the cc residual-reweighting rejects most cc links
# (only ~487 cc-resolved). The higher proven DAMP keeps the inversion stable and retains cc links.
PROVEN_DAMP = [485, 593, 547, 529, 475, 448, 425]


def set_proven_damp(inp_text):
    """Overwrite the DAMP column (last token) of each data-weighting row with PROVEN_DAMP, in order.
    Leaves every other column (NITER, WT*, WR*, WD*) untouched."""
    lines = inp_text.splitlines()
    out, in_wt, di = [], False, 0
    for ln in lines:
        if "data weighting" in ln:
            in_wt = True; out.append(ln); continue
        if "1D model" in ln:
            in_wt = False
        if in_wt and ln.strip() and not ln.lstrip().startswith("*"):
            toks = ln.split()
            if di < len(PROVEN_DAMP):
                toks[-1] = str(PROVEN_DAMP[di]); di += 1
                ln = "    " + "  ".join(toks)
        out.append(ln)
    if di != len(PROVEN_DAMP):
        raise ValueError(f"expected {len(PROVEN_DAMP)} weighting rows, set {di}")
    return "\n".join(out) + "\n"


def force_isolv2(inp_text):
    """Force ISOLV=2 (LSQR). SVD (ISOLV=1) overflows MAXDATA0 on the full ~7.7M-link cc set; LSQR is the
    solver the working runs use and scales to this size. Leaves ISTART and NSET unchanged."""
    import re as _re
    def _sub(m):
        nums = m.group(2).split()
        nums[1] = "2"                                   # ISTART ISOLV NSET -> set ISOLV=2
        return m.group(1) + "    " + "  ".join(nums)
    out, n = _re.subn(r"(\*---\s*solution control: ISTART ISOLV NSET\s*\n)(\s*\d+\s+\d+\s+\d+)",
                      _sub, inp_text)
    if n != 1:
        raise ValueError("could not locate the ISTART ISOLV NSET line")
    return out


def swap_velocity(inp_text):
    """Return inp_text with its 1D-model block (from '*--- 1D model' up to '*--- event selection')
    replaced by the kim2011 model. Everything else (weighting/DAMP/ISOLV) is preserved verbatim."""
    pat = re.compile(r"\*---\s*1D model.*?(?=\*---\s*event selection)", re.DOTALL)
    if not pat.search(inp_text):
        raise ValueError("could not locate the 1D-model block in 02.dt.cc/hypoDD.inp")
    return pat.sub(KIM2011_MODEL, inp_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only validate 02.dt.cc inputs, do not run hypoDD")
    a = ap.parse_args()

    need = ["event.dat", "dt.ct", "station.dat", "dt.cc_0.7_combined", "hypoDD.inp"]
    missing = [f for f in need if not os.path.exists(os.path.join(GEN, f))]
    if missing:
        sys.exit(f"02.dt.cc missing inputs {missing} — run run_uf_hypodd.py --through dtcc first.")
    gen_reloc = os.path.join(GEN, "hypoDD.reloc")
    if not os.path.exists(gen_reloc):
        print(f"WARNING: {gen_reloc} absent — the default-variant dtcc may not have finished.", file=sys.stderr)
    # report event.dat cuspid span (sanity: should be the current 2779-member set, ids 200000..)
    ids = [int(ln.split()[-1]) for ln in open(os.path.join(GEN, "event.dat")) if ln.split()]
    print(f"02.dt.cc/event.dat: {len(ids)} events, cuspid {min(ids)}..{max(ids)}")
    print(f"02.dt.cc/dt.cc_0.7_combined: {os.path.getsize(os.path.join(GEN,'dt.cc_0.7_combined'))/1e6:.0f} MB")
    if a.check:
        return

    os.makedirs(KIM, exist_ok=True)
    # 1. clean stale outputs (old-set reloc iterations etc.) so nothing masquerades as fresh
    for f in os.listdir(KIM):
        if f.startswith("hypoDD.") and f not in ("hypoDD.inp",):
            os.remove(os.path.join(KIM, f))
    # 2. hypoDD.inp = fresh generic inp with kim2011 velocity; KEEP the generic run's adaptive CN-40-80
    #    DAMP (no ad-hoc override). With the corrected cuspid headers the cc links finally connect the
    #    right events, so the standard CN-40-80 tuning should resolve ~all events.
    # CN 40-80 damping chosen from the DAMP sweep on the corrected combined (damp 600 -> CND ~55, in band;
    # cc-resolved/event count are robust across damping, ~2150/~2745). LSQR (ISOLV=2).
    inp = swap_velocity(open(os.path.join(GEN, "hypoDD.inp")).read())
    inp = force_isolv2(inp)
    _o, _w = [], False                                  # set constant DAMP=600 on every weighting row
    for _ln in inp.splitlines():
        if "data weighting" in _ln: _w = True; _o.append(_ln); continue
        if "1D model" in _ln: _w = False
        if _w and _ln.strip() and not _ln.lstrip().startswith("*"):
            _t = _ln.split(); _t[-1] = "600"; _ln = "    " + "  ".join(_t)
        _o.append(_ln)
    inp = "\n".join(_o) + "\n"
    open(os.path.join(KIM, "hypoDD.inp"), "w").write(inp)
    # 3. inputs: symlink the big shared files (auto-fresh), copy the small per-set files
    for f, link in [("dt.ct", True), ("dt.cc_0.7_combined", True), ("event.dat", False), ("station.dat", False)]:
        dst = os.path.join(KIM, f)
        if os.path.lexists(dst):
            os.remove(dst)
        if link:
            os.symlink(os.path.join("..", "02.dt.cc", f), dst)
        else:
            shutil.copyfile(os.path.join(GEN, f), dst)
    # 4. run hypoDD (Fortran; tolerate non-utf8 bytes in its stdout, as the pipeline does)
    print("running hypoDD (kim2011 velocity) ...", flush=True)
    proc = subprocess.run(["hypoDD", "hypoDD.inp"], cwd=KIM,
                          capture_output=True, text=True, errors="replace")
    open(os.path.join(KIM, "hypoDD.stdout"), "w").write(proc.stdout)   # full stdout incl. "# cross corr dtimes"
    for line in proc.stdout.splitlines():
        if "cross corr" in line.lower() or "catalog dtimes" in line.lower():
            print("  " + line.strip())
    tail = "\n".join(proc.stdout.splitlines()[-15:])
    print(tail)
    reloc = os.path.join(KIM, "hypoDD.reloc")
    if proc.returncode != 0 or not os.path.exists(reloc):
        print(proc.stderr[-2000:], file=sys.stderr)
        sys.exit(f"hypoDD failed (rc={proc.returncode}); reloc exists={os.path.exists(reloc)}")
    rids = [int(ln.split()[0]) for ln in open(reloc) if ln.split()]
    print(f"\nkim2011 dt.cc reloc: {len(rids)} events, id {min(rids)}..{max(rids)} -> {reloc}")


if __name__ == "__main__":
    main()
