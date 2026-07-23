#!/usr/bin/env python
"""DAMP sweep for the kim2011 dt.cc relocation on the CORRECTED combined (cuspids fixed).

Goal: find the constant DAMP whose condition number (CND) lands in HypoDD's recommended 40-80 band while
preserving cc-resolved events and a low cc RMS. Each run: kim2011 velocity, ISOLV=2 (LSQR, no SVD overflow),
constant DAMP, on the corrected dt.cc_0.7_combined. Runs all DAMPs CONCURRENTLY in isolated dirs.

Reads from each run's hypoDD.log: final 'acond (CND)=' (condition number), 'absolute cc rms', and the
cc-resolved count (nccp+nccs>0) from hypoDD.reloc.
"""
import glob, os, re, shutil, subprocess, sys
sys.path.insert(0, "/home/msseo/works/02.Ulsan_Fault_detection/analysis/uf_subregion_hypodd")
from run_kim2011_dtcc import swap_velocity, force_isolv2

GEN = "/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/uf_subregion_reuse/2.HypoDD/02.dt.cc"
SWEEP = "/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/uf_subregion_reuse/2.HypoDD/sweep_damp"
DAMPS = [300, 800, 1500, 3000]


def set_const_damp(inp_text, d):
    out, in_wt = [], False
    for ln in inp_text.splitlines():
        if "data weighting" in ln: in_wt = True; out.append(ln); continue
        if "1D model" in ln: in_wt = False
        if in_wt and ln.strip() and not ln.lstrip().startswith("*"):
            t = ln.split(); t[-1] = str(d); ln = "    " + "  ".join(t)
        out.append(ln)
    return "\n".join(out) + "\n"


def setup(d):
    dd = os.path.join(SWEEP, f"d{d}"); os.makedirs(dd, exist_ok=True)
    for f in os.listdir(dd):
        if f.startswith("hypoDD."): os.remove(os.path.join(dd, f))
    inp = set_const_damp(force_isolv2(swap_velocity(open(os.path.join(GEN, "hypoDD.inp")).read())), d)
    open(os.path.join(dd, "hypoDD.inp"), "w").write(inp)
    for f, link in [("dt.ct", True), ("dt.cc_0.7_combined", True), ("event.dat", False), ("station.dat", False)]:
        dst = os.path.join(dd, f)
        if os.path.lexists(dst): os.remove(dst)
        os.symlink(os.path.join(GEN, f), dst) if link else shutil.copyfile(os.path.join(GEN, f), dst)  # ABSOLUTE target
    return dd


def parse(dd):
    log = os.path.join(dd, "hypoDD.log")
    cnd = ccrms = ctrms = None
    if os.path.exists(log):
        txt = open(log, errors="replace").read()
        cnds = re.findall(r"acond \(CND\)=\s*([\d.]+)", txt)
        if cnds: cnd = float(cnds[-1])
        cc = re.findall(r"absolute cc rms \[s\] =\s*([\d.]+)", txt)
        if cc: ccrms = float(cc[-1])
        ct = re.findall(r"absolute ct rms \[s\] =\s*([\d.]+)", txt)
        if ct: ctrms = float(ct[-1])
    reloc = os.path.join(dd, "hypoDD.reloc")
    nev = ccres = 0
    if os.path.exists(reloc):
        for ln in open(reloc):
            c = ln.split()
            if len(c) >= 19:
                nev += 1
                if float(c[17]) + float(c[18]) > 0: ccres += 1
    return cnd, ccrms, ctrms, nev, ccres


procs = {}
for d in DAMPS:
    dd = setup(d)
    fout = open(os.path.join(dd, "hypoDD.stdout"), "w")
    procs[d] = (subprocess.Popen(["hypoDD", "hypoDD.inp"], cwd=dd, stdout=fout,
                                 stderr=subprocess.STDOUT, text=True, errors="replace"), dd, fout)
    print(f"launched DAMP={d}", flush=True)
print(f"\nall {len(DAMPS)} runs launched (ISOLV=2, corrected combined); waiting ...", flush=True)
for d, (p, dd, fout) in procs.items():
    p.wait(); fout.close()

print("\n=== DAMP SWEEP (corrected cuspids, kim2011, LSQR) ===", flush=True)
print(f"{'DAMP':>6} {'CND':>9} {'cc_rms_s':>9} {'ct_rms_s':>9} {'cc-resolved':>11} {'nev':>6}  {'in_40-80?':>9}")
for d in DAMPS:
    cnd, ccrms, ctrms, nev, ccres = parse(procs[d][1])
    inband = "YES" if (cnd is not None and 40 <= cnd <= 80) else ""
    print(f"{d:>6} {str(cnd):>9} {str(ccrms):>9} {str(ctrms):>9} {ccres:>11} {nev:>6}  {inband:>9}", flush=True)
