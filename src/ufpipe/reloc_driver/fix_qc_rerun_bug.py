#!/usr/bin/env python
"""Fix the QC-subset dt.cc bug at its ROOT: the QC cluster's HypoInverse RE-RUN produced wrong origins/locations
(a redundant, pick-mis-staged relocation), and `rereference` stamped those wrong origins into the QC-cluster SACs,
so `xcorr`/`dt.cc` were measured against them. The picks themselves are fine; the ORIGIN REFERENCE they were
expressed against is wrong for a majority of events (median |Δorigin| 0.07 s, hundreds > 0.05 s).

FIX (chosen: re-reference + re-measure dt.cc from scratch, ground truth):
  Replace the QC cluster's HypoInverse solution (1.HypoInv/kim2011/{.sum,.arc}) with the FULL-run solution subset
  to the QC members (the single HypoInverse solution QC was actually computed on), renumbered to the QC cuspids
  (200000+qc_row) that match the existing event dirs. Then re-run the pipeline from `rereference`:
      rereference (correct origins) -> ph2dt (clean event.dat/dt.ct) -> xcorr -> dtcc  (pq-gpu, interp_hz=1000)
  and finally the adaptive kim2011/ISTART=2 HypoDD. The old corrupted 1.HypoInv is backed up to *.rerun_backup.

This does NOT touch detection/association or the full-run HypoInverse. It re-measures dt.cc against the correct
origins -- the expensive but unambiguous path (PN+ ~177k pairs, original ~164k; stead/eqt ~18k/13k).

DRY-RUN by default (prints the plan + verifies the .sum/.arc subset). Pass --apply to execute.
  python fix_qc_rerun_bug.py                       # dry-run, all 4
  python fix_qc_rerun_bug.py --apply               # all 4, back-to-back (rereference..dtcc..HypoDD)
  python fix_qc_rerun_bug.py --picker phasenet_plus --apply
"""
import argparse, os, shutil, subprocess, sys, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DT = "/home/msseo/works/02.Ulsan_Fault_detection/detection_test"
PQ = "/home/msseo/works/15.PocketQuake"
PIPE = os.path.join(PQ, "external", "korea-cluster-relocation")
RUNS = os.path.join(PIPE, "pipeline", "runs")
RELOC = "/home/msseo/works/02.Ulsan_Fault_detection/analysis/relocation"
HYPO = "/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv"
ENV = {**os.environ, "PYTHONPATH": f"{PQ}:{PIPE}:{HYPO}"}
sys.path.insert(0, PQ); sys.path.insert(0, PIPE)
sys.path.insert(0, HERE)
import year_paths as YP
OFFSET = 200000

PICKER_NAMES = ["phasenet_plus", "original", "stead", "eqt"]


def pickers_for(year):
    """picker -> (full_slug, qc_slug, root_basename) for a given year (via year_paths)."""
    return {p: (YP.slug(year, p), YP.slug_qc(year, p), os.path.basename(YP.root_dir(year, p)))
            for p in PICKER_NAMES}


def qc_to_fullrow(root):
    """qc_row (0..N-1, == QC cuspid-200000 == event-dir index) -> full-run members row (== full cuspid-200000)."""
    mem_full = pd.read_csv(f"{DT}/{root}/members.txt", header=None)[0].tolist()
    mem_qc = pd.read_csv(f"{DT}/{root}/members_qc.txt", header=None)[0].tolist()
    fp = {e: i for i, e in enumerate(mem_full)}
    assert all(e in fp for e in mem_qc), "QC member not in full members"
    return [fp[e] for e in mem_qc], mem_qc          # qc_row -> full_row ; qc_row -> event_idx


def build_full_sum_index(full_slug):
    """Full-run .sum indexed by full_row (= id%OFFSET). Uses the pipeline sumio (keeps the cuspid `id`)."""
    from pipeline.core import sumio
    sm = sumio.read_sum(f"{RUNS}/{full_slug}/1.HypoInv/kim2011/{full_slug}.sum")
    sm["full_row"] = sm.id.astype(int) % OFFSET
    return sm.set_index("full_row")


def subset_renumber_sum(full_slug, qc_slug, full_rows):
    """Write the QC .sum as the FULL-run rows for QC members, id renumbered to 200000+qc_row (raw-line copy so the
    HYPOINVERSE column format is byte-preserved; only the ID-NUM field is rewritten)."""
    src = f"{RUNS}/{full_slug}/1.HypoInv/kim2011/{full_slug}.sum"
    dst = f"{RUNS}/{qc_slug}/1.HypoInv/kim2011/{qc_slug}.sum"
    lines = open(src).readlines()
    header = lines[0] if lines and not lines[0][:4].isdigit() else None
    body = lines[1:] if header else lines
    # map full .sum body-row -> its cuspid; we need full_row -> line. sumio row order == body order.
    # find the ID-NUM column: last field on each summary line (10-char). Detect from the pipeline sumio id vs raw.
    from pipeline.core import sumio
    sm = sumio.read_sum(src)
    ids = sm.id.astype(int).tolist()                # same order as body
    fr_of_line = {i % OFFSET: k for k, i in enumerate(ids)}   # full_row -> body line index
    out = [header] if header else []
    for qc_row, fr in enumerate(full_rows):
        ln = body[fr_of_line[fr]]
        new_id = OFFSET + qc_row
        # replace the OLD cuspid (fixed 10-char field) with the new, preserving width.
        old_id = ids[fr_of_line[fr]]
        os_ = ln.rfind(f"{old_id:>10}")
        if os_ < 0:
            os_ = ln.rfind(str(old_id))
            ln = ln[:os_] + f"{new_id}" + ln[os_+len(str(old_id)):]
        else:
            ln = ln[:os_] + f"{new_id:>10}" + ln[os_+10:]
        out.append(ln)
    with open(dst, "w") as f:
        f.writelines(out)
    return dst, len(full_rows)


def subset_renumber_arc(full_slug, qc_slug, full_rows):
    """Write the QC .arc as the FULL-run event blocks for QC members, cuspid (cols 136:146) renumbered to
    200000+qc_row. Preserves phase lines exactly. Full-run arc cuspid = 200000 + full_row order."""
    src = f"{RUNS}/{full_slug}/1.HypoInv/kim2011/{full_slug}.arc"
    dst = f"{RUNS}/{qc_slug}/1.HypoInv/kim2011/{qc_slug}.arc"
    lines = open(src).readlines()
    # An HYPOINVERSE .arc block = a 179-char event header (cols 136:146 = cuspid) + phase lines + a terminator
    # SHADOW card (mostly spaces, cuspid RIGHT-justified at the end, e.g. '...      200000\n'). Blocks are delimited
    # by the NEXT header, so split on headers (a line whose first 8 chars are digits). The terminator card also
    # carries the cuspid -> renumber it too.
    def is_header(ln):
        return len(ln) >= 146 and ln[:8].isdigit()
    hdr_idx = [k for k, ln in enumerate(lines) if is_header(ln)]
    blocks = {}                                      # full cuspid -> list of lines (header .. before next header)
    for a, b in zip(hdr_idx, hdr_idx[1:] + [len(lines)]):
        blocks[int(lines[a][136:146])] = lines[a:b]
    out = []
    for qc_row, fr in enumerate(full_rows):
        full_cusp = OFFSET + fr
        blk = list(blocks[full_cusp]); new_id = OFFSET + qc_row
        blk[0] = blk[0][:136] + f"{new_id:>10}" + blk[0][146:]      # header cuspid
        # terminator shadow card = the LAST line of the block if it holds the old cuspid at its tail
        old_tag = f"{full_cusp}"
        for j in range(len(blk) - 1, 0, -1):
            if blk[j].strip() == old_tag:                          # spaces + cuspid
                blk[j] = blk[j].replace(f"{full_cusp}", f"{new_id}"); break
        out.extend(blk)
    with open(dst, "w") as f:
        f.writelines(out)
    return dst, len(full_rows)


def run(cmd, cwd, conda_env=None):
    if conda_env:
        cmd = ["conda", "run", "-n", conda_env, "python3"] + cmd[1:]
    print(f"\n$ (cwd={cwd}) {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=ENV, check=True)


def fix_one(picker, apply, year=2016):
    full_slug, qc_slug, root = pickers_for(year)[picker]
    full_rows, mem_qc = qc_to_fullrow(root)
    n = len(mem_qc)
    hyp_qc = f"{RUNS}/{qc_slug}/1.HypoInv/kim2011"
    print(f"\n=== {picker} ===  QC members {n}   (full-run HypoInverse -> QC cuspids 200000..{OFFSET+n-1})")

    # verify the full .sum covers every QC member (origins valid)
    fsum = build_full_sum_index(full_slug)
    miss = [fr for fr in full_rows if fr not in fsum.index]
    bad = [fr for fr in full_rows if fr in fsum.index and pd.isna(fsum.loc[fr].time)]
    print(f"  full .sum coverage: {n-len(miss)}/{n} rows present, {len(bad)} with NaN origin  (want 0/0)")
    assert not miss and not bad, "full-run .sum does not cover all QC members cleanly"

    # compare old (rerun) vs new (full) origin for a few, to show the correction magnitude
    from pipeline.core import sumio
    try:
        old = sumio.read_sum(f"{hyp_qc}/{qc_slug}.sum"); old["r"] = old.id.astype(int) % OFFSET
        old = old.set_index("r")
        dts = []
        for qc_row, fr in enumerate(full_rows):
            if qc_row in old.index:
                a, b = fsum.loc[fr].time, old.loc[qc_row].time
                if isinstance(a, pd.Timestamp) and isinstance(b, pd.Timestamp) and pd.notna(a) and pd.notna(b):
                    dts.append(abs((a - b).total_seconds()))
        if dts:
            s = pd.Series(dts)
            print(f"  origin correction |Δot|: median {s.median():.3f}s  >0.05s {int((s>0.05).sum())}  "
                  f">0.2s {int((s>0.2).sum())}  (these are the dt.cc errors being removed)")
    except Exception as e:
        print(f"  (could not read old rerun .sum for comparison: {e})")

    if not apply:
        print("  [dry-run] would: backup 1.HypoInv -> .rerun_backup, write corrected .sum + .arc (full-run rows,")
        print("            renumbered to QC cuspids), then re-run rereference->ph2dt->xcorr->dtcc (pq-gpu) +")
        print("            adaptive kim2011/ISTART=2 HypoDD into 02.dt.cc. Pass --apply to execute.")
        return

    # 1) back up the corrupted re-run HypoInverse, then overwrite .sum + .arc with the full-run subset
    bak = f"{hyp_qc}.rerun_backup"
    if not os.path.exists(bak):
        shutil.copytree(hyp_qc, bak); print(f"  backed up {hyp_qc} -> {bak}")
    sdst, ns = subset_renumber_sum(full_slug, qc_slug, full_rows)
    adst, na = subset_renumber_arc(full_slug, qc_slug, full_rows)
    print(f"  wrote corrected .sum ({ns} events) + .arc ({na} events) from the full run")

    # 2) re-run the relative chain from rereference (correct origins) through dtcc, then adaptive HypoDD
    t0 = time.perf_counter()
    run([sys.executable, "-m", "pipeline.cli.run_pipeline", "--cluster", qc_slug,
         "--stage-from", "rereference", "--through", "dtcc",
         "--velmodels", "kim2011", "--arc-velmodel", "kim2011"], PIPE, conda_env="pq-gpu")
    dtcc = f"{RUNS}/{qc_slug}/2.HypoDD/02.dt.cc"
    for f in ("event.dat", "dt.ct", "station.dat", "hypoDD.inp"):
        assert os.path.exists(f"{dtcc}/{f}"), f"{dtcc}/{f} missing after re-run"
    run([sys.executable, os.path.join(HERE, "run_hypodd_kim2011_istart2.py"), dtcc], HERE)
    rl = f"{dtcc}/hypoDD.reloc"
    nrel = sum(1 for _ in open(rl)) if os.path.exists(rl) else 0
    print(f"  => {picker} corrected dt.cc: {nrel} relocated in {time.perf_counter()-t0:.0f}s  ({rl})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picker", choices=PICKER_NAMES, default=None, help="default: all 4 back-to-back")
    ap.add_argument("--apply", action="store_true")
    YP.add_year_arg(ap)
    a = ap.parse_args()
    for pk in ([a.picker] if a.picker else PICKER_NAMES):
        fix_one(pk, a.apply, a.year)
    if not a.apply:
        print("\nDRY-RUN complete. Re-run with --apply to execute (re-measures dt.cc; the two dense pickers are long).")


if __name__ == "__main__":
    main()
