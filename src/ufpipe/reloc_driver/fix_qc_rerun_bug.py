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
import numpy as np
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


def _root_dir(root):
    """Resolve `root` (a full reloc working-dir path, or a bare `reloc_<year>_uf[_<picker>]` basename) to the
    directory that actually holds members.txt. Reloc dirs live under outputs/reloc/ since the 2026-07 migration
    (YP.RELOC_OUT); a legacy detection_test/ path is honoured only if that's where it actually is."""
    if os.path.isabs(root) and os.path.isdir(root):
        return root
    base = os.path.basename(root.rstrip("/"))
    for cand in (os.path.join(YP.RELOC_OUT, base), os.path.join(DT, base)):
        if os.path.exists(os.path.join(cand, "members.txt")):
            return cand
    return os.path.join(YP.RELOC_OUT, base)          # default to the current convention (clear error if absent)


def qc_to_fullrow(root):
    """qc_row (0..N-1, == QC cuspid-200000 == event-dir index) -> full-run members row (== full cuspid-200000)."""
    rd = _root_dir(root)
    mem_full = pd.read_csv(f"{rd}/members.txt", header=None)[0].tolist()
    mem_qc = pd.read_csv(f"{rd}/members_qc.txt", header=None)[0].tolist()
    fp = {e: i for i, e in enumerate(mem_full)}
    assert all(e in fp for e in mem_qc), "QC member not in full members"
    return [fp[e] for e in mem_qc], mem_qc          # qc_row -> full_row ; qc_row -> event_idx


def build_full_sum_index(full_slug):
    """Full-run .sum indexed by full_row (= id%OFFSET). Uses the pipeline sumio (keeps the cuspid `id`)."""
    from pipeline.core import sumio
    sm = sumio.read_sum(f"{RUNS}/{full_slug}/1.HypoInv/kim2011/{full_slug}.sum")
    sm["full_row"] = sm.id.astype(int) % OFFSET
    return sm.set_index("full_row")


def member_to_cuspid(root, full_slug):
    """members row -> full-run engine CUSPID, matched by ORIGIN TIME (monotonic two-pointer, 10 s tol,
    auto tz offset). Returns (dict row->cuspid, dict cuspid->sum-body-line-index).

    NEVER trust ``cuspid == OFFSET + members row``: the engine derives its event list from its own
    staging, and same-second doublets can insert/drop entries — 2011 had 446 id slots for 445 members,
    so every cuspid >= 18 was OFF BY ONE and id arithmetic would have attached WRONG origins to most
    events (silently, had the KeyError not fired). Time alignment against catalog_kma is exact; a
    member with no matching solution is simply absent from the map (caller drops it loudly)."""
    from pipeline.core import sumio
    rd = _root_dir(root)
    cat = pd.read_csv(f"{rd}/catalog_kma.csv")
    cat_t = pd.to_datetime(dict(year=cat.Year, month=cat.Month, day=cat.Day, hour=cat.Hour,
                                minute=cat.Minute, second=cat.Second.clip(0, 59))).to_numpy()
    sm = sumio.read_sum(f"{RUNS}/{full_slug}/1.HypoInv/kim2011/{full_slug}.sum")
    line_of_cuspid = {int(i): k for k, i in enumerate(sm.id.astype(int).tolist())}
    # sumio yields obspy UTCDateTime origin times — convert to numpy datetimes (None -> NaT)
    sm_t = pd.to_datetime([t.datetime if t is not None else pd.NaT for t in sm.time]).to_numpy()
    ok = ~pd.isna(sm_t)
    order = np.argsort(sm_t[ok], kind="stable")
    sm_ids = sm.id.astype(int).to_numpy()[ok][order]
    sm_ts = sm_t[ok][order]

    def med_nn(o):
        s = sm_ts - np.timedelta64(int(o), "s")
        k = np.clip(np.searchsorted(cat_t, s), 0, len(cat_t) - 1)
        d = np.abs(cat_t[k] - s).astype("timedelta64[s]").astype(float)
        km = np.clip(k - 1, 0, len(cat_t) - 1)
        dm = np.abs(cat_t[km] - s).astype("timedelta64[s]").astype(float)
        return np.median(np.minimum(d, dm))
    off = min([0, 32400, -32400], key=med_nn)
    sm_ts = sm_ts - np.timedelta64(int(off), "s")

    TOL = np.timedelta64(10, "s")
    m2c = {}
    j = 0
    for i in range(len(cat_t)):
        while j < len(sm_ts) and sm_ts[j] < cat_t[i] - TOL:
            j += 1
        if j < len(sm_ts) and abs(sm_ts[j] - cat_t[i]) <= TOL:
            m2c[i] = int(sm_ids[j])
            j += 1
    return m2c, line_of_cuspid


def subset_renumber_sum(full_slug, qc_slug, pairs):
    """Write the QC .sum as the FULL-run rows for QC members, id renumbered to 200000+qc_row (raw-line copy so the
    HYPOINVERSE column format is byte-preserved; only the ID-NUM field is rewritten).

    `pairs` = [(qc_row, full_cuspid or None), ...] from member_to_cuspid — a None cuspid (member with
    no full-run solution, e.g. an unlocated same-second doublet twin) is SKIPPED: its qc_row id is
    simply absent from the subset .sum, so downstream stages exclude that event instead of crashing
    (or worse, inheriting a wrong origin)."""
    src = f"{RUNS}/{full_slug}/1.HypoInv/kim2011/{full_slug}.sum"
    dst = f"{RUNS}/{qc_slug}/1.HypoInv/kim2011/{qc_slug}.sum"
    lines = open(src).readlines()
    header = lines[0] if lines and not lines[0][:4].isdigit() else None
    body = lines[1:] if header else lines
    from pipeline.core import sumio
    sm = sumio.read_sum(src)
    ids = sm.id.astype(int).tolist()                # same order as body
    line_of = {i: k for k, i in enumerate(ids)}     # cuspid -> body line index
    out = [header] if header else []
    n = 0
    for qc_row, cusp in pairs:
        if cusp is None:
            continue
        ln = body[line_of[cusp]]
        new_id = OFFSET + qc_row
        # replace the OLD cuspid (fixed 10-char field) with the new, preserving width.
        os_ = ln.rfind(f"{cusp:>10}")
        if os_ < 0:
            os_ = ln.rfind(str(cusp))
            ln = ln[:os_] + f"{new_id}" + ln[os_+len(str(cusp)):]
        else:
            ln = ln[:os_] + f"{new_id:>10}" + ln[os_+10:]
        out.append(ln)
        n += 1
    with open(dst, "w") as f:
        f.writelines(out)
    return dst, n


def subset_renumber_arc(full_slug, qc_slug, pairs):
    """Write the QC .arc as the FULL-run event blocks for QC members, cuspid (cols 136:146) renumbered to
    200000+qc_row. Preserves phase lines exactly. Blocks are looked up by the engine's ACTUAL cuspid
    (from `pairs`, time-matched), never by 200000+members-row arithmetic. None cuspids are skipped."""
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
    n = 0
    for qc_row, cusp in pairs:
        if cusp is None or cusp not in blocks:
            continue
        blk = list(blocks[cusp]); new_id = OFFSET + qc_row
        blk[0] = blk[0][:136] + f"{new_id:>10}" + blk[0][146:]      # header cuspid
        # terminator shadow card = the LAST line of the block if it holds the old cuspid at its tail
        old_tag = f"{cusp}"
        for j in range(len(blk) - 1, 0, -1):
            if blk[j].strip() == old_tag:                          # spaces + cuspid
                blk[j] = blk[j].replace(f"{cusp}", f"{new_id}"); break
        out.extend(blk)
        n += 1
    with open(dst, "w") as f:
        f.writelines(out)
    return dst, n


def qc_pairs(root, full_slug, max_drop_frac=0.02):
    """(qc_row, cuspid) pairs for the injection, via time matching; LOUD about any drops.

    A dropped member (no full-run solution) is scientifically correct to exclude — it has no valid
    origin to re-reference against. But a LARGE drop fraction means something structural is wrong
    (stale staging, wrong catalog), so fail hard beyond `max_drop_frac`."""
    full_rows, mem_qc = qc_to_fullrow(root)
    m2c, _ = member_to_cuspid(root, full_slug)
    pairs = [(qc_row, m2c.get(fr)) for qc_row, fr in enumerate(full_rows)]
    dropped = [(qc_row, full_rows[qc_row], mem_qc[qc_row]) for qc_row, c in pairs if c is None]
    if dropped:
        print(f"  !! {len(dropped)}/{len(pairs)} QC member(s) have NO full-run HypoInverse solution "
              f"(unlocated same-second doublet twins etc.) — EXCLUDED from the injected subset:")
        for qc_row, fr, eidx in dropped[:10]:
            print(f"       qc_row {qc_row}  members_row {fr}  event_idx {eidx}")
    if len(dropped) > max(2, max_drop_frac * len(pairs)):
        raise RuntimeError(f"{len(dropped)} of {len(pairs)} QC members unmatched in the full-run .sum — "
                           f"far beyond doublet losses; check for stale staging (picks/, stp_download/SAC).")
    return pairs, mem_qc


def run(cmd, cwd, conda_env=None):
    if conda_env:
        cmd = ["conda", "run", "-n", conda_env, "python3"] + cmd[1:]
    print(f"\n$ (cwd={cwd}) {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=ENV, check=True)


def fix_one(picker, apply, year=2016):
    full_slug, qc_slug, root = pickers_for(year)[picker]
    pairs, mem_qc = qc_pairs(root, full_slug)
    n = len(mem_qc)
    hyp_qc = f"{RUNS}/{qc_slug}/1.HypoInv/kim2011"
    print(f"\n=== {picker} ===  QC members {n}   (full-run HypoInverse -> QC cuspids 200000..{OFFSET+n-1})")

    # compare old (rerun) vs new (full) origin for a few, to show the correction magnitude
    from pipeline.core import sumio
    try:
        fsum = sumio.read_sum(f"{RUNS}/{full_slug}/1.HypoInv/kim2011/{full_slug}.sum")
        fsum = fsum.set_index(fsum.id.astype(int))
        old = sumio.read_sum(f"{hyp_qc}/{qc_slug}.sum"); old["r"] = old.id.astype(int) % OFFSET
        old = old.set_index("r")
        dts = []
        for qc_row, cusp in pairs:
            if cusp is not None and qc_row in old.index:
                a, b = fsum.loc[cusp].time, old.loc[qc_row].time
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
    sdst, ns = subset_renumber_sum(full_slug, qc_slug, pairs)
    adst, na = subset_renumber_arc(full_slug, qc_slug, pairs)
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
