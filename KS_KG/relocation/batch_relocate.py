#!/usr/bin/env python
"""Batch driver — relocate ALL 5-25 Hz multiplet families with the validated single-family REUSE scheme.

For every family in the 5-25 Hz single-linkage CC>=0.9 clustering (sorted largest-first) it runs the
existing per-family scripts (make_catalog -> scaffold_offline -> stage -> PocketQuake gather -> HypoInverse
+ ph2dt + dt.ct + GPU dt.cc + HypoDD -> bootstrap -> save_results). It is:
  * ROBUST   — every step in its own try/except; a family that fails (too few dt.cc links, missing reloc,
               nonzero exit) is logged and SKIPPED (falls back to absolute-only); the loop never aborts.
  * RESUMABLE — families already relocated (hypoDD.reloc + reloc_<slug>.csv present) are `done_cached`;
               families already downgraded to absolute-only (with a .sum) are skipped too.
  * AUDITED  — one row per family written to batch_manifest.csv after each family (rewritten each time, so
               it always reflects current state), plus a human log in batch.log.

Usage:
  python batch_relocate.py                       # all families, bootstrap n=200
  python batch_relocate.py --families 738,1044   # a subset
  python batch_relocate.py --limit 3             # first 3 (largest) — smoke test
  python batch_relocate.py --no-bootstrap        # skip bootstrap
"""
import argparse
import csv
import os
import signal
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PQ = "/home/msseo/works/15.PocketQuake"
PIPE = os.path.join(PQ, "external", "korea-cluster-relocation")
HYPO = "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/HypoInv"
RUNS = os.path.join(PIPE, "pipeline", "runs")
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": f"{PQ}:{PIPE}:{HYPO}"}
MANIFEST = os.path.join(HERE, "batch_manifest.csv")
LOG = os.path.join(HERE, "batch.log")
FIELDS = ["family_id", "n", "slug", "status", "n_relocated", "t_seconds",
          "stage_failed", "error_msg", "timestamp"]
BOOT_CORES = 48
STEP_TIMEOUT = 600          # s per subprocess step; HypoDD occasionally hangs on ill-conditioned tiny
                            # families — kill the whole process group and fail that family, don't block.

sys.path.insert(0, HYPO)
import uf_waveform_similarity as wf            # noqa: E402
sys.path.insert(0, PQ); sys.path.insert(0, PIPE)
from pipeline import config                    # noqa: E402
from pipeline.core import sumio, hypodd        # noqa: E402
sys.path.insert(0, HERE)
from save_results import save_results_one       # noqa: E402

STATION, COMP, WIN, PRIMARY, MAXLAG = "KG.HDB", "HHZ", (-0.5, 7.5), (5, 25), 0.2


# --------------------------------------------------------------------------- family enumeration
def family_table():
    """The 5-25 Hz single-linkage CC>=0.9 repeater table (same clustering as make_catalog.py),
    sorted largest-first. Columns include 'cluster' (family id) and 'n' (member count)."""
    kept = wf.make_bands(wf.list_events(station=STATION, comp=COMP), station=STATION, comp=COMP,
                         bands=[PRIMARY, (1, 10), (2, 8), (4, 12)], win=WIN,
                         cache_dir=wf.CACHE_DIR, verbose=False)["kept"]
    meta = wf.load_event_meta(kept)
    tag = (f"{STATION}_{COMP}_w{WIN[0]}_{WIN[1]}_b{PRIMARY[0]}-{PRIMARY[1]}_lag{MAXLAG}_n{len(kept)}"
           .replace(".", "p"))
    cc = np.load(os.path.join(wf.CACHE_DIR, f"cc_{tag}.npy"))
    labels, _, _ = wf.ward_clusters(cc, threshold=1 - 0.9, method="single")
    rep = wf.repeater_table(meta, labels, cc, min_size=3)
    return rep.sort_values("n", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- small helpers
def _stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    line = f"[{_stamp()}] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def _tail(err, k=400):
    s = (getattr(err, "stderr", "") or str(err))[-k:]
    return " ".join(s.split())


def reloc_path(slug):
    return os.path.join(RUNS, slug, "2.HypoDD", "02.dt.cc", "hypoDD.reloc")


def sum_path(slug):
    return os.path.join(RUNS, slug, "1.HypoInv", "kim2011", f"{slug}.sum")


def n_reloc(slug):
    p = reloc_path(slug)
    if not os.path.exists(p):
        return 0
    try:
        return len(sumio.read_reloc(p))
    except Exception:
        return 0


def _run(cmd, cwd):
    """Like subprocess.run(check=True) but with a STEP_TIMEOUT that kills the whole process GROUP
    (so a hung HypoDD grandchild can't survive). A timeout surfaces as CalledProcessError(124) so the
    existing per-step error handling treats it as a normal step failure."""
    p = subprocess.Popen(cmd, cwd=cwd, env=ENV, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = p.communicate(timeout=STEP_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.communicate()
        raise subprocess.CalledProcessError(124, cmd, "", f"TIMEOUT after {STEP_TIMEOUT}s")
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd, out, err)
    return out


def _script(name, *args):
    return _run([PY, os.path.join(HERE, name), *args], HERE)


def _rp(slug, stage_from, through, extra=()):
    return _run([PY, "-m", "pipeline.cli.run_pipeline", "--cluster", slug,
                 "--stage-from", stage_from, "--through", through, *extra], PIPE)


# --------------------------------------------------------------------------- per-family run
def run_family(fid, n, do_boot, boot_n, prior_status=None, redo=False):
    slug = f"f{fid}_reuse"
    outdir = os.path.join(HERE, f"family{fid}")
    t0 = time.perf_counter()

    def res(status, **kw):
        return dict(status=status, n_relocated=kw.get("n_relocated", 0),
                    stage_failed=kw.get("stage_failed", ""), error_msg=kw.get("error_msg", ""),
                    t=time.perf_counter() - t0)

    if not redo:                                   # resume guards
        if n_reloc(slug) > 0 and os.path.exists(os.path.join(outdir, f"reloc_{slug}.csv")):
            return res("done_cached", n_relocated=n_reloc(slug))
        if prior_status == "absolute_only" and os.path.exists(sum_path(slug)):
            return res("absolute_only")

    stage = "make_catalog"
    try:
        _script("make_catalog.py", "--family", str(fid), "--outdir", f"family{fid}")
        with open(os.path.join(outdir, "scaffold_args.txt")) as f:
            parts = f.read().split()               # "--epicenter LAT,LON --region-bounds A,B,C,D"
        epi, rb = parts[1], parts[3]
        stage = "scaffold"
        _script("scaffold_offline.py", slug, "--catalog", os.path.join(outdir, "catalog_kma.csv"),
                "--epicenter", epi, "--region-bounds", rb)
        stage = "stage"
        _script("stage.py", slug, "--reuse-picks", "--members", os.path.join(outdir, "members.txt"))
        stage = "gather"
        _rp(slug, "stations", "waveforms")         # gather raw -> 100km, preserves SAC a/t0
        stage = "dtcc"
        try:
            _rp(slug, "hypoinverse", "dtcc", ("--arc-velmodel", "kim2011"))
        except subprocess.CalledProcessError as e:
            log(f"f{fid}: dt.cc/HypoDD failed -> absolute-only ({_tail(e)})")
            _rp(slug, "hypoinverse", "hypoinverse", ("--arc-velmodel", "kim2011"))  # ensure a .sum exists
            return res("absolute_only", stage_failed="dtcc", error_msg=_tail(e))
        nr = n_reloc(slug)
        if nr == 0:
            log(f"f{fid}: dt.cc produced no/empty reloc -> absolute-only")
            return res("absolute_only", stage_failed="dtcc_empty")
        if do_boot:
            stage = "bootstrap"
            try:
                hypodd.bootstrap_relocation(config.load_cluster(slug), branch="dtcc",
                                            n=boot_n, seed=0, cores=BOOT_CORES)
            except Exception as e:                 # noqa: BLE001 — bootstrap must never abort the family
                log(f"f{fid}: bootstrap failed ({type(e).__name__}: {e})")
        stage = "save_results"
        save_results_one(slug, outdir)
        return res("done", n_relocated=nr)
    except subprocess.CalledProcessError as e:
        log(f"f{fid}: FAILED at {stage}: {_tail(e)}")
        return res(f"failed_{stage}", n_relocated=n_reloc(slug), stage_failed=stage, error_msg=_tail(e))
    except Exception as e:                          # noqa: BLE001
        log(f"f{fid}: ERROR at {stage}: {type(e).__name__}: {e}")
        return res("failed_other", n_relocated=n_reloc(slug), stage_failed=stage,
                   error_msg=f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- manifest IO
def load_manifest():
    m = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            for row in csv.DictReader(f):
                m[int(row["family_id"])] = row
    return m


def write_manifest(m):
    rows = sorted(m.values(), key=lambda r: int(r["family_id"]))
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-bootstrap", action="store_true", help="skip the bootstrap step")
    ap.add_argument("--boot-n", type=int, default=200, help="bootstrap replicas per family (default 200)")
    ap.add_argument("--families", help="comma-separated subset of family ids")
    ap.add_argument("--limit", type=int, help="only the first N (largest) families — smoke test")
    ap.add_argument("--redo", action="store_true", help="ignore resume guards and re-run")
    a = ap.parse_args()

    rep = family_table()
    fam_n = {int(r.cluster): int(r.n) for r in rep.itertuples()}
    ids = [int(x) for x in a.families.split(",")] if a.families else list(fam_n.keys())
    if a.limit:
        ids = ids[:a.limit]

    manifest = load_manifest()
    log(f"=== batch start: {len(ids)} families | bootstrap={'off' if a.no_bootstrap else a.boot_n} "
        f"| redo={a.redo} ===")
    for i, fid in enumerate(ids, 1):
        n = fam_n.get(fid, 0)
        prior = manifest.get(fid, {}).get("status")
        r = run_family(fid, n, do_boot=not a.no_bootstrap, boot_n=a.boot_n,
                       prior_status=prior, redo=a.redo)
        manifest[fid] = dict(family_id=fid, n=n, slug=f"f{fid}_reuse", status=r["status"],
                             n_relocated=r["n_relocated"], t_seconds=round(r["t"], 1),
                             stage_failed=r["stage_failed"], error_msg=r["error_msg"][:300],
                             timestamp=_stamp())
        write_manifest(manifest)
        log(f"[{i}/{len(ids)}] f{fid} (n={n}) -> {r['status']} "
            f"({r['n_relocated']} reloc, {r['t']:.0f}s)")

    done = sum(1 for v in manifest.values() if v["status"] in ("done", "done_cached"))
    absonly = sum(1 for v in manifest.values() if v["status"] == "absolute_only")
    failed = sum(1 for v in manifest.values() if v["status"].startswith("failed"))
    log(f"=== batch done: {done} relocated, {absonly} absolute-only, {failed} failed "
        f"(manifest: {MANIFEST}) ===")


if __name__ == "__main__":
    main()
