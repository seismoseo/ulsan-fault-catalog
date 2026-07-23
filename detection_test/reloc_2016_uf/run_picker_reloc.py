#!/usr/bin/env python
"""Full UF-box relocation for ONE picker in ONE year (association-onward), picker is the only variable.

YEAR-GENERAL (--year, default 2016). All year-dependent names go through year_paths.py; year=2016 resolves to
the exact existing 2016 paths/slugs (reloc_2016_uf*, uf_2016*). Assumes detection + association are done for the
year (catalogs/catalog_<picker>_<year>_<mm>_pyocto.csv exist). Chains:
  1. build_sac_and_pyocto --picker --year  -> reloc_<year>_uf_<p>/{pyocto, event_sac(native), station_table, merged_archive}
  2. build_catalog_kma   --picker --year   -> catalog_kma + members (all UF-box events)
  3. scaffold uf_<year>_<p> + stage + pipeline HypoInverse(kim2011)  -> .sum (the FULL run)
  4. build_qc_catalog    --picker --year   -> uf_cluster QC -> members_qc / catalog_kma_qc
  5. [--through dtcc] scaffold uf_<year>_<p>_qc + stage + INJECT full-run HypoInverse (fix; no redundant re-run)
     + rereference + all-pairs GPU xcorr(interp_hz=1000) + adaptive kim2011/ISTART=2 HypoDD (dt.cc + dt.ct copy).

  python run_picker_reloc.py --picker original --through hypoinverse            # 2016, up to QC (fast; validate)
  python run_picker_reloc.py --picker original --through dtcc                   # 2016, full (xcorr ~6h dense)
  python run_picker_reloc.py --picker original --year 2019 --through dtcc       # any year (inputs must exist)
"""
import argparse, os, shutil, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import year_paths as YP
DT = "/home/msseo/works/02.Ulsan_Fault_detection/detection_test"
PQ = "/home/msseo/works/15.PocketQuake"
PIPE = os.path.join(PQ, "external", "korea-cluster-relocation")
RELOC = "/home/msseo/works/02.Ulsan_Fault_detection/analysis/relocation"
HYPO = "/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv"
RUNS = os.path.join(PIPE, "pipeline", "runs")
ENV = {**os.environ, "PYTHONPATH": f"{PQ}:{PIPE}:{HYPO}"}
PY = sys.executable


def run(cmd, cwd, conda_env=None):
    if conda_env:
        cmd = ["conda", "run", "-n", conda_env, "python3"] + cmd[1:]
    print(f"\n$ (cwd={cwd}) {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=ENV, check=True)


def link_results(year, picker, slug, qc_slug):
    """Publish the external pipeline results back into the UF working dir as SYMLINKS, so every output is
    reachable from detection_test/reloc_<year>_uf[_<p>]/results/ without leaving the working tree. Idempotent
    (re-creates links each run). Links point at RUNS/<slug>; nothing is copied."""
    ROOT = YP.root_dir(year, picker)
    res = os.path.join(ROOT, "results"); os.makedirs(res, exist_ok=True)
    # (link_name, external target under RUNS)
    targets = {
        "hypoDD.reloc.dtcc":       f"{RUNS}/{qc_slug}/2.HypoDD/02.dt.cc/hypoDD.reloc",       # dt.ct+dt.cc reloc
        "hypoDD.reloc.dtct":       f"{RUNS}/{qc_slug}/2.HypoDD/01b.dtct_qc/hypoDD.reloc",     # dt.ct-only reloc
        "hypoDD.reloc.pipeline":   f"{RUNS}/{qc_slug}/2.HypoDD/02.dt.cc/hypoDD.reloc.pipeline_default",  # baseline
        "HypoInv.full.sum":        f"{RUNS}/{slug}/1.HypoInv/kim2011/{slug}.sum",             # full absolute loc
        "HypoInv.qc.sum":          f"{RUNS}/{qc_slug}/1.HypoInv/kim2011/{qc_slug}.sum",       # QC subset (=full subset)
        "dt.cc.02":                f"{RUNS}/{qc_slug}/2.HypoDD/02.dt.cc",                      # full dt.cc dir (dir link)
        "run.dir":                 f"{RUNS}/{qc_slug}",                                       # whole external run dir
    }
    n = 0
    for name, tgt in targets.items():
        link = os.path.join(res, name)
        if os.path.islink(link) or os.path.exists(link):
            try: os.remove(link)
            except OSError: continue
        if os.path.exists(tgt):
            os.symlink(tgt, link); n += 1
    print(f"  [{picker}] linked {n} results -> {os.path.relpath(res, DT)}/", flush=True)


def inject_full_hypoinverse(picker, full_slug, qc_slug, root):
    """Overwrite the QC cluster's HypoInverse .sum/.arc with the FULL-run solution subset to the QC members
    (renumbered to the QC cuspids 200000+qc_row that match the event dirs). This REPLACES the redundant, buggy
    QC HypoInverse re-run so rereference/ph2dt/xcorr/dt.cc all use the one solution QC gated on. Reuses the
    validated subset+renumber helpers in fix_qc_rerun_bug.py."""
    import fix_qc_rerun_bug as FX
    full_rows, mem_qc = FX.qc_to_fullrow(os.path.basename(root.rstrip("/")))
    hyp_qc = os.path.join(RUNS, qc_slug, "1.HypoInv", "kim2011")
    os.makedirs(hyp_qc, exist_ok=True)
    FX.subset_renumber_sum(full_slug, qc_slug, full_rows)
    FX.subset_renumber_arc(full_slug, qc_slug, full_rows)
    print(f"  [{picker}] injected full-run HypoInverse ({len(mem_qc)} events) into {qc_slug} "
          f"(no redundant re-run)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picker", required=True)
    ap.add_argument("--through", default="dtcc", choices=["hypoinverse", "dtcc"])
    ap.add_argument("--clean-cache", action="store_true",
                    help="after dt.cc completes, delete the QC cluster's wf_interp_cache (a derived xcorr "
                         "speed-cache, ~tens of GB/picker) — recommended for long multi-year runs")
    ap.add_argument("--link-only", action="store_true",
                    help="don't run anything; just (re)create the results/ symlinks for a completed run")
    YP.add_year_arg(ap)
    a = ap.parse_args()
    p = a.picker; yr = a.year
    ROOT = YP.root_dir(yr, p)
    slug = YP.slug(yr, p)
    slug_qc = YP.slug_qc(yr, p)
    sta_table = YP.station_table(yr, p)
    t0 = time.perf_counter()
    if a.link_only:                                   # just (re)publish result symlinks for a completed run
        link_results(yr, p, slug, slug_qc); return
    print(f"=== {yr} {p}: slug {slug}, qc {slug_qc} ===", flush=True)

    # 1-2: SAC store + pyocto year files + catalog_kma/members (all UF-box)
    run([PY, os.path.join(HERE, "build_sac_and_pyocto.py"), "--picker", p, "--year", str(yr)], HERE)
    run([PY, os.path.join(HERE, "build_catalog_kma.py"), "--picker", p, "--year", str(yr)], HERE)
    # 3: scaffold + stage + HypoInverse (kim2011) on ALL UF-box events -> .sum for QC
    run([PY, os.path.join(HERE, "scaffold_2016.py"), "--slug", slug,
         "--catalog", os.path.join(ROOT, "catalog_kma.csv"), "--station-table", sta_table], PIPE)
    run([PY, os.path.join(RELOC, "stage.py"), slug, "--reuse-picks",
         "--members", os.path.join(ROOT, "members.txt"), "--wf-root", os.path.join(ROOT, "event_sac"),
         "--catalog", os.path.join(ROOT, "members_event_idx.csv")], RELOC)
    run([PY, "-m", "pipeline.cli.run_pipeline", "--cluster", slug, "--stage-from", "stations",
         "--through", "waveforms"], PIPE)
    run([PY, "-m", "pipeline.cli.run_pipeline", "--cluster", slug, "--stage-from", "hypoinverse",
         "--through", "hypoinverse", "--velmodels", "kim2011", "--arc-velmodel", "kim2011"], PIPE)
    # 4: uf_cluster QC of the .sum -> members_qc / catalog_kma_qc
    run([PY, os.path.join(HERE, "build_qc_catalog.py"), "--picker", p, "--year", str(yr)], HERE)
    if a.through == "hypoinverse":
        link_results(yr, p, slug, slug_qc)
        print(f"\n=== {yr} {p}: through QC in {time.perf_counter()-t0:.0f}s ==="); return

    # 5: QC-subset scaffold + stage + rereference + GPU xcorr -> assemble 02.dt.cc
    run([PY, os.path.join(HERE, "scaffold_2016.py"), "--slug", slug_qc,
         "--catalog", os.path.join(ROOT, "catalog_kma_qc.csv"), "--station-table", sta_table], PIPE)
    run([PY, os.path.join(RELOC, "stage.py"), slug_qc, "--reuse-picks",
         "--members", os.path.join(ROOT, "members_qc.txt"), "--wf-root", os.path.join(ROOT, "event_sac"),
         "--catalog", os.path.join(ROOT, "members_event_idx_qc.csv")], RELOC)
    # stations + waveforms only (NOT hypoinverse/ph2dt) -- then REUSE the full-run HypoInverse solution.
    run([PY, "-m", "pipeline.cli.run_pipeline", "--cluster", slug_qc, "--stage-from", "stations",
         "--through", "waveforms", "--velmodels", "kim2011", "--arc-velmodel", "kim2011"], PIPE)
    # CRITICAL: the QC-subset HypoInverse re-run is REDUNDANT and mis-staged picks by a second-resolution timestamp
    # key (an adjacent event's raw picks overwrote the associated picks), producing WRONG origins/locations. That
    # corrupted rereference (SAC origins) -> dt.cc -> event.dat/dt.ct. Instead, inject the FULL-run HypoInverse
    # .sum/.arc subset to the QC members (renumbered to QC cuspids) so rereference/ph2dt use the SAME solution QC
    # gated on. See fix_qc_rerun_bug.py for the full diagnosis.
    inject_full_hypoinverse(p, slug, slug_qc, ROOT)
    try:
        run([PY, "-m", "pipeline.cli.run_pipeline", "--cluster", slug_qc, "--stage-from", "rereference",
             "--through", "dtcc", "--velmodels", "kim2011", "--arc-velmodel", "kim2011"], PIPE, conda_env="pq-gpu")
    except subprocess.CalledProcessError as e:
        print(f"[expected] pipeline dtcc HypoDD failed ({e.returncode}); 02.dt.cc assembled -> our adaptive HypoDD.", flush=True)
    dtcc = os.path.join(RUNS, slug_qc, "2.HypoDD", "02.dt.cc")
    for f in ("event.dat", "dt.ct", "station.dat", "hypoDD.inp"):
        assert os.path.exists(os.path.join(dtcc, f)), f"{dtcc}/{f} missing"
    # adaptive kim2011/ISTART=2 dt.cc, then a dt.ct-only copy
    run([PY, os.path.join(HERE, "run_hypodd_kim2011_istart2.py"), dtcc], HERE)
    dtct = os.path.join(RUNS, slug_qc, "2.HypoDD", "01b.dtct_qc"); os.makedirs(dtct, exist_ok=True)
    for f in ("event.dat", "dt.ct", "station.dat"):
        shutil.copyfile(os.path.join(dtcc, f), os.path.join(dtct, f))
    # dt.ct-only hypoDD.inp template: reuse the pipeline-default from the PN+ 2016 dt.ct run (a valid
    # kim2011 dt.ct .inp; the adaptive driver overwrites the weighting/damping anyway, so it is a seed only).
    inp_tpl = os.path.join(RUNS, "uf_2016", "2.HypoDD", "01.dt.ct", "hypoDD.inp.pipeline_default")
    if not os.path.exists(inp_tpl):                       # any year's own dt.cc .inp is an equivalent seed
        inp_tpl = os.path.join(dtcc, "hypoDD.inp")
    shutil.copyfile(inp_tpl, os.path.join(dtct, "hypoDD.inp"))
    run([PY, os.path.join(HERE, "run_hypodd_kim2011_istart2.py"), dtct], HERE)
    for nm, d in [("dt.cc", dtcc), ("dt.ct", dtct)]:
        rl = os.path.join(d, "hypoDD.reloc")
        n = sum(1 for _ in open(rl)) if os.path.exists(rl) else 0
        print(f"  {p} {nm}: {n} relocated")

    # publish results back into the UF working dir as symlinks (external runs/ -> reloc_<year>_uf[_<p>]/)
    link_results(yr, p, slug, slug_qc)

    # optional: drop the derived interp cache (only speeds a RE-run of THIS year's xcorr; huge over 16 years)
    if a.clean_cache:
        cache = os.path.join(RUNS, slug_qc, "wf_interp_cache")
        if os.path.isdir(cache):
            shutil.rmtree(cache, ignore_errors=True)
            print(f"  [clean-cache] removed {slug_qc}/wf_interp_cache", flush=True)

    print(f"\n=== {yr} {p}: through dtcc in {time.perf_counter()-t0:.0f}s ===")


if __name__ == "__main__":
    main()
