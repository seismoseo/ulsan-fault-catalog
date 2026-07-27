#!/usr/bin/env python
"""Build (and status-check) the per-year notebook set — one tidy folder per year.

    python notebooks/make_year_notebooks.py --years 2010                 # one year
    python notebooks/make_year_notebooks.py --years 2010-2024            # the whole record
    python notebooks/make_year_notebooks.py --status                     # just show progress, build nothing

Each year gets its own folder with the YEAR/MODEL already filled in, so you can work through years
independently without editing (or clobbering) a shared notebook:

    notebooks/<year>/00.Run_pipeline_<year>.ipynb      the 6-stage cockpit (run + check + maps)
    notebooks/<year>/01.Record_sections_<year>.ipynb   record sections + the augmented-events figure

The emitted .ipynb files are gitignored (the builders are tracked) — regenerate any time. Regenerating
OVERWRITES a year's notebooks, so parameter edits you made inside a notebook are lost; change the
defaults in the builders if you want them to persist.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BUILDERS = ["build_yearly_run_nb.py", "build_recsec_nb.py"]


def parse_years(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def status_row(year, model):
    """What exists on disk for this year (so you can see where each year stands)."""
    sys.path.insert(0, os.path.join(REPO, "src"))
    from ufpipe import config
    import glob
    picks = len(glob.glob(os.path.join(config.picks_dir(model, year), f"picks_{year}.*.csv")))
    def n_rows(p):
        try:
            return sum(1 for _ in open(p)) - 1
        except OSError:
            return 0
    ev = n_rows(config.pyocto_events(model, year))
    sm = n_rows(os.path.join(config.MODELS, model, "HypoInv", "kim2011", f"UF{year}.sum"))
    qc = 0
    if sm:
        try:
            from uflib import uf_cluster as uf
            qc = len(uf.apply_qc(uf.read_sum(
                os.path.join(config.MODELS, model, "HypoInv", "kim2011", f"UF{year}.sum")).dropna(subset=["time"])))
        except Exception:
            qc = -1
    base = "reloc_%d_uf" % year if model == "phasenet_plus" else "reloc_%d_uf_%s" % (year, model)
    rl = os.path.join(REPO, "outputs", "reloc", base, "results", "hypoDD.reloc.dtcc")
    dtcc = sum(1 for _ in open(rl)) if os.path.exists(rl) else 0
    nb = os.path.isdir(os.path.join(HERE, str(year)))
    return dict(year=year, nb=nb, picks=picks, assoc=ev, located=sm, qc=qc, dtcc=dtcc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", default=None, help="e.g. '2010', '2010-2024', '2016,2018'")
    ap.add_argument("--model", default="phasenet_plus")
    ap.add_argument("--status", action="store_true", help="print the per-year progress table and exit")
    a = ap.parse_args()

    years = parse_years(a.years) if a.years else sorted(
        int(d) for d in os.listdir(HERE) if d.isdigit() and os.path.isdir(os.path.join(HERE, d))
    ) or list(range(2010, 2025))

    if not a.status:
        for y in years:
            for b in BUILDERS:
                subprocess.run([sys.executable, os.path.join(HERE, b),
                                "--year", str(y), "--model", a.model], cwd=REPO, check=True)
        print()

    print(f"=== per-year status ({a.model}) ===")
    print(f"  {'year':>4}  {'nb':>3}  {'picks':>6}  {'assoc':>7}  {'located':>7}  {'QC':>6}  {'dt.cc':>6}")
    for y in years:
        r = status_row(y, a.model)
        print(f"  {r['year']:>4}  {'yes' if r['nb'] else ' - ':>3}  {r['picks']:>6}  {r['assoc']:>7}  "
              f"{r['located']:>7}  {r['qc']:>6}  {r['dtcc']:>6}")
    print("  (picks = daily files; assoc/located/QC = events; dt.cc = relocated events)")
    print("  NOTE assoc=0 with located>0 means that year's catalog predates 2026-07: its association file is")
    print("       still named pyocto_kim1983_<year>.csv and its locations used the stale station list.")
    print("       Re-run the year from `association` in its notebook to bring it onto the current pipeline.")


if __name__ == "__main__":
    main()
