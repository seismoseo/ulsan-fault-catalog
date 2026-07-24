"""CLI: orchestrate the full catalog pipeline over one or more years.

Stages, in order:  detection -> association -> augment -> phs -> locate -> relocate

Examples:
  # one year, end to end (detection ... HYPOINVERSE)
  python -m ufpipe.run_pipeline --model original --years 2024
  # the whole record
  python -m ufpipe.run_pipeline --model original --years 2010-2024
  # resume from association (picks already exist)
  python -m ufpipe.run_pipeline --model original --years 2015,2016 --stage-from association
  # relocation only (HypoDD dt.ct + dt.cc) on an already-located year
  python -m ufpipe.run_pipeline --model original --years 2016 --stage-from relocate --through dtcc
  # quick plumbing test
  python -m ufpipe.run_pipeline --model original --years 2024 --days 1-3

NOTE the `relocate` stage feeds on the per-month, per-picker association produced by detection_test/lib
(KS/KG/GJ/NS, daily-chunked), NOT ufpipe's whole-year pyocto catalog. See src/ufpipe/relocate.py.
"""
import os
import sys
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
import config
import relocate

STAGES = ["detection", "association", "augment", "phs", "locate", "relocate"]


def parse_years(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def parse_days(s):
    if not s:
        return None
    if "-" in s:
        a, b = s.split("-")
        return range(int(a), int(b) + 1)
    return [int(s)]


def main():
    ap = argparse.ArgumentParser(description="Run the full KS_KG catalog pipeline.")
    ap.add_argument("--model", default="original")
    ap.add_argument("--years", required=True, help="e.g. '2010-2024' or '2015,2016'")
    ap.add_argument("--days", default=None, help="restrict detection to a day range (testing)")
    ap.add_argument("--velmodel", default=config.DEFAULT_VELMODEL)
    ap.add_argument("--stage-from", default="detection", choices=STAGES,
                    help="start at this stage (skip earlier ones)")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="allow writing into model='stead'")
    ap.add_argument("--strict", action="store_true",
                    help="use config.REGION_STRICT for PyOcto association (Stage-2 tighter params)")
    ap.add_argument("--through", default="dtcc", choices=["hypoinverse", "dtcc"],
                    help="relocate stage: stop at QC'd absolute location (hypoinverse) or full dt.cc (default)")
    ap.add_argument("--clean-cache", action="store_true",
                    help="relocate stage: delete the interp cache after dt.cc (recommended for multi-year runs)")
    a = ap.parse_args()

    stages = STAGES[STAGES.index(a.stage_from):]
    days = parse_days(a.days)
    summary = []

    for yr in parse_years(a.years):
        print(f"\n############### {a.model} {yr} : {', '.join(stages)} ###############")
        try:
            if "detection" in stages:
                core.run_detection_year(a.model, yr, days=days, device=a.device,
                                        workers=a.workers, force=a.force)
            if "association" in stages:
                core.run_association_year(a.model, yr, force=a.force, strict=a.strict)
            if "augment" in stages:
                core.run_augment_year(a.model, yr, force=a.force)
            if "phs" in stages:
                core.write_phs(a.model, yr, force=a.force)
            if "locate" in stages:
                core.run_hypoinverse_year(a.model, yr, velmodel=a.velmodel, force=a.force)
            if "relocate" in stages:
                relocate.run_relocate_year(a.model, yr, through=a.through, clean_cache=a.clean_cache)
            summary.append((yr, "OK"))
        except Exception as e:
            traceback.print_exc()
            summary.append((yr, f"FAIL: {e}"))

    print("\n==================== PIPELINE SUMMARY ====================")
    for yr, st in summary:
        print(f"  {yr}: {st}")
    if any(st != "OK" for _, st in summary):
        sys.exit(1)


if __name__ == "__main__":
    main()
