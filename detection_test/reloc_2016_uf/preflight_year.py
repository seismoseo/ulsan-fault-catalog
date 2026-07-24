#!/usr/bin/env python
# DEPRECATED (2026-07): ORPHANED — nothing calls this. ufpipe.relocate has its own _preflight() that checks
# ufpipe's per-year association instead of the lib per-month inputs. See detection_test/lib/DEPRECATED.md.
"""Preflight: for a given --year, report which detection+association INPUTS exist per picker, so you know what
to generate before a full-year `run_picker_reloc.py --year <Y>`. The orchestration (stages 2-9) is year-general;
this only checks the stage 0-1 inputs (station cache + picks + association) that must exist first.

  python preflight_year.py --year 2019
  python preflight_year.py --year 2016 --picker phasenet_plus
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import year_paths as YP

PICKERS = ["phasenet_plus", "original", "stead", "eqt"]
PICKS = os.path.join(YP.HERE, "picks")


def picks_parquet(picker, mm, year):
    # detection writes picks/picks_<picker>_<year>_<mm>.parquet (PN+ via run_pnplus_month, others via seisbench)
    return os.path.join(PICKS, f"picks_{picker}_{year}_{mm:02d}.parquet")


def check(year, pickers):
    print(f"=== YEAR {year} input readiness (stages 0-1) ===\n")
    sc = [os.path.exists(YP.station_cache(mm, year)) for mm in range(1, 13)]
    print(f"station cache (lib/build_stations.py): {sum(sc)}/12 months"
          + ("" if all(sc) else f"  MISSING: {[f'{m:02d}' for m in range(1,13) if not sc[m-1]]}"))
    print()
    ready = all(sc)
    hdr = f"{'picker':14} {'picks':>7} {'assoc':>7}   status"
    print(hdr); print("-" * len(hdr))
    for p in pickers:
        npk = sum(os.path.exists(picks_parquet(p, mm, year)) for mm in range(1, 13))
        nas = sum(os.path.exists(YP.catalog_pyocto(p, mm, year))
                  and os.path.exists(YP.assign_pyocto(p, mm, year)) for mm in range(1, 13))
        ok = all(sc) and npk == 12 and nas == 12
        ready &= (npk == 12 and nas == 12)
        print(f"{p:14} {npk:>4}/12 {nas:>4}/12   {'READY' if ok else 'incomplete'}")
    print()
    if ready:
        print(f"All inputs present. Run the relocation for each picker:")
        for p in pickers:
            print(f"  python run_picker_reloc.py --picker {p} --year {year} --through dtcc")
    else:
        print(f"Generate the missing stage 0-1 inputs first (from detection_test/), per month mm=01..12:")
        print(f"  python lib/build_stations.py       --month {year}-<mm>")
        print(f"  python lib/run_seisbench_picker.py --model <p> --month {year}-<mm>   # PN+: run_pnplus_month.py")
        print(f"  python lib/associate_daily.py      --picker <p> --month {year}-<mm>")
    return ready


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picker", choices=PICKERS, default=None, help="default: all 4")
    YP.add_year_arg(ap)
    a = ap.parse_args()
    ok = check(a.year, [a.picker] if a.picker else PICKERS)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
