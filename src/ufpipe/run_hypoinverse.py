"""CLI: run HYPOINVERSE (hyp1.40) for one year + velocity model.

Example:
  python run_hypoinverse.py --model original --year 2024 --velmodel kim2011
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
import config


def main():
    ap = argparse.ArgumentParser(description="Locate events with HYPOINVERSE.")
    ap.add_argument("--model", default="original")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--velmodel", default=config.DEFAULT_VELMODEL, help="crustal model dir (kim1983/kim2011)")
    ap.add_argument("--force", action="store_true", help="allow writing into model='stead'")
    ap.add_argument("--networks", default=None,
                    help="networks for the generated station file (default: KS,KG,GJ,NS)")
    ap.add_argument("--no-regen-sta", dest="regen_sta", action="store_false",
                    help="do NOT regenerate STA/UF<year>_hyp.sta from the year table (keep the existing file)")
    a = ap.parse_args()
    core.run_hypoinverse_year(a.model, a.year, velmodel=a.velmodel, force=a.force,
                              networks=(a.networks.split(",") if a.networks else None),
                              regen_sta=a.regen_sta)


if __name__ == "__main__":
    main()
