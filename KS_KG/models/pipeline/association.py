"""CLI: PyOcto association for one year (picks -> events + assignments).

Example:
  python association.py --model original --year 2024
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core


def main():
    ap = argparse.ArgumentParser(description="Associate picks into events with PyOcto.")
    ap.add_argument("--model", default="original")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--force", action="store_true", help="allow writing into model='stead'")
    ap.add_argument("--strict", action="store_true",
                    help="use config.REGION_STRICT (tighter pick_match_tolerance + minimums)")
    a = ap.parse_args()
    core.run_association_year(a.model, a.year, force=a.force, strict=a.strict)


if __name__ == "__main__":
    main()
