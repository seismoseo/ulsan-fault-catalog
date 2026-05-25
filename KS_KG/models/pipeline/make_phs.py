"""CLI: build the HYPOINVERSE phase (.phs) file for one year from PyOcto output.

Example:
  python make_phs.py --model original --year 2024
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core


def main():
    ap = argparse.ArgumentParser(description="Write HYPO71 .phs file from PyOcto events+assignments.")
    ap.add_argument("--model", default="original")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--force", action="store_true", help="allow writing into model='stead'")
    a = ap.parse_args()
    core.write_phs(a.model, a.year, force=a.force)


if __name__ == "__main__":
    main()
