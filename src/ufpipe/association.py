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
                    help="use config.ASSOC_GATE_STRICT (stronger origin/depth constraint)")
    ap.add_argument("--networks", default=None,
                    help="comma-separated networks whose stations provide coords (default: KS,KG,GJ,NS)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes over days (daily chunks are independent)")
    a = ap.parse_args()
    networks = a.networks.split(",") if a.networks else None
    core.run_association_year(a.model, a.year, force=a.force, strict=a.strict,
                              networks=networks, workers=a.workers)


if __name__ == "__main__":
    main()
