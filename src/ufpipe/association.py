"""CLI: PyOcto association for one year (picks -> events + assignments).

Example:
  python -m ufpipe.association --model original --year 2024
  python -m ufpipe.association --model phasenet_plus --year 2010 \
      --gate n_picks=5,n_p=3,n_s=2,n_ps=1 --pick-match-tol 1.0 --overlap-s 150
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core


def parse_gate(s):
    """'n_picks=4,n_p=2,n_s=2,n_ps=1' -> dict (all four keys required by PyOcto's from_area)."""
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        k = k.strip()
        if k not in ("n_picks", "n_p", "n_s", "n_ps"):
            raise SystemExit(f"--gate: unknown key {k!r} (valid: n_picks,n_p,n_s,n_ps)")
        out[k] = int(v)
    for req in ("n_picks", "n_p", "n_s", "n_ps"):
        out.setdefault(req, {"n_picks": 4, "n_p": 2, "n_s": 2, "n_ps": 1}[req])
    return out


def _pair(s):
    a, b = s.split(",")
    return (float(a), float(b))


def add_assoc_args(ap):
    """Attach the daily-chunked-association override flags (shared by association.py + run_pipeline.py).
    Every default is None -> run_association_year falls back to config, so omitting them is byte-identical
    to the previous behaviour."""
    g = ap.add_argument_group("association overrides (default: config.ASSOC_*)")
    g.add_argument("--gate", default=None, help="e.g. 'n_picks=4,n_p=2,n_s=2,n_ps=1' (overrides --strict)")
    g.add_argument("--pick-match-tol", type=float, default=None, help="PyOcto residual cap (s); the primary knob")
    g.add_argument("--overlap-s", type=float, default=None, help="daily-chunk overlap (s)")
    g.add_argument("--zlim", type=_pair, default=None, metavar="Z0,Z1", help="depth search range km, e.g. '0,30'")
    g.add_argument("--center", type=_pair, default=None, metavar="LAT,LON", help="region center deg, e.g. '35.856,129.224'")
    g.add_argument("--lat-pad", type=float, default=None, help="region half-height (deg)")
    g.add_argument("--lon-pad", type=float, default=None, help="region half-width (deg)")
    g.add_argument("--time-before", type=float, default=None, help="PyOcto origin-search window (s)")
    return ap


def assoc_overrides(a):
    """Collect the set (non-None) association flags from a parsed argparse namespace into an overrides dict."""
    o = {}
    if getattr(a, "gate", None):
        o["gate"] = parse_gate(a.gate)
    for attr, key in [("pick_match_tol", "pick_match_tol"), ("overlap_s", "overlap_s"),
                      ("zlim", "zlim"), ("center", "center"), ("lat_pad", "lat_pad"),
                      ("lon_pad", "lon_pad"), ("time_before", "time_before")]:
        v = getattr(a, attr, None)
        if v is not None:
            o[key] = v
    return o


def main():
    ap = argparse.ArgumentParser(description="Associate picks into events with PyOcto (daily-chunked).")
    ap.add_argument("--model", default="original")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--force", action="store_true", help="allow writing into model='stead'")
    ap.add_argument("--strict", action="store_true",
                    help="use config.ASSOC_GATE_STRICT (ignored if --gate is given)")
    ap.add_argument("--networks", default=None,
                    help="comma-separated networks whose stations provide coords (default: KS,KG,GJ,NS)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes over days (daily chunks are independent)")
    add_assoc_args(ap)
    a = ap.parse_args()
    networks = a.networks.split(",") if a.networks else None
    core.run_association_year(a.model, a.year, force=a.force, strict=a.strict,
                              networks=networks, workers=a.workers, overrides=assoc_overrides(a))


if __name__ == "__main__":
    main()
