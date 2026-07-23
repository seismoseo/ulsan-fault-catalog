#!/usr/bin/env python
"""Year-general path/slug helpers for the UF picker-comparison relocation (orchestration layer).

The lower layers (detection/association/station-cache) are already `--month YYYY-MM` general. Only the
relocation orchestration hard-coded 2016. This module centralizes every year-dependent name so the build
scripts + driver take a single `--year` (default 2016) and back-compatibly resolve the EXACT existing 2016
paths/slugs when year==2016 -- so 2016 work is untouched.

Naming scheme (year-general):
  ROOT dir     reloc_<year>_uf/           (PN+)     reloc_<year>_uf_<picker>/   (others)
  cluster slug uf_<year>                  (PN+)     uf_<year>_<picker>          (others)
  qc slug      uf_<year>_qc               (PN+)     uf_<year>_<picker>_qc       (others)
  station cache cache/stations_<year>_<mm>.csv       (built by lib/build_stations.py per month)
  catalogs      catalogs/catalog_<picker>_<year>_<mm>_pyocto.csv + assign_..._pyocto.parquet
  station table <ROOT>/station_table/stations_<year>.csv
  pyocto year   <ROOT>/pyocto/pyocto_kim2011_<year>.csv (+ _assignment_)
"""
import os

HERE = "/home/msseo/works/02.Ulsan_Fault_detection/detection_test"
PQ = "/home/msseo/works/15.PocketQuake"
PIPE = os.path.join(PQ, "external", "korea-cluster-relocation")
RUNS = os.path.join(PIPE, "pipeline", "runs")
VELMODEL = "kim2011"


def root_dir(year, picker):
    """Per-(year,picker) working dir under detection_test/. PN+ uses the bare reloc_<year>_uf/."""
    base = os.path.join(HERE, f"reloc_{year}_uf")
    return base if picker == "phasenet_plus" else os.path.join(HERE, f"reloc_{year}_uf_{picker}")


def shared_dir(year):
    """Picker-independent dir (station_table + merged_archive live here) = the PN+ root for that year."""
    return os.path.join(HERE, f"reloc_{year}_uf")


def slug(year, picker):
    """HypoDD/pipeline cluster slug (full UF-box run)."""
    return f"uf_{year}" if picker == "phasenet_plus" else f"uf_{year}_{picker}"


def slug_qc(year, picker):
    """HypoDD/pipeline cluster slug (QC subset -> dt.cc)."""
    return f"{slug(year, picker)}_qc"


def station_cache(mm, year):
    return os.path.join(HERE, "cache", f"stations_{year}_{mm:02d}.csv")


def station_table(year, picker):
    return os.path.join(root_dir(year, picker), "station_table", f"stations_{year}.csv")


def catalog_pyocto(picker, mm, year):
    return os.path.join(HERE, "catalogs", f"catalog_{picker}_{year}_{mm:02d}_pyocto.csv")


def assign_pyocto(picker, mm, year):
    return os.path.join(HERE, "catalogs", f"assign_{picker}_{year}_{mm:02d}_pyocto.parquet")


def pyocto_year(year, picker):
    return os.path.join(root_dir(year, picker), "pyocto", f"pyocto_{VELMODEL}_{year}.csv")


def pyocto_assignment_year(year, picker):
    return os.path.join(root_dir(year, picker), "pyocto", f"pyocto_assignment_{VELMODEL}_{year}.csv")


def add_year_arg(ap):
    """Attach the standard --year argument (default 2016 = the existing run)."""
    ap.add_argument("--year", type=int, default=2016, help="calendar year (default 2016)")
    return ap


# ---- self-check: year=2016 must resolve to the EXACT existing paths ------------------------------
if __name__ == "__main__":
    import sys
    y = 2016
    checks = {
        "root PN+":      (root_dir(y, "phasenet_plus"), f"{HERE}/reloc_2016_uf"),
        "root original": (root_dir(y, "original"),      f"{HERE}/reloc_2016_uf_original"),
        "slug PN+":      (slug(y, "phasenet_plus"),     "uf_2016"),
        "slug_qc orig":  (slug_qc(y, "original"),       "uf_2016_original_qc"),
        "sta cache m1":  (station_cache(1, y),          f"{HERE}/cache/stations_2016_01.csv"),
        "sta table PN+": (station_table(y, "phasenet_plus"), f"{HERE}/reloc_2016_uf/station_table/stations_2016.csv"),
        "cat orig m9":   (catalog_pyocto("original", 9, y), f"{HERE}/catalogs/catalog_original_2016_09_pyocto.csv"),
        "pyocto PN+":    (pyocto_year(y, "phasenet_plus"), f"{HERE}/reloc_2016_uf/pyocto/pyocto_kim2011_2016.csv"),
    }
    bad = 0
    for name, (got, want) in checks.items():
        ok = got == want
        bad += not ok
        print(f"  {'OK ' if ok else 'BAD'} {name}: {got}" + ("" if ok else f"  != {want}"))
    print(f"\n{'ALL 2016 paths back-compatible' if not bad else f'{bad} MISMATCH'}")
    sys.exit(1 if bad else 0)
