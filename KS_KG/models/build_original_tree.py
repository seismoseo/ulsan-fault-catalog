#!/usr/bin/env python3
"""
Scaffold  KS_KG/models/{stead,original}  for a parallel PhaseNet "original" run.

This script introduces a *picker-model* dimension (stead vs original) on top of the
existing pipeline, WITHOUT modifying any pre-existing file or script.

  * It writes ONLY under  KS_KG/models/  (a hard guard enforces this).
  * Existing outputs become the "stead" run and are exposed read-only via symlinks
    under  models/stead/ .
  * Copies of the per-year notebooks are placed under  models/original/  with the
    model switched to "original" and every output path redirected into the
    original tree so the two runs never collide.

Re-runnable: regenerates the copies, leaves originals untouched.

Run:  python3 KS_KG/models/build_original_tree.py
"""
import json
import os
import shutil
from pathlib import Path

ROOT   = Path("/home/msseo/works/02.Ulsan_Fault_detection/KS_KG")
MODELS = ROOT / "models"
ORIG   = MODELS / "original"
STEAD  = MODELS / "stead"
DET    = ROOT / "detection_location"
HYP    = ROOT / "HypoInv"
YEARS  = list(range(2010, 2025))

MODELS_ABS = os.path.abspath(MODELS)
warnings = []
report   = []


# ---------------------------------------------------------------- safe writers
def guard(dest):
    """Refuse any write whose location is not inside KS_KG/models/."""
    d = os.path.abspath(dest)
    if not (d == MODELS_ABS or d.startswith(MODELS_ABS + os.sep)):
        raise RuntimeError(f"REFUSING to write outside models/: {d}")
    return d


def mkdir(p):
    guard(p)
    os.makedirs(p, exist_ok=True)


def symlink(target, linkpath):
    """Create/replace a symlink located inside models/ pointing at `target`."""
    guard(linkpath)
    if os.path.islink(linkpath):
        os.unlink(linkpath)
    elif os.path.exists(linkpath):
        raise RuntimeError(f"exists and is not a symlink, refuse to clobber: {linkpath}")
    os.symlink(os.path.abspath(target), linkpath)


# ---------------------------------------------------------------- notebook ops
def load_nb(path):
    return json.loads(Path(path).read_text())


def clear_outputs(nb):
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None


def apply_repls(nb, repls):
    """Apply (old,new) substring replacements to code cells. Return fire counts."""
    counts = {o: 0 for o, _ in repls}
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        s = "".join(c["source"])
        for o, n in repls:
            cnt = s.count(o)
            if cnt:
                counts[o] += cnt
                s = s.replace(o, n)
        c["source"] = s.splitlines(keepends=True)
    return counts


def code_text(nb):
    return "\n".join("".join(c["source"]) for c in nb.get("cells", []) if c.get("cell_type") == "code")


def save_nb(nb, path):
    guard(path)
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def process(src, dst, repls, must_fire=None, tag=""):
    """Copy notebook src->dst applying repls + clearing outputs. Validate."""
    if not Path(src).exists():
        warnings.append(f"[MISSING] {tag}: source not found {src}")
        return
    nb = load_nb(src)
    counts = apply_repls(nb, repls)
    clear_outputs(nb)
    save_nb(nb, dst)
    fired = {k: v for k, v in counts.items() if v}
    report.append(f"  {tag}: {Path(dst).relative_to(MODELS)}  repls={fired}")
    if must_fire:
        total = sum(counts.get(k, 0) for k in must_fire)
        if total == 0:
            warnings.append(f"[CHECK] {tag}: expected output-path redirect did NOT fire ({must_fire})")
    # leftover cross-write detectors
    txt = code_text(nb)
    for bad in ('/KS_KG/pyocto/', '/2014_sequence/'):
        if bad in txt:
            warnings.append(f"[REVIEW] {tag}: copy still contains '{bad}' -> manual check {Path(dst).name}")


# ==================================================================== 1. dirs
for d in (MODELS, ORIG, STEAD,
          ORIG / "detection_location",
          ORIG / "pyocto",
          ORIG / "station_table",
          ORIG / "HypoInv",
          ORIG / "HypoInv" / "PHS",
          ORIG / "HypoInv" / "kim1983",
          ORIG / "HypoInv" / "kim2011",
          ORIG / "hypodd"):
    mkdir(d)

# ==================================================== 2. stead navigability
for name in ("detection_location", "picks", "pyocto", "HypoInv", "station_table"):
    if (ROOT / name).exists():
        symlink(ROOT / name, STEAD / name)

# ==================================================== 3. shared HypoInv inputs
symlink(HYP / "STA", ORIG / "HypoInv" / "STA")                       # station files (picker-independent)
for vm in ("kim1983", "kim2011"):
    for ph in ("p", "s"):
        crh = HYP / vm / f"{vm}_{ph}.crh"
        if crh.exists():
            symlink(crh, ORIG / "HypoInv" / vm / f"{vm}_{ph}.crh")    # crh inputs; .sum/.prt/.arc land in real dir

# ==================================================== 4. per-year notebooks
report.append("Detection / Association / Plot notebooks:")
for y in YEARS:
    ydst = ORIG / "detection_location" / str(y)
    mkdir(ydst)

    # 01 detection: switch picker model + fix legacy waveform base_dir
    #   (2014_sequence/continuous is an outdated path; data now lives in KS_KG/continuous)
    det01 = DET / str(y) / f"01.Run_multi-station_detection_{y}.ipynb"
    process(det01, ydst / det01.name,
            [('from_pretrained("stead")', 'from_pretrained("original")'),
             ('/2014_sequence/continuous', '/KS_KG/continuous')],
            tag=f"01 {y}")

    # 02 association: redirect picks-in, pyocto-out, station side-writes
    assoc_repls = [
        ("/KS_KG/detection_location/", "/KS_KG/models/original/detection_location/"),
        ("/KS_KG/pyocto/",             "/KS_KG/models/original/pyocto/"),
        ("/KS_KG/station_table/stations_", "/KS_KG/models/original/station_table/stations_"),
    ]
    if y == 2013:
        assoc_repls.append(("/KS_KG/picks/picks_2013",
                            "/KS_KG/models/original/detection_location/2013/picks/picks_2013"))
    if y == 2010:
        assoc_repls += [
            ("/2014_sequence/picks/picks_2010",
             "/KS_KG/models/original/detection_location/2010/picks/picks_2010"),
            ("/2014_sequence/pyocto/",        "/KS_KG/models/original/pyocto/"),
            ("/2014_sequence/station_table/", "/KS_KG/station_table/"),
            ("/2014_sequence/velocity_model/", "/KS_KG/velocity_model/"),
        ]
    assoc = DET / str(y) / f"02.PyOcto_association_{y}.ipynb"
    must = ["/KS_KG/pyocto/"] + (["/2014_sequence/pyocto/"] if y == 2010 else [])
    process(assoc, ydst / assoc.name, assoc_repls, must_fire=must, tag=f"02 {y}")

    # 03 plot: redirect pyocto reads
    plot = DET / str(y) / f"03.Plot_association_results_{y}.ipynb"
    process(plot, ydst / plot.name,
            [("/KS_KG/pyocto/", "/KS_KG/models/original/pyocto/"),
             ("/2014_sequence/pyocto/", "/KS_KG/models/original/pyocto/")],
            tag=f"03 {y}")

# ==================================================== 5. HypoInv assets
report.append("HypoInv assets:")
# 04 make-PHS notebook (single, parameterized by `year` variable)
phs_repls = [
    ("/KS_KG/pyocto/",       "/KS_KG/models/original/pyocto/"),
    ("/KS_KG/HypoInv/PHS/",  "/KS_KG/models/original/HypoInv/PHS/"),
]
process(HYP / "04.Make_input_PHS_file_for_HypoInv.ipynb",
        ORIG / "HypoInv" / "04.Make_input_PHS_file_for_HypoInv.ipynb",
        phs_repls, must_fire=["/KS_KG/HypoInv/PHS/"], tag="04 PHS")

# UF{year}.sh: copy verbatim (relative paths resolve from the new HypoInv dir)
for y in YEARS:
    sh = HYP / f"UF{y}.sh"
    if sh.exists():
        dst = guard(ORIG / "HypoInv" / f"UF{y}.sh")
        shutil.copy2(sh, dst)
        os.chmod(dst, 0o755)
report.append(f"  UF<year>.sh copied verbatim for {YEARS[0]}-{YEARS[-1]}")

# ==================================================== report
print("\n".join(report))
print("\n=== WARNINGS / FLAGS ===")
print("\n".join(warnings) if warnings else "(none)")
print("\nDONE. All writes confined to:", MODELS_ABS)
