"""ufpipe stage 6 — relocation (HypoDD dt.ct + dt.cc, with GPU cross-correlation).

Last stage of the end-to-end pipeline: detection -> association -> augment -> phs -> locate -> **relocate**.
Turns the QC'd absolute-location catalog into a double-difference relocated catalog for sub-100 m precision.

SELF-FED (2026-07). The reloc inputs (event-idx-keyed native-rate SAC store + per-year PyOcto tables +
multi-network station table) are built HERE, from ufpipe's OWN detection + association outputs, by
``src/ufpipe/reloc_inputs.py``. The old per-month ``detection_test/lib`` feeder is no longer used — ufpipe
detection+association now cover KS/KG/GJ/NS with daily-chunked association, so the pipeline is self-contained.

The heavy downstream orchestration (scaffold -> HYPOINVERSE -> uf_cluster QC -> inject full-run HYPOINVERSE
[the origin-correctness fix] -> re-reference -> GPU xcorr [pq-gpu env] -> adaptive kim2011/ISTART=2 HypoDD
v2.1beta -> link results) still runs via the validated driver
``src/ufpipe/reloc_driver/run_picker_reloc.py`` with ``--skip-build`` (it drives the external
15.PocketQuake korea-cluster-relocation engine + handles the pq-gpu shell-out). We build inputs, then hand off.

``--model`` maps identically to the reloc ``--picker`` (original/stead/eqt/phasenet_plus).
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# The downstream reloc driver (scaffold/HypoInverse/xcorr/HypoDD orchestration) — live code in src/.
_RELOC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reloc_driver")


def _reloc_root(picker, year):
    """Per-(picker, year) reloc working dir under outputs/reloc/ (single source: reloc_driver/year_paths).
    The completed 2016 4-picker PILOT dirs remain frozen at detection_test/reloc_2016_uf*/ (archive)."""
    sys.path.insert(0, _RELOC_DIR)
    import year_paths as YP
    return YP.root_dir(year, picker)


def _preflight(model, year):
    """True if ufpipe's OWN per-year detection+association outputs exist for (model, year).
    Prints exactly what to run if not (the ufpipe detection/association CLIs)."""
    ev = config.pyocto_events(model, year)
    asg = config.pyocto_assign(model, year)
    ok = os.path.exists(ev) and os.path.exists(asg)
    if ok:
        print(f"[relocate] preflight OK: ufpipe association present for {model} {year}")
    else:
        print(f"[relocate] preflight: MISSING ufpipe association for {model} {year}:")
        print(f"    events : {ev}  {'[ok]' if os.path.exists(ev) else '[missing]'}")
        print(f"    assign : {asg}  {'[ok]' if os.path.exists(asg) else '[missing]'}")
        print(f"  Generate them first (ufpipe's own detection + daily-chunked association):")
        print(f"    conda run -n eqnet python -m ufpipe.detection   --model {model} --year {year}")
        print(f"    conda run -n base  python -m ufpipe.association --model {model} --year {year}")
    return ok


def run_relocate_year(model, year, through="dtcc", clean_cache=True, strict_inputs=True):
    """Relocate one (model==picker, year) from ufpipe's own outputs.

    through : "hypoinverse" -> stop at the QC'd absolute-location subset (fast; no xcorr);
              "dtcc"        -> full double-difference relocation (GPU xcorr ~6 h for dense pickers).
    clean_cache : after dt.cc, delete the ~tens-of-GB interp cache (recommended for multi-year runs).
    strict_inputs : if True, abort when ufpipe's per-year association is missing.
    """
    picker = model                            # identity map: original/stead/eqt/phasenet_plus
    print(f"[relocate] {picker} {year}: through={through} (self-fed from ufpipe detection+association)", flush=True)
    if not _preflight(model, year) and strict_inputs:
        raise SystemExit(
            f"[relocate] ufpipe association for {model} {year} is missing — run ufpipe detection + "
            f"association first (see the commands printed above), then re-run. "
            f"Pass strict_inputs=False to attempt anyway.")

    # Step 1 (was detection_test/lib per-month feeder): build the reloc input store from ufpipe's OWN outputs.
    root = _reloc_root(picker, year)
    os.makedirs(root, exist_ok=True)
    import reloc_inputs
    reloc_inputs.build_reloc_inputs(model, year, root)

    # Steps 2-15: hand off to the validated driver with --skip-build (it no longer rebuilds step 1).
    cmd = [sys.executable, os.path.join(_RELOC_DIR, "run_picker_reloc.py"),
           "--picker", picker, "--year", str(year), "--through", through, "--skip-build"]
    if clean_cache and through == "dtcc":
        cmd.append("--clean-cache")
    print(f"[relocate] $ (cwd={_RELOC_DIR}) {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=_RELOC_DIR, check=True)

    results = os.path.join(root, "results")
    print(f"[relocate] {picker} {year}: done -> {results}", flush=True)
    return results
