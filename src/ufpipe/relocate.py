"""ufpipe stage 6 — relocation (HypoDD dt.ct + dt.cc, with GPU cross-correlation).

This is the last stage of the end-to-end pipeline: detection -> association -> augment -> phs -> locate ->
**relocate**. It turns the QC'd absolute-location catalog into a double-difference relocated catalog for
sub-100 m relative precision.

THIN WRAPPER by design. The heavy, validated orchestration already exists as a year-general driver:
``detection_test/reloc_2016_uf/run_picker_reloc.py`` (the 15-step chain: SAC store -> scaffold -> HYPOINVERSE ->
uf_cluster QC -> inject full-run HYPOINVERSE [the origin-correctness fix] -> re-reference -> GPU xcorr [pq-gpu env,
interp_hz=1000] -> adaptive kim2011/ISTART=2 HypoDD v2.1beta -> link results). We do NOT duplicate it — we
preflight the inputs, then shell out to it with the right cwd/args. The dt.cc/xcorr/HypoDD engine stays external
(15.PocketQuake korea-cluster-relocation); the driver handles PYTHONPATH + the pq-gpu shell-out internally.

INPUT DEPENDENCY (the one place the two pipelines meet): relocation feeds on the per-month, per-picker association
produced by ``detection_test/lib/`` (KS/KG/GJ/NS, daily-chunked — required for the dense NS array), i.e.
``detection_test/catalogs/catalog_<picker>_<year>_<mm>_pyocto.csv`` + assign parquet. It does NOT consume ufpipe's
whole-year ``outputs/models/<model>/pyocto`` catalog (that KS/KG-only whole-year associator is intractable on dense
months). ``--model`` maps identically to the reloc ``--picker`` (original/stead/eqt/phasenet_plus).
"""
import os
import subprocess
import sys

# The validated reloc driver + its preflight live here (year-general).
_RELOC_DIR = "/home/msseo/works/02.Ulsan_Fault_detection/detection_test/reloc_2016_uf"


def _preflight(picker, year):
    """Return True if the per-month detection+association inputs exist for (picker, year); print what to run if not.
    Reuses the existing preflight_year.check() (KS/KG/GJ/NS per-month readiness)."""
    sys.path.insert(0, _RELOC_DIR)
    import preflight_year as PF
    if picker not in PF.PICKERS:
        raise ValueError(f"relocate: picker {picker!r} not in {PF.PICKERS}")
    return PF.check(year, [picker])          # prints the readiness table + the exact commands to generate inputs


def run_relocate_year(model, year, through="dtcc", clean_cache=True, strict_inputs=True):
    """Relocate one (model==picker, year) via the validated reloc driver.

    through : "hypoinverse" -> stop at the QC'd absolute-location subset (fast; no xcorr);
              "dtcc"        -> full double-difference relocation (GPU xcorr ~6 h for dense pickers).
    clean_cache : after dt.cc, delete the ~tens-of-GB interp cache (recommended for multi-year runs).
    strict_inputs : if True, abort when the per-month lib inputs are missing (prints how to generate them).
    """
    picker = model                            # identity map: original/stead/eqt/phasenet_plus
    print(f"[relocate] {picker} {year}: through={through} (feeds on detection_test/lib per-month association)",
          flush=True)
    ready = _preflight(picker, year)
    if not ready and strict_inputs:
        raise SystemExit(
            f"[relocate] inputs for {picker} {year} are incomplete — generate the per-month detection + "
            f"association first (see the commands printed above), then re-run. "
            f"Pass strict_inputs=False to proceed on a partial year.")

    cmd = [sys.executable, os.path.join(_RELOC_DIR, "run_picker_reloc.py"),
           "--picker", picker, "--year", str(year), "--through", through]
    if clean_cache and through == "dtcc":
        cmd.append("--clean-cache")
    print(f"[relocate] $ (cwd={_RELOC_DIR}) {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=_RELOC_DIR, check=True)

    # Results are published as symlinks by the driver's link_results():
    #   detection_test/reloc_<year>_uf[_<picker>]/results/{hypoDD.reloc.dtcc, .dtct, HypoInv.full.sum, ...}
    root = os.path.join("/home/msseo/works/02.Ulsan_Fault_detection/detection_test",
                        f"reloc_{year}_uf" if picker == "phasenet_plus" else f"reloc_{year}_uf_{picker}")
    results = os.path.join(root, "results")
    print(f"[relocate] {picker} {year}: done -> {results}", flush=True)
    return results
