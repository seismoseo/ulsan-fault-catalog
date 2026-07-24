# `detection_test/lib/` is DEPRECATED (2026-07)

These per-month scripts were the **feeder** for the relocation stage: they ran detection
(`run_pnplus_month.py` / `run_seisbench_picker.py`), built the multi-network station table
(`build_stations.py`), and daily-chunked PyOcto association (`associate_daily.py`) per month,
writing `detection_test/{cache,picks,catalogs}/`.

**They are no longer used.** `ufpipe` now covers this itself, per year, for all four networks:

| was (`detection_test/lib/`) | now (`ufpipe`) |
|---|---|
| `build_stations.py --month` | `ufpipe/stations.py` — `build_year_table(year)` (KS/KG/GJ/NS) |
| `run_pnplus_month.py` / `run_seisbench_picker.py --month` | `python -m ufpipe.detection --model <m> --year <Y>` |
| `associate_daily.py --month` | `python -m ufpipe.association --model <m> --year <Y>` (daily-chunked) |
| `build_sac_and_pyocto.py` (reloc input builder) | `ufpipe/reloc_inputs.py` (fed from ufpipe's own outputs) |

The `relocate` stage (`ufpipe/relocate.py`) builds its inputs from ufpipe's own per-year detection +
association via `ufpipe.reloc_inputs`, then hands off to the downstream driver
`src/ufpipe/reloc_driver/run_picker_reloc.py --skip-build` (scaffold → HYPOINVERSE → QC → xcorr → HypoDD; the
external 15.PocketQuake engine). Only that driver and the PocketQuake engine remain live in
`detection_test/`; everything in this `lib/` folder is kept for reference/reproducibility of the original
4-picker comparison pilot but is not part of the pipeline.

The one shared constant source that still matters, `gj_config.py`, has been superseded by the
`config.ASSOC_*` / `config.KIM2011` / `config.REGION_CENTER` constants in `ufpipe/config.py` (same values).

To reproduce or relocate any year now:
```bash
conda run -n eqnet python -m ufpipe.detection   --model <picker> --year <Y>
conda run -n base  python -m ufpipe.association --model <picker> --year <Y>
conda run -n base  python -m ufpipe.run_pipeline --model <picker> --years <Y> --stage-from relocate --through dtcc
```
