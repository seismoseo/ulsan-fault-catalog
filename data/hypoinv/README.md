# KS_KG / HypoInv — directory guide

HYPOINVERSE location + post-location analysis for the Ulsan-fault catalog (KS + KG networks,
2010–2024). The **phasenet_plus** picker model is canonical; **stead** is retired (archived).

This directory was reorganized on 2026-06-09: the 37 mixed notebooks at the root were split into
**active (root)** vs **archived (`archive/`)**. Active notebooks were kept at the root *on purpose* —
they `import uf_cluster` / `uf_waveform_similarity` with a bare import and read caches/catalogs with
paths relative to this directory, so they must run with the working directory set here. Nothing was
deleted; nothing tracked was moved.

On 2026-06-10 the **repeater + anti-repeater** analyses (06/07/09/10 notebooks + their builders) moved
into **[`repeaters/`](repeaters/README.md)**. Those notebooks were made **cwd-independent** (setup cell
does `sys.path.insert(0, <HypoInv>)` + absolute `CACHE = wf.CACHE_DIR`), so they run from anywhere and
still hit the cache here. The shared modules + cache + catalogs stay at this root.

## What stays at the root (do not move — code references these by path)

The tracked `.py` modules use absolute `HYPO_DIR` paths (`uf_waveform_similarity.py:35-40`), so the
following are part of the stable data/code layer and must remain at the root:

- **Modules:** `uf_waveform_similarity.py`, `uf_cluster.py`, `event_sac_export.py`,
  `cross_component_blast.py`, `annual_location_density_plots.py`
- **Notebook builders:** `build_wf_nb.py`, `build_seq_nb.py` (+ `build_antirepeater_nb.py`)
- **Catalogs (phasenet_plus):** `catalog_phasenet_plus_2010_2024_blastclean.csv` (the headline
  catalog), `..._blastclean_with_ml[/_sheen].csv`, `..._declustered.csv`,
  `catalog_phasenet_plus_2010_2024.csv`, `cluster_summary_phasenet_plus_2010_2024.csv`,
  `subcatalog_phasenet_plus_uf_zone[_declustered].csv`, `cross_component_blast_candidates.csv`
- **Special-case catalogs:** `2014_Gyeongju_sequence.csv`, `gyeongju_kma_2010_2013.csv`
- **Data / model dirs:** `STA/`, `PHS/`, `kim1983/`, `kim2011/`, `wf_similarity_cache/`,
  `event_waveforms_ulsanfault/`, `event_waveforms_blastclean/`, `filtered_waveforms/`, `logs/`

## Active analysis notebooks (root, phasenet_plus)

| Notebook | What it does |
|---|---|
| `catalog_summary_phasenet_plus.ipynb` | master catalog summary (maps, FMD, stats) |
| `catalog_model_comparison.ipynb` | picker-model comparison |
| `03_blast_decluster_hdbscan_phasenet_plus.ipynb` | HDBSCAN + hour-of-day blast discrimination |
| `04_subregion_seismicity_phasenet_plus.ipynb` | east-of-fault subregion / long-term seismicity |
| `05_error_ellipses_phasenet_plus.ipynb` | HYPOINVERSE 95% location error ellipses |
| `05_cluster_spacetime_HHZ_phasenet_plus.ipynb` | space–time clustering / 3-D view |
| `07.Combine_2010_2024_location_results.ipynb` | merge per-year `.sum` → master catalog |
| `07.Examine_2014_location_result.ipynb` | 2014 Gyeongju sequence special case |

Waveform-similarity (blast screen) and the new anti-repeater study are **generated** from builders:
- `build_wf_nb.py {HHZ|HHN|HHE}` → `04_waveform_similarity_hdb_<COMP>_phasenet_plus.ipynb`
- `build_seq_nb.py {HHZ|HHN|HHE}` → `05_cluster_spacetime_<COMP>_phasenet_plus.ipynb`
- `build_antirepeater_nb.py {HHZ|HHN|HHE}` → `06_anti_repeaters_KGHDB_<COMP>_phasenet_plus.ipynb`

## `archive/` — kept, not deleted

| Folder | Contents | Why archived |
|---|---|---|
| `prep/` | `01.Make_PHS`, `02.Make_STA`, `04.Make_input_PHS`, `05.Cut_event_waveforms`, `06.Add_PhaseNet_picks`, `06.Export_event_waveforms` | one-time pipeline-prep notebooks, superseded by the automated `KS_KG/models/pipeline/` scripts |
| `stead/` | `03_blast_decluster_hdbscan`, `04_subregion_seismicity`, `05_error_ellipses`, `catalog_summary` notebooks **+** all `catalog_stead_*`, `cluster_summary_stead_*`, `subcatalog_stead_*` CSVs | the **stead** picker model is retired (phasenet_plus is canonical). These notebooks/catalogs are referenced only by each other |
| `per_year_epicenters/` | `03.Draw_HypInv_Epicenters_2010…2024` (15) | hardcoded-path per-year epicenter maps; superseded by `catalog_summary_phasenet_plus` |
| `autogen_wf_similarity/` | `04_waveform_similarity_hdb_HH{Z,N,E}_phasenet_plus` (3, ~28 MB each) | **regenerable** any time via `build_wf_nb.py` |
| `superseded/` | `07.Combine_2010_2013_location_results` | replaced by the 2010–2024 combine |
| `catalog_variants/` | `catalog_phasenet_plus_2010_2024_all.csv` | unreferenced intermediate catalog |

**Note for re-running archived notebooks:** archived stead notebooks read their catalogs from the
same `archive/stead/` folder; if a path no longer resolves, copy the needed catalog back to the
working directory or fix the path cell. Everything is preserved on disk — only moved.

All `archive/` contents (and root `*.ipynb`/`*.csv`) remain **git-ignored** (see the repo
`.gitignore`); the repo tracks code + small metadata only.
