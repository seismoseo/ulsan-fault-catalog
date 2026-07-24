# Development & version control

## First-time setup (per clone)

```bash
git clone git@github.com:seismoseo/ulsan-fault-catalog.git
cd ulsan-fault-catalog
bash tools/setup-git-filters.sh        # enable notebook output stripping
# two-env split on this server: pip install -e . in BOTH eqnet (detection) and base (association);
# fresh machine: conda env create -f environment.yml && conda activate uf-catalog
conda run -n eqnet pip install -e . --no-deps && conda run -n base pip install -e . --no-deps
```

`setup-git-filters.sh` registers a git **clean filter** (`tools/nbstrip.py`) so notebook outputs are
stripped from what git stores, while your working copies keep their rendered plots. This keeps the
repo small and diffs readable. It is a local setting (`.git/config`), hence the per-clone step.

## Day-to-day git workflow

```bash
git status                 # confirm only intended files changed (NO data!)
git add <files>
git commit -m "..."
git push
```

Before committing, **always sanity-check that no data slipped in**:

```bash
# nothing large staged:
git diff --cached --name-only | xargs -r du -h 2>/dev/null | sort -rh | head
# data paths must stay ignored:
git check-ignore KS_KG NS KS_KG/picks 01.PhaseNet_detection_test.ipynb
```

If a notebook keeps showing as "modified" with no real change, the strip filter isn't installed —
re-run `bash tools/setup-git-filters.sh`.

## Commit conventions

- Small, focused commits with imperative messages (e.g. `Add HypoDD stage to pipeline`).
- Keep code and its docs in the same commit when they change together.
- Never commit secrets or data; `.gitignore` is the safety net, but verify with `git status`.

## Extending the pipeline

**A new picker model** (e.g. `instance`):
```bash
python src/ufpipe/run_pipeline.py --model instance --years 2024
```
Outputs land in `outputs/models/instance/…`. Add the picker to the comparison set in the docs as needed.
(If you want pre-built notebooks for it too, generalize `build_original_tree.py`.)

**A new velocity model**: add `data/hypoinv/<name>/<name>_p.crh` + `_s.crh`, then
`run_hypoinverse.py --velmodel <name>` (the crh files are symlinked into each model's `HypoInv/`).

**HypoDD (relative relocation)**: implemented — the `relocate` stage (`src/ufpipe/relocate.py` +
`reloc_inputs.py`) is self-fed from ufpipe's own association and drives the external PocketQuake
HypoDD/xcorr engine via `src/ufpipe/reloc_driver/run_picker_reloc.py --skip-build`.

**The NS + GJ networks**: integrated — the per-year multi-network station table
(`src/ufpipe/stations.py`) covers KS/KG/GJ/NS; NS reads the pre-decimated `NS_100hz/` mirror.

## Where to change defaults

All paths, detection/association parameters, and the region box live in
[`src/ufpipe/config.py`](../src/ufpipe/config.py). Change them there rather than
editing individual scripts.

## Safety guard

The scripts refuse `--model stead` writes unless `--force`, because `models/stead/` symlinks the
original reference run. Don't override this unless you intend to overwrite the baseline.
