# Development & version control

## First-time setup (per clone)

```bash
git clone git@github.com:seismoseo/ulsan-fault-catalog.git
cd ulsan-fault-catalog
bash tools/setup-git-filters.sh        # enable notebook output stripping
conda env create -f environment.yml && conda activate ulsan
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
git check-ignore KS_KG/continuous NS KS_KG/picks 01.PhaseNet_detection_test.ipynb
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
python KS_KG/models/pipeline/run_pipeline.py --model instance --years 2024
```
Outputs land in `KS_KG/models/instance/…`. Add the picker to the comparison set in the docs as needed.
(If you want pre-built notebooks for it too, generalize `build_original_tree.py`.)

**A new velocity model**: add `KS_KG/HypoInv/<name>/<name>_p.crh` + `_s.crh`, then
`run_hypoinverse.py --velmodel <name>`.

**HypoDD (relative relocation)**: implement under `KS_KG/models/<model>/hypodd/`, consuming the
HYPOINVERSE catalog + differential times. Add a `core.run_hypodd_year(...)` and a `hypodd.py` CLI,
and extend `run_pipeline.py`'s stage list. Reference implementation: `/home/msseo/works/relocDD-py/`.

**The NS network**: deferred. Reuse the same `--model`/path convention; the main new work is
station discovery + metadata for the additional stations (mostly active post-2018/19).

## Where to change defaults

All paths, detection/association parameters, and the region box live in
[`KS_KG/models/pipeline/config.py`](../KS_KG/models/pipeline/config.py). Change them there rather than
editing individual scripts.

## Safety guard

The scripts refuse `--model stead` writes unless `--force`, because `models/stead/` symlinks the
original reference run. Don't override this unless you intend to overwrite the baseline.
