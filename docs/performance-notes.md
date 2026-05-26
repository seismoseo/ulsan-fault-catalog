# Detection performance & failure modes — reference notes

Findings from diagnosing the slow/stalling 2010–2024 `original` re-run (2026-05-26), and
the fixes applied in `KS_KG/models/pipeline/{core,config}.py`. Keep this for later reference.

## TL;DR
- **GPU is preferred for all inference.** Detection runs `device="cuda"` and now warns loudly
  instead of silently falling back to CPU. The GPU was never the bottleneck (sat at ~6 %).
- Two independent problems made the `original` run effectively non-progressing:
  1. a **per-day worker-pool fork storm** (the general slowness), and
  2. a **pathologically fragmented station-day** that hangs obspy `merge()` (a hard stall at
     2010 day 082).

## Problem 1 — per-day ProcessPoolExecutor fork storm  (the ~35-day ETA)

**Symptom.** `original` ran ~9.3 min/day (ETA ~35 days); parent at ~3 % CPU; ~64 forked
`run_pipeline --model original` workers, **22 GB RSS each**, mostly idle (`STAT=S`, 0 % CPU);
GPU ~6 %. PhaseNet+ (no process pool) ran ~1.16 min/day by comparison.

**Root cause.** `core.detect_day` created
`concurrent.futures.ProcessPoolExecutor(max_workers=workers)` **inside the per-day loop** with
`workers=None` ⇒ `os.cpu_count()` (~64) workers. Each worker was **forked from the ~23 GB parent
that had already loaded PhaseNet + torch + CUDA**, just to preprocess ~7 stations, then torn down
— repeated for every day × every year (5,475×). Forking a CUDA-initialised, multi-GB process
dozens of times per day dominated wall-time; the workers themselves sat idle.

**Fix** (`core.run_detection_year` + `detect_day`):
- Create **one** `ProcessPoolExecutor` for the whole year, **reused** across days (pass it into
  `detect_day` via `executor=`).
- Create it with the **`forkserver`** start method (`mp_context=multiprocessing.get_context("forkserver")`)
  so workers are **lean** (a clean server forks them — they never inherit torch/CUDA/model) and
  the fork-after-CUDA-init hazard is gone.
- Create the pool **before** `sbm.PhaseNet.from_pretrained(...)`.
- **Cap workers** at `min(len(stations), config.DETECT_WORKERS)` (default `DETECT_WORKERS=16`) —
  more workers than stations is pointless.

**Verified.** Re-running 2010 day 001 with the fix produced **byte-identical picks** (733 = 733)
to the pre-fix CSV, on GPU (`device=cuda`), with 7 lean forkserver workers, in ~20 s (incl. model
load) vs ~9.3 min/day before. Picks are unchanged because only the preprocessing *parallelism*
changed, not the preprocessing or inference logic.

Note: `run_pipeline.py` must stay import-safe (it has the `if __name__ == "__main__":` guard) —
required because forkserver re-imports `__main__`.

## Problem 2 — fragmented station-day is very slow in obspy merge()  (2010.082)

**Symptom.** After the pool fix, the run appeared to stall at 2010 day 082: one worker pegged at
~100 % CPU for minutes; no picks written for a long time.

**Root cause.** `preprocess_station` does `read → interpolate → merge(method=1, fill_value=0)`.
For 2010.082, station **YSB** returns **55,572 traces** — but inspection showed this is the *full
3-component day* (~86,400 s × 3) stored as ~4.5 s **contiguous** miniSEED records, NOT telemetry
dropout: the data is real and continuous. obspy `merge()` is ~O(n²) in trace count, so it takes
~100 s for 55k records (not a true hang). A divide-and-conquer / concat merge isn't faster — the
cost is the 55k Python `Trace` objects, which every approach pays.

**Scope** (full 2010 scan + 2011–2024 sample): fragmentation is **YSB-only and essentially
2010-only** (142 of YSB's 365 days, a Jan–May block; 1 minor 2012 day; 2011 & 2013–2024 clean).
No other station is ever affected.

**Decision: LOSSLESS — keep, don't skip.** The data is genuine, and the *old* run had silently
**lost** YSB on these days (its giant forked workers errored on the merge → 0 YSB picks), so the
re-run *recovers* data. `config.MAX_SEGMENTS=2000` now only **logs** "fragmented — slow merge" and
processes the day normally; `HARD_MAX_SEGMENTS=300000` is a last-resort skip for a truly corrupt
file. Cost: ~142 YSB-2010 days × ~140 s ≈ 5 h (one-time), confined to 2010. (To re-decide,
`preprocess_station` is the single place that handles it.)

## Problem 3 — CPU contention on the shared box  (the other "why slow")

**Symptom.** Even after the pool fix, the SeisBench job's per-day merges were ~5× slower than in
isolation; `original` sat at ~2 % CPU while the box load was ~60.

**Root cause.** PhaseNet+ ran with numpy/torch threads defaulting to **all 64 cores (193 threads)**,
oversubscribing the machine and starving the (single-threaded) YSB merges. This is a **shared**
server (~5 users) — grabbing all cores is also antisocial.

**Fix.** Detection now sizes its preprocessing pool and `torch.set_num_threads` from the process
CPU affinity (`os.sched_getaffinity`), capped by `config.MAX_CORES`, so launching under
`taskset -c <cores>` auto-scopes the whole job (and all threads) to that budget. Current budget:
`original` → 8 cores, `phasenet_plus` → 16 cores (≈24 / 64). See "Polite CPU use" above.

## Polite CPU use (SHARED 64-core server)

This box is shared (≈5 users). Detection jobs must **not** grab all cores. Two layers:

1. **Hard cap with `taskset`** — pins a job and all its threads to a fixed core set:
   ```bash
   # original (SeisBench): preprocessing pool — give it ~6 cores
   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-5 \
     nohup python run_pipeline.py --model original --years 2010-2024 --velmodel kim2011 >> ... &
   # phasenet_plus (EQNet): numpy/torch heavy — give it ~8 cores
   OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 6-13 \
     nohup python run_pipeline.py --model phasenet_plus --years 2010-2024 --velmodel kim2011 >> ... &
   ```
   Total ≈14 / 64 cores. Live-cap a *running* job without restart:
   `taskset -acp 6-13 <pid>` (the `-a` covers all threads).
2. **Soft caps in code/env:** `config.DETECT_WORKERS` (preprocess pool size),
   `config.TORCH_THREADS` (`torch.set_num_threads`), and `OMP_NUM_THREADS`/`MKL_NUM_THREADS`
   (numpy/BLAS) — set these so a job's thread count ≈ its `taskset` core count (avoids
   193-threads-on-8-cores thrash). Left uncapped, PhaseNet+ grabbed 49 cores / 193 threads.

GPU is shared too but detection is not GPU-bound (~6% util), so GPU contention is minor.

## Operational notes
- The run is **resumable** (`skip_existing` skips days whose `picks_<year>.<day>.csv` exists), so
  the `original` job can be killed and relaunched without losing completed days.
- `pkill -f "run_pipeline.py --model original"` **self-matches** the issuing shell (its argv
  contains the pattern) and kills it too — use a bracket-safe pattern, e.g.
  `pkill -f "[r]un_pipeline.py --model original"`.
- Both background jobs shared one GPU + competed for CPU; PhaseNet+ alone used ~44 cores. Capping
  the `original` pool and keeping inference on GPU reduces the contention.
