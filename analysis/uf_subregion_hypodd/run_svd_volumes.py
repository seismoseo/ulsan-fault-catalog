#!/usr/bin/env python
"""Per-volume HypoDD SVD relocation + bootstrap uncertainties (Phase 3).

Relocates the FULL 1 km^3 in-cube event sets around the two largest UF events (nb30 volumes:
M3.89 2014-09-23 -> 111 events "m389"; M3.73 2023-11-29 -> 58 events "m373", incl. the co-located
C6 swarm, background and ML-less events) with HypoDD **SVD (ISOLV=1)** for formal errors, plus a
**global-resampling bootstrap** (vendored from PocketQuake korea-cluster-relocation
pipeline/core/hypodd.py, lines ~190-745) for percentile 95% uncertainties. Catalog-start seeding
(independent re-inversion; --seed-from-whole-box optional). Adaptive bootstrap: n=200 probe, auto
extend to n=1000 iff a single run takes <5 min.

Stages (idempotent; --force to redo): select | extract | run | boot | analyze | all
Usage:  python run_svd_volumes.py --volume all --stage all [--nboot adaptive|200|1000]
        [--cores 24] [--seed 0] [--quick] [--force] [--seed-from-whole-box]
Runs in base env (numpy/pandas + kma_absolute_location.nnd/clustering only).
"""
import argparse, dataclasses, glob, json, os, re, shutil, subprocess, sys, tempfile, time
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/msseo/works/16.kma_absolute_location")
from kma_absolute_location import nnd, clustering                      # canonical NND + UTM

# ----------------------------------------------------------------- constants (mirror nb30)
# Restructured 2026-07: KS_KG/local_magnitudes -> analysis/local_magnitudes; uf_subregion_hypodd -> analysis/.
REPO    = "/home/msseo/works/02.Ulsan_Fault_detection"
RELOC   = f"{REPO}/analysis/local_magnitudes/catalog_ml_heo_ufonly_reloc.csv"
RUN03   = ("/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/"
           "uf_subregion_reuse/2.HypoDD/03.dt.cc_kim2011")
WF100   = ("/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/"
           "uf_subregion_reuse/waveforms_100km")
MEIDX   = f"{REPO}/analysis/uf_subregion_hypodd/uf_subregion/members_event_idx.csv"
BASE    = f"{REPO}/analysis/uf_subregion_hypodd/svd_volumes"
HYPODD  = os.path.expanduser("~/bin/hypoDD")
DF_UF, B_NND, LINKR, HALF = 1.2, 1.0, 1.0, 0.5
ERRCAP_M = 150.0     # a well-located event has bootstrap 95% half-width < this on every axis (n_boot>=50);
                     # excludes events whose intra-volume links can't hold them (e.g. 0-cc-link strays that
                     # flew to 26 km depth once the outside-cube links were dropped). Used for plane/separation
                     # fits AND the notebook's "constrained" display gate.
EXPECT  = {"m389": dict(n=111, msmag=3.89), "m373": dict(n=58, msmag=3.73)}   # hard gates (nb30)
RC24 = ["id","lat","lon","depth","x","y","z","ex","ey","ez","yr","mo","dy","hr","mi","sc",
        "mag","nccp","nccs","nctp","ncts","rcc","rct","cid"]

def read_reloc(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=RC24)
    for c in ("x","y","z","ex","ey","ez"): df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def meta_path(vol): return os.path.join(BASE, vol, "run_meta.json")
def load_meta(vol):
    p = meta_path(vol)
    return json.load(open(p)) if os.path.exists(p) else {}
def save_meta(vol, upd):
    m = load_meta(vol); m.update(upd)
    json.dump(m, open(meta_path(vol), "w"), indent=1, default=str)

# ================================================================= stage A: select (nb30 logic, vendored)
def stage_select(force=False):
    outs = {v: os.path.join(BASE, v, "volume_events.csv") for v in EXPECT}
    if not force and all(os.path.exists(p) for p in outs.values()):
        print("[select] cached volume_events.csv found for all volumes"); return
    rl = pd.read_csv(RELOC); rl["event_time"] = pd.to_datetime(rl.event_time, format="ISO8601", utc=True, errors="coerce")
    rl = rl.dropna(subset=["lat","lon","depth","event_time","ml_ufcorr_reloc"]).copy()
    g = rl[rl.n_used >= 3].copy()
    g["event_id"] = g.event_idx.astype(int).astype(str)
    g["t_year"] = g.event_time.dt.strftime("%Y%m%d%H%M%S").map(nnd.decimal_year)   # CANONICAL
    g = g.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep","ml_ufcorr_reloc":"kma_mag"})
    g = g.sort_values("t_year").reset_index(drop=True)
    nd = nnd.compute_nnd(g, b=B_NND, D=DF_UF, mmin=None, metric="3d")
    e0, _ = nnd.fit_eta0(nd.eta.values)
    lab = nnd.build_families(nd, e0, g.event_id.values, link_rmax_km=LINKR)
    g["Cluster"] = g.event_id.map(lab).fillna(-1).astype(int)
    CLU = dict(zip(g.event_idx.astype(int), g.Cluster.astype(int)))
    fammax = g[g.Cluster >= 0].groupby("Cluster").kma_mag.max().sort_values(ascending=False)
    targets = list(fammax.head(2).index)                       # [C(M3.89), C(M3.73)]

    # full relocated catalog (id -> ts -> event_idx map; identical to nb30)
    r0 = read_reloc(os.path.join(RUN03, "hypoDD.reloc")); r0["ncc"] = r0.nccp + r0.nccs
    dirs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(WF100, "20*")))
    id2ts = {200000 + i: ts for i, ts in enumerate(dirs)}
    mei = pd.read_csv(MEIDX).sort_values("event_idx")
    mei["ts"] = pd.to_datetime(mei.time, utc=True, format="ISO8601").dt.floor("s").dt.strftime("%Y%m%d%H%M%S")
    ts2e = {}
    for e, t in zip(mei.event_idx.astype(int), mei.ts): ts2e.setdefault(t, e)
    allc = r0.copy(); allc["event_idx"] = allc.id.map(id2ts).map(ts2e)
    allc = allc.merge(rl[["event_idx","ml_ufcorr_reloc","n_used"]], on="event_idx", how="left").drop_duplicates("id")
    allc["has_ml"] = allc.ml_ufcorr_reloc.notna() & (allc.n_used >= 3)
    scf = allc.sc.clip(0, 59.999)
    allc["time"] = pd.to_datetime(dict(year=allc.yr, month=allc.mo, day=allc.dy, hour=allc.hr, minute=allc.mi,
                   second=scf.astype(int), microsecond=((scf - scf.astype(int)) * 1e6).astype(int)), utc=True, errors="coerce")
    allc = allc.dropna(subset=["time","lat","lon","depth"]).copy()
    allc["magU"] = np.where(allc.has_ml, allc.ml_ufcorr_reloc, allc.mag)
    au, _ = clustering.to_utm(allc.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep"}))
    allc["x_m"] = au.x_m.values; allc["y_m"] = au.y_m.values; allc["depth_m"] = au.depth_m.values
    l0 = pd.read_csv(os.path.join(RUN03, "hypoDD.loc"), sep=r"\s+", header=None,
                     names=["id","lat","lon","depth","x","y","z","ex","ey","ez","yr","mo","dy","hr","mi","sc","mag","cid"])[["id","lat","lon","depth"]]
    lu, _ = clustering.to_utm(l0.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep"}))
    l0["x0"], l0["y0"], l0["z0"] = lu.x_m.values, lu.y_m.values, lu.depth_m.values
    allc = allc.merge(l0[["id","x0","y0","z0"]], on="id", how="left")
    allc["shift_km"] = np.sqrt((allc.x_m - allc.x0)**2 + (allc.y_m - allc.y0)**2 + (allc.depth_m - allc.z0)**2) / 1000.0
    allc["nlinks"] = allc.nccp + allc.nccs + allc.nctp + allc.ncts
    # ct-only events (0 surviving cc obs, e.g. the M3.89 mainshock whose waveform is too dissimilar to its
    # aftershocks to cross-correlate) are located by catalog dt only — kept & relocated, but flagged so they do
    # NOT drive the high-precision plane/shape fit (they lack cc precision). shift/nlinks gates unchanged.
    allc["suspect"] = (allc.shift_km > 3.0) | (allc.nlinks < 10) | (allc.ncc == 0)

    HW = HALF * 1000.0
    for vol, k in zip(("m389","m373"), targets):
        memset = set(g.loc[g.Cluster == k, "event_idx"].astype(int))
        mem = allc[allc.event_idx.isin(memset)]
        cE, cN, cZ = float(mem.x_m.mean()), float(mem.y_m.mean()), float(mem.depth_m.mean())
        ms = mem.loc[mem.magU.idxmax()]
        # A cluster relocation must not omit a member. Take the UNION of the 1 km context cube and the FULL NND
        # family, so every grouped event is relocated — in particular the M3.89 mainshock, whose ct-only whole-box
        # position lands just outside the cube. (For m373 all family members already fall in the cube; no change.)
        in_cube = (allc.x_m.between(cE-HW, cE+HW)) & (allc.y_m.between(cN-HW, cN+HW)) & (allc.depth_m.between(cZ-HW, cZ+HW))
        box = allc[in_cube | allc.event_idx.isin(memset)].copy()
        box["is_member"] = box.event_idx.isin(memset)
        box["nnd_cluster"] = box.event_idx.map(CLU)
        box["cat"] = np.where(box.is_member, "member",
                     np.where(box.nnd_cluster.isna(), "not-in-pop",
                     np.where(box.nnd_cluster == -1, "background", "other-family")))
        box["e_km"] = (box.x_m - cE)/1000; box["n_km"] = (box.y_m - cN)/1000; box["z_km"] = (box.depth_m - cZ)/1000
        box = box.sort_values("time").reset_index(drop=True)
        exp = EXPECT[vol]
        # the mainshock is the same physical event (M~3.89/3.73); its cube membership shifts when the whole-box
        # relocation changes (e.g. the 2026-07 adaptive-damping re-relocation moved abs positions ~1 km), so the
        # count is a WARNING not a hard gate. The mainshock magnitude must still match (guards a wrong-cluster pick).
        assert abs(float(ms.magU) - exp["msmag"]) < 0.15, (
            f"{vol}: mainshock M{float(ms.magU):.2f} != expected M{exp['msmag']} — wrong cluster picked")
        if len(box) != exp["n"]:
            print(f"  [note] {vol}: n_in_cube {len(box)} != nb30's {exp['n']} (expected after re-relocation)")
        d = os.path.join(BASE, vol); os.makedirs(d, exist_ok=True)
        cols = ["id","event_idx","cat","nnd_cluster","is_member","has_ml","suspect","lat","lon","depth",
                "x_m","y_m","depth_m","e_km","n_km","z_km","magU","time","shift_km","nlinks","ncc"]
        box[cols].to_csv(outs[vol], index=False)
        save_meta(vol, dict(volume=vol, family=int(k), n_in_cube=len(box),
                            center_utm=[cE, cN, cZ], mainshock_id=int(ms.id),
                            mainshock_mag=float(ms.magU), mainshock_time=str(ms.time),
                            cat_counts=box.cat.value_counts().to_dict(),
                            n_suspect=int(box.suspect.sum())))
        print(f"[select] {vol}: {len(box)} events (family {k}, mainshock M{ms.magU:.2f} id {int(ms.id)})")

# ================================================================= stage B: extract
def subset_dt(src, dst, ids):
    keep = False; nb_in = nb_keep = nobs = 0
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            if line.lstrip().startswith("#"):
                t = line.split(); nb_in += 1
                keep = int(t[1]) in ids and int(t[2]) in ids
                if keep: nb_keep += 1
            elif keep:
                nobs += 1
            if keep:
                out.write(line)
    return nb_in, nb_keep, nobs

def stage_extract(vol, force=False):
    d = os.path.join(BASE, vol)
    if not force and os.path.exists(os.path.join(d, "dt.ct")) and os.path.exists(os.path.join(d, "event.dat")):
        print(f"[extract] {vol}: cached subset inputs found"); return
    ve = pd.read_csv(os.path.join(d, "volume_events.csv")); ids = set(ve.id.astype(int))
    ncc = subset_dt(os.path.join(RUN03, "dt.cc_0.7_combined"), os.path.join(d, "dt.cc_pristine"), ids)
    shutil.copyfile(os.path.join(d, "dt.cc_pristine"), os.path.join(d, "dt.cc_0.7_combined"))
    nct = subset_dt(os.path.join(RUN03, "dt.ct"), os.path.join(d, "dt.ct"), ids)
    rows = [ln for ln in open(os.path.join(RUN03, "event.dat")) if int(ln.split()[-1]) in ids]
    assert len(rows) == len(ids), f"event.dat subset {len(rows)} != {len(ids)}"
    open(os.path.join(d, "event.dat"), "w").writelines(rows)
    shutil.copyfile(os.path.join(RUN03, "station.dat"), os.path.join(d, "station.dat"))
    save_meta(vol, dict(n_cc_pairs=ncc[1], n_cc_obs=ncc[2], n_ct_pairs=nct[1], n_ct_obs=nct[2]))
    print(f"[extract] {vol}: cc {ncc[1]} pairs/{ncc[2]} obs | ct {nct[1]} pairs/{nct[2]} obs | events {len(rows)}")

# ================================================================= stage C: hypoDD.inp
DAMPS_SVD = (8, 8, 8, 6, 6, 6, 6)          # gwangyang _DTCC_ITERS; inert under SVD, sane on LSQR fallback

def _isnum(s):
    try: float(s); return True
    except ValueError: return False

def make_inp(isolv=1, damps=DAMPS_SVD, istart=1):
    txt = open(os.path.join(RUN03, "hypoDD.inp")).read()
    # set ISTART + ISOLV (keep NSET). ISTART=2 = start from event.dat (catalog/absolute) locations rather than
    # the cluster centroid (ISTART=1) — essential for large-scale relocation so the extent isn't collapsed.
    txt = re.sub(r"(\*--- solution control: ISTART ISOLV NSET\n\s*)\d+\s+\d+(\s+\d+)", rf"\g<1>{istart}  {isolv}\g<2>", txt)
    # weighting rows = the 10-numeric-token rows inside the "data weighting" section (robust to ANY DAMP value,
    # so it still works after the production template's DAMP was changed to the adaptive per-set schedule).
    lines, wi, insec = txt.splitlines(keepends=True), 0, False
    for i, ln in enumerate(lines):
        if "data weighting" in ln: insec = True; continue
        if "1D model" in ln: insec = False
        t = ln.split()
        if insec and len(t) == 10 and all(_isnum(x) for x in t):
            d = damps[wi] if wi < len(damps) else damps[-1]
            lines[i] = "    " + "  ".join(t[:-1] + [str(d)]) + "\n"
            wi += 1
    assert wi == 7, f"expected 7 weighting rows, mutated {wi}"
    return "".join(lines)

def stage_inp(vol, seed_from_whole_box=False, force=False):
    d = os.path.join(BASE, vol); p = os.path.join(d, "hypoDD.inp")
    if not force and os.path.exists(p):
        print(f"[inp] {vol}: cached hypoDD.inp found"); return
    open(p, "w").write(make_inp(isolv=1))
    if seed_from_whole_box:
        main = read_reloc(os.path.join(RUN03, "hypoDD.reloc"))
        seeded = _seed_event_dat(os.path.join(d, "event.dat"), main)   # READ first (open('w') below truncates!)
        assert seeded.count("\n") >= 1, f"{vol}: seeded event.dat is empty"
        with open(os.path.join(d, "event.dat"), "w") as f: f.write(seeded)
    save_meta(vol, dict(isolv_requested=1, damps=list(DAMPS_SVD), seed_from_whole_box=bool(seed_from_whole_box)))
    print(f"[inp] {vol}: wrote SVD hypoDD.inp (ISOLV=1, DAMP {DAMPS_SVD})")

# ================================================================= stage D: run (+ MAXDATA0 ladder)
class _MaxData0Overflow(RuntimeError): pass
class _MaxEve0Overflow(RuntimeError): pass

def _exec_hypodd_once(d, timeout=None):
    """Vendored from korea-cluster-relocation pipeline/core/hypodd.py:_exec_hypodd_once (attribution)."""
    os.makedirs(os.path.join(d, "reloc"), exist_ok=True)
    _stale = os.path.join(d, "hypoDD.reloc")
    if os.path.exists(_stale): os.remove(_stale)   # a stale reloc must not mask a failed run
    proc = subprocess.run([HYPODD, "hypoDD.inp"], cwd=d, capture_output=True, text=True,
                          errors="replace", timeout=timeout)
    sum_text = (proc.stdout or "") + (("\n--- stderr ---\n" + proc.stderr) if proc.stderr else "")
    open(os.path.join(d, "hypoDD.sum"), "w").write(sum_text)
    for f in glob.glob(os.path.join(d, "*.reloc.0*")):
        shutil.move(f, os.path.join(d, "reloc", os.path.basename(f)))
    reloc = os.path.join(d, "hypoDD.reloc")
    if not os.path.exists(reloc) or os.path.getsize(reloc) == 0:
        comb = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if "MAXDATA0" in comb: raise _MaxData0Overflow("SVD MAXDATA0 overflow")
        if "MAXEVE0"  in comb: raise _MaxEve0Overflow("SVD MAXEVE0 overflow")
        tail = comb.strip().splitlines()[-5:]
        raise RuntimeError("hypoDD produced no relocations:\n  " + "\n  ".join(tail))
    return reloc

def _prune_cc(d, cap=None, ccmin=None):
    """Rewrite dt.cc_0.7_combined from the pristine subset: cap obs/pair by highest ccweight and/or ccmin."""
    src, dst = os.path.join(d, "dt.cc_pristine"), os.path.join(d, "dt.cc_0.7_combined")
    with open(src) as f, open(dst, "w") as out:
        hdr, obs = None, []
        def flush():
            if hdr is None: return
            keep = [o for o in obs if ccmin is None or float(o.split()[2]) >= ccmin]
            if cap is not None and len(keep) > cap:
                keep = sorted(keep, key=lambda o: -float(o.split()[2]))[:cap]
            if keep: out.write(hdr); out.writelines(keep)
        for line in f:
            if line.lstrip().startswith("#"):
                flush(); hdr, obs = line, []
            else:
                obs.append(line)
        flush()

_CND_RE = re.compile(r"acond \(CND\)=\s*([0-9.]+)")
_KV_RE = re.compile(r"([a-z_]+)=\s*(-?[0-9.]+)")

def _log_cnds(d):
    p = os.path.join(d, "hypoDD.log")
    return [float(m) for m in _CND_RE.findall(open(p, errors="ignore").read())] if os.path.exists(p) else []

# --- PocketQuake-style per-set adaptive LSQR damping (vendored from pipeline/core/hypodd.py) ---------------
_INP_SIGS = None
def _template_sigs():
    """The 7 weighting-set signatures (cols WTCCP..WDCT) from the hypoDD.inp template, in order."""
    global _INP_SIGS
    if _INP_SIGS is None:
        rows, insec = [], False
        for ln in open(os.path.join(RUN03, "hypoDD.inp")):
            if "data weighting" in ln: insec = True; continue
            if "1D model" in ln: insec = False
            t = ln.split()
            if insec and len(t) == 10 and all(_isnum(x) for x in t):
                rows.append(tuple(float(x) for x in t[1:9]))
        _INP_SIGS = rows
        assert len(_INP_SIGS) == 7, f"expected 7 weighting sets, found {len(_INP_SIGS)}"
    return _INP_SIGS

_NITER = None
def _template_niter():
    """NITER (iterations per weighting set) from the template — same for all 7 rows."""
    global _NITER
    if _NITER is None:
        insec = False
        for ln in open(os.path.join(RUN03, "hypoDD.inp")):
            if "data weighting" in ln: insec = True; continue
            if "1D model" in ln: insec = False
            t = ln.split()
            if insec and len(t) == 10 and all(_isnum(x) for x in t): _NITER = int(float(t[0])); break
    return _NITER or 4

def _iter_cnds_by_sig(log_path):
    """(weighting-signature, CND) per LSQR iteration from hypoDD.log, in order. Signature = the 8 echoed
    weighting columns; used only to detect when one weighting SET gives way to the next (consecutive change)."""
    out, cur = [], None
    if not os.path.exists(log_path): return out
    for line in open(log_path, errors="ignore"):
        if "Weighting parameters for this iteration" in line: cur = {}
        elif cur is not None and ("wt_ccp=" in line or "wt_ctp=" in line):
            cur.update({k: float(v) for k, v in _KV_RE.findall(line)})
        elif cur is not None and "acond (CND)=" in line:
            m = _CND_RE.search(line)
            if m:
                sig = tuple(cur.get(k) for k in ("wt_ccp","wt_ccs","maxr_cc","maxd_cc","wt_ctp","wt_cts","maxr_ct","maxd_ct"))
                out.append((sig, float(m.group(1))))
            cur = None
    return out

def _max_cnd_per_set(log_path, niter=None):
    """Worst (max) CND per weighting set, grouping CONSECUTIVE iterations by their weighting signature (robust
    to variable NITER — e.g. the whole box runs 6 ct-phase + 4 cc-phase iterations per set, not a clean i//4).
    Within a set the echoed signature is self-consistent, so a signature CHANGE marks a new set (order 0..6)."""
    it = _iter_cnds_by_sig(log_path)
    if not it:                                                   # fallback: fixed NITER order mapping
        niter = niter or _template_niter()
        cnds = [float(m) for m in _CND_RE.findall(open(log_path, errors="ignore").read())] if os.path.exists(log_path) else []
        res = {}
        for i, c in enumerate(cnds):
            if i // niter < 7: res[i // niter] = max(res.get(i // niter, 0.0), c)
        return res
    res = {}; gi = -1; prev = object()
    for sig, cnd in it:
        if sig != prev: gi += 1; prev = sig
        if gi < 7: res[gi] = max(res.get(gi, 0.0), cnd)
    return res

def _adaptive_damp(d, cnd_range=(40.0, 80.0), max_attempts=12, damp0=60, istart=1):
    """PocketQuake's adaptive LSQR damping: independently nudge EACH weighting set's DAMP until that set's
    worst-iteration CND lands in cnd_range, so the whole run is well-conditioned (not just the last iteration).
    Runs in dir d (event.dat/dt/station already present). Returns (best_damps 7-tuple, per_set_cnd dict).
    newd = damp * (CND/mid)**0.5 (higher DAMP -> lower CND); clamped 1..2000; keeps the best of max_attempts.
    istart=2 = start from event.dat locations (avoids the ISTART=1 centroid-collapse contraction on large clusters)."""
    lo, hi = cnd_range; mid = (lo + hi) / 2.0
    damps = [int(damp0)] * 7; best, hist = None, []
    for _ in range(max_attempts):
        open(os.path.join(d, "hypoDD.inp"), "w").write(make_inp(isolv=2, damps=tuple(damps), istart=istart))
        _exec_hypodd_once(d)
        cnds = _max_cnd_per_set(os.path.join(d, "hypoDD.log"))
        if not cnds: break
        score = max(max(0.0, c - hi) + max(0.0, lo - c) for c in cnds.values())
        hist.append((list(damps), {k: round(v, 1) for k, v in sorted(cnds.items())}, round(score, 1)))
        if best is None or score < best[0]: best = (score, list(damps), dict(cnds))
        if score <= 0.0: break
        for i, c in cnds.items():
            damps[i] = int(min(2000, max(1, round(damps[i] * (c / mid) ** 0.5))))
    bdamps = best[1] if best else damps
    open(os.path.join(d, "hypoDD.inp"), "w").write(make_inp(isolv=2, damps=tuple(bdamps), istart=istart))
    _exec_hypodd_once(d)                                         # final run corresponds to the best damping
    with open(os.path.join(d, "damping_calibration.txt"), "w") as f:
        f.write(f"adaptive per-set LSQR damping, target CND {lo:.0f}-{hi:.0f}\nattempt: damps -> {{set:maxCND}} (score)\n")
        for a, (dm, cn, sc) in enumerate(hist): f.write(f"  {a}: {dm} -> {cn}  ({sc})\n")
        f.write(f"chosen: {list(bdamps)}\n")
    return tuple(bdamps), (best[2] if best else {})

def stage_run(vol, force=False):
    d = os.path.join(BASE, vol)
    if not force and os.path.exists(os.path.join(d, "hypoDD.reloc")) and load_meta(vol).get("solver"):
        print(f"[run] {vol}: cached hypoDD.reloc found ({load_meta(vol).get('solver')})"); return
    ladder = [("as-is", {}), ("cap20", dict(cap=20)), ("cap12", dict(cap=12)), ("cap8", dict(cap=8)),
              ("cc>=0.75", dict(ccmin=0.75)), ("cc>=0.80", dict(ccmin=0.80))]
    t0 = time.time(); solver = None; rung_used = None
    for rung, prune in ladder:
        if prune: _prune_cc(d, **prune)
        try:
            t1 = time.time(); _exec_hypodd_once(d); solver, rung_used = "SVD", rung
            save_meta(vol, dict(main_run_s=round(time.time() - t1, 1))); break
        except _MaxData0Overflow:
            print(f"[run] {vol}: MAXDATA0 at rung '{rung}' -> next prune"); continue
        except _MaxEve0Overflow:
            print(f"[run] {vol}: MAXEVE0 -> LSQR fallback (formal errors N/A)"); break
    if solver is None:                                          # LSQR fallback with DAMP scan
        _prune_cc(d)                                            # restore pristine cc
        best = None
        for damp in (600, 300, 150, 80, 40, 20):
            open(os.path.join(d, "hypoDD.inp"), "w").write(make_inp(isolv=2, damps=(damp,)*7))
            try:
                t1 = time.time(); _exec_hypodd_once(d); cnds = _log_cnds(d)
                fin = cnds[-1] if cnds else np.nan
                if best is None or abs(fin - 60) < abs(best[1] - 60): best = (damp, fin, time.time() - t1)
                if 40 <= fin <= 80: break
            except Exception as e:
                print(f"[run] {vol}: LSQR damp={damp} failed ({e})")
        assert best is not None, f"{vol}: no solver converged"
        damp, fin, tt = best
        open(os.path.join(d, "hypoDD.inp"), "w").write(make_inp(isolv=2, damps=(damp,)*7))
        _exec_hypodd_once(d); solver, rung_used = f"LSQR(damp={damp})", "lsqr-fallback"
        save_meta(vol, dict(main_run_s=round(tt, 1)))
    rel = read_reloc(os.path.join(d, "hypoDD.reloc"))
    ve = pd.read_csv(os.path.join(d, "volume_events.csv"))
    dropped = sorted(set(ve.id.astype(int)) - set(rel.id.astype(int)))
    dropcat = ve[ve.id.isin(dropped)].cat.value_counts().to_dict()
    save_meta(vol, dict(solver=solver, prune_rung=rung_used, n_relocated=len(rel),
                        n_dropped=len(dropped), dropped_ids=dropped, dropped_by_cat=dropcat,
                        cnds=_log_cnds(d)[-7:], n_clusters_cid=int(rel.cid.nunique()),
                        depth_range=[float(rel.depth.min()), float(rel.depth.max())],
                        total_stage_s=round(time.time() - t0, 1)))
    print(f"[run] {vol}: {solver} rung={rung_used} | relocated {len(rel)}/{len(ve)} "
          f"(dropped {dropped}) | cid n={rel.cid.nunique()} | {time.time()-t0:.0f}s")

# ================================================================= stage D2: reported (primary) solution
def stage_primary(vol, force=False):
    """Reported per-volume solution = LSQR-CND seeded on whole-box LSQR.

    The DD differential data barely constrains the absolute CENTROID depth (a near-null-space direction):
    undamped SVD wanders down it (whole-box 11.5 km -> 9.5 km for m373, seed-dependent), while light-damped
    LSQR (CND 40-80) is softly anchored to the seed and holds the physical whole-box centroid (11.49 km at
    every damping). SVD and LSQR-CND give the SAME relative structure (~10 m median, identical thickness), so
    LSQR-CND on the whole-box seed is strictly better here: physical absolute depth, same shape, no dropouts.
    The SVD run is preserved as hypoDD.reloc.svd (diagnostic that documents the null-space drift)."""
    d = os.path.join(BASE, vol)
    if not force and load_meta(vol).get("primary_solver"):
        print(f"[primary] {vol}: cached ({load_meta(vol).get('primary_solver')})"); return
    reloc = os.path.join(d, "hypoDD.reloc")
    # 1. preserve the SVD run as a diagnostic (only if the current reloc is the SVD one)
    if os.path.exists(reloc) and str(load_meta(vol).get("solver", "")).startswith("SVD"):
        shutil.copyfile(reloc, os.path.join(d, "hypoDD.reloc.svd"))
        sv = read_reloc(reloc)
        save_meta(vol, dict(svd_depth_range=[float(sv.depth.min()), float(sv.depth.max())]))
    # 2. rebuild event.dat from the canonical whole-box LSQR positions (volume_events.csv)
    ve = pd.read_csv(os.path.join(d, "volume_events.csv"))
    seeded = _seed_event_dat(os.path.join(d, "event.dat"),
                             pd.DataFrame(dict(id=ve.id, lat=ve.lat, lon=ve.lon, depth=ve.depth)))
    assert seeded.count("\n") >= 1, f"{vol}: whole-box seed empty"
    with open(os.path.join(d, "event.dat"), "w") as f: f.write(seeded)
    # 3. PocketQuake-style PER-SET adaptive damping: tune each weighting set's DAMP so its worst-iteration CND
    #    lands in 40-80 (not a single global DAMP checked only at the last iteration — that under/over-damps
    #    the cc/ct sets differently and misses small clusters). Keeps the resulting reloc.
    t1 = time.time()
    bdamps, per_set_cnd = _adaptive_damp(d, cnd_range=(40.0, 80.0), max_attempts=12, istart=2); tt = time.time() - t1
    inband = sum(1 for c in per_set_cnd.values() if 40 <= c <= 80)
    rel = read_reloc(reloc)
    dropped = sorted(set(ve.id.astype(int)) - set(rel.id.astype(int)))
    save_meta(vol, dict(primary_solver="LSQR-CND(adaptive)", primary_damps=list(bdamps),
                        primary_cnd_per_set={int(k): round(float(v), 1) for k, v in per_set_cnd.items()},
                        primary_cnd_inband=f"{inband}/{len(per_set_cnd)}",
                        primary_cnd=round(float(np.median(list(per_set_cnd.values()))), 1) if per_set_cnd else None,
                        primary_run_s=round(tt, 1), main_run_s=round(tt, 1),
                        n_relocated=len(rel), n_dropped=len(dropped), dropped_ids=dropped,
                        dropped_by_cat=ve[ve.id.isin(dropped)].cat.value_counts().to_dict(),
                        n_clusters_cid=int(rel.cid.nunique()),
                        depth_range=[float(rel.depth.min()), float(rel.depth.max())]))
    print(f"[primary] {vol}: LSQR-CND adaptive DAMP={list(bdamps)} | per-set CND in-band {inband}/{len(per_set_cnd)} "
          f"(med {np.median(list(per_set_cnd.values())):.0f}) | relocated {len(rel)}/{len(ve)} | "
          f"depth med {rel.depth.median():.2f} km (SVD was {load_meta(vol).get('svd_depth_range')})")

# ================================================================= stage E: bootstrap (vendored + adapted)
def _parse_dt_blocks(path):
    blocks, cur = [], None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip(): continue
            if line.lstrip().startswith("#"):
                cur = (line, []); blocks.append(cur)
            elif cur is not None:
                cur[1].append(line)
    return blocks

def _write_dt_blocks(path, blocks):
    with open(path, "w") as f:
        for header, obs in blocks:
            if not obs: continue
            f.write(header + "\n"); f.writelines(o + "\n" for o in obs)

def _resample_within_pair(blocks, rng):
    # WITHIN-PAIR (block) bootstrap: resample each pair's observations WITH REPLACEMENT, keeping the pair's obs
    # COUNT fixed (so every pair survives -> full graph connectivity preserved -> every event relocates in every
    # replica -> no whole-replica failures, n_boot ~ n for all events). Captures the station/phase-selection
    # (observation-noise) uncertainty. Chosen over global resampling because global empties some pairs, which
    # disconnects weakly-linked events and fails whole replicas (the inconsistency the user flagged). Validated:
    # m373 aligned ez95 25 m (within-pair) vs 32 m (global) with 0 failures vs 16-76%.
    out = []
    for h, obs in blocks:
        if not obs: continue
        idx = rng.integers(0, len(obs), len(obs))
        out.append((h, [obs[k] for k in idx]))
    return out

def _resample_global(blocks, rng):
    # GLOBAL bootstrap: pool EVERY observation across all pairs, draw len(pool) with replacement, regroup by
    # pair. Whole pairs can vanish (0 draws) or be duplicated -> captures PAIR-SELECTION variance on top of the
    # observation noise, giving a LARGER, more complete uncertainty than within-pair. Cost: some pairs empty ->
    # weakly-linked events can drop from a replica (whole-replica variability). This is the classic dt-bootstrap.
    pool = [(bi, o) for bi, (_h, obs) in enumerate(blocks) for o in obs]
    if not pool: return blocks
    regrouped = [[] for _ in blocks]
    for k in rng.integers(0, len(pool), size=len(pool)):
        bi, o = pool[k]; regrouped[bi].append(o)
    return [(blocks[bi][0], regrouped[bi]) for bi in range(len(blocks)) if regrouped[bi]]

def _seed_event_dat(src_event_dat, reloc_df):
    pos = {int(r.id): (float(r.lat), float(r.lon), float(r.depth)) for r in reloc_df.itertuples()}
    out = []
    for line in open(src_event_dat):
        t = line.split()
        if len(t) >= 10 and t[-1].lstrip("-").isdigit() and int(t[-1]) in pos:
            la, lo, dp = pos[int(t[-1])]
            t[2], t[3], t[4] = f"{la:.4f}", f"{lo:.4f}", f"{dp:.3f}"
            out.append("  ".join(t) + "\n")
        else:
            out.append(line if line.endswith("\n") else line + "\n")
    return "".join(out)

def _calibrate_lsqr(d, seeded, base_blocks):
    """Calibrate PER-SET LSQR damping (PocketQuake-style: each weighting set's worst-iteration CND in 40-80)
    on the un-resampled, solution-seeded data; the SAME per-set DAMP is reused for every bootstrap replica.
    Returns (inp_text, median_CND, damps_list)."""
    pd_ = tempfile.mkdtemp(prefix="cal_")
    try:
        shutil.copyfile(os.path.join(d, "station.dat"), os.path.join(pd_, "station.dat"))
        with open(os.path.join(pd_, "event.dat"), "w") as f: f.write(seeded)
        for fn, blk in base_blocks.items(): _write_dt_blocks(os.path.join(pd_, fn), blk)
        bdamps, cnds = _adaptive_damp(pd_, cnd_range=(40.0, 80.0), max_attempts=12, istart=2)
        med = float(np.median(list(cnds.values()))) if cnds else float("nan")
        return make_inp(isolv=2, damps=tuple(bdamps), istart=2), med, list(bdamps)
    finally:
        shutil.rmtree(pd_, ignore_errors=True)

def stage_boot(vol, nboot=200, seed=0, cores=48, force=False, boot_solver="svd", resample="within-pair"):
    from concurrent.futures import ThreadPoolExecutor
    d = os.path.join(BASE, vol); bdir = os.path.join(d, "bootstrap"); os.makedirs(bdir, exist_ok=True)
    out_csv, out_npz = os.path.join(bdir, "bootstrap_errors.csv"), os.path.join(bdir, "bootstrap_samples.npz")
    n = int(nboot)                                              # FIXED n for every cluster (no adaptive)
    resample_fn = _resample_global if resample == "global" else _resample_within_pair
    if not force and os.path.exists(out_csv):
        head = open(out_csv).readline()
        if f"n={n} " in head and f"seed={seed} " in head and f"resample={resample}" in head and f"bootsolver={boot_solver}" in head:
            print(f"[boot] {vol}: cached n={n} {resample} {boot_solver} bootstrap found"); return
    main = read_reloc(os.path.join(d, "hypoDD.reloc"))            # = reported LSQR-CND-on-whole-box solution
    main_xyz = {int(r.id): (float(r.x), float(r.y), float(r.z)) for r in main.itertuples()}
    seeded = _seed_event_dat(os.path.join(d, "event.dat"), main)   # replicas start from the reported solution
    base_blocks = {fn: _parse_dt_blocks(os.path.join(d, fn)) for fn in ("dt.ct", "dt.cc_0.7_combined")}
    meta = load_meta(vol); T = float(meta.get("main_run_s", 300.0)); boot_timeout = max(300, 3 * T)
    # Reported solution is LSQR-CND on the whole-box seed; bootstrap replicas re-run the SAME LSQR-CND (DAMP
    # calibrated to CND 40-80). ez95 here is the RELATIVE-precision scatter of that estimator; the absolute
    # centroid depth carries a separate ~km null-space uncertainty NOT captured by any bootstrap (see stage_primary).
    if boot_solver == "lsqr":
        bdamps = load_meta(vol).get("primary_damps")               # reuse the reported solution's per-set adaptive DAMP
        if bdamps:
            inp_boot = make_inp(isolv=2, damps=tuple(bdamps), istart=2); damp_cal = bdamps; cnd_cal = load_meta(vol).get("primary_cnd", float("nan"))
            print(f"[boot] {vol}: LSQR-CND bootstrap reusing primary per-set DAMP={bdamps} (CND~{cnd_cal}); ez95 = relative precision (absolute depth ~km, null-space)")
        else:
            inp_boot, cnd_cal, damp_cal = _calibrate_lsqr(d, seeded, base_blocks)
            print(f"[boot] {vol}: LSQR-CND bootstrap adaptive DAMP={damp_cal} (CND {cnd_cal:.0f}); ez95 = relative precision (absolute depth ~km, null-space)")
    else:
        inp_boot = open(os.path.join(d, "hypoDD.inp")).read(); damp_cal = None
    tmp_root = tempfile.mkdtemp(prefix=f"boot_{vol}_")

    def _one(i):
        rng = np.random.default_rng(seed + i)
        rd = tempfile.mkdtemp(prefix=f"r{i}_", dir=tmp_root)
        try:
            shutil.copyfile(os.path.join(d, "station.dat"), os.path.join(rd, "station.dat"))
            open(os.path.join(rd, "hypoDD.inp"), "w").write(inp_boot)
            open(os.path.join(rd, "event.dat"), "w").write(seeded)
            for fn, blk in base_blocks.items():
                _write_dt_blocks(os.path.join(rd, fn), resample_fn(blk, rng))
            try:
                rl = read_reloc(_exec_hypodd_once(rd, timeout=boot_timeout))
                return {int(r.id): (float(r.x), float(r.y), float(r.z)) for r in rl.itertuples()}
            except (_MaxData0Overflow, _MaxEve0Overflow):        # SVD-mode replicas only
                try:
                    open(os.path.join(rd, "hypoDD.inp"), "w").write(make_inp(isolv=2, damps=(80,)*7, istart=2))
                    rl = read_reloc(_exec_hypodd_once(rd, timeout=boot_timeout))
                    return {int(r.id): (float(r.x), float(r.y), float(r.z)) for r in rl.itertuples()}
                except Exception: return {}
            except Exception: return {}
        finally:
            shutil.rmtree(rd, ignore_errors=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=int(cores)) as ex:
        replicas = list(ex.map(_one, range(n)))
    shutil.rmtree(tmp_root, ignore_errors=True)
    nfail = sum(1 for r in replicas if not r)

    samples = {}; samp_rows = []
    for ri, rep in enumerate(replicas):
        common = [e for e in rep if e in main_xyz]
        if len(common) < 2: continue
        off = np.median([[rep[e][k] - main_xyz[e][k] for k in range(3)] for e in common], axis=0)  # median-align
        for e, p in rep.items():
            if e in main_xyz:
                q = (p[0]-off[0], p[1]-off[1], p[2]-off[2])
                samples.setdefault(e, []).append(q)
                samp_rows.append([ri, e, q[0], q[1], q[2]])     # REPLICA-INDEXED (plane/centroid bootstraps)

    def _hw(a): return (np.percentile(a, 97.5, axis=0) - np.percentile(a, 2.5, axis=0)) / 2.0
    rows = []
    for e in sorted(main_xyz):                                   # NO error/n_boot flagging — report every event
        s = np.asarray(samples.get(e, []), float); nb = len(s)
        row = dict(id=e, n_boot=nb, x_med=main_xyz[e][0], y_med=main_xyz[e][1], z_med=main_xyz[e][2],
                   ex95=np.nan, ey95=np.nan, ez95=np.nan)
        if nb >= 2:                                              # need >=2 for a percentile; report whatever we have
            med, hw = np.median(s, axis=0), _hw(s)
            row.update(x_med=med[0], y_med=med[1], z_med=med[2], ex95=hw[0], ey95=hw[1], ez95=hw[2])
        rows.append(row)
    out = pd.DataFrame(rows)
    with open(out_csv, "w") as f:
        f.write(f"# bootstrap_errors n={n} seed={seed} vol={vol} resample={resample} bootsolver={boot_solver} "
                f"ci=percentile2.5-97.5 align=median init=wholebox-lsqr-cnd\n")
        out.to_csv(f, index=False)
    np.savez(out_npz, data=np.asarray(samp_rows, float) if samp_rows else np.empty((0, 5)))
    json.dump(dict(n=n, seed=seed, failed=nfail, failed_frac=round(nfail/max(n,1), 3),
                   wall_s=round(time.time()-t0, 1), resample=resample, boot_solver=boot_solver, lsqr_damp=damp_cal,
                   n_boot_min=int(out.n_boot.min()), n_boot_median=int(out.n_boot.median()),
                   timeout_s=boot_timeout), open(os.path.join(bdir, "bootstrap_meta.json"), "w"), indent=1)
    print(f"[boot] {vol}: n={n} {resample} {boot_solver} done in {time.time()-t0:.0f}s | failed replicas {nfail} ({100*nfail/n:.0f}%) | "
          f"n_boot/event min/median {int(out.n_boot.min())}/{int(out.n_boot.median())} | "
          f"median ex95={out.ex95.median():.1f} ey95={out.ey95.median():.1f} ez95={out.ez95.median():.1f} m")

# ================================================================= stage F: analysis
def plane_geom(x, y, z):
    """Vendored shape-gated PCA plane fit (build_cluster_deepdive_nb.py). x,y,z in metres, z +down."""
    P = np.c_[x - np.mean(x), y - np.mean(y), z - np.mean(z)]
    if len(P) < 4: return dict(shape="n/a", strike=None, dip=None, L1=np.nan, L2=np.nan, L3=np.nan)
    S, Vt = np.linalg.svd(P, full_matrices=False)[1:]
    sd = S / np.sqrt(len(P) - 1); L1, L2, L3 = (2 * sd / 1000)
    if L2 >= 0.4 * L1 and L3 <= 0.35 * L2:
        n_ = Vt[2]; n_ = -n_ if n_[2] < 0 else n_
        dip = float(np.degrees(np.arccos(abs(n_[2]) / np.linalg.norm(n_))))
        strike = float((np.degrees(np.arctan2(n_[0], n_[1])) + 90) % 180); shape = "planar"
    elif L2 < 0.4 * L1:
        v = Vt[0]; strike = float(np.degrees(np.arctan2(v[0], v[1])) % 180); dip = None; shape = "linear"
    else:
        strike = dip = None; shape = "blob"
    return dict(shape=shape, strike=strike, dip=dip, L1=round(L1, 3), L2=round(L2, 3), L3=round(L3, 3))

def _branch(strikes, ref):
    s = np.asarray(strikes, float)
    s = np.where(s - ref > 90, s - 180, s); s = np.where(s - ref < -90, s + 180, s)
    return s

def stage_analyze(vol, force=False):
    d = os.path.join(BASE, vol); adir = os.path.join(d, "analysis"); os.makedirs(adir, exist_ok=True)
    outj = os.path.join(adir, "summary.json")
    if not force and os.path.exists(outj):
        print(f"[analyze] {vol}: cached summary.json found"); return
    ve = pd.read_csv(os.path.join(d, "volume_events.csv"))
    be = pd.read_csv(os.path.join(d, "bootstrap", "bootstrap_errors.csv"), comment="#")
    rel = read_reloc(os.path.join(d, "hypoDD.reloc")).merge(
        ve[["id","cat","nnd_cluster","suspect","magU","x_m","y_m","depth_m"]], on="id", how="left").merge(
        be[["id","n_boot","ex95","ey95","ez95"]], on="id", how="left")
    # whole-box (LSQR) vs reported per-volume (LSQR-CND) per-event displacement, rigid-centroid offset removed
    off = np.median(np.c_[rel.x - (rel.x_m - rel.x_m.median()), rel.y - (rel.y_m - rel.y_m.median()),
                          rel.z - (rel.depth_m - rel.depth_m.median())], axis=0)
    rel["wb_shift_m"] = np.sqrt((rel.x - (rel.x_m - rel.x_m.median()) - off[0])**2 +
                                (rel.y - (rel.y_m - rel.y_m.median()) - off[1])**2 +
                                (rel.z - (rel.depth_m - rel.depth_m.median()) - off[2])**2)
    # geometry DESCRIPTOR (not an error gate): fit the shape only on events still INSIDE the 1 km volume after
    # relocation — i.e. within IN_VOL_KM of the cluster median. This excludes events that RELOCATED OUT of the
    # box (the anticipated subsetting artefact, e.g. the 0-cc-link event that slid to 26 km) on a PHYSICAL
    # in-volume criterion, NOT a bootstrap-error threshold (per user: do not flag by error>=150 m).
    IN_VOL_KM = 0.75
    cx, cy, cz = rel.x.median(), rel.y.median(), rel.z.median()
    rel["in_volume"] = np.sqrt((rel.x-cx)**2 + (rel.y-cy)**2 + (rel.z-cz)**2) / 1000.0 <= IN_VOL_KM
    core = rel[rel.in_volume]                                     # in-volume events for the shape descriptor
    fit_main = plane_geom(core.x.values, core.y.values, core.z.values)
    z = np.load(os.path.join(d, "bootstrap", "bootstrap_samples.npz"))["data"]
    S = pd.DataFrame(z, columns=["ri","id","x","y","zz"]); S["id"] = S.id.astype(int)
    core_ids = set(core.id.astype(int)); prows = []
    for ri, gS in S.groupby("ri"):
        gg = gS[gS.id.isin(core_ids)]
        if len(gg) < 0.8 * len(core_ids): continue
        prows.append(dict(ri=int(ri), **plane_geom(gg.x.values, gg.y.values, gg.zz.values)))
    PB = pd.DataFrame(prows); PB.to_csv(os.path.join(adir, "plane_bootstrap.csv"), index=False)
    rel[["id","cat","nnd_cluster","magU","wb_shift_m","in_volume","n_boot","ex95","ey95","ez95"]].to_csv(
        os.path.join(adir, "event_quality.csv"), index=False)   # per-event: whole-box vs primary shift + boot error
    summ = dict(volume=vol, fit_main=fit_main, n_plane_replicas=len(PB),
                n_in_volume=int(rel.in_volume.sum()), n_relocated=len(rel), in_vol_km=IN_VOL_KM,
                relocated_out_ids=sorted(rel.loc[~rel.in_volume, "id"].astype(int)),
                big_shift_ids=sorted(rel.loc[rel.wb_shift_m > 300, "id"].astype(int)),   # whole-box vs primary > 300 m
                shape_votes=PB["shape"].value_counts().to_dict() if len(PB) else {})
    if len(PB):                                                  # guard each: tiny clusters give all-NaN replicas
        for q in ("L1","L2","L3"):
            a = PB[q].dropna().values
            if len(a): summ[f"{q}_ci95"] = [round(float(np.percentile(a, p)), 3) for p in (2.5, 97.5)]
        if fit_main["strike"] is not None and PB.strike.notna().sum():
            st = _branch(PB.strike.dropna().values, fit_main["strike"])
            summ["strike_ci95"] = [round(float(np.percentile(st, p)), 1) for p in (2.5, 97.5)]
            if fit_main["dip"] is not None and PB.dip.notna().sum() > 10:
                summ["dip_ci95"] = [round(float(np.percentile(PB.dip.dropna(), p)), 1) for p in (2.5, 97.5)]
    if vol == "m373":                                            # M3.73-sequence vs co-located-swarm separation
        wl = core_ids
        # the two family labels change when the catalog is re-relocated: use the mainshock's OWN family and the
        # OTHER most-populous NND family in the cube (not hard-coded 7 & 6).
        ms_fam = ve.loc[ve.magU.idxmax(), "nnd_cluster"]
        fam_counts = ve[ve.nnd_cluster.notna() & (ve.nnd_cluster != -1) & (ve.nnd_cluster != ms_fam)].nnd_cluster.value_counts()
        other_fam = fam_counts.index[0] if len(fam_counts) else None
        ids7 = set(ve.loc[ve.nnd_cluster == ms_fam, "id"].astype(int)) & wl
        ids6 = set(ve.loc[ve.nnd_cluster == other_fam, "id"].astype(int)) & wl if other_fam is not None else set()
        rows = []
        for ri, gS in S.groupby("ri"):
            a, b = gS[gS.id.isin(ids7)], gS[gS.id.isin(ids6)]
            if len(a) < 3 or len(b) < 3: continue
            ca, cb = a[["x","y","zz"]].mean().values, b[["x","y","zz"]].mean().values
            wa = np.median(np.linalg.norm(a[["x","y","zz"]].values - ca, axis=1))
            wb = np.median(np.linalg.norm(b[["x","y","zz"]].values - cb, axis=1))
            rows.append(dict(ri=int(ri), sep3d=float(np.linalg.norm(ca - cb)),
                             seph=float(np.hypot(*(ca - cb)[:2])), width7=wa, width6=wb))
        SEP = pd.DataFrame(rows, columns=["ri","sep3d","seph","width7","width6"])   # keep header even if empty
        SEP.to_csv(os.path.join(adir, "separation_bootstrap.csv"), index=False)
        if len(SEP):
            lo, med, hi = (float(np.percentile(SEP.sep3d, p)) for p in (2.5, 50, 97.5))
            wmax = float(max(SEP.width7.median(), SEP.width6.median()))
            summ["separation_m"] = dict(lo=round(lo,1), med=round(med,1), hi=round(hi,1),
                                        max_group_width_m=round(wmax,1),
                                        resolved=bool(lo > wmax / 2), n_replicas=len(SEP))
    json.dump(summ, open(outj, "w"), indent=1, default=str)
    print(f"[analyze] {vol}: shape={fit_main['shape']} strike={fit_main['strike']} dip={fit_main['dip']} "
          f"L={fit_main['L1']}x{fit_main['L2']}x{fit_main['L3']} km | plane replicas {len(PB)}"
          + (f" | sep {summ.get('separation_m')}" if vol == "m373" else ""))

# ================================================================= main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", default="all", choices=["m389","m373","all"])
    ap.add_argument("--stage", default="all", choices=["select","extract","run","primary","boot","analyze","all"])
    ap.add_argument("--nboot", type=int, default=200)           # FIXED n for every cluster
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed-from-whole-box", action="store_true")
    ap.add_argument("--boot-solver", default="lsqr", choices=["svd","lsqr"])   # lsqr matches the reported LSQR-CND primary (relative precision); svd bootstrap also captures null-space depth wander
    ap.add_argument("--boot-resample", default="global", choices=["within-pair","global"])  # global = larger, more complete uncertainty
    a = ap.parse_args()
    vols = ["m389","m373"] if a.volume == "all" else [a.volume]
    os.makedirs(BASE, exist_ok=True)
    if a.stage in ("select","all"): stage_select(force=a.force)
    for v in vols:
        if a.stage in ("extract","all"): stage_extract(v, force=a.force)
        if a.stage in ("extract","run","all"): stage_inp(v, a.seed_from_whole_box, force=a.force)
        if a.stage in ("run","all"): stage_run(v, force=a.force)
        if a.stage in ("primary","run","all"): stage_primary(v, force=a.force)
        if a.stage in ("boot","all"): stage_boot(v, int(a.nboot), a.seed, a.cores, force=a.force, boot_solver=a.boot_solver, resample=a.boot_resample)
        if a.stage in ("analyze","all"): stage_analyze(v, force=a.force)
    print("DONE")
