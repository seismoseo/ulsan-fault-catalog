#!/usr/bin/env python
"""Companion to run_svd_volumes.py / run_svd_volumes_next8.py — per-volume LSQR-CND relocation + global
bootstrap for the 8 largest clusters BY MEMBER COUNT (the families analysed in nb32; NND labels 0-7 under
descending-size labeling). Ranks by member count instead of mainshock magnitude — four of the eight (families
0,1,6,7) coincide with the by-magnitude set already relocated elsewhere, but every volume is relocated here
independently so nb35 reads one self-contained tree.

Same methodology as the top-2 / next8 runners: SVD kept only as the depth-drift diagnostic (hypoDD.reloc.svd);
reported solver = LSQR-CND seeded on the whole-box LSQR locations (absolute centroid depth is a DD null space);
bootstrap = global resampling, LSQR-CND, n=200; ez95 = relative precision (absolute depth ~km, uncaptured).
Selection UNIONS the 1 km context cube with the FULL NND family so no grouped member is omitted, and flags
ct-only events suspect (kept & relocated, excluded from the plane/shape fit) — mirrors run_svd_volumes.py.

Reuses every downstream stage from run_svd_volumes.py by pointing rsv.BASE at svd_volumes_top8count/.

Usage:  python run_svd_volumes_top8count.py [--stage select|extract|run|primary|boot|analyze|all] [--cores 48] [--force]
"""
import os, glob, json, argparse
import numpy as np, pandas as pd
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("run_svd_volumes", os.path.join(_HERE, "run_svd_volumes.py"))
rsv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rsv)
nnd, clustering = rsv.nnd, rsv.clustering

# point every reused stage at the top8count output tree
BASE = f"{rsv.KG}/uf_subregion_hypodd/svd_volumes_top8count"
rsv.BASE = BASE
os.makedirs(BASE, exist_ok=True)

def stage_select8count(force=False):
    """Vendored from rsv.stage_select but selecting the 8 largest families by MEMBER COUNT (nb32), naming
    volumes c<k> by the CURRENT NND family labels. No hard gate. Writes BASE/volumes.txt."""
    if not force and os.path.exists(os.path.join(BASE, "volumes.txt")):
        print("[select] cached volumes.txt found"); return
    rl = pd.read_csv(rsv.RELOC); rl["event_time"] = pd.to_datetime(rl.event_time, format="ISO8601", utc=True, errors="coerce")
    rl = rl.dropna(subset=["lat","lon","depth","event_time","ml_ufcorr_reloc"]).copy()
    g = rl[rl.n_used >= 3].copy()
    g["event_id"] = g.event_idx.astype(int).astype(str)
    g["t_year"] = g.event_time.dt.strftime("%Y%m%d%H%M%S").map(nnd.decimal_year)     # CANONICAL
    g = g.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep","ml_ufcorr_reloc":"kma_mag"})
    g = g.sort_values("t_year").reset_index(drop=True)
    nd = nnd.compute_nnd(g, b=rsv.B_NND, D=rsv.DF_UF, mmin=None, metric="3d")
    e0, _ = nnd.fit_eta0(nd.eta.values)
    lab = nnd.build_families(nd, e0, g.event_id.values, link_rmax_km=rsv.LINKR)
    g["Cluster"] = g.event_id.map(lab).fillna(-1).astype(int)
    CLU = dict(zip(g.event_idx.astype(int), g.Cluster.astype(int)))
    famsize = g[g.Cluster >= 0].groupby("Cluster").size().sort_values(ascending=False)
    targets = list(famsize.head(8).index)                      # 8 largest by MEMBER COUNT (nb32)

    # full relocated catalog (id -> ts -> event_idx map; identical to nb30/nb31/nb32)
    r0 = rsv.read_reloc(os.path.join(rsv.RUN03, "hypoDD.reloc")); r0["ncc"] = r0.nccp + r0.nccs
    dirs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(rsv.WF100, "20*")))
    id2ts = {200000 + i: ts for i, ts in enumerate(dirs)}
    mei = pd.read_csv(rsv.MEIDX).sort_values("event_idx")
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
    l0 = pd.read_csv(os.path.join(rsv.RUN03, "hypoDD.loc"), sep=r"\s+", header=None,
                     names=["id","lat","lon","depth","x","y","z","ex","ey","ez","yr","mo","dy","hr","mi","sc","mag","cid"])[["id","lat","lon","depth"]]
    lu, _ = clustering.to_utm(l0.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep"}))
    l0["x0"], l0["y0"], l0["z0"] = lu.x_m.values, lu.y_m.values, lu.depth_m.values
    allc = allc.merge(l0[["id","x0","y0","z0"]], on="id", how="left")
    allc["shift_km"] = np.sqrt((allc.x_m - allc.x0)**2 + (allc.y_m - allc.y0)**2 + (allc.depth_m - allc.z0)**2) / 1000.0
    allc["nlinks"] = allc.nccp + allc.nccs + allc.nctp + allc.ncts
    # ct-only events (0 surviving cc obs) are located by catalog dt only — kept & relocated, but flagged so they
    # do NOT drive the high-precision plane/shape fit (mirrors run_svd_volumes.py).
    allc["suspect"] = (allc.shift_km > 3.0) | (allc.nlinks < 10) | (allc.ncc == 0)

    HW = rsv.HALF * 1000.0
    selected = []
    for k in targets:
        vol = f"c{int(k)}"
        memset = set(g.loc[g.Cluster == k, "event_idx"].astype(int))
        mem = allc[allc.event_idx.isin(memset)]
        cE, cN, cZ = float(mem.x_m.mean()), float(mem.y_m.mean()), float(mem.depth_m.mean())
        ms = mem.loc[mem.magU.idxmax()]
        # UNION the 1 km context cube with the FULL NND family so no grouped member is omitted (mirrors m389 fix).
        in_cube = (allc.x_m.between(cE-HW, cE+HW)) & (allc.y_m.between(cN-HW, cN+HW)) & (allc.depth_m.between(cZ-HW, cZ+HW))
        box = allc[in_cube | allc.event_idx.isin(memset)].copy()
        box["is_member"] = box.event_idx.isin(memset)
        box["nnd_cluster"] = box.event_idx.map(CLU)
        box["cat"] = np.where(box.is_member, "member",
                     np.where(box.nnd_cluster.isna(), "not-in-pop",
                     np.where(box.nnd_cluster == -1, "background", "other-family")))
        box["e_km"] = (box.x_m - cE)/1000; box["n_km"] = (box.y_m - cN)/1000; box["z_km"] = (box.depth_m - cZ)/1000
        box = box.sort_values("time").reset_index(drop=True)
        d = os.path.join(BASE, vol); os.makedirs(d, exist_ok=True)
        cols = ["id","event_idx","cat","nnd_cluster","is_member","has_ml","suspect","lat","lon","depth",
                "x_m","y_m","depth_m","e_km","n_km","z_km","magU","time","shift_km","nlinks","ncc"]
        box[cols].to_csv(os.path.join(d, "volume_events.csv"), index=False)
        rsv.save_meta(vol, dict(volume=vol, family=int(k), n_in_cube=len(box), rank_by="count", famsize=int(famsize[k]),
                                center_utm=[cE, cN, cZ], mainshock_id=int(ms.id),
                                mainshock_mag=float(ms.magU), mainshock_time=str(ms.time),
                                cat_counts=box.cat.value_counts().to_dict(),
                                n_suspect=int(box.suspect.sum())))
        selected.append((vol, float(ms.magU), int(famsize[k])))
        print(f"[select] {vol}: {len(box)} events (family {int(k)}, {int(famsize[k])} members, mainshock M{ms.magU:.2f} id {int(ms.id)} {str(ms.time)[:10]})")
    with open(os.path.join(BASE, "volumes.txt"), "w") as f:      # canonical volume list (rank-by-count order) for nb35
        for v, mg, sz in selected: f.write(f"{v} {mg:.2f} {sz}\n")

def run_vol(vol, cores, force):
    """Full downstream pipeline for one volume, reusing rsv stages (BASE already redirected)."""
    try:
        rsv.stage_extract(vol, force=force)
        rsv.stage_inp(vol, seed_from_whole_box=True, force=force)
        rsv.stage_run(vol, force=force)                          # SVD diagnostic -> hypoDD.reloc(.svd)
        rsv.stage_primary(vol, force=force)                     # reported LSQR-CND on whole-box seed
        rsv.stage_boot(vol, nboot=200, seed=0, cores=cores, force=force, boot_solver="lsqr", resample="global")
        rsv.stage_analyze(vol, force=force)
        return True
    except Exception as e:
        print(f"[FAIL] {vol}: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["select","extract","run","primary","boot","analyze","all"])
    ap.add_argument("--volume", default="all")
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.stage in ("select", "all"): stage_select8count(force=a.force)
    vtxt = os.path.join(BASE, "volumes.txt")                    # current volume list (from stage_select8count)
    all_vols = [ln.split()[0] for ln in open(vtxt)] if os.path.exists(vtxt) else []
    vols = all_vols if a.volume == "all" else [a.volume]
    for v in vols:
        if a.stage == "select": break
        if a.stage == "all":
            run_vol(v, a.cores, a.force)
        else:
            fn = {"extract": lambda: rsv.stage_extract(v, force=a.force),
                  "run": lambda: (rsv.stage_inp(v, seed_from_whole_box=True, force=a.force), rsv.stage_run(v, force=a.force), rsv.stage_primary(v, force=a.force)),
                  "primary": lambda: rsv.stage_primary(v, force=a.force),
                  "boot": lambda: rsv.stage_boot(v, nboot=200, seed=0, cores=a.cores, force=a.force, boot_solver="lsqr", resample="global"),
                  "analyze": lambda: rsv.stage_analyze(v, force=a.force)}[a.stage]
            fn()
    print("DONE")
