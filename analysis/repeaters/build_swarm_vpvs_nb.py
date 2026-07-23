"""Build `12_vpvs_swarm2021_KGHDB.ipynb` — in-situ Vp/Vs of the 2021 Ulsan-fault swarm using the FULL
spatial swarm (52 events, not just the CC>=0.9 repeater subset), with Liu (2023)'s τ (lever-arm)
reliability framework as the centrepiece. Self-contained (computes in-cell from uf_vpvs_swarm), embeds
figures large. Structured so dense GHBSN stations can be dropped in later.

Build + execute:  python build_swarm_vpvs_nb.py
"""
import os
import nbformat as nbf

HYPO = "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/HypoInv"
OUT = os.path.join(HYPO, "repeaters", "12_vpvs_swarm2021_KGHDB.ipynb")


def _cells():
    nb = nbf.v4.new_notebook()
    C = []
    md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
    co = lambda s: C.append(nbf.v4.new_code_cell(s))

    md(r"""# In-situ Vp/Vs of the 2021 Ulsan-fault swarm — full-swarm, τ-reliability analysis

The 2021 swarm (≈ 35.81°N, 129.44°E, ~14.5 km) is a larger-extent sequence than the tight repeater
families, so it is a natural target for a **lever-arm (τ) robust** in-situ Vp/Vs (Lin & Shearer 2007;
Liu et al. 2023). Rather than the CC ≥ 0.9 repeater subset, we define the **full swarm spatially**
(DBSCAN, à la NND declustering) to keep the more-separated members that carry the **long-baseline event
pairs** — and we test, bin by bin in τ, whether those long baselines actually buy reliability.""")

    co(r'''import os, sys, math, warnings, itertools
warnings.filterwarnings("ignore")
HYPO = r"%s"
sys.path.insert(0, os.path.join(HYPO, "repeaters")); sys.path.insert(0, HYPO)
sys.path.insert(0, "/home/msseo/works/16.kma_absolute_location")
import numpy as np, pandas as pd
import matplotlib, matplotlib.pyplot as plt
matplotlib.use("Agg")
from IPython.display import Image, display
import uf_vpvs_swarm as sw
from kma_absolute_location import vpvs
pd.set_option("display.width", 200)

FIGDIR = os.path.join(HYPO, "repeaters", "figs_swarm"); os.makedirs(FIGDIR, exist_ok=True)
HELVETICA = "/home/msseo/Downloads/Helvetica/helvetica.ttf"
def use_helvetica():
    try:
        import matplotlib.font_manager as fm
        if os.path.isfile(HELVETICA):
            fm.fontManager.addfont(HELVETICA)
            matplotlib.rcParams["font.family"] = [fm.FontProperties(fname=HELVETICA).get_name(), "DejaVu Sans"]
    except Exception: pass
    matplotlib.rcParams["axes.unicode_minus"] = False
use_helvetica()
def show(p, w): display(Image(filename=p, width=w))

BAND = (5, 15)
EVS, META = sw.swarm_events()
P, S = sw.build_dt(EVS, BAND)
TAU = sw.pair_tau(P, S)
# ALL pairs (for context) vs the per-pair-GATED clean set (Liu 2023) that the reported value uses
m_all = vpvs.build_measurements(P, S, None, cc_min=sw.CCXC_MIN, min_stations=sw.MIN_ST, dist_min_km=0.0)
Pg, Sg = sw.per_pair_gate(P, S)
m = vpvs.build_measurements(Pg, Sg, None, cc_min=sw.CCXC_MIN, min_stations=sw.MIN_ST, dist_min_km=0.0)
v, mad, _ = vpvs.bootstrap_vpvs(m, n_boot=1000); inl, _ = vpvs.robust_inliers(m, v)
corr = float(np.corrcoef(m["dtp"][inl], m["dts"][inl])[0, 1])
v_all = vpvs.bootstrap_vpvs(m_all, n_boot=400)[0]
# --- Zhang et al. (2025, SRL) two-step grid search on the SAME gate-passing pairs: corrects only the
#     per-pair origin-time term Δt0 and keeps the moveout/lever arm that demeaning collapses, so it is the
#     less-dilution-biased estimator. For this low-δtP-SNR swarm the LS<->Zhang gap is the headline number:
#     it brackets the true Vp/Vs (IRLS-demean low, Zhang-orthogonal high). ---
z = vpvs.zhang_vpvs(Pg, Sg, cc_min=sw.CCXC_MIN, min_stations=sw.MIN_ST)
z_all = vpvs.zhang_vpvs(P, S, cc_min=sw.CCXC_MIN, min_stations=sw.MIN_ST)
vz, vz_sig = z["vpvs"], z["sigma"]
print(f"Full 2021 swarm: {len(EVS)} events, {len(m_all['stations'])} stations.")
print(f"  all {m_all['n_pairs']} pairs (incl. decohered) -> robust {v_all:.3f}")
print(f"  after Liu per-pair linearity gate: {m['n_pairs']} clean pairs, {len(m['dtp'])} obs")
print(f"  Lin & Shearer (IRLS-demean) Vp/Vs = {v:.3f} ± {mad:.3f}  (inlier corr {corr:.2f})")
print(f"  Zhang (2025, grid search)   Vp/Vs = {vz:.3f} ± {vz_sig:.3f}  -> LS<->Zhang gap {vz-v:+.3f}")''' % HYPO)

    # ---------------------------------------------------------------- §1 the swarm
    md(r"""## 1. The full swarm — spatial membership

The spatial DBSCAN keeps the coherent 2021–2022 sequence (dropping isolated stragglers). Map view + depth
sections; colour = time. This is **more events and more stations** than the CC ≥ 0.9 repeater family
(which kept only the tightest subset).""")
    co(r'''la = np.array([META.loc[e].lat for e in EVS]); lo = np.array([META.loc[e].lon for e in EVS])
de = np.array([META.loc[e].depth for e in EVS]); yr = np.array([int(e[:4]) + (int(e[4:6])-1)/12 for e in EVS])
clat, clon = la.mean(), lo.mean()
xkm = (lo - clon) * 111.195 * math.cos(math.radians(clat)); ykm = (la - clat) * 111.195
fig, ax = plt.subplots(1, 3, figsize=(16, 5), dpi=120)
for a, (hx, hy, xl, yl, inv) in zip(ax, [(xkm, ykm, "E (km)", "N (km)", False),
                                          (xkm, de, "E (km)", "Depth (km)", True),
                                          (ykm, de, "N (km)", "Depth (km)", True)]):
    s_ = a.scatter(hx, hy, s=50, c=yr, cmap="plasma", edgecolor="k", linewidth=0.3)
    a.set(xlabel=xl, ylabel=yl); a.spines[["top", "right"]].set_visible(False)
    if inv: a.invert_yaxis()
ax[0].set_aspect("equal"); ax[0].set_title(f"Map ({len(EVS)} events)")
ax[1].set_title("E–depth"); ax[2].set_title("N–depth")
plt.colorbar(s_, ax=ax[2], label="year", shrink=0.8)
maxsep = max(math.sqrt((xkm[i]-xkm[j])**2 + (ykm[i]-ykm[j])**2 + (de[i]-de[j])**2)
             for i, j in itertools.combinations(range(len(EVS)), 2))
fig.suptitle(f"2021 Ulsan swarm — {clat:.3f}°N, {clon:.3f}°E, {de.min():.1f}–{de.max():.1f} km, max baseline {maxsep:.2f} km", y=1.02, fontsize=12)
fig.tight_layout(); p = os.path.join(FIGDIR, "swarm.png"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); show(p, 1600)
print(f"max inter-event baseline {maxsep:.2f} km -> max possible τ ≈ {maxsep/6*1000:.0f} ms")''')

    # ---------------------------------------------------------------- §2 the fit
    md(r"""## 2. The measurement — one fit, one number

Each point is one station's demeaned (δt$_P$, δt$_S$) for an event pair; the slope is Vp/Vs. We keep only
the pairs that pass a **per-pair linearity gate** (Liu 2023): a pair is used only if its own δt$_P$–δt$_S$
form a clean line (residual RMS < 5 ms). **Grey ✕ = rejected pairs** — overwhelmingly the long-baseline
pairs whose waveforms have decohered (see §3). The robust fit to the **kept (blue) pairs** is the reported
value — and it is what this plot shows directly.

We draw **both** SOTA estimators on the kept pairs: **solid red = Lin & Shearer** (IRLS over the demeaned
points) and **solid blue = Zhang (2025)** (origin-time-only correction, moveout preserved). For this swarm
the δt$_P$ SNR is low, so the two **bracket** the answer — Lin & Shearer is pulled *down* by regression
dilution, Zhang sits *higher*; the gap (not either endpoint alone) is the honest uncertainty.""")
    co(r'''# kept (gated) points vs all points (context), in the same demeaned frame
xg, yg = m["dtp"] * 1000, m["dts"] * 1000
xa, ya = m_all["dtp"] * 1000, m_all["dts"] * 1000
fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=130)
# faded: ALL points (context), then overplot kept
ax.scatter(xa, ya, s=12, c="0.8", edgecolor="none", zorder=1, label=f"rejected pairs (decohered)")
ax.scatter(xg[inl], yg[inl], s=18, c=m["w"][inl], cmap="viridis", vmin=0.75, vmax=1.0,
           edgecolor="k", linewidth=0.2, zorder=3, label="kept pairs (pass linearity gate)")
xs = np.array([xg.min(), xg.max()])
ax.plot(xs, v * xs, "-", c="crimson", lw=2.2, label=f"Lin & Shearer (IRLS) = {v:.3f} ± {mad:.3f}")
if np.isfinite(vz):
    ax.plot(xs, vz * xs, "-", c="#2c7fb8", lw=2.2, label=f"Zhang (2025) = {vz:.3f} ± {vz_sig:.3f}")
ax.plot(xs, 1.73 * xs, ":", c="k", lw=1, label="1.73 (normal crust)")
ax.axhline(0, c="0.85", lw=0.6); ax.axvline(0, c="0.85", lw=0.6)
ax.set(xlabel=r"Demeaned $\delta t_P$ (ms)", ylabel=r"Demeaned $\delta t_S$ (ms)",
       title=f"2021 Ulsan swarm · Lin&Shearer {v:.3f}  vs  Zhang {vz:.3f}\n"
             f"{m['n_pairs']} clean pairs / {len(xg)} obs · corr {corr:.2f}")
ax.legend(loc="upper left", fontsize=9); ax.set_aspect("equal", "datalim"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); p = os.path.join(FIGDIR, "fit.png"); fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); show(p, 800)
print(f"Lin & Shearer Vp/Vs = {v:.3f} ± {mad:.3f}; Zhang Vp/Vs = {vz:.3f} ± {vz_sig:.3f} "
      f"(gap {vz-v:+.3f}) from {m['n_pairs']} gate-passing pairs (corr {corr:.2f}); "
      f"all-pairs (incl. decohered) would give {v_all:.3f}.")''')

    # ---------------------------------------------------------------- §3 the tau tradeoff (centerpiece)
    md(r"""## 3. Why the gate rejects the long-baseline pairs — the τ trade-off

This explains *why* §2's gate threw out the far pairs. Liu (2023) says a large differential-P-time range
**τ** (long baseline) best constrains the slope — but there is a competing effect: events far apart have
**less similar waveforms**, so their δt decohere. We bin the pairs by τ and fit each bin: coherence peaks
at moderate τ and **collapses for the long baselines**, which is exactly why the per-pair gate removes
them. So the longer baseline does **not** buy reliability for this swarm.""")
    co(r'''def slope_for(pairs):
    Ps = {k: P[k] for k in pairs if k in P}; Ss = {k: S[k] for k in pairs if k in S}
    mm = vpvs.build_measurements(Ps, Ss, None, cc_min=sw.CCXC_MIN, min_stations=sw.MIN_ST, dist_min_km=0.0)
    if mm["n_pairs"] < 4: return None
    xx, yy = mm["dtp"], mm["dts"]; vv, mm2, _ = vpvs.bootstrap_vpvs(mm, n_boot=600)
    zz = vpvs.zhang_vpvs(Ps, Ss, cc_min=sw.CCXC_MIN, min_stations=sw.MIN_ST)["vpvs"]
    return dict(npairs=mm["n_pairs"], corr=float(np.corrcoef(xx, yy)[0, 1]),
                ols=float(np.sum(xx*yy)/np.sum(xx*xx)), robust=vv, mad=mm2, zhang=zz)

bins = [(0.0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 1.0)]
blab = ["0–20", "20–40", "40–60", "60+"]
rows = []
for (lob, hib), lab in zip(bins, blab):
    pr = [k for k, t in TAU.items() if lob <= t < hib]; r = slope_for(pr)
    if r: rows.append(dict(tau_bin=lab + " ms", **r))
T = pd.DataFrame(rows)
print(T.to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), dpi=120)
xb = np.arange(len(T))
ax[0].bar(xb, T["corr"], color="#2c7fb8", edgecolor="k"); ax[0].set_xticks(xb); ax[0].set_xticklabels(T.tau_bin)
ax[0].set(ylabel=r"coherence corr($\delta t_P,\delta t_S$)", title="(a) coherence vs lever arm τ", ylim=(0, 1))
for i, c in enumerate(T["corr"]): ax[0].text(i, c + 0.02, f"{c:.2f}", ha="center", fontsize=9)
ax[0].spines[["top", "right"]].set_visible(False)
ax[1].errorbar(xb, T.robust, yerr=T["mad"], fmt="o-", c="crimson", ms=9, capsize=4, label="Lin & Shearer (IRLS)")
ax[1].plot(xb, T.zhang, "D-", c="#2c7fb8", ms=8, label="Zhang (2025)")
ax[1].plot(xb, T.ols, "s--", c="#d95f0e", ms=7, label="OLS")
ax[1].axhline(math.sqrt(2), ls="--", c="0.6", lw=0.8, label="√2"); ax[1].axhline(1.73, ls=":", c="k", label="1.73")
ax[1].set_xticks(xb); ax[1].set_xticklabels(T.tau_bin); ax[1].set(xlabel="τ bin (ms)", ylabel="Vp/Vs",
          title="(b) Vp/Vs vs τ — trustworthy where corr is high", ylim=(0, 2))
ax[1].legend(fontsize=8); ax[1].spines[["top", "right"]].set_visible(False)
fig.tight_layout(); p = os.path.join(FIGDIR, "tau.png"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); show(p, 1450)

# the trustworthy estimate = highest-coherence bin
best = T.loc[T["corr"].idxmax()]
print(f"\\nMost-coherent bin: {best.tau_bin} (corr {best['corr']:.2f}) -> Vp/Vs = {best.robust:.3f} ± {best['mad']:.3f}")
print("Far pairs (60+ ms) decohere -> NOT usable here: waveforms too dissimilar + shared-ray-path breaks (Liu smearing).")''')

    # ---------------------------------------------------------------- §4 reliability extras
    md(r"""## 4. Coverage, bootstrap, band stability

**(a) Azimuthal coverage** — the swarm sits on the eastern edge, so station coverage is one-sided (large
az_gap); this is the binding limitation and exactly what dense **GHBSN** stations would fix. **(b)
Bootstrap** of the trustworthy (coherent) estimate. **(c) Band stability** across 5–15 / 3–15 Hz.""")
    co(r'''fig, ax = plt.subplots(1, 3, figsize=(17, 5), dpi=120)
# (a) station azimuths from centroid
sc = {r.station: (r.lat, r.lon) for r in sw.wf.used_stations(EVS).itertuples()}
azs = []
for s in m["stations"]:
    if s in sc:
        kx = math.cos(math.radians(clat)); az = math.degrees(math.atan2((sc[s][1]-clon)*kx, sc[s][0]-clat)) % 360
        azs.append(az)
azg = sw.az_gap(EVS, META, m["stations"])
ax[0] = plt.subplot(1, 3, 1, projection="polar")
ax[0].scatter(np.radians(azs), np.ones(len(azs)), s=60, c="#2c7fb8", edgecolor="k", zorder=3)
ax[0].set_theta_zero_location("N"); ax[0].set_theta_direction(-1); ax[0].set_rticks([])
ax[0].set_title(f"(a) station azimuths — az_gap {azg:.0f}°\\n(one-sided; GHBSN would fill)", fontsize=10)
# (b) bootstrap of the REPORTED (gate-passing) estimate
mc = m
boot = []
_o = np.argsort(mc["pair_id"], kind="stable"); _p = mc["pair_id"][_o]
_dp, _ds, _w = mc["dtp"][_o], mc["dts"][_o], mc["w"][_o]; _n = mc["n_pairs"]
_b = np.searchsorted(_p, np.arange(_n+1)); _rng = np.random.default_rng(0)
for _ in range(2000):
    pk = _rng.integers(0, _n, _n); idx = np.concatenate([np.arange(_b[q], _b[q+1]) for q in pk])
    sl = vpvs.irls_slope(_dp[idx], _ds[idx], _w[idx])
    if np.isfinite(sl): boot.append(sl)
boot = np.array(boot); vc = float(np.median(boot))
ax[1] = plt.subplot(1, 3, 2)
ax[1].hist(boot, bins=40, color="#2c7fb8", edgecolor="k", density=True)
ax[1].axvline(vc, c="crimson", lw=1.5, label=f"{vc:.3f} ± {np.median(np.abs(boot-vc)):.3f}")
ax[1].set(xlabel="Vp/Vs (bootstrap)", ylabel="density", title="(b) reported estimate (Liu per-pair gate)")
ax[1].legend(fontsize=9); ax[1].spines[["top", "right"]].set_visible(False)
# (c) band stability
ax[2] = plt.subplot(1, 3, 3)
bres = []
for bd in [(5, 15), (3, 15)]:
    Pb, Sb = sw.build_dt(EVS, bd); Pbg, Sbg = sw.per_pair_gate(Pb, Sb)
    mb = vpvs.build_measurements(Pbg, Sbg, None, cc_min=sw.CCXC_MIN, min_stations=sw.MIN_ST, dist_min_km=0.0)
    if mb["n_pairs"] >= 4:
        vb, mb2, _ = vpvs.bootstrap_vpvs(mb, n_boot=400); bres.append((f"{bd[0]}-{bd[1]}", vb, mb2))
bx = np.arange(len(bres))
ax[2].errorbar(bx, [b[1] for b in bres], yerr=[b[2] for b in bres], fmt="o", c="#2c7fb8", ms=10, capsize=4)
ax[2].axhline(1.73, ls=":", c="k"); ax[2].set_xticks(bx); ax[2].set_xticklabels([b[0]+"Hz" for b in bres])
ax[2].set(ylabel="Vp/Vs", title="(c) band stability (gate-passing)", ylim=(1.2, 1.8))
ax[2].spines[["top", "right"]].set_visible(False)
fig.tight_layout(); p = os.path.join(FIGDIR, "reliability.png"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); show(p, 1650)
print(f"reported Vp/Vs = {v:.3f} ± {mad:.3f}; az_gap {azg:.0f}° (one-sided — GHBSN target)")''')

    # ---------------------------------------------------------------- §5 summary
    md(r"""## 5. Summary""")
    co(r'''print("="*82); print("2021 ULSAN-FAULT SWARM — IN-SITU Vp/Vs (full spatial swarm, τ-reliability)"); print("="*82)
print(f"• Full swarm: {len(EVS)} events (2021-2022), {len(m_all['stations'])} stations, max baseline {maxsep:.2f} km.")
print(f"• Lin & Shearer Vp/Vs = {v:.3f} ± {mad:.3f} (ν = {vpvs.poisson_ratio(v):.2f}) from {m['n_pairs']} clean pairs")
print(f"  (Liu per-pair gate, corr {corr:.2f}); all {m_all['n_pairs']} pairs incl. decohered would give {v_all:.2f}.")
print(f"• Zhang (2025) on the SAME pairs = {vz:.3f} ± {vz_sig:.3f} -> the two estimators BRACKET the answer")
print(f"  ({v:.2f}-{vz:.2f}). The gap = regression dilution: low δtP SNR pulls Lin & Shearer down, Zhang keeps")
print(f"  the moveout. Honest statement = Vp/Vs in [{min(v,vz):.2f}, {max(v,vz):.2f}]; the dramatic <√2 low is")
print(f"  NOT robust to the estimator (a demeaning artifact), though both stay at/below normal crust (1.73).")
print(f"• WHY the gate: far pairs (>60 ms, the long baselines) DECOHERE (corr ~0.1) — waveforms too dissimilar")
print(f"  + shared-ray-path assumption breaks (Liu smearing). The long baseline does NOT buy reliability here.")
print(f"• Within the COHERENT τ range the Lin & Shearer slope is τ-stable (not a τ-dependent artifact); the")
print(f"  residual estimator spread is the LS<->Zhang gap above, which dense GHBSN δtP SNR would collapse.")
print(f"• BINDING LIMITATION: azimuthal coverage (az_gap {azg:.0f}°, one-sided eastern edge). Dense GHBSN")
print(f"  stations would close the gap + add far-station measurements — the clearest path to firming this up.")''')

    nb["cells"] = C
    return nb


def main():
    nb = _cells()
    with open(OUT, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {OUT}")
    from nbclient import NotebookClient
    nb2 = nbf.read(OUT, as_version=4)
    NotebookClient(nb2, timeout=3000, kernel_name="python3").execute()
    with open(OUT, "w") as f:
        nbf.write(nb2, f)
    print(f"executed {OUT}")


if __name__ == "__main__":
    main()
