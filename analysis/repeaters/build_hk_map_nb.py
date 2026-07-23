"""Build `13_vpvs_Hk_vs_insitu_map.ipynb` — map comparison of the GHBSN H-κ receiver-function Vp/Vs
(crustal average, densely sampled in space; interp_k.txt) against the clusterwise IN-SITU Vp/Vs of the
Ulsan-fault repeater families. Two panels: in-situ by SOTA Lin & Shearer vs by SOTA Zhang, both on the same
H-κ colour background — because the bulk-vs-local comparison depends on which in-situ estimator is used.

Build + execute:  python build_hk_map_nb.py
"""
import os
import nbformat as nbf

HYPO = "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/HypoInv"
OUT = os.path.join(HYPO, "repeaters", "13_vpvs_Hk_vs_insitu_map.ipynb")


def _cells():
    nb = nbf.v4.new_notebook()
    C = []
    md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
    co = lambda s: C.append(nbf.v4.new_code_cell(s))

    md(r"""# GHBSN H-κ (bulk crust) vs clusterwise in-situ Vp/Vs — Ulsan fault

Two very different Vp/Vs measurements in the same region:

- **H-κ receiver functions (GHBSN, `interp_k.txt`):** the **bulk-crustal average** Vp/Vs beneath each
  station (whole 0→Moho column), densely interpolated in space — the coloured **background field**.
- **In-situ (Lin & Shearer 2007):** the **local source-volume** Vp/Vs of each repeater family (tens of m
  at ~10–15 km) — the **circles**.

Because the in-situ slope is **estimator-dependent** for these tight clusters (Lin & Shearer demeaning is
dilution-biased low; Zhang's grid search is higher — see notebook 36), we show **both** in-situ estimators
on the same H-κ background, so the bulk-vs-local comparison is honest.""")

    co(r'''import os, sys, warnings, math
warnings.filterwarnings("ignore")
HYPO = r"%s"
sys.path.insert(0, os.path.join(HYPO, "repeaters")); sys.path.insert(0, HYPO)
sys.path.insert(0, "/home/msseo/works/16.kma_absolute_location")
import numpy as np, pandas as pd
from IPython.display import Image, display
import pygmt
import uf_vpvs as uv, uf_cluster as ufc
from kma_absolute_location import vpvs

FIGDIR = os.path.join(HYPO, "repeaters", "figs_hk"); os.makedirs(FIGDIR, exist_ok=True)
HK = "/home/msseo/works/07.SeismoStats/interp_k.txt"
def show(p, w): display(Image(filename=p, width=w))

# --- H-k grid (lon, lat, k); 0 = no-data ---
hk = np.loadtxt(HK)
lons = np.unique(hk[:, 0]); lats = np.unique(hk[:, 1])
dlon = np.median(np.diff(lons)); dlat = np.median(np.diff(lats))
REG = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
hk_v = hk[hk[:, 2] >= 0.5]                                   # drop no-data; missing nodes -> NaN in grid
grid = pygmt.xyz2grd(data=hk_v, region=REG, spacing=[dlon, dlat])

# --- in-situ families: Lin & Shearer (csv) + Zhang (computed) ---
rep, ev_of, meta = uv.families()
res = pd.read_csv(os.path.join(HYPO, "repeaters", "uf_vpvs_results.csv"))
ok = res[res.status == "ok"].copy()
zv = []
for _, r in ok.iterrows():
    P, S, _ = uv.build_family_dt(int(r.family), ev_of[int(r.family)], meta)
    zv.append(round(vpvs.zhang_vpvs(P, S, cc_min=uv.CCXC_MIN, min_stations=uv.MIN_STATIONS)["vpvs"], 3))
ok["zhang"] = zv
# H-k value sampled at each family
def hk_at(la, lo):
    return float(hk_v[np.argmin((hk_v[:,0]-lo)**2+(hk_v[:,1]-la)**2), 2])
ok["hk"] = [hk_at(a, b) for a, b in zip(ok.lat, ok.lon)]
print(f"H-k grid {REG}, spacing {dlon:.4f}/{dlat:.4f}; {len(ok)} in-situ families")
print(f"MEDIANS  in-situ Lin&Shearer {ok.vpvs.median():.3f} | in-situ Zhang {ok.zhang.median():.3f} | H-k bulk {ok.hk.median():.3f}")''' % HYPO)

    # ---------------------------------------------------------------- §1 the map
    md(r"""## 1. The map — H-κ Vp/Vs field, in-situ clusters as circles

Same colour scale (roma, **1.70–1.90**) for the H-κ background field **and** the in-situ circles, so a
circle **redder (lower) than its background** = local in-situ Vp/Vs *below* the bulk crust; **matching the
blue background** = no local anomaly. **(a)** in-situ by Lin & Shearer, **(b)** by Zhang. Black lines =
mapped faults (USF/YSF/MoRF/MiRF).""")
    co(r'''VMIN, VMAX = 1.70, 1.90
fig = pygmt.Figure()
pygmt.makecpt(cmap="roma", series=[VMIN, VMAX, 0.01], reverse=True, continuous=True)
FAULTS = [(129.31, 35.70, "USF", 285), (129.205, 35.85, "YSF", 70),
          (129.16, 35.90, "MoRF", 60), (129.035, 35.75, "MiRF", 65)]
for col, (label, vals) in enumerate([("(a) in-situ Lin and Shearer", ok.vpvs.values),
                                      ("(b) in-situ Zhang", ok.zhang.values)]):
    if col:
        fig.shift_origin(xshift="w+1.4c")
    with pygmt.config(MAP_FRAME_TYPE="plain", FORMAT_GEO_MAP="ddd.x", MAP_TICK_LENGTH_PRIMARY="0.2c",
                      FONT_ANNOT_PRIMARY="10p", FONT_LABEL="12p", FONT_TITLE="14p"):
        fig.basemap(region=REG, projection="M8.2c", frame=["a0.2f0.1", f"WSne+t{label}"])
    fig.grdimage(grid, cmap=True, transparency=30)
    fig.coast(shorelines="1/0.4p,black", water="white")
    try: ufc.plot_faults(fig)
    except Exception as e: print("faults:", e)
    for fx, fy, ft, fa in FAULTS:
        fig.text(x=fx, y=fy, text=ft, angle=fa, font="11p,Helvetica-Bold,black", justify="CB")
    # in-situ clusters (same cpt as the field); drop families where Zhang is unconstrained (NaN)
    mk = ~np.isnan(vals)
    fig.plot(x=ok.lon.values[mk], y=ok.lat.values[mk], fill=vals[mk], cmap=True, style="c0.36c", pen="0.9p,black")
fig.basemap(map_scale="jTR+w10k+o0.4c/0.6c+lkm")
with pygmt.config(FONT_LABEL="14p,Helvetica", FONT_ANNOT_PRIMARY="11p"):
    fig.colorbar(cmap=True, frame="a0.05+lVp/Vs", position="JBC+h+w11c/0.45c+o-5.3c/1.3c")
p = os.path.join(FIGDIR, "hk_map.png"); fig.savefig(p, dpi=300); show(p, 1500)''')

    # ---------------------------------------------------------------- §2 quantitative
    md(r"""## 2. In-situ vs bulk — the numbers

Per family, the in-situ value (both estimators) against the co-located H-κ bulk value.""")
    co(r'''import matplotlib, matplotlib.pyplot as plt
matplotlib.use("Agg")
HELV = "/home/msseo/Downloads/Helvetica/helvetica.ttf"
try:
    import matplotlib.font_manager as fm
    if os.path.isfile(HELV): fm.fontManager.addfont(HELV); matplotlib.rcParams["font.family"]=[fm.FontProperties(fname=HELV).get_name(),"DejaVu Sans"]
except Exception: pass
matplotlib.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
o = ok.sort_values("hk")
x = np.arange(len(o))
ax[0].scatter(x, o.hk, s=55, c="#cc3311", marker="_", linewidth=2.5, label="H-κ bulk crust", zorder=3)
ax[0].scatter(x, o.vpvs, s=40, c="#d95f0e", label="in-situ Lin & Shearer", zorder=3)
ax[0].scatter(x, o.zhang, s=40, c="#2c7fb8", marker="s", label="in-situ Zhang", zorder=3)
for xi, r in zip(x, o.itertuples()):
    ax[0].plot([xi, xi], [min(r.vpvs, r.zhang), r.hk], c="0.8", lw=1, zorder=1)
ax[0].axhline(1.73, ls=":", c="k", lw=1, label="1.73")
ax[0].set(xlabel="family (sorted by H-κ)", ylabel="Vp/Vs", title="(a) in-situ (both estimators) vs H-κ bulk")
ax[0].set_xticks([]); ax[0].legend(fontsize=8); ax[0].spines[["top","right"]].set_visible(False)
# (b) distributions
parts = [ok.vpvs.dropna().values, ok.zhang.dropna().values, ok.hk.dropna().values]
bp = ax[1].boxplot(parts, positions=[0,1,2], widths=0.6, patch_artist=True, showfliers=False)
for patch, c in zip(bp["boxes"], ["#d95f0e","#2c7fb8","#cc3311"]): patch.set_facecolor(c); patch.set_alpha(0.6)
for i, v in enumerate(parts): ax[1].scatter(np.full(len(v),i)+np.random.uniform(-0.1,0.1,len(v)), v, s=18, c="k", zorder=3)
ax[1].axhline(1.73, ls=":", c="k"); ax[1].set_xticks([0,1,2]); ax[1].set_xticklabels(["in-situ\nLin&Shearer","in-situ\nZhang","H-κ\nbulk"])
ax[1].set(ylabel="Vp/Vs", title="(b) distributions"); ax[1].spines[["top","right"]].set_visible(False)
fig.tight_layout(); p = os.path.join(FIGDIR, "hk_compare.png"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); show(p, 1350)
print(f"in-situ Lin&Shearer median {ok.vpvs.median():.3f} | in-situ Zhang median {ok.zhang.median():.3f} | H-κ bulk median {ok.hk.median():.3f}")''')

    md(r"""## 3. Reading the map

- The **H-κ bulk crust** is high (~1.86) and fairly uniform across the GHBSN footprint.
- **In-situ by Lin & Shearer** sits well **below** it (~1.46) — the apparent "local low-Vp/Vs fault zone".
  But we now know this is **largely a demeaning-dilution bias** for these tight clusters.
- **In-situ by Zhang** rises to ~1.72 — **close to, but still slightly below, the bulk H-κ**. So a *modest*
  local reduction may be real, but the dramatic low anomaly is not.
- **Caveat:** H-κ (column average, ~30 km path) and in-situ (tens-of-m source patch at ~12 km) sample
  different volumes; an exact match is not expected. The honest statement is that the in-situ values are
  **consistent with the bulk crust to within the estimator spread**, and pinning down any true local
  anomaly needs higher δt SNR (dense GHBSN) to collapse the Lin & Shearer ↔ Zhang gap.""")

    nb["cells"] = C
    return nb


def main():
    nb = _cells()
    with open(OUT, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {OUT}")
    from nbclient import NotebookClient
    nb2 = nbf.read(OUT, as_version=4)
    NotebookClient(nb2, timeout=2400, kernel_name="python3").execute()
    with open(OUT, "w") as f:
        nbf.write(nb2, f)
    print(f"executed {OUT}")


if __name__ == "__main__":
    main()
