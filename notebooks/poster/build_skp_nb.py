#!/usr/bin/env python
"""Generate notebooks/poster/03.SKP_regional_background.ipynb — peninsula-scale context.

Poster analysis 3. Two things the poster needs for orientation and for the central comparison:

  * the WHOLE southern Korean Peninsula: all HypoSVI-relocated KMA seismicity (2010-2024) and the
    same catalogue DECLUSTERED, so background (spontaneous) and triggered rates can be compared
    region-wide, not just inside the Ulsan box;
  * a ZOOM on the Gyeongju area showing the 2016 M5.5 aftershock zone and the Ulsan-Fault volume
    side by side — the visual argument that they are DISTINCT: a decaying aftershock sequence
    versus a persistent, largely spontaneous cluster ~10 km to the east.

Declustering uses the same canonical `kma_absolute_location.nnd` module and the same background
definition as notebook 01 (spontaneous = eta >= eta0, i.e. above the diagonal — a family's first
event is spontaneous). Parameters differ from 01 by design: this is a REGIONAL catalogue with
kilometre-scale absolute (HypoSVI) locations, not a 27 km box of waveform-relocated events, so the
generic Z&B Df=1.6, a wider family-linking cap, and the 2-D metric are appropriate here.

    python notebooks/poster/build_skp_nb.py

Kernel: base (pygmt + sklearn + xarray). The regional NND is O(N^2) over ~11.7k events — a few
minutes, single-threaded; it will not disturb the detections.
"""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "03.SKP_regional_background.ipynb")
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Southern Korean Peninsula — seismicity and its spontaneous background

Regional context for the Ulsan-Fault study, in two parts.

**Part A — the whole peninsula.** All HypoSVI-relocated KMA seismicity, 2010–2024, then the same
catalogue declustered by nearest-neighbour distance so the **spontaneous (background)** field can be
mapped separately from triggered sequences. Aftershock sequences dominate the raw event counts;
removing them shows where seismicity is *persistently* generated.

**Part B — the Gyeongju zoom.** The 2016 M5.5 Gyeongju aftershock zone and the Ulsan-Fault volume
are adjacent but behave differently: one is a decaying sequence around a single mainshock, the other
a persistent, largely spontaneous cluster. Seen at peninsula scale they blur together; zoomed in and
split by spontaneous/triggered they are clearly distinct.

Declustering uses the canonical `kma_absolute_location.nnd` and the **same background definition as
notebook 01** — spontaneous = η ≥ η₀ (above the diagonal), so a sequence's first event counts as
spontaneous. **Kernel: base.**""")

co(r'''# ============================ PARAMETERS ============================
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.ndimage import gaussian_filter
import xarray as xr
import pygmt

REPO   = "/home/msseo/works/02.Ulsan_Fault_detection"
NNDPKG = "/home/msseo/works/16.kma_absolute_location"
SVI    = NNDPKG + "/runs/kma_batch/results_final.csv"      # HypoSVI-relocated KMA catalog
FIGS   = REPO + "/notebooks/poster/figs"
sys.path.insert(0, NNDPKG); from kma_absolute_location import nnd

# --- regional NND parameters -------------------------------------------------------------------
# DIFFERENT from notebook 01 BY DESIGN. There the catalogue is a 27 km box of dt.cc-relocated events
# (~100 m precision) so Df was measured (1.2), the metric 3-D and the link cap 1 km. Here it is the
# whole peninsula with HypoSVI ABSOLUTE locations (km-scale errors, depths poorly resolved), so the
# generic Z&B values are the defensible choice and depth is left out of the metric.
DF_SKP   = 1.6            # generic Zaliapin-Ben-Zion fractal dimension
B_SKP    = 1.0            # fixed b
METRIC   = "2d"           # HypoSVI depths are weakly constrained at regional scale
LINKR    = nnd.LINK_RMAX_KM   # 10 km — the module default, appropriate at peninsula scale
MMIN     = None           # no Mc cut (eta depends on the PARENT magnitude)

RC   = [125.3, 130.8, 33.8, 38.5]          # peninsula map extent
UF   = [129.25, 129.55, 35.60, 35.90]      # Ulsan-Fault study box
ZOOM = [128.95, 129.65, 35.55, 36.05]      # tight Gyeongju + Ulsan zoom (S3)
# canonical full study area, from uflib.uf_cluster.REGION — spans WEST to the Milyang fault, so the
# regional plot (S5) shows the whole Gyeongju region rather than only the Gyeongju-Ulsan corridor.
sys.path.insert(0, REPO + "/src"); from uflib import uf_cluster as _ufc
GREG = list(_ufc.REGION)                   # [128.5, 130.0, 35.3, 36.5]
GJ_LON, GJ_LAT, GJ_T = 129.191, 35.766, "2016-09-12T11:32:54"   # 2016 M5.5 Gyeongju
POH_LON, POH_LAT     = 129.366, 36.109                           # 2017 M5.4 Pohang

try:
    fm.findfont("Helvetica", fallback_to_default=False); plt.rcParams["font.family"]="Helvetica"
except Exception:
    plt.rcParams["font.family"]="DejaVu Sans"
plt.rcParams.update({"figure.dpi":120,"savefig.dpi":400,"legend.framealpha":1.0,
                     "legend.facecolor":"white","axes.unicode_minus":False})
os.makedirs(FIGS, exist_ok=True)

def savegmt(fig, name):
    for ext in ("pdf","png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"))
    print(f"  saved figs/{name}.pdf + .png")

def savemp(fig, name):
    for ext in ("pdf","png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight")
    print(f"  saved figs/{name}.pdf + .png")

def annotate_mainshocks(fig, size="0.5c", label=True):
    """Mark the 2016 Gyeongju / 2017 Pohang mainshocks WITHOUT hiding the events beneath them:
    an OPEN star (no fill) plus an offset label, instead of a filled symbol that blanks the
    densest part of each aftershock cloud."""
    for lon, lat, nm, dx, dy in [(GJ_LON, GJ_LAT, "2016 M5.5 Gyeongju", -0.06, -0.14),
                                 (POH_LON, POH_LAT, "2017 M5.4 Pohang",  0.06,  0.14)]:
        fig.plot(x=[lon], y=[lat], style=f"a{size}", pen="1.4p,black")      # OPEN star
        if label:
            fig.text(x=lon+dx, y=lat+dy, text=nm, font="9p,Helvetica-Bold,black",
                     justify="RM" if dx < 0 else "LM", fill="white@30", pen="0.3p,gray50")

sv = pd.read_csv(SVI).dropna(subset=["svi_lat","svi_lon","kma_mag"]).copy()
sv = sv[sv.status.isin(["located","svi_only"])].copy()
print(f"KMA HypoSVI catalog: {len(sv):,} located events, ML {sv.kma_mag.min():.1f}..{sv.kma_mag.max():.1f}")''')

md(r"""## 1 — Decluster the regional catalogue

Same canonical module and the same background definition as notebook 01. `nnd.decimal_year` parses
the `event_id` (`YYYYMMDDHHMMSS`) directly — this catalogue is already in the module's native
schema.""")

co(r'''g = sv.copy()
g["event_id"] = g.event_id.astype(str)
g["t_year"]   = g.event_id.map(nnd.decimal_year)          # CANONICAL (native schema here)
g = g.dropna(subset=["t_year"]).sort_values("t_year").reset_index(drop=True)
g["event_time"] = pd.to_datetime(g.event_id, format="%Y%m%d%H%M%S", utc=True, errors="coerce")

nd = nnd.compute_nnd(g, b=B_SKP, D=DF_SKP, mmin=MMIN, metric=METRIC, rmax=nnd.RMAX_KM)
e0, info = nnd.fit_eta0(nd.eta.values, method="gmm"); LE0 = float(np.log10(e0))
labels = nnd.build_families(nd, e0, g.event_id.values, link_rmax_km=LINKR)
g["Cluster"] = g.event_id.map(labels).fillna(-1).astype(int)

# SAME definition as notebook 01: spontaneous = the event's own eta >= eta0 (above the diagonal),
# so a sequence's FIRST event is spontaneous (it has no parent), only its offspring are triggered.
triggered = set(nd.loc[nd.eta < e0, "event_id"].astype(str))
g["background"] = ~g.event_id.isin(triggered)
n_bg = int(g.background.sum()); n_tr = len(g)-n_bg
n_roots = int(((g.Cluster>=0) & g.background).sum())
assert info["means"][0] < LE0 < info["means"][1], "eta0 must lie between the GMM modes"
print(f"regional NND: Df={DF_SKP}, b={B_SKP}, {METRIC}, link_rmax={LINKR} km")
print(f"  log10(eta0) = {LE0:.2f}  (GMM means {np.round(info['means'],2)})")
print(f"  spontaneous {n_bg:,} ({100*n_bg/len(g):.0f}%)  -- incl. {n_roots:,} sequence first events")
print(f"  triggered   {n_tr:,} ({100*n_tr/len(g):.0f}%)   families {int(g.Cluster.max())+1:,}")''')

md(r"""## S1 — All seismicity vs spontaneous only, southern Korean Peninsula

Left: every located event. Right: the spontaneous set. Aftershock sequences (Gyeongju 2016, Pohang
2017) dominate the raw map; with them removed, what remains is where the peninsula *persistently*
generates earthquakes.""")

co(r'''fig = pygmt.Figure()
for i,(d_, lab) in enumerate([(g, f"All seismicity (n={len(g):,})"),
                              (g[g.background], f"Spontaneous / background (n={n_bg:,})")]):
    if i: fig.shift_origin(xshift="10.6c")
    fig.basemap(region=RC, projection="M10c", frame=["WSne" if i==0 else "wSne","xa1","ya1"])
    fig.coast(land="gray92", water="lightskyblue1", shorelines="0.5p,gray40",
              borders="1/0.4p,gray60", resolution="i")
    fig.plot(x=d_.svi_lon, y=d_.svi_lat, style="c0.05c", pen="0.25p,gray25", transparency=40)
    fig.plot(x=[UF[0],UF[1],UF[1],UF[0],UF[0]], y=[UF[2],UF[2],UF[3],UF[3],UF[2]], pen="1.8p,red")
    annotate_mainshocks(fig, size="0.45c", label=(i==0))
    fig.text(x=125.6, y=38.28, text=lab, font="8p,Helvetica,black", justify="LM",
             fill="white@20", pen="0.4p,gray40")
    fig.basemap(map_scale="jBL+w50k+o0.4c/0.4c")
savegmt(fig, "S1_skp_all_vs_background"); fig.show(width=1400)''')

md(r"""## S2 — Smoothed density: all vs spontaneous

The same two populations as smoothed density on a shared colour scale. The change between panels is
the point: sequences light up the raw field, while the spontaneous field shows the persistent
sources.""")

co(r'''def dgrid(lon, lat, ext, sp=0.04, sm=1.3):
    xb=np.arange(ext[0],ext[1]+sp,sp); yb=np.arange(ext[2],ext[3]+sp,sp)
    H,_,_=np.histogram2d(lon,lat,bins=[xb,yb]); H=gaussian_filter(H,sm)
    return xr.DataArray(H.T, coords={"lat":(yb[:-1]+yb[1:])/2,"lon":(xb[:-1]+xb[1:])/2},
                        dims=["lat","lon"])

dg_all = dgrid(g.svi_lon.values, g.svi_lat.values, RC)
dg_bg  = dgrid(g[g.background].svi_lon.values, g[g.background].svi_lat.values, RC)
# SEPARATE colour scale per panel: the Gyeongju/Pohang sequences dominate the raw field, so a shared
# scale would flatten the spontaneous panel into near-uniform pale. Independent p99 caps let each
# panel use its full dynamic range — the point being WHERE the density sits, not its absolute value.
vcap_all = float(np.percentile(dg_all.values[dg_all.values>0.05], 99))
vcap_bg  = float(np.percentile(dg_bg.values[dg_bg.values>0.05], 99))

fig = pygmt.Figure()
for i,(gr, lab, vc) in enumerate([(dg_all,"All seismicity", vcap_all),
                                  (dg_bg, "Spontaneous only", vcap_bg)]):
    if i: fig.shift_origin(xshift="11.2c")
    # hot reversed, and the LOW end forced to white so empty ground is white, not pale yellow
    pygmt.makecpt(cmap="hot", series=[0, vc], reverse=True, background="o")
    fig.basemap(region=RC, projection="M10c", frame=["WSne" if i==0 else "wSne","xa1","ya1"])
    fig.grdimage(grid=gr.where(gr > 0.02), cmap=True, nan_transparent=True)
    fig.coast(water="lightskyblue1", shorelines="0.5p,gray40", borders="1/0.4p,gray60", resolution="i")
    fig.plot(x=[UF[0],UF[1],UF[1],UF[0],UF[0]], y=[UF[2],UF[2],UF[3],UF[3],UF[2]], pen="1.8p,red")
    annotate_mainshocks(fig, size="0.45c")
    fig.text(x=125.6, y=38.28, text=lab, font="8p,Helvetica,black", justify="LM",
             fill="white@20", pen="0.4p,gray40")
    fig.colorbar(frame=[f'x+l"{lab} — events per cell"'], position="JBC+w7.5c/0.32c+h+o0c/0.9c")
savegmt(fig, "S2_skp_density"); fig.show(width=1500)
print(f"density caps (p99): all {vcap_all:.1f}   spontaneous {vcap_bg:.1f} events/cell")''')

md(r"""## S3 — Gyeongju zoom: the aftershock zone vs the Ulsan Fault

The comparison the poster rests on. Two adjacent volumes, ~10 km apart, behaving differently:

* around the **2016 M5.5 Gyeongju** epicentre — a dense, overwhelmingly **triggered** aftershock
  cloud that decays with time;
* the **Ulsan Fault** box — a persistent cluster with a much higher **spontaneous** fraction that
  was active before 2016 and remained active after.

Colour separates spontaneous (blue) from triggered (red); symbol size scales with magnitude.""")

co(r'''zm = g[(g.svi_lon.between(ZOOM[0],ZOOM[1])) & (g.svi_lat.between(ZOOM[2],ZOOM[3]))].copy()
# ML -> symbol size, following the reference figure's scaling (0.045 * 1.7**M): gentle enough that an
# M5.5 does not swallow its own aftershock zone, unlike a 2**M law.
zm["sz"] = 0.045*1.7**np.clip(zm.kma_mag.values, 0.0, None)
zb, zt = zm[zm.background], zm[~zm.background]

fig = pygmt.Figure()
fig.basemap(region=ZOOM, projection="M13c", frame=["WSne","xa0.2","ya0.2"])
fig.coast(land="gray98", water="lightblue", shorelines="0.4p,gray45", resolution="f")
# OPEN circles, outline-coloured, drawn triggered-first so the sparse spontaneous events stay visible
fig.plot(data=pd.DataFrame({"x":zt.svi_lon.values,"y":zt.svi_lat.values,"s":zt.sz.values}),
         style="c", pen="0.7p,firebrick")
fig.plot(data=pd.DataFrame({"x":zb.svi_lon.values,"y":zb.svi_lat.values,"s":zb.sz.values}),
         style="c", pen="0.7p,royalblue")
fig.plot(x=[UF[0],UF[1],UF[1],UF[0],UF[0]], y=[UF[2],UF[2],UF[3],UF[3],UF[2]], pen="2.0p,red")
fig.text(x=(UF[0]+UF[1])/2, y=UF[3]+0.035, text="Ulsan Fault study area",
         font="11p,Helvetica-Bold,red", justify="BC")
fig.plot(x=[GJ_LON], y=[GJ_LAT], style="a0.6c", pen="1.5p,black")     # OPEN: does not mask events
fig.text(x=GJ_LON-0.02, y=GJ_LAT+0.05, text="2016 M5.5 Gyeongju",
         font="10p,Helvetica-Bold,black", justify="RB", fill="white@30", pen="0.3p,gray50")

# boxed legend, lower-left. OPAQUE white (not white@25) so the data underneath does not ghost
# through the swatches, and generous row spacing so the M-labels never collide with the colour key.
bx0,bx1 = ZOOM[0]+0.012, ZOOM[0]+0.165
by0,by1 = ZOOM[2]+0.012, ZOOM[2]+0.210
fig.plot(x=[bx0,bx1,bx1,bx0,bx0], y=[by0,by0,by1,by1,by0], fill="white", pen="0.8p,black")
cx = bx0+0.024
for k,m in enumerate([5,4,3,2,1]):
    ly = by1-0.024-k*0.028
    fig.plot(x=[cx], y=[ly], style=f"c{0.045*1.7**m:.3f}c", pen="0.8p,black")
    fig.text(x=cx+0.034, y=ly, text=f"M {m}", font="8p,Helvetica,black", justify="LM")
fig.plot(x=[cx], y=[by0+0.042], style="c0.16c", pen="0.9p,royalblue")
fig.text(x=cx+0.034, y=by0+0.042, text="spontaneous", font="8p,Helvetica,black", justify="LM")
fig.plot(x=[cx], y=[by0+0.018], style="c0.16c", pen="0.9p,firebrick")
fig.text(x=cx+0.034, y=by0+0.018, text="triggered", font="8p,Helvetica,black", justify="LM")
fig.basemap(map_scale="jBR+w10k+o0.5c/0.5c")
savegmt(fig, "S3_gyeongju_zoom"); fig.show(width=1300)

# quantify the contrast between the two volumes
GJR = 0.12   # deg (~11 km) around the Gyeongju epicentre = the aftershock zone
dgj = np.hypot((zm.svi_lon-GJ_LON)*np.cos(np.radians(35.8)), zm.svi_lat-GJ_LAT)
gj  = zm[dgj <= GJR]
uf  = zm[(zm.svi_lon.between(UF[0],UF[1])) & (zm.svi_lat.between(UF[2],UF[3]))]
print(f"{'volume':<26}{'n':>7}{'spontaneous':>13}{'triggered':>11}{'spont %':>9}")
for lab,d_ in (("Gyeongju aftershock zone",gj), ("Ulsan Fault box",uf)):
    print(f"{lab:<26}{len(d_):>7}{int(d_.background.sum()):>13}{int((~d_.background).sum()):>11}"
          f"{100*d_.background.mean():>8.0f}%")''')

md(r"""## S5 — Gyeongju region, all seismicity (no spontaneous/triggered split)

The full study area (`uflib.uf_cluster.REGION`), which extends west to the **Milyang fault** — the
plain seismicity map, with no declustering applied. Circles are open and scaled by magnitude; the
Ulsan-Fault box and the two mainshocks are marked. Companion panel: the same events as smoothed
density, so the spatial pattern is readable where symbols overlap.""")

co(r'''gr_ = g[(g.svi_lon.between(GREG[0],GREG[1])) & (g.svi_lat.between(GREG[2],GREG[3]))].copy()
gr_["sz"] = 0.040*1.7**np.clip(gr_.kma_mag.values, 0.0, None)
print(f"Gyeongju region ({GREG}): {len(gr_):,} events, ML {gr_.kma_mag.min():.1f}..{gr_.kma_mag.max():.1f}")

fig = pygmt.Figure()
fig.basemap(region=GREG, projection="M12c", frame=["WSne","xa0.5","ya0.5"])
fig.coast(land="gray98", water="lightblue", shorelines="0.4p,gray45", resolution="f")
fig.plot(data=pd.DataFrame({"x":gr_.svi_lon.values,"y":gr_.svi_lat.values,"s":gr_.sz.values}),
         style="c", pen="0.5p,gray25")                     # OPEN circles, no population split
fig.plot(x=[UF[0],UF[1],UF[1],UF[0],UF[0]], y=[UF[2],UF[2],UF[3],UF[3],UF[2]], pen="2.0p,red")
fig.text(x=(UF[0]+UF[1])/2, y=UF[3]+0.05, text="Ulsan Fault", font="10p,Helvetica-Bold,red",
         justify="BC")
annotate_mainshocks(fig, size="0.55c")
bx0,bx1 = GREG[0]+0.03, GREG[0]+0.30; by0,by1 = GREG[2]+0.03, GREG[2]+0.42
fig.plot(x=[bx0,bx1,bx1,bx0,bx0], y=[by0,by0,by1,by1,by0], fill="white", pen="0.8p,black")
cx = bx0+0.05
for k,m in enumerate([5,4,3,2,1]):
    ly = by1-0.055-k*0.068
    fig.plot(x=[cx], y=[ly], style=f"c{0.040*1.7**m:.3f}c", pen="0.7p,black")
    fig.text(x=cx+0.07, y=ly, text=f"M {m}", font="8p,Helvetica,black", justify="LM")
fig.basemap(map_scale="jBR+w20k+o0.5c/0.5c")
savegmt(fig, "S5_gyeongju_region_all"); fig.show(width=1200)''')

co(r'''# companion: smoothed density of the same events (own colour scale, white background)
dg_reg = dgrid(gr_.svi_lon.values, gr_.svi_lat.values, GREG, sp=0.01, sm=1.5)
vcap_r = float(np.percentile(dg_reg.values[dg_reg.values>0.05], 99))
pygmt.makecpt(cmap="hot", series=[0, vcap_r], reverse=True, background="o")
fig = pygmt.Figure()
fig.basemap(region=GREG, projection="M12c", frame=["WSne","xa0.5","ya0.5"])
fig.grdimage(grid=dg_reg.where(dg_reg > 0.02), cmap=True, nan_transparent=True)
fig.coast(water="lightblue", shorelines="0.4p,gray45", resolution="f")
fig.plot(x=[UF[0],UF[1],UF[1],UF[0],UF[0]], y=[UF[2],UF[2],UF[3],UF[3],UF[2]], pen="2.0p,red")
annotate_mainshocks(fig, size="0.55c")
fig.colorbar(frame=['x+l"Events per cell (smoothed)"'], position="JBC+w8c/0.35c+h+o0c/1.0c")
fig.basemap(map_scale="jBR+w20k+o0.5c/0.5c")
savegmt(fig, "S5b_gyeongju_region_density"); fig.show(width=1200)
print(f"region density cap (p99) = {vcap_r:.1f} events/cell")''')

md(r"""## S4 — The same contrast in time

Cumulative counts for the two volumes. The Gyeongju zone is near-flat before September 2016 and then
jumps — a sequence. The Ulsan Fault accumulates through the whole period — persistent seismicity
that the mainshock perturbed but did not create.""")

co(r'''GJT = pd.Timestamp(GJ_T, tz="UTC")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True)
for ax, (lab, d_, col) in zip(axes, [("Gyeongju aftershock zone (r < 11 km)", gj, "#d62728"),
                                     ("Ulsan Fault study area", uf, "#1f77b4")]):
    s = d_.sort_values("event_time")
    ax.plot(s.event_time, np.arange(1,len(s)+1), lw=2.0, color=col, label=f"all (n={len(s)})")
    sb = s[s.background]; st = s[~s.background]
    ax.plot(sb.event_time, np.arange(1,len(sb)+1), lw=1.5, color=col, ls="--",
            label=f"spontaneous (n={len(sb)}, {100*len(sb)/max(len(s),1):.0f}%)")
    ax.plot(st.event_time, np.arange(1,len(st)+1), lw=1.5, color=col, ls=":",
            label=f"triggered (n={len(st)})")
    ax.axvline(GJT, color="0.25", lw=1.2)
    ax.set(xlabel="Year", ylabel="Cumulative events", title=lab)
    ax.set_xlim(pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC"))
    leg=ax.legend(loc="upper left", fontsize=9); leg.set_zorder(10)
fig.tight_layout()
savemp(fig, "S4_zoom_cumulative"); plt.show()''')

md(r"""## Summary""")

co(r'''rows = [
    ("KMA HypoSVI events (located, 2010-2024)", f"{len(g):,}"),
    ("regional NND parameters", f"Df={DF_SKP}, b={B_SKP}, {METRIC}, link_rmax={LINKR:g} km"),
    ("log10(eta_0)", f"{LE0:.2f}"),
    ("spontaneous (peninsula)", f"{n_bg:,} ({100*n_bg/len(g):.0f}%)"),
    ("triggered (peninsula)", f"{n_tr:,} ({100*n_tr/len(g):.0f}%)"),
    ("Gyeongju zone: n / spontaneous %", f"{len(gj):,} / {100*gj.background.mean():.0f}%"),
    ("Ulsan Fault box: n / spontaneous %", f"{len(uf):,} / {100*uf.background.mean():.0f}%"),
]
S = pd.DataFrame(rows, columns=["quantity","value"])
print(S.to_string(index=False))
S.to_csv(os.path.join(FIGS, "skp_numbers.csv"), index=False)
g[["event_id","event_time","svi_lon","svi_lat","svi_dep","kma_mag","Cluster","background"]] \
    .to_csv(os.path.join(FIGS, "skp_events.csv"), index=False)
print("\n-> figs/skp_numbers.csv  +  figs/skp_events.csv")''')

nb = nbf.v4.new_notebook(cells=cells,
                         metadata={"kernelspec":{"name":"python3","display_name":"Python 3 (base)",
                                                 "language":"python"}})
with open(NB, "w") as f:
    nbf.write(nb, f)
print(f"wrote {NB} ({len(cells)} cells, unexecuted)")
