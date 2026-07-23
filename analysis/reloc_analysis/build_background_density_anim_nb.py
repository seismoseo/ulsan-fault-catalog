#!/usr/bin/env python
"""Generate 39.UF_background_density_animation.ipynb — an animated GIF of how the BACKGROUND (declustered)
seismicity density of the UF subregion varies over 2010-2024.

Visualization choice (disclosed): seismicity is a point process, so a single "density movie" requires an
estimator. We use the standard SLIDING-WINDOW SMOOTHED RATE MAP: a 2-yr window stepping 2 months; per frame a
2-D histogram (0.004 deg grid) Gaussian-smoothed (~0.7 km), normalised to events/yr/km^2, drawn on a FIXED
colour scale so frames are directly comparable; the actual epicentres in the window stay overlaid as neutral
open circles (the point process itself, so the smoothing never hides the data); M>=3 spontaneous events get a
star. A timeline strip below shows the cumulative background count with the current window shaded. Window
length trades temporal resolution vs statistical stability: the background rate is ~65 events/yr, so a 2-yr
window holds ~130 events — enough for a stable smoothed map; a 1-yr window would be noise-dominated.

Background = the ZBZ-declustered spontaneous set (eta>=eta0 at the adopted Df=1.2, keeps family roots +
singletons; identical recipe to nb27 Sec 5c), on the de-blasted ML-resolved population. Alternatives considered
(noted in-notebook): cumulative map coloured by time (no rate info), lon-vs-time plume (loses 2-D geometry),
per-frame points only (unreadable at ~130 pts).

Style: dataviz rules — sequential single-hue ramp (Blues, white low end), fixed vmin/vmax, one axis per panel,
recessive grid, neutral-ink points, black SOTA coastline + grey fault traces, Helvetica.
Output: UF_background_density_2010_2024.gif (loop, ~150 ms/frame) + a 6-frame contact sheet in the notebook."""
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# Animated background-seismicity density of the Ulsan Fault (2010–2024)

Seismicity is a **point process**, so "density vs time" needs an explicit estimator. The choice here is the
standard **sliding-window smoothed rate map**, animated:

- **Window**: 2 years, stepping 2 months (~78 frames). The background rate is ~65 events/yr, so each frame holds
  ~130 events — a 1-yr window would be noise-dominated; 2 yr is the stability/resolution compromise.
- **Rate estimate**: per-frame 2-D histogram on a 0.004° grid, Gaussian-smoothed (~0.7 km, the nb26/nb36 kernel),
  normalised to **events · yr⁻¹ · km⁻²**.
- **Fixed colour scale** across all frames (single-hue sequential ramp) — otherwise frames are not comparable.
- **The points stay visible**: the actual epicentres in the window are overlaid as neutral open circles, so the
  smoothing never hides the underlying point process; spontaneous events with **M ≥ 3** get a star.
- A **timeline strip** below shows the cumulative background count with the current window shaded, so each frame
  is anchored in the 15-yr history.

**Background** = the ZBZ-declustered *spontaneous* set (η ≥ η₀ at the adopted D_f = 1.2 — keeps family
roots/mainshocks and singletons; the exact nb27 §5c recipe) on the **de-blasted** ML-resolved population.

*Alternatives considered:* a cumulative map coloured by time (no rate information), longitude-vs-time plumes
(lose the 2-D fault geometry), points-only frames (unreadable at ~130 points). The sliding-window rate map with
points overlaid keeps rate, geometry and the point process all visible.""")

# ------------------------------------------------------------------ §0 load + decluster
co(r"""import warnings; warnings.filterwarnings("ignore")
import os, sys, numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.font_manager as fm
from scipy.ndimage import gaussian_filter
from PIL import Image
sys.path.insert(0,"/home/msseo/works/16.kma_absolute_location"); from kma_absolute_location import nnd
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":110,"font.size":10,"axes.grid":False,"axes.unicode_minus":False})

KG="/home/msseo/works/02.Ulsan_Fault_detection/KS_KG"
REGION=[129.25,129.55,35.60,35.90]; ASP=1/np.cos(np.deg2rad(35.75))
rl=pd.read_csv(f"{KG}/local_magnitudes/catalog_ml_heo_ufonly_reloc.csv")
rl=rl[~rl.event_idx.isin(set(pd.read_csv(f"{KG}/local_magnitudes/blast_event_idx_deblast.csv")
                             .event_idx.dropna().astype(int)))].copy()      # DE-BLAST (nb22 §7)
g=rl[rl.n_used>=3].dropna(subset=["lat","lon","depth","ml_ufcorr_reloc"]).copy()
g["event_time"]=pd.to_datetime(g.event_time,format="ISO8601",utc=True,errors="coerce")
g=g.dropna(subset=["event_time"]).sort_values("event_time").reset_index(drop=True)
g["t_year"]=g.event_time.dt.strftime("%Y%m%d%H%M%S").map(nnd.decimal_year); g["event_id"]=np.arange(len(g))
g=g.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep","ml_ufcorr_reloc":"kma_mag"})
nd=nnd.compute_nnd(g,b=1.0,D=1.2,mmin=None,metric="3d"); e0,_=nnd.fit_eta0(nd.eta.values,method="gmm")
_clu=set(nd.loc[nd.eta<e0,"event_id"])
bg=g[~g.event_id.isin(_clu)].copy()                       # spontaneous: eta>=eta0 (+ first event) — nb27 §5c recipe
print(f"population {len(g)} (de-blasted, n_used>=3) -> background/spontaneous {len(bg)} "
      f"({100*len(bg)/len(g):.0f}%), span {bg.event_time.min():%Y-%m}..{bg.event_time.max():%Y-%m}")
# coast + fault segments (nb28 idiom)
def _load_segs(path):
    segs=[]; cur=[]
    if not os.path.exists(path): return segs
    for ln in open(path):
        if ln.startswith((">","#")):
            if len(cur)>1: segs.append(np.array(cur))
            cur=[]; continue
        p=ln.split()
        if len(p)>=2:
            try: cur.append([float(p[0]),float(p[1])])
            except ValueError: pass
    if len(cur)>1: segs.append(np.array(cur))
    return segs
FSEG=_load_segs(f"{KG}/HypoInv/faults_lonlat.gmt"); CSEG=_load_segs(f"{KG}/reloc_analysis/coastline_lonlat.gmt")""")

# ------------------------------------------------------------------ §1 frames + fixed scale
md(r"""## 1 · Precompute the frame grids and the fixed colour scale

All window grids are computed first so a single global maximum fixes the colour scale (a per-frame scale would
fake rate changes). Grid cell ≈ 0.36 × 0.44 km (area 0.16 km²); rate = counts / 2 yr / cell-area.""")
co(r"""SP=0.004; SIG=1.5                                            # grid step (deg), Gaussian sigma (cells) ~0.7 km
xb=np.arange(REGION[0],REGION[1]+SP,SP); yb=np.arange(REGION[2],REGION[3]+SP,SP)
_dx=SP*111.320*np.cos(np.deg2rad(35.75)); _dy=SP*110.574; CELL_KM2=_dx*_dy
W=pd.DateOffset(years=1)                                     # half-window
CENTERS=pd.date_range("2011-01-01","2023-11-01",freq="2MS",tz="UTC")
def wgrid(tc):
    m=(bg.event_time>=tc-W)&(bg.event_time<tc+W)
    H,_,_=np.histogram2d(bg.svi_lon[m],bg.svi_lat[m],bins=[xb,yb])
    return gaussian_filter(H,SIG).T/2.0/CELL_KM2, m           # events/yr/km^2
GRIDS=[wgrid(tc) for tc in CENTERS]
VMAX=max(gr.max() for gr,_ in GRIDS)
print(f"{len(CENTERS)} frames ({CENTERS[0]:%Y-%m}..{CENTERS[-1]:%Y-%m}), window 2 yr step 2 mo")
print(f"events per window: median {int(np.median([m.sum() for _,m in GRIDS]))}, "
      f"range {min(m.sum() for _,m in GRIDS)}-{max(m.sum() for _,m in GRIDS)}")
print(f"fixed colour scale: 0..{VMAX:.2f} events/yr/km^2 (global max over all frames; sqrt colour mapping)")""")

# ------------------------------------------------------------------ §2 render + GIF
md(r"""## 2 · Render the frames and write the GIF""")
co(r"""EXT=[REGION[0],REGION[1],REGION[2],REGION[3]]
_ct=bg.event_time.sort_values(); _cum=np.arange(1,len(_ct)+1)
def make_frame(i):
    tc=CENTERS[i]; gr,m=GRIDS[i]; w=bg[m]
    fig=plt.figure(figsize=(7.0,9.2))
    gs=fig.add_gridspec(2,1,height_ratios=[5.4,1],hspace=0.16,left=0.11,right=0.88,top=0.965,bottom=0.065)
    ax=fig.add_subplot(gs[0])
    im=ax.imshow(gr,origin="lower",extent=EXT,cmap="Blues",norm=mpl.colors.PowerNorm(0.5,vmin=0,vmax=VMAX),
                 aspect=ASP,interpolation="bilinear",zorder=1)   # sqrt scale: fixed+honest, low end legible
    for s in FSEG: ax.plot(s[:,0],s[:,1],color="0.45",lw=0.7,zorder=2)
    for s in CSEG: ax.plot(s[:,0],s[:,1],color="black",lw=0.8,zorder=3)
    ax.scatter(w.svi_lon,w.svi_lat,s=9,facecolor="white",edgecolor="0.25",lw=0.5,zorder=4)
    big=w[w.kma_mag>=3.0]
    if len(big): ax.scatter(big.svi_lon,big.svi_lat,s=170,marker="*",facecolor="none",edgecolor="black",lw=1.1,zorder=5)
    ax.text(0.02,0.975,f"{tc-W:%Y-%m} – {tc+W:%Y-%m}\n{int(m.sum())} background events",
            transform=ax.transAxes,ha="left",va="top",fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35",fc="white",ec="0.55",lw=0.7))
    ax.set(xlim=REGION[:2],ylim=REGION[2:],xlabel="Longitude",ylabel="Latitude",
           title="Ulsan Fault — background (declustered) seismicity rate, 2-yr sliding window")
    cax=fig.add_axes([0.90,0.30,0.022,0.50])
    cb=fig.colorbar(im,cax=cax); cb.set_label("background events · yr$^{-1}$ · km$^{-2}$")
    # timeline strip: cumulative background count, current window shaded
    axt=fig.add_subplot(gs[1])
    axt.step(_ct,_cum,where="post",color="0.35",lw=1.2)
    axt.axvspan(tc-W,tc+W,color="#3b6fb6",alpha=0.22,zorder=0)
    axt.set(ylabel="cum. count",xlim=(_ct.iloc[0],_ct.iloc[-1]))
    axt.xaxis.set_major_locator(mdates.YearLocator(2)); axt.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axt.grid(alpha=0.25)
    fig.canvas.draw()
    img=Image.frombuffer("RGBA",fig.canvas.get_width_height(),fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig); return img

frames=[make_frame(i) for i in range(len(CENTERS))]
GIF=f"{KG}/reloc_analysis/UF_background_density_2010_2024.gif"
frames[0].save(GIF,save_all=True,append_images=frames[1:]+[frames[-1]]*6,   # hold the last frame ~1 s
               duration=150,loop=0,optimize=True)
print(f"wrote {GIF}  ({os.path.getsize(GIF)/1e6:.1f} MB, {len(frames)} frames, 150 ms/frame, loops)")""")

# ------------------------------------------------------------------ §3 contact sheet
md(r"""## 3 · Contact sheet — six snapshots

Static check of the animation (same fixed scale): six windows spanning the catalog.""")
co(r"""PICK=[np.argmin(np.abs(CENTERS-pd.Timestamp(t,tz="UTC"))) for t in
      ("2011-06-01","2014-01-01","2016-06-01","2019-01-01","2021-06-01","2023-06-01")]
_w,_h=frames[0].size; sc=0.42; tw,th=int(_w*sc),int(_h*sc)
sheet=Image.new("RGB",(tw*3,th*2),"white")
for k,idx in enumerate(PICK):
    sheet.paste(frames[idx].resize((tw,th),Image.LANCZOS),((k%3)*tw,(k//3)*th))
from IPython.display import display
display(sheet)
print("frames shown:",", ".join(f"{CENTERS[i]:%Y-%m}" for i in PICK))
print(f"\nGIF: {GIF}")""")

nb["cells"]=C
import os as _os
_out=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"39.UF_background_density_animation.ipynb")
nbf.write(nb,_out)
print("wrote",_out,"with",len(C),"cells")
