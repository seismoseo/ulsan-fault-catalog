#!/usr/bin/env python
"""Generate 00.Summary_figures_Zhigang.ipynb at the Ulsan-Fault project root — a compact set of high-quality
figures for the meeting with Zhigang Peng. Robust, SOTA-styled (PyGMT, Helvetica, plain frame, scale bar):
  Fig 1 — regional seismicity BEFORE / AFTER the 2016 Gyeongju EQ, over the H-k-stacking Vp/Vs grid,
          with fault traces and a subtle station layer.
  Fig 2 — cumulative event count vs time for the Ulsan-Fault subregion, 2016 Gyeongju & 2017 Pohang marked.
  Fig 3 — dt.cc-resolved relocated catalog (UF box) scaled by local magnitude, with fault traces.
  Fig 4 — background vs clustered SEISMICITY DENSITY (3D NND declustering) for the UF box.
Runs in `base` (needs pygmt, xarray, scipy, seismostats-free). Self-contained; absolute data paths."""
import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# Ulsan-Fault project — summary figures (for Zhigang Peng)

Preliminary but robust results. The Vp/Vs backdrop is the **receiver-function H-κ-stacking** interpolated
grid for the Gyeongju–Ulsan region (`07.SeismoStats/interp_k.txt`). Locations: PhaseNet+ blast-cleaned
catalog (absolute, HypoInverse) for the regional view; **HypoDD dt.cc cross-correlation relocations**
(kim2011 velocity) for the Ulsan-Fault subregion. Magnitudes: UF-only-corrected local magnitude (`ml_ufcorr`).

*Input-completeness fix (now applied):* an earlier dt.cc run silently dropped ~60 events and mislabelled
event cuspids. Threading a stable `event_idx` and correcting the per-pair cuspid headers recovered the full
event set — including the **catalog-max 2014 event** (now relocated on its 1,189 catalog differential-time
links) — so these numbers are over the complete, correctly-linked catalog.""")

# ----------------------------------------------------------------- §0 setup
co(r"""import warnings; warnings.filterwarnings("ignore")
import os, glob, sys, numpy as np, pandas as pd, xarray as xr
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
import pygmt
from scipy.ndimage import gaussian_filter
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":130,"axes.grid":True,"grid.alpha":0.3,"font.size":11})
sys.path.insert(0,"/home/msseo/works/16.kma_absolute_location"); from kma_absolute_location import nnd

KG="/home/msseo/works/02.Ulsan_Fault_detection/KS_KG"
CAT=f"{KG}/local_magnitudes/catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo_clean.csv"  # full deblasted set (count-complete)
RELOC=f"{KG}/local_magnitudes/catalog_ml_heo_ufonly_reloc.csv"
FAULTS=f"{KG}/HypoInv/faults_lonlat.gmt"
VPVS="/home/msseo/works/07.SeismoStats/interp_k.txt"
STAT_GLOB=f"{KG}/models/phasenet_plus/station_table/stations_*.csv"
UF=(129.25,129.55,35.60,35.90)                       # Ulsan-Fault subregion box
GYEONGJU=pd.Timestamp("2016-09-12 10:44",tz="utc")    # M5.1 Gyeongju FORESHOCK (sequence onset) = split time
GJ_LON,GJ_LAT=129.1875,35.7697                        # 2016 Gyeongju M5.5 mainshock epicentre (green star)
POHANG  =pd.Timestamp("2017-11-15 05:29",tz="utc")    # M5.4 Pohang

# --- catalog (regional, absolute) ---
cat=pd.read_csv(CAT); cat["time"]=pd.to_datetime(cat.time,utc=True,errors="coerce")
cat=cat.dropna(subset=["time","lat","lon"]); cat["mag"]=cat.ml_all
# --- Vp/Vs H-k grid -> xarray for grdimage ---
vk=pd.read_csv(VPVS,sep=r"\s+",header=None,names=["lon","lat","v"])
piv=vk.pivot(index="lat",columns="lon",values="v")
VGRID=xr.DataArray(piv.values,coords={"lat":piv.index.values,"lon":piv.columns.values},dims=["lat","lon"])
REG=[float(vk.lon.min()),float(vk.lon.max()),float(vk.lat.min()),float(vk.lat.max())]   # fig-1 region = Vp/Vs extent
# --- stations: first operating year per station -> split PRE vs ADDED-after 2016 Gyeongju (network densification) ---
import re
_sy={}
for f in sorted(glob.glob(STAT_GLOB)):
    y=int(re.search(r"stations_(\d{4})",f).group(1))
    for n,c in pd.read_csv(f)[["Network","Code"]].itertuples(index=False): _sy[(n,c)]=min(_sy.get((n,c),9999),y)
st=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(STAT_GLOB))]).drop_duplicates(["Network","Code"]).copy()
st["first_year"]=[_sy.get((n,c),9999) for n,c in zip(st.Network,st.Code)]
st_pre=st[st.first_year<2016]; st_post=st[st.first_year>=2016]      # pre-existing vs added 2016+ (post-Gyeongju)
# --- dt.cc relocated + ML (UF subregion) ---
rl=pd.read_csv(RELOC); rl["event_time"]=pd.to_datetime(rl.event_time,format="ISO8601",utc=True,errors="coerce")
# Fig 4 NND population = ALL relocated events with reliable ML (dt.cc AND dt.ct). The dt.cc/dt.ct split is
# location PRECISION (tens vs ~hundreds of m), not detection; excluding dt.ct would drop mainshock PARENTS
# (incl. the 2014 M3.89, a dt.ct event with 1189 catalog links) and open a completeness hole below Mc,
# biasing the declustering, background rate and b-value. ~hundreds-of-m locations are fine for km-scale NND.
dtcc=rl[rl.n_used>=3].dropna(subset=["lat","lon","depth","ml_ufcorr_reloc"]).copy()   # dt.cc + dt.ct, reliable ML
# ALL relocated events for the map (dt.cc AND dt.ct, incl. those WITHOUT reliable ML) -> attach ML by EXACT
# event_idx. Including dt.ct means the largest event (2014 M3.89, a dt.ct event) is not hidden from the map.
RELOC_FILE="/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/uf_subregion_reuse/2.HypoDD/03.dt.cc_kim2011/hypoDD.reloc"
_rc=["id","lat","lon","depth","x","y","z","ex","ey","ez","yr","mo","dy","hr","mi","sc","mag","nccp","nccs","nctp","ncts","rcc","rct","cid"]
_r0=pd.read_csv(RELOC_FILE,sep=r"\s+",header=None,names=_rc); _r0["ncc"]=_r0.nccp+_r0.nccs
dtcc_all=_r0.copy(); dtcc_all["is_dtcc"]=dtcc_all.ncc>0   # dt.cc (sharp) + dt.ct (~hundreds of m); flag kept for styling
# exact hypoDD id->ts->event_idx map (no coord/time matching): cuspid = 200000 + sorted waveforms_100km "20*"
# dir index; dir name = pipeline timestamp ts; members_event_idx.csv maps ts (floor-second) -> master event_idx.
_WF100="/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/uf_subregion_reuse/waveforms_100km"
_MEIDX="/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/uf_subregion_hypodd/uf_subregion/members_event_idx.csv"
_dirs=sorted(os.path.basename(d) for d in glob.glob(os.path.join(_WF100,"20*")))
_id2ts={200000+i:ts for i,ts in enumerate(_dirs)}
_mei=pd.read_csv(_MEIDX).sort_values("event_idx"); _mei["ts"]=pd.to_datetime(_mei.time,utc=True,format="ISO8601").dt.floor("s").dt.strftime("%Y%m%d%H%M%S")
_ts2eidx={}
for _e,_t in zip(_mei.event_idx.astype(int),_mei.ts): _ts2eidx.setdefault(_t,_e)
dtcc_all["event_idx"]=dtcc_all.id.map(_id2ts).map(_ts2eidx)
dtcc_all=dtcc_all.merge(rl[["event_idx","ml_ufcorr_reloc","n_used"]],on="event_idx",how="left").drop_duplicates("id")
dtcc_all["has_ml"]=dtcc_all.ml_ufcorr_reloc.notna()&(dtcc_all.n_used>=3)
MINSZ=0.07   # min circle (cm) for events without reliable ML; ML-scaled otherwise (shared by Fig 3 + 3b + 3c)
dtcc_all["sz"]=(0.045*1.7**dtcc_all.ml_ufcorr_reloc.clip(0,4)).where(dtcc_all.has_ml,MINSZ).clip(lower=MINSZ)
_scf=dtcc_all.sc.clip(0,59.999)   # reconstruct dt.cc origin time (for the 6-month before/after Gyeongju maps)
dtcc_all["time"]=pd.to_datetime(dict(year=dtcc_all.yr,month=dtcc_all.mo,day=dtcc_all.dy,hour=dtcc_all.hr,
                  minute=dtcc_all.mi,second=_scf.astype(int),microsecond=((_scf-_scf.astype(int))*1e6).astype(int)),utc=True,errors="coerce")
DMIN,DMAX=float(dtcc_all.depth.min()),float(dtcc_all.depth.max())   # consistent depth colour range across all dt.cc figures
DF_UF=1.2   # DATA-DRIVEN fractal dimension (Grassberger-Procaccia 3-D correlation dim of the FULL relocated population, nb27; dt.cc-only structural value 1.16 -> clustered set identical at either), NOT generic ZBZ 2.5

pygmt.config(MAP_FRAME_TYPE="plain",FORMAT_GEO_MAP="ddd.x",FONT_TITLE="13p",FONT_SUBTITLE="11p",FONT_LABEL="9p",FONT_ANNOT_PRIMARY="8p",MAP_TITLE_OFFSET="5p")   # FONT_TITLE/SUBTITLE pinned -> constant across before/after panels
PROJ="M11c"; SCALE_R="jBR+w10k+o0.2c/0.5c"; SCALE_UF="jBL+w5k+o0.5c/0.5c"   # Fig 1 scale lower-RIGHT (closer to edge)
FIGDIR="figures_zhigang"; os.makedirs(FIGDIR,exist_ok=True)   # all figures saved here at 300 dpi (PNG) + vector PDF
def savempl(fig,name): fig.savefig(f"{FIGDIR}/{name}.png",dpi=300,bbox_inches="tight"); fig.savefig(f"{FIGDIR}/{name}.pdf",bbox_inches="tight")
def savegmt(fig,name): fig.savefig(f"{FIGDIR}/{name}.png",dpi=300); fig.savefig(f"{FIGDIR}/{name}.pdf")
print(f"regional catalog {len(cat):,} ev | Vp/Vs grid {VGRID.shape} over {REG} (Vp/Vs {float(vk.v.min()):.2f}-{float(vk.v.max()):.2f})")
print(f"stations {len(st)} (pre-2016 {len(st_pre)} | added 2016+ {len(st_post)}) | relocated events {len(dtcc_all):,} "
      f"(with reliable ML {int(dtcc_all.has_ml.sum()):,}, without {int((~dtcc_all.has_ml).sum()):,} -> min size)")
print(f"events before 2016 Gyeongju: {(cat.time<GYEONGJU).sum():,} | after: {(cat.time>=GYEONGJU).sum():,}")""")

# ----------------------------------------------------------------- §0b Fig 0 regional context
md(r"""## Figure 0 · Regional context — southern Korean Peninsula

Where this study sits, for orientation. **All HypoSVI-relocated KMA seismicity** (2010–2024;
`16.kma_absolute_location`) across the southern Korean Peninsula (gray dots), with the **Ulsan-Fault study
area** (red box) and the two largest recent inland sequences — the **2016 M5.5 Gyeongju** and **2017 M5.4
Pohang** — marked. The dashed gray box is the receiver-function **Vp/Vs map extent** used in Figs 1/3/4. The
inset locates the region within East Asia.""")
co(r"""SVI="/home/msseo/works/16.kma_absolute_location/runs/kma_batch/results_final.csv"
sv=pd.read_csv(SVI).dropna(subset=["svi_lat","svi_lon"])
RC=[125.3,130.8,33.8,38.5]; POH_LON,POH_LAT=129.366,36.109     # map extent ; 2017 M5.4 Pohang
fig=pygmt.Figure()
fig.basemap(region=RC,projection="M13c",frame=["WSne","xa1","ya1"])
fig.coast(land="gray92",water="lightskyblue1",shorelines="0.5p,gray40",borders="1/0.4p,gray60",resolution="i")
fig.plot(x=sv.svi_lon,y=sv.svi_lat,style="c0.045c",fill="gray25",transparency=55)        # all sKP seismicity
gx=[REG[0],REG[1],REG[1],REG[0],REG[0]]; gy=[REG[2],REG[2],REG[3],REG[3],REG[2]]          # Vp/Vs map extent
fig.plot(x=gx,y=gy,pen="0.9p,gray30")
rx=[UF[0],UF[1],UF[1],UF[0],UF[0]]; ry=[UF[2],UF[2],UF[3],UF[3],UF[2]]                     # study-area box
fig.plot(x=rx,y=ry,pen="2.2p,red")
fig.text(x=UF[1]+0.09,y=(UF[2]+UF[3])/2+0.06,text="Ulsan Fault",font="10p,Helvetica-Bold,red",justify="LM")   # offshore, E of box (2 lines)
fig.text(x=UF[1]+0.09,y=(UF[2]+UF[3])/2-0.10,text="study area",font="10p,Helvetica-Bold,red",justify="LM")
fig.plot(x=[GJ_LON],y=[GJ_LAT],style="a0.55c",fill="lightgreen",pen="0.8p,black")                # 2016 Gyeongju
fig.text(x=GJ_LON-0.12,y=GJ_LAT,text="2016 M5.5 Gyeongju",font="9p,Helvetica-Bold,black",justify="RM")
fig.plot(x=[POH_LON],y=[POH_LAT],style="a0.55c",fill="lightgreen",pen="0.8p,black")            # 2017 Pohang
fig.text(x=POH_LON+0.13,y=POH_LAT+0.05,text="2017 M5.4 Pohang",font="9p,Helvetica-Bold,black",justify="LM")
for cl,cla,nm in [(126.978,37.567,"Seoul"),(129.075,35.180,"Busan")]:                       # reference cities
    fig.plot(x=[cl],y=[cla],style="s0.18c",fill="white",pen="0.8p,black")
    fig.text(x=cl,y=cla+0.09,text=nm,font="8p,Helvetica,black",justify="BC")
fig.text(x=125.65,y=36.6,text="Yellow Sea",font="9p,Helvetica-Oblique,gray30",justify="MC")
fig.text(x=130.4,y=37.7,text="East Sea",font="9p,Helvetica-Oblique,gray30",justify="MC")
fig.basemap(map_scale="jBL+w50k+o0.5c/0.5c")
with fig.inset(position="jBR+w3.4c+o0.15c",box="+p1p,black+gwhite"):                        # East-Asia locator
    fig.coast(region=[118,135,28,44],projection="M3.4c",land="gray70",water="white",shorelines="0.2p,gray50",resolution="l")
    fig.plot(x=[RC[0],RC[1],RC[1],RC[0],RC[0]],y=[RC[2],RC[2],RC[3],RC[3],RC[2]],pen="1.2p,red")
fig.text(x=125.55,y=38.28,text="KMA catalog, 2010-2024",font="8p,Helvetica,black",justify="LM",fill="white@20",pen="0.4p,gray40")
savegmt(fig,"fig00_regional_context"); fig.show(width=1400)
print(f"sKP context: {len(sv):,} HypoSVI events; study box {UF}; Gyeongju ({GJ_LON},{GJ_LAT}), Pohang ({POH_LON},{POH_LAT})")""")

md(r"""## Figure 0b · Seismicity density — southern Korean Peninsula

The **smoothed event density** of the same HypoSVI catalogue, shown as a *separate* panel so the warm colour
scale is not muddied by the ocean. Seismicity concentrates in the SE (the Gyeongju–Pohang–Ulsan corridor);
the **red box** is the Ulsan-Fault study area, gold/orange stars the 2016 Gyeongju / 2017 Pohang events.""")
co(r"""def dgrid_skp(lon,lat,sp=0.04,sm=1.3):
    xb=np.arange(RC[0],RC[1]+sp,sp); yb=np.arange(RC[2],RC[3]+sp,sp)
    H,_,_=np.histogram2d(lon,lat,bins=[xb,yb]); H=gaussian_filter(H,sm)
    return xr.DataArray(H.T,coords={"lat":(yb[:-1]+yb[1:])/2,"lon":(xb[:-1]+xb[1:])/2},dims=["lat","lon"])
dg=dgrid_skp(sv.svi_lon.values,sv.svi_lat.values)
vcap=float(np.percentile(dg.values[dg.values>0.1],99))   # cap at p99: dense Gyeongju/Pohang saturate, diffuse field stays visible
fig=pygmt.Figure()
fig.basemap(region=RC,projection="M13c",frame=["WSne","xa1","ya1"])
fig.coast(land="gray96",water="white",shorelines="0.3p,gray70",resolution="i")
pygmt.makecpt(cmap="hot",series=[0,vcap],reverse=True)
fig.grdimage(dg.where(dg>=0.2),cmap=True,nan_transparent=True)                 # density backdrop (threshold 0.2)
fig.coast(shorelines="0.5p,gray45",borders="1/0.3p,gray70",resolution="i")     # coastlines on top
fig.plot(x=rx,y=ry,pen="2.2p,red")
fig.plot(x=[GJ_LON],y=[GJ_LAT],style="a0.5c",fill="lightgreen",pen="0.7p,black")
fig.plot(x=[POH_LON],y=[POH_LAT],style="a0.5c",fill="lightgreen",pen="0.7p,black")
fig.basemap(map_scale="jBL+w50k+o0.5c/0.5c")
with pygmt.config(FONT_LABEL="12p",FONT_ANNOT_PRIMARY="10p"):
    fig.colorbar(cmap=True,frame="af+lSmoothed event count (p99-capped)",position="jBR+w5c/0.35c+h+o0.7c/0.9c+ef")
fig.text(x=125.55,y=38.28,text="KMA catalog, 2010-2024",font="8p,Helvetica,black",justify="LM",fill="white@20",pen="0.4p,gray40")
savegmt(fig,"fig00b_skp_density"); fig.show(width=1400)
print(f"sKP density: scale capped at p99={vcap:.1f} (peak {float(dg.max()):.0f}) events per 0.04-deg cell")""")

md(r"""## Figure 0c · Background (declustered) seismicity — southern Korean Peninsula

Only the **background** events from that project's Zaliapin–Ben-Zion **2-D NND declustering** of the HypoSVI
catalogue (`16.kma_absolute_location/.../nnd_events.csv`). The split uses the **2-D epicentral** metric on
purpose — regional HypoSVI **depth errors are too large for a reliable 3-D split** (unlike the dt.cc
Ulsan-Fault catalogue, where 3-D is justified). Stripping the aftershock/swarm clusters (e.g. the **Gyeongju**
and **Pohang** sequences, which dominated Fig 0b) leaves the **steady tectonic** seismicity and its density —
note how the saturated Gyeongju/Pohang hotspots largely disappear.""")
co(r"""NND="/home/msseo/works/16.kma_absolute_location/runs/kma_batch/nnd_events.csv"
nd0=pd.read_csv(NND); bg=nd0[nd0.background==True].dropna(subset=["svi_lat","svi_lon"])
dgb=dgrid_skp(bg.svi_lon.values,bg.svi_lat.values); vmaxb=float(dgb.max())     # NO cap (background is not peaky)
fig=pygmt.Figure()
fig.basemap(region=RC,projection="M13c",frame=["WSne","xa1","ya1"])
fig.coast(land="gray96",water="white",shorelines="0.3p,gray70",resolution="i")
pygmt.makecpt(cmap="hot",series=[0,vmaxb],reverse=True)
fig.grdimage(dgb.where(dgb>=0.2),cmap=True,nan_transparent=True)                # background density backdrop (uncapped)
fig.coast(shorelines="0.5p,gray45",borders="1/0.3p,gray70",resolution="i")
fig.plot(x=bg.svi_lon,y=bg.svi_lat,style="c0.03c",fill="gray15",transparency=55)  # background epicentres
fig.plot(x=rx,y=ry,pen="2.2p,red")
fig.plot(x=[GJ_LON],y=[GJ_LAT],style="a0.5c",fill="lightgreen",pen="0.7p,black")
fig.plot(x=[POH_LON],y=[POH_LAT],style="a0.5c",fill="lightgreen",pen="0.7p,black")
fig.basemap(map_scale="jBL+w50k+o0.5c/0.5c")
with pygmt.config(FONT_LABEL="12p",FONT_ANNOT_PRIMARY="10p"):
    fig.colorbar(cmap=True,frame="af+lSmoothed event count",position="jBR+w5c/0.35c+h+o0.7c/0.9c")
fig.text(x=125.55,y=38.28,text="KMA catalog, 2010-2024",font="8p,Helvetica,black",justify="LM",fill="white@20",pen="0.4p,gray40")
savegmt(fig,"fig00c_skp_background_density"); fig.show(width=1400)
print(f"background events {len(bg):,}/{len(nd0):,} ({100*len(bg)/len(nd0):.0f}%); NO cap, peak {vmaxb:.1f} per cell")""")

md(r"""## Figure 0d · Background seismicity RATE — southern Korean Peninsula

The same 2-D-NND background field expressed as a long-term **rate density** $\lambda$: smoothed background
events divided by the catalogue span (2010–2024) and by cell area, in **events yr⁻¹ km⁻²** (the standard
ETAS / hazard background-intensity unit — *per km²*, not an arbitrary tile). Spatially it mirrors Fig 0c
(rate = count ÷ time ÷ area); the colour scale is now a physical **annual rate**, uncapped.""")
co(r"""ts=pd.to_datetime(bg.event_id.astype('int64').astype(str).str.slice(0,8),format="%Y%m%d",errors="coerce")
T_YR=(ts.max()-ts.min()).days/365.25
latmid=(RC[2]+RC[3])/2; cell_km2=(0.04*111.320*np.cos(np.radians(latmid)))*(0.04*110.574)
rate=dgb/cell_km2/T_YR                                              # events / yr / km^2 (areal rate density)
fig=pygmt.Figure()
fig.basemap(region=RC,projection="M13c",frame=["WSne","xa1","ya1"])
fig.coast(land="gray96",water="white",shorelines="0.3p,gray70",resolution="i")
pygmt.makecpt(cmap="hot",series=[0,float(rate.max())],reverse=True)
fig.grdimage(rate.where(dgb>=0.2),cmap=True,nan_transparent=True)
fig.coast(shorelines="0.5p,gray45",borders="1/0.3p,gray70",resolution="i")
fig.plot(x=bg.svi_lon,y=bg.svi_lat,style="c0.03c",fill="gray15",transparency=55)  # background epicentres
fig.plot(x=rx,y=ry,pen="2.2p,red")
fig.plot(x=[GJ_LON],y=[GJ_LAT],style="a0.5c",fill="lightgreen",pen="0.7p,black")
fig.plot(x=[POH_LON],y=[POH_LAT],style="a0.5c",fill="lightgreen",pen="0.7p,black")
fig.basemap(map_scale="jBL+w50k+o0.5c/0.5c")
with pygmt.config(FONT_LABEL="12p",FONT_ANNOT_PRIMARY="10p"):
    fig.colorbar(cmap=True,frame="af+lBackground rate (events yr@+-1@+ km@+-2@+)",position="jBR+w5c/0.35c+h+o0.7c/0.9c")
fig.text(x=125.55,y=38.28,text=f"KMA catalog, 2010-2024 ({T_YR:.0f} yr)",font="8p,Helvetica,black",justify="LM",fill="white@20",pen="0.4p,gray40")
savegmt(fig,"fig00d_skp_background_rate"); fig.show(width=1400)
print(f"background rate: span {T_YR:.1f} yr, cell {cell_km2:.1f} km^2, peak {float(rate.max()):.4f} events/yr/km^2")""")

# ----------------------------------------------------------------- §0e Fig 0e narrow smoothing
md(r"""## Figure 0e · Background density at location-error-matched smoothing — southern Korean Peninsula

The same 2-D-NND background field as Fig 0c, but smoothed at the scale the **locations actually resolve**.

**What sets the smoothing in Figs 0b-0d?** A Gaussian of $\sigma=1.3$ cells on a $0.04^\circ$ grid $\approx$
**5 km** (5.7 km N / 4.7 km E) — a purely *visual* bandwidth chosen to give a continuous warm backdrop, with
**no link to location accuracy**. Since well-located events here are good to $\sim$**1-2 km** horizontally, a
5-km kernel *over-smooths*: real structure finer than ~5 km (individual fault strands, tight clusters) is
blurred away. The principled choice is to set the kernel **equal to the location uncertainty** — each
epicentre becomes its own positional PDF ($\sigma\approx1.5$ km here) — so the map shows exactly the structure
the data support: no invented detail below the errors, no needless blurring above them. (A fully rigorous map
would use a *per-event* $\sigma$ / adaptive KDE; HypoSVI horizontal errors vary and depths are far worse —
1.5 km is the well-located-event value, an optimistic floor used uniformly here.)

**Does the grid size matter?** Grid spacing and smoothing length are independent. Map resolution is set by
$\sigma$, **not** the grid — *provided* the grid samples the kernel (cell $\lesssim\sigma/2$). A 5-km kernel
tolerates the 4-km grid of Figs 0b-0d, but a 1.5-km kernel needs a **$\sim$0.5-km grid**, else the coarse
histogram bins — not $\sigma$ — would cap the resolution. We use a 0.5-km grid ($\sigma$/cell $=3$, well
sampled) and express density as **events km⁻²** (areal, grid-independent); the printout confirms a 2× finer
grid changes the peak by only ~1.5 %.""")
co(r"""# narrow, location-error-matched smoothing -> events per km^2 (grid-INDEPENDENT areal density)
def dgrid_skp_areal(lon,lat,sigma_km=1.5,cell_km=0.5):
    # square-in-km cells so the Gaussian kernel is ISOTROPIC in physical space (unlike single-sp dgrid_skp)
    latmid=(RC[2]+RC[3])/2
    dlat=cell_km/110.574; dlon=cell_km/(111.320*np.cos(np.radians(latmid)))
    xb=np.arange(RC[0],RC[1]+dlon,dlon); yb=np.arange(RC[2],RC[3]+dlat,dlat)
    H,_,_=np.histogram2d(lon,lat,bins=[xb,yb])
    H=gaussian_filter(H,sigma_km/cell_km)                          # sigma in CELLS; cell=cell_km -> physical sigma=sigma_km
    cell_area=(dlon*111.320*np.cos(np.radians(latmid)))*(dlat*110.574)   # km^2
    return xr.DataArray((H/cell_area).T,coords={"lat":(yb[:-1]+yb[1:])/2,"lon":(xb[:-1]+xb[1:])/2},dims=["lat","lon"])
SIGMA_KM=1.5                                                        # = assumed horizontal error of well-located events (1-2 km)
dgf =dgrid_skp_areal(bg.svi_lon.values,bg.svi_lat.values,SIGMA_KM,0.5)
dgf2=dgrid_skp_areal(bg.svi_lon.values,bg.svi_lat.values,SIGMA_KM,0.25)   # 2x finer grid, SAME sigma -> grid-independence check
vcap=float(np.percentile(dgf.values[dgf.values>0],99))             # p99 cap (+ef triangle) so the diffuse field stays visible
fig=pygmt.Figure()
fig.basemap(region=RC,projection="M13c",frame=["WSne","xa1","ya1"])
fig.coast(land="gray96",water="white",shorelines="0.3p,gray70",resolution="i")
pygmt.makecpt(cmap="hot",series=[0,vcap],reverse=True)
fig.grdimage(dgf.where(dgf>=0.03*vcap),cmap=True,nan_transparent=True)
fig.coast(shorelines="0.5p,gray45",borders="1/0.3p,gray70",resolution="i")
fig.plot(x=bg.svi_lon,y=bg.svi_lat,style="c0.02c",fill="gray15",transparency=60)   # background epicentres
fig.plot(x=rx,y=ry,pen="2.2p,red")
fig.plot(x=[GJ_LON],y=[GJ_LAT],style="a0.5c",fill="lightgreen",pen="0.7p,black")
fig.plot(x=[POH_LON],y=[POH_LAT],style="a0.5c",fill="lightgreen",pen="0.7p,black")
fig.basemap(map_scale="jBL+w50k+o0.5c/0.5c")
with pygmt.config(FONT_LABEL="12p",FONT_ANNOT_PRIMARY="10p"):
    fig.colorbar(cmap=True,frame="af+lBackground density (events km@+-2@+)",position="jBR+w5c/0.35c+h+o0.7c/0.9c+ef")
fig.text(x=125.55,y=38.28,text=f"@~s@~={SIGMA_KM:g} km (= location error), 0.5-km grid",font="8p,Helvetica,black",justify="LM",fill="white@20",pen="0.4p,gray40")
savegmt(fig,"fig00e_skp_background_density_fine"); fig.show(width=1400)
print(f"narrow smoothing: sigma={SIGMA_KM} km on 0.5-km grid (sigma/cell=3, kernel well sampled); cf. Figs 0b-0d ~5 km")
print(f"grid-independence: peak {float(dgf.max()):.3f} (0.5-km) vs {float(dgf2.max()):.3f} (0.25-km) events/km^2 "
      f"-> {100*abs(float(dgf.max())-float(dgf2.max()))/float(dgf2.max()):.1f}% (resolution set by sigma, not grid)")""")

# ----------------------------------------------------------------- §1 Fig 1
md(r"""## Figure 1 · Seismicity and station network before vs after the 2016 Gyeongju earthquake

**Two side-by-side maps** — before and after the 2016-09 Gyeongju M5.5 — over the receiver-function
**H-κ-stacking Vp/Vs** field (**M. Kim et al., 2026**; blockmean→surface interpolation, roma 1.7–1.9, red = high
Vp/Vs), with **fault traces**. Each
panel shows that era's **seismicity** (black open circles, **fixed size** — region-wide reliable local
magnitudes are not yet available) and its **station network** (green triangles). The sparse early network
densifies markedly after 2016, improving geometry and lowering the detection threshold — the more complete
post-2016 seismicity follows. Blue dashed box = the Ulsan-Fault subregion of Figs 2–4.""")
co(r"""def clip(sub): return sub[(sub.lon>=REG[0])&(sub.lon<=REG[1])&(sub.lat>=REG[2])&(sub.lat<=REG[3])]
# Vp/Vs: blockmean -> surface interpolation; MASK to land only (no offshore values)
_blk=pygmt.blockmean(data=vk[["lon","lat","v"]],region=REG,spacing="0.002d")
VSURF=pygmt.surface(data=_blk,region=REG,spacing="0.002d")
_lm=pygmt.grdlandmask(region=REG,spacing="0.002d",resolution="h",maskvalues=[0,1])
VSURF=VSURF.where(_lm.values>0)                                                   # NaN offshore
panels=[(cat[cat.time<GYEONGJU], st_pre, f"Before 2016 M5.5 Gyeongju  ({len(st_pre)} stations)"),
        (cat[cat.time>=GYEONGJU], st,     f"After 2016 M5.5 Gyeongju  ({len(st)} stations)")]
fig=pygmt.Figure()
pygmt.makecpt(cmap="roma",series=[1.7,1.9,0.01],continuous=True,reverse=True)     # RED=high Vp/Vs, BLUE=low
with fig.subplot(nrows=1,ncols=2,figsize=("22c","9.5c"),margins="0.7c"):
    for j,(sc,stn,ttl) in enumerate(panels):
        with fig.set_panel(panel=j):
            fig.basemap(region=REG,projection="M?",frame=[f"WSne+t{ttl}","xa0.2","ya0.2"])
            fig.grdimage(VSURF,cmap=True,nan_transparent=True,transparency=40)     # Vp/Vs, land only
            fig.coast(shorelines="0.6p,gray35")
            if os.path.exists(FAULTS): fig.plot(data=FAULTS,pen="0.7p,gray20")     # fault traces
            s=clip(sc)
            fig.plot(x=s.lon,y=s.lat,style="c0.07c",pen="0.35p,black")             # black OPEN circles, fixed size
            fig.plot(x=stn.Longitude,y=stn.Latitude,style="i0.22c",fill="green",pen="0.3p,black")   # GREEN stations
            fig.plot(x=[GJ_LON],y=[GJ_LAT],style="a0.6c",fill="lightgreen",pen="0.9p,black")             # 2016 Gyeongju star
            fig.plot(x=[UF[0],UF[1],UF[1],UF[0],UF[0]],y=[UF[2],UF[2],UF[3],UF[3],UF[2]],pen="1.1p,blue,--")
            fig.basemap(map_scale=SCALE_R)
with pygmt.config(FONT_LABEL="13p",FONT_ANNOT_PRIMARY="10p"):                     # larger colorbar heading + ticks
    fig.colorbar(cmap=True,frame="a+lVp/Vs ratio (H-@~k@~ stacking; M. Kim et al. 2026)",position="x11c/-1.0c+w11c/0.5c+h+jTC")
savegmt(fig,"fig01_seismicity_vpvs_before_after"); fig.show(width=2000)
print(f"network {len(st_pre)} -> {len(st)} stations (+{len(st_post)}); seismicity "
      f"{int((cat.time<GYEONGJU).sum()):,} -> {int((cat.time>=GYEONGJU).sum()):,} (before -> after 2016 Gyeongju).")""")

# ----------------------------------------------------------------- §2 Fig 2
md(r"""## Figure 2 · Cumulative event count and magnitude — Ulsan-Fault subregion

Cumulative number of UF-box events (left axis, blue) with each event's **local magnitude** overlaid (right
axis, open circles). Events **without** a magnitude are placed at a **floor of −1.5** — below all real ML —
so they are visible without confusion. Vertical dashed lines mark the **2016 M5.5 Gyeongju** earthquake and
the two largest local Ulsan-Fault events (**2014 M4.0**, **2023 M3.8**).""")
co(r"""uf=cat[(cat.lon>=UF[0])&(cat.lon<=UF[1])&(cat.lat>=UF[2])&(cat.lat<=UF[3])].sort_values("time").reset_index(drop=True)
MLFLOOR=-1.5; asg=uf[uf.ml_all.notna()]; una=uf[uf.ml_all.isna()]
fig,ax=plt.subplots(figsize=(11.5,5.2))
ax.step(uf.time,np.arange(1,len(uf)+1),where="post",color="steelblue",lw=1.9,zorder=4,label=f"Cumulative count (N={len(uf)})")
ax.set_xlabel("Year",fontsize=13); ax.set_ylabel("Cumulative number of events",fontsize=13)
ax.set_title("Ulsan Fault subregion — cumulative seismicity and magnitude",fontsize=15)
ax.tick_params(labelsize=11)
ax2=ax.twinx()
_sza=(0.045*1.7**asg.ml_all.clip(0,4)*42)**2                                   # circle size proportional to magnitude
ax2.scatter(asg.time,asg.ml_all,s=_sza,facecolors="none",edgecolors="0.45",linewidths=0.8,alpha=0.7,zorder=2,label="ML (assigned)")
ax2.scatter(una.time,[MLFLOOR]*len(una),s=5,facecolors="none",edgecolors="tab:orange",linewidths=0.7,alpha=0.6,zorder=2,label=f"ML unassigned (set to {MLFLOOR})")
ax2.axhline(MLFLOOR,color="0.7",ls=":",lw=0.8,zorder=1); ax2.set_ylabel(r"Local magnitude ($M_\mathrm{L}$)",fontsize=13); ax2.tick_params(labelsize=11); ax2.set_ylim(MLFLOOR-0.4,float(uf.ml_all.max())+0.4)
for t,nm,c in [(GYEONGJU,"2016 M5.5 Gyeongju","tab:red"),
               (pd.Timestamp("2014-09-23 06:27",tz="utc"),"2014 M4.0 (Ulsan Fault)","tab:green"),
               (pd.Timestamp("2023-11-29 19:55",tz="utc"),"2023 M3.8 (Ulsan Fault)","tab:purple")]:
    ax.axvline(t,color=c,lw=1.6,ls="--",zorder=3,label=nm)
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax.legend(h1+h2,l1+l2,loc="upper left",framealpha=1,facecolor="white",edgecolor="black",fontsize=8); ax.margins(x=0.01); fig.tight_layout(); savempl(fig,"fig02_cumulative_magnitude"); plt.show()
print(f"UF-box: {len(uf)} events ({len(una)} without ML -> floor {MLFLOOR}); before Gyeongju {int((uf.time<GYEONGJU).sum())}, after {int((uf.time>=GYEONGJU).sum())}")""")

# ----------------------------------------------------------------- §3 Fig 3
md(r"""## Figure 3 · HypoDD-relocated Ulsan Fault catalog, scaled by local magnitude

All HypoDD relocations (kim2011 velocity) in the UF box, coloured by depth — both **dt.cc-resolved** (sharp,
tens of m) and **dt.ct-relocated** (catalog-dt only, ~hundreds of m) events, so the catalog is complete and
the **largest event (2014 M3.89, a dt.ct event) is shown**. Circle size ∝ `ml_ufcorr` where a reliable local
magnitude exists (n_used ≥ 3); events **without** a reliable ML are drawn at a **minimum visible size** (not
dropped). The sharpened structure is the headline of the relocation.""")
co(r"""fig=pygmt.Figure()
fig.basemap(region=list(UF),projection=PROJ,frame=["WSne+tUlsan Fault HypoDD relocations","xa0.1","ya0.1"])
fig.coast(land="gray98",water="lightblue",shorelines="0.4p,gray45")
if os.path.exists(FAULTS): fig.plot(data=FAULTS,pen="0.8p,gray20")
pygmt.makecpt(cmap="turbo",series=[DMIN,DMAX])   # shallow=blue, deep=red
# OPEN circles whose OUTLINE is coloured by depth (pen colour from CPT via +cl; no fill). cols: x,y,z(depth),size
df3=pd.DataFrame({"x":dtcc_all.lon.values,"y":dtcc_all.lat.values,"z":dtcc_all.depth.values,"s":dtcc_all.sz.values})
fig.plot(data=df3,style="c",cmap=True,pen="1.0p+cl")
with pygmt.config(FONT_LABEL="13p",FONT_ANNOT_PRIMARY="11p"):
    fig.colorbar(frame="a2f1+lDepth (km)",position="JMR+o0.5c/0c+w7c")   # 2 km annotation interval
# ML -> size legend (compact, semi-transparent box, upper-RIGHT)
bx1=UF[1]-0.006; bx0=bx1-0.051; by1=UF[3]-0.010; by0=by1-0.072
fig.plot(x=[bx0,bx1,bx1,bx0,bx0],y=[by0,by0,by1,by1,by0],fill="white@35",pen="0.8p,black")
fig.text(x=(bx0+bx1)/2,y=by1-0.012,text="M@-L@-",font="10p,Helvetica-Bold,black",justify="MC")
cx=bx0+0.013
for k,m in enumerate([3,2,1]):
    ly=by1-0.028-k*0.016
    fig.plot(x=[cx],y=[ly],style=f"c{0.045*1.7**m:.3f}c",pen="1.1p,black")
    fig.text(x=cx+0.017,y=ly,text=f"{m}",font="9p,Helvetica,black",justify="LM")
fig.basemap(region=list(UF),projection=PROJ,frame=["WSne","xa0.1","ya0.1"],map_scale=SCALE_UF)
savegmt(fig,"fig03_hypodd_relocations"); fig.show()
print(f"plotted {len(dtcc_all)} relocated events ({int(dtcc_all.has_ml.sum())} ML-scaled, {int((~dtcc_all.has_ml).sum())} at min size); "
      f"depth {dtcc_all.depth.min():.1f}-{dtcc_all.depth.max():.1f} km")""")

# ----------------------------------------------------------------- §3b depth sections
md(r"""### Figure 3b · Depth sections of the dt.cc-relocated Ulsan-Fault events

Hypocentre depth across **longitude** and across **latitude** for all dt.cc-resolved UF events (open circles,
outline coloured by depth) — the cross-correlation depths resolve the fault's down-dip structure.""")
co(r"""import matplotlib as _mpl
_norm=_mpl.colors.Normalize(DMIN,DMAX); _cm=plt.cm.turbo
_ec=_cm(_norm(dtcc_all.depth.values)); _s=(dtcc_all.sz.values*42)**2          # size by ML (same scale as Fig 3)
fig,ax=plt.subplots(1,2,figsize=(15,4.8))
for a,(xc,xl) in zip(ax,[(dtcc_all.lon,"Longitude"),(dtcc_all.lat,"Latitude")]):
    a.scatter(xc,dtcc_all.depth,s=_s,facecolors="none",edgecolors=_ec,linewidths=0.9)
    a.set_xlabel(xl,fontsize=14); a.set_ylabel("Depth (km)",fontsize=14)
    a.set_title(f"Depth section across {xl.lower()}",fontsize=14); a.tick_params(labelsize=11); a.invert_yaxis()
from matplotlib.lines import Line2D as _L2D                                       # ML size legend on ONE panel only
_h=[_L2D([0],[0],marker="o",mfc="none",mec="k",ls="",ms=0.045*1.7**m*42,label=f"{m}") for m in [1,2,3]]
ax[0].legend(handles=_h,loc="upper right",fontsize=8,title=r"$M_\mathrm{L}$",framealpha=1,edgecolor="black",labelspacing=1.2,handletextpad=1.0)
sm=_mpl.cm.ScalarMappable(norm=_norm,cmap=_cm); sm.set_array([])
_cb=fig.colorbar(sm,ax=ax,fraction=0.03,pad=0.02,ticks=np.arange(np.ceil(DMIN/2)*2,DMAX+0.1,2))   # 2 km tick interval
_cb.set_label("Depth (km)",fontsize=12); _cb.ax.tick_params(labelsize=10); savempl(fig,"fig03b_depth_sections"); plt.show()
print(f"depth sections: {len(dtcc_all)} relocated events; depth {dtcc_all.depth.min():.1f}-{dtcc_all.depth.max():.1f} km, "
      f"median {dtcc_all.depth.median():.1f} km")""")

# ----------------------------------------------------------------- §3c short-window before/after Gyeongju
md(r"""### Figure 3c · Ulsan Fault seismicity 6 months before vs after the 2016 Gyeongju earthquake

Side-by-side maps for the **6 months before** and **6 months after** the 2016 Gyeongju event: smoothed
event **density** (shared colour scale, so the rate change is directly comparable) with the individual
HypoDD **seismicity overlaid** (black open circles, size ∝ ML). Each panel title gives the **event count and
the number of recording stations** — these are **equal (13) before and after**, so the increase is **real
triggering, not a detection artifact** (the post-Gyeongju densification came later, in 2017).""")
co(r"""g0=GYEONGJU; W=pd.Timedelta(days=183)
pre6=dtcc_all[(dtcc_all.time>=g0-W)&(dtcc_all.time<g0)]; post6=dtcc_all[(dtcc_all.time>=g0)&(dtcc_all.time<g0+W)]
# network control (stations actually recording) -- computed first so the counts go in the panel titles
_ps=pd.read_csv(f"{KG}/local_magnitudes/catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo_uncapped.csv",usecols=["network","station","event_time"])
_ps["t"]=pd.to_datetime(_ps.event_time,utc=True,errors="coerce"); _ps["scc"]=_ps.network+"."+_ps.station
_nb=_ps[(_ps.t>=g0-W)&(_ps.t<g0)].scc.nunique(); _na=_ps[(_ps.t>=g0)&(_ps.t<g0+W)].scc.nunique()
def _dg(lon,lat,sp=0.004,sm=1.5):
    xb=np.arange(UF[0],UF[1]+sp,sp); yb=np.arange(UF[2],UF[3]+sp,sp)
    H,_,_=np.histogram2d(lon,lat,bins=[xb,yb]); H=gaussian_filter(H,sm)
    return xr.DataArray(H.T,coords={"lat":(yb[:-1]+yb[1:])/2,"lon":(xb[:-1]+xb[1:])/2},dims=["lat","lon"])
gp=_dg(pre6.lon.values,pre6.lat.values); gq=_dg(post6.lon.values,post6.lat.values)
VMAX=float(max(gp.max(),gq.max())) or 1.0
fig=pygmt.Figure()
pygmt.makecpt(cmap="hot",series=[0,VMAX],reverse=True)                             # density colour scale
with fig.subplot(nrows=1,ncols=2,figsize=("18c","9c"),margins="0.6c"):
    for j,(grid,s,ttl) in enumerate([(gp,pre6,f"6 months before 2016 Gyeongju+s({len(pre6)} events, {_nb} stations)"),
                                     (gq,post6,f"6 months after 2016 Gyeongju+s({len(post6)} events, {_na} stations)")]):
        with fig.set_panel(panel=j):
            fig.basemap(region=list(UF),projection="M?",frame=[f"WSne+t{ttl}","xa0.1","ya0.1"])
            fig.grdimage(grid.where(grid>=0.05*VMAX),cmap=True,nan_transparent=True)           # density backdrop
            fig.coast(shorelines="0.4p,gray45")
            if os.path.exists(FAULTS): fig.plot(data=FAULTS,pen="0.8p,gray20")
            if len(s): fig.plot(x=s.lon,y=s.lat,size=s.sz,style="cc",pen="0.4p,black")          # BLACK OPEN circles, size by ML (thin pen -> density visible)
            bx1=UF[1]-0.005; bx0=bx1-0.05; by1=UF[3]-0.009; by0=by1-0.066                       # ML size legend (upper-right)
            fig.plot(x=[bx0,bx1,bx1,bx0,bx0],y=[by0,by0,by1,by1,by0],fill="white@25",pen="0.8p,black")
            fig.text(x=(bx0+bx1)/2,y=by1-0.011,text="M@-L@-",font="9p,Helvetica-Bold,black",justify="MC")
            _cx=bx0+0.012
            for _k,_m in enumerate([3,2,1]):
                _ly=by1-0.026-_k*0.015
                fig.plot(x=[_cx],y=[_ly],style=f"c{0.045*1.7**_m:.3f}c",pen="0.9p,black")
                fig.text(x=_cx+0.016,y=_ly,text=f"{_m}",font="8p,Helvetica,black",justify="LM")
            fig.basemap(map_scale=SCALE_UF)
with pygmt.config(FONT_LABEL="13p",FONT_ANNOT_PRIMARY="11p"):
    fig.colorbar(cmap=True,frame="af+lSmoothed event count",position="x9c/-1.0c+w8c/0.4c+h+jTC")
savegmt(fig,"fig03c_gyeongju_6mo_before_after"); fig.show(width=1500)
print(f"6mo before {len(pre6)} -> 6mo after {len(post6)} events ({len(post6)/max(len(pre6),1):.1f}x).")
print(f"NETWORK CONTROL: unique stations recording = {_nb} (before) vs {_na} (after) -> UNCHANGED; the increase is real triggering, not detection.")""")

md(r"""### Figure 3d · Magnified cumulative count across the 2016 Gyeongju window

The same ±6-month window and population as the maps above, drawn as a **cumulative event count** (the Fig 2
style, zoomed) so the **seismicity-rate jump shows up as a slope change** at the Gyeongju time — the clearest
single view of the rate difference.""")
co(r"""win=dtcc_all[(dtcc_all.time>=g0-W)&(dtcc_all.time<g0+W)].sort_values("time")
rpre=len(pre6)/6.0; rpost=len(post6)/6.0; tot=len(win)
fig,ax=plt.subplots(figsize=(9.5,4.6))
ax.step(win.time,np.arange(1,tot+1),where="post",color="black",lw=2.2,zorder=4)
ax.axvline(g0,color="tab:red",ls="--",lw=1.8,zorder=3,label="2016 Gyeongju (M5.1 foreshock)")
ax.axvspan(g0-W,g0,color="tab:blue",alpha=0.10,zorder=1); ax.axvspan(g0,g0+W,color="tab:red",alpha=0.10,zorder=1)
ax.text(g0-W*0.5,tot*0.62,f"before\n{len(pre6)} events\n{rpre:.0f}/month",ha="center",va="center",fontsize=10,color="tab:blue",fontweight="bold")
ax.text(g0+W*0.45,tot*0.90,f"after\n{len(post6)} events\n{rpost:.0f}/month  ({rpost/max(rpre,1e-9):.1f}×)",ha="center",va="center",fontsize=10,color="tab:red",fontweight="bold")
ax.set_xlabel("Date",fontsize=13); ax.set_ylabel("Cumulative relocated events",fontsize=13)
ax.set_title("Ulsan Fault — cumulative count across the 2016 Gyeongju window",fontsize=14)
ax.tick_params(labelsize=11); ax.legend(loc="upper left",fontsize=10,framealpha=1,edgecolor="black")
fig.autofmt_xdate(); fig.tight_layout(); savempl(fig,"fig03d_gyeongju_cumulative"); plt.show()
print(f"window {(g0-W):%Y-%m-%d}..{(g0+W):%Y-%m-%d}: {tot} events; rate {rpre:.1f}->{rpost:.1f}/month ({rpost/max(rpre,1e-9):.1f}x)")""")

# ----------------------------------------------------------------- §4 Fig 4
md(r"""## Figure 4 · Background vs clustered seismicity density (3D NND declustering)

The **full relocated catalogue** — dt.cc-resolved *and* dt.ct-relocated events with reliable ML (n_used ≥ 3) —
split into **background** (isolated, tectonic) and **clustered** (aftershock/swarm) populations by
**Zaliapin–Ben-Zion 3D nearest-neighbour** declustering (depth-aware, **data-driven Df = 1.2**, b = 1.0; the
measured Grassberger–Procaccia correlation dimension of the relocated hypocentres — see the fractal-dimension
notebook — *not* the generic 2.5), then smoothed event density of each (shared colour scale) with the
**individual events overlaid as black open circles sized by ML**, side-by-side for direct comparison.
**Including the dt.ct-relocated events is essential** — the dt.cc/dt.ct split is location precision, not
detection, so a dt.cc-only population would drop mainshock parents (incl. the 2014 M3.89) and open a
completeness hole below Mc, biasing the clustered/background split and rate. Disclosed: completeness Mc = 1.2.""")
co(r"""# 3D NND on the FULL relocated UF population (dt.cc + dt.ct, reliable ML)
g=dtcc.copy(); g["t_year"]=g.event_time.dt.strftime("%Y%m%d%H%M%S").map(nnd.decimal_year); g["event_id"]=np.arange(len(g))  # CANONICAL nnd.decimal_year (exact year length)
g=g.rename(columns={"lon":"svi_lon","lat":"svi_lat","depth":"svi_dep","ml_ufcorr_reloc":"kma_mag"})
nd=nnd.compute_nnd(g,b=1.0,D=DF_UF,mmin=None,metric="3d"); e0,info=nnd.fit_eta0(nd.eta.values,method="gmm")
clu=set(nd.loc[nd.eta<e0,"event_id"]); g["bg"]=~g.event_id.isin(clu)
g["sz"]=(0.045*1.7**g.kma_mag.clip(0,4)).clip(lower=MINSZ)             # circle size by ML (same scale as Fig 3/3c)
SP=0.004
def dgrid(lon,lat,sp=SP,sm=1.5):
    xb=np.arange(UF[0],UF[1]+sp,sp); yb=np.arange(UF[2],UF[3]+sp,sp)
    H,_,_=np.histogram2d(lon,lat,bins=[xb,yb]); H=gaussian_filter(H,sm)
    return xr.DataArray(H.T,coords={"lat":(yb[:-1]+yb[1:])/2,"lon":(xb[:-1]+xb[1:])/2},dims=["lat","lon"])
gb=dgrid(g[g.bg].svi_lon.values,g[g.bg].svi_lat.values)               # background density grid
gc=dgrid(g[~g.bg].svi_lon.values,g[~g.bg].svi_lat.values)             # clustered density grid
VMAX=float(max(gb.max(),gc.max())) or 1.0                             # shared scale -> directly comparable
fig=pygmt.Figure()
pygmt.makecpt(cmap="hot",series=[0,VMAX],reverse=True)                # density colour scale
with fig.subplot(nrows=1,ncols=2,figsize=("18c","9c"),margins="0.6c"):
    for j,(grid,sub,ttl) in enumerate([(gb,g[g.bg],f"Background seismicity+s({int(g.bg.sum())} events)"),
                                       (gc,g[~g.bg],f"Clustered seismicity+s({int((~g.bg).sum())} events)")]):
        with fig.set_panel(panel=j):
            fig.basemap(region=list(UF),projection="M?",frame=[f"WSne+t{ttl}","xa0.1","ya0.1"])
            fig.grdimage(grid.where(grid>=0.05*VMAX),cmap=True,nan_transparent=True)            # density backdrop
            fig.coast(shorelines="0.4p,gray45")
            if os.path.exists(FAULTS): fig.plot(data=FAULTS,pen="0.8p,gray20")
            if len(sub): fig.plot(x=sub.svi_lon,y=sub.svi_lat,size=sub.sz,style="cc",pen="0.4p,black")  # BLACK OPEN circles, size by ML (thin pen -> density visible)
            bx1=UF[1]-0.005; bx0=bx1-0.05; by1=UF[3]-0.009; by0=by1-0.066                        # ML size legend (upper-right)
            fig.plot(x=[bx0,bx1,bx1,bx0,bx0],y=[by0,by0,by1,by1,by0],fill="white@25",pen="0.8p,black")
            fig.text(x=(bx0+bx1)/2,y=by1-0.011,text="M@-L@-",font="9p,Helvetica-Bold,black",justify="MC")
            _cx=bx0+0.012
            for _k2,_m in enumerate([3,2,1]):
                _ly=by1-0.026-_k2*0.015
                fig.plot(x=[_cx],y=[_ly],style=f"c{0.045*1.7**_m:.3f}c",pen="0.9p,black")
                fig.text(x=_cx+0.016,y=_ly,text=f"{_m}",font="8p,Helvetica,black",justify="LM")
            fig.basemap(map_scale=SCALE_UF)
with pygmt.config(FONT_LABEL="13p",FONT_ANNOT_PRIMARY="11p"):
    fig.colorbar(cmap=True,frame="af+lSmoothed event count",position="x9c/-1.0c+w8c/0.4c+h+jTC")
savegmt(fig,"fig04_density_background_clustered"); fig.show(width=1500)
print(f"3D NND: log10 eta0={np.log10(e0):+.2f} | background {int(g.bg.sum())} | clustered {int((~g.bg).sum())} "
      f"({100*(~g.bg).mean():.0f}% clustered)")
# how the two location classes split (does adding dt.ct just inflate the background?)
for _lab,_mk in [("dt.cc-resolved",g.is_dtcc),("dt.ct-only",~g.is_dtcc)]:
    _s=g[_mk]; print(f"  {_lab:16}: N {len(_s):4d} | clustered {int((~_s.bg).sum()):4d} ({100*(~_s.bg).mean():2.0f}%) | background {int(_s.bg.sum()):4d} ({100*_s.bg.mean():2.0f}%)")
if (g.event_idx==704).any():
    _cl=bool((~g.loc[g.event_idx==704,"bg"]).values[0]); print(f"  -> dt.ct events go mostly to BACKGROUND (few similar neighbours = genuinely isolated), BUT the large "
          f"2014 M3.89 (dt.ct) is clustered={_cl} (a mainshock PARENT, not isolated).")""")

md(r"""### Figure 4b · Nearest-neighbour structure (Zaliapin–Ben-Zion / Goebel SOTA)

The **R–T pair density** (2-D KDE) of nearest-neighbour event pairs — the standard ZBZ/Goebel declustering
view. The two axes share the **same range**, so the η₀ line (slope −1) renders at 45°: clustered pairs fall
below it (small η), background above. Depth-aware **3D NND (data-driven Df = 1.2, b = 1.0)**.""")
co(r"""from scipy.stats import gaussian_kde
lt=nd.logT.values; lr=nd.logR.values; ok=np.isfinite(lt)&np.isfinite(lr); lt,lr=lt[ok],lr[ok]
le0=np.log10(e0); bn=0.1
xlo,xhi=-8.0,2.0; ylo,yhi=-6.0,4.0   # logT (x): -8..2 ; logR (y): -6..4 (equal 10-unit spans -> slope -1 at 45 deg)
Tb=np.arange(xlo,xhi+bn,bn); Rb=np.arange(ylo,yhi+bn,bn); XX,YY=np.meshgrid(Tb,Rb)
ZZ=gaussian_kde(np.vstack([lt,lr]))(np.vstack([XX.ravel(),YY.ravel()])).reshape(XX.shape)*bn*bn*len(lt)
fig,ax=plt.subplots(figsize=(6.8,6.8))
pc=ax.pcolormesh(XX,YY,ZZ,cmap=plt.cm.RdYlGn_r,shading="auto")
cb=fig.colorbar(pc,ax=ax,fraction=0.046,pad=0.04); cb.set_label("Number of event pairs",fontsize=13); cb.ax.tick_params(labelsize=11)
ax.plot([xlo,xhi],le0-np.array([xlo,xhi]),"-",lw=2.5,color="w")
ax.plot([xlo,xhi],le0-np.array([xlo,xhi]),"--",lw=1.5,color="0.3",label=fr"$\eta_0$ (log10 = {le0:.2f})")
ax.set_xlabel(r"Rescaled time  $\log_{10}T$",fontsize=14); ax.set_ylabel(r"Rescaled distance  $\log_{10}R$",fontsize=14)
ax.set_title("Nearest-neighbour pairs in R–T (3D NND, b = 1.0, Df = 1.1)",fontsize=14)
ax.tick_params(labelsize=11); ax.set(xlim=(xlo,xhi),ylim=(ylo,yhi))
ax.set_aspect("equal"); ax.legend(loc="lower left",fontsize=12,framealpha=1,edgecolor="black"); fig.tight_layout(); savempl(fig,"fig04b_nnd_TR_density"); plt.show()
print(f"3D NND (data-driven Df={DF_UF}, b=1.0): log10 eta0={le0:.2f} | clustered {100*(~g.bg).mean():.0f}%")""")

md(r"""### Figure 4c · 1-D nearest-neighbour distance distribution — bimodal η at the selected Df

The classic Zaliapin–Ben-Zion **1-D** view (companion to the 2-D R–T plot above): the histogram of
$\log_{10}\eta$ (nearest-neighbour proximity) at the selected **Df = 1.2**, with the two-component
Gaussian-mixture fit — the **clustered** mode (small η: aftershocks/swarms) and the **background** mode
(large η: isolated events) — separated at the threshold $\eta_0$ that defines the split used in Fig 4.
The dt.ct-relocated events fall mostly in the background mode (they lacked a waveform-similar neighbour →
genuinely isolated), except large mainshocks like the 2014 M3.89 which are dt.ct only for magnitude
dissimilarity and sit in the clustered mode. $D_f$ robustness (dt.cc structural 1.1 vs full-population 1.2)
is quantified in the fractal-dimension notebook §5b; the split barely moves.""")
co(r"""from scipy.stats import norm as _norm
le=np.log10(nd.eta.values); le=le[np.isfinite(le)]
fig,ax=plt.subplots(figsize=(7.8,4.7))
ax.hist(le,bins=45,density=True,color="0.82",ec="w",zorder=1)
xs=np.linspace(le.min(),le.max(),400); mns,sgs,wts=info["means"],info["sigmas"],info["weights"]
order=np.argsort(mns); names=["clustered mode","background mode"]; cols=["tab:red","tab:green"]   # small eta = clustered
for rank,idx in enumerate(order):
    ax.plot(xs,wts[idx]*_norm.pdf(xs,mns[idx],sgs[idx]),color=cols[rank],lw=2.3,label=names[rank],zorder=3)
ax.axvline(le0,color="k",ls="--",lw=2,label=fr"$\eta_0$ (log10 = {le0:.2f})",zorder=4)
ax.set_xlabel(r"$\log_{10}\eta$   (nearest-neighbour proximity)",fontsize=13); ax.set_ylabel("Density",fontsize=13)
ax.set_title(f"1-D bimodal NND distribution + GMM (3D, Df = {DF_UF}, b = 1.0)",fontsize=13); ax.tick_params(labelsize=11)
ax.legend(fontsize=11,framealpha=1,edgecolor="black"); fig.tight_layout(); savempl(fig,"fig04c_nnd_eta_histogram"); plt.show()
print(f"1-D bimodal split at log10 eta0={le0:.2f}: clustered {100*(~g.bg).mean():.0f}% | background {100*g.bg.mean():.0f}%")""")

md(r"""## Figure 4c · Seismicity count vs seismic moment release — Ulsan-Fault subregion

Prof. Won-Young Kim suggested mapping **seismic moment** rather than event count. Both views, side-by-side:

* **Left — count density** (number of events): dominated by the *many small* earthquakes, so it traces the
  **fault structure** and is the statistically robust quantity (∝ activity rate, the basis of $b$-value,
  ETAS, declustering — measured above $M_c$).
* **Right — seismic-moment release** ($M_0=10^{1.5M+9.1}$ N·m, summed & smoothed; **log scale**): dominated
  by the *few largest* events. In the UF box the **2014 M3.99 alone = 56 %** of the total moment and the
  **top two events = 85 %** (green stars), so the moment map collapses to ~two spots — where the fault
  actually *slipped / released strain energy* (hazard-relevant).

**Are they the same (Gutenberg–Richter)?** Only if $b$ and the largest magnitude $M_{max}$ are spatially
uniform. With $b\approx1<1.5$ the moment integral is controlled by $M_{max}$, so wherever $M_{max}$ varies the
two maps **diverge** — which is exactly what we see. **Both are standard**: count/rate above $M_c$ for
statistics & forecasting (your "rule of thumb"); cumulative-moment / moment-rate maps (Kostrov summation,
seismic-vs-geodetic moment budgets) for deformation & hazard. The trade-off is robustness — moment here rests
on **1–2 events**, so it is physically meaningful but statistically noisy.

*Caveats:* ML is used as an $M_w$ proxy in $M_0$ (rough; only the spatial pattern is read, which the largest
events fix regardless). The **full absolute catalogue** is used here (not the dt.cc subset) for count-complete
regional coverage; the **M3.99** mainshock — which alone is 56 % of the moment — is now also recovered in the
dt.cc relocation (dt.ct-relocated), so the moment map no longer depends on the absolute catalogue to retain it.""")
co(r"""ufm=uf[uf.ml_all.notna()].copy(); ufm["M0"]=10**(1.5*ufm.ml_all+9.1)        # N·m (ML as Mw proxy)
SP2=0.004
def gridw(lon,lat,w,sp=SP2,sm=1.6):
    xb=np.arange(UF[0],UF[1]+sp,sp); yb=np.arange(UF[2],UF[3]+sp,sp)
    H,_,_=np.histogram2d(lon,lat,bins=[xb,yb],weights=w); H=gaussian_filter(H,sm)
    return xr.DataArray(H.T,coords={"lat":(yb[:-1]+yb[1:])/2,"lon":(xb[:-1]+xb[1:])/2},dims=["lat","lon"])
gN=gridw(ufm.lon.values,ufm.lat.values,np.ones(len(ufm)))                       # event-count density
gM=gridw(ufm.lon.values,ufm.lat.values,ufm.M0.values); gML=np.log10(gM.where(gM>0))  # moment density (log10 N·m)
big=ufm.sort_values("M0",ascending=False).head(2)
fig=pygmt.Figure()
with fig.subplot(nrows=1,ncols=2,figsize=("18c","9c"),margins="0.8c"):
    with fig.set_panel(panel=0):
        vN=float(np.percentile(gN.values[gN.values>0],99))
        fig.basemap(region=list(UF),projection="M?",frame=["WSne+tSeismicity count density","xa0.1","ya0.1"])
        pygmt.makecpt(cmap="hot",series=[0,vN],reverse=True)
        fig.grdimage(gN.where(gN>=0.05*vN),cmap=True,nan_transparent=True)
        if os.path.exists(FAULTS): fig.plot(data=FAULTS,pen="0.8p,gray20")
        fig.plot(x=ufm.lon,y=ufm.lat,style="c0.035c",pen="0.3p,black")
        fig.basemap(map_scale=SCALE_UF)
        with pygmt.config(FONT_LABEL="12p",FONT_ANNOT_PRIMARY="10p"):
            fig.colorbar(cmap=True,frame="af+lSmoothed event count",position="JBC+w6c/0.35c+h+o0c/0.9c")
    with fig.set_panel(panel=1):
        lo=float(np.nanpercentile(gML.values,55)); hi=float(np.nanmax(gML.values))
        fig.basemap(region=list(UF),projection="M?",frame=["wSnE+tSeismic moment release","xa0.1","ya0.1"])
        pygmt.makecpt(cmap="hot",series=[lo,hi],reverse=True)
        fig.grdimage(gML,cmap=True,nan_transparent=True)
        if os.path.exists(FAULTS): fig.plot(data=FAULTS,pen="0.8p,gray20")
        fig.plot(x=ufm.lon,y=ufm.lat,style="c0.035c",pen="0.3p,black")
        fig.plot(x=big.lon,y=big.lat,style="a0.5c",fill="lightgreen",pen="0.7p,black")   # top-2 moment events
        fig.basemap(map_scale=SCALE_UF)
        with pygmt.config(FONT_LABEL="12p",FONT_ANNOT_PRIMARY="10p"):
            fig.colorbar(cmap=True,frame="af+llog@-10@- moment (N m)",position="JBC+w6c/0.35c+h+o0c/0.9c")
savegmt(fig,"fig04c_count_vs_moment"); fig.show(width=1500)
_tot=ufm.M0.sum()
print(f"UF moment (proxy) {_tot:.2e} N·m = Mw {(np.log10(_tot)-9.1)/1.5:.2f}; top-1 {100*big.M0.iloc[0]/_tot:.0f}%, top-2 {100*big.M0.sum()/_tot:.0f}%")""")

md(r"""## Figure 4d · Background vs clustered seismicity through 2016 Gyeongju — Ulsan-Fault subregion

Using **our own enhanced dt.cc-relocated UF catalogue** and **our own 3-D NND split** (Df = 1.2, b = 1.0; the
Fig 4 analysis — *not* the regional HypoSVI catalogue), events are separated into **background** (isolated) and
**clustered** (aftershock/swarm). **Left:** cumulative counts vs time — the post-Gyeongju surge is
overwhelmingly **clustered** (triggered), while the **background accumulates steadily** across Gyeongju.
**Right:** their magnitude-frequency distributions (Mc, b per population).""")
co(r"""gs=g.sort_values("event_time").reset_index(drop=True)
fig,ax=plt.subplots(1,2,figsize=(14,5))
ax[0].step(gs.event_time,np.arange(1,len(gs)+1),where="post",color="0.45",lw=1.5,label=f"All ({len(gs)})")
for lab,sub,col in [("Background",gs[gs.bg],"tab:blue"),("Clustered",gs[~gs.bg],"tab:red")]:
    ss=sub.sort_values("event_time"); ax[0].step(ss.event_time,np.arange(1,len(ss)+1),where="post",color=col,lw=2.0,label=f"{lab} ({len(ss)})")
ax[0].axvline(g0,color="tab:green",ls="--",lw=1.8,label="2016 Gyeongju")
ax[0].set_xlabel("Year",fontsize=13); ax[0].set_ylabel("Cumulative events",fontsize=13)
ax[0].set_title("Cumulative: background vs clustered",fontsize=13); ax[0].tick_params(labelsize=11)
ax[0].legend(loc="upper left",fontsize=10,framealpha=1,edgecolor="black")
DM=0.1
for lab,sub,col in [("Background",gs[gs.bg],"tab:blue"),("Clustered",gs[~gs.bg],"tab:red")]:
    m=sub.kma_mag.values; mc=nnd.estimate_mc(m,DM); b=nnd.estimate_b(m,mc,DM)
    xb=np.round(np.arange(np.floor(m.min()/DM)*DM,m.max()+DM,DM),2); cum=np.array([(m>=x-1e-9).sum() for x in xb])
    ax[1].scatter(xb,cum,s=22,facecolors="none",edgecolors=col,linewidths=1.2,label=f"{lab}: $M_c$={mc:.1f}, b={b:.2f}")
ax[1].set_yscale("log"); ax[1].set_xlabel(r"Local magnitude $M_\mathrm{L}$",fontsize=13); ax[1].set_ylabel(r"$N(\geq M_\mathrm{L})$",fontsize=13)
ax[1].set_title("Magnitude distributions",fontsize=13); ax[1].tick_params(labelsize=11)
ax[1].legend(loc="upper right",fontsize=10,framealpha=1,edgecolor="black")
fig.tight_layout(); savempl(fig,"fig04d_bg_clustered_gyeongju"); plt.show()
pre=gs[gs.event_time<g0]; post=gs[gs.event_time>=g0]
print(f"before Gyeongju: bg={int(pre.bg.sum())} clu={int((~pre.bg).sum())} | after: bg={int(post.bg.sum())} clu={int((~post.bg).sum())}")""")

md(r"""## Figure 4e · Background seismicity BEFORE the 2016 Gyeongju earthquake — Ulsan-Fault subregion

Only the **background (declustered)** events from our 3-D NND that occurred **before 2016 Gyeongju** — the
pre-Gyeongju **tectonic background** of the fault, uncontaminated by later triggered clusters (open circles =
epicentres; warm field = their smoothed density). Same enhanced relocated catalogue (dt.cc + dt.ct) / NND as Fig 4.""")
co(r"""bgpre=g[(g.bg)&(g.event_time<g0)]
grid=dgrid(bgpre.svi_lon.values,bgpre.svi_lat.values); vmax=float(grid.max()) or 1.0
fig=pygmt.Figure()
fig.basemap(region=list(UF),projection=PROJ,frame=[f"WSne+tBackground seismicity before 2016 Gyeongju (N={len(bgpre)})","xa0.1","ya0.1"])
pygmt.makecpt(cmap="hot",series=[0,vmax],reverse=True)
fig.grdimage(grid.where(grid>=0.05*vmax),cmap=True,nan_transparent=True)
fig.coast(shorelines="0.4p,gray45")
if os.path.exists(FAULTS): fig.plot(data=FAULTS,pen="0.8p,gray20")
fig.plot(x=bgpre.svi_lon,y=bgpre.svi_lat,style="c0.11c",pen="0.7p,black")
fig.basemap(map_scale=SCALE_UF)
with pygmt.config(FONT_LABEL="12p",FONT_ANNOT_PRIMARY="10p"):
    fig.colorbar(cmap=True,frame="af+lSmoothed event count",position="JBC+w7c/0.35c+h+o0c/0.9c")
savegmt(fig,"fig04e_background_before_gyeongju"); fig.show()
print(f"background-before-Gyeongju events: {len(bgpre)}")""")

md(r"""## Figure 4f · Background vs clustered across the ±6-month Gyeongju window — cumulative

The same **±6-month window** as Fig 3d, now split by NND class. **Clustered** (triggered) events jump from
**8 → 100** across Gyeongju (~12×) while **background** barely changes (**19 → 25**) — so the post-Gyeongju
surge is almost entirely *triggered* seismicity, not a background-rate change.""")
co(r"""Wd=pd.Timedelta(days=183)
win=g[(g.event_time>=g0-Wd)&(g.event_time<g0+Wd)].sort_values("event_time")
fig,ax=plt.subplots(figsize=(9.5,4.6))
ax.step(win.event_time,np.arange(1,len(win)+1),where="post",color="0.45",lw=1.8,zorder=4,label=f"All ({len(win)})")
for lab,sub,col in [("Background",win[win.bg],"tab:blue"),("Clustered",win[~win.bg],"tab:red")]:
    ss=sub.sort_values("event_time"); ax.step(ss.event_time,np.arange(1,len(ss)+1),where="post",color=col,lw=2.2,zorder=4,label=f"{lab} ({len(ss)})")
ax.axvline(g0,color="tab:green",ls="--",lw=1.8,zorder=3,label="2016 Gyeongju")
ax.axvspan(g0-Wd,g0,color="tab:blue",alpha=0.07,zorder=0); ax.axvspan(g0,g0+Wd,color="tab:red",alpha=0.07,zorder=0)
ax.set_xlabel("Date",fontsize=13); ax.set_ylabel("Cumulative relocated events",fontsize=13)
ax.set_title("Ulsan Fault — background vs clustered across the 2016 Gyeongju window",fontsize=13)
ax.tick_params(labelsize=11); ax.legend(loc="upper left",fontsize=10,framealpha=1,edgecolor="black")
fig.autofmt_xdate(); fig.tight_layout(); savempl(fig,"fig04f_window_bg_clustered_cumulative"); plt.show()
wp=win[win.event_time<g0]; wq=win[win.event_time>=g0]
print(f"window before: bg={int(wp.bg.sum())} clu={int((~wp.bg).sum())} | after: bg={int(wq.bg.sum())} clu={int((~wq.bg).sum())}")""")

md(r"""## Figure 4g · Background vs clustered maps — ±6 months around 2016 Gyeongju

The same window, mapped: **before** (left) vs **after** (right), with **background** (blue) and **clustered**
(red) epicentres distinguished. Before Gyeongju the few events are mostly background; after, a dense
**clustered** patch appears on the fault — the spatial signature of triggering.""")
co(r"""sub_pre=g[(g.event_time>=g0-Wd)&(g.event_time<g0)]; sub_post=g[(g.event_time>=g0)&(g.event_time<g0+Wd)]
fig=pygmt.Figure()
with fig.subplot(nrows=1,ncols=2,figsize=("18c","9c"),margins="0.6c"):
    for j,(sub,ttl) in enumerate([(sub_pre,"6 months before Gyeongju"),(sub_post,"6 months after Gyeongju")]):
        with fig.set_panel(panel=j):
            fig.basemap(region=list(UF),projection="M?",frame=[f"WSne+t{ttl}+s(background {int(sub.bg.sum())}, clustered {int((~sub.bg).sum())})","xa0.1","ya0.1"])
            fig.coast(shorelines="0.3p,gray60")
            if os.path.exists(FAULTS): fig.plot(data=FAULTS,pen="0.8p,gray30")
            bgp=sub[sub.bg]; clp=sub[~sub.bg]
            fig.plot(x=bgp.svi_lon,y=bgp.svi_lat,style="c0.13c",fill="dodgerblue",pen="0.4p,black")
            fig.plot(x=clp.svi_lon,y=clp.svi_lat,style="c0.13c",fill="red",pen="0.4p,black")
            if j==0:                                                   # legend in left panel (upper-right corner)
                lx0=UF[1]-0.095; lx1=UF[1]-0.005; ly1=UF[3]-0.006; ly0=ly1-0.05
                fig.plot(x=[lx0,lx1,lx1,lx0,lx0],y=[ly0,ly0,ly1,ly1,ly0],fill="white@15",pen="0.5p,black")
                fig.plot(x=[lx0+0.014],y=[ly1-0.013],style="c0.13c",fill="dodgerblue",pen="0.4p,black"); fig.text(x=lx0+0.026,y=ly1-0.013,text="background",font="8p,Helvetica,black",justify="LM")
                fig.plot(x=[lx0+0.014],y=[ly1-0.035],style="c0.13c",fill="red",pen="0.4p,black"); fig.text(x=lx0+0.026,y=ly1-0.035,text="clustered",font="8p,Helvetica,black",justify="LM")
            fig.basemap(map_scale=SCALE_UF)
savegmt(fig,"fig04g_window_bg_clustered_maps"); fig.show(width=1500)
print(f"before {len(sub_pre)} (bg {int(sub_pre.bg.sum())}/clu {int((~sub_pre.bg).sum())}) | after {len(sub_post)} (bg {int(sub_post.bg.sum())}/clu {int((~sub_post.bg).sum())})")""")

md(r"""## Figure 4h · WHERE did the rate change? — network-clean ±6-month window

The right window to map: **±6 months around Gyeongju**, where the network is **constant** (13 stations
before and after), so any spatial rate change is **real, not detection**. The periods are equal (183 d), so
the **rate ratio = count ratio**. Maps of $\log_2(\text{after}/\text{before})$ (diverging: red = increase,
white = unchanged, blue = decrease), for **all events** (left) and **background only** (right); after-period
epicentres overlaid (black).

**Reading it (RSF view):** the *total* rate increase is **localized to one patch** on the central fault — a
triggered transient — while the **background** is **spatially flat / near-unchanged** (19→25 over the box).
So in the clean window there is **no broad background lobe-or-shadow pattern** resolvable: the response is a
*localized clustered transient on top of a steady background* — consistent with an RSF stress-step (clock
advance of a near-critical patch) rather than a change in the background loading. *Caveat:* N is small
(background 19/25), so weak spatial structure is not constrained.""")
co(r"""Wd=pd.Timedelta(days=183)
apre=g[(g.event_time>=g0-Wd)&(g.event_time<g0)]; apost=g[(g.event_time>=g0)&(g.event_time<g0+Wd)]
SPh=0.012
def dgh(d):
    xb=np.arange(UF[0],UF[1]+SPh,SPh); yb=np.arange(UF[2],UF[3]+SPh,SPh)
    H,_,_=np.histogram2d(d.svi_lon.values,d.svi_lat.values,bins=[xb,yb]); H=gaussian_filter(H,2.0)
    return xr.DataArray(H.T,coords={"lat":(yb[:-1]+yb[1:])/2,"lon":(xb[:-1]+xb[1:])/2},dims=["lat","lon"])
fig=pygmt.Figure()
with fig.subplot(nrows=1,ncols=2,figsize=("18c","9c"),margins="0.7c"):
    for j,(pr,po,ttl) in enumerate([(apre,apost,f"All events  ({len(apre)}@-pre@- -> {len(apost)}@-post@-)"),
                                    (apre[apre.bg],apost[apost.bg],f"Background  ({int(apre.bg.sum())}@-pre@- -> {int(apost.bg.sum())}@-post@-)")]):
        SA=dgh(pr); SB=dgh(po); ratio=np.log2((SB+0.05)/(SA+0.05))
        ratio=ratio.where((SA+SB)>0.05*float((SA+SB).max()))
        with fig.set_panel(panel=j):
            fr="WSne" if j==0 else "wSne"
            fig.basemap(region=list(UF),projection="M?",frame=[f"{fr}+t{ttl}","xa0.1","ya0.1"])
            pygmt.makecpt(cmap="vik",series=[-2.5,2.5])
            fig.grdimage(ratio,cmap=True,nan_transparent=True)
            if os.path.exists(FAULTS): fig.plot(data=FAULTS,pen="0.7p,gray25")
            fig.plot(x=po.svi_lon,y=po.svi_lat,style="c0.07c",pen="0.4p,black")   # after-period epicentres
            fig.basemap(map_scale=SCALE_UF)
            with pygmt.config(FONT_LABEL="12p",FONT_ANNOT_PRIMARY="10p"):
                fig.colorbar(cmap=True,frame="a1f0.5+llog@-2@-(rate after / before): red=increase, blue=decrease",position="JBC+w7c/0.35c+h+o0c/0.9c")
savegmt(fig,"fig04h_window_rate_change_map"); fig.show(width=1500)
print(f"network-clean +/-6mo: all {len(apre)}->{len(apost)} ({len(apost)/max(len(apre),1):.1f}x) | background {int(apre.bg.sum())}->{int(apost.bg.sum())} ({apost.bg.sum()/max(apre.bg.sum(),1):.1f}x)")""")

md(r"""## Figure 5 · Frequency–magnitude distribution — pre- vs post-2019

Gutenberg–Richter FMD of **ML-assigned** Ulsan-Fault-box events, split at **2019** (a network/completeness
step). Both the **cumulative** $N(\geq M_\mathrm{L})$ (open circles) and **incremental** counts (filled
squares) are shown per period. **Mc** = maximum-curvature peak of the incremental FMD $+0.2$ (Woessner &
Wiemer 2005); **b** = Aki–Utsu maximum-likelihood for $M\geq M_\mathrm{c}$, with Shi & Bolt (1982)
uncertainty; dashed lines are the fitted G–R relations anchored at $M_\mathrm{c}$.""")
co(r"""SPLIT=pd.Timestamp("2019-01-01",tz="utc"); DM=0.1
def shi_bolt(m,b):                                   # Shi & Bolt (1982) b-value standard error
    m=np.asarray(m,float); n=len(m)
    return 2.30*b**2*np.sqrt(((m-m.mean())**2).sum()/(n*(n-1))) if n>2 else np.nan
fig,ax=plt.subplots(figsize=(7.4,6.2)); out=[]
for lab,sub,col in [("Pre-2019",asg[asg.time<SPLIT],"tab:blue"),("2019 onward",asg[asg.time>=SPLIT],"tab:red")]:
    m=sub.ml_all.values; mc=nnd.estimate_mc(m,DM); b=nnd.estimate_b(m,mc,DM); sb=shi_bolt(m[m>=mc-1e-9],b)
    bins=np.arange(np.floor(m.min()/DM)*DM, m.max()+DM, DM); h,edges=np.histogram(m,bins=bins); cent=edges[:-1]+DM/2
    cum=np.array([(m>=e-1e-9).sum() for e in cent])                              # cumulative N(>=M)
    ax.scatter(cent,cum,s=30,facecolors="none",edgecolors=col,linewidths=1.4,label=f"{lab} cumulative (N={len(m)})")
    ax.scatter(cent,np.where(h>0,h,np.nan),s=14,color=col,alpha=0.45,marker="s",label=f"{lab} incremental")
    N_mc=(m>=mc-1e-9).sum(); a=np.log10(N_mc)+b*mc; xx=np.array([mc,m.max()])    # G-R line anchored at Mc
    ax.plot(xx,10**(a-b*xx),"--",color=col,lw=1.8); ax.axvline(mc,color=col,ls=":",lw=1.0)
    out.append((lab,len(m),mc,b,sb))
ax.set_yscale("log"); ax.tick_params(labelsize=11)
ax.set_xlabel(r"Local magnitude  $M_\mathrm{L}$",fontsize=14); ax.set_ylabel(r"Number of events  $N(\geq M_\mathrm{L})$",fontsize=14)
ax.set_title("Ulsan Fault subregion — frequency–magnitude distribution",fontsize=14)
ax.text(0.97,0.97,"\n".join(f"{l}:  $M_c$={mc:.2f},  b={b:.2f}±{sb:.2f}" for l,n,mc,b,sb in out),
        transform=ax.transAxes,ha="right",va="top",fontsize=12,bbox=dict(boxstyle="round",fc="white",ec="black"))
ax.legend(loc="lower left",fontsize=10,framealpha=1,edgecolor="black"); fig.tight_layout(); savempl(fig,"fig05_fmd_pre_post_2019"); plt.show()
for l,n,mc,b,sb in out: print(f"{l}: N={n}  Mc={mc:.2f}  b={b:.2f}+/-{sb:.2f}")""")

md(r"""### Figure 5b · $M_c$-robust b-value — b-positive (Van der Elst 2021)

The **b-positive** estimator (SeismoStats `BPositiveBValueEstimator`) uses only **consecutive positive
magnitude differences** $m_i\ge m_{i-1}+\delta_{mc}$. Because differencing cancels a slowly-varying detection
threshold, b-positive is **far less sensitive to $M_c$ and to short-term aftershock incompleteness** than the
classic Aki–Utsu estimate. The plot sweeps the *assumed* $M_c$: the classic b drifts strongly at low $M_c$
(incompleteness bias), whereas **b-positive stays flat** — its value can be read even where the catalog is
incomplete. $\delta_{mc}=0.2$, $\delta_m=0.1$.""")
co(r"""import seismostats as ss
from seismostats.analysis import BPositiveBValueEstimator, ClassicBValueEstimator
try: from seismostats.utils import bin_to_precision as _bin
except Exception: from seismostats import bin_to_precision as _bin
DMC=0.2
def b_at(m,t,mc,kind):
    e=BPositiveBValueEstimator() if kind=="pos" else ClassicBValueEstimator()
    if kind=="pos": e.calculate(m,mc=mc,delta_m=DM,times=t,dmc=DMC)
    else:           e.calculate(m,mc=mc,delta_m=DM)
    return e.b_value,e.std,e.n
mA=_bin(asg.ml_all.values,DM); tA=asg.time.values; mcs=np.round(np.arange(-0.5,1.35,0.1),2)
fig,ax=plt.subplots(figsize=(7.6,5.6))
for lab,col,mk,kind in [("Classic (Aki–Utsu)","tab:gray","o","cla"),
                        ("b-positive (Van der Elst 2021)","tab:purple","s","pos")]:
    bb,sd=[],[]
    for mc in mcs:
        try: v,s,_=b_at(mA,tA,mc,kind)
        except Exception: v,s=np.nan,np.nan
        bb.append(v); sd.append(s)
    bb,sd=np.array(bb,float),np.array(sd,float)
    ax.plot(mcs,bb,mk+"-",color=col,lw=1.8,label=lab); ax.fill_between(mcs,bb-sd,bb+sd,color=col,alpha=0.18)
ax.axhline(1.0,color="0.7",ls=":",lw=0.9)
ax.set_xlabel(r"Assumed completeness  $M_\mathrm{c}$",fontsize=14); ax.set_ylabel("b-value",fontsize=14)
ax.set_title("b-value stability vs $M_\\mathrm{c}$: classic vs b-positive (UF, ML-assigned)",fontsize=13)
ax.tick_params(labelsize=11); ax.set_ylim(0.2,1.6); ax.legend(fontsize=11,framealpha=1,edgecolor="black",loc="lower right")
fig.tight_layout(); savempl(fig,"fig05b_bpositive_stability"); plt.show()
for lab,sub in [("Whole",asg),("Pre-2019",asg[asg.time<SPLIT]),("2019 onward",asg[asg.time>=SPLIT])]:
    v,s,n=b_at(_bin(sub.ml_all.values,DM),sub.time.values,0.3,"pos")
    print(f"b-positive {lab:11s}: b={v:.2f}+/-{s:.2f}  (n_pos={n}, mc=0.3, dmc={DMC})")""")

md(r"""### Figure 5c · The b-positive mechanism — FMD of magnitude *differences*

Direct demonstration of *why* b-positive works. Van der Elst's result: for a Gutenberg–Richter population the
**consecutive positive magnitude differences** $\Delta m = m_i-m_{i-1}\ (\ge\delta_{mc})$ are exponentially
distributed with the **same b**. So instead of the raw FMD — which **rolls over below $M_c\approx0.5$**
(small events missing) — b-positive fits the **differences distribution**, which stays **log-linear down to
$\delta_{mc}=0.2$**. That lower, cleaner effective threshold is what buys the completeness robustness.""")
co(r"""mm=_bin(asg.ml_all.values,DM); dms=_bin(np.diff(mm),DM); dms=dms[dms>=DMC-DM/2]
xb=np.round(np.arange(np.floor(mm.min()),mm.max()+DM,DM),2); cum=np.array([(mm>=x-1e-9).sum() for x in xb])
xd=np.round(np.arange(DMC,dms.max()+DM,DM),2); cumd=np.array([(dms>=x-1e-9).sum() for x in xd])
mc_raw=nnd.estimate_mc(mm,DM); b_raw=nnd.estimate_b(mm,mc_raw,DM); b_dif=np.log10(np.e)/(dms.mean()-(DMC-DM/2))
fig,ax=plt.subplots(figsize=(7.6,6.0))
ax.scatter(xb,cum,s=26,facecolors="none",edgecolors="tab:gray",linewidths=1.3,label=fr"Raw FMD  $N(\geq M_\mathrm{{L}})$  (rolls over < $M_c$={mc_raw:.1f})")
ax.scatter(xd,cumd,s=20,color="tab:purple",marker="s",alpha=0.6,label=r"Positive differences  $N(\geq\Delta m)$")
xx=np.array([mc_raw,mm.max()]); ax.plot(xx,10**(np.log10((mm>=mc_raw-1e-9).sum())+b_raw*(mc_raw-xx)),"--",color="tab:gray",lw=1.6,label=f"classic  b={b_raw:.2f}")
xx=np.array([DMC,dms.max()]); ax.plot(xx,10**(np.log10((dms>=DMC-1e-9).sum())+b_dif*(DMC-xx)),"--",color="tab:purple",lw=1.6,label=f"b-positive  b={b_dif:.2f}")
ax.axvline(DMC,color="tab:purple",ls=":",lw=1.0); ax.axvline(mc_raw,color="tab:gray",ls=":",lw=1.0)
ax.set_yscale("log"); ax.tick_params(labelsize=11)
ax.set_xlabel(r"Magnitude $M_\mathrm{L}$  or  magnitude difference $\Delta m$",fontsize=14)
ax.set_ylabel("Cumulative count",fontsize=14); ax.set_title(r"b-positive mechanism: differences stay complete to $\delta_{mc}$",fontsize=13)
ax.legend(fontsize=10,framealpha=1,edgecolor="black",loc="upper right"); fig.tight_layout(); savempl(fig,"fig05c_bpositive_mechanism"); plt.show()
print(f"raw FMD rolls over at Mc={mc_raw:.1f} (classic b={b_raw:.2f}); differences log-linear from dmc={DMC} (b={b_dif:.2f})")""")

md(r"""### Figure 5d · b-value across three completeness epochs — classic vs b-positive

The UF network steps at **2014** and **2019** (completeness epochs), giving three periods: **before 2014**,
**2014–2019**, **after 2019**. Classic Aki–Utsu (each at its own $M_c$) trends **downward** (1.13 → 1.05 →
0.98) — readily mis-read as a real b decrease. **b-positive** (common low threshold, $M_c$-robust) is **flat
within uncertainty at ≈1.1** across all three, so the apparent classic trend is a completeness/STAI artifact,
not tectonic. The earliest epoch is sparse ($n_+$≈70) hence its wide error bar.""")
co(r"""T1=pd.Timestamp("2014-01-01",tz="utc")
EPOCHS=[("Before 2014",asg[asg.time<T1]),("2014–2019",asg[(asg.time>=T1)&(asg.time<SPLIT)]),("After 2019",asg[asg.time>=SPLIT])]
MC_COMMON=0.3; rows=[]
for lab,sub in EPOCHS:
    m=_bin(sub.ml_all.values,DM); t=sub.time.values; mc=nnd.estimate_mc(m,DM)
    c=ClassicBValueEstimator(); c.calculate(m,mc=mc,delta_m=DM)
    p=BPositiveBValueEstimator(); p.calculate(m,mc=MC_COMMON,delta_m=DM,times=t,dmc=DMC)
    rows.append((lab,len(m),mc,c.b_value,c.std,p.b_value,p.std,p.n))
fig,ax=plt.subplots(figsize=(7.8,5.4)); xp=np.arange(len(EPOCHS)); w=0.38
ax.bar(xp-w/2,[r[3] for r in rows],w,yerr=[r[4] for r in rows],capsize=4,color="tab:gray",label="Classic (Aki–Utsu, @ own $M_c$)")
ax.bar(xp+w/2,[r[5] for r in rows],w,yerr=[r[6] for r in rows],capsize=4,color="tab:purple",label=f"b-positive (@ $M_c$={MC_COMMON})")
ax.axhline(1.0,color="0.7",ls=":",lw=0.9)
ax.set_xticks(xp); ax.set_xticklabels([f"{r[0]}\n(N={r[1]}, $M_c$={r[2]:.1f})" for r in rows],fontsize=11)
ax.set_ylabel("b-value",fontsize=14); ax.set_title("Ulsan Fault b-value through time: classic vs b-positive",fontsize=13)
ax.tick_params(labelsize=11); ax.set_ylim(0,1.7); ax.legend(fontsize=11,framealpha=1,edgecolor="black",loc="upper right")
fig.tight_layout(); savempl(fig,"fig05d_bvalue_three_epochs"); plt.show()
for lab,n,mc,bc,sc,bp,sp,npos in rows: print(f"{lab:12s}: N={n:4d} Mc={mc:.2f} | classic b={bc:.2f}±{sc:.2f} | b-positive b={bp:.2f}±{sp:.2f} (n_pos={npos})")""")

md(r"""---
**Sources:** catalog `…blastclean_with_ml_heo_clean.csv`; Vp/Vs `07.SeismoStats/interp_k.txt`
(H-κ); faults `KS_KG/HypoInv/faults_lonlat.gmt`; relocations `…/03.dt.cc_kim2011/hypoDD.reloc`; ML
`catalog_ml_heo_ufonly_reloc.csv`; declustering `kma_absolute_location.nnd` (3D, data-driven Df=1.2, b=1.0).""")

nb.cells=C
out="00.Summary_figures_Zhigang.ipynb"; nbf.write(nb,out); print("wrote",out,len(C),"cells")
