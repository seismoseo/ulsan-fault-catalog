import warnings; warnings.filterwarnings("ignore")
import numpy as np, pygmt, matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":130,"font.size":10})
def rd(f,n):
    d={}
    for ln in open(f):
        p=ln.split()
        if len(p)>=n: v=[float(x) for x in p[:n]]; d[int(v[0])]=v
    return d
loc=rd("hypoDD.loc_backup",18); ct=rd("hypoDD.reloc_dtct_only",24); cc=rd("hypoDD.reloc",24)
ids=sorted(set(loc)&set(ct)&set(cc))
def col(d,c): return np.array([d[i][c] for i in ids])
sets=[("HypoInverse (absolute)",loc),("dt.ct relocation",ct),("dt.cc≥0.9 relocation",cc)]
REGION=[129.25,129.55,35.60,35.90]
# ---- PyGMT epicentre comparison (3 panels) ----
fig=pygmt.Figure()
with fig.subplot(nrows=1,ncols=3,figsize=("27c","9c"),frame=["WSne","xa0.1","ya0.1"],margins="0.4c"):
    pygmt.makecpt(cmap="turbo",series=[2,20],reverse=True)
    for j,(name,d) in enumerate(sets):
        with fig.set_panel(j):
            fig.basemap(region=REGION,projection="M?",frame=["WSne","xa0.1f0.05","ya0.1f0.05",f"+t{name}"])
            fig.coast(shorelines="0.4p,gray50")
            fig.plot(x=col(d,2),y=col(d,1),fill=col(d,3),cmap=True,style="c0.07c",pen="0.2p,gray20")
    fig.colorbar(position="JBC+w10c/0.4c+h+o0c/1c",frame=["xa4+lDepth (km)"])
fig.savefig("figs/uf_reloc_compare_map.png",dpi=200)
print("saved figs/uf_reloc_compare_map.png")
# ---- depth cross-section + stats (matplotlib) ----
fig2,ax=plt.subplots(1,3,figsize=(16,4.6),sharex=True,sharey=True)
for a,(name,d) in zip(ax,sets):
    a.scatter(col(d,2),col(d,3),s=4,alpha=0.35,c=col(d,3),cmap="turbo_r",vmin=2,vmax=20)
    a.set(title=name,xlabel="Longitude (°E)",ylim=(22,2)); a.grid(alpha=0.3)
ax[0].set_ylabel("Depth (km)")
fig2.suptitle("Depth vs longitude — absolute → dt.ct → dt.cc (vertical collapse)",y=1.0)
fig2.tight_layout(); fig2.savefig("figs/uf_reloc_compare_depth.png",dpi=150,bbox_inches="tight")
print("saved figs/uf_reloc_compare_depth.png")
# stats
def std_km(d):
    la=col(d,1); return (np.std(col(d,2))*111.19*np.cos(np.deg2rad(la.mean())), np.std(col(d,1))*111.19, np.std(col(d,3)))
for name,d in sets:
    sx,sy,sz=std_km(d); print(f"  {name:28} spread: lon {sx:.2f} lat {sy:.2f} dep {sz:.2f} km")
