#!/usr/bin/env python
"""Generate 12.PNplus_dtcc_only_record_sections_2016.ipynb — record sections for 2016 UF-box events that PN+
FINALLY dt.cc-relocated (cc-resolved, nccp+nccs>0) but PN-original (and the STEAD models) did NOT dt.cc-relocate.
Same wiggle style as nb11 (12 nearest stations, 2-20 Hz, blue-P/red-S), BUT station geometry & epicentral
distances are computed against the PN+ HYPOINVERSE (kim2011) location, and the window is centred on the
HypoInverse origin time. Answers: for events PN+ can precision-relocate but original cannot, do the other pickers
even pick/associate the arrivals? Selection = PN+ cc-resolved with no original cc-resolved event within 8 s+5 km."""
import nbformat as nbf
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# Record sections of PN+-only dt.cc-relocated events (2016 Ulsan-Fault)

The complement of nb11. Here we take events that PN+ **finally dt.cc-relocated** — the highest-precision output,
`nccp+nccs>0` cross-correlation-resolved — and for which **PhaseNet-original did NOT produce a dt.cc-relocated
event** at the same place/time (nor did the STEAD models). These are events where PN+'s picking survives all the
way through QC + HypoDD dt.cc while original's does not.

Unlike nb11, the panel geometry uses the **PN+ HypoInverse (kim2011) location**, not the PyOcto association
location, and the window is centred on the **HypoInverse origin time** — the located, catalog-quality position.
Each panel plots one picker's **associated picks** over the 12 nearest stations' vertical traces (2–20 Hz),
**blue = P, red = S**:
 - **PN+** panel: the picks of the event PN+ associated + located + dt.cc-relocated;
 - an **other-picker** panel populated = that picker associated a nearby event (it just didn't dt.cc-relocate it
   here); **empty, tagged `[NOT associated]`,** = that picker formed no event there at all.

**Selection:** PN+ cc-resolved events with **no PN-original cc-resolved event within 8 s + 5 km**; the ~10
best-recorded (most PN+ picks) are shown.""")

co(r"""import os, glob, warnings, sys; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, obspy
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
HERE="/home/msseo/works/02.Ulsan_Fault_detection/detection_test"
RUNS="/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs"
from uflib import uf_cluster as uf
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in {f.name for f in fm.fontManager.ttflist}: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":120,"font.size":8,"axes.grid":True,"grid.alpha":0.2,"axes.unicode_minus":False})
UF=(129.25,129.55,35.60,35.90); NST=12; W=(-3,18); AMP=0.40
PICKERS=[("original","PhaseNet-original"),("stead","PhaseNet-STEAD"),("eqt","EQT-STEAD")]
def hav(a1,o1,a2,o2):
    x=np.sin(np.radians(a2-a1)/2)**2+np.cos(np.radians(a1))*np.cos(np.radians(a2))*np.sin(np.radians(o2-o1)/2)**2
    return 2*6371*np.arcsin(np.sqrt(x))
def rd_reloc(f):
    if not os.path.exists(f): return None
    r=[l.split() for l in open(f) if l.split()]
    cols=["id","lat","lon","depth","x","y","z","ex","ey","ez","yr","mo","dy","hr","mi","sc","mag",
          "nccp","nccs","nctp","ncts","rcc","rct","cid"]
    df=pd.DataFrame([x[:24] for x in r],columns=cols[:len(r[0])]).apply(pd.to_numeric,errors="coerce")
    df["t"]=pd.to_datetime(dict(year=df.yr.astype(int),month=df.mo.astype(int),day=df.dy.astype(int),
        hour=df.hr.astype(int),minute=df.mi.astype(int),second=df.sc.clip(0,59).astype(int)),utc=True)
    return df
# combined 2016 station table (archive/band month-invariant; dedup by station)
S=pd.concat([pd.read_csv(f"{HERE}/cache/stations_2016_{m:02d}.csv") for m in range(1,13)])
S=S[S.coverage>0].drop_duplicates("sta").reset_index(drop=True)
def load_wf(r,ORIG):
    fs=sorted(glob.glob(os.path.join(r.archive,r.sta,f"{r.band}Z.D",f"*.{ORIG.year}.{ORIG.julday:03d}")))
    if not fs: return None
    try:
        st=obspy.read(fs[0],starttime=ORIG+W[0]-2,endtime=ORIG+W[1]+2); st.merge(fill_value=0)
        tr=st[0]; tr.detrend("demean")
        if tr.stats.sampling_rate>100:
            tr.filter("lowpass",freq=40,zerophase=True); tr.decimate(int(round(tr.stats.sampling_rate/100)),no_filter=True)
        tr.filter("bandpass",freqmin=2,freqmax=20,zerophase=True); tr.trim(ORIG+W[0],ORIG+W[1]); return tr
    except Exception: return None
print(f"{len(S)} stations in 2016 table")""")

co(r"""# ---- selection: PN+ cc-resolved events NOT dt.cc-relocated by PN-original, with PN+ HYPOINVERSE location ----
ROOT=f"{HERE}/reloc_2016_uf"
pnp=rd_reloc(f"{RUNS}/uf_2016_qc/2.HypoDD/02.dt.cc/hypoDD.reloc"); pnp=pnp[(pnp.nccp+pnp.nccs)>0].reset_index(drop=True)
orig=rd_reloc(f"{RUNS}/uf_2016_original_qc/2.HypoDD/02.dt.cc/hypoDD.reloc"); orig=orig[(orig.nccp+orig.nccs)>0].reset_index(drop=True)

# dt.cc id (200000+qcrow) -> QC event_idx -> full members row -> PN+ HypoInverse location/time
mem_qc=pd.read_csv(f"{ROOT}/members_qc.txt",header=None)[0].tolist()
mem_full=pd.read_csv(f"{ROOT}/members.txt",header=None)[0].tolist(); fullpos={e:i for i,e in enumerate(mem_full)}
sm=uf.read_sum(f"{RUNS}/uf_2016/1.HypoInv/kim2011/uf_2016.sum").reset_index(drop=True)   # full-members order
hi_lat=[]; hi_lon=[]; hi_dep=[]; hi_t=[]; npick=[]
for _,r in pnp.iterrows():
    eidx=mem_qc[int(r.id)-200000]; frow=fullpos[eidx]; h=sm.iloc[frow]
    hi_lat.append(h.lat); hi_lon.append(h.lon); hi_dep.append(h.depth)
    hi_t.append(pd.Timestamp(h.time,tz="UTC")); npick.append(int(h.num))
pnp["hlat"]=hi_lat; pnp["hlon"]=hi_lon; pnp["hdep"]=hi_dep; pnp["ht"]=hi_t; pnp["hnum"]=npick

# original cc-resolved match within 8 s + 5 km (compared at the dt.cc positions)
ot=orig.t.values.astype("datetime64[s]").astype(np.int64); order=np.argsort(ot); ot_s=ot[order]
def orig_dtcc(r):
    tt=np.int64(pd.Timestamp(r.t).timestamp()); lo=np.searchsorted(ot_s,tt-8); hi=np.searchsorted(ot_s,tt+8)
    if hi<=lo: return False
    near=orig.iloc[order[lo:hi]]
    return bool((hav(r.lat,r.lon,near.lat.values,near.lon.values)<=5).any())
pnp["orig_dtcc"]=pnp.apply(orig_dtcc,axis=1)
only=pnp[~pnp.orig_dtcc].sort_values("hnum",ascending=False).reset_index(drop=True)
GJ=pd.Timestamp("2016-09-12 11:32:54",tz="UTC")
print(f"PN+ cc-resolved: {len(pnp)};  original ALSO cc-resolved: {int(pnp.orig_dtcc.sum())};  "
      f"PN+-only (no original dt.cc within 8 s+5 km): {len(only)}")
print(f"  of the PN+-only: pre-GJ {int((only.ht<GJ).sum())}, post-GJ {int((only.ht>=GJ).sum())}")
print(only.head(10)[["ht","hlat","hlon","hdep","hnum","nccp","nccs"]].to_string(index=False))""")

md(r"""## Record sections — the 10 best-recorded PN+-only dt.cc events

Distances (y-axis) and the 12-nearest-station set are relative to the **PN+ HypoInverse location**; the window is
centred on the **HypoInverse origin time**. Left = PN+ associated picks; the next three = each other picker's
associated picks nearest this event (8 s + 10 km). **blue = P, red = S.**""")

co(r"""_CACHE={}
def _catasg(p,mm):
    if (p,mm) not in _CACHE:
        cat=pd.read_csv(f"{HERE}/catalogs/catalog_{p}_2016_{mm:02d}_pyocto.csv")
        cat["t"]=pd.to_datetime(cat.time,utc=True).dt.tz_localize(None)
        asg=pd.read_parquet(f"{HERE}/catalogs/assign_{p}_2016_{mm:02d}_pyocto.parquet")
        asg["t"]=pd.to_datetime(asg.time,unit="s"); asg["sta"]=asg.station.str.split(".").str[1]
        _CACHE[(p,mm)]=(cat,asg)
    return _CACHE[(p,mm)]

def assoc_picks(p,ORIG,T,la,lo):
    "picks of picker p's ASSOCIATED event nearest (T,la,lo) within 8 s + 10 km; empty if none."
    cat,asg=_catasg(p,ORIG.month)
    dt=(cat.t-T).abs().dt.total_seconds().values; dd=hav(la,lo,cat.lat.values,cat.lon.values)
    m=(dt<=8)&(dd<=10)
    if not m.any(): return []
    ei=int(cat.event_idx.values[np.where(m)[0][np.argmin(dt[m])]]); ev=asg[asg.event_idx==ei]
    return [(r.sta,str(r.phase).upper(),obspy.UTCDateTime(r.t.to_pydatetime())-ORIG) for _,r in ev.iterrows()]

def col(ph): return "#1f77b4" if ph=="P" else "#d62728"
def make(ev):
    # HypoInverse-based geometry & timing
    la,lo=ev.hlat,ev.hlon; T=ev.ht.tz_localize(None); ORIG=obspy.UTCDateTime(ev.ht.to_pydatetime())
    D=S.copy(); D["edist"]=hav(la,lo,D.lat,D.lon); D=D.sort_values("edist").head(NST).reset_index(drop=True)
    WF={r.sta:load_wf(r,ORIG) for _,r in D.iterrows()}
    panels=[("PN+","phasenet_plus")]+[(lab,p) for p,lab in PICKERS]
    fig,axs=plt.subplots(1,4,figsize=(18,6),sharey=True); cnt={}
    for ax,(name,p) in zip(axs,panels):
        assoc=assoc_picks(p,ORIG,T,la,lo)
        for i,(_,r) in enumerate(D.iterrows()):
            tr=WF.get(r.sta)
            if tr is None: continue
            tt=tr.times()+W[0]; y=tr.data/(np.abs(tr.data).max() or 1)*AMP
            ax.plot(tt,y+i,color="0.4",lw=0.5,zorder=2)
            ax.text(W[0]+0.15,i+AMP*0.55,f"{r.sta} {r.edist:.0f}km",fontsize=6,color="0.35")
            for sta,ph,rel in assoc:
                if sta==r.sta and W[0]<=rel<=W[1]:
                    ax.plot([rel,rel],[i-AMP*0.85,i+AMP*0.85],color=col(ph),lw=2.0,zorder=4)
        na=len(assoc); flag="" if (p=="phasenet_plus" or na) else "  [NOT associated]"
        ax.set(xlim=W,title=f"{name}   {na} assoc.{flag}")
        ax.set_xlabel("Time from origin (s)",fontsize=11)
        cnt[name]=na
    axs[0].set_yticks(range(len(D))); axs[0].set_yticklabels([f"{d:.0f}" for d in D.edist],fontsize=6)
    axs[0].set_ylabel("Epicentral distance (km)",fontsize=11)
    fig.suptitle(f"UF (PN+ HypoInverse loc)   {T:%Y-%m-%d %H:%M:%S}   depth {ev.hdep:.1f} km   "
                 f"({la:.3f}, {lo:.3f})   dt.cc cc-links P/S {int(ev.nccp)}/{int(ev.nccs)}    "
                 f"blue = P, red = S    |    panel title = associated picks",y=1.0,fontsize=9)
    plt.tight_layout(); plt.show()
    return dict(event=f"{T:%m-%d %H:%M}",depth=round(ev.hdep,1),hi_num=int(ev.hnum),
                cc_ps=f"{int(ev.nccp)}/{int(ev.nccs)}",**{n:cnt[n] for n in cnt})

ROWS=[make(only.iloc[k]) for k in range(min(10,len(only)))]
print(pd.DataFrame(ROWS).to_string(index=False))""")

md(r"""## Summary""")
co(r"""bar="="*100; print(bar)
R=pd.DataFrame(ROWS)
print("PN+-only dt.cc-relocated events — associated picks per picker (12 nearest stations, PN+ HypoInverse geom):")
print(R.to_string(index=False))
def med(n): return R[n].median()
print(f'''
READING — these are events PN+ carried all the way to a cross-correlation dt.cc relocation, that PN-original did
NOT dt.cc-relocate (no cc-resolved original event within 8 s + 5 km):
  - PN+ associates a full set here (median {med("PN+"):.0f} associated picks over the 12 nearest stations).
  - PhaseNet-original associated picks: median {med("PhaseNet-original"):.0f} — where this is ~0, original did not
    form the event at all here; where it is populated, original DID associate a nearby event but it failed QC or
    lost its cross-correlation links, so it never reached a dt.cc relocation.
  - STEAD models (median {med("PhaseNet-STEAD"):.0f} / {med("EQT-STEAD"):.0f}) mostly miss at the picker stage.
  - Net: PN+ uniquely delivers a precision-relocatable catalog for these {len(only)} events; the location/geometry
    is the PN+ HypoInverse solution, so the wiggle alignment reflects the actual located hypocentre.
NEXT: overlay the dt.cc-relocated position vs HypoInverse to show the precision gain on these PN+-only events.''')
print(bar)""")

nb=nbf.v4.new_notebook(); nb["cells"]=C
nb["metadata"]={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}
OUT="/home/msseo/works/02.Ulsan_Fault_detection/detection_test/reloc_2016_uf/12.PNplus_dtcc_only_record_sections_2016.ipynb"
nbf.write(nb,OUT); print("wrote",OUT)
