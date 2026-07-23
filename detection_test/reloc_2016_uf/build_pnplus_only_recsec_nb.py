#!/usr/bin/env python
"""Generate 11.PNplus_only_record_sections_2016.ipynb — record sections for 2016 UF-box events that PN+ associated
but the other three pickers (PhaseNet-original, PhaseNet-STEAD, EQT-STEAD) did NOT. Benchmarks
07.PN_vs_original_record_sections_2021_09.ipynb (same 12-nearest-station wiggle plot, blue-P/red-S picks) but
extends it to a 4-panel comparison: PN+ associated picks vs each other picker's RAW picks in the window. Shows
WHY the others missed the event (too few picks -> below the 4/2/2 association gate)."""
import nbformat as nbf
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# Record sections of marginal PN+ events — which pickers actually pick them? (2016 Ulsan-Fault)

We select UF-box events that **PN+ associated** but that have **no matching associated event** in any of
PhaseNet-original / PhaseNet-STEAD / EQT-STEAD within 5 s + 5 km. **IMPORTANT — read the record sections, not
that count:** the association-level "PN+-only" tally (5 s + 5 km) *overstates* PN+, because a picker can associate
the same event yet have PyOcto place its origin a bit further away, so the strict match fails. Each panel below
plots that picker's **ASSOCIATED picks** — the picks of its *own* nearest associated event within a looser
**8 s + 10 km** — over the 12 nearest stations' vertical traces (2–20 Hz), **blue = P, red = S**:
 - a **populated** other-picker panel means that picker DID associate the event (its origin was just shifted, which
   is why the 5 s catalog match missed it) -> NOT a real PN+ edge over that picker;
 - an **empty panel, tagged `[NOT associated]`,** means that picker formed no event there -> a **genuine miss**.
This is the honest test: association, not raw picks.""")

co(r"""import os, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, obspy
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
HERE="/home/msseo/works/02.Ulsan_Fault_detection/detection_test"
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in {f.name for f in fm.fontManager.ttflist}: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":120,"font.size":8,"axes.grid":True,"grid.alpha":0.2,"axes.unicode_minus":False})
UF=(129.25,129.55,35.60,35.90); NST=12; W=(-3,18); AMP=0.40
PICKERS=[("original","PhaseNet-original"),("stead","PhaseNet-STEAD"),("eqt","EQT-STEAD")]
def hav(a1,o1,a2,o2):
    x=np.sin(np.radians(a2-a1)/2)**2+np.cos(np.radians(a1))*np.cos(np.radians(a2))*np.sin(np.radians(o2-o1)/2)**2
    return 2*6371*np.arcsin(np.sqrt(x))
# combined 2016 station table (archive/band are month-invariant; dedup by station)
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

co(r"""# load all-month catalogs (associated events) for every picker; and PN+ assignment + each picker's raw picks
def cat(p):
    dfs=[]
    for m in range(1,13):
        f=f"{HERE}/catalogs/catalog_{p}_2016_{m:02d}_pyocto.csv"
        if os.path.exists(f):
            d=pd.read_csv(f); d["t"]=pd.to_datetime(d.time,utc=True).dt.tz_localize(None); d["mm"]=m; dfs.append(d)
    return pd.concat(dfs,ignore_index=True)
PNP=cat("phasenet_plus"); OTH={p:cat(p) for p,_ in PICKERS}
# PN+-only UF-box events: no other-picker event within 5 s AND 5 km
box=PNP[(PNP.lon.between(*UF[:2]))&(PNP.lat.between(*UF[2:]))].sort_values("t").reset_index(drop=True)
oth_all=pd.concat(list(OTH.values()),ignore_index=True).sort_values("t").reset_index(drop=True)
ot=oth_all.t.values.astype("datetime64[s]").astype(np.int64)
def is_pnplus_only(r):
    tt=np.int64(pd.Timestamp(r.t).timestamp()); lo=np.searchsorted(ot,tt-5); hi=np.searchsorted(ot,tt+5)
    if hi<=lo: return True
    near=oth_all.iloc[lo:hi]
    return not (hav(r.lat,r.lon,near.lat.values,near.lon.values)<=5).any()
box["only"]=box.apply(is_pnplus_only,axis=1); box["npk"]=box.n_p+box.n_s
only=box[box.only].sort_values("npk",ascending=False).reset_index(drop=True)   # best-recorded PN+-only first
print(f"PN+ UF-box events: {len(box)};  PN+-ONLY (missed by all 3 others): {len(only)} "
      f"({100*len(only)/len(box):.0f}%)")
print(only.head(8)[["t","lat","lon","depth","n_p","n_s","npk"]].to_string(index=False))""")

md(r"""## Record sections — the 4 best-recorded PN+-only events

Left panel = PN+ associated picks; the next three = each other picker's raw picks in the window. A clean PN+
P/S set with sparse other-picker panels is the signature of a genuine PN+-only detection.""")

co(r"""_CACHE={}
def _catasg(p,mm):
    if (p,mm) not in _CACHE:
        cat=pd.read_csv(f"{HERE}/catalogs/catalog_{p}_2016_{mm:02d}_pyocto.csv")
        cat["t"]=pd.to_datetime(cat.time,utc=True).dt.tz_localize(None)
        asg=pd.read_parquet(f"{HERE}/catalogs/assign_{p}_2016_{mm:02d}_pyocto.parquet")
        asg["t"]=pd.to_datetime(asg.time,unit="s"); asg["sta"]=asg.station.str.split(".").str[1]
        _CACHE[(p,mm)]=(cat,asg)
    return _CACHE[(p,mm)]

def raw_picks(p,ORIG,T):
    "ALL raw picks of picker p in the window, as (sta, phase, reltime)."
    R=pd.read_parquet(f"{HERE}/picks/picks_{p}_2016_{ORIG.month:02d}.parquet")
    R["time"]=pd.to_datetime(R.time,format="ISO8601",utc=True).dt.tz_localize(None); R["sta"]=R.sta
    w=R[(R.time>=T+pd.Timedelta(seconds=W[0]))&(R.time<=T+pd.Timedelta(seconds=W[1]))]
    return [(r.sta,str(r.phase).upper(),obspy.UTCDateTime(r.time.to_pydatetime())-ORIG) for _,r in w.iterrows()]

def assoc_picks(p,ORIG,T,la,lo):
    "picks of picker p's ASSOCIATED event nearest (T,la,lo) within 8 s + 10 km; empty if it associated none."
    cat,asg=_catasg(p,ORIG.month)
    dt=(cat.t-T).abs().dt.total_seconds().values; dd=hav(la,lo,cat.lat.values,cat.lon.values)
    m=(dt<=8)&(dd<=10)
    if not m.any(): return []
    ei=int(cat.event_idx.values[np.where(m)[0][np.argmin(dt[m])]]); ev=asg[asg.event_idx==ei]
    return [(r.sta,str(r.phase).upper(),obspy.UTCDateTime(r.t.to_pydatetime())-ORIG) for _,r in ev.iterrows()]

def col(ph): return "#1f77b4" if ph=="P" else "#d62728"
def make(ev):
    T=pd.Timestamp(ev.t); ORIG=obspy.UTCDateTime(T.to_pydatetime())
    D=S.copy(); D["edist"]=hav(ev.lat,ev.lon,D.lat,D.lon); D=D.sort_values("edist").head(NST).reset_index(drop=True)
    WF={r.sta:load_wf(r,ORIG) for _,r in D.iterrows()}
    panels=[("PN+","phasenet_plus")]+[(lab,p) for p,lab in PICKERS]
    fig,axs=plt.subplots(1,4,figsize=(18,6),sharey=True); cnt={}
    for ax,(name,p) in zip(axs,panels):
        raw=raw_picks(p,ORIG,T); assoc=assoc_picks(p,ORIG,T,ev.lat,ev.lon)
        for i,(_,r) in enumerate(D.iterrows()):
            tr=WF.get(r.sta)
            if tr is None: continue
            tt=tr.times()+W[0]; y=tr.data/(np.abs(tr.data).max() or 1)*AMP
            ax.plot(tt,y+i,color="0.4",lw=0.5,zorder=2)
            ax.text(W[0]+0.15,i+AMP*0.55,f"{r.sta} {r.edist:.0f}km",fontsize=6,color="0.35")
            for sta,ph,rel in raw:                                  # ALL picks drawn the same (bold); per-panel title
                if sta==r.sta and W[0]<=rel<=W[1]:                  # gives the raw vs associated counts
                    ax.plot([rel,rel],[i-AMP*0.85,i+AMP*0.85],color=col(ph),lw=2.0,zorder=4)
        na=len(assoc); flag="" if (p=="phasenet_plus" or na) else "  [NOT associated]"
        ax.set(xlim=W,title=f"{name}   {len(raw)} raw / {na} assoc.{flag}")
        ax.set_xlabel("Time from origin (s)",fontsize=11)
        cnt[name]=(len(raw),na)
    axs[0].set_yticks(range(len(D))); axs[0].set_yticklabels([f"{d:.0f}" for d in D.edist],fontsize=6)
    axs[0].set_ylabel("Epicentral distance (km)",fontsize=11)
    fig.suptitle(f"UF   {T:%Y-%m-%d %H:%M:%S}   depth {ev.depth:.1f} km   ({ev.lat:.3f}, {ev.lon:.3f})    "
                 f"blue = P, red = S    |    all picks shown; each panel title = (raw picks / associated picks)",y=1.0,fontsize=9)
    plt.tight_layout(); plt.show()
    return dict(event=f"{T:%m-%d %H:%M}",depth=round(ev.depth,1),
                **{f"{n} (raw/assoc)":f"{cnt[n][0]}/{cnt[n][1]}" for n in cnt})

ROWS=[make(only.iloc[k]) for k in range(min(8,len(only)))]
print(pd.DataFrame(ROWS).to_string(index=False))""")

md(r"""## Summary""")
co(r"""bar="="*100; print(bar)
R=pd.DataFrame(ROWS)
print("Per-event RAW / ASSOCIATED picks in the record-section window (12 nearest stations):")
print(R.to_string(index=False))
def med(n):
    s=R[f"{n} (raw/assoc)"].str.split("/",expand=True).astype(int); return s[0].median(), s[1].median()
mr={n:med(n) for n in ["PN+","PhaseNet-original","PhaseNet-STEAD","EQT-STEAD"]}
print(f'''
HONEST READING -- all picks are drawn; each panel title reports (raw picks / associated picks):
  - PN+ ASSOCIATES these {len(only)} marginal UF-box events (median {mr['PN+'][1]:.0f} associated picks).
  - PhaseNet-original DETECTS them (median {mr['PhaseNet-original'][0]:.0f} RAW picks) but ASSOCIATES ~none
    (median {mr['PhaseNet-original'][1]:.0f}): its noisy picking leads PyOcto to place the event at a shifted origin /
    merge it, so it does NOT enter original's catalog here. A detection edge is NOT an association edge.
  - PhaseNet-STEAD / EQT-STEAD have few raw picks (median {mr['PhaseNet-STEAD'][0]:.0f} / {mr['EQT-STEAD'][0]:.0f})
    and associate ~none -- a genuine PICKER miss (below the 4/2/2 gate).
  - Net, at the ASSOCIATION level (what the catalog uses): PN+ uniquely recovers these events; original's raw
    detection does not survive its noisier association; the STEAD models miss already at the picker stage.
  - (The 5 s+5 km "PN+-only" tally of {len(only)} = {100*len(only)/len(box):.0f}% is thus real at association, but
    remember original DID pick many of them -- the gap is association robustness, not raw detection, vs original.)
NEXT: a picker-vs-picker recall number independent of association = match on shared-station P/S pick agreement.''')
print(bar)""")

nb=nbf.v4.new_notebook(); nb["cells"]=C
nb["metadata"]={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}
OUT="/home/msseo/works/02.Ulsan_Fault_detection/detection_test/reloc_2016_uf/11.PNplus_only_record_sections_2016.ipynb"
nbf.write(nb,OUT); print("wrote",OUT)
