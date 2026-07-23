#!/usr/bin/env python
"""Generate 24.HDB_failure_window_record_sections.ipynb — SOTA Wood-Anderson record sections for the
HDB sensor-FAILURE-window events (Dec 2014 – Apr 2015), the readings EXCLUDED from the constant-network
catalog (nb23). Purpose: show by eye that HDB's amplitude there is *corrupted* (residual −1.2 … −2.2 ML),
not a clean constant offset — so it cannot be recovered by an epoch term and is dropped, while the rest of
the 2012–2015 HDB era IS epoch-corrected. Reuses the exact ml_pipeline measurement (band-pass 2-20 Hz →
response removal → Wood-Anderson → Sheen [dist/4,dist/2] hypocentral S-window → snr_pp gate).
Run in `base`, cwd = local_magnitudes."""
import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# HDB sensor-failure window — record sections of the excluded events

In the constant-network catalog (nb23) the HDB 2012-11 → 2015-05 era **is** treated as a proper epoch and
corrected — **except** a short **sensor-failure window (Dec 2014 – Apr 2015)**, where HDB's readings are
*excluded* rather than offset-corrected. This notebook shows, with the exact pipeline measurement, **why**:
HDB's Wood-Anderson amplitude there is **corrupted** (it under-reads by 1.2–2.2 ML), and in at least one
case (**2015-03-19**) it records the event with **high SNR but a wrong amplitude** — proof the gain is
broken, not that the signal is weak. A single constant epoch offset cannot recover such data (you would be
adding ≈ +2 ML to corrupted samples), so the defensible choice is exclusion.

**Measurement (identical to ml_pipeline / nb20):** demean/detrend → band-pass 2–20 Hz → remove response →
Wood-Anderson → **noise** `[P−6,P−1] s`, **signal** Sheen `[dist/4, dist/2] s` (HYPOCENTRAL dist) →
`snr_pp = peak/noise_peak`, reading used iff `snr_pp ≥ 2`. HDB is highlighted in every panel.""")

co(r"""import warnings; warnings.filterwarnings("ignore")
import os, re, glob, numpy as np, pandas as pd, obspy
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
import matplotlib.lines as mlines; from matplotlib.patches import Patch
import ml_pipeline as mp
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":130,"font.size":9.5,"legend.framealpha":1.0,"legend.facecolor":"white","legend.edgecolor":"0.6"})
NOISE_WIN,NOISE_PAD,SIG_DIV_START,SIG_DIV_END,SNR_THR = 5.0,1.0,4.0,2.0,2.0
EVT="../HypoInv"
inv=mp.load_combined_inventory("responses/master")
TAUP=mp.kim2011_taup_model()
# the 7 HDB failure-window events (event_time, HDB ML, HDB residual vs event median)
FAIL=[("2014-12-06 20:10:01", 0.07,-2.22),("2014-12-11 00:30:57",-0.54,-2.19),
      ("2015-01-09 17:33:55",-0.82,-1.64),("2015-03-16 02:59:32",-0.81,-1.19),
      ("2015-03-19 12:51:15", 0.00,-2.03),("2015-03-20 14:47:56",-0.70,-1.80),
      ("2015-04-30 01:22:32",-0.79,-1.86)]
def evdir_from_time(tstr):
    key=re.sub(r"[^0-9]","",tstr)[:12]      # YYYYMMDDHHMM
    for root in ("event_waveforms_ulsanfault","event_waveforms_blastclean"):
        c=sorted(glob.glob(f"{EVT}/{root}/{key}*"))
        if c: return c[0]
    return None
for t,_,_ in FAIL: print(f"  {t}  ->  {os.path.basename(evdir_from_time(t) or 'MISSING')}")""")

md(r"""## 1 · Per-station measurement (same function as nb20)""")
co(r"""def measure(evdir):
    R=[]
    for f in sorted(glob.glob(evdir+"/*Z.sac")):
        tr=obspy.read(f)[0]
        if not tr.stats.channel.endswith("Z"): continue
        s=tr.stats.sac; a=s.get("a",-12345); t0=s.get("t0",-12345); b=float(s.get("b",0))
        dist=mp.hypocentral_km(s)
        if not np.isfinite(dist): continue
        if a>-1e4: P=float(a); psrc="pick"
        else:
            epi=float(s.get("dist",np.nan)); dep=float(s.get("evdp",np.nan))
            arr=TAUP.get_travel_times(source_depth_in_km=max(dep,0.0),distance_in_degree=epi/111.195,
                                      phase_list=("p","P","Pn","Pg")) if (np.isfinite(epi) and np.isfinite(dep)) else []
            if not arr: continue
            P=float(min(x.time for x in arr)); psrc="taup"
        disp=mp.remove_response_to_disp(obspy.Stream([tr]),inv)
        if not len(disp): continue
        wa=disp[0].copy().simulate(paz_simulate=mp.WOOD_ANDERSON_PAZ); d=np.asarray(wa.data)*1000.0
        sr=wa.stats.sampling_rate; n=len(d); x=wa.times()+b
        S=(t0 if t0>-1e4 else np.nan); pr=P-b
        i_n0=int((pr-NOISE_PAD-NOISE_WIN)*sr); i_n1=int((pr-NOISE_PAD)*sr)
        if i_n0<0 or i_n1<=i_n0: continue
        i_s0=int((dist/SIG_DIV_START-b)*sr); i_s1=int((dist/SIG_DIV_END-b)*sr)
        if i_s0<0 or i_s1>n or i_s1<=i_s0+1: continue
        nz=d[i_n0:i_n1]; nrms=float(np.sqrt(np.mean(nz**2))); npk=float(np.max(np.abs(nz)))
        il=int(np.argmax(np.abs(d[i_s0:i_s1]))); pk=float(abs(d[i_s0:i_s1][il])); pidx=i_s0+il
        snr_pp=pk/npk if npk>0 else np.nan
        ML=mp.ml_heo2024(pk,dist,"Z") if pk>0 else np.nan
        R.append(dict(sta=tr.stats.station,x=x,d=d,dist=dist,P=P,S=S,psrc=psrc,
                      xn0=x[i_n0],xn1=x[i_n1-1],xs0=x[i_s0],xs1=x[i_s1-1],
                      tpk=x[pidx],apk=d[pidx],pk=pk,nrms=nrms,npk=npk,snr=snr_pp,ML=ML,used=(snr_pp>=SNR_THR)))
    R.sort(key=lambda r:r["dist"]); return R""")

md(r"""## 2 · Record sections of the 7 excluded events (HDB highlighted)

Origin at t = 0. **P** (red pick / orange-dashed theoretical), **S** (blue), **★** peak → ML, grey = noise
window, yellow = signal window. **HDB is drawn in magenta** with a bold label. Each title shows the event
ML **from the healthy anchors only** (median over used stations excluding HDB) vs **HDB's own ML** — the gap
is the corruption.""")
co(r"""def draw(ax,name,evdir):
    R=measure(evdir)
    if not R: ax.set_title(f"{name}: no data"); return None
    hl=[r for r in R if r["sta"]=="HDB"]; oth=[r for r in R if r["sta"]!="HDB" and r["used"]]
    ml_healthy=np.median([r["ML"] for r in oth]) if oth else np.nan
    ml_hdb=hl[0]["ML"] if hl else np.nan
    XL=(-3.0,min(45.0,max(r["xs1"] for r in R)+2))
    rng=(max(r["dist"] for r in R)-min(r["dist"] for r in R)) or 8.0; sc=0.85*rng/max(len(R)-1,1)
    for r in R:
        ishdb=(r["sta"]=="HDB"); m=(r["x"]>=XL[0])&(r["x"]<=XL[1]); xx=r["x"][m]; dd=r["d"][m]; amp=np.max(np.abs(dd)) or 1.0
        y0,y1=r["dist"]-sc,r["dist"]+sc
        ax.add_patch(plt.Rectangle((max(r["xn0"],XL[0]),y0),r["xn1"]-max(r["xn0"],XL[0]),y1-y0,color="0.6",alpha=0.20,lw=0))
        ax.add_patch(plt.Rectangle((r["xs0"],y0),min(r["xs1"],XL[1])-r["xs0"],y1-y0,color="#ffe08a",alpha=0.30,lw=0))
        tc="m" if ishdb else ("0.1" if r["used"] else "#c44e52")
        ax.plot(xx,r["dist"]+sc*dd/amp,lw=1.1 if ishdb else 0.6,color=tc,zorder=5 if ishdb else 2)
        pc="tab:red" if r["psrc"]=="pick" else "#ff8c00"; pls="-" if r["psrc"]=="pick" else "--"
        if np.isfinite(r["P"]): ax.plot([r["P"]]*2,[r["dist"]-sc*0.8,r["dist"]+sc*0.8],color=pc,lw=1.3,ls=pls)
        if np.isfinite(r["S"]): ax.plot([r["S"]]*2,[r["dist"]-sc*0.8,r["dist"]+sc*0.8],color="tab:blue",lw=1.3)
        if XL[0]<=r["tpk"]<=XL[1]: ax.plot(r["tpk"],r["dist"]+sc*r["apk"]/amp,"*",color="#d62728",ms=10,mec="k",mew=0.4,zorder=6)
        lab=f"{r['sta']} {r['dist']:.0f}km"
        ax.text(XL[1]-0.3,r["dist"]+sc*0.55,lab,fontsize=7.5 if ishdb else 6.5,va="bottom",ha="right",
                color="m" if ishdb else "0.1",fontweight="bold" if ishdb else "normal")
        ax.text(XL[1]-0.3,r["dist"]-sc*0.95,f"{r['snr']:.1f} {'U' if r['used'] else 'R'}",fontsize=6,va="top",ha="right",
                color="m" if ishdb else ("#1a7a1a" if r["used"] else "#c44e52"),fontweight="bold")
    ax.axvline(0,color="0.5",lw=0.9,ls=":")
    ax.set(xlim=XL,xlabel="Time since origin (s)",ylabel="Hypocentral dist (km)",
           title=f"{name}\nhealthy-anchor ML={ml_healthy:+.2f}   HDB ML={ml_hdb:+.2f}  (ΔHDB={ml_hdb-ml_healthy:+.2f})")
    return dict(ml_healthy=ml_healthy,ml_hdb=ml_hdb,nsta=len(R))
def legend(fig,y=-0.01):
    h=[mlines.Line2D([],[],color="m",lw=2,label="HDB (failure)"),mlines.Line2D([],[],color="0.1",lw=2,label="other, used"),
       mlines.Line2D([],[],color="#c44e52",lw=2,label="rejected (snr_pp<2)"),
       mlines.Line2D([],[],color="tab:red",lw=2,label="P pick"),mlines.Line2D([],[],color="tab:blue",lw=2,label="S pick"),
       mlines.Line2D([],[],color="#d62728",marker="*",ls="",ms=11,mec="k",label="peak → ML"),
       Patch(color="0.6",alpha=0.4,label="noise"),Patch(color="#ffe08a",alpha=0.5,label="signal (Sheen S)")]
    lg=fig.legend(handles=h,loc="lower center",ncol=8,fontsize=8,bbox_to_anchor=(0.5,y)); lg.set_zorder(50)
rows=[]
fig,axes=plt.subplots(2,4,figsize=(20,10))
for ax,(t,_,_) in zip(axes.ravel(),FAIL):
    r=draw(ax,t[:16],evdir_from_time(t))
    if r: rows.append(dict(event=t,**r))
axes.ravel()[-1].axis("off")
legend(fig); fig.suptitle("HDB sensor-failure window — HDB (magenta) under-reads by 1–2 ML vs the healthy anchors",y=1.0,fontsize=12.5)
fig.tight_layout(); plt.show()
summary=pd.DataFrame(rows); print(summary.to_string(index=False))""")

md(r"""## 3 · The smoking gun — 2015-03-19, HDB high-SNR but wrong amplitude

For this event HDB has `snr_pp ≈ 27` (it clearly records the earthquake) yet its ML is ≈ 2 units below the
healthy anchors. So the issue is a **broken gain**, not a weak/noisy signal — overlaying HDB against a
healthy near anchor (MKL) on the same WA scale makes the amplitude deficit explicit. A constant epoch
offset would have to add ≈ +2 ML uniformly, but the corruption is not a clean constant (the window varies
1.2–2.2 ML across events). In nb23 this window is given its **own HDB epoch** with offset ≈ −1.94 ML, so
the readings are kept and corrected — though, as the spread shows, that single offset is necessarily
imperfect per reading (it just doesn't matter: HDB is 1 of 5 stations and the event ML is a median).""")
co(r"""ev=evdir_from_time("2015-03-19 12:51:15"); R=measure(ev)
hdb=next((r for r in R if r["sta"]=="HDB"),None)
near=min([r for r in R if r["sta"]!="HDB"],key=lambda r:r["dist"]) if len(R)>1 else None
fig,ax=plt.subplots(figsize=(11,4.6))
for r,c,lab in [(near,"0.1",f"{near['sta']} (healthy, {near['dist']:.0f} km)"),(hdb,"m",f"HDB (failure, {hdb['dist']:.0f} km)")]:
    if r is None: continue
    ax.plot(r["x"],r["d"],lw=0.9,color=c,label=f"{lab}  ML={r['ML']:+.2f}  snr_pp={r['snr']:.0f}")
    ax.axvspan(r["xs0"],r["xs1"],color="#ffe08a",alpha=0.15)
    ax.plot(r["tpk"],r["apk"],"*",color="#d62728",ms=13,mec="k",zorder=6)
ax.axvline(0,color="0.5",ls=":",lw=0.9); ax.set(xlim=(-3,28),xlabel="Time since origin (s)",ylabel="Wood-Anderson displacement (mm)",
       title="2015-03-19 — HDB records the event strongly (snr_pp≈27) but its WA amplitude is ~6× too small")
ax.legend(loc="upper right"); fig.tight_layout(); plt.show()
print(f"HDB peak {hdb['pk']:.3g} mm vs {near['sta']} peak {near['pk']:.3g} mm at similar distance -> amplitude ratio {near['pk']/hdb['pk']:.1f}x")""")

md(r"""## 4 · Summary

* The HDB failure window is **7 readings (Dec 2014 – Apr 2015)**, each under-reading by **1.2–2.2 ML**.
* The deficit is a **corrupted gain**, not weak signal: on 2015-03-19 HDB has `snr_pp ≈ 27` yet sits ~2 ML
  low (peak ~23× too small *despite being the closest station*).
* nb23 gives this window its **own HDB epoch** (offset ≈ −1.94 ML), so the 7 readings are **kept and
  corrected** rather than dropped. Because the corruption is non-constant (1.2–2.2 ML), the single offset
  is imperfect per reading — but the **event ML is a median over 5 stations**, so it is **unchanged either
  way**: UF `b = 1.07`, slope `−0.0050`, `n≥3 = 888` are identical whether the window is corrected or
  excluded (the 7 events just gain HDB back, `n_const` 4→5, with ML shifts of only −0.08…+0.09).
* Take-home: for the constant-network catalog the choice is immaterial; correcting keeps the data, and
  these record sections are the evidence that the per-reading correction there should not be trusted in
  isolation (only via the multi-station median).""")

nb["cells"]=C
import os
os.chdir("/home/msseo/works/02.Ulsan_Fault_detection/analysis/local_magnitudes")
nbf.write(nb,"24.HDB_failure_window_record_sections.ipynb")
print("wrote 24.HDB_failure_window_record_sections.ipynb with",len(C),"cells")
