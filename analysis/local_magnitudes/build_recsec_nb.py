#!/usr/bin/env python
"""Generate 20.UF_record_sections.ipynb — SOTA Wood-Anderson record sections for representative UF
events, showing EXACTLY how the local-magnitude pipeline measures noise (SNR) and peak (ML amplitude).
Final scheme (matches ml_pipeline):
  * bandpass 2-20 Hz -> remove response -> Wood-Anderson (Heo 2024 / Uhrhammer-Collins).
  * P reference = PhaseNet+ pick (SAC.a) where present, else THEORETICAL kim2011 P via TauP.
  * NOISE  window: [P-6 s, P-1 s] (5 s ending 1 s before P)              -> RMS / peak -> SNR denominator.
  * SIGNAL window: Sheen (2018) [dist/4, dist/2] s after origin, HYPOCENTRAL dist -> max|WA| -> ML amp.
  * SNR = peak / noise_PEAK (zero-to-peak / peak-to-peak); reading used iff snr_pp >= 2.0.
  * dist = hypocentral = sqrt(epicentral^2 + depth^2); same dist feeds Heo Eq.3 attenuation.
Exports use b=-30 (origin at t=0; SAC a/t0 are origin-relative): P sample = (a-b)*sr.
Runs in `base`, cwd = local_magnitudes."""
import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# UF Wood-Anderson record sections — how SNR and ML amplitude are measured

These record sections show, for representative Ulsan-Fault events, **exactly the waveforms, windows and
SNR the local-magnitude pipeline uses** — so the amplitude feeding ML can be checked by eye.

**Windowing scheme** (Heo et al. 2024 scale + our SNR quality gate):

| step | definition |
|---|---|
| preprocessing | demean/detrend → **band-pass 2–20 Hz** → remove response → **Wood-Anderson** (Uhrhammer & Collins 1990) |
| **P reference** | PhaseNet+ pick (`SAC.a`) where present; otherwise the **theoretical kim2011 P** (TauP) — so a missed-P station with a clear S still contributes |
| **noise window** | `[P − 6 s, P − 1 s]` (5 s, ending 1 s before P) |
| **signal window** | **Sheen (2018) `[dist/4, dist/2]` s after origin** → max \|WA\| → **ML (S-phase) amplitude** |
| **SNR** | `peak / noise_PEAK` (zero-to-peak / peak-to-peak) |
| gate | reading **used** iff `snr_pp ≥ 2.0` |

**`dist` is the HYPOCENTRAL distance** `√(epicentral² + depth²)` (`mp.hypocentral_km`). The SAC `dist`
header is only *epicentral*; for near-source stations (epicentral ≲ depth) an epicentral `dist/4 … dist/2`
lands the S window **at or before P** (verification §2) and biases the Heo Eq. 3 term low. The same
hypocentral `dist` feeds the attenuation term, so window and correction are consistent.

The exports put **origin at t = 0** with `b = −30`; SAC `a` (P) / `t0` (S) are origin-relative, so the
P sample is `(a − b)·sr`.""")

co(r"""import warnings; warnings.filterwarnings("ignore")
import os, re, glob, numpy as np, pandas as pd, obspy
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
import matplotlib.lines as mlines; from matplotlib.patches import Patch
import ml_pipeline as mp
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":130,"font.size":9.5,"legend.framealpha":1.0,
    "legend.facecolor":"white","legend.edgecolor":"0.6"})
# windowing parameters (final scheme = ml_pipeline defaults)
NOISE_WIN, NOISE_PAD, SIG_DIV_START, SIG_DIV_END, SNR_THR = 5.0, 1.0, 4.0, 2.0, 2.0
EVT="../HypoInv"; UF=(129.25,129.55,35.60,35.90)
inv=mp.load_combined_inventory("responses/master")
TAUP=mp.kim2011_taup_model()       # theoretical kim2011 P for stations with no PhaseNet+ pick
cat=pd.read_csv("catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo_clean.csv")
uf=cat[(cat.lon.between(UF[0],UF[1]))&(cat.lat.between(UF[2],UF[3]))&(cat.mag_status=='ok')].copy()
def _stem(t):
    m=re.match(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})",str(t)); return "".join(m.groups()) if m else None
def evdir(s):
    for r in ("event_waveforms_blastclean","event_waveforms_ulsanfault"):
        d=f"{EVT}/{r}/{s}"
        if os.path.isdir(d): return d
uf["dir"]=uf.time.map(lambda t: evdir(_stem(t)))
uf=uf[uf.dir.notna()]
def pick_near(mag,**q):
    g=uf.query("n_used>=6 and time>'2016'")
    for k,v in q.items(): g=g.query(f"{k}{v}")
    return g.iloc[(g.magnitude-mag).abs().argsort()].iloc[0]
print(f"UF ok-events with waveforms: {len(uf)}   TauP model: kim2011 (matches picks to ±0.1 s)")""")

md(r"""## 1  Per-station measurement function

For one event directory this reads the vertical SACs, deconvolves to displacement (2–20 Hz), simulates
Wood-Anderson, and applies the windows above. **P reference**: the PhaseNet+ pick if present, else the
theoretical kim2011 P (TauP) — recorded as `psrc` (`pick`/`taup`). Returns per station: the WA trace,
P/S, noise + signal window extents, the measured peak (→ ML) and its time, `nrms`/`npk`, `snr_pp`,
`p_source`, and used/rejected.""")
co(r"""def measure(evdir):
    R=[]
    for f in sorted(glob.glob(evdir+"/*Z.sac")):
        tr=obspy.read(f)[0]
        if not tr.stats.channel.endswith("Z"): continue
        s=tr.stats.sac; a=s.get("a",-12345); t0=s.get("t0",-12345); b=float(s.get("b",0))
        dist=mp.hypocentral_km(s)                # HYPOCENTRAL = sqrt(epi^2 + evdp^2)
        if not np.isfinite(dist): continue
        # P reference: PhaseNet+ pick, else theoretical kim2011 P
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
        sr=wa.stats.sampling_rate; n=len(d); x=wa.times()+b            # origin at 0
        S=(t0 if t0>-1e4 else np.nan); pr=P-b
        i_n0=int((pr-NOISE_PAD-NOISE_WIN)*sr); i_n1=int((pr-NOISE_PAD)*sr)
        if i_n0<0 or i_n1<=i_n0: continue
        i_s0=int((dist/SIG_DIV_START-b)*sr); i_s1=int((dist/SIG_DIV_END-b)*sr)   # Sheen [dist/4, dist/2]
        if i_s0<0 or i_s1>n or i_s1<=i_s0+1: continue
        nz=d[i_n0:i_n1]; nrms=float(np.sqrt(np.mean(nz**2))); npk=float(np.max(np.abs(nz)))
        il=int(np.argmax(np.abs(d[i_s0:i_s1]))); pk=float(abs(d[i_s0:i_s1][il])); pidx=i_s0+il
        snr_pp=pk/npk if npk>0 else np.nan                                       # peak/peak (zero-to-peak) SNR
        ML=mp.ml_heo2024(pk,dist,"Z") if pk>0 else np.nan
        R.append(dict(sta=tr.stats.station,x=x,d=d,dist=dist,P=P,S=S,psrc=psrc,
                      xn0=x[i_n0],xn1=x[i_n1-1],xs0=x[i_s0],xs1=x[i_s1-1],
                      tpk=x[pidx],apk=d[pidx],pk=pk,nrms=nrms,npk=npk,snr=snr_pp,ML=ML,used=(snr_pp>=SNR_THR)))
    R.sort(key=lambda r:r["dist"]); return R""")

md(r"""## 2  Verification — epicentral vs hypocentral window + ML distance-term bias

The bug fixed here: the SAC `dist` header is **epicentral**, but the Sheen window and the Heo Eq. 3 term
both need **hypocentral** distance. For the flagged events — **HDB 2014-09-23**, **YGBA 2021-08-19**, and
the **near-source stations of the small 2024-03-12 event** — the table shows P/S, the **epicentral**
`[dist/4,dist/2]` window (used before) vs the **hypocentral** one (the fix), whether each brackets S, and
`dML_distterm` = how much ML was **under-estimated** purely from the epicentral distance in Heo Eq. 3.""")
co(r"""def verify(stem, only=None, dmax=99):
    d=evdir(stem); rows=[]
    for f in sorted(glob.glob(d+"/*Z.sac")):
        tr=obspy.read(f)[0]; s=tr.stats.sac; a=s.get("a",-12345); t0=s.get("t0",-12345)
        if a<-1e4: continue
        st=tr.stats.station
        if only and st not in only: continue
        epi=float(s.get("dist",np.nan)); dep=float(s.get("evdp",np.nan)); hyp=float(np.hypot(epi,dep))
        if epi>dmax: continue
        we=(epi/SIG_DIV_START,epi/SIG_DIV_END); wh=(hyp/SIG_DIV_START,hyp/SIG_DIV_END)
        S=(t0 if t0>-1e4 else np.nan)
        dML=float(mp.ml_heo2024(1.0,hyp,"Z")-mp.ml_heo2024(1.0,epi,"Z"))   # amplitude cancels -> pure distance term
        rows.append(dict(sta=st,epi=round(epi,1),depth=round(dep,1),hypo=round(hyp,1),
                         P=round(a,2),S=(np.nan if np.isnan(S) else round(S,2)),
                         epi_win=f"[{we[0]:.1f},{we[1]:.1f}]",hypo_win=f"[{wh[0]:.1f},{wh[1]:.1f}]",
                         epi_has_S=bool((not np.isnan(S)) and we[0]<=S<=we[1] and we[0]>a),
                         hypo_has_S=bool((not np.isnan(S)) and wh[0]<=S<=wh[1] and wh[0]>a),
                         dML_distterm=round(dML,2)))
    return pd.DataFrame(rows).sort_values("epi")
for label,stem,only,dm in [("HDB 2014-09-23","20140923062757",["HDB"],99),
                           ("YGBA 2021-08-19","20210819035916",["YGBA"],99),
                           ("2024-03-12 near-source (<12 km)","20240312105825",None,12)]:
    print(f"### {label}"); print(verify(stem,only,dm).to_string(index=False)); print()""")

md(r"""## 3  Record sections — large / moderate / small

Distance vs time (origin at t = 0). **P (red = pick, orange dashed = theoretical kim2011 P)**, **S (blue)**
ticks form the moveout; **★** = the peak that becomes ML (on the S/coda); **grey** = noise window,
**yellow** = signal window — shaded exactly where the pipeline measures. Each trace is labelled with its
**snr_pp** and **USED/REJ**. Titles show **catalog(old) → new(hypocentral) ML**.""")
co(r"""def draw(ax,name,ev,XL=None,fs_lab=None):
    R=measure(ev.dir); ds=[r["dist"] for r in R]
    _u=[r for r in R if r["used"]]; _newml=np.median([r["ML"] for r in _u]) if _u else np.nan
    if not R: ax.set_title(f"{name}: no data"); return _newml,len(_u),len(R)
    XL=(-3.0, min(45.0, max(r["xs1"] for r in R)+2))
    rng=(max(ds)-min(ds)) or 8.0; sc=0.85*rng/max(len(R)-1,1)
    fs=fs_lab if fs_lab else (6 if len(R)>14 else 7)
    for r in R:
        m=(r["x"]>=XL[0])&(r["x"]<=XL[1]); xx=r["x"][m]; dd=r["d"][m]; amp=np.max(np.abs(dd)) or 1.0
        y0,y1=r["dist"]-sc,r["dist"]+sc
        ax.add_patch(plt.Rectangle((max(r["xn0"],XL[0]),y0),r["xn1"]-max(r["xn0"],XL[0]),y1-y0,color="0.6",alpha=0.22,lw=0))
        ax.add_patch(plt.Rectangle((r["xs0"],y0),min(r["xs1"],XL[1])-r["xs0"],y1-y0,color="#ffe08a",alpha=0.30,lw=0))
        ax.plot(xx,r["dist"]+sc*dd/amp,lw=0.6,color="0.1" if r["used"] else "#c44e52")
        pc="tab:red" if r["psrc"]=="pick" else "#ff8c00"; pls="-" if r["psrc"]=="pick" else "--"
        if np.isfinite(r["P"]): ax.plot([r["P"]]*2,[r["dist"]-sc*0.8,r["dist"]+sc*0.8],color=pc,lw=1.5,ls=pls)
        if np.isfinite(r["S"]): ax.plot([r["S"]]*2,[r["dist"]-sc*0.8,r["dist"]+sc*0.8],color="tab:blue",lw=1.5)
        if XL[0]<=r["tpk"]<=XL[1]: ax.plot(r["tpk"],r["dist"]+sc*r["apk"]/amp,"*",color="#d62728",ms=10,mec="k",mew=0.4,zorder=6)
        ax.text(XL[1]-0.3,r["dist"]+sc*0.55,f"{r['sta']} {r['dist']:.0f}km",fontsize=fs,va="bottom",ha="right")
        ax.text(XL[1]-0.3,r["dist"]-sc*0.95,f"{r['snr']:.1f} {'U' if r['used'] else 'R'}",
                fontsize=fs-0.5,va="top",ha="right",color="#1a7a1a" if r["used"] else "#c44e52",fontweight="bold")
    ax.axvline(0,color="0.5",lw=0.9,ls=":"); ax.set(xlim=XL,xlabel="Time relative to origin (s)",ylabel="Hypocentral distance (km)",
           title=f"{name}: {ev.time[:19]}\ncatalog ML={ev.magnitude:+.2f} → new ML={_newml:+.2f}  (used {len(_u)}/{len(R)})")
    return _newml,len(_u),len(R)

def legend(fig,ncol=6,y=-0.02):
    h=[mlines.Line2D([],[],color="tab:red",lw=2,label="P pick (PhaseNet+)"),
       mlines.Line2D([],[],color="#ff8c00",lw=2,ls="--",label="P theoretical (kim2011 TauP)"),
       mlines.Line2D([],[],color="tab:blue",lw=2,label="S pick"),
       mlines.Line2D([],[],color="#d62728",marker="*",ls="",ms=11,mec="k",label="Peak → ML"),
       Patch(color="0.6",alpha=0.4,label="Noise [P−6,P−1] s"),Patch(color="#ffe08a",alpha=0.5,label="Signal [dist/4,dist/2] (Sheen S)"),
       mlines.Line2D([],[],color="#c44e52",lw=2,label="Rejected (snr_pp<2)")]
    lg=fig.legend(handles=h,loc="lower center",ncol=ncol,fontsize=8.3,bbox_to_anchor=(0.5,y)); lg.set_zorder(50)

BIG=uf.query("magnitude>3").sort_values("magnitude",ascending=False).iloc[0]
MID=pick_near(1.2)
_m=uf[uf.time.str.startswith("2024-03-12 10:58")]; SML=_m.iloc[0] if len(_m) else pick_near(0.4)
fig,axes=plt.subplots(1,3,figsize=(17,7.2))
for ax,(name,ev) in zip(axes,[("Large (reference)",BIG),("Moderate",MID),("Small",SML)]): draw(ax,name,ev)
legend(fig)
fig.suptitle("UF Wood-Anderson record sections — windows shaded exactly as the pipeline measures noise (SNR) and peak (ML)",y=1.0,fontsize=11.5)
fig.tight_layout(); plt.show()""")

md(r"""## 4  More examples — a magnitude spread

Six further events spanning ML ≈ 0.5 → 2.6, each shown as a record section with its **catalog → new ML**
and used/total station count, so the amplitude measurement can be inspected across event sizes.""")
co(r"""TARGETS=[2.6,2.0,1.5,1.0,0.7,0.5]
EX=[pick_near(t) for t in TARGETS]
fig,axes=plt.subplots(2,3,figsize=(17,11))
for ax,ev,t in zip(axes.ravel(),EX,TARGETS): draw(ax,f"ML~{t:.1f}",ev,fs_lab=6)
legend(fig,ncol=7,y=-0.01)
fig.suptitle("UF record sections across magnitudes — how the S-window peak (★) and SNR set each station ML",y=1.0,fontsize=11.5)
fig.tight_layout(); plt.show()""")

md(r"""## 5  Theoretical-P recovery (TauP) — stations with a missed P but a clear S

PhaseNet+ does not always pick P even when S is strong (near or noisy stations). Rather than drop those
stations, the pipeline places the noise window on the **theoretical kim2011 P** (orange dashed). Two
showcases:

* **2019-02-15 02:04** — *every* used station comes from theoretical P; **without TauP the event has
  zero usable stations** (no ML at all), with TauP it is measured.
* **2024-10-28 03:55** — TauP adds near stations (e.g. DUC ≈ 15 km) to an already well-recorded event;
  the ML is unchanged but rests on more observations (more robust, no distortion).

The table compares event ML and station count **with vs without** theoretical P.""")
co(r"""def ml_on_off(stem):
    d=evdir(stem)
    on =mp.per_station_ml(d,inv,attenuation_fn=mp.ml_heo2024,use_taup_for_missing_p=True ).dropna(subset=["ML"])
    off=mp.per_station_ml(d,inv,attenuation_fn=mp.ml_heo2024,use_taup_for_missing_p=False).dropna(subset=["ML"])
    return on,off
rows=[]
for label,stem in [("2019-02-15 02:04",  "20190215020420"),
                   ("2024-10-28 03:55",  "20241028035505"),
                   ("2024-03-12 10:58",  "20240312105825")]:
    on,off=ml_on_off(stem)
    rows.append(dict(event=label,
                     ML_with_TauP=round(on.ML.median(),2) if len(on) else np.nan, n_with=len(on),
                     taup_used=int((on.p_source=="taup").sum()),
                     ML_no_TauP=round(off.ML.median(),2) if len(off) else np.nan, n_no=len(off)))
print(pd.DataFrame(rows).to_string(index=False))

# record sections for the two TauP showcases
S1=uf[uf.time.str.startswith("2019-02-15 02:04")].iloc[0]
S2=uf[uf.time.str.startswith("2024-10-28 03:55")].iloc[0]
fig,axes=plt.subplots(1,2,figsize=(13,7.2))
draw(axes[0],"All-theoretical-P (no PhaseNet+ P)",S1)
draw(axes[1],"TauP adds near stations (DUC~15km)",S2)
legend(fig,ncol=7,y=-0.02)
fig.suptitle("Theoretical-P recovery — orange dashed = kim2011 TauP P; these stations would be dropped without it",y=1.0,fontsize=11.5)
fig.tight_layout(); plt.show()""")

md(r"""## 6  Single-event detail + per-station table

The per-station numbers that go into the small event ML (median of the **used** station MLs), including
the P source (`pick`/`taup`).""")
co(r"""ev=SML; R=measure(ev.dir)
tab=pd.DataFrame([{"station":r["sta"],"dist_km":round(r["dist"],1),"P":r["psrc"],"noise_pk_mm":r["npk"],
                   "peak_mm":r["pk"],"snr_pp":round(r["snr"],1),"ML":round(r["ML"],2),
                   "used":r["used"]} for r in R])
print(f"Event {ev.time[:19]}  catalog ML={ev.magnitude:+.2f}")
print(tab.to_string(index=False))
used=tab[tab.used]
print(f"\nevent ML = median of {len(used)} used-station MLs = {used.ML.median():+.2f}  (catalog {ev.magnitude:+.2f})")""")

md(r"""## 7  Summary

* **`dist` is hypocentral** `√(epicentral²+depth²)`. With the old epicentral `dist` the Sheen
  `[dist/4,dist/2]` window landed **at/before P** for near stations (§2: epi windows fail to bracket S;
  hypocentral ones succeed), and the **Heo Eq. 3 term was biased low** (`dML_distterm` ≈ 0.3–0.8 ML at
  <10 km), under-estimating near-station ML — the broken short-distance distance–ML residual.
* **No-P-pick stations are no longer dropped**: the noise window uses the **theoretical kim2011 P** (TauP,
  matched to picks within ±0.1 s). This recovers events that would otherwise be unmeasurable (2019-02-15:
  0 → 3 stations) and strengthens others (2024-10-28: 8 → 10) without changing their ML.
* The **peak that becomes ML sits on the S/coda** (★) for every used station; **noise (grey)** is 5 s
  pre-P, **signal (yellow)** is the hypocentral Sheen S window — both shaded exactly where measured.
* Gate = **`snr_pp ≥ 2.0`** (zero-to-peak peak / peak noise); event ML = **median of used station MLs**.
* Scale = Heo et al. (2024): 2–20 Hz, Wood-Anderson, vertical peak, hypocentral distance term Eq. 3.""")

nb["cells"]=C
import os
os.chdir("/home/msseo/works/02.Ulsan_Fault_detection/analysis/local_magnitudes")
nbf.write(nb,"20.UF_record_sections.ipynb")
print("wrote 20.UF_record_sections.ipynb with",len(C),"cells")
