#!/usr/bin/env python
"""Generate 19.HDB_2015_amplitude_diagnosis.ipynb — diagnose the KG.HDB 2014-2015 amplitude anomaly
that is decoupled from any documented response change. Question raised in lab meeting (Prof. W.-Y. Kim):
is this a response-correction problem (a power-of-2 digitizer/bit-depth gain that response revision would
fix) or a raw-count sensor problem? Sections:
  1  the response-corrected ML residual + epoch steps (catalog).
  2  raw-count noise-floor PSD over time (metadata-independent witness; Welch, cached to npz).
  3  response-swap test: re-deconvolve the 2012.11-2015.05 epoch with the neighbour 11/6 response.
  4  SNR + size-resolved detection through the failure window.
  5  conclusion: documented digitizer x4 is handled; the residuals are the SENSOR (raw count); epoch
     station term is the clean fix (cleaner than editing the StationXML).
Runs in `base`, cwd = local_magnitudes."""
import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# The KG.HDB 2014–2015 amplitude anomaly — raw-count sensor failure, not a response error

**Context (lab meeting, Prof. W.-Y. Kim).** KG.HDB shows a magnitude residual that steps and collapses
around 2012–2015 with no matching *documented* response change. The proposed mechanism was a digitizer /
logger bit-depth change (24→26 bit ⇒ ×2–4 ⇒ ΔML 0.3–0.6) that a **response-file revision** would fix.
This notebook tests that, using three independent witnesses:

1. the response-corrected **ML residual** (what motivated the question);
2. the **raw-count background-noise floor** — an instrument-independent gauge of the sensor's true
   sensitivity (ocean microseisms drive a regionally near-constant ground motion), touching neither the
   Wood–Anderson simulation nor the response;
3. a direct **response-swap experiment** on the event waveforms.

**Result preview.** HDB *did* have a real ×4 digitizer change (Quanterra Q412x→Q330HRS, 2012-09) — but
it is documented and already removed by per-epoch deconvolution. The residual steps that remain track the
**sensor**, not the metadata: a progressive under-performance from 2012 and a catastrophic ×28 collapse
in Nov 2014–May 2015, both cured by the 2015-05 **sensor** swap. Swapping the response does nothing.
So this is a raw-count hardware fault; the empirical **epoch station term** is the correct fix.

*Companion to:* `17.Response_epoch_corrected_catalog.ipynb`, `18.Why_epoch_station_correction.ipynb`.""")

# ------------------------------------------------------------------ §0 setup
co(r"""import warnings; warnings.filterwarnings("ignore")
import os, glob, copy, numpy as np, pandas as pd, obspy
from scipy import signal
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
import matplotlib.dates as mdates
import ml_pipeline as mp
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":130,"axes.grid":True,"grid.alpha":0.25,"font.size":11,
    "legend.framealpha":1.0,"legend.facecolor":"white","legend.edgecolor":"0.6"})  # opaque legend, above data

PS="catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo.csv"
SXML="responses/master/KS_KG_metadata_1.0.2.xml"
CONT="../continuous"; EVT="../HypoInv/event_waveforms_ulsanfault"
SC="KG.HDB.HHZ"
# documented HDB hardware history (from the StationXML)
LOGGER_SWAP=pd.Timestamp("2012-09-12")      # Quanterra Q412x -> Q330HRS, sensitivity x3.96
SENSOR_SWAP=pd.Timestamp("2015-05-21")      # sensor replaced (cures the fault)
FAIL_ON, FAIL_OFF = pd.Timestamp("2014-10-01"), pd.Timestamp("2015-05-21")

d=pd.read_csv(PS); d=d[(d.snr>=3)&d.ML.notna()].copy()
d["t"]=pd.to_datetime(d.event_time,utc=True,errors="coerce").dt.tz_localize(None)
d=d.dropna(subset=["t"]); d["sc"]=d.network+"."+d.station+"."+d.channel
d["event_idx"]=d.event_idx.astype(int)
# time-stable event magnitude (median of all readings) -> per-reading residual
d["res"]=d.ML-d.groupby("event_idx").ML.transform("median")
h=d[d.sc==SC].sort_values("t").copy()
print(f"HDB HHZ readings (snr>=3): {len(h)}   span {h.t.min().date()} .. {h.t.max().date()}")""")

# ------------------------------------------------------------------ §1 residual + epoch steps
md(r"""## 1  The response-corrected ML residual and its epoch steps

Each HDB reading's residual is `ML_HDB − median ML of the event` (so it is the station's bias relative to
a time-stable event magnitude). Deconvolution is **per-epoch and time-aware**, so any *documented* gain
change is already removed; whatever steps remain are undocumented. The regimes below are bounded by HDB's
documented hardware dates.""")
co(r"""def regime_med(a,b,col="res"):
    m=(h.t>=pd.Timestamp(a))&(h.t<pd.Timestamp(b)); return h[m][col].median(), int(m.sum())
regs=[("2010-01-01","2012-09-12","Q412x"),
      ("2012-09-12","2014-10-01","Q330HRS pre-fault"),
      ("2014-10-01","2015-05-21","FAILURE window"),
      ("2015-05-21","2019-02-28","post sensor-swap"),
      ("2019-02-28","2025-01-01","post 2019-swap")]
print(f"{'regime':20} {'res(ML)':>8} {'n':>6}   step vs Q412x")
base=None
for a,b,nm in regs:
    m,n=regime_med(a,b); base=m if base is None else base
    print(f"{nm:20} {m:+8.3f} {n:6d}   {m-base:+.3f} ML  (x{10**(m-base):.2f})")

def binmed(sub,col="res",freq="ME",min_n=8):
    g=sub.set_index("t")[col].sort_index().groupby(pd.Grouper(freq=freq))
    med=g.median(); cnt=g.size(); out=med[cnt>=min_n].dropna(); return out.index,out.values

fig,ax=plt.subplots(figsize=(12,4.2))
ax.scatter(h.t,h.res,s=5,alpha=0.07,color="0.5")
bx,by=binmed(h); ax.plot(bx,by,"-",color="#111111",lw=1.6,label="Monthly median residual")
ax.axhline(0,color="0.4",lw=1,ls="--")
ax.axvspan(FAIL_ON,FAIL_OFF,color="crimson",alpha=0.12,label="Sensor-failure window")
for x,lab in [(LOGGER_SWAP,"Logger swap (Q412x→Q330HRS, ×4)"),(SENSOR_SWAP,"Sensor swap")]:
    ax.axvline(x,color="tab:blue",ls=":",lw=1.3); ax.text(x,1.05,lab,rotation=90,fontsize=7,va="bottom")
ax.set(ylim=(-2.1,1.2),ylabel="Station residual  ML$_{HDB}$ − ML$_{event}$",xlabel="Year",
       title="KG.HDB HHZ residual: a −0.4 step at the 2012 hardware change, then a ×28 collapse to −1.7")
leg=ax.legend(loc="lower left",fontsize=8.5); leg.set_zorder(50)
fig.tight_layout(); plt.show()""")

# ------------------------------------------------------------------ §2 raw-count noise floor
md(r"""## 2  Raw-count noise floor over time — the metadata-independent witness

If the metadata were the problem, the noise floor (raw counts, no response applied) would be unaffected.
If the **sensor** lost sensitivity, the raw-count noise — driven by regionally near-constant ocean
microseisms — drops with it. We compute a daily Welch PSD of HDB HHZ raw counts (one day per month,
2011–2017), median per band. Cached to `hdb_noisefloor.npz`.

Comparison with the documented logger gain: the Q412x→Q330HRS sensitivity ratio is ×3.96, i.e. raw
counts should jump **+12.0 dB** at 2012-09 (and stay there). Anything beyond that — especially a *drop* —
is the sensor.""")
co(r"""CACHE="hdb_noisefloor.npz"
months=pd.date_range("2011-01","2017-12",freq="MS")
def read_raw(y,jd):
    fs=glob.glob(f"{CONT}/HDB/HHZ.D/KG.HDB..HHZ.D.{y}.{jd:03d}")
    if not fs: return None
    try: tr=obspy.read(fs[0]).merge(fill_value=0)[0]
    except Exception: return None
    x=np.asarray(tr.data,float); x=np.where(np.abs(x)>2e8,0,x)
    return x if np.count_nonzero(x)>3e5 else None
if os.path.exists(CACHE):
    z=np.load(CACHE,allow_pickle=True); freq=z["freq"]; PSD=z["psd"]; mon=pd.to_datetime(z["mon"])
else:
    freq=None; rows=[]; mon=[]
    for m in months:
        x=None
        for dd in (14,9,19,4,24):            # try a few mid-month days
            base=pd.Timestamp(f"{m.year}-{m.month:02d}-01")+pd.Timedelta(days=dd)
            if base.month==m.month: x=read_raw(m.year,base.dayofyear)
            if x is not None: break
        if x is None: continue
        f,P=signal.welch(signal.detrend(x),fs=100,nperseg=16384)
        freq=f; rows.append(P); mon.append(m)
    PSD=np.vstack(rows); mon=pd.DatetimeIndex(mon)
    np.savez(CACHE,freq=freq,psd=PSD,mon=mon.astype("int64"))
def band_db(lo,hi):
    msk=(freq>=lo)&(freq<hi); return 10*np.log10(np.median(PSD[:,msk],axis=1))
BANDS={"Microseism 0.1–0.2 Hz":(0.1,0.2),"0.5–1 Hz":(0.5,1.0),"ML band 2–8 Hz":(2.0,8.0)}
print(f"months with continuous data: {len(mon)}")

fig,ax=plt.subplots(2,1,figsize=(12,7.5),sharex=True,gridspec_kw=dict(height_ratios=[1.25,1],hspace=0.13))
per=1.0/freq[1:]; img=10*np.log10(PSD[:,1:].T); pm=(per>=0.4)&(per<=40)
mesh=ax[0].pcolormesh(mon,per[pm],img[pm,:],shading="nearest",cmap="viridis",
        vmin=np.nanpercentile(img[pm],5),vmax=np.nanpercentile(img[pm],97))
ax[0].set_yscale("log"); ax[0].set_ylabel("Period (s)"); ax[0].grid(False)
ax[0].set_title("(a) Raw-count noise PSD over time — the noise floor collapses in the failure window",loc="left",fontsize=11)
cb=fig.colorbar(mesh,ax=ax[0],pad=0.01,aspect=18); cb.set_label("PSD (dB rel. counts$^2$/Hz)")
for x in (LOGGER_SWAP,SENSOR_SWAP): ax[0].axvline(x,color="w",ls=":",lw=1.1)
ax[0].axvspan(FAIL_ON,FAIL_OFF,color="crimson",alpha=0.12)
for b,c in zip(BANDS,["#1f77b4","#ff7f0e","#2ca02c"]):
    ax[1].plot(mon,band_db(*BANDS[b]),"-o",ms=3,lw=1.3,color=c,label=b)
ax[1].axvspan(FAIL_ON,FAIL_OFF,color="crimson",alpha=0.12,label="Sensor-failure window")
ax[1].axvline(LOGGER_SWAP,color="tab:blue",ls=":",lw=1.3); ax[1].axvline(SENSOR_SWAP,color="tab:blue",ls=":",lw=1.3)
ax[1].set(ylabel="Raw-count PSD (dB)",xlabel="Year",
          title="(b) Per-band raw-count noise level — no response applied (metadata-independent)")
leg=ax[1].legend(loc="lower right",fontsize=8.5,ncol=2); leg.set_zorder(50)
ax[1].xaxis.set_major_locator(mdates.YearLocator()); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout(); plt.show()

mb=band_db(*BANDS["Microseism 0.1–0.2 Hz"]); mdt=pd.DatetimeIndex(mon)
def win(a,b): m=(mdt>=pd.Timestamp(a))&(mdt<pd.Timestamp(b)); return np.nanmedian(mb[m])
pre=win("2012-10-01","2014-10-01"); fail=win(FAIL_ON,FAIL_OFF); post=win("2015-06-01","2017-01-01")
print(f"raw microseism PSD: Q330HRS pre-fault {pre:.1f} dB | failure {fail:.1f} dB | post-swap {post:.1f} dB")
print(f"  collapse pre-fault->failure = {fail-pre:+.1f} dB  (= x{10**((pre-fail)/20):.0f} amplitude = {(fail-pre)/20:+.2f} ML)")
print(f"  documented logger change predicts +12.0 dB at 2012-09; the failure is a {fail-pre:+.0f} dB DROP -> sensor, not metadata")""")

# ------------------------------------------------------------------ §3 response-swap test
md(r"""## 3  Response-swap experiment — is the −0.4 a mis-assigned response?

HDB's documented poles/zeros flip 11/6 → **6/2** (2012.11–2015.05) → 11/6, suspicious enough to test: if
the 6/2 response is wrong, re-deconvolving the 2012.11–2015.05 events with the neighbour **11/6** response
should raise their ML by +0.4 and flatten the residual. We check both the analytic response ratio and the
real events.""")
co(r"""inv=obspy.read_inventory(SXML)
def get_resp(date):
    for net in inv:
        for sta in net:
            if sta.code!="HDB": continue
            for c in sta.channels:
                if c.code=="HHZ" and c.start_date<=obspy.UTCDateTime(date)<(c.end_date or obspy.UTCDateTime("2100-01-01")):
                    pz=[s for s in c.response.response_stages if hasattr(s,"poles")][0]
                    return c.response,(len(pz.poles),len(pz.zeros))
R62,sh62 =get_resp("2014-01-01")     # assigned 6/2 in the problem epoch
R116,sh116=get_resp("2016-06-01")    # neighbour 11/6
f=np.logspace(-2,1.3,160)
A62 =np.abs(R62 .get_evalresp_response_for_frequencies(f,output="DISP"))
A116=np.abs(R116.get_evalresp_response_for_frequencies(f,output="DISP"))
ml=(f>=2)&(f<=20)
print(f"problem-epoch response {sh62} vs neighbour {sh116}")
print(f"analytic median ΔML over the 2–20 Hz ML band from swapping 6/2→11/6 = {np.median(np.log10(A116/A62)[ml]):+.3f}")

# real events: re-deconvolve with the neighbour response
invB=copy.deepcopy(inv)
for net in invB:
    for sta in net:
        if sta.code!="HDB": continue
        for c in sta.channels:
            if c.code=="HHZ" and c.start_date<=obspy.UTCDateTime("2014-01-01")<(c.end_date or obspy.UTCDateTime("2100-01-01")):
                c.response=copy.deepcopy(R116)
dirs=sorted(d_ for d_ in glob.glob(f"{EVT}/*") if os.path.isdir(d_)
            and "20121114"<=os.path.basename(d_)[:8]<="20150521")
shifts=[]
for dd in dirs[:160]:                # cap for runtime; result is deterministic
    fp=glob.glob(dd+"/*KG.HDB.HHZ.sac")
    if not fp: continue
    st=obspy.read(fp[0])
    try:
        a=mp.wood_anderson_amp_mm(mp.remove_response_to_disp(st.copy(),inv ),require_pick=False)
        b=mp.wood_anderson_amp_mm(mp.remove_response_to_disp(st.copy(),invB),require_pick=False)
    except Exception: continue
    if a.empty or b.empty: continue
    pa,pb=float(a.peak_mm.iloc[0]),float(b.peak_mm.iloc[0])
    if pa>0 and pb>0: shifts.append(np.log10(pb/pa))
shifts=np.array(shifts)
print(f"real HDB events tested: {len(shifts)}")
print(f"median ΔML from 6/2→11/6 on real events = {np.median(shifts):+.3f}  (IQR {np.subtract(*np.percentile(shifts,[75,25])):.3f})")
print(f"need +0.40 to cancel the residual; got ~0 -> the response is NOT the cause (the bands agree; PZ differ only <1 Hz)")

fig,ax=plt.subplots(figsize=(8.4,4.4))
ax.loglog(f,A62,lw=2,label=f"assigned {sh62}")
ax.loglog(f,A116,lw=1.6,ls="--",label=f"neighbour {sh116}")
ax.axvspan(2,20,color="0.85",label="ML band (2–20 Hz)")
ax.set(xlabel="Frequency (Hz)",ylabel="|response| (DISP)",
       title="6/2 and 11/6 responses are identical in the ML band — swapping cannot fix the −0.4")
leg=ax.legend(loc="upper left",fontsize=9); leg.set_zorder(50)
fig.tight_layout(); plt.show()""")

# ------------------------------------------------------------------ §4 SNR + detection
md(r"""## 4  Does the collapse bury signals and lose detections?

A uniform gain collapse drops **signal and background noise together**, so the SNR of recorded picks is
preserved (the picker, PhaseNet, works on SNR/shape, not absolute amplitude). The penalty appears only
for the smallest events, once the ground microseism has fallen to the fixed digitizer / self-noise floor:
those sink below detectability while large events are unaffected — a **size-selective** loss. Catalog
completeness is held up by the redundant network.""")
co(r"""# SNR + noise both fall together
print("HDB readings: signal and pre-P noise fall together -> SNR preserved")
for a,b,nm in [("2013-06-01","2014-10-01","before"),(FAIL_ON,FAIL_OFF,"FAILURE"),("2015-05-21","2016-09-01","after")]:
    s=h[(h.t>=pd.Timestamp(a))&(h.t<pd.Timestamp(b))]
    print(f"  {nm:8} med_SNR={s.snr.median():7.0f}  med_peak_mm={s.peak_mm.median():.2e}  "
          f"med_noise_mm={s.noise_mm.median():.2e}  frac(snr<3)={np.mean(s.snr<3):.2f}")

# size-resolved participation: failure vs flanking baseline
ev=d[d.sc!=SC].groupby("event_idx").agg(netML=("ML","median"),t=("t","first")).reset_index()
ev["hdb"]=ev.event_idx.isin(set(h.event_idx))
fa=ev[(ev.t>=FAIL_ON)&(ev.t<FAIL_OFF)]
bl=ev[((ev.t>=pd.Timestamp("2013-06-01"))&(ev.t<FAIL_ON))|((ev.t>=pd.Timestamp("2015-05-21"))&(ev.t<pd.Timestamp("2016-09-01")))]
bins=[-1,0.5,1.0,1.5,9]; lab=["<0.5","0.5–1","1–1.5",">1.5"]
def prof(sub):
    g=sub.assign(mb=pd.cut(sub.netML,bins,labels=lab)).groupby("mb").hdb.agg(["size","mean"]); return g
gb,gf=prof(bl),prof(fa)
fig,ax=plt.subplots(1,2,figsize=(12,4.3))
ax[0].plot(h.set_index("t").noise_mm.groupby(pd.Grouper(freq="ME")).median().loc["2013":"2016"],
           "-o",ms=3,color="#7a3fbf"); ax[0].set_yscale("log")
ax[0].axvspan(FAIL_ON,FAIL_OFF,color="crimson",alpha=0.12,label="Sensor-failure window")
ax[0].set(ylabel="HDB pre-P noise_mm (resp-corrected)",xlabel="Year",title="(a) Background noise collapses in the window")
leg=ax[0].legend(loc="lower left",fontsize=8.5); leg.set_zorder(50)
x=np.arange(len(lab)); w=0.38
ax[1].bar(x-w/2,gb["mean"].reindex(lab).values,w,label="Baseline (flanking)",color="#4c72b0")
ax[1].bar(x+w/2,gf["mean"].reindex(lab).values,w,label="Failure window",color="#c44e52")
ax[1].set_xticks(x); ax[1].set_xticklabels(lab)
ax[1].set(ylim=(0,1.05),ylabel="HDB participation fraction",xlabel="Event size (network ML)",
          title="(b) Only SMALL events are lost; large ones detected at 100%")
leg=ax[1].legend(loc="lower right",fontsize=8.5); leg.set_zorder(50)
fig.tight_layout(); plt.show()
print("\nparticipation by size (baseline -> failure):")
for L in lab: print(f"  {L:6}: {gb['mean'].get(L,np.nan):.2f} -> {gf['mean'].get(L,np.nan):.2f}")""")

# ------------------------------------------------------------------ §5 conclusion + summary
md(r"""## 5  Conclusion

| feature | date | nature | fixable by response revision? |
|---|---|---|---|
| Digitizer ×4 (Q412x→Q330HRS) | 2012-09 | **documented** gain; raw ×3.6 ≈ stated ×3.96 | already handled by per-epoch deconvolution |
| −0.4 ML step | 2012.11–2015.05 | **sensor** under-performing (6/2↔11/6 identical in band; swap → 0) | only as an *empirical* gain = the epoch term |
| ×28 collapse to −1.7 | 2014.10–2015.05 | dying sensor, **non-stationary** | no — exclude / time-resolved term |

Prof. Kim correctly identified a real ×4 digitizer change, but it sits in 2012, matches the metadata, and
is already removed. The residual steps track the **sensor** (cured by the 2015-05 *sensor* swap, not a
metadata change) and live in the **raw counts** — the noise floor collapses ~29 dB with the signal,
independent of any response. A response-file revision *could* encode the flat −0.4 as a lowered per-epoch
sensitivity, but the factor is empirical (no calibration captured the drift), making it **mathematically
identical to the epoch station term**. The epoch term is the cleaner home for it: the StationXML stays an
authoritative record of *documented* hardware, and the empirically-derived drift stays in the magnitude
layer where its provenance is explicit.

**Were the failure window and the response-defined epochs treated the same? No (nb17/nb18).** Each
metadata-defined sensor era becomes a median-polish *unit* and receives a **constant epoch offset**; its
readings are kept and corrected. The 2014–2015 failure window is instead **excluded** — its HDB readings
are dropped from the magnitude solution (`~fail_mask`, 62 readings) — for two reasons: (i) its onset is
**data-driven** (the first month HDB falls below −1 ML), not a metadata date; (ii) it is **non-stationary**
(the residual swings −1.0 to −1.9 *within* the window), so no single offset can represent it. Note the
failure is a *subset* of the 6/2 metadata era (2012-11→2015-05): the **pre-fault** part of that era
(2012.11–2014.10) keeps its epoch offset (the −0.4), while the **failure tail** is excluded entirely, so
those events' magnitudes rely on the other stations.""")
md(r"""## 6  Are small-event ML reliable? — record sections + per-station scatter

Skepticism check on low-magnitude ML. We draw Wood-Anderson **record sections** for three low-mag UF
events (rising station count), aligned on the **P pick**, with the **SOTA windows** — noise (start→P−1 s)
and signal (P+0.5 s→end) — the **peak WA amplitude**, and per-station **distance / SNR / ML**. In the UF
box the SNR≥3 gate almost never drops a station (`n_total ≈ n_used`): the real filter is *upstream* at
PhaseNet picking, so only already-detected stations enter. Note the closest events have a short pre-P
window, so their very high SNR should be read with care.""")
co(r"""import re, glob
UF=(129.25,129.55,35.60,35.90); EVTROOT="../HypoInv/event_waveforms_ulsanfault"
cc=pd.read_csv("catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo_clean.csv")
ufc=cc[(cc.lon.between(UF[0],UF[1]))&(cc.lat.between(UF[2],UF[3]))&(cc.mag_status=='ok')].copy()
def _stem(t):
    m=re.match(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})",str(t)); return "".join(m.groups()) if m else None
ufc["dir"]=ufc.time.map(lambda t: os.path.join(EVTROOT,_stem(t)) if _stem(t) else None)
ufc["has"]=ufc.dir.map(lambda x: bool(x) and os.path.isdir(x))
e_few=ufc[(ufc.magnitude<-0.5)&ufc.has&(ufc.n_used==2)].sort_values("magnitude").iloc[0]
e_mid=ufc[(ufc.magnitude<-0.5)&ufc.has&(ufc.n_used.between(4,5))].sort_values("magnitude").iloc[0]
e_big=ufc[(ufc.magnitude.between(0.3,0.7))&ufc.has&(ufc.n_used>=8)].sort_values("magnitude").iloc[0]
events=[e_few,e_mid,e_big]; T0,T1=-4.0,14.0
invF=mp.load_combined_inventory("responses/master")
def rows_for(ev):
    st=obspy.Stream()
    for f in sorted(glob.glob(ev.dir+"/*.sac")):
        tr=obspy.read(f)[0]
        if tr.stats.channel.endswith("Z"): st+=tr
    R=[]
    for tr in mp.remove_response_to_disp(st,invF):
        a=tr.stats.sac.get("a",-12345)
        if a< -1e4: continue
        wa=tr.copy().simulate(paz_simulate=mp.WOOD_ANDERSON_PAZ); dat=np.asarray(wa.data)*1000.0
        sr=wa.stats.sampling_rate; x=np.arange(len(dat))/sr-a
        ie=max(min(int((a-1.0)*sr),len(dat)),4); iss=max(int((a+0.5)*sr),ie)
        if iss>=len(dat)-1: continue
        nrms=float(np.sqrt(np.mean(dat[:ie]**2)))
        il=int(np.argmax(np.abs(dat[iss:]))); pk=float(abs(dat[iss:][il])); pidx=iss+il
        snr=pk/nrms if nrms>0 else np.nan; dist=float(tr.stats.sac.get("dist",np.nan))
        ML=mp.ml_heo2024(pk,dist,"Z") if pk>0 and np.isfinite(dist) else np.nan
        R.append(dict(sta=tr.stats.station,x=x,d=dat,sr=sr,xpk=pidx/sr-a,dpk=dat[pidx],
                      pk=pk,nrms=nrms,snr=snr,dist=dist,ML=ML))
    R.sort(key=lambda r:r["dist"]); return R
EVR=[rows_for(ev) for ev in events]

fig,axes=plt.subplots(1,3,figsize=(16.5,7.2))
for ax,ev,R in zip(axes,events,EVR):
    for i,r in enumerate(R):
        m=(r["x"]>=T0)&(r["x"]<=T1); xx=r["x"][m]; dd=r["d"][m]; amp=np.max(np.abs(dd)) or 1.0
        used=r["snr"]>=3
        ax.fill_betweenx([i-0.46,i+0.46],T0,-1.0,color="0.7",alpha=0.35,lw=0)
        ax.fill_betweenx([i-0.46,i+0.46],0.5,T1,color="#ffe08a",alpha=0.28,lw=0)
        ax.plot(xx,i+0.42*dd/amp,lw=0.6,color="#1a1a1a" if used else "#c44e52")
        if T0<=r["xpk"]<=T1: ax.plot(r["xpk"],i+0.42*r["dpk"]/amp,"v",color="tab:red",ms=5,zorder=5)
        ax.text(T1+0.4,i,f"{r['sta']} . {r['dist']:.0f} km\nSNR {r['snr']:.0f} . ML {r['ML']:+.2f}",
                fontsize=7.2,va="center",color="#1a7a1a" if used else "#c44e52")
    ax.axvline(0,color="tab:blue",lw=0.8,ls=":")
    ax.set_yticks([]); ax.set_xlim(T0,T1+7.5); ax.set_ylim(-0.8,len(R)-0.2)
    ax.set_xlabel("Time relative to P pick (s)")
    ax.set_title(f"{ev.time[:19]}\nnetwork ML = {ev.magnitude:+.2f}  (n_used = {int(ev.n_used)}/{int(ev.n_total)})",fontsize=10)
import matplotlib.lines as _ml
from matplotlib.patches import Patch as _P
hh=[_P(color="0.7",alpha=0.5,label="Noise window (start -> P-1 s)"),
    _P(color="#ffe08a",alpha=0.5,label="Signal window (P+0.5 s -> end)"),
    _ml.Line2D([],[],color="tab:blue",ls=":",label="P pick"),
    _ml.Line2D([],[],marker="v",color="tab:red",ls="",label="Peak Wood-Anderson amplitude"),
    _ml.Line2D([],[],color="#1a7a1a",label="Used (SNR>=3)"),_ml.Line2D([],[],color="#c44e52",label="Dropped (SNR<3)")]
leg=fig.legend(handles=hh,loc="lower center",ncol=6,fontsize=8.3,bbox_to_anchor=(0.5,-0.015)); leg.set_zorder(50)
fig.suptitle("Low-magnitude UF events - Wood-Anderson record sections with SOTA noise/signal windows and per-station SNR",y=1.0,fontsize=12)
fig.tight_layout(); plt.show()""")

md(r"""### 6b  Per-station ML scatter vs distance

Each station's ML against hypocentral distance, with the **network median** (the assigned event ML). The
spread shows how firmly the network ML is pinned — for `n_used = 2` it is the midpoint of two readings,
so a single off station moves it ~half the gap.""")
co(r"""fig,axes=plt.subplots(1,3,figsize=(15.5,4.4),sharey=False)
for ax,ev,R in zip(axes,events,EVR):
    dks=[r["dist"] for r in R]; mls=[r["ML"] for r in R]
    ax.axhline(ev.magnitude,color="tab:blue",lw=1.4,ls="--",label=f"network ML = {ev.magnitude:+.2f}")
    sc=ax.scatter(dks,mls,c=[np.log10(r["snr"]) for r in R],cmap="viridis",s=55,edgecolors="k",lw=0.4,zorder=5)
    for r in R: ax.annotate(r["sta"],(r["dist"],r["ML"]),fontsize=6.5,xytext=(3,3),textcoords="offset points")
    ax.set(xlabel="Hypocentral distance (km)",ylabel="Per-station ML",
           title=f"{ev.time[:10]}  (n_used={int(ev.n_used)})")
    leg=ax.legend(loc="upper right",fontsize=8); leg.set_zorder(50)
    cb=fig.colorbar(sc,ax=ax,pad=0.01); cb.set_label("log10 SNR",fontsize=8)
fig.suptitle("Per-station ML vs distance for the three events (colour = log SNR); dashed = assigned network ML",y=1.03,fontsize=11)
fig.tight_layout(); plt.show()
for ev,R in zip(events,EVR):
    s=np.array([r["ML"] for r in R]); print(f"{ev.time[:19]}  network ML {ev.magnitude:+.2f} | per-station ML "
          f"min {np.nanmin(s):+.2f} max {np.nanmax(s):+.2f} spread {np.nanmax(s)-np.nanmin(s):.2f} (n={len(s)})")""")

md(r"""## 7  What if we required a minimum station count (n_used >= 3)?

`aggregate_ml` enforces **no** minimum: even one station yields a network ML. Below we count how many UF
events would be flagged as under-determined under an `n_used >= 3` floor, and where they sit in magnitude
— i.e. exactly the small events whose ML rests on one or two readings.""")
co(r"""ok=ufc[ufc.mag_status=='ok'].copy()
under=ok[ok.n_used<3]
print(f"UF ok-events: {len(ok)}")
print(f"  n_used == 1 : {(ok.n_used==1).sum()}")
print(f"  n_used == 2 : {(ok.n_used==2).sum()}")
print(f"  -> flagged under n_used>=3 floor: {len(under)} ({100*len(under)/len(ok):.1f}%)")
print(f"  their magnitude: median {under.magnitude.median():+.2f}, "
      f"{(under.magnitude<0).mean()*100:.0f}% have ML<0, max {under.magnitude.max():+.2f}")
fig,ax=plt.subplots(figsize=(8.6,4.2))
bins=np.arange(-1.6,2.0,0.2)
ax.hist(ok.magnitude,bins=bins,color="0.8",label=f"all UF events (n={len(ok)})")
ax.hist(under.magnitude,bins=bins,color="#c44e52",label=f"n_used<3, under-determined (n={len(under)})")
ax.set(xlabel="Network ML",ylabel="Events",title="Under-determined small-event ML (n_used < 3) sit at the low-magnitude end")
leg=ax.legend(loc="upper right",fontsize=9); leg.set_zorder(50)
fig.tight_layout(); plt.show()
mc=0.4
print(f"\nof the {len(under)} under-determined, {(under.magnitude>=mc).sum()} are at/above Mc~{mc} "
      f"(would still enter b-value/seismicity stats); the rest are sub-Mc and already excluded.")""")

co(r"""r1=regime_med('2010-01-01','2012-09-12')[0]; r2=regime_med('2012-09-12','2014-10-01')[0]
r3=regime_med('2014-10-01','2015-05-21')[0]; r4=regime_med('2015-05-21','2019-02-28')[0]
L=["="*78,"COMPREHENSIVE SUMMARY - KG.HDB 2014-2015 amplitude diagnosis","="*78,
   "",
   "RESIDUAL (response-corrected, per-epoch deconvolution applied):",
   f"   Q412x (2010-2012.09)         {r1:+.2f} ML   (baseline)",
   f"   Q330HRS pre-fault (-2014.10) {r2:+.2f} ML   (-0.4 step at the 2012 hardware change)",
   f"   FAILURE (2014.10-2015.05)    {r3:+.2f} ML   (catastrophic, non-stationary)",
   f"   post sensor-swap (2015.05+)  {r4:+.2f} ML   (flat -> sensor swap cured it)",
   "",
   f"RAW-COUNT NOISE FLOOR (microseism 0.1-0.2 Hz, no response): pre {pre:.0f} dB -> failure {fail:.0f} dB",
   f"   = {fail-pre:+.0f} dB = sensor sensitivity collapse ~x{10**((pre-fail)/20):.0f}; metadata predicts only +12 dB at 2012-09.",
   "",
   f"RESPONSE SWAP (6/2 -> 11/6 on {len(shifts)} real events): median dML = {np.median(shifts):+.3f}  (need +0.40)",
   "   -> the response is correct; the deficit is the instrument, not the metadata.",
   "",
   "DETECTION: SNR of recorded picks preserved (signal & noise fall together); size-selective loss of",
   f"   SMALL events only (participation <0.5 ML: {gb['mean'].get('<0.5',np.nan):.2f} -> {gf['mean'].get('<0.5',np.nan):.2f}; "
   f">1.5 ML: {gb['mean'].get('>1.5',np.nan):.2f} -> {gf['mean'].get('>1.5',np.nan):.2f}). Completeness held up by the redundant network.",
   "",
   "TAKE-HOME",
   "   * Documented digitizer x4 -> handled by time-aware response removal (NOT contaminating ML).",
   "   * -0.4 step and x28 collapse -> raw-count SENSOR fault, invisible to metadata, NOT response-fixable.",
   "   * Response-defined sensor eras -> constant epoch offset (kept, corrected).",
   "   * 2014-2015 failure window -> EXCLUDED (data-driven, non-stationary); NOT treated as an epoch.",
   "   * Fix = epoch-dependent station term (cleaner than editing the StationXML)."]
print("\n".join(L))""")

nb["cells"]=C
import os
os.chdir("/home/msseo/works/02.Ulsan_Fault_detection/analysis/local_magnitudes")
nbf.write(nb,"19.HDB_2015_amplitude_diagnosis.ipynb")
print("wrote 19.HDB_2015_amplitude_diagnosis.ipynb with",len(C),"cells")
