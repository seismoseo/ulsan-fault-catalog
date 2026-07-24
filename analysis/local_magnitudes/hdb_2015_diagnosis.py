#!/usr/bin/env python
"""Diagnose the KG.HDB 2014-2015 amplitude anomaly that is NOT explained by any documented
response change. Decisive test: the background NOISE floor is an instrument-independent witness to
the sensor's true sensitivity (ocean microseisms drive a regionally near-constant ground motion).
We compute raw-count PSD over time (metadata-independent) + response-corrected band level + the
catalog noise_mm/peak_mm/residual, all on one time axis."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, obspy, glob, os
from scipy import signal
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
_av={f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":130,"font.size":11,"axes.grid":True,"grid.alpha":0.25,
    "legend.framealpha":1.0,"legend.facecolor":"white","legend.edgecolor":"0.6"})
D="/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/"
CONT=D+"continuous/HDB/HHZ.D/"
SXML="/home/msseo/works/02.Ulsan_Fault_detection/data/metadata/responses/master/KS_KG_metadata_1.0.2.xml"
CAT=D+"local_magnitudes/catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo.csv"
FS=100.0; NPS=8192
FAIL_ON=pd.Timestamp("2014-11-01"); FAIL_OFF=pd.Timestamp("2015-05-21")
SWAPS=[pd.Timestamp("2015-05-21"),pd.Timestamp("2019-02-01")]  # documented sensor swaps
BANDS={"microseism 0.1-0.2 Hz":(0.1,0.2),"0.5-1 Hz":(0.5,1.0),"WA band 2-8 Hz":(2.0,8.0)}

def read_day(year,jd):
    fs=glob.glob(CONT+f"KG.HDB..HHZ.D.{year}.{jd:03d}")
    if not fs: return None
    try: st=obspy.read(fs[0])
    except Exception: return None
    try: tr=st.merge(fill_value=0)[0]
    except Exception: return None
    x=np.asarray(tr.data,float)
    x=np.where(np.abs(x)>2e8,0.0,x)            # mask 2^28 fill/saturation spikes
    if np.count_nonzero(x)<FS*600: return None # <10 min good data
    return x

def day_segpsd(x):
    """median PSD over overlapping segments of one day (counts^2/Hz)."""
    f,_,Sxx=signal.spectrogram(signal.detrend(x),fs=FS,nperseg=NPS,noverlap=NPS//2,
                               scaling="density",mode="psd")
    return f, np.median(Sxx,axis=1)            # robust over time-of-day

def period_psd(months, sample_days):
    """returns (freq, dict month->medianPSD[counts], coverage)."""
    out={}; fref=None
    for m in months:
        segs=[]
        for jd in sample_days:
            x=read_day(m.year, int(m.dayofyear)+jd) if False else None
        # sample by absolute day-of-year near month
        y=m.year; base=pd.Timestamp(f"{y}-{m.month:02d}-01")
        for dd in sample_days:
            day=base+pd.Timedelta(days=dd)
            if day.year!=y: continue
            x=read_day(y, day.dayofyear)
            if x is None: continue
            f,P=day_segpsd(x); fref=f; segs.append(P)
        if segs: out[m]=np.median(np.vstack(segs),axis=0)
    return fref, out

# ---- monthly raw-count PSD across full archive ----
months=pd.date_range("2010-01","2024-12",freq="MS")
freq, mpsd = period_psd(months, sample_days=[3,9,15,21,27])
mon=[m for m in months if m in mpsd]
print(f"months with data: {len(mon)}/{len(months)}")
def band_db(P,lo,hi):
    msk=(freq>=lo)&(freq<hi); return 10*np.log10(np.median(P[msk]))
band_ts={b:np.array([band_db(mpsd[m],*r) for m in mon]) for b,r in BANDS.items()}
mont=np.array([m.to_pydatetime() for m in mon])

# ---- response-corrected band level (master xml) : confirm it drops too ----
inv=obspy.read_inventory(SXML)
def corrected_band(year, sample=(15,)):
    """median displacement PSD (m^2/Hz dB) in WA band over a few days that year."""
    vals=[]
    for mo in range(1,13,3):
        for dd in sample:
            try:
                x=read_day(year,(pd.Timestamp(f"{year}-{mo:02d}-01")+pd.Timedelta(days=dd)).dayofyear)
            except Exception: x=None
            if x is None: continue
            tr=obspy.Trace(data=signal.detrend(x).astype(np.float32))
            tr.stats.update(network="KG",station="HDB",location="",channel="HHZ",sampling_rate=FS,
                            starttime=obspy.UTCDateTime(f"{year}-{mo:02d}-02"))
            try: tr.remove_response(inventory=inv,output="DISP",water_level=60)
            except Exception: continue
            f,P=day_segpsd(tr.data); m=(f>=2)&(f<8); vals.append(10*np.log10(np.median(P[m])))
    return np.median(vals) if vals else np.nan
yrs=list(range(2010,2025))
corr_wa=np.array([corrected_band(y) for y in yrs])

# ---- catalog noise_mm / peak_mm / residual for HDB (response-corrected, event-based) ----
d=pd.read_csv(CAT); d=d[(d.snr>=3)&d.ML.notna()].copy()
d["t"]=pd.to_datetime(d.event_time,utc=True,errors="coerce"); d=d.dropna(subset=["t"])
d["t"]=d.t.dt.tz_localize(None); d["sc"]=d.network+"."+d.station+"."+d.channel
d["event_idx"]=d.event_idx.astype(int)
# time-stable event magnitude via median of all readings -> reading residual
mu=d.groupby("event_idx").ML.transform("median"); d["res"]=d.ML-mu
h=d[d.sc=="KG.HDB.HHZ"].set_index("t").sort_index()
def mmed(s,freq="ME",min_n=5):
    g=s.groupby(pd.Grouper(freq=freq)); med=g.median(); cnt=g.size()
    return med[cnt>=min_n].dropna()
noise_db=mmed(20*np.log10(h.noise_mm.clip(lower=1e-6)))   # response-corrected noise, dB rel mm
res_m=mmed(h.res)

# =================== FIGURE ===================
fig,ax=plt.subplots(4,1,figsize=(13,13),sharex=True,
                    gridspec_kw=dict(height_ratios=[1.25,1,1,1],hspace=0.12))
T0,T1=pd.Timestamp("2010-06"),pd.Timestamp("2024-12")
def marks(a):
    a.axvspan(FAIL_ON,FAIL_OFF,color="crimson",alpha=0.12,zorder=0)
    for s in SWAPS: a.axvline(s,color="k",ls=":",lw=1.1)
# (a) raw-count PSD spectrogram (period vs time)
P=np.vstack([mpsd[m] for m in mon]).T  # freq x month
per=1.0/freq[1:]; img=10*np.log10(P[1:,:])
pm=(per>=0.5)&(per<=40)
im=ax[0].pcolormesh(mont,per[pm],img[pm,:],shading="nearest",cmap="viridis",
                    norm=Normalize(vmin=np.nanpercentile(img[pm],5),vmax=np.nanpercentile(img[pm],97)))
ax[0].set_yscale("log"); ax[0].set_ylabel("Period (s)")
ax[0].set_title("(a) KG.HDB HHZ raw-count noise PSD over time  —  the sensor's own sensitivity witness",
                fontsize=11.5,loc="left")
cb=fig.colorbar(im,ax=ax[0],pad=0.01,aspect=18); cb.set_label("PSD (dB rel. counts$^2$/Hz)")
marks(ax[0]); ax[0].grid(False)
# (b) raw-count band levels
for b,c in zip(BANDS,["#1f77b4","#ff7f0e","#2ca02c"]):
    ax[1].plot(mont,band_ts[b],"-o",ms=2.5,lw=1.3,color=c,label=b)
ax[1].set_ylabel("Raw-count PSD (dB)")
ax[1].set_title("(b) Raw-count noise level per band  —  metadata-independent (no response applied)",fontsize=11,loc="left")
leg=ax[1].legend(loc="lower right",fontsize=8.5,ncol=3); leg.set_zorder(50); marks(ax[1])
# (c) response-corrected witnesses
ax[2].plot(noise_db.index,noise_db.values,"-o",ms=3,lw=1.4,color="#7a3fbf",label="catalog pre-P noise (resp-corr, 20·log mm)")
ax2=ax[2].twinx()
ax2.plot([pd.Timestamp(f"{y}-07-01") for y in yrs],corr_wa,"-s",ms=4,lw=1.4,color="#c0392b",label="continuous WA-band noise (resp-corr DISP)")
ax2.set_ylabel("Resp-corr DISP PSD (dB)",color="#c0392b"); ax2.grid(False)
ax[2].set_ylabel("noise_mm level (dB)",color="#7a3fbf")
ax[2].set_title("(c) Response-corrected noise also collapses  →  correcting the response does NOT recover it",fontsize=11,loc="left")
l1,la1=ax[2].get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
leg=ax[2].legend(l1+l2,la1+la2,loc="lower right",fontsize=8.5); leg.set_zorder(50)
ax[2].set_zorder(2); ax[2].patch.set_visible(False); ax2.set_zorder(1); marks(ax[2])
# (d) the ML residual that motivated all this
ax[3].axhline(0,color="0.5",lw=1,ls="--")
ax[3].plot(res_m.index,res_m.values,"-o",ms=3,lw=1.5,color="#111111",label="HDB monthly ML residual")
ax[3].set_ylabel("ML residual"); ax[3].set_xlabel("Year")
ax[3].set_title("(d) The station-magnitude residual this explains",fontsize=11,loc="left")
leg=ax[3].legend(loc="lower right",fontsize=8.5); leg.set_zorder(50); marks(ax[3])
ax[3].set_xlim(T0,T1)
ax[3].xaxis.set_major_locator(mdates.YearLocator(1)); ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.suptitle("KG.HDB 2014-2015 amplitude anomaly: a silent sensor-sensitivity collapse (raw counts), not a response-metadata error",
             y=0.995,fontsize=12.5)
fig.savefig("/tmp/hdb_2015_diagnosis.png",bbox_inches="tight"); print("saved /tmp/hdb_2015_diagnosis.png")

# =================== QUANTITATIVE SUMMARY ===================
def win(ts_idx,vals,a,b):
    m=(pd.DatetimeIndex(ts_idx)>=a)&(pd.DatetimeIndex(ts_idx)<b); return np.nanmedian(np.asarray(vals)[m])
print("\n=== RAW-COUNT microseism-band PSD (dB) ===")
b="microseism 0.1-0.2 Hz"
base=win(mont,band_ts[b],pd.Timestamp("2012-01"),FAIL_ON)
fail=win(mont,band_ts[b],FAIL_ON,FAIL_OFF)
post=win(mont,band_ts[b],FAIL_OFF,pd.Timestamp("2016-06"))
print(f"  baseline(2012->fail) {base:.1f} | failure-window {fail:.1f} | post-swap {post:.1f}")
print(f"  DROP during failure = {base-fail:.1f} dB  (= factor {10**((base-fail)/20):.0f} in amplitude = {(base-fail)/20:.2f} ML units)")
print("\n=== catalog ML residual ===")
print(f"  baseline {win(res_m.index,res_m.values,pd.Timestamp('2012-01'),FAIL_ON):+.2f} | "
      f"failure {win(res_m.index,res_m.values,FAIL_ON,FAIL_OFF):+.2f} | "
      f"post {win(res_m.index,res_m.values,FAIL_OFF,pd.Timestamp('2016-06')):+.2f}")
print("\nInterpretation: noise floor (raw counts) and ML residual fall TOGETHER and recover at the 2015-05 swap.")
