#!/usr/bin/env python
"""Generate 08.Network_decimation_ML_bias.ipynb — network-decimation synthetic test for temporal
ML bias from network densification (Heo 2024 -logA0). Takes recent, densely-recorded events as
'truth', decimates each to every past year's operating-station set, and measures the resulting
ML offset bias(year) — the bias the sparser historical network would have imposed. Then corrects
the catalog and shows the b-value/Mc impact. Pure arithmetic on the per-station ML table; does not
touch any running job."""
import nbformat as nbf

PS = "catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo.csv"
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md("""# Network-decimation test for temporal ML bias (Heo 2024)

**Question.** Did the sparse early network (2010-) impose an artefactual bias on ML relative to the
dense recent network, given an imperfect regional -logA0?

**Method (synthetic decimation).** Recent, densely-recorded events are the 'truth' (full-network
median ML). For each past year *Y* we keep only the stations operating in *Y* (and that recorded the
event), re-take the median, and measure `ΔML = ML_decimated − ML_full`. Averaged over many reference
events this gives **bias(Y)** — the offset the year-*Y* network would have produced on a known event.
We then subtract bias(Y) from the historical catalog and check the b-value / Mc impact.

*Caveat:* decimating recent events can only use stations present in both eras (the persistent subset),
so this is a tight **lower bound** on the bias (captures fewer-stations + geometry; not the site
response of since-decommissioned stations).""")

co("""import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib as mpl, matplotlib.font_manager as fm
_av = {f.name for f in fm.fontManager.ttflist}
for _f in ("Helvetica","Arial","Nimbus Sans","TeX Gyre Heros","DejaVu Sans"):
    if _f in _av: mpl.rcParams["font.family"]=_f; break
mpl.rcParams.update({"figure.dpi":120,"axes.grid":True,"grid.alpha":0.3})

PS_FILE="catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo.csv"
SNR_MIN=3.0; DM=0.1
REF_YEARS=(2022,2023,2024)   # dense, stable era = 'truth'
REF_NMIN=8                   # min full-network stations for a reference event
DEC_NMIN=3                   # min decimated stations to trust a decimated median
OP_MIN_FRAC=0.02             # a station 'operates' in year Y if it recorded >=2% of that year's events

d = pd.read_csv(PS_FILE)
d = d[(d.snr>=SNR_MIN) & d.ML.notna()].copy()
d['year'] = pd.to_datetime(d.event_time, utc=True, errors='coerce').dt.year
d = d.dropna(subset=['year']); d['year']=d.year.astype(int)
print(f"{len(d):,} per-station rows | {d.event_idx.nunique():,} events | years {d.year.min()}-{d.year.max()}")

# full-network event ML (median across station-channels) — self-consistent with the pipeline
ml_full = d.groupby('event_idx').ML.median()
n_full  = d.groupby('event_idx').station.nunique()
ev_year = d.groupby('event_idx').year.first()""")

md("""## 1 · Network densification — operating stations and stations-per-event by year""")
co("""years = sorted(d.year.unique())
# S(Y): stations operating in year Y (recorded >= OP_MIN_FRAC of that year's events)
SY = {}
for y in years:
    dy = d[d.year==y]; nev = dy.event_idx.nunique()
    vc = dy.groupby('station').event_idx.nunique()
    SY[y] = set(vc[vc >= max(3, OP_MIN_FRAC*nev)].index)
nsta = pd.Series({y:len(SY[y]) for y in years})
med_nstaev = d.groupby('year').apply(lambda g: g.groupby('event_idx').station.nunique().median())

fig,ax=plt.subplots(1,2,figsize=(12,4))
ax[0].bar(nsta.index,nsta.values,color='steelblue'); ax[0].set(xlabel='Year',ylabel='Operating stations',
    title='Network size by year'); ax[0].tick_params(axis='x',labelrotation=45)
ax[1].plot(med_nstaev.index,med_nstaev.values,'o-',color='tab:green')
ax[1].set(xlabel='Year',ylabel='Median stations per event',title='Stations contributing per event')
ax[1].tick_params(axis='x',labelrotation=45)
fig.tight_layout(); plt.show()
print('operating stations/yr:'); print(nsta.to_string())""")

md("""## 2 · The decimation test — bias(Y)

Reference events (dense era, well-recorded) are decimated to each past year's network; `ΔML =
decimated − full`. The curve `bias(Y)` is the systematic ML offset that year's network would impose.""")
co("""ref = d[d.year.isin(REF_YEARS)].copy()
ref_ev = n_full[(n_full>=REF_NMIN) & n_full.index.isin(ref.event_idx)].index
ref = ref[ref.event_idx.isin(ref_ev)]
print(f"{len(ref_ev):,} reference events (>= {REF_NMIN} stations, {REF_YEARS})")

rows=[]
for y in years:
    sub = ref[ref.station.isin(SY[y])]
    mld = sub.groupby('event_idx').ML.median()
    ndc = sub.groupby('event_idx').station.nunique()
    keep = ndc[ndc>=DEC_NMIN].index
    dml = (mld.loc[keep] - ml_full.loc[keep]).dropna()
    if len(dml):
        rows.append(dict(year=y, bias=dml.median(), q25=dml.quantile(.25), q75=dml.quantile(.75),
                         std=dml.std(), n=len(dml), n_sta=len(SY[y]),
                         med_ndec=int(ndc.loc[keep].median())))
B = pd.DataFrame(rows).set_index('year')

fig,ax=plt.subplots(figsize=(11,4.6))
ax.axhline(0,color='0.6',lw=0.8,ls='--')
ax.fill_between(B.index,B.q25,B.q75,color='tab:red',alpha=0.2,label='IQR')
ax.plot(B.index,B.bias,'o-',color='tab:red',lw=1.8,label='Median ΔML (decimated − full)')
ax.set(xlabel='Historical network year (Y)',ylabel='ML bias  ΔML',
       title='Network-decimation bias — recent events seen by each year\\'s network')
ax.set_xticks(B.index); ax.tick_params(axis='x',labelrotation=45); ax.legend()
fig.tight_layout(); plt.show()
print(B.round(3).to_string())""")

md("""## 3 · Mechanism — bias grows as the station count shrinks""")
co("""# per-event ΔML vs decimated station count, pooled over early years (the sparse regime)
pts=[]
for y in [yy for yy in years if yy<=2016]:
    sub=ref[ref.station.isin(SY[y])]
    mld=sub.groupby('event_idx').ML.median(); ndc=sub.groupby('event_idx').station.nunique()
    k=ndc[ndc>=1].index
    for e in k: pts.append((ndc.loc[e], mld.loc[e]-ml_full.loc[e]))
P=pd.DataFrame(pts,columns=['n_sta','dML'])
binned=P.groupby(pd.cut(P.n_sta,[0,1,2,3,5,8,12,100])).dML.agg(['median','std','count'])
fig,ax=plt.subplots(figsize=(8,4.5))
ax.scatter(P.n_sta+np.random.default_rng(0).normal(0,.08,len(P)),P.dML,s=5,alpha=.15,color='0.5')
ax.plot([i.mid for i in binned.index],binned['median'],'o-',color='tab:red',label='binned median')
ax.axhline(0,color='0.6',lw=0.8,ls='--')
ax.set(xlabel='Decimated station count',ylabel='ΔML',title='ML bias vs station count (sparse-era configs)',
       xlim=(0,20)); ax.legend()
fig.tight_layout(); plt.show()
print(binned.round(3).to_string())""")

md("""## 4 · Correction and impact — homogenised ML

Subtract bias(Y) from every event by its year, then compare annual median ML and the Gutenberg-Richter
b-value before vs after. If the densification bias is real, the corrected series is flatter in time.""")
co("""from seismostats.analysis import estimate_mc_maxc, ClassicBValueEstimator
from seismostats.utils import bin_to_precision
biasmap = B.bias.to_dict()
ev = pd.DataFrame({'ml':ml_full,'year':ev_year}).dropna()
ev['ml_corr'] = ev.ml - ev.year.map(biasmap).fillna(0.0)

def _b(mm):
    m=bin_to_precision(np.sort(mm.astype(float)),DM); mc,_=estimate_mc_maxc(m,fmd_bin=DM)
    be=ClassicBValueEstimator(); be.calculate(m[m>=mc],mc=mc,delta_m=DM); return mc,be.b_value,be.std

ann = ev.groupby('year').agg(ml_med=('ml','median'),mlc_med=('ml_corr','median'),n=('ml','size'))
fig,ax=plt.subplots(1,2,figsize=(13,4.4))
ax[0].plot(ann.index,ann.ml_med,'o-',color='0.5',label='original ML')
ax[0].plot(ann.index,ann.mlc_med,'s-',color='tab:blue',label='decimation-corrected ML')
ax[0].set(xlabel='Year',ylabel='Annual median ML',title='Annual median ML — before vs after correction')
ax[0].tick_params(axis='x',labelrotation=45); ax[0].legend()
for col,lab,c in [('ml','original','0.5'),('ml_corr','corrected','tab:blue')]:
    mm=ev[col].to_numpy(); mc,b,se=_b(mm)
    m=bin_to_precision(np.sort(mm),DM); e=np.arange(np.floor(m.min()/DM)*DM,np.ceil(m.max()/DM)*DM+DM,DM)
    cum=np.array([(m>=x).sum() for x in e])
    ax[1].semilogy(e,cum,'.',color=c,ms=4,label=f'{lab}: Mc={mc:.2f} b={b:.2f}±{se:.2f}')
ax[1].set(xlabel='ML',ylabel='N(≥ML)',title='Full-catalog FMD — before vs after'); ax[1].legend(fontsize=8)
fig.tight_layout(); plt.show()""")

md("""## 5 · Summary""")
co("""print('NETWORK-DECIMATION ML-BIAS TEST — summary\\n'+'='*52)
print(f'reference events: {len(ref_ev):,} ({REF_YEARS}, >= {REF_NMIN} stations)')
print(f'max |bias|: {B.bias.abs().max():.3f} ML at year {B.bias.abs().idxmax()}  '
      f'(network {B.loc[B.bias.abs().idxmax(),\"n_sta\"]:.0f} stations)')
print(f'bias in densest recent year ({years[-1]}): {B.bias.iloc[-1]:+.3f} ML (≈0 expected = self-consistency check)')
early=B[B.index<=2014].bias; late=B[B.index>=2021].bias
print(f'mean bias 2010-2014: {early.mean():+.3f} | 2021-2024: {late.mean():+.3f} ML')
print('\\nper-year bias table:'); print(B[['n_sta','med_ndec','n','bias','std']].round(3).to_string())
print('\\nTake-homes:')
print(' - bias(Y)~0 in the dense reference years confirms the method is self-consistent.')
print(' - a non-zero, time-trending bias(Y) in early years = the densification artefact, magnitude above.')
print(' - apply -bias(Y) by event year for a temporally homogeneous ML catalog before b/Mc/rate studies.')""")

nb.cells=C
out="08.Network_decimation_ML_bias.ipynb"; nbf.write(nb,out); print("wrote",out,len(C),"cells")
