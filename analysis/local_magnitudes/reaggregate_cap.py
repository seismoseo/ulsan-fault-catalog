"""Apply the <=60 km distance cap to per-station ML and re-aggregate the event catalog.
The cap excludes far stations (path-attenuation bias that grows with distance and inflates the
network-median ML as the network expands). Equivalent to re-running with max_dist_km=60, but fast
(per-reading ML is unchanged; only the event aggregation excludes >60 km readings).
Keeps ml_all = UNCAPPED event median for location; magnitude = capped (<=60 km) median for statistics."""
import sys; sys.path.insert(0,'.')
import ml_pipeline as mp, pandas as pd, numpy as np
CAP=60.0
PS='catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo.csv'
EV='catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo.csv'
ps=pd.read_csv(PS)
ps.to_csv(PS.replace('.csv','_uncapped.csv'),index=False)               # backup full (for location/reference)
# uncapped event median (ml_all) BEFORE capping
ml_all=ps[ps.ML.notna()].groupby('event_idx').ML.median()
ncap=int((ps.ML.notna()&(ps.dist_km>CAP)).sum())
ps.loc[ps.dist_km>CAP,'ML']=np.nan                                       # apply cap
ps.to_csv(PS,index=False)
print(f"capped {ncap} used readings beyond {CAP:g} km; backup -> {PS.replace('.csv','_uncapped.csv')}")
# re-aggregate event catalog from capped per_station
agg={int(k):mp.aggregate_ml(g) for k,g in ps.groupby('event_idx')}
ev=pd.read_csv(EV)
for i,a in agg.items():
    ev.at[ev.index[i],'magnitude']=a['ml_median']; ev.at[ev.index[i],'magnitude_std']=a['ml_std']
    ev.at[ev.index[i],'n_used']=a['n_used']; ev.at[ev.index[i],'n_total']=a['n_total']
    ev.at[ev.index[i],'snr_median']=a['snr_median']
    ev.at[ev.index[i],'mag_status']=('ok' if a['n_used']>0 else 'low_snr')
ev['ml_all']=ev.index.map(lambda i: ml_all.get(i,np.nan))               # uncapped, for location
ev.to_csv(EV,index=False)
ok=ev[ev.magnitude.notna()]
print(f"re-aggregated: {len(ok)} events with capped magnitude; n_used>=3: {int((ev.n_used>=3).sum())}")
# UF-box sanity
UF=(129.25,129.55,35.60,35.90); box=ok[(ok.lon.between(*UF[:2]))&(ok.lat.between(*UF[2:]))]
print(f"UF box: {len(box)} with magnitude, n_used>=3: {int((box.n_used>=3).sum())}, median ML {box.magnitude.median():.2f}")
