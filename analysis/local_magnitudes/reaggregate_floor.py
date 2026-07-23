"""Apply the dead-trace floor to the per-station catalog and re-aggregate the event catalog.
Keeps ALL events (magnitude for location); the n_used>=3 stats filter is applied in the
analysis notebooks, NOT here."""
import sys; sys.path.insert(0,'.')
import ml_pipeline as mp, pandas as pd, numpy as np
PS='catalog_phasenet_plus_2010_2024_blastclean_per_station_ml_heo.csv'
EV='catalog_phasenet_plus_2010_2024_blastclean_with_ml_heo.csv'
ps=pd.read_csv(PS); n0=len(ps)
dead=ps.peak_mm < mp.DEAD_TRACE_FLOOR_MM
print(f"dead-trace readings (peak<{mp.DEAD_TRACE_FLOOR_MM:g}): {int(dead.sum())} of {n0}")
ps=ps[~dead].copy(); ps.to_csv(PS,index=False)             # floored per-station
ev=pd.read_csv(EV); ev_before=ev[['time','magnitude','n_used']].copy()
# re-aggregate ONLY events present in per_station (others = no readings, keep as-is)
agg={int(k):mp.aggregate_ml(g) for k,g in ps.groupby('event_idx')}
changed=0
for i in list(agg.keys()):
    a=agg[i]
    row=ev.iloc[i]
    newmag = a['ml_median']; newn=a['n_used']
    if not (np.isclose(row.magnitude, newmag, equal_nan=True) and row.n_used==newn):
        ev.at[ev.index[i],'magnitude']=a['ml_median']; ev.at[ev.index[i],'magnitude_std']=a['ml_std']
        ev.at[ev.index[i],'n_used']=a['n_used']; ev.at[ev.index[i],'n_total']=a['n_total']
        ev.at[ev.index[i],'snr_median']=a['snr_median']
        ev.at[ev.index[i],'mag_status']=('ok' if a['n_used']>0 else 'low_snr')
        changed+=1
ev.to_csv(EV,index=False)
print(f"events updated by floor: {changed}")
# show what changed
m=ev[['time','magnitude','n_used']].merge(ev_before,on='time',suffixes=('_new','_old'))
d=m[(m.magnitude_new!=m.magnitude_old)|(m.n_used_new!=m.n_used_old)]
print(f"events with changed magnitude or n_used: {len(d)}")
print("biggest magnitude changes:")
d=d.assign(dmag=(d.magnitude_new-d.magnitude_old).abs()).sort_values('dmag',ascending=False)
print(d.head(6)[['time','magnitude_old','magnitude_new','n_used_old','n_used_new']].to_string(index=False))
print(f"events now with magnitude (all): {ev.magnitude.notna().sum()}  n_used>=3: {int((ev.n_used>=3).sum())}")
