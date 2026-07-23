"""Cross-component blast-candidate analysis for the KG.HDB waveform-similarity screen.

Cluster ids are NOT comparable across HHZ/HHN/HHE (each component is clustered independently),
but the EVENTS are. For each component we reproduce the notebook pipeline (warm make_bands + CC
caches), apply the exact `blast_like` criterion, collect the set of member events of every
blast_like family, and intersect those event sets across the three components.

The robust still-remaining-blast set = events flagged on ALL THREE components. Run with `Seis`.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from uflib import uf_waveform_similarity as wf

STATION   = "KG.HDB"
WIN       = (-0.5, 7.5)
BANDS     = [(1, 10), (2, 8), (4, 12), (5, 15)]
PRIMARY   = (1, 10)
MAXLAG    = 0.2
CC_THRESH = 0.6
LINKAGE   = "average"
MIN_SIZE  = 4
CACHE     = os.path.join(os.path.dirname(__file__), "wf_similarity_cache")
COMPS     = ["HHZ", "HHN", "HHE"]


def band_cc(comp, band, kept):
    tag = (f"{STATION}_{comp}_w{WIN[0]}_{WIN[1]}_b{band[0]}-{band[1]}_lag{MAXLAG}"
           f"_n{len(kept)}").replace(".", "p")
    f = os.path.join(CACHE, f"cc_{tag}.npy")
    return np.load(f) if os.path.exists(f) else None


def blast_for_component(comp):
    """Return (blast_events:set, evid_blast:DataFrame, meta, labels) for one component."""
    events = wf.list_events(station=STATION, comp=comp)
    res = wf.make_bands(events, station=STATION, comp=comp, bands=BANDS, win=WIN,
                        cache_dir=CACHE, verbose=False)
    kept, info = res["kept"], res["info"]
    meta = wf.load_event_meta(kept)

    cc = band_cc(comp, PRIMARY, kept)
    if cc is None:
        cc = wf.similarity_matrix(res["bands"][PRIMARY], maxlag=MAXLAG)
    labels, _, _ = wf.ward_clusters(cc, threshold=1 - CC_THRESH, method=LINKAGE)

    evid = wf.cluster_evidence(meta, labels, cc, min_size=MIN_SIZE)
    evid["blast_like"] = ((evid["mean_cc"] >= 0.6) & (evid["daytime_frac"] >= 0.6)
                          & (evid["rayleigh_p"] < 0.05) & (evid["spread_km"] <= 5))
    blast_ids = set(int(c) for c in evid.loc[evid["blast_like"], "cluster"])

    kept_arr = np.asarray(kept)
    labels = np.asarray(labels)
    blast_events = set(kept_arr[np.isin(labels, list(blast_ids))].tolist())
    print(f"[{comp}] kept {len(kept)} | {evid['blast_like'].sum()} blast_like families "
          f"({len(blast_events)} member events)")
    return blast_events, evid[evid["blast_like"]].copy(), meta, labels


def main():
    sets, metas = {}, {}
    for c in COMPS:
        sets[c], _, metas[c], _ = blast_for_component(c)

    z, n, e = sets["HHZ"], sets["HHN"], sets["HHE"]
    all3 = z & n & e
    any_ = z | n | e
    print("\n=== cross-component blast-candidate EVENTS ===")
    print(f"  HHZ {len(z)} | HHN {len(n)} | HHE {len(e)}")
    print(f"  HHZ∩HHN {len(z & n)} | HHZ∩HHE {len(z & e)} | HHN∩HHE {len(n & e)}")
    print(f"  all three (robust)     : {len(all3)}")
    print(f"  any component (union)  : {len(any_)}")
    print(f"  HHZ only {len(z - n - e)} | HHN only {len(n - z - e)} | HHE only {len(e - z - n)}")

    # per-event support count (1..3) with hypocentre/hour from the HHZ meta where available
    rows = []
    base = metas["HHZ"].set_index("event")
    for ev in sorted(any_):
        support = "".join(c[-1] for c in COMPS if ev in sets[c])  # e.g. 'ZNE'
        r = dict(event=ev, n_comp=sum(ev in sets[c] for c in COMPS), comps=support)
        if ev in base.index:
            row = base.loc[ev]
            r.update(lat=row.get("lat"), lon=row.get("lon"), depth=row.get("depth"),
                     hour_kst=row.get("hour_kst"))
        rows.append(r)
    df = pd.DataFrame(rows).sort_values(["n_comp", "event"], ascending=[False, True])
    out = os.path.join(os.path.dirname(__file__),
                       "cross_component_blast_candidates.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(df)} union events; {len(all3)} on all 3, "
          f"{(df['n_comp'] >= 2).sum()} on >=2)")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
