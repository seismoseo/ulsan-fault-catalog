"""Build the rough de-blasted catalog notebook for KG.HDB (HHZ), derived from the waveform-similarity
blast screen (04_waveform_similarity_hdb_HHZ). Usage: python build_deblast_nb.py

Blast identification uses ONLY two measures — **waveform similarity** (the events cluster at the
station) and **daytime fraction** — because the blast events are **severely mislocated**, so no
location-based criterion (depth, epicentral spread) is trustworthy. A high daytime fraction
(predominantly working-hours origins) is the anthropogenic signature. The vetted natural cluster(s)
in NATURAL_OVERRIDE (cl 1158) are kept out of the blast set. The blastclean catalog is then split into
a blast catalog and a rough de-blasted catalog, both mapped over the UF subregion coloured by
hour-of-day (KST).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md("""# Rough de-blasted catalog at `KG.HDB` (HHZ) — from the waveform-similarity blast screen

Turns the **exploratory** blast screen (`04_waveform_similarity_hdb_HHZ_phasenet_plus`) into a
**rough final de-blasted catalog**. The screen clusters KG.HDB waveforms; a quarry pit repeats the
same source→path, so its events form a **tight waveform family** that fires in **working hours**.

**Classification uses only two measures — waveform similarity + daytime fraction.** The blast events
are **severely mislocated**, so depth and epicentral spread carry no information and are deliberately
**not** used. A family that is both waveform-similar *and* strongly daytime is a blast.

1. reproduce the HHZ 1-10 Hz clustering of the screen (cached CC),
2. flag `blast_like` = tight family (`mean_cc ≥ BLAST_MEAN_CC`) **and** strongly daytime
   (`daytime_frac ≥ BLAST_DAYFRAC`),
3. keep manually-vetted natural families out of the blast set (`NATURAL_OVERRIDE`),
4. split the blastclean catalog into a **blast catalog** and a **de-blasted catalog**, and
5. map both over the UF subregion, **coloured by hour-of-day (KST)**.

> **Why `daytime_frac ≥ 0.9` (not 0.6).** The daytime window is ~11 h, so a *random* (natural) family
> sits near 0.46 daytime; the old 0.6 cut admitted families that are ~40 % **night-time** — too many
> for a quarry — and it caught **cl 1158** (0.60 daytime, a genuine repeating natural cluster). The
> data show a clean gap: a handful of families are ≥95 % daytime, the rest ≤80 %. A 0.9 cut isolates
> the clean blasts and drops cl 1158 *automatically*. `NATURAL_OVERRIDE` keeps it out explicitly too.

> **Rough / preliminary.** Only KG.HDB-recorded events are screened; magnitude is not used.""")

md("## Parameters")
code("""STATION    = "KG.HDB"
COMP       = "HHZ"             # vertical — robust, rotation-immune basis
WIN        = (-0.5, 7.5)
BANDS      = [(1, 10), (2, 8), (4, 12), (5, 15)]
PRIMARY    = (1, 10)           # blast-screen band
MAXLAG     = 0.2
CC_THRESHOLD = 0.6             # average-linkage cut (defines the waveform families)
LINKAGE      = "average"
MIN_SIZE     = 4               # min cluster size to evaluate
# blast_like = waveform-similar AND strongly daytime — the ONLY two measures (no location: blasts
# are severely mislocated, so depth / epicentral spread are not trustworthy).
BLAST_MEAN_CC = 0.6            # family tightness (events already cluster at CC_THRESHOLD)
BLAST_DAYFRAC = 0.9            # >= 90% of origins in working hours (06-17 KST) — anthropogenic
# clusters flagged daytime+similar that are actually natural earthquakes (manually vetted) — kept in
# the de-blasted catalog. cl 1158 = deep repeating natural cluster (also excluded by the 0.9 cut).
NATURAL_OVERRIDE = [1158]
CACHE      = "wf_similarity_cache"
DEBLAST_CSV = "catalog_phasenet_plus_2010_2024_deblasted_rough.csv\"""")

code("""import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import uf_waveform_similarity as wf
import uf_cluster as ufc
wf.use_helvetica()
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)""")

md("""## 1 · Reproduce the families and flag blasts (waveform similarity + daytime only)""")
code("""events = wf.list_events(station=STATION, comp=COMP)
res  = wf.make_bands(events, station=STATION, comp=COMP, bands=BANDS, win=WIN, cache_dir=CACHE, verbose=False)
kept = res["kept"]; meta = wf.load_event_meta(kept)

def band_cc(band):
    tag = f"{STATION}_{COMP}_w{WIN[0]}_{WIN[1]}_b{band[0]}-{band[1]}_lag{MAXLAG}_n{len(kept)}".replace(".", "p")
    f = os.path.join(CACHE, f"cc_{tag}.npy")
    if os.path.exists(f):
        return np.load(f)
    cc = wf.similarity_matrix(res["bands"][band], maxlag=MAXLAG); np.save(f, cc); return cc

CC = band_cc(PRIMARY)
labels, Z, order = wf.ward_clusters(CC, threshold=1 - CC_THRESHOLD, method=LINKAGE)
evid = wf.cluster_evidence(meta, labels, CC, min_size=MIN_SIZE)
# blast_like: ONLY waveform similarity + daytime fraction (no rayleigh_p, no spread_km, no depth)
evid["blast_like"] = (evid["mean_cc"] >= BLAST_MEAN_CC) & (evid["daytime_frac"] >= BLAST_DAYFRAC)
blast_ids_raw = evid.loc[evid["blast_like"], "cluster"].tolist()
print(f"{len(blast_ids_raw)} blast_like families (mean_cc>={BLAST_MEAN_CC}, daytime>={BLAST_DAYFRAC}): "
      f"{sorted(blast_ids_raw)}  ({int(evid.loc[evid['blast_like'],'n'].sum())} events)")
# show the tight families sorted by daytime fraction so the daytime gap is visible
hi = evid[evid["mean_cc"] >= BLAST_MEAN_CC].sort_values("daytime_frac", ascending=False)
hi[["cluster","n","mean_cc","daytime_frac","depth_med","lat_c","lon_c","blast_like"]].head(25)""")

md("""## 2 · Reclassify vetted natural clusters → blast & de-blasted catalogs

`NATURAL_OVERRIDE` (cl 1158) is removed from the blast set; the de-blasted catalog is the blastclean
catalog minus the remaining blast-family events (matched on origin-time id).""")
code("""blast_ids = [c for c in blast_ids_raw if c not in NATURAL_OVERRIDE]
removed = [c for c in blast_ids_raw if c in NATURAL_OVERRIDE]
m = meta.copy(); m["fam"] = labels
blast_events = set(m.loc[m["fam"].isin(blast_ids), "event"])
print(f"blast families kept: {sorted(blast_ids)}  ({len(blast_events)} events)")
print(f"reclassified to natural: {removed}  ("
      f"{len(set(m.loc[m['fam'].isin(removed), 'event']))} events returned)")

cat = pd.read_csv(wf.BLASTCLEAN)
cat["time"] = pd.to_datetime(cat["time"], utc=True)
cat = ufc.add_kst_columns(cat, ufc.KST)
cat["event"] = cat["time"].dt.strftime("%Y%m%d%H%M%S")
cat["is_blast"] = cat["event"].isin(blast_events)
blast_cat   = cat[cat["is_blast"]].copy()
deblast_cat = cat[~cat["is_blast"]].copy()
print(f"\\nblastclean: {len(cat)} | blast: {len(blast_cat)} | de-blasted: {len(deblast_cat)}")
deblast_cat.drop(columns=["is_blast"]).to_csv(DEBLAST_CSV, index=False)
print(f"wrote rough de-blasted catalog -> {DEBLAST_CSV}")""")

md("""## 3 · Maps over the UF subregion — coloured by hour-of-day (KST)

Both maps share the subregion frame, fault traces, KG.HDB (yellow square) and known quarry centroids
(red ✗). **Colour = hour-of-day (KST)**, cyclic cpt. Origin *time* is reliable even though the
locations are not — so hour-of-day is the meaningful axis here. A blast family reads as one **daytime**
colour; the natural background spreads across all 24 h. (Depth / epicentral position are intentionally
not shown — the blast events are severely mislocated.)""")
code("""def _in_sub(d, s=ufc.SUBREGION):
    return d[d["lon"].between(s[0], s[1]) & d["lat"].between(s[2], s[3])]
print(f"subregion events: de-blasted {len(_in_sub(deblast_cat))} | blast {len(_in_sub(blast_cat))}")""")

md("### 3a · De-blasted (natural) catalog — hour-of-day (KST)")
code("""wf.map_catalog_subregion(deblast_cat, color_by="hour",
    title=f"De-blasted catalog ({len(_in_sub(deblast_cat))} in subregion) — hour of day (KST)").show()""")
md("### 3b · Blast catalog — hour-of-day (KST)")
code("""wf.map_catalog_subregion(blast_cat, color_by="hour", size="0.28c",
    title=f"Blast catalog ({len(_in_sub(blast_cat))} in subregion) — hour of day (KST)").show()""")

md("""## 4 · How to read this

- **Blast catalog** should read as a tight set of **daytime** (working-hours) colours. The map
  *positions* are unreliable (blasts are severely mislocated) — the **colour** (hour) is the signal.
- **De-blasted (natural) catalog** should show origins across **all 24 h** (no daytime bias),
  including the reclassified repeating cluster 1158.
- Classification is **location-free**: only waveform similarity (the family) + daytime fraction.
  To tighten/loosen, edit `BLAST_DAYFRAC` (0.9 = ≥90 % daytime); add any other vetted natural family
  to `NATURAL_OVERRIDE`.
- `DEBLAST_CSV` is written for downstream use but is **rough/preliminary** (only KG.HDB-recorded
  events were screened).""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/HypoInv/08_deblasted_catalog_KGHDB_HHZ_phasenet_plus.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(C), "cells")
