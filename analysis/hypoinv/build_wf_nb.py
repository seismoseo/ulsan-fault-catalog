"""Build the comprehensive waveform-similarity notebook (nbformat), ready to nbconvert.
Usage: python build_wf_nb.py [HHZ|HHN|HHE]   (default HHZ). Component drives PARAMS + filename."""
import sys
import nbformat as nbf

NB_COMP = sys.argv[1] if len(sys.argv) > 1 else "HHZ"
assert NB_COMP in ("HHZ", "HHN", "HHE"), NB_COMP

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md(f"""# Waveform-similarity screening for still-remaining quarry blasts — Ulsan Fault

**Common station:** `KG.HDB` (**{NB_COMP}**) · **working set:** `event_waveforms_ulsanfault` (≈2.8k events that
already survived the spatial/temporal blast decluster).

**Idea.** Quarry blasts from one pit repeat the same source→path, so at a fixed station they share
near-identical waveforms; tectonic events do not (genuine repeaters/aftershocks correlate too, but
separate out by hour-of-day and location in the evidence table). We:

1. load `KG.HDB.{NB_COMP}` per event, align on **P** — two deterministic sources only: a station
   **pick** (`{{ev}}_picks.csv`) else a synthetic **fallback** (`origin + median P travel-time`),
   the fallbacks refined by cross-correlation to the picked stack — picked events keep P at *t*=0;
2. bandpass + cut a **short P-aligned window** (never the raw 120 s) + L2-normalise — several bands;
3. build an N×N **max-lag cross-correlation** similarity matrix per band;
4. **Ward** hierarchical clustering on (1 − CC) → dendrogram + clustered heatmap;
5. per-cluster **waveform gathers** (visual similarity);
6. **spatial map** + a per-cluster **blast-likeness evidence table** (intra-cluster CC, spatial
   compactness, daytime fraction / hour-of-day reusing `uf_cluster`).

This is **exploratory** — it surfaces candidate blast families; it does not remove events. A tight
(high `mean_cc`), spatially compact, **daytime-concentrated** family = strong still-remaining-blast
candidate; a tight but **night/uniform** family = tectonic.
""")

md("## Parameters")
code(f"""# --- analysis parameters (edit + re-run top-to-bottom) ---------------------------------
STATION    = "KG.HDB"          # NET.STA common station
COMP       = "{NB_COMP}"             # component (HHZ vertical / HHN north / HHE east)
WIN        = (-0.5, 7.5)       # s relative to P — SHORT phase window (P→S→early coda), NOT 120 s
BANDS      = [(1, 10), (2, 8), (4, 12), (5, 15)]   # Hz
PRIMARY    = (1, 10)           # band used for dendrogram / gathers / evidence / map
MAXLAG     = 0.2              # s, CC lag search (alignment is refined, so small)
# Clustering criterion: group events whose waveforms correlate >= CC_THRESHOLD (data-driven
# family count; events with no similar sibling stay as singletons = the unclustered set).
CC_THRESHOLD = 0.6            # families merge below distance 1-CC_THRESHOLD
LINKAGE      = "average"      # interpretable with a CC threshold (mean 1-CC); 'ward' also available
YEARS      = None            # FULL catalog (all years). Set e.g. range(2016,2017) for one year.
MIN_SIZE   = 4                # min members for a cluster to appear in the evidence table
# Per-event gathers/spectrograms can't stack thousands of rows: at full-catalog scale show only
# the top families (the evidence table lists ALL of them) and drop the singleton block.
MAX_CLUSTERS_PLOT = 10        # families shown in the per-event gathers/spectrograms
MAX_PER_CLUSTER   = 25        # member cap per family in those gathers
SHOW_SINGLETONS   = False     # full catalog: don't stack the (many) singleton rows
CACHE      = "wf_similarity_cache"
""")

code("""import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from uflib import uf_waveform_similarity as wf
from uflib import uf_cluster as ufc
wf.use_helvetica()
os.makedirs(CACHE, exist_ok=True)
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
print("station", STATION, COMP, "| window", WIN, "s | bands", BANDS)""")

md("## 1 · Load, align, and build per-band features")
code("""events = wf.list_events(station=STATION, comp=COMP)
if YEARS is not None:
    yrs = {str(y) for y in YEARS}
    events = [e for e in events if e[:4] in yrs]
print(f"events with {STATION}.{COMP}: {len(events)}")

res  = wf.make_bands(events, station=STATION, comp=COMP, bands=BANDS, win=WIN, cache_dir=CACHE)
kept, info, shifts = res["kept"], res["info"], res["shifts"]
meta = wf.load_event_meta(kept)
print("alignment source:", info["p_source"].value_counts().to_dict())
print(f"fallback events refined by xcorr: {(shifts != 0).sum()} | "
      f"joined to blastclean catalog: {meta['joined'].sum()}/{len(meta)}")""")

md("""> **Note — per-component event coverage (why HHZ/HHN/HHE counts differ slightly).**
> At `KG.HDB`, `event_waveforms_ulsanfault` holds **HHZ 2770 / HHN 2772 / HHE 2771** events. This is
> **not** a processing artefact — it is **4 events with an incomplete set of component files on disk**
> (a per-channel dropout at those times):
>
> | event (UTC) | present | missing |
> |---|---|---|
> | `20100531170439` | HHE, HHN | HHZ |
> | `20131113192434` | HHN | HHZ, HHE |
> | `20140104222332` | HHE, HHN | HHZ |
> | `20140911190328` | HHZ | HHN, HHE |
>
> So **HHN = 2770 + 3 − 1 = 2772** and **HHE = 2770 + 2 − 1 = 2771**. `list_events(comp=…)` lists only
> events whose `{ev}.KG.HDB.{comp}.sac` exists; every event present for a component is processed
> identically. (~0.1 % of events — no material effect on the families.) The separately-reported
> `kept` count also drops a few dead / window-off-edge traces in `build_features`, which is likewise
> per-component.""")

md("### Alignment QC")
code("""fig, ax = plt.subplots(1, 3, figsize=(13, 3.2), dpi=120)
info["p_source"].value_counts().plot.bar(ax=ax[0], color="steelblue")
ax[0].set(title="P datum source", ylabel="events"); ax[0].tick_params(axis="x", rotation=0)
ax[1].hist(shifts[shifts != 0] / wf.SR, bins=30, color="indianred")
ax[1].set(title="Fallback xcorr shift", xlabel="shift (s)", ylabel="events")
# spot-check: overlay a few aligned PRIMARY-band traces (pick vs fallback)
Xp = res["bands"][PRIMARY]; t = np.arange(Xp.shape[1]) / wf.SR + WIN[0]
fb = np.where(info["p_source"].values == "fallback")[0][:3]
pk = np.where(info["p_source"].values == "pick")[0][:8]
for i in pk: ax[2].plot(t, Xp[i], color="0.6", lw=0.4)
for i in fb: ax[2].plot(t, Xp[i], color="crimson", lw=0.8)
ax[2].axvline(0, color="b", ls="--", lw=0.6); ax[2].set(title="Aligned traces (red=fallback)", xlabel="Time from P (s)")
ax[2].set_yticks([]); fig.tight_layout()""")

md("""## 2 · Inter-event similarity matrices (per band)

Max-lag normalised cross-correlation; reordered by the Ward dendrogram so repeating families show as
bright diagonal blocks, each **identified family (≥ `MIN_SIZE`) outlined in white**. Bands differ:
blasts are more emergent/low-frequency, quakes more impulsive/high-frequency.""")
code("""def band_cc(band):
    tag = f"{STATION}_{COMP}_w{WIN[0]}_{WIN[1]}_b{band[0]}-{band[1]}_lag{MAXLAG}_n{len(kept)}".replace(".", "p")
    f = os.path.join(CACHE, f"cc_{tag}.npy")
    if os.path.exists(f):
        return np.load(f)
    cc = wf.similarity_matrix(res["bands"][band], maxlag=MAXLAG)
    np.save(f, cc); return cc

CC = {b: band_cc(b) for b in BANDS}
# flat clusters by CORRELATION THRESHOLD: merge while average (1-CC) < 1-CC_THRESHOLD
CLUS = {b: wf.ward_clusters(CC[b], threshold=1 - CC_THRESHOLD, method=LINKAGE) for b in BANDS}
fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=110)
for ax, b in zip(axes.ravel(), BANDS):
    lab_b, _, order = CLUS[b]
    im = ax.imshow(CC[b][np.ix_(order, order)], cmap="magma", vmin=0, vmax=1,
                   aspect="equal", interpolation="nearest")
    wf.outline_clusters(ax, lab_b, order, min_size=MIN_SIZE)   # white box per identified family
    iu = np.triu_indices(len(CC[b]), 1)
    ax.set(title=f"{b[0]}–{b[1]} Hz  (mean CC {CC[b][iu].mean():.2f})",
           xlabel="event (clustered)", ylabel="event (clustered)")
fig.colorbar(im, ax=axes, label="Cross-correlation", shrink=0.6)
fig.suptitle(f"{STATION} {COMP} — waveform similarity, {len(kept)} events", y=0.93);""")

md("""## 3 · How clusters are identified (correlation-threshold criterion)

Clustering pipeline, per band:

1. **Feature** — each event → one P-aligned, band-passed, **L2-normalised** HHZ window, so only
   waveform *shape* matters (not amplitude/magnitude).
2. **Similarity** — `CC[i,j]` = max-lag normalised cross-correlation (cell 2); `1` = identical shape.
3. **Distance** — `D = 1 − CC`.
4. **Agglomerate** — `scipy.cluster.hierarchy.linkage(method=LINKAGE)` builds a tree (dendrogram);
   `average` linkage merges the two groups with the highest **mean pairwise CC**. A merge's
   *height* = `1 − (mean CC)`, so it is directly interpretable.
5. **Flat clusters — the criterion** — cut the tree at a **correlation threshold**: two groups stay
   together only while their mean CC ≥ `CC_THRESHOLD` (= cut distance `1 − CC_THRESHOLD`), via
   `fcluster(Z, 1 − CC_THRESHOLD, 'distance')`. So the **number of clusters is data-driven**, not a
   chosen count — raise `CC_THRESHOLD` → more, tighter families; lower it → fewer, looser ones.
6. **Singletons = unclustered** — an event whose waveform never reaches `CC_THRESHOLD` with any
   group stays a **size-1 cluster**. These are the "non-clustered" events. (Note: blasts *repeat*,
   so they form tight clusters; singletons are usually *unique* events, not blasts.)

`'ward'` linkage is also available (`LINKAGE='ward'`) but its merge heights aren't in CC units, so
the threshold isn't a CC value — hence `'average'` is the default here.""")
code("""labels, Z, order = CLUS[PRIMARY]
sizes = np.bincount(labels)[1:]
n_clu = int((sizes >= 2).sum()); n_sing = int((sizes == 1).sum())
print(f"CC_THRESHOLD={CC_THRESHOLD} ({LINKAGE} linkage) -> {n_clu} multi-event families "
      f"+ {n_sing} singletons (unclustered) over {len(labels)} events")
print(f"family sizes (top 15): {sorted(sizes[sizes >= 2], reverse=True)[:15]}")
wf.plot_dendrogram(Z, color_threshold=1 - CC_THRESHOLD,
                   title=f"{LINKAGE.title()} dendrogram (1 − CC), {PRIMARY[0]}–{PRIMARY[1]} Hz "
                         f"— cut at CC={CC_THRESHOLD}")
plt.axhline(1 - CC_THRESHOLD, color="red", ls="--", lw=0.8);""")

md("""## 4 · Cluster waveform gathers (primary band)

Traces (not stacked), P-aligned at *t*=0 (blue dashed), **S arrivals as short black bars**,
**coloured by family**. A blast family = near-identical coloured wiggles with a consistent S–P
moveout.

> **Full-catalog note:** to stay legible these gathers show only the **top `MAX_CLUSTERS_PLOT`
> families** (by size), capped at `MAX_PER_CLUSTER` members each, and **omit singletons**
> (`SHOW_SINGLETONS=False`). The **evidence table (§5) lists *all* families**; lower the caps /
> set `YEARS` to one year to inspect everything.""")
code("""evid = wf.cluster_evidence(meta, labels, CC[PRIMARY], min_size=MIN_SIZE)
# one colour per family, shared across the gathers + map (size order)
fams = sorted([int(c) for c in np.unique(labels) if (labels == c).sum() >= MIN_SIZE],
              key=lambda c: -(labels == c).sum())
COLORS = wf.cluster_colors(fams)
wf.plot_cluster_sections(res["bands"][PRIMARY], labels, kept, win=WIN, station=STATION,
                         min_show=MIN_SIZE, order_by="size", colors=COLORS,
                         max_clusters=MAX_CLUSTERS_PLOT, max_per_cluster=MAX_PER_CLUSTER,
                         show_singletons=SHOW_SINGLETONS);""")
md("""**Same gather, UNFILTERED.** Identical clusters/colours (identified on the *filtered* 1–10 Hz
waveforms above), but here the traces are the **raw** data — minimal preprocessing (demean + linear
detrend only, *no bandpass*), same P-alignment. Sanity check that a family's similarity is real in
the raw record, not an artefact of filtering.""")
code("""Xraw = wf.raw_matrix(res, station=STATION, comp=COMP, win=WIN)   # band=None: no bandpass
wf.plot_cluster_sections(Xraw, labels, kept, win=WIN, station=STATION, min_show=MIN_SIZE,
                         order_by="size", colors=COLORS, max_clusters=MAX_CLUSTERS_PLOT,
                         max_per_cluster=MAX_PER_CLUSTER, show_singletons=SHOW_SINGLETONS,
                         title=f"{STATION} {COMP} — UNFILTERED (raw, demean+detrend only), "
                               f"clusters from {PRIMARY[0]}–{PRIMARY[1]} Hz");""")
md("""**Same gather, 1 Hz highpass.** Again the same clusters/colours (from the 1–10 Hz clustering),
but traces shown with a single **1 Hz high-pass** (removes long-period drift, keeps the full high-
frequency content uncapped) — between the raw and the band-passed views.""")
code("""Xhp = wf.display_matrix(res, band=("highpass", 1.0), station=STATION, comp=COMP, win=WIN)
wf.plot_cluster_sections(Xhp, labels, kept, win=WIN, station=STATION, min_show=MIN_SIZE,
                         order_by="size", colors=COLORS, max_clusters=MAX_CLUSTERS_PLOT,
                         max_per_cluster=MAX_PER_CLUSTER, show_singletons=SHOW_SINGLETONS,
                         title=f"{STATION} {COMP} — 1 Hz HIGHPASS, "
                               f"clusters from {PRIMARY[0]}–{PRIMARY[1]} Hz");""")
md("""**Every family as its own full-size chronological gather — nothing omitted within families.**
The gathers above group by family but **cap** what they show (`max_clusters`/`max_per_cluster`/
`show_singletons` → at most a few hundred of the ~2.7k events). Here each **family** (≥ `MIN_SIZE`)
is drawn as its **own separate, full-width figure** (not a cramped subplot) with **all** its members
**chronologically** (oldest at top), 1 Hz-highpass, P-aligned at *t*=0 — **constant per-trace height**,
so a panel's height grows with its member count and the UTC origin times stay legible. A tight
same-shape stack that **recurs across many years** is the quarry-blast signature. (Singletons don't
repeat — they're the non-blast background — so they're skipped; for a literal single all-events stack
incl. singletons, `wf.plot_all_chronological(...)` is available, best on a one-year `kept`.)""")
code("""wf.plot_clusters_individually(Xhp, labels, kept, win=WIN, station=STATION, comp=COMP,
                              min_show=MIN_SIZE, colors=COLORS, order_by="size");""")
md("""**Same gather, coloured by hour-of-day (KST).** Identical clusters/ordering (left labels keep
the family colour), but each **trace is coloured by its origin hour** with the cyclic **HSV**
colormap (the same one as `uf.hour_map` / the per-year hour maps). **Blast tell-tale:** a family
whose traces are all one colour band fired at a single time-of-day (daytime) → anthropogenic; a
tectonic family shows mixed colours (24 h).""")
code("""hour = wf.event_hours(kept)   # KST hour from each event's origin time (dir name) — every event
wf.plot_cluster_sections(res["bands"][PRIMARY], labels, kept, win=WIN, station=STATION,
                         min_show=MIN_SIZE, order_by="size", colors=COLORS,
                         max_clusters=MAX_CLUSTERS_PLOT, max_per_cluster=MAX_PER_CLUSTER,
                         show_singletons=SHOW_SINGLETONS,
                         trace_values=hour, value_cmap="hsv", value_range=(0, 24),
                         value_label="Hour of day (KST)",
                         title=f"{STATION} {COMP} {PRIMARY[0]}–{PRIMARY[1]} Hz — coloured by "
                               f"hour-of-day (KST)");""")
md("""### Per-event spectrograms (time–frequency, 0.5–40 Hz), grouped by cluster

The time–frequency twin of the gather: **common x = time from P**, each event a **spectrogram strip**
(**y = 0.5–40 Hz**, colour = per-event relative power), stacked by family. (Computed on the *raw*
windowed signal; full-resolution in the notebook — zoom to inspect.) Compare spectral character
across families — blasts often differ from quakes (richer low-frequency / Rg, spectral scalloping),
so a family with a distinctive, *consistent* time–frequency signature is a strong candidate.""")
code("""wf.plot_cluster_spectrograms(Xraw, labels, kept, win=WIN, station=STATION, comp=COMP,
                             fmin=0.5, fmax=40, min_show=MIN_SIZE, order_by="size", colors=COLORS,
                             max_clusters=MAX_CLUSTERS_PLOT, max_per_cluster=MAX_PER_CLUSTER,
                             show_singletons=SHOW_SINGLETONS,
                             hours=wf.event_hours(kept),   # HSV hour tab + value at right (every event)
                             title=f"{STATION} {COMP} — per-event spectrogram 0.5–40 Hz, "
                                   f"clusters from {PRIMARY[0]}–{PRIMARY[1]} Hz");""")
md("""Compact per-cluster **stack** (coloured) over members (grey) — same families, summarised. The
panel **`cc`** is the family's **intra-cluster mean pairwise cross-correlation** (average of all
member-pair CC values from §2 — i.e. how tight the family is).""")
code("""wf.plot_cluster_gathers(res["bands"][PRIMARY], labels, evid, win=WIN, max_clusters=8,
                        colors=COLORS);""")

md("""## 5 · Blast-likeness evidence table

Per cluster (≥ `MIN_SIZE` members), sorted by intra-cluster `mean_cc`. **`blast_like`** highlights
families that are tight **and** daytime-concentrated **and** non-uniform in hour-of-day (the
anthropogenic signature). Tight-but-nightly families are tectonic repeaters.""")
code("""evid = evid.copy()
evid["blast_like"] = ((evid["mean_cc"] >= 0.6) & (evid["daytime_frac"] >= 0.6)
                      & (evid["rayleigh_p"] < 0.05) & (evid["spread_km"] <= 5))
n_cand = int(evid["blast_like"].sum())
print(f"{len(evid)} clusters ≥{MIN_SIZE}; {n_cand} flagged blast_like "
      f"({evid.loc[evid['blast_like'],'n'].sum()} events)")
evid.style.apply(lambda r: ["background-color:#ffe0e0" if r["blast_like"] else "" for _ in r], axis=1)\
    if hasattr(evid, "style") else evid""")

md("## 6 · Spatial map of waveform clusters")
code("""wf.map_clusters(meta, labels, evid, title=f"{STATION} waveform clusters ({PRIMARY[0]}–{PRIMARY[1]} Hz)",
                station=STATION, colors=COLORS)""")
md("""**Enlarged view — Ulsan-fault subregion.** The same cluster map zoomed to the
`uf_cluster.SUBREGION` box (+0.03°) so the families along the fault are legible.""")
code("""sub = [ufc.SUBREGION[0] - 0.03, ufc.SUBREGION[1] + 0.03,
       ufc.SUBREGION[2] - 0.03, ufc.SUBREGION[3] + 0.03]
wf.map_clusters(meta, labels, evid, station=STATION, colors=COLORS, reg=sub, top=20,
                title=f"{STATION} clusters — Ulsan subregion ({PRIMARY[0]}–{PRIMARY[1]} Hz)")""")
md("""**All families — same subregion.** Same zoom but with **every** evidence family coloured
(`top=None`), not just the 20 highest-`mean_cc` ones — the full clustered population in the
Ulsan-fault box (singletons/sub-`MIN_SIZE` groups remain the small grey dots).""")
code("""wf.map_clusters(meta, labels, evid, station=STATION, colors=COLORS, reg=sub, top=None,
                title=f"{STATION} all clusters — Ulsan subregion ({PRIMARY[0]}–{PRIMARY[1]} Hz)")""")

md("""## 7 · Blast-candidate close-ups

Focus on the **`blast_like`** families from §5 (tight CC + daytime + compact). The two views below
show **where** they are (coloured by hour-of-day) and **what** they look like (waveforms) — the
basis for confirming them before any removal.""")
code("""blast_ids = evid.loc[evid["blast_like"], "cluster"].tolist()
# dedicated DISTINCT palette for the (few) blast families, shared by the blast waveform gather and
# the hour histograms — the global COLORS can land small blast families in similar hues.
BLAST_COLORS = wf.cluster_colors(blast_ids)
print(f"{len(blast_ids)} blast-candidate families: {blast_ids} "
      f"({int(evid.loc[evid['blast_like'], 'n'].sum())} events)")""")
md("""**Blast events, coloured by hour-of-day (KST).** Subregion zoom: blast-candidate events are the
large hour-coloured circles (cyclic colormap, matching `uf.hour_map`); every other event is faint
grey context. A genuine quarry cluster reads as a compact pocket in a single **daytime** colour
band.""")
code("""wf.map_blast_hours(meta, labels, evid, station=STATION)""")
md("""**Blast-candidate waveforms** (1 Hz highpass, P-aligned at *t*=0, S bars), one colour block per
family — the near-identical repeating wiggles confirm a common (anthropogenic) source.""")
code("""wf.plot_cluster_sections(Xhp, labels, kept, win=WIN, station=STATION, clusters=blast_ids,
                         colors=BLAST_COLORS, show_singletons=False,
                         title=f"{STATION} {COMP} — blast candidates, 1 Hz highpass");""")
md("""**Hour-of-day histograms — one panel per blast-candidate family** (KST; daytime `06–17 h`
shaded). The temporal signature behind the `blast_like` flag: a genuine quarry family piles into a
single daytime bar (small `rayleigh_p`), whereas a tectonic repeater would spread across 24 h.""")
code("""wf.plot_blast_hour_histograms(meta, labels, evid, station=STATION, colors=BLAST_COLORS);""")

md("""## 8 · How to read this / next steps

- **Blast candidates** = rows flagged `blast_like` (tight `mean_cc`, `daytime_frac` high, `rayleigh_p`
  small, compact `spread_km`), ideally sitting near a known quarry centroid (red ✗ on the map) — see
  the §7 close-ups.
- **Tectonic families** = high `mean_cc` but night-ish / uniform hour-of-day, or spatially elongated.
- Compare across **bands** (cell 2): a real blast family stays coherent across bands.
- **Removal is a follow-up**: once the candidate families are confirmed here, drop their member events
  from the catalog and re-export — kept as a separate, reviewable step.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = f"/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv/04_waveform_similarity_hdb_{NB_COMP}_phasenet_plus.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(C), "cells")
