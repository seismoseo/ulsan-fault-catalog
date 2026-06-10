"""Build the MULTI-STATION repeater-confirmation notebook for KG.HDB families, one per component.
Usage: python build_multistation_nb.py [HHZ|HHN|HHE]   (default HHZ).

Extends the single-station (KG.HDB) repeating-cluster analysis to a NETWORK scope. We keep the HDB
clustering (5-15 Hz, CC>=0.9) as the candidate generator, then CONFIRM each family by intra-family
cross-correlation at the nearby stations that recorded its members, each on its NATIVE vertical
channel (HH/HG/EL). This is adaptive to the station network that grew 12->56 stations (2010-2024):
old families have few stations (flagged 'insufficient coverage', not rejected), recent ones many.

Motivation: the anti-repeater investigation showed an HDB-only waveform signal can be a single-station
artifact (a true co-located repeater reproduces at nearby stations; the candidate pairs collapsed).
"""
import sys
import nbformat as nbf

NB_COMP = sys.argv[1] if len(sys.argv) > 1 else "HHZ"
assert NB_COMP in ("HHZ", "HHN", "HHE"), NB_COMP
HYPO = "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/HypoInv"

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md(f"""# Multi-station confirmation of `KG.HDB` repeater families ({NB_COMP}, 5-15 Hz, single-linkage)

The repeating-earthquake families are detected at the single common station **KG.HDB** (5-15 Hz,
CC >= 0.9). But a strong HDB waveform similarity can be a **single-station** effect — the
anti-repeater study showed candidate pairs that looked compelling at HDB **decorrelated** at the
nearby stations, whereas a genuine co-located repeater reproduced everywhere (`cc_pos` 0.88-0.99).

So we **confirm** each HDB family across the network: at every nearby station that recorded the
family's members (each on its **native** vertical channel HH/HG/EL), measure the **intra-family mean
CC**; a family is **network-confirmed** if >= `MIN_CONF` stations reproduce it at mean CC >= `CONF_CC`.

**Adaptive to the time-varying network** (12->56 stations over 2010-2024): confirmation uses whatever
stations recorded each family's members. A family from a sparse era (too few stations) is labelled
**`insufficient` coverage — NOT rejected**, distinct from one that had stations but failed (a likely
HDB-only artifact).""")

md("## Parameters")
code(f"""STATION    = "KG.HDB"
COMP       = "{NB_COMP}"          # HDB clustering channel (HHZ vertical is the robust basis)
WIN        = (-0.5, 7.5)
BANDS      = [(5, 15), (1, 10), (2, 8), (4, 12)]
PRIMARY    = (5, 15)           # clustering + confirmation band (user-set)
MAXLAG     = 0.2
CC_REPEAT  = 0.90              # HDB family threshold
LINKAGE    = "single"          # single-linkage: events join a family if CC >= CC_REPEAT for >=1 pair
MIN_FAMILY = 3                 # min members to call a family (need >=3 for a meaningful network CC)
# --- network confirmation ---
STATION_K  = 8                 # up to this many closest stations per family
MAX_KM     = 40.0              # only stations within this radius of the family centroid
CONF_CC    = 0.6               # a station 'confirms' if its intra-family mean CC >= this
MIN_MEMBERS = 3                # a station must record >= this many members to count
MIN_CONF   = 2                 # >= this many confirming stations => network-confirmed
CACHE      = "wf_similarity_cache\"""")

code(f"""import os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, "{HYPO}")   # run from repeaters/ or anywhere
import uf_waveform_similarity as wf
import uf_cluster as ufc
wf.use_helvetica()
CACHE = wf.CACHE_DIR           # absolute cache dir (cwd-independent)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)""")

md("""## 1 · KG.HDB repeater families (5-15 Hz, CC >= 0.9, **single-linkage**)

**Linkage = single (friends-of-friends):** two events are joined into the same family if their CC is
>= `CC_REPEAT` (0.9) for **at least one pair** — clusters grow by chaining any near-duplicate pair,
rather than requiring the whole group to be mutually similar (that is average/UPGMA linkage). This is
the looser, pair-based definition of a repeating sequence; `wf.ward_clusters(CC, 1-0.9, "single")`
cuts the single-linkage tree at distance 1-CC.""")
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
labels, Z, order = wf.ward_clusters(CC, threshold=1 - CC_REPEAT, method=LINKAGE)
rep = wf.repeater_table(meta, labels, CC, min_size=MIN_FAMILY)
print(f"{len(rep)} HDB families (>= {MIN_FAMILY} members) at {PRIMARY[0]}-{PRIMARY[1]} Hz, CC>={CC_REPEAT}; "
      f"{int(rep['n'].sum())} events; largest n={int(rep['n'].max()) if len(rep) else 0}")
rep.head(20)""")

md("""## 2 · Nearby stations are era-dependent

The station network grew over 2010-2024, so the stations available to confirm a family depend on
**when** it occurred. Compare an early family vs a recent one (closest stations, native channel).""")
code("""def show_nearby(fam):
    g = meta.assign(fam=labels); gg = g[(g.fam == fam) & g.joined]
    ev = list(meta.assign(fam=labels).query("fam == @fam")["event"])
    c = (gg.lat.mean(), gg.lon.mean())
    ns = wf.nearby_stations(ev, c, max_km=MAX_KM)
    print(f"family {fam}: {len(ev)} members, years {sorted({e[:4] for e in ev})}, "
          f"{len(ns)} stations within {MAX_KM:.0f} km")
    return ns
# earliest and latest families by first-event year
order_by_year = rep.assign(yr=rep["t_first"].str[:4]).sort_values("yr")
early = int(order_by_year.iloc[0]["cluster"]); late = int(order_by_year.iloc[-1]["cluster"])
display(show_nearby(early).head(10)); display(show_nearby(late).head(10))""")

md("""## 3 · Network confirmation table

Per family: `n_sta_avail` (nearby stations recording >= MIN_MEMBERS members), `n_sta_conf` (of those,
how many reproduce the family at mean CC >= CONF_CC), `net_mean_cc` (median intra-family mean CC over
those stations), `confirmed`, and `coverage` (`ok` vs `insufficient`). Sorted: confirmed first.""")
code("""net = wf.network_confirm(meta, labels, rep, band=PRIMARY, maxlag=MAXLAG, win=WIN,
                         station_K=STATION_K, max_km=MAX_KM, conf_cc=CONF_CC,
                         min_members=MIN_MEMBERS, min_conf=MIN_CONF)
cols = ["cluster","n","mean_cc","depth_med","spread_km","t_first","t_last",
        "n_sta_avail","n_sta_conf","net_mean_cc","confirmed","coverage"]
net = net.sort_values(["confirmed","net_mean_cc"], ascending=[False, False])
n_conf = int(net["confirmed"].sum()); n_ins = int((net["coverage"] == "insufficient").sum())
print(f"{n_conf} network-confirmed | {len(net)-n_conf-n_ins} had coverage but unconfirmed | "
      f"{n_ins} insufficient coverage (sparse era)")
net[cols]""")

md("""## 4 · Per-family network gathers — the visual proof

For the top confirmed families: member traces (grey) + the family stack (red) at each nearby station.
A confirmed family shows the same repeating wiggle across stations; an HDB-only one wouldn't.""")
code("""TOPF = net[net["confirmed"]].head(4)["cluster"].tolist()
for fam in TOPF:
    fig = wf.plot_family_network(meta, labels, int(fam), band=PRIMARY, win=WIN, station_K=6,
                                 max_km=MAX_KM, min_members=MIN_MEMBERS)
    if fig is not None:
        plt.show()""")
md("""**Counter-example** — a family with coverage that did **not** confirm (network CC low off HDB),
i.e. a likely single-station artifact (if any exists).""")
code("""bad = net[(net["coverage"] == "ok") & (~net["confirmed"])].sort_values("net_mean_cc")
if len(bad):
    fig = wf.plot_family_network(meta, labels, int(bad.iloc[0]["cluster"]), band=PRIMARY, win=WIN,
                                 station_K=6, max_km=MAX_KM, min_members=MIN_MEMBERS)
    if fig is not None: plt.show()
else:
    print("(no covered-but-unconfirmed family — all families with coverage confirmed)")""")

md("""## 5 · Waveform similarity across events at EACH station — 10 largest clusters (no stack)

For the **10 largest** families: the member event waveforms as **offset wiggles** (earliest -> latest)
at every nearby station, band-filtered to PRIMARY (5-15 Hz). **No stack** — the raw cross-event
similarity at each station is shown directly. A genuine repeating family looks near-identical row-to-row
at the close stations; chained/incoherent members show up as rows that don't match.""")
code("""TOP10 = rep.head(10)["cluster"].tolist()
for fam in TOP10:
    fig = wf.plot_family_station_gathers(meta, labels, int(fam), band=PRIMARY, win=WIN,
            station_K=6, max_km=MAX_KM, min_members=MIN_MEMBERS, max_traces=40)
    if fig is not None:
        plt.show()""")

md("""## 6 · Recurrence timeline — largest families, 2016 Gyeongju mainshock marked

The same temporal view used for the KG.HDB-only analysis (`plot_repeater_sequences`): one full-width
row per family (largest first), a marker at every member origin time, coloured by family. The **2016
Gyeongju M5.8 mainshock** is the red dashed line.""")
code("""fig = wf.plot_repeater_sequences(meta, labels, rep, top=30, mark_gyeongju=True,
        title=f"KG.HDB {COMP} repeater families ({PRIMARY[0]}-{PRIMARY[1]} Hz, single-linkage CC>={CC_REPEAT}) "
              f"- recurrence timeline (largest 30)")
plt.show()""")

md("""## 7 · Map — confirmed vs unconfirmed vs insufficient-coverage families (PyGMT, UF subregion)""")
code("""try:
    conf_ids = net.loc[net["confirmed"], "cluster"].tolist()
    fig = wf.map_cluster_links(meta, labels, net[net["confirmed"]], link="centroid",
            title=f"KG.HDB {COMP} network-CONFIRMED repeater families ({PRIMARY[0]}-{PRIMARY[1]} Hz)")
    fig.show()
except Exception as e:
    print("PyGMT map skipped:", type(e).__name__, e)""")

md(f"""## 8 · How to read this

- **`confirmed`** = the family repeats at >= {{MIN_CONF}} nearby stations (intra-family mean CC >=
  {{CONF_CC}}) on their native channel — a genuine, network-verified repeating-earthquake family.
- **covered but unconfirmed** = stations were available but the waveform similarity did **not** hold
  off KG.HDB — a candidate **single-station artifact** (the lesson from the anti-repeater study).
- **`insufficient` coverage** = the family's era had < {{MIN_CONF}} usable nearby stations; it is
  **not rejected**, only unverifiable network-wide (these are mostly early-period families).
- **Channel fairness**: every station is used on its **native** vertical channel (HH/HG/EL), selected
  by distance — never assuming HHZ (which would silently drop the closest newer stations).
- **Component**: this is `{NB_COMP}`. HHZ is the robust basis; HHN/HHE are a parameter swap.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = f"{HYPO}/repeaters/10_multistation_repeaters_KGHDB_{NB_COMP}_phasenet_plus.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(C), "cells")
