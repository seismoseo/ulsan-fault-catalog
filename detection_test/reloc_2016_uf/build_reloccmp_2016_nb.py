#!/usr/bin/env python
"""Generate 09.UF2016_reloccmp.ipynb — nb21-style 3-way relocation comparison for the 2016 Ulsan-Fault
GJ-array run: HypoInverse (absolute) vs dt.ct (catalog) vs dt.cc (CC>=0.7), all kim2011 / ISTART=2 / adaptive
LSQR damping on the uf_cluster-QC'd 596-event subset. Differences from nb21: EQUAL-SIZE markers (no local
magnitudes), and ADDED hour-of-day maps + histograms (quarry-blast / diurnal structure). Mirrors the SOTA
PyGMT style of HypoInv/build_reloccmp_nb.py (region, fault trace, turbo depth, 10 km scale)."""
import nbformat as nbf
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# Ulsan-Fault 2016 relocation — HypoInverse vs dt.ct vs dt.cc (GJ-array, kim2011)

Whole-box HypoDD relocation of the 2016 Ulsan-Fault-subregion events recorded with the **KS + KG + GJ
temporary array**, comparing three location sets:

| set | data | meaning |
|---|---|---|
| **HypoInverse** | — | absolute single-event locations (HypoDD starting positions) |
| **dt.ct** | catalog | catalog-pick differential times only |
| **dt.cc** | cross-correlation + catalog | waveform CC≥0.7 differential times (the high-resolution result) |

All three legs use the **kim2011** velocity, **ISTART=2** (start from catalog locations), **ISOLV=2** (LSQR)
with **per-set adaptive damping** (condition number driven into 40–80). Event set = the 596 events passing the
`uf_cluster` QC (`erh<5, erz<5, gap<270, num>5, rms<1.0`) of the HypoInverse `.sum`. Markers are **equal size**
(no local magnitudes). Times are **KST** (the PocketQuake catalog convention).""")

co(r"""import os, numpy as np, pandas as pd, pygmt, matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# --- Helvetica house style (graceful fallback to a metric-compatible clone) ---
_avail = {f.name for f in fm.fontManager.ttflist}
for _f in ["Helvetica", "Arial", "Nimbus Sans", "Liberation Sans", "FreeSans"]:
    if _f in _avail:
        mpl.rcParams["font.family"] = _f; break
mpl.rcParams.update({"font.size": 10, "figure.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.3, "legend.framealpha": 1.0, "legend.facecolor": "white"})

RUNROOT = "/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/pipeline/runs/uf_2016_qc"
RUN = f"{RUNROOT}/2.HypoDD"
FAULT = "/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv/faults_lonlat.gmt"
COAST = "/home/msseo/works/02.Ulsan_Fault_detection/analysis/reloc_analysis/coastline_lonlat.gmt"
REGION = [129.25, 129.55, 35.60, 35.90]
MARK = "c0.15c"                                   # EQUAL-size marker (no magnitude scaling)
STA = pd.read_csv(f"{RUNROOT}/station_table/used_stations_100km.csv")   # used stations (Network,Code,Lat,Lon)

def basemap_context(fig, title, proj="M13c"):
    "Common map furniture: frame + coastline + fault trace + used-station triangles + scale bar."
    fig.basemap(region=REGION, projection=proj, frame=[f"WSne+t{title}", "xa0.1f0.05", "ya0.1f0.05"])
    if os.path.exists(COAST): fig.plot(data=COAST, pen="0.6p,60")          # coastline
    if os.path.exists(FAULT): fig.plot(data=FAULT, pen="0.9p,black")       # fault traces
    fig.plot(x=STA.Longitude, y=STA.Latitude, style="i0.34c", fill="dodgerblue3", pen="0.4p,black")  # stations
    fig.basemap(map_scale="jBL+w10k+o0.3c/0.3c+c35.75")

def rd(path):
    "Read a hypoDD .loc/.reloc into a DataFrame keyed by cuspid (id). Columns positional (HypoDD 1.x)."
    cols = ["id","lat","lon","depth","x","y","z","ex","ey","ez","yr","mo","dy","hr","mi","sc",
            "mag","nccp","nccs","nctp","ncts","rcc","rct","cid"]
    rows = [ln.split() for ln in open(path) if ln.split()]
    d = pd.DataFrame([r[:24] for r in rows], columns=cols[:len(rows[0])]).apply(pd.to_numeric, errors="coerce")
    return d.set_index("id")

loc = rd(f"{RUN}/02.dt.cc/hypoDD.loc")            # HypoInverse absolute (starting positions)
ct  = rd(f"{RUN}/01b.dtct_qc/hypoDD.reloc")       # dt.ct catalog-only
cc  = rd(f"{RUN}/02.dt.cc/hypoDD.reloc")          # dt.cc cross-correlation (primary product)
for d in (loc, ct, cc):                            # hypoDD time columns are UTC -> KST = +9h (PocketQuake convention)
    t_utc = pd.to_datetime(dict(year=d.yr, month=d.mo, day=d.dy, hour=d.hr, minute=d.mi,
                                second=d.sc.clip(0,59)), errors="coerce", utc=True)
    t_kst = t_utc + pd.Timedelta(hours=9)
    d["kst_hour"] = t_kst.dt.hour + t_kst.dt.minute/60.0
    d["weekday"] = t_kst.dt.dayofweek
ccres = cc[(cc.nccp + cc.nccs) > 0]               # events with >=1 surviving cc link (the repeaters)
SETS = [("HypoInverse", loc), ("dt.ct", ct), ("dt.cc", cc)]
print(f"HypoInverse {len(loc)}  |  dt.ct {len(ct)}  |  dt.cc {len(cc)}  (cc-resolved {len(ccres)})")
print(f"faults: {os.path.exists(FAULT)}   region {REGION}")""")

md(r"""## 1 · Three-way location maps (depth-coloured, equal markers)

The same events located three ways. Absolute HypoInverse locations scatter; dt.ct sharpens them onto the fault
framework; dt.cc collapses the well-correlated pairs onto near-linear structures. Colour = depth (km); every
marker is the same size.""")

co(r"""fig = pygmt.Figure()
with pygmt.config(MAP_FRAME_TYPE="plain", FORMAT_GEO_MAP="ddd.xx"):
    pygmt.makecpt(cmap="turbo", series=[2, 20], reverse=True)
    for i, (name, d) in enumerate(SETS):                     # horizontal row of 3 panels
        basemap_context(fig, f"{name} (n={len(d)})", proj="M9c")   # coast + fault + stations + scale
        fig.plot(x=d.lon, y=d.lat, style=MARK, fill=d.depth, cmap=True, pen="0.2p,gray30")
        fig.shift_origin(xshift="10c")
    fig.shift_origin(xshift="-30c")
    fig.colorbar(frame=["x+lDepth", "y+lkm"], position="JBC+o0c/1.3c+w14c/0.4c+h")
fig.show(width=980)""")

md(r"""## 1b · The dt.cc-resolved catalog (highest precision)

The subset relocated with **surviving waveform cross-correlation links** (`nccp+nccs > 0`) — the tightly
correlated, near-repeating events whose relative positions are constrained to the sub-metre level. These are
the events "only relocated by dt.cc": the dt.ct-only events (no cc link) are dropped here. Same depth colour,
equal markers, coastline + fault + used-station triangles.""")

co(r"""fig = pygmt.Figure()
with pygmt.config(MAP_FRAME_TYPE="plain", FORMAT_GEO_MAP="ddd.xx"):
    basemap_context(fig, f"dt.cc-resolved events (n={len(ccres)}, >=1 cc link)")
    pygmt.makecpt(cmap="turbo", series=[2, 20], reverse=True)
    fig.plot(x=ccres.lon, y=ccres.lat, style=MARK, fill=ccres.depth, cmap=True, pen="0.3p,gray20")
    fig.colorbar(frame=["x+lDepth", "y+lkm"], position="JMR+o0.5c/0c+w9c")
fig.show(width=820)
print(f"cc-resolved: {len(ccres)} of {len(cc)} dt.cc events ({100*len(ccres)/len(cc):.0f}%)")
print(f"  cc links/event: P median {ccres.nccp.median():.0f}, S median {ccres.nccs.median():.0f}")
print(f"  depth {ccres.depth.min():.1f}-{ccres.depth.max():.1f} km (median {ccres.depth.median():.1f}); "
      f"rel err EX {ccres.ex.median():.0f} EY {ccres.ey.median():.0f} EZ {ccres.ez.median():.0f} m")""")

md(r"""## 2 · Displacement HypoInverse → dt.ct → dt.cc

Median horizontal / vertical shift each relocation step applies, measured on the events common to the relevant
pair (cuspids are shared across the three legs). Large HypoInverse→dt.cc shifts = events whose absolute location
was poorly constrained; small shifts = already-tight events.""")

co(r"""def disp(A, B):
    j = A.index.intersection(B.index)
    dx = (B.loc[j,"lon"]-A.loc[j,"lon"])*np.cos(np.radians(35.75))*111.19
    dy = (B.loc[j,"lat"]-A.loc[j,"lat"])*111.19
    h = np.hypot(dx, dy)*1000.0; v = (B.loc[j,"depth"]-A.loc[j,"depth"]).abs()*1000.0
    return len(j), np.median(h), np.median(v)
for a,b in [(loc,ct),(loc,cc),(ct,cc)]:
    an = {id(loc):"HypoInverse",id(ct):"dt.ct",id(cc):"dt.cc"}
    n,h,v = disp(a,b); print(f"{an[id(a)]:12s} -> {an[id(b)]:6s}: n={n:4d}  median |H| {h:5.0f} m   |V| {v:5.0f} m")
print(f"\ncc-resolved (>=1 surviving cc link): {len(ccres)} of {len(cc)} dt.cc events ({100*len(ccres)/len(cc):.0f}%)")
print(f"dt.cc relative errors (m): EX {cc.ex.median():.0f}  EY {cc.ey.median():.0f}  EZ {cc.ez.median():.0f} (median)")""")

md(r"""## 3 · Relocated vs unrelocated — why ~half the QC'd events drop out

HypoDD relocates only events **connected in the differential-time graph**. Absolute location quality (the QC)
is necessary but not sufficient: a well-located but spatially **isolated** event has too few neighbour links to
relocate relatively. Below, the dt.cc-relocated events (depth-coloured) against the QC'd-but-unrelocated events
(grey) — the unrelocated set is diffuse and peripheral, not a distinct (e.g. quarry) population.""")

co(r"""import json, glob
qc = set(json.load(open("/home/msseo/works/02.Ulsan_Fault_detection/detection_test/reloc_2016_uf/qc_cuspids.json")))
# unrelocated = QC'd events NOT in the dt.cc reloc; positions = HypoInverse (event.dat) for those in the graph
ed = {int(l.split()[-1]): l.split() for l in open(f"{RUN}/02.dt.cc/event.dat") if l.split()}
unrel_ids = [c for c in loc.index if c not in cc.index]
ula = np.array([loc.loc[c,"lat"] for c in unrel_ids]); ulo = np.array([loc.loc[c,"lon"] for c in unrel_ids])
# link counts per event (dt.ct + dt.cc)
def links(fn):
    from collections import Counter; c=Counter()
    for l in open(fn):
        if l.startswith("#"):
            t=l.split(); c[int(t[1])]+=1; c[int(t[2])]+=1
    return c
lct = links(f"{RUN}/02.dt.cc/dt.ct")
rel_ct = np.array([lct.get(c,0) for c in cc.index]); un_ct = np.array([lct.get(c,0) for c in unrel_ids])
print(f"relocated dt.ct links/event: median {np.median(rel_ct):.0f}")
print(f"unrelocated dt.ct links/event: median {np.median(un_ct):.0f};  fully isolated (0 links): {(un_ct==0).sum()}/{len(unrel_ids)}")
fig = pygmt.Figure()
with pygmt.config(MAP_FRAME_TYPE="plain", FORMAT_GEO_MAP="ddd.xx"):
    basemap_context(fig, "Relocated (dt.cc) vs unrelocated (isolated)")     # coast + fault + stations + scale
    fig.plot(x=ulo, y=ula, style="c0.13c", pen="0.6p,gray55")           # unrelocated: grey opens
    pygmt.makecpt(cmap="turbo", series=[2, 20], reverse=True)
    fig.plot(x=cc.lon, y=cc.lat, style=MARK, fill=cc.depth, cmap=True, pen="0.2p,gray30")
    fig.colorbar(frame=["x+lDepth","y+lkm"], position="JMR+o0.5c/0c+w9c")
fig.show(width=780)""")

md(r"""## 4 · Hour-of-day (KST) of the dt.cc-resolved events

Quarry blasts fire in daytime working hours (**06-18 KST, shaded**) and, being fixed-site repeaters,
cross-correlate and relocate; tectonic earthquakes are ~uniform over 24 h. Below (dt.cc-resolved subset only):
the hour-of-day **histogram** (bars coloured by the cyclic `hsv` hour colormap of the first-round blast-decluster
notebooks) and the hour-of-day **map** (GMT `cyclic` colormap, same as nb21) showing *where* daytime events cluster.""")

co(r"""_cyc = plt.get_cmap("hsv")                                    # first-round cyclic hour colormap
h = np.histogram(ccres.kst_hour, bins=np.arange(25))[0]
fig, ax = plt.subplots(figsize=(7.6, 4.0))
for k in range(24):
    ax.bar(k, h[k], width=1.0, align="edge", color=_cyc((k+0.5)/24.0), edgecolor="white", lw=0.3, zorder=2)
ax.axvspan(6, 18, color="0.80", alpha=0.45, zorder=0)                  # daytime 06-18 KST
frac = float(((ccres.kst_hour >= 6) & (ccres.kst_hour < 18)).mean())
ax.set(xlim=(0, 24), xticks=[0, 6, 12, 18, 24], xlabel="Hour of day (KST)", ylabel="Events",
       title=f"dt.cc-resolved hour-of-day (n={len(ccres)}, daytime frac={frac:.2f})")
fig.tight_layout(); plt.show()
print(f"dt.cc-resolved (n={len(ccres)}): daytime(06-18) {frac*100:.0f}%   noon(11-13) "
      f"{((ccres.kst_hour>=11)&(ccres.kst_hour<13)).mean()*100:.0f}%")""")

co(r"""# hour-of-day MAP of the dt.cc-resolved events, GMT 'cyclic' colormap (first-round nb21 style)
fig = pygmt.Figure()
with pygmt.config(MAP_FRAME_TYPE="plain", FORMAT_GEO_MAP="ddd.xx"):
    basemap_context(fig, f"dt.cc-resolved events by hour of day (KST), n={len(ccres)}")  # coast+fault+stations+scale
    pygmt.makecpt(cmap="cyclic", series=[0, 24, 1], continuous=True)     # nb21 hour-of-day colormap (GMT 'cyclic')
    fig.plot(x=ccres.lon, y=ccres.lat, style=MARK, fill=ccres.kst_hour, cmap=True, pen="0.2p,gray20")
    fig.colorbar(frame=["xa6+lHour of day (KST)"], position="JMR+o0.5c/0c+w9c")
fig.show(width=820)""")

md(r"""## 5 · Depth cross-sections of the dt.cc-resolved events

E-W and N-S projections of the **dt.cc-resolved** catalog (depth positive-down): the vertical extent and
dipping structure resolved by the sub-metre cross-correlation locations.""")

co(r"""fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
km_lon = np.cos(np.radians(35.75))*111.19
ax[0].scatter((ccres.lon-REGION[0])*km_lon, ccres.depth, s=16, c=ccres.depth, cmap="turbo_r", vmin=2, vmax=20,
              edgecolor="0.3", linewidth=0.2)
ax[0].set(xlabel="E-W distance (km)", ylabel="Depth (km)",
          title=f"E-W depth section (dt.cc-resolved, n={len(ccres)})"); ax[0].invert_yaxis()
ax[1].scatter((ccres.lat-REGION[2])*111.19, ccres.depth, s=16, c=ccres.depth, cmap="turbo_r", vmin=2, vmax=20,
              edgecolor="0.3", linewidth=0.2)
ax[1].set(xlabel="S-N distance (km)", ylabel="Depth (km)",
          title=f"S-N depth section (dt.cc-resolved, n={len(ccres)})"); ax[1].invert_yaxis()
fig.tight_layout(); plt.show()""")

md(r"""## 6 · Cumulative seismicity over 2016

Cumulative event count through 2016. **Left**: all detected UF-box events (the full seismicity, dominated by the
step at the 2016-09-12 M5.8 Gyeongju mainshock). **Right**: the relocation subsets (dt.cc-relocated and the
dt.cc-resolved subset), which accumulate more gradually. The M5.8 mainshock (2016-09-12 11:32 UTC) is dashed.""")

co(r"""import matplotlib.dates as mdates
MEI = "/home/msseo/works/02.Ulsan_Fault_detection/detection_test/reloc_2016_uf/members_event_idx.csv"
allt = pd.to_datetime(pd.read_csv(MEI).time, utc=True).dt.tz_localize(None).sort_values()   # all detected (3867)
def times(d):                                                    # UTC datetimes from a reloc DataFrame
    return pd.to_datetime(dict(year=d.yr, month=d.mo, day=d.dy, hour=d.hr, minute=d.mi,
                               second=d.sc.clip(0, 59))).sort_values()
cct, ccrest = times(cc), times(ccres)
ms = pd.Timestamp("2016-09-12 11:32")
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
ax[0].plot(allt.values, np.arange(1, len(allt)+1), lw=1.7, color="0.2")
ax[0].axvline(ms, color="crimson", lw=1.2, ls="--")
ax[0].set(title=f"All detected UF-box events (n={len(allt)})", xlabel="2016", ylabel="Cumulative count")
ax[1].plot(cct.values, np.arange(1, len(cct)+1), lw=1.5, color="#cc7733", label=f"dt.cc-relocated (n={len(cct)})")
ax[1].plot(ccrest.values, np.arange(1, len(ccrest)+1), lw=2.1, color="#3366aa", label=f"dt.cc-resolved (n={len(ccrest)})")
ax[1].axvline(ms, color="crimson", lw=1.2, ls="--", label="M5.8 Gyeongju (Sep 12)")
ax[1].set(title="Relocation subsets", xlabel="2016", ylabel="Cumulative count")
ax[1].legend(loc="upper left", framealpha=1)
for a in ax:
    a.xaxis.set_major_locator(mdates.MonthLocator((1, 3, 5, 7, 9, 11)))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b")); a.grid(alpha=0.3)
fig.suptitle("Cumulative seismicity over 2016", y=1.02); fig.tight_layout(); plt.show()
print(f"pre-Sep12 fraction: all detected {(allt<ms).mean()*100:.0f}%, dt.cc-resolved {(ccrest<ms).mean()*100:.0f}%")""")

md(r"""## 7 · Summary""")

co(r"""bar = "="*118
print(bar); print("2016 ULSAN-FAULT GJ-ARRAY RELOCATION — SUMMARY".center(118)); print(bar)
tab = pd.DataFrame({
    "set": ["HypoInverse (abs)", "dt.ct", "dt.cc"],
    "n_events": [len(loc), len(ct), len(cc)],
    "depth_med_km": [round(loc.depth.median(),1), round(ct.depth.median(),1), round(cc.depth.median(),1)],
    "depth_range_km": [f"{loc.depth.min():.1f}-{loc.depth.max():.1f}",
                       f"{ct.depth.min():.1f}-{ct.depth.max():.1f}", f"{cc.depth.min():.1f}-{cc.depth.max():.1f}"],
})
print(tab.to_string(index=False))
_,h_lc,v_lc = disp(loc,cc)
day_cc = ((cc.kst_hour>=7)&(cc.kst_hour<19)).mean()*100
noon_cc = ((cc.kst_hour>=11)&(cc.kst_hour<13)).mean()*100
print(f'''
TAKE-HOMES
  - Event set: 596 uf_cluster-QC'd events (erh<5, erz<5, gap<270, num>5, rms<1.0) of 3867 detected.
  - dt.cc relocated {len(cc)} events (kim2011, ISTART=2, adaptive LSQR damping, CND 40-80); {len(ccres)} cc-resolved.
  - Median HypoInverse -> dt.cc shift: {h_lc:.0f} m horizontal, {v_lc:.0f} m vertical; dt.cc relative error ~1 m.
  - Depth {cc.depth.min():.1f}-{cc.depth.max():.1f} km (median {cc.depth.median():.1f}); collapses onto the UF trace + NE strands.
  - ~half the QC'd events are UNRELOCATED - spatially isolated (median {int(np.median(un_ct))} vs {int(np.median(rel_ct))} dt.ct links;
    {int((un_ct==0).sum())} with zero links), NOT a quarry-blast population (blasts cluster and relocate).
  - Diurnal: {day_cc:.0f}% of dt.cc events daytime, {noon_cc:.0f}% at noon (11-13 KST) -> a blast contribution is present
    in BOTH relocated and unrelocated sets; a uf_cluster.flag_blasts declustering pass is the separate next step.
NEXT: quarry-blast declustering (uf_cluster) for a tectonic-only catalog; GJ-array vs KS/KG-only comparison.''')
print(bar)""")

nb = nbf.v4.new_notebook(); nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
OUT = "/home/msseo/works/02.Ulsan_Fault_detection/detection_test/reloc_2016_uf/09.UF2016_reloccmp.ipynb"
nbf.write(nb, OUT)
print("wrote", OUT)
