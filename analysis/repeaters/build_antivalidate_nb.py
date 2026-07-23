"""Build the multi-station anti-repeater validation notebook — FAIR ACROSS CHANNELS.
Usage: python build_antivalidate_nb.py

Takes notebook 06's HDB anti-pair candidates (top-N by cc_lag0, a tight ~1 km cluster) and shows how
the SAME pairs correlate / anti-correlate at the **genuinely closest** stations — selected by DISTANCE
to the cluster and read on each station's **native vertical channel** (HHZ, HGZ, or ELZ). The earlier
version used `comp='HHZ'` everywhere, which silently dropped every HG/EL station — including the
closest ones (KG.SIG 7 km, KS.YGBA 10 km, KG.BOG 13 km on HG). No data was missing; it was a band-code
filter. The catalog is unaffected (locations use the picks from all channels).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md("""# Anti-repeater candidates — multi-station check, fair across HH / HG / EL channels

Notebook 06 ranks HDB pairs by **`cc_lag0`** (signed CC at the exact P datum) and its top candidates
form a tight ~1 km cluster of strongly lag-0-**anti-correlated** HDB pairs. We test those exact pairs
at the stations **closest to the cluster**, each on its **native vertical channel**.

**Why this matters.** Stations split by SEED band-code: **HH** (broadband: HDB, MKL, YSB, CHS, JEJB…),
**HG** (SIG, YGBA, BOG, CGU…), **EL** (short-period: CGD, DKJ, HAK…). An analysis fixed to `HHZ`
silently excludes HG/EL — and the *closest* stations to this cluster are HG. So we pick stations by
**distance** and use whatever vertical channel each one actually records. (No data is missing — the
HG/EL stations are fully archived; the catalog locations already use their picks.)

**The test.** A genuine co-located anti-repeater is the same waveform up to **sign** at every station
(`max(|cc_pos|,|cc_neg|)` high everywhere, sign set by take-off geometry). Decorrelation at the close
stations ⇒ the HDB lag-0 negative is a single-station effect. Read `cc_pos` (similar at all?) together
with `cc_lag0` (anti at the datum). **Caveat:** HG (≈accelerometer-class) and HH (broadband) are not
directly comparable for fine 1-25 Hz correlation of small events — low `cc_pos` at an HG station can be
instrument/SNR, not source; §5 offers a control.""")

md("## Parameters")
code("""STATION   = "KG.HDB"          # screening station (HHZ)
COMP      = "HHZ"             # screening channel
WIN       = (-0.5, 7.5)
BANDS     = [(1, 25), (1, 10), (2, 8), (4, 12), (5, 15)]
BAND      = (1, 25)
MAXLAG    = 0.2
TOPN      = 12               # notebook 06's top-N pairs by most-negative cc_lag0
N_STATIONS = 6               # how many of the closest stations to validate against
N_OVERLAY = 6                # candidate pairs to draw as multi-station overlays
CACHE     = "wf_similarity_cache\"""")

code("""import os, sys, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
from obspy.geodetics.base import gps2dist_azimuth
from uflib import uf_waveform_similarity as wf
from uflib import uf_cluster as ufc
wf.use_helvetica()
CACHE = wf.CACHE_DIR          # absolute cache dir (overrides the relative default; cwd-independent)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 50)
WF = wf.WF_ROOT""")

md("""## 1 · Notebook 06's candidate pairs (HDB, rank by `cc_lag0`, top N)""")
code("""resH = wf.make_bands(wf.list_events(station=STATION, comp=COMP), station=STATION, comp=COMP,
                     bands=BANDS, win=WIN, cache_dir=CACHE, verbose=False)
keptH = resH["kept"]; meta = wf.load_event_meta(keptH)
SH = wf.signed_similarity(resH["bands"][BAND], maxlag=MAXLAG)
iu = np.triu_indices(len(keptH), k=1)
order = np.argsort(SH["cc_lag0"][iu])[:TOPN]
PAIRS = [dict(i=int(iu[0][o]), j=int(iu[1][o])) for o in order]
evset = sorted(set([p["i"] for p in PAIRS]) | set([p["j"] for p in PAIRS]))
cand_events = [keptH[i] for i in evset]
clat = float(meta.iloc[evset]["lat"].mean()); clon = float(meta.iloc[evset]["lon"].mean())
print(f"top-{TOPN} pairs -> {len(cand_events)} events, centroid ({clat:.4f}, {clon:.4f}); "
      f"years {sorted({e[:4] for e in cand_events})}")
pd.DataFrame([dict(ev_i=keptH[p["i"]], ev_j=keptH[p["j"]],
                   cc_lag0=round(float(SH["cc_lag0"][p["i"], p["j"]]), 3),
                   cc_pos=round(float(SH["cc_pos"][p["i"], p["j"]]), 3)) for p in PAIRS])""")

md("""## 2 · Select the CLOSEST stations (by distance, native vertical channel)

Distances from station coordinates (`STA/UF<year>.sta`); native vertical channel = whichever of
`HHZ / HGZ / ELZ` the station actually has SAC for. We keep the `N_STATIONS` closest that cover all
candidate events.""")
code("""# station coords from the per-year HYPOINVERSE station tables (union over candidate years)
coords = {}
for y in sorted({e[:4] for e in cand_events}):
    f = os.path.join(wf.STA_DIR, f"UF{y}.sta")
    if os.path.exists(f):
        for ln in open(f):
            pr = ln.strip().split(",")
            if len(pr) >= 3:
                try: coords[pr[0]] = (float(pr[1]), float(pr[2]))
                except ValueError: pass

def native_vz(stacode):                       # vertical channel a station records (HH/HG/EL...)
    for ch in ("HHZ", "HGZ", "ELZ"):
        if any(os.path.exists(os.path.join(WF, e, f"{e}.{stacode}.{ch}.sac")) for e in cand_events[:6]):
            return ch
    return None

rows = []
for st, (la, lo) in coords.items():
    vz = native_vz(st)
    if vz is None:
        continue
    cov = sum(os.path.exists(os.path.join(WF, e, f"{e}.{st}.{vz}.sac")) for e in cand_events)
    if cov == 0:
        continue
    d = gps2dist_azimuth(clat, clon, la, lo)[0] / 1000.0
    rows.append(dict(station=st, channel=vz, dist_km=round(d, 1), coverage=f"{cov}/{len(cand_events)}", _cov=cov))
stab = pd.DataFrame(rows).sort_values("dist_km").reset_index(drop=True)
# validators: the N closest with full coverage (HDB is the screen, always first)
full = stab[stab["_cov"] == len(cand_events)]
SEL = full.head(N_STATIONS)[["station", "channel", "dist_km"]].values.tolist()
print("closest fully-covering stations (all channel band-codes):")
display(stab[stab["_cov"] == len(cand_events)].head(12).drop(columns="_cov"))
print("selected for validation:", [(s, c, f"{d}km") for s, c, d in SEL])""")

md("""## 3 · Build features at each selected station (native channel) and the signed CC

Each station is P-aligned on its own pick (fallback+xcorr if unpicked); the ±MAXLAG search absorbs
small jitter. We report `cc_lag0` (signed at the datum), `cc_pos` (best +, "similar at all?") and
`cc_neg` (best −) for every candidate pair.""")
code("""SIG = {}
for st, ch, _ in SEL:
    r = wf.make_bands(wf.list_events(station=st, comp=ch), station=st, comp=ch,
                      bands=BANDS, win=WIN, cache_dir=CACHE, verbose=False)
    S = wf.signed_similarity(r["bands"][BAND], maxlag=MAXLAG)
    SIG[st] = (S, {e: k for k, e in enumerate(r["kept"])}, ch, r)
    print(f"  {st}.{ch}: {len(r['kept'])} events")

def _tab(key):
    out = []
    for p in PAIRS:
        ei, ej = keptH[p["i"]], keptH[p["j"]]
        rec = {"ev_i": ei, "ev_j": ej}
        for st, ch, d in SEL:
            S, idx, _, _ = SIG[st]
            rec[f"{st.split('.')[1]}/{ch[:2]}({d:.0f})"] = (
                round(float(S[key][idx[ei], idx[ej]]), 2) if (ei in idx and ej in idx) else np.nan)
        out.append(rec)
    return pd.DataFrame(out)

for key, lab in [("cc_lag0", "cc_lag0 (signed, exact P datum)"),
                 ("cc_pos",  "cc_pos (best positive — similar at all?)"),
                 ("cc_neg",  "cc_neg (best negative)")]:
    print("\\n###", lab); display(_tab(key))""")

md("""## 4 · Waveform overlays at the closest stations

Per pair, at each selected station: event *i* (black) vs *j* at its best **+CC** lag (blue, repeater
fit) and best **−CC** lag flipped (red, anti fit). Distance + channel in each row title.""")
code("""RESp = {st: SIG[st][3] for st, _, _ in SEL}
STATIONS = [st for st, _, _ in SEL]
for p in PAIRS[:N_OVERLAY]:
    wf.plot_antipair_stations(RESp, keptH[p["i"]], keptH[p["j"]], stations=STATIONS, band=BAND,
                              win=WIN, maxlag=MAXLAG, zoom=(-0.3, 4.0))
    plt.show()""")

md("""## 5 · How to read / verdict + an HG-vs-HH control

- **Validated reversal:** `max(|cc_pos|,|cc_neg|)` high at HDB **and** the close stations, with the
  winning sign flipping by geometry.
- **HDB-only artifact:** the close stations decorrelate (low `cc_pos` *and* low `|cc_neg|`).
- **Channel caveat (read this):** the closest stations here are **HG** (≈ accelerometer-class). HG and
  HH are not directly comparable for fine 1-25 Hz correlation of *small* events — a low `cc_pos` at an
  HG station may be instrument/SNR, not source. Distance can't be the cause when an HG station at 7 km
  decorrelates while an HH station at 10 km stays correlated; the **band-code** can.

**Control — can the HG channels resolve a *known* repeater at all?** Pick the tightest HDB repeater
family (high `cc_pos`) and check `cc_pos` at the same HG stations. If HG can't reproduce even a genuine
repeater, the HG decorrelation above is uninformative about the source; if it can, it is meaningful.""")
code("""# tightest positive-CC HDB pair (a near-certain repeater) as the control
pos = SH["cc_pos"][iu]; bo = int(np.argmax(pos))
ci, cj = int(iu[0][bo]), int(iu[1][bo]); ce_i, ce_j = keptH[ci], keptH[cj]
print(f"control repeater pair {ce_i} x {ce_j}: HDB cc_pos = {pos[bo]:.2f}")
ctl = {"pair": f"{ce_i}x{ce_j}", "HDB": round(float(SH['cc_pos'][ci, cj]), 2)}
for st, ch, d in SEL:
    S, idx, _, _ = SIG[st]
    ctl[f"{st.split('.')[1]}/{ch[:2]}"] = round(float(S['cc_pos'][idx[ce_i], idx[ce_j]]), 2) if (ce_i in idx and ce_j in idx) else np.nan
print("control cc_pos across stations (does HG resolve a true repeater?):")
display(pd.DataFrame([ctl]))""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = "/home/msseo/works/02.Ulsan_Fault_detection/data/hypoinv/repeaters/09_antirepeater_multistation_1-25Hz_phasenet_plus.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(C), "cells")
