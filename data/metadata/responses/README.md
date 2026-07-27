# Instrument-response inventory (KS / KG / GJ / NS)

Every network that contributes picks must contribute a response. Local magnitude removes the
instrument response and simulates a Wood-Anderson seismograph, so a station with no response is
**silently dropped** from ML — the magnitude is then computed from a biased subset of the network
rather than failing loudly. This directory is the single source of truth for responses, the way
`../stations/` is for coordinates.

```
responses/
├── master/                          # KS + KG StationXML — 142 MB, gitignored
│   └── KS_KG_metadata_1.0.2.xml
├── fetched/                         # 7 KS stations the master file omits
│   ├── zips/                        #   raw NECIS zips (gitignored)
│   └── extracted/                   #   RESP.KS.<STA>..<CHAN>   21 files
├── gj/                              # GJ 2016-2017 temporary arrays
│   └── RESP.GJ.<STA>..HH{E,N,Z}     #   30 stations x 3 = 90 files
├── ns/                              # NS dense array (GHBSN)
│   └── RESP.NS.<STA>..HH{E,N,Z}     #   200 stations x 3 = 600 files
├── ns_derived/                      # DERIVED clones for N201-N220 — see below
│   └── RESP.NS.N2xx..HH{E,N,Z}      #   20 stations x 3 = 60 files
└── source/                          # raw MARA drop, kept for provenance (gitignored)
    ├── NS-HH.dataless               #   Dabeen Heo, 2020-06-03
    ├── RESP_NS200.zip               #   obspy-dataless2resp NS-HH.dataless
    ├── sacpz_1.00.zip               #   SACPZ form of the same, unused by the pipeline
    └── MEMO.{DATALESS,RESP}
```

`obspy.read_inventory()` reads StationXML and SEED RESP alike, so the loader concatenates them
without any conversion step.

## Loading

```python
from ufpipe import responses
inv = responses.load_inventory()                 # all four networks, master-first
responses.coverage_report(2021)                  # per-station: covered / n_channels
```

or from the ML side, which delegates to the same function:

```python
import ml_pipeline as mp
inv = mp.load_full_inventory()
```

`master/` is loaded first and the per-network RESP directories after, so where two sources cover
the same `(network, station, channel)` the authoritative entry wins.

Command line:

```bash
python -m ufpipe.responses --coverage 2021      # coverage against that year's station table
python -m ufpipe.responses --derive-ns          # regenerate ns_derived/ (idempotent)
```

Verified coverage is **100 % of the station table for every year 2010-2024**, all three channels.

## GJ and NS provenance

Both came from the MARA server (`RESP_from_mara/`, 2026-07-27).

* **GJ** — 30 stations, SEED RESP, epoch `2010,001`-`2099,001`. Covers all 29 GJ stations the
  2016-2017 station table needs; `PK07` is spare.
* **NS** — 200 stations, produced by `obspy-dataless2resp NS-HH.dataless` from a dataless dated
  2020-06-03. All 600 files are **byte-identical apart from the station code and the epoch start
  date** — the whole array shares one instrument configuration (sensor gain 754.3 x digitiser
  4e5 = 3.0172e8 counts/(m/s), internally consistent to the last digit).
  Three deployment epochs appear: `2016,256` (N001-N007), `2019,001` (139 stations),
  `2019,213` (54 stations). These match the archive: the only pre-2019 NS data on disk is
  N001-N007 in 2017-2018, so no real trace falls outside its response epoch.

## `ns_derived/` — read this before using N2xx magnitudes

The 2020 dataless predates the **N201-N220** block (first data 2022.237), so those 20 stations
have no authoritative response. `ufpipe.responses.write_derived_ns` clones the common NS response
for them, stamping each file with the station's own first-day-on-disk as the epoch start date.

**The assumption:** that the 2022 expansion continued the same instrument configuration as the rest
of the array. Corroborating but not conclusive — the N2xx stations record at the same 200 Hz, with
3 components, at count amplitudes in the same range as the covered stations. There is no
manufacturer record here to confirm it. If the array was re-equipped in 2022, ML from those
stations carries that error. Treat N2xx magnitudes as provisional and delete this directory the
moment real metadata arrives. Every file carries a `DERIVED by ufpipe.responses` header so it can
never be mistaken for authoritative metadata.

## Known quirk

Loading the KS/KG master StationXML emits
`WARNING (norm_resp): computed and reported sensitivities differ by more than 5 percent`.
This predates the GJ/NS work and comes from the master file, not from these RESP files — GJ and NS
stage gains multiply to their reported sensitivities exactly.

## `fetched/` — the 7 missing KS stations

```
BAEA  DAJA  GJAA  HYDA  NARA  SRGA  UICA
```

For each, the raw NECIS zip (`zips/RESP_KS_<STA>.zip`) and three extracted SEED RESP files
(`extracted/RESP.KS.<STA>..HG{E,N,Z}`). Produced by the **NECIS RESP fetcher** in the
`necis-downloader` project:

- repo: <https://github.com/seismoseo/necis-downloader>, pinned commit `158f1d1`
- CLI: `fetch_responses.py` → POSTs to `/necis-dbf/usernl/ob/observatoryListEarthDown.do`
- credentials: `NECIS_USER` / `NECIS_PASS` in the `necis-downloader` `.env`

Station codes live in [fetch_missing.sh](fetch_missing.sh) so the recipe stays in version control.

### Re-running the fetch

```bash
bash fetch_missing.sh     # from this directory
```

The wrapper expects `necis-downloader` at `$HOME/works/Claude` — override with `NECISDL`.
Output is byte-identical modulo NECIS server-side date stamps in the zip filenames.

### Verifying

```python
from ml_pipeline import load_combined_inventory
inv = load_combined_inventory("responses/master", "responses/fetched")   # note: fetched/, not fetched/extracted/
print(sum(1 for net in inv for sta in net if net.code == "KS"))
```
