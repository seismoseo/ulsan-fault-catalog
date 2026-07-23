# Instrument-response inventory for the Ulsan-Fault ML pipeline

Two-source inventory consumed by `ml_pipeline.load_combined_inventory(...)`:

```
responses/
├── master/                            # 142 MB, gitignored
│   └── KS_KG_metadata_1.0.2.xml       # combined StationXML covering 52 / 59 stations
└── fetched/                           # 7 stations the master file is missing
    ├── zips/                          # raw NECIS zips, one per station (tracked)
    │   └── RESP_KS_<STA>.zip
    └── extracted/                     # SEED RESP files, ready for obspy.read_inventory()
        └── RESP.KS.<STA>..<CHAN>      # 21 files (7 stations × 3 channels)
```

`master/` is a one-time copy of the v1.0.2 KS+KG metadata and is **excluded from git**
(see [.gitignore](../../../.gitignore)). `fetched/` is **tracked** so re-fetching is
optional, not mandatory.

## What `fetched/` contains

The seven KS stations the master file does not cover (per the v1.0.0 coverage report
in `01.Calculate_ML_with_full_response.ipynb` §2):

```
BAEA  DAJA  GJAA  HYDA  NARA  SRGA  UICA
```

For each station the directory holds the raw NECIS zip (`zips/RESP_KS_<STA>.zip`,
~1 MB) and the three extracted SEED RESP files (`extracted/RESP.KS.<STA>..HG{E,N,Z}`).
`obspy.read_inventory()` reads either format directly, so the loader concatenates the
master StationXML and the SEED RESP files without any conversion step.

## How `fetched/` was produced

By the **NECIS RESP fetcher** that ships with the `necis-downloader` project:

- repo: <https://github.com/seismoseo/necis-downloader>
- pinned commit: `158f1d1`  (*"Add NECIS RESP fetcher (necis/responses.py + fetch_responses.py CLI)"*)
- CLI: `fetch_responses.py` → POSTs to `/necis-dbf/usernl/ob/observatoryListEarthDown.do` with the same authenticated session as the waveform downloaders.
- credentials: `NECIS_USER` / `NECIS_PASS` in the `necis-downloader` `.env` (same KMA NECIS account the continuous- and event-waveform fetchers use).

The seven station codes are stored in [fetch_missing.sh](fetch_missing.sh) so the
recipe stays in version control.

## Re-running the fetch

```bash
# from this directory:
bash fetch_missing.sh
```

The wrapper expects the `necis-downloader` repo to be checked out at
`$HOME/works/Claude` (the default) — override with the `NECISDL` env var if your
checkout lives elsewhere. It runs `fetch_responses.py` for the seven stations
listed above and writes into `fetched/zips/` + `fetched/extracted/`. Output is
byte-identical modulo NECIS server-side date stamps inside the zip filenames; the
extracted `RESP.*` files themselves are identical.

If you start from a clean machine:

```bash
# 1. Clone necis-downloader at the pinned commit
git clone https://github.com/seismoseo/necis-downloader $HOME/works/Claude
cd $HOME/works/Claude
git checkout 158f1d1
pip install -e .                      # installs playwright + requests + obspy deps
playwright install chromium           # one-time browser dependency
cp .env.example .env                  # then edit NECIS_USER / NECIS_PASS

# 2. Re-fetch
cd <ULSAN_REPO>/KS_KG/local_magnitudes/responses
bash fetch_missing.sh
```

## Verifying

```python
import obspy
from ml_pipeline import load_combined_inventory
inv = load_combined_inventory("responses/master", "responses/fetched/extracted")
# Should print 59 KS stations:
print(sum(1 for net in inv for sta in net if net.code == "KS"))
```

Cross-checks against the per-event SAC trees under
[`../HypoInv/event_waveforms_blastclean/`](../../HypoInv/) and
[`../HypoInv/event_waveforms_ulsanfault/`](../../HypoInv/) are run by
`01.Calculate_ML_with_full_response.ipynb` §3.
