#!/usr/bin/env bash
#
# Re-fetch NECIS instrument-response (SEED RESP) files for the 7 KS stations the
# master StationXML (responses/master/KS_KG_metadata_1.0.2.xml) does not cover.
#
# Pinned to the `necis-downloader` commit that introduced the RESP fetcher:
#   github.com/seismoseo/necis-downloader @ 158f1d1
#
# Requires `NECIS_USER` / `NECIS_PASS` set in the necis-downloader `.env`.
# Override `NECISDL` if your checkout of necis-downloader lives elsewhere.
#
set -euo pipefail

NECISDL="${NECISDL:-$HOME/works/Claude}"
STATIONS="${STATIONS:-BAEA,DAJA,GJAA,HYDA,NARA,SRGA,UICA}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$NECISDL/fetch_responses.py" ]]; then
    echo "error: $NECISDL/fetch_responses.py not found." >&2
    echo "       Set NECISDL=<path-to-necis-downloader-checkout> and retry." >&2
    exit 2
fi

echo "[fetch_missing] necis-downloader at: $NECISDL"
echo "[fetch_missing] stations           : $STATIONS"
echo "[fetch_missing] output             : $HERE/fetched"

# Run from the necis-downloader checkout so its .env / package layout resolve.
cd "$NECISDL"
python fetch_responses.py --network KS \
    --stations "$STATIONS" \
    --out "$HERE/fetched"
