#!/usr/bin/env bash
# Second archive pass: (1) rate-limited (429) rows and leftover raw files at normal
# settings; (2) a gentle pass over the 403/404/410 rows with 2 workers and a 5 s
# per-show delay, in case the blocks were triggered by our own request rate.
set -u
cd "$(dirname "$0")/.."
echo "[$(date -u +%FT%TZ)] pass 1: 429s and leftovers"
.venv/bin/python step4_diarize/archive_download.py --workers 8 --encoders 6 2>&1 | grep -E "targets:|progress|=== done" | cut -c1-160
echo "[$(date -u +%FT%TZ)] pass 2: gentle retry of permanent failures"
PPC_SHOW_DELAY=5 .venv/bin/python step4_diarize/archive_download.py --workers 2 --encoders 4 --retry-permanent 2>&1 | grep -E "targets:|progress|=== done" | cut -c1-160
echo "[$(date -u +%FT%TZ)] archive retry done"
