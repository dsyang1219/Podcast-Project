#!/usr/bin/env bash
# Resume the paused recovery run: put its parked queue files back, then relaunch the recovery manifest
# (download_audio.py skips episodes already transcribed or already queued).
cd "$(dirname "$0")/../.."
mv data/audio_queue_recovery_paused/* data/audio_queue/ 2>/dev/null; rm -f data/audio_queue/STOP
export PPC_HTTP_TIMEOUT=120 PPC_HTTP_RETRIES=4 LD_LIBRARY_PATH=/opt/ctranslate2/lib:${LD_LIBRARY_PATH:-}
MANIFEST=data/output/sample_out/sample_manifest_recovery.csv WORKERS=6 MAX_QUEUED=120 START_ABOVE_FLOOR=2 step3_transcribe/run_ladder.sh
