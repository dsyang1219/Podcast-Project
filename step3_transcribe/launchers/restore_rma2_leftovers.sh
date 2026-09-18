#!/usr/bin/env bash
# Put rma2-shard episodes 29-50 back in the queue once every show has cleared the 25 floor.
cd "$(dirname "$0")/../.."; mv data/audio_queue_rma2_leftover/* data/audio_queue/ 2>/dev/null; echo "restored: $(ls data/audio_queue | grep -c '\.meta$') in queue"
