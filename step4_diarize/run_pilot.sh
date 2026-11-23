#!/usr/bin/env bash
# Diarization pilot: run every variant on the pilot sample, then compare.
set -u
cd "$(dirname "$0")/.."
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
PY=../whisperx-env/bin/python
P=data/output/diarization/pilot_episodes.txt
S40=data/output/diarization/pilot_stock40.txt
run() { echo "[$(date -u +%FT%TZ)] >>> $*"; $PY step4_diarize/diarize.py --episodes-file "$@" 2>&1 | grep -E "^\[.*(finished|FAIL|progress)" | cut -c1-160; }
run $P  --suffix s3act                                            # default: community-1, stride 3, active, fp16
run $P  --suffix s3act31 --model pyannote/speaker-diarization-3.1 # AHC + relative mcs
run $P  --suffix s3dom   --embed-mode dominant                    # SDBench per-chunk
run $P  --suffix s1act   --stride 0.1                             # stride-1 reference (= stock output, 5x faster)
run $S40 --suffix stock  --stock                                  # pyannote as shipped, first 40 episodes
echo "[$(date -u +%FT%TZ)] >>> compare"
$PY step4_diarize/pilot_eval.py compare --reference s1act 2>&1 | grep -E "^\||wrote"
$PY step4_diarize/pilot_eval.py compare --reference stock --episodes-file $S40 --append 2>&1 | grep -E "^\||wrote"
$PY step4_diarize/identify_hosts.py --variant s3act 2>&1 | tail -6
echo "[$(date -u +%FT%TZ)] >>> pilot done"
