#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
SHOWS="rogan shapiro beck tucker thedaily parnas kirk kelly ryan bongino"
while true; do
  rsync -aq -e "ssh -i $HOME/.ssh/id_ed25519_rma1 -o BatchMode=yes" --include='HOLDOUT_*/' --include='HOLDOUT_*/**' --exclude='*' rma1.caltech.edu:/home/dsyang/Podcast-Project/political-podcast-corpus/data/transcripts/ data/transcripts/ 2>/dev/null
  ok=1; line=""
  for s in $SHOWS; do n=$(ls data/transcripts/HOLDOUT_$s/*.json 2>/dev/null | wc -l); line="$line $s=$n"; [ "$n" -ge 25 ] || ok=0; done
  echo "[$(date -u +%FT%TZ)] floor watch:$line" >> step3_transcribe/logs/floor_watch.log
  if [ "$ok" = "1" ]; then
    echo "[$(date -u +%FT%TZ)] ALL 10 SHOWS >= 25. Restoring leftovers and re-enabling the thermal gate." | tee -a step3_transcribe/logs/chain_holdout.log step3_transcribe/logs/floor_watch.log
    TR=$(python3 -c "import json;print(' '.join(json.load(open('data/output/ladder_status.json')).get('transcribe_pids',[])))"); DL=$(python3 -c "import json;print(json.load(open('data/output/ladder_status.json')).get('download_pid',''))")
    for p in $TR $DL; do [ "$p" != "-1" ] && kill $p 2>/dev/null; done
    for p in $(pgrep -f '^bash \./run_ladder\.sh$'); do kill $p 2>/dev/null; done; sleep 3
    for f in data/audio_queue/*.meta.claimed; do [ -e "$f" ] && mv "$f" "${f%.claimed}"; done
    step3_transcribe/launchers/restore_rma2_leftovers.sh >> step3_transcribe/logs/chain_holdout.log 2>&1
    GATE=1 setsid nohup step3_transcribe/launchers/launch_rma1shard_on_rma2.sh >/dev/null 2>&1 </dev/null & disown
    echo "[$(date -u +%FT%TZ)] regated run launched; watcher exiting" >> step3_transcribe/logs/chain_holdout.log; exit 0
  fi
  sleep 120
done
