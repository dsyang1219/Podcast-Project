# Step 3 — download and transcribe

**What this step does.** Takes a manifest from step 2, downloads each episode's audio, and transcribes it on the
GPU. Two long-running processes form a producer/consumer pair through a queue directory, and a shell supervisor
keeps both alive for the days or weeks a run takes.

**Input.** A manifest CSV (`data/output/sample_out/sample_manifest_*.csv`).

**Output.** `data/transcripts/<show_id>/<episode_id>_<slug>.json` (segments with timestamps) and `.txt`
(plain text), plus `data/output/transcribe_status.csv` and `data/output/ladder_status.json` (live status).

## How the pieces fit

```
run_ladder.sh  (supervisor, runs at the root)
   |-- download_audio.py   manifest -> data/audio_queue/<episode>.<ext> + .meta   (producer)
   '-- transcribe.py  x N  claims .meta -> .meta.claimed, transcribes, writes transcript  (consumer)
```

| file | role |
|---|---|
| `run_ladder.sh` | Starts one downloader and `TRANSCRIBERS` GPU workers, restarts any that die, writes `ladder_status.json` on every state change. Configured by environment variables: `MANIFEST`, `WORKERS` (download threads), `MAX_QUEUED` (back-pressure), `BATCH_SIZE`, `TRANSCRIBERS`, `START_ABOVE_FLOOR` (thermal gate), `ARCHIVE=0` (do not keep audio), `PREDECODE`, `GPU_VENV`. |
| `download_audio.py` | Streams each enclosure URL to the queue with retry and back-off (settings imported from `step1_frame.config`), writes the `.meta` file last so a worker never claims a partial file, rebuilds its skip sets from disk at start (already transcribed or already queued episodes are skipped), and writes `STOP` when the manifest is exhausted. |
| `transcribe.py` | faster-whisper `BatchedInferencePipeline` on the CTranslate2 model `deepdml/faster-whisper-large-v3-turbo-ct2`, float16, batch 64, English, Silero VAD, no diarization. Compares the audio hash against any existing transcript before working, so a restart never re-transcribes. Optional thermal start gate and CPU/GPU fallbacks are documented in its docstring. |
| `launchers/` | The one-off shell scripts that chained the September 2026 runs on rma2 (holdout shows, the rma1 shard, recovery passes). Kept as the run history; paths have been updated to this layout. |
| `logs/` | Older run logs (`chain_*.log`, `download_audio.log`). New runs of `run_ladder.sh` write here. |

## Running

```
MANIFEST=data/output/sample_out/sample_manifest_band15.csv WORKERS=6 MAX_QUEUED=120 START_ABOVE_FLOOR=2 \
ARCHIVE=0 PPC_HTTP_TIMEOUT=120 PPC_HTTP_RETRIES=4 LD_LIBRARY_PATH=/opt/ctranslate2/lib \
setsid nohup bash step3_transcribe/run_ladder.sh > step3_transcribe/logs/run_$(date -u +%m%dT%H%M).log 2>&1 &
```

Run `transcribe.py --smoke-test` once on a new machine to confirm the GPU is actually used. Hardware, environment
and throughput notes for the two NVIDIA GB10 machines are in `handoff/notes/RMA2_SETUP.md` and
`handoff/notes/PIPELINE_LOAD_PROFILE.md`.

## Run history note

The band-15 run was first launched on 12 September 2026 under the old folder name, was killed by a planned reboot
of rma2 the same afternoon after 1,307 episodes, and was relaunched from this folder. Its logs are in `logs/`
(`band15_rma2_0912T0447.log` for the first launch, `band15_rma2_restart_*.log` afterwards, plus the shared
`ladder_download.log` / `ladder_transcribe.log`). Nothing depends on the old `transcription/` name any more.
