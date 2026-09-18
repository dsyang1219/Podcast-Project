# How the transcription pipeline batches work, and where its load spikes come from

Written for the rma1/rma2 administrators, 10 September 2026. Everything here is read from the code as it runs (`step3_transcribe/run_ladder.sh`, `step3_transcribe/download_audio.py`, `step3_transcribe/transcribe.py`), not from memory. One correction up front: **the pipeline does not use WhisperX.** The virtualenv on rma1 is named `whisperx-env` for historical reasons, but the worker is faster-whisper's `BatchedInferencePipeline` on a CTranslate2 model, with no diarization and no alignment pass.

## 1. Process layout on each node

One supervisor shell script (`step3_transcribe/run_ladder.sh`) starts and babysits two long-lived processes:

| process | what it does | count |
|---|---|---|
| `download_audio.py` | fetches audio from podcast feeds, pre-decodes it to WAV, drops it in a queue directory | 1 |
| `transcribe.py` | holds the Whisper model on the GPU, claims files from the queue one at a time, writes transcripts | 1 per node (`TRANSCRIBERS=1`) |

The supervisor wakes every 30 s. If either child has died it restarts it after `RESTART_SLEEP`. That restart behaviour matters for load: **a restarted transcriber reloads the model from disk into GPU memory**, which is the single largest transient the pipeline produces (see §4). If a worker were crashing repeatedly, the node would see a model-load spike every restart interval.

The two processes talk only through the filesystem: the downloader writes `<episode>.wav` plus a `<episode>.meta` sidecar into `data/audio_queue/`; the transcriber claims a file by atomically renaming `.meta` → `.meta.claimed`, transcribes it, and deletes the audio. There is no shared memory and no IPC.

## 2. The downloader: where the CPU and network bursts come from

- **HTTP concurrency.** A thread pool of `--workers` connections (16 by default; our launches use 6 on rma2 and 8 on rma1), with politeness pacing of one request per second per show and a per-show circuit breaker that parks a show for the run after repeated failures.
- **Backpressure.** The downloader stops fetching when `MAX_QUEUED` episodes are sitting in the queue (120 in our launches; default 250) and polls until the transcriber has consumed some. So it runs in **bursts**: it sleeps while the queue is full, then when the transcriber frees slots it wakes all its threads at once.
- **Pre-decoding.** Every downloaded file is transcoded with ffmpeg to 16 kHz mono 16-bit WAV before it enters the queue, because faster-whisper otherwise decodes the container itself on the GPU worker's critical path (25 s per mp3 versus 1.35 s from WAV). These ffmpeg processes run at `nice -n 10`, each takes about one core for ~20 s per episode, and they are capped at **6 concurrent** by a semaphore. That cap exists precisely because 16 unbounded transcodes were observed to monopolise the 20 cores and starve the GPU worker's VAD.
- **Disk.** WAV at 16 kHz mono is ~115 MB per audio-hour; a full queue of 120 episodes at a median 47 minutes is roughly 11 GB of scratch, written and deleted continuously.

So on the CPU side, the characteristic pattern is: quiet while the queue is full, then a burst of up to 6 ffmpeg processes plus 6–8 concurrent downloads for a minute or two, then quiet again. On a long-episode stretch the bursts are further apart; on a short-episode stretch (news briefs) they come every few minutes.

## 3. The GPU worker: how audio is batched into Whisper

Per episode, in order:

1. **Thermal gate (optional, on in our runs).** Before claiming the next file the worker polls `nvidia-smi` for GPU temperature, estimates the idle "floor" as the 10th percentile of recent readings, and waits (up to 300 s per episode, 600 s hard cap) until the die is within `--start-above-floor` degrees of it (2 °F in our launches). rma1's launcher also sets `--target-temp-f 138`. Note the worker's own comment: idling does *not* let the GPU leave P0 — a live CUDA context pins it at ~14 W — so the gate reduces duty cycle, nothing else.
2. **Claim** one file (atomic rename).
3. **Decode + VAD (CPU).** `BatchedInferencePipeline.transcribe(path, batch_size=64, language="en")` reads the WAV into memory as 16 kHz float32 (a 3-hour episode is ~690 MB), runs Silero voice-activity detection over the whole file on CPU, and cuts the speech into segments of at most 30 s.
4. **Batched inference (GPU).** Segments are grouped into batches of **64**. Each batch is 64 × ≤30 s ≈ up to 32 minutes of audio: the log-mel features for the batch are computed, the encoder runs once over the batch, and the decoder generates text for all 64 segments together. A 47-minute episode is therefore about 2 batches; a 3-hour interview is about 6. Each batch is a solid burst of GPU work lasting a few seconds; between batches and between episodes the GPU is idle while the CPU does VAD, JSON writing and garbage collection. Measured with one worker that idle share is about half the wall clock, which is why throughput is ~100–140× real time rather than the ~400× the encoder alone could sustain.
5. **Write** JSON and text transcripts, append a status row, delete the audio (or, on holdout runs only, hand it to one background thread that Opus-encodes it at `nice -n 15` — off for the current runs).

Batch size and memory, as measured on these boxes: the model plus a batch-64 working set holds **~10 GB of unified memory**; batch 128 raises that to ~15 GB for 1.24× throughput with identical output; two workers roughly double it. We run one worker at 64 deliberately to keep sustained draw and heat down (38 W vs 51 W GPU power at 64 vs 128).

## 4. Where the spikes are

In descending order of size, the transients a node sees from this pipeline:

1. **Model load** at worker start: the 1.6 GB CTranslate2 model is read from disk and materialised in GPU memory, plus CUDA context creation. Happens once per run — unless the supervisor is restarting a crashing worker, in which case it repeats every restart. If the spikes you saw were periodic, check `step3_transcribe/logs/ladder_transcribe.log` on rma1 for repeated "Loading … model" lines and `data/output/ladder_status.json` for `restarting` states.
2. **Long episodes.** A 3-hour file means ~700 MB of float32 audio in unified memory, VAD over the whole file on CPU, then six back-to-back GPU batches with no idle gap between them — the longest continuous full-power GPU period the pipeline produces (about two minutes at 100×). GB10's die heats at ~13 °F/s at burst onset; that is what the gate is for.
3. **Downloader wake-ups**: up to 6 ffmpeg processes and 6–8 HTTP threads starting together when queue space frees (§2).
4. **Steady state** is modest: one GPU worker at ~40 W, one to two ffmpeg processes, a few hundred MB/min of disk churn.

Nothing in the pipeline spawns processes per episode on the GPU side, nothing forks the model, and there is no multiprocessing pool: one model, one process, one file at a time.

## 5. What I would ask you to check on rma1

- `journalctl -k` / `dmesg` around the outage for OOM-killer or thermal-shutdown lines. If the OOM killer fired, its victim line will name the process; if it names `python` at ~10 GB, that is our worker, and I would like to know what else was resident, because unified memory means the GPU worker competes with every CPU process on the box.
- Our own logs on rma1, in `/home/dsyang/Podcast-Project/political-podcast-corpus/transcription/`: `ladder_transcribe.log` (per-episode duration, wall time, real-time factor, gate waits) and `ladder_download.log` (burst timing), plus `data/output/ladder_status.json`. The last episode logged before the outage tells you whether it died mid-batch on a long file.
- Whether anything else was on the GPU at the time. Earlier this month another user's job shared rma1 and our gate accommodated it, but two GPU tenants plus our downloader bursts is the configuration I would expect to trip a marginal power or thermal limit.

## 6. Levers we can pull without root, if you want the load flattened

- `BATCH_SIZE=32` halves the per-batch working set (slower, same output).
- `MAX_QUEUED=40` and `WORKERS=4` shrink the downloader bursts.
- `GPU_THREAD_PCT` caps the worker's share of streaming multiprocessors through CUDA MPS, which keeps the GPU continuously busy at lower power instead of duty-cycling it — the cleanest way to run cooler, and unused so far only because the MPS daemon has to be running.
- A hard cap on episode length (skip files over, say, 2.5 hours) removes the longest continuous bursts at the cost of a handful of interview episodes.

Tell me which of these you would like set and I will relaunch rma1's shard with them once my key is back on the box. rma2 is running the same pipeline on its half of the manifest and has been stable for four days.
