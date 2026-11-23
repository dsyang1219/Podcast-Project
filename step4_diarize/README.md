# Step 4 — Speaker diarization and host identification

Adds speaker turns to every transcribed episode and labels which speaker is the
host, so the register measures can be rebuilt on host speech alone and the
corpus can be released with a speaker layer.

## Why this design

Stock pyannote embeds each second of audio 20–30 times (10 s window, 1 s step,
one embedding per window and local speaker slot, including slots with no
speech). Profiling on this GPU (step `profile_stages.py`, 49-minute episode,
7 Nov 2026) put 96% of the time in that embedding loop. Three changes, all
of which leave the models untouched:

| change | effect on a 49-min episode | source |
|---|---|---|
| segmentation step 1 s → 3 s | 3× fewer windows | Yamaguchi 2026 (arXiv:2606.08505), SDBench 2025 (arXiv:2507.16136) |
| embed only (window, speaker) slots with speech | 2,955 → 1,129 passes, output identical (pyannote discards silent slots anyway) | this repo |
| fp16 autocast for the ResNet (filterbank stays fp32: it overflows in fp16) | 1.9× per batch | Yamaguchi 2026 ("fp16 + bs64") |

Measured: stock 21–25× realtime; stride 1 with the two embedding changes 110×
and *identical* output to stock (DER 0.000); stride 3 with all three ~300×
and DER 0.003 against stock on the test episodes. The 75,359-hour corpus
therefore takes about 250 GPU-hours instead of 3,500. The "dominant" mode
(one embedding per window, SDBench's per-chunk variant) adds only ~10% and
is off by default. Community-1 uses VBx clustering, so the relative minimum
cluster size from Yamaguchi applies only to the 3.1 pipeline (`--model
pyannote/speaker-diarization-3.1`); both are in the pilot.

Two practical fixes: pyannote's reader trusts the Opus header duration,
which can exceed the decoded length by a few seconds and crash the last
window, so `diarize.py` decodes with ffmpeg and passes a waveform; and the
decode of the next episode runs on a CPU thread while the GPU works on the
current one.

SPoRC (Litterer, Jurgens & Card, ACL 2025) diarized 220,000 podcast hours
with stock pyannote 2.1 at ~22,000 GPU-hours, which is what this avoids.

## Files

| file | does |
|---|---|
| `archive_download.py` | re-fetches the audio of already-transcribed episodes into `data/audio_archive/<show_id>/<episode_id>.opus` (16 kHz mono Opus 24 kb/s, ~11 MB/h); resumable; permanent failures in `data/output/archive_failures.csv` |
| `run_archive_after_ladder.sh` | waits for the transcription ladder to stop, then runs the archive download (shared origins, shared CPU package) |
| `accelerated_pipeline.py` | patches a pretrained pyannote pipeline: stride, batch size, optional dominant-speaker embedding, relative minimum cluster size |
| `diarize.py` | diarizes archived episodes under the start-temperature gate; writes `data/diarization/<show_id>/<episode_id>[.variant].json` with turns, per-speaker centroids and the Whisper segments labelled by speaker |
| `profile_stages.py`, `profile_run.sh` | per-stage timing of one episode for a configuration; the shell wrapper pauses the archive encoders and always resumes them |
| `run_pilot.sh` | the pilot: every variant on `data/output/diarization/pilot_episodes.txt`, stock on the first 40, then the comparison tables and host identification |
| `identify_hosts.py` | per show, clusters speaker centroids across episodes; hosts are the recurring voices; checks against self-introductions by name; writes `data/output/diarization/episode_speakers.csv`, `show_hosts.csv`, and optionally host-only text under `data/transcripts_host/` |
| `pilot_eval.py` | picks the pilot sample and hand-label sheets, compares variants (DER vs stock, speed), scores host labels against hand labels; writes `results/diarization_pilot.md` |

## Order of operations

1. Audio. The ladder now runs with `ARCHIVE=1`, so episodes transcribed from
   26 Oct 2026 on are archived as they finish. For the ~65,000 earlier
   episodes, `run_archive_after_ladder.sh` starts the re-download when the
   ladder stops (download is fast; the Opus encode is the slow part, ~27 s
   per episode per core).
2. Token. `diarize.py` needs a Hugging Face token with the licences accepted
   for `pyannote/segmentation-3.0` and `pyannote/speaker-diarization-community-1`
   (or `-3.1`). Export it as `HF_TOKEN`.
3. Pilot, after the GPU is free:
   ```
   ../whisperx-env/bin/python step4_diarize/pilot_eval.py sample --per-show 2 --hand-label 20
   setsid nohup bash step4_diarize/run_pilot.sh > step4_diarize/logs/pilot.log 2>&1 &
   ```
   (launched 7 Nov 2026 21:16 UTC on 404 episodes from 202 shows.)
   Fill the hand-label sheets (H host, G guest/other, A ad or clip, X unsure),
   then `identify_hosts.py --variant s3act` and `pilot_eval.py score --variant s3act`.
4. Corpus run with the variant the pilot supports (default is community-1,
   stride 3, active-slot embedding, fp16): `diarize.py` with no filters, then
   `identify_hosts.py --write-host-text`.

## Output format

`data/diarization/<show_id>/<episode_id>.json`

```
turns      [{start, end, speaker}]         pyannote turns, overlaps allowed
speakers   {SPEAKER_00: {duration, share, centroid[256]}}
segments   Whisper segments + speaker (from the exclusive diarization) + purity
provenance model, window, step, batch, embed_mode, clustering, mcs, wall, realtime factor
```

`purity` is the share of a Whisper segment covered by its winning speaker; the
host-only text uses segments with purity ≥ 0.8.

## Known limits

- Stride 3 under-counts speakers when a voice has few embeddings (short
  interjections, many-speaker panels). The relative minimum cluster size
  recovers most of that; the pilot quantifies the rest for our audio.
- `dominant` embedding mode discards the minority speaker in windows with two
  active voices and relies on neighbouring windows to recover them. It buys only
  ~10% over `active` here and is off by default.
- Host identification by recurrence needs several diarized episodes per show;
  shows below `--min-episodes` are flagged `low_confidence` and fall back to
  the most recurrent or dominant voice.
- About 1% of episodes have no retrievable audio (HTTP 403/404); they stay
  text-only and are listed in `archive_failures.csv`.
