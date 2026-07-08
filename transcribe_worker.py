#!/usr/bin/env python3
"""
Transcription worker — loads Whisper + diarization models once,
then watches ~/Podcast-Project/queue/ for audio files to process.
Supports multiple parallel workers via atomic .meta → .meta.claimed rename.

Queue protocol (written by youtube_pipeline.sh):
  - Audio file:  QUEUE_DIR/<video_id>.<ext>
  - Meta file:   QUEUE_DIR/<video_id>.meta  →  "video_id|||safe_title"
  - Stop signal: QUEUE_DIR/STOP  (created after all downloads finish)
"""
import whisperx
from whisperx.diarize import DiarizationPipeline
import os, sys, gc, glob, time
import torch

NO_AUDIO_WAIT_SECS = 60  # give downloader this long before declaring audio missing

QUEUE_DIR     = os.path.expanduser("~/Podcast-Project/queue")
OUTPUT_DIR    = os.path.expanduser("~/Podcast-Project/output")
PROCESSED_LOG = os.path.expanduser("~/Podcast-Project/processed.txt")
HF_TOKEN      = os.environ["HF_TOKEN"]
DEVICE        = "cuda"
BATCH_SIZE    = 128
COMPUTE_TYPE  = "float16"

os.makedirs(QUEUE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Whisper large-v3...", flush=True)
model = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE, language="en")

print("Loading diarization model...", flush=True)
diarize_model = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)

print("All models loaded. Watching queue...", flush=True)
sys.stdout.flush()

# Track how long each video_id has been claimed with no audio found
no_audio_since: dict[str, float] = {}


def is_already_processed(video_id: str) -> bool:
    try:
        with open(PROCESSED_LOG) as f:
            return video_id in f.read().splitlines()
    except FileNotFoundError:
        return False


def process(audio_path, video_id, safe_title):
    print(f"\n=== Transcribing: {safe_title} ===", flush=True)

    audio = whisperx.load_audio(audio_path)

    result = model.transcribe(audio, batch_size=BATCH_SIZE, language="en")
    language = result.get("language", "en")

    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    result.setdefault("language", language)

    from whisperx.utils import get_writer
    writer = get_writer("vtt", OUTPUT_DIR)
    writer(result, audio_path, {
        "max_line_width": None,
        "max_line_count": None,
        "highlight_words": False,
    })

    stem = os.path.splitext(os.path.basename(audio_path))[0]
    src_vtt = os.path.join(OUTPUT_DIR, stem + ".vtt")
    dst_vtt = os.path.join(OUTPUT_DIR, safe_title + ".vtt")
    if os.path.exists(src_vtt):
        os.rename(src_vtt, dst_vtt)

    os.remove(audio_path)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"=== Done: {safe_title} ===", flush=True)


while True:
    metas = sorted(glob.glob(os.path.join(QUEUE_DIR, "*.meta")))

    claimed_path = None
    meta_path = None
    for m in metas:
        c = m + ".claimed"
        try:
            os.rename(m, c)
            claimed_path = c
            meta_path = m
            break
        except (FileNotFoundError, OSError):
            continue

    if claimed_path:
        with open(claimed_path) as f:
            video_id, safe_title = f.read().strip().split("|||", 1)

        stem = os.path.splitext(meta_path)[0]
        audio_files = [f for f in glob.glob(stem + ".*")
                       if not f.endswith(".meta") and not f.endswith(".claimed")]

        if audio_files:
            try:
                process(audio_files[0], video_id, safe_title)
                os.remove(claimed_path)
                with open(PROCESSED_LOG, "a") as f:
                    f.write(video_id + "\n")
            except Exception as e:
                print(f"  ERROR processing {video_id}: {e}", flush=True)
                for af in audio_files:
                    if os.path.exists(af):
                        os.remove(af)
                if os.path.exists(claimed_path):
                    os.remove(claimed_path)
                partial_vtt = os.path.join(OUTPUT_DIR, video_id + ".vtt")
                if os.path.exists(partial_vtt):
                    os.remove(partial_vtt)
        else:
            # No audio file found yet.
            # Case 1: already processed by another worker (stale .meta from a race).
            if is_already_processed(video_id):
                print(f"  Stale .meta for already-processed {video_id} — cleaning up.", flush=True)
                os.remove(claimed_path)
                no_audio_since.pop(video_id, None)
            # Case 2: audio missing too long — download likely failed.
            elif time.monotonic() - no_audio_since.setdefault(video_id, time.monotonic()) > NO_AUDIO_WAIT_SECS:
                print(f"  WARNING: No audio for {video_id} after {NO_AUDIO_WAIT_SECS}s — skipping.", flush=True)
                os.remove(claimed_path)
                with open(PROCESSED_LOG, "a") as f:
                    f.write(video_id + "\n")
                no_audio_since.pop(video_id, None)
            # Case 3: downloader still writing — unclaim and wait.
            else:
                os.rename(claimed_path, meta_path)
                time.sleep(1)
        continue

    if os.path.exists(os.path.join(QUEUE_DIR, "STOP")):
        # Only exit if no other worker is mid-episode
        if not glob.glob(os.path.join(QUEUE_DIR, "*.claimed")):
            print("Queue empty and STOP received. Exiting.", flush=True)
            break

    time.sleep(2)
