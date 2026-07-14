#!/usr/bin/env python3
"""
GPU transcription worker for the H25 sample manifest — consumer half of a
producer/consumer pair with download_audio.py.

Watches data/audio_queue/ using the same atomic .meta -> .meta.claimed
rename-claim protocol as ~/Podcast-Project/transcribe_worker.py, but:
  - faster-whisper BatchedInferencePipeline, not whisperx + diarization
  - output keyed by show_id/episode_id (json + txt), not video_id/title (vtt)
  - resumable via audio_sha256 comparison against an existing transcript,
    not just presence of a processed-log line

Run `--smoke-test` once before any bulk run (see README / plan) to confirm
this box's ctranslate2 build actually executes on the GB10 GPU rather than
silently falling back to CPU.
"""
import argparse
import gc
import glob
import json
import os
import re
import subprocess
import sys
import time
from csv import DictWriter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUEUE_DIR = PROJECT_ROOT / "data/audio_queue"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data/transcripts"
STATUS_CSV = PROJECT_ROOT / "data/output/transcribe_status.csv"
STATUS_FIELDS = ["episode_id", "show_id", "engine", "model", "language",
                  "n_segments", "n_words", "realtime_factor", "wall_seconds"]

NO_META_POLL_SECS = 2.0


class WhisperEngine:
    name = "whisper"

    def __init__(self, model: str, compute_type: str, batch_size: int, language: str | None):
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        self.batch_size = batch_size
        self.language = language  # None => auto-detect
        base = WhisperModel(model, device="cuda", compute_type=compute_type)
        self.pipeline = BatchedInferencePipeline(model=base)

    def transcribe(self, audio_path: str):
        segments, info = self.pipeline.transcribe(
            audio_path, batch_size=self.batch_size, language=self.language)
        segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
        return segs, info.language, info.duration


class NemoEngine:
    name = "nemo"

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "NeMo evaluation not yet wired in. Per the plan: evaluate "
            "Parakeet-TDT / Canary on ~20 episodes before implementing this "
            "path — it's a separate toolkit install, not a drop-in swap."
        )

    def transcribe(self, audio_path: str):
        raise NotImplementedError


ENGINES = {"whisper": WhisperEngine, "nemo": NemoEngine}


def sanitize_title(title: str, max_len: int = 80) -> str:
    t = re.sub(r'[/\\:*?"<>|]', "_", title)
    t = re.sub(r"\s+", " ", t).strip()
    t = t[:max_len].rstrip()
    return t or "untitled"


def find_existing_json(show_id: str, episode_id: str) -> Path | None:
    """episode_id is the stable identifier; title (and thus filename) can
    change if the manifest is regenerated, so look up by episode_id prefix
    rather than assuming an exact filename."""
    show_dir = TRANSCRIPTS_DIR / show_id
    if not show_dir.exists():
        return None
    matches = sorted(show_dir.glob(f"{episode_id}_*.json"))
    return matches[0] if matches else None


def already_done(show_id: str, episode_id: str, audio_sha256: str) -> bool:
    json_path = find_existing_json(show_id, episode_id)
    if json_path is None:
        return False
    try:
        data = json.loads(json_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("audio_sha256") == audio_sha256


def write_status_row(row: dict) -> None:
    STATUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not STATUS_CSV.exists()
    with open(STATUS_CSV, "a", newline="") as f:
        w = DictWriter(f, fieldnames=STATUS_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def process(engine, audio_path: str, show_id: str, episode_id: str,
            audio_sha256: str, title: str, model_name: str) -> tuple[float, float]:
    t0 = time.monotonic()
    segments, language, duration = engine.transcribe(audio_path)
    wall = time.monotonic() - t0

    n_words = sum(len(s["text"].split()) for s in segments)
    realtime_factor = (duration / wall) if wall > 0 else float("nan")

    out_dir = TRANSCRIPTS_DIR / show_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # drop any stale output under an older filename (e.g. title changed
    # between manifest regenerations) so we don't accumulate orphans
    for stale in out_dir.glob(f"{episode_id}_*"):
        stale.unlink(missing_ok=True)

    stem = f"{episode_id}_{sanitize_title(title)}"
    record = {
        "episode_id": episode_id,
        "show_id": show_id,
        "episode_title": title,
        "engine": engine.name,
        "model": model_name,
        "language": language,
        "duration": duration,
        "audio_sha256": audio_sha256,
        "segments": segments,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2))
    (out_dir / f"{stem}.txt").write_text(
        " ".join(s["text"] for s in segments).strip() + "\n")

    write_status_row({
        "episode_id": episode_id, "show_id": show_id, "engine": engine.name,
        "model": model_name, "language": language,
        "n_segments": len(segments), "n_words": n_words,
        "realtime_factor": round(realtime_factor, 2), "wall_seconds": round(wall, 1),
    })
    return realtime_factor, wall


def run(engine, model_name: str) -> None:
    print("Watching queue...", flush=True)
    while True:
        metas = sorted(glob.glob(str(QUEUE_DIR / "*.meta")))

        claimed = None
        for m in metas:
            c = m + ".claimed"
            try:
                os.rename(m, c)
                claimed = c
                break
            except (FileNotFoundError, OSError):
                continue

        if claimed:
            show_id, episode_id, audio_sha256, title = (
                Path(claimed).read_text().strip().split("|||", 3))
            stem = claimed[: -len(".meta.claimed")]
            audio_files = [f for f in glob.glob(stem + ".*")
                           if not f.endswith(".meta") and not f.endswith(".claimed")]

            if not audio_files:
                print(f"  WARNING: no audio for {episode_id} — dropping stale claim", flush=True)
                os.remove(claimed)
                continue

            audio_path = audio_files[0]
            try:
                if already_done(show_id, episode_id, audio_sha256):
                    print(f"  Already transcribed: {episode_id} — skipping", flush=True)
                else:
                    rtf, wall = process(engine, audio_path, show_id, episode_id,
                                         audio_sha256, title, model_name)
                    print(f"  Done: {episode_id} ({show_id}) "
                          f"rtf={rtf:.1f}x wall={wall:.1f}s", flush=True)
            except Exception as e:
                print(f"  ERROR processing {episode_id}: {e}", flush=True)
            finally:
                Path(audio_path).unlink(missing_ok=True)
                if os.path.exists(claimed):
                    os.remove(claimed)
                gc.collect()
            continue

        if (QUEUE_DIR / "STOP").exists():
            if not glob.glob(str(QUEUE_DIR / "*.claimed")):
                print("Queue empty and STOP received. Exiting.", flush=True)
                break

        time.sleep(NO_META_POLL_SECS)


def smoke_test(engine_name: str, model_name: str, compute_type: str,
               batch_size: int, clip_path: Path | None) -> None:
    synthesized = clip_path is None
    if synthesized:
        clip_path = PROJECT_ROOT / "transcription" / "_smoke_test_tone.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
             "-ar", "16000", "-ac", "1", str(clip_path)],
            check=True, capture_output=True,
        )
        print(f"No clip given — synthesized a 5s test tone at {clip_path}.")
        print("This only proves the GPU path executes; pass a real ~5min "
              "speech clip (--smoke-test /path/to/clip.mp3) to also sanity-"
              "check that decoded text is real speech, not gibberish.")

    print(f"Loading {model_name} on cuda ({compute_type})...", flush=True)
    t0 = time.monotonic()
    engine = ENGINES[engine_name](model_name, compute_type, batch_size, "en")
    print(f"Model loaded in {time.monotonic() - t0:.1f}s", flush=True)

    t0 = time.monotonic()
    segments, language, duration = engine.transcribe(str(clip_path))
    wall = time.monotonic() - t0
    text = " ".join(s["text"] for s in segments)

    rtf = duration / wall if wall > 0 else float("nan")
    print(f"Transcribed {duration:.1f}s of audio in {wall:.1f}s ({rtf:.1f}x realtime)")
    print(f"Detected language: {language}")
    print(f"Decoded text sample: {text[:200]!r}")

    import ctranslate2
    n_devices = ctranslate2.get_cuda_device_count()
    print(f"CUDA devices visible to ctranslate2: {n_devices}")

    if n_devices == 0 or (not synthesized and wall > duration * 2):
        print("WARNING: this looks like a silent CPU fallback (no CUDA "
              "devices visible, or running slower than 0.5x realtime). "
              "Abort the bulk run and investigate before proceeding.",
              file=sys.stderr)
        sys.exit(1)

    print("Smoke test OK.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=list(ENGINES), default="whisper")
    ap.add_argument("--model", default="large-v3",
                    help="large-v3 (default) | distil-large-v3")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--compute-type", default="float16",
                    help="float16 (default) | int8_float16 (fallback)")
    ap.add_argument("--language", default="en")
    ap.add_argument("--auto-lang", action="store_true",
                    help="auto-detect language instead of forcing --language")
    ap.add_argument("--smoke-test", nargs="?", const="", default=None,
                    metavar="CLIP_PATH",
                    help="transcribe one clip and exit, confirming CUDA "
                         "execution before a bulk run. No path -> synthesized tone.")
    args = ap.parse_args()

    language = None if args.auto_lang else args.language

    if args.smoke_test is not None:
        clip = Path(args.smoke_test) if args.smoke_test else None
        smoke_test(args.engine, args.model, args.compute_type, args.batch_size, clip)
        return

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.engine} model {args.model} ({args.compute_type})...", flush=True)
    engine = ENGINES[args.engine](args.model, args.compute_type, args.batch_size, language)
    print("Model loaded. Watching queue...", flush=True)

    run(engine, args.model)


if __name__ == "__main__":
    main()
