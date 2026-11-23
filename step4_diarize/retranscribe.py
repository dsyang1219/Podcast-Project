#!/usr/bin/env python3
"""Re-transcribe episodes whose archived audio differs from the file that was
transcribed originally (dynamic ad insertion), so that Whisper timestamps line
up with the audio the diarization ran on.

Same engine and settings as step3_transcribe/transcribe.py (faster-whisper
large-v3-turbo, batched, fp16). Output goes to data/transcripts_v2/<show_id>/
<episode_id>.json (+ .txt) with the original schema plus "source": "archive".
The original transcripts are not touched; diarize.py, identify_hosts.py and
the labelling kit prefer a v2 transcript when one exists.

Targets:
  --episodes-file F        show_id,episode_id lines
  --from-diarization       every diarized episode whose provenance shows
                           |audio duration - transcript duration| > --tolerance
  (either way, episodes that already have a v2 transcript are skipped)

    HF_TOKEN not needed. Do not run alongside the diarization run except for a
    handful of episodes: both are GPU jobs.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = PROJECT_ROOT / "data/audio_archive"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data/transcripts"
V2_DIR = PROJECT_ROOT / "data/transcripts_v2"
DIAR_DIR = PROJECT_ROOT / "data/diarization"
LOG_PATH = PROJECT_ROOT / "step4_diarize/logs/retranscribe.log"
MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
AUDIO_SUFFIXES = (".opus", ".mp3", ".m4a", ".wav", ".ogg", ".aac", ".flac")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def find_audio(show_id: str, episode_id: str) -> Path | None:
    for suf in AUDIO_SUFFIXES:
        p = ARCHIVE_DIR / show_id / f"{episode_id}{suf}"
        if p.exists():
            return p
    return None


def v2_path(show_id: str, episode_id: str) -> Path:
    return V2_DIR / show_id / f"{episode_id}.json"


def targets_from_diarization(tolerance: float) -> list[tuple[str, str]]:
    out = []
    for f in glob.glob(str(DIAR_DIR / "*" / "*.json")):
        if os.path.basename(f).count(".") > 1:
            continue  # pilot variants
        try:
            r = json.load(open(f))
        except json.JSONDecodeError:
            continue
        td = r.get("provenance", {}).get("transcript_duration")
        if td is None:
            continue
        if abs(float(r["duration"]) - float(td)) > tolerance:
            out.append((r["show_id"], r["episode_id"]))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes-file")
    ap.add_argument("--from-diarization", action="store_true")
    ap.add_argument("--tolerance", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--language", default="en")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", default=None, help="k/n: process every n-th target starting at k (parallel workers)")
    ap.add_argument("--priority-shows", default=None,
                    help="file of show_ids to process first (e.g. the fixed-host shows the analysis needs)")
    ap.add_argument("--priority-episodes", default=None,
                    help="file of show_id,episode_id lines to process before everything else (e.g. the sandbox)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[str, str]] = []
    if args.episodes_file:
        for line in Path(args.episodes_file).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                s, e = line.strip().split(",")[:2]
                pairs.append((s.strip(), e.strip()))
    if args.from_diarization:
        pairs += targets_from_diarization(args.tolerance)
    pairs = [(s, e) for s, e in dict.fromkeys(pairs) if not v2_path(s, e).exists() and find_audio(s, e)]
    if args.priority_shows:
        first = {l.strip() for l in Path(args.priority_shows).read_text().splitlines() if l.strip()}
        pairs = [p for p in pairs if p[0] in first] + [p for p in pairs if p[0] not in first]
    if args.priority_episodes:
        top = {tuple(l.strip().split(",")[:2]) for l in Path(args.priority_episodes).read_text().splitlines() if l.strip()}
        pairs = [p for p in pairs if p in top] + [p for p in pairs if p not in top]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        pairs = pairs[k::n]
    if args.limit:
        pairs = pairs[: args.limit]
    log(f"targets: {len(pairs)}" + (" (dry run)" if args.dry_run else ""))
    if args.dry_run or not pairs:
        return

    from faster_whisper import WhisperModel, BatchedInferencePipeline
    log(f"loading {MODEL} ({args.compute_type})")
    pipeline = BatchedInferencePipeline(model=WhisperModel(MODEL, device="cuda", compute_type=args.compute_type))

    done = failed = 0
    audio_s = wall_s = 0.0
    for i, (show_id, episode_id) in enumerate(pairs, 1):
        if v2_path(show_id, episode_id).exists():
            continue  # another worker got there first
        audio = find_audio(show_id, episode_id)
        orig = sorted((TRANSCRIPTS_DIR / show_id).glob(f"{episode_id}_*.json"))
        title = json.load(open(orig[0])).get("episode_title") if orig else None
        t0 = time.monotonic()
        try:
            segments, info = pipeline.transcribe(str(audio), batch_size=args.batch_size, language=args.language)
            segs = [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()} for s in segments]
        except Exception as e:
            failed += 1
            log(f"  FAIL {episode_id} ({show_id}): {type(e).__name__}: {str(e)[:160]}")
            continue
        wall = time.monotonic() - t0
        sha = hashlib.sha256(audio.read_bytes()).hexdigest()
        rec = {"episode_id": episode_id, "show_id": show_id, "episode_title": title,
               "engine": "whisper", "model": MODEL, "language": info.language,
               "duration": float(info.duration), "audio_sha256": sha, "segments": segs,
               "source": "archive", "audio": str(audio.relative_to(PROJECT_ROOT)),
               "original_transcript": str(orig[0].relative_to(PROJECT_ROOT)) if orig else None,
               "realtime_factor": round(info.duration / wall, 1) if wall else None}
        out = v2_path(show_id, episode_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.part")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        tmp.replace(out)
        out.with_suffix(".txt").write_text("\n".join(s["text"] for s in segs) + "\n")
        done += 1
        audio_s += info.duration
        wall_s += wall
        log(f"  Done {episode_id} ({show_id}) {info.duration:.0f}s rtf={info.duration / wall:.0f}x")
        if i % 50 == 0:
            log(f"progress {i}/{len(pairs)} done={done} failed={failed} mean rtf={audio_s / max(1e-6, wall_s):.0f}x")
    log(f"finished: done={done} failed={failed} audio={audio_s / 3600:.1f}h wall={wall_s / 3600:.2f}h")


if __name__ == "__main__":
    main()
