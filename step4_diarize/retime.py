#!/usr/bin/env python3
"""Re-time an original transcript onto the archived audio without re-transcribing.

The archived audio differs from the originally transcribed file only by the
ads spliced in per download; the spoken content is identical. So the original
text is right and only its timestamps are shifted. This script:

  1. runs a small Whisper model (tiny.en) on the archived audio with word
     timestamps, which is 5-10x faster than the large-v3-turbo re-transcription;
  2. aligns the original transcript's words to the small model's words
     (sequence alignment on normalised tokens, anchored on exact matches);
  3. gives every original segment the new start/end of its first/last matched
     word. Segments with too few matched words (old ads that are not in the
     new audio) keep their text but get `matched: false` and null times.

Output: data/transcripts_v2/<show_id>/<episode_id>.json with the original
text and schema, "source": "retimed". Validate against episodes that were
fully re-transcribed with --validate (compares segment starts).

    ../whisperx-env/bin/python step4_diarize/retime.py --validate 50
    ../whisperx-env/bin/python step4_diarize/retime.py --from-diarization
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = PROJECT_ROOT / "data/audio_archive"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data/transcripts"
V2_DIR = PROJECT_ROOT / "data/transcripts_v2"
DIAR_DIR = PROJECT_ROOT / "data/diarization"
LOG_PATH = PROJECT_ROOT / "step4_diarize/logs/retime.log"
AUDIO_SUFFIXES = (".opus", ".mp3", ".m4a", ".wav", ".ogg", ".aac", ".flac")
SMALL_MODEL = "tiny.en"
_WORD = re.compile(r"[a-z0-9']+")


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


def norm_tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("’", "'"))


# --------------------------------------------------------------------------- core

def retime_segments(old_segments: list[dict], new_words: list[tuple[str, float, float]],
                    min_match: float = 0.5) -> tuple[list[dict], dict]:
    """Map each old segment onto new word times.

    old_segments: [{start, end, text}]; new_words: [(token, start, end)].
    Returns new segments (same text, new times, matched flag) and stats.
    """
    old_tok, old_seg_of = [], []
    for i, s in enumerate(old_segments):
        for t in norm_tokens(s["text"]):
            old_tok.append(t)
            old_seg_of.append(i)
    new_tok = [w[0] for w in new_words]
    sm = difflib.SequenceMatcher(None, old_tok, new_tok, autojunk=False)
    old_to_new = np.full(len(old_tok), -1, dtype=np.int64)
    for a, b, n in sm.get_matching_blocks():
        if n >= 2:  # require at least two consecutive matching words as an anchor
            old_to_new[a:a + n] = np.arange(b, b + n)

    out, n_matched_words = [], 0
    per_seg_hits = [[] for _ in old_segments]
    for k, (seg_i, j) in enumerate(zip(old_seg_of, old_to_new)):
        if j >= 0:
            per_seg_hits[seg_i].append(j)
            n_matched_words += 1
    for i, s in enumerate(old_segments):
        hits = per_seg_hits[i]
        n_words = sum(1 for x in old_seg_of if x == i)
        if n_words and len(hits) / n_words >= min_match:
            start = float(new_words[min(hits)][1])
            end = float(new_words[max(hits)][2])
            out.append({"start": round(start, 3), "end": round(max(end, start + 0.1), 3), "text": s["text"],
                        "matched": True, "orig_start": s["start"], "orig_end": s["end"]})
        else:
            out.append({"start": None, "end": None, "text": s["text"], "matched": False,
                        "orig_start": s["start"], "orig_end": s["end"]})
    # monotonicity guard: drop matches that go backwards in time (spurious anchors)
    last = -1.0
    for o in out:
        if o["matched"]:
            if o["start"] < last - 1.0:
                o.update({"start": None, "end": None, "matched": False})
            else:
                last = o["end"]
    stats = {"n_segments": len(out), "n_matched": sum(o["matched"] for o in out),
             "word_match_rate": n_matched_words / max(1, len(old_tok))}
    return out, stats


def small_model(compute_type: str = "float16"):
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    return BatchedInferencePipeline(model=WhisperModel(SMALL_MODEL, device="cuda", compute_type=compute_type))


def new_words_for(pipeline, audio: Path, batch_size: int) -> tuple[list[tuple[str, float, float]], float]:
    segments, info = pipeline.transcribe(str(audio), batch_size=batch_size, language="en", word_timestamps=True)
    words = []
    for s in segments:
        for w in (s.words or []):
            for t in norm_tokens(w.word):
                words.append((t, float(w.start), float(w.end)))
    return words, float(info.duration)


def load_original(show_id: str, episode_id: str) -> tuple[dict, Path] | tuple[None, None]:
    hits = sorted((TRANSCRIPTS_DIR / show_id).glob(f"{episode_id}_*.json"))
    if not hits:
        return None, None
    return json.loads(hits[0].read_text()), hits[0]


def mismatched_targets(tolerance: float) -> list[tuple[str, str]]:
    out = []
    for f in glob.glob(str(DIAR_DIR / "*" / "*.json")):
        if os.path.basename(f).count(".") > 1:
            continue
        try:
            r = json.load(open(f))
        except json.JSONDecodeError:
            continue
        td = r.get("provenance", {}).get("transcript_duration")
        if td is not None and abs(float(r["duration"]) - float(td)) > tolerance:
            out.append((r["show_id"], r["episode_id"]))
    return sorted(set(out))


# --------------------------------------------------------------------------- modes

def run_retime(pairs, args) -> None:
    pipeline = small_model(args.compute_type)
    done = failed = 0
    audio_s = wall_s = 0.0
    for i, (show_id, episode_id) in enumerate(pairs, 1):
        out = V2_DIR / show_id / f"{episode_id}.json"
        if out.exists() and not args.overwrite:
            continue
        orig, opath = load_original(show_id, episode_id)
        audio = find_audio(show_id, episode_id)
        if orig is None or audio is None:
            continue
        t0 = time.monotonic()
        try:
            words, duration = new_words_for(pipeline, audio, args.batch_size)
            segs, st = retime_segments(orig["segments"], words)
        except Exception as e:
            failed += 1
            log(f"  FAIL {episode_id} ({show_id}): {type(e).__name__}: {str(e)[:160]}")
            continue
        wall = time.monotonic() - t0
        rec = {**{k: v for k, v in orig.items() if k != "segments"}, "duration": duration,
               "segments": segs, "source": "retimed", "retime_model": SMALL_MODEL,
               "audio": str(audio.relative_to(PROJECT_ROOT)), "original_transcript": str(opath.relative_to(PROJECT_ROOT)),
               "retime_stats": st, "realtime_factor": round(duration / wall, 1) if wall else None}
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.part")
        tmp.write_text(json.dumps(rec, ensure_ascii=False))
        tmp.replace(out)
        done += 1
        audio_s += duration
        wall_s += wall
        log(f"  Done {episode_id} ({show_id}) {duration:.0f}s rtf={duration / wall:.0f}x matched {st['n_matched']}/{st['n_segments']} segs, words {st['word_match_rate']:.2f}")
        if i % 50 == 0:
            log(f"progress {i}/{len(pairs)} done={done} failed={failed} mean rtf={audio_s / max(1e-6, wall_s):.0f}x")
    log(f"finished: done={done} failed={failed} audio={audio_s / 3600:.1f}h wall={wall_s / 3600:.2f}h "
        f"mean rtf={audio_s / max(1e-6, wall_s):.0f}x")


def run_validate(n: int, args) -> None:
    """Compare re-timed segment starts with the fully re-transcribed (archive)
    transcripts for n episodes that have one."""
    truth = sorted(p for p in V2_DIR.glob("*/*.json") if json.loads(p.read_text()).get("source") == "archive")
    truth = truth[:n]
    pipeline = small_model(args.compute_type)
    errs, matched_rates, rtfs = [], [], []
    for p in truth:
        show_id, episode_id = p.parent.name, p.stem
        orig, _ = load_original(show_id, episode_id)
        audio = find_audio(show_id, episode_id)
        if orig is None or audio is None:
            continue
        t0 = time.monotonic()
        words, duration = new_words_for(pipeline, audio, args.batch_size)
        segs, st = retime_segments(orig["segments"], words)
        rtfs.append(duration / (time.monotonic() - t0))
        matched_rates.append(st["n_matched"] / max(1, st["n_segments"]))
        # truth: for each re-timed segment, the archive segment with the most word overlap -> compare starts
        tsegs = json.loads(p.read_text())["segments"]
        ttok = [set(norm_tokens(s["text"])) for s in tsegs]
        tstart = np.array([s["start"] for s in tsegs])
        for s in segs:
            if not s["matched"]:
                continue
            tok = set(norm_tokens(s["text"]))
            if len(tok) < 4:
                continue
            # candidates within 90 s of the re-timed start
            idx = np.where(np.abs(tstart - s["start"]) < 90)[0]
            if not len(idx):
                errs.append(90.0)
                continue
            best = max(idx, key=lambda k: len(tok & ttok[k]) / max(1, len(tok)))
            if len(tok & ttok[best]) / len(tok) < 0.5:
                continue
            errs.append(abs(tsegs[best]["start"] - s["start"]))
        log(f"  {episode_id}: rtf {rtfs[-1]:.0f}x, segments matched {matched_rates[-1]:.0%}")
    e = np.array(errs)
    log(f"VALIDATION on {len(rtfs)} episodes: mean rtf {np.mean(rtfs):.0f}x | segments re-timed {np.mean(matched_rates):.1%} | "
        f"start error vs full re-transcription: median {np.median(e):.2f}s, 90th {np.percentile(e, 90):.2f}s, "
        f"within 1 s {np.mean(e <= 1):.1%}, within 2 s {np.mean(e <= 2):.1%}, >5 s {np.mean(e > 5):.1%} (n={len(e)} segments)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", type=int, metavar="N", help="validate against N fully re-transcribed episodes")
    ap.add_argument("--from-diarization", action="store_true")
    ap.add_argument("--episodes-file")
    ap.add_argument("--tolerance", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.validate:
        run_validate(args.validate, args)
        return
    pairs = []
    if args.episodes_file:
        for line in Path(args.episodes_file).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                s, e = line.strip().split(",")[:2]
                pairs.append((s.strip(), e.strip()))
    if args.from_diarization:
        pairs += mismatched_targets(args.tolerance)
    pairs = [(s, e) for s, e in dict.fromkeys(pairs) if not (V2_DIR / s / f"{e}.json").exists()]
    if args.limit:
        pairs = pairs[: args.limit]
    log(f"targets: {len(pairs)}")
    if pairs:
        run_retime(pairs, args)


if __name__ == "__main__":
    main()
