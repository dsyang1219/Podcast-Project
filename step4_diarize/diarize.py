#!/usr/bin/env python3
"""Diarize archived episodes and attach speaker labels to the Whisper segments.

Reads   data/audio_archive/<show_id>/<episode_id>.opus   (see archive_download.py)
        data/transcripts/<show_id>/<episode_id>_*.json   (Whisper segments)
Writes  data/diarization/<show_id>/<episode_id>.json  with
        turns      speaker turns (start, end, speaker), overlaps allowed
        speakers   per speaker: total duration, share, 256-d centroid embedding
        segments   the Whisper segments with the speaker that covers most of
                   each one and the covered fraction ("purity")
        provenance pipeline configuration, wall time, realtime factor

Default configuration is the accelerated pipeline (stride 3 s, batch 32,
relative minimum cluster size); --stock runs pyannote as shipped for the
pilot comparison. Runs under the same start-temperature gate as the
transcriber so the GPU is never driven continuously.

    HF_TOKEN=hf_... ../whisperx-env/bin/python step4_diarize/diarize.py --limit 5
    HF_TOKEN=hf_... ../whisperx-env/bin/python step4_diarize/diarize.py --episodes-file pilot.txt --stock --suffix stock

Do not run while the transcription ladder is using the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from accelerated_pipeline import AccelConfig, build_pipeline, describe  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = PROJECT_ROOT / "data/audio_archive"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data/transcripts"
OUT_DIR = PROJECT_ROOT / "data/diarization"
LOG_PATH = PROJECT_ROOT / "step4_diarize/logs/diarize.log"
AUDIO_SUFFIXES = (".opus", ".mp3", ".m4a", ".wav", ".ogg", ".aac", ".flac")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------- thermal gate

def gpu_temp_f() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        return float(out) * 9 / 5 + 32
    except Exception:
        return None


class StartGate:
    """Delay each episode until the die is within `margin` F of the box's
    current idle floor. Port of step3_transcribe/transcribe.py's adaptive gate:
    a sampler thread reads the GPU temperature every 5 s, the floor is the 10th
    percentile of the last 30 minutes (so it tracks room temperature and other
    tenants' load, and one cold sample cannot pin the gate out of reach), and
    every wait is bounded by `max_wait`."""

    SAMPLE_EVERY_S = 5.0
    FLOOR_WINDOW_S = 1800.0
    FLOOR_PCTL = 0.10
    FLOOR_MIN_SAMPLES = 12

    def __init__(self, margin: float = 2.0, max_wait: float = 300.0, below_f: float | None = None):
        import collections, threading
        self.margin, self.max_wait, self.below_f = margin, max_wait, below_f
        self.buf = collections.deque(maxlen=int(self.FLOOR_WINDOW_S / self.SAMPLE_EVERY_S))
        self.latest = gpu_temp_f()
        self.waits: list[float] = []
        self.timeouts = 0
        self._stop = threading.Event()
        threading.Thread(target=self._sample, daemon=True).start()

    def _sample(self) -> None:
        while not self._stop.wait(self.SAMPLE_EVERY_S):
            t = gpu_temp_f()
            if t is not None:
                self.latest = t
                self.buf.append(t)

    def floor(self) -> float | None:
        if not self.buf:
            return self.latest
        v = sorted(self.buf)
        if len(v) < self.FLOOR_MIN_SAMPLES:
            return v[0]
        return v[int(self.FLOOR_PCTL * (len(v) - 1))]

    def wait(self) -> float:
        t0 = time.monotonic()
        while True:
            t, f = self.latest, self.floor()
            limit = self.below_f if self.below_f is not None else (None if f is None else f + self.margin)
            if t is None or limit is None or t <= limit:
                break
            if time.monotonic() - t0 >= self.max_wait:
                self.timeouts += 1
                break
            time.sleep(5.0)
        waited = time.monotonic() - t0
        if waited > 0:
            self.waits.append(waited)
        return waited

    def summary(self) -> str:
        f = self.floor()
        if self.below_f is not None:
            desc = f"gate={self.below_f:.0f}F (fixed) now={self.latest:.0f}F"
        else:
            desc = f"gate={f + self.margin:.0f}F (floor {f:.0f}F +{self.margin:.0f}) now={self.latest:.0f}F" if f is not None else "gate: no reading"
        if not self.waits:
            return desc
        recent = self.waits[-20:]
        return f"{desc} mean_wait={sum(recent) / len(recent):.0f}s timeouts={self.timeouts}"


# --------------------------------------------------------------------------- IO

def find_audio(show_id: str, episode_id: str) -> Path | None:
    d = ARCHIVE_DIR / show_id
    for suf in AUDIO_SUFFIXES:
        p = d / f"{episode_id}{suf}"
        if p.exists():
            return p
    return None


V2_DIR = PROJECT_ROOT / "data/transcripts_v2"


def find_transcript(show_id: str, episode_id: str) -> Path | None:
    """Prefer a v2 transcript (re-transcribed from the archived audio, so its
    timestamps match) over the original one."""
    v2 = V2_DIR / show_id / f"{episode_id}.json"
    if v2.exists():
        return v2
    hits = sorted((TRANSCRIPTS_DIR / show_id).glob(f"{episode_id}_*.json"))
    return hits[0] if hits else None


def list_targets(args) -> list[tuple[str, str]]:
    """(show_id, episode_id) pairs with audio and a transcript, not yet diarized."""
    pairs: list[tuple[str, str]] = []
    if args.episodes_file:
        for line in Path(args.episodes_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                show_id, episode_id = line.split(",")[:2]
                pairs.append((show_id.strip(), episode_id.strip()))
    else:
        shows = set(args.shows.split(",")) if args.shows else None
        for show_dir in sorted(ARCHIVE_DIR.iterdir()):
            if not show_dir.is_dir() or show_dir.name.startswith("_"):
                continue
            if shows and show_dir.name not in shows:
                continue
            for p in sorted(show_dir.iterdir()):
                if p.suffix in AUDIO_SUFFIXES:
                    pairs.append((show_dir.name, p.stem))
    out = []
    for show_id, episode_id in pairs:
        if not args.overwrite and out_path(show_id, episode_id, args.suffix).exists():
            continue
        if find_audio(show_id, episode_id) and find_transcript(show_id, episode_id):
            out.append((show_id, episode_id))
    if args.limit:
        out = out[: args.limit]
    return out


def out_path(show_id: str, episode_id: str, suffix: str) -> Path:
    name = f"{episode_id}.{suffix}.json" if suffix else f"{episode_id}.json"
    return OUT_DIR / show_id / name


# --------------------------------------------------------------------------- core

def assign_segments(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Label each Whisper segment with the speaker covering most of it.

    Uses the exclusive (non-overlapping) diarization. purity = covered share of
    the segment by the winning speaker; low purity flags segments that straddle
    a speaker change, which downstream text analyses can drop or split.
    """
    if not turns:
        return [dict(s, speaker="UNK", purity=0.0) for s in segments]
    starts = np.array([t["start"] for t in turns])
    ends = np.array([t["end"] for t in turns])
    spk = [t["speaker"] for t in turns]
    out = []
    for s in segments:
        a, b = float(s["start"]), float(s["end"])
        ov = np.clip(np.minimum(ends, b) - np.maximum(starts, a), 0, None)
        if ov.sum() <= 0:
            out.append(dict(s, speaker="UNK", purity=0.0))
            continue
        tot: dict[str, float] = {}
        for k, o in zip(spk, ov):
            if o > 0:
                tot[k] = tot.get(k, 0.0) + float(o)
        best = max(tot, key=tot.get)
        out.append(dict(s, speaker=best, purity=round(tot[best] / max(1e-6, b - a), 3)))
    return out


def load_waveform(path: Path, sample_rate: int = 16000):
    """Decode with ffmpeg to a (1, n) float32 tensor at 16 kHz mono.

    Passing a waveform instead of a path keeps pyannote away from container
    metadata: Opus/Ogg headers can declare a few seconds more than the stream
    decodes to, and pyannote's reader then raises on the last window.
    """
    import torch
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(sample_rate), "-f", "f32le", "-"],
        capture_output=True, timeout=1800)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode('utf-8', 'replace')[-200:]}")
    wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return torch.from_numpy(wav)[None], sample_rate


def diarize_one(pipeline, cfg: AccelConfig, show_id: str, episode_id: str, suffix: str,
                preloaded=None) -> dict:
    audio = find_audio(show_id, episode_id)
    tpath = find_transcript(show_id, episode_id)
    transcript = json.loads(tpath.read_text())
    t0 = time.monotonic()
    waveform, sr = preloaded if preloaded is not None else load_waveform(audio)
    decode_s = time.monotonic() - t0
    result = pipeline({"waveform": waveform, "sample_rate": sr, "uri": episode_id})
    wall = time.monotonic() - t0

    # pyannote 4 returns DiarizeOutput; 3.x returns an Annotation
    if hasattr(result, "speaker_diarization"):
        ann, excl, cents = result.speaker_diarization, result.exclusive_speaker_diarization, result.speaker_embeddings
    else:
        ann, excl, cents = result, result, None

    labels = list(ann.labels())
    turns = [{"start": round(seg.start, 3), "end": round(seg.end, 3), "speaker": lab}
             for seg, _, lab in ann.itertracks(yield_label=True)]
    excl_turns = [{"start": round(seg.start, 3), "end": round(seg.end, 3), "speaker": lab}
                  for seg, _, lab in excl.itertracks(yield_label=True)]
    total = sum(ann.label_duration(l) for l in labels) or 1.0
    speakers = {}
    for i, lab in enumerate(labels):
        d = float(ann.label_duration(lab))
        entry = {"duration": round(d, 2), "share": round(d / total, 4)}
        if cents is not None and i < len(cents) and not np.any(np.isnan(cents[i])):
            entry["centroid"] = [round(float(x), 5) for x in cents[i]]
        speakers[lab] = entry

    duration = float(waveform.shape[1] / sr)
    tdur = transcript.get("duration")
    aligned = tdur is not None and abs(duration - float(tdur)) <= 0.5
    return {
        "show_id": show_id,
        "episode_id": episode_id,
        "episode_title": transcript.get("episode_title"),
        "duration": duration,
        "n_speakers": len(labels),
        "turns": turns,
        "exclusive_turns": excl_turns,
        "speakers": speakers,
        "segments": assign_segments(transcript["segments"], excl_turns),
        "provenance": {**describe(pipeline, cfg), "audio": str(audio.relative_to(PROJECT_ROOT)),
                       "transcript": str(tpath.relative_to(PROJECT_ROOT)),
                       "wall_seconds": round(wall, 1), "decode_seconds": round(decode_s, 1),
                       "transcript_duration": tdur,
                       "transcript_source": transcript.get("source", "original"),
                       "aligned": bool(aligned),
                       "realtime_factor": round(duration / wall, 1) if wall > 0 and duration else None,
                       "variant": suffix or "default"},
    }


def reassign_from_v2(suffix: str = "") -> None:
    """For every diarized episode that now has a v2 (audio-aligned) transcript
    but was labelled from the original one, recompute the segment speaker
    labels from the stored turns. No GPU, no re-diarization."""
    n = 0
    for v2 in sorted(V2_DIR.glob("*/*.json")):
        show_id, episode_id = v2.parent.name, v2.stem
        p = out_path(show_id, episode_id, suffix)
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        if rec.get("provenance", {}).get("transcript_source") == "archive":
            continue
        transcript = json.loads(v2.read_text())
        turns = rec.get("exclusive_turns") or rec["turns"]
        rec["segments"] = assign_segments(transcript["segments"], turns)
        tdur = float(transcript["duration"])
        rec["provenance"].update({"transcript": str(v2.relative_to(PROJECT_ROOT)), "transcript_duration": tdur,
                                  "transcript_source": "archive",
                                  "aligned": abs(float(rec["duration"]) - tdur) <= 0.5})
        tmp = p.with_suffix(".json.part")
        tmp.write_text(json.dumps(rec, ensure_ascii=False))
        tmp.replace(p)
        n += 1
    log(f"reassigned segment labels from v2 transcripts for {n} episodes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reassign-only", action="store_true",
                    help="no diarization: relabel segments of already-diarized episodes from v2 transcripts")
    ap.add_argument("--model", default=AccelConfig.model)
    ap.add_argument("--stock", action="store_true", help="pyannote as shipped (stride 1 s, all speakers, stock mcs)")
    ap.add_argument("--stride", type=float, default=0.3, help="segmentation step as a fraction of the 10 s window")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--embed-mode", choices=["full", "active", "dominant"], default="active")
    ap.add_argument("--fp32", action="store_true", help="disable fp16 autocast for the embedding model")
    ap.add_argument("--mcs-fraction", type=float, default=0.01, help="0 disables the relative minimum cluster size")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--shows", default=None, help="comma-separated show_ids")
    ap.add_argument("--episodes-file", default=None, help="text file of show_id,episode_id lines")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--suffix", default="", help="output name suffix for pilot variants, e.g. 'stock'")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-gate", action="store_true", help="skip the start-temperature gate")
    ap.add_argument("--gate-margin", type=float, default=2.0)
    ap.add_argument("--gate-max-wait", type=float, default=120.0)
    ap.add_argument("--gate-below-f", type=float, default=None,
                    help="fixed cap: wait until the die is at or below this many F before each episode "
                         "(the adaptive floor gate is useless for a GPU-bound run: the floor just drifts up)")
    ap.add_argument("--dry-run", action="store_true", help="list targets and exit")
    args = ap.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.reassign_only:
        reassign_from_v2(args.suffix)
        return
    targets = list_targets(args)
    log(f"targets: {len(targets)} episodes" + (" (dry run)" if args.dry_run else ""))
    if args.dry_run or not targets:
        for s, e in targets[:20]:
            print(s, e)
        return
    if not args.token:
        sys.exit("need a Hugging Face token: set HF_TOKEN or pass --token (model licences must be accepted)")

    cfg = AccelConfig.stock(args.model, args.device) if args.stock else AccelConfig(
        model=args.model, stride=args.stride, batch_size=args.batch, embed_mode=args.embed_mode,
        mcs_fraction=args.mcs_fraction or None, half=not args.fp32, device=args.device)
    log(f"config: {cfg}")
    pipeline = build_pipeline(cfg, token=args.token)
    log(f"pipeline: {describe(pipeline, cfg)}")

    gate = None if args.no_gate else StartGate(margin=args.gate_margin, max_wait=args.gate_max_wait, below_f=args.gate_below_f)
    done, failed, audio_s, wall_s = 0, 0, 0.0, 0.0
    # decode the next episode on a CPU thread while the GPU works on the current one
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1)
    futures = [pool.submit(load_waveform, find_audio(s_, e_)) for s_, e_ in targets[:1]]
    for i, (show_id, episode_id) in enumerate(targets, 1):
        if i < len(targets):
            futures.append(pool.submit(load_waveform, find_audio(*targets[i])))
        fut = futures[i - 1]
        if gate is not None:
            gate.wait()
        try:
            rec = diarize_one(pipeline, cfg, show_id, episode_id, args.suffix, preloaded=fut.result())
        except Exception as e:  # keep going; one bad file must not stop the run
            failed += 1
            log(f"  FAIL {episode_id} ({show_id}): {type(e).__name__}: {str(e)[:200]}")
            continue
        finally:
            futures[i - 1] = None
        p = out_path(show_id, episode_id, args.suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.part")
        tmp.write_text(json.dumps(rec, ensure_ascii=False))
        tmp.replace(p)
        done += 1
        audio_s += rec["duration"]
        wall_s += rec["provenance"]["wall_seconds"]
        log(f"  Done {episode_id} ({show_id}) spk={rec['n_speakers']} rtf={rec['provenance']['realtime_factor']}x "
            f"wall={rec['provenance']['wall_seconds']}s")
        if i % 25 == 0:
            log(f"progress {i}/{len(targets)} done={done} failed={failed} "
                f"mean rtf={audio_s / max(1e-6, wall_s):.0f}x | {gate.summary() if gate else 'no gate'}")
    pool.shutdown(wait=False)
    log(f"finished: done={done} failed={failed} audio={audio_s / 3600:.1f}h wall={wall_s / 3600:.2f}h "
        f"mean rtf={audio_s / max(1e-6, wall_s):.0f}x")


if __name__ == "__main__":
    main()
