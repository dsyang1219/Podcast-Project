#!/usr/bin/env python3
"""Re-time an original transcript onto the archived audio by probing.

Dynamic ad insertion splices different ads into each download, so the archived
audio and the originally transcribed file differ only at a few splice points
(pre-roll, one to three mid-rolls, post-roll). Between splices the content is
the same audio, shifted by a constant. So the map from old to new time is
piecewise constant and needs a handful of numbers per episode, not a full
re-transcription.

Method
  1. Transcribe short probes of the new audio (20 s every PROBE_EVERY seconds,
     denser in the first minutes) with the same large-v3-turbo model, word
     timestamps on.
  2. Locate each probe's words in the old transcript by 4-gram anchors; the
     old time of a word is interpolated inside its segment by character
     position. offset = old_time - new_time, median over the probe's words.
  3. Where consecutive probes disagree by more than JUMP seconds, bisect with
     extra probes until the splice is localised to within ~20 s of new audio.
  4. Every old segment gets new times = old times - offset of its region. Old
     segments that fall in a span removed from the new audio (old ads) keep
     their text with null times and matched=false.

Output: data/transcripts_v2/<show_id>/<episode_id>.json, "source": "probe",
original text and schema. Validate with --validate N against episodes that were
fully re-transcribed (source "archive"): start error, and agreement of the
resulting segment speaker labels.

    LD_LIBRARY_PATH=/opt/ctranslate2/lib ../whisperx-env/bin/python step4_diarize/retime_probe.py --validate 40
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = PROJECT_ROOT / "data/audio_archive"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data/transcripts"
V2_DIR = PROJECT_ROOT / "data/transcripts_v2"
DIAR_DIR = PROJECT_ROOT / "data/diarization"
LOG_PATH = PROJECT_ROOT / "step4_diarize/logs/retime_probe.log"
MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
AUDIO_SUFFIXES = (".opus", ".mp3", ".m4a", ".wav", ".ogg", ".aac", ".flac")
SR = 16000
PROBE_LEN = 30.0
PROBE_EVERY = 180.0
EARLY_EVERY = 45.0      # denser probes in the first minutes (pre-roll ads)
EARLY_UNTIL = 360.0
JUMP = 1.5              # seconds: offsets differing by more are different regions
MIN_TOKENS = 6
MIN_MATCH = 0.6
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


def load_waveform(path: Path) -> np.ndarray:
    proc = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                           "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, timeout=1800)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError("ffmpeg decode failed")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def norm_tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("’", "'"))


# --------------------------------------------------------------------------- old transcript index

class OldIndex:
    """Tokens of the old transcript with interpolated old times and a 4-gram index."""

    def __init__(self, segments: list[dict]):
        self.tok: list[str] = []
        self.time: list[float] = []          # interpolated (coarse) old time of the token
        self.anchor: list[tuple | None] = []  # ("start", t) for a segment's first token, ("end", t) for its last
        self.seg_of: list[int] = []
        for i, s in enumerate(segments):
            text = s["text"]
            toks = list(_WORD.finditer(text.lower().replace("’", "'")))
            dur = max(0.1, float(s["end"]) - float(s["start"]))
            for j, m in enumerate(toks):
                frac = m.start() / max(1, len(text))
                self.tok.append(m.group(0))
                self.time.append(float(s["start"]) + frac * dur)
                a = None
                if j == 0:
                    a = ("start", float(s["start"]))
                elif j == len(toks) - 1:
                    a = ("end", float(s["end"]))
                self.anchor.append(a)
                self.seg_of.append(i)
        self.grams: dict[tuple, list[int]] = defaultdict(list)
        for k in range(len(self.tok) - 3):
            self.grams[tuple(self.tok[k:k + 4])].append(k)

    def locate(self, words: list[tuple[str, float, float]]) -> tuple[float | None, int]:
        """words: [(token, new_start, new_end)]. Returns (offset old-new, n matched) or (None, n).

        The offset is read only at segment boundaries of the old transcript,
        whose times are accurate: first token -> old segment start vs the
        probe word's start; last token -> old segment end vs the word's end.
        Interpolated word times are used only to pick the alignment, never
        for the offset."""
        toks = [w[0] for w in words]
        if len(toks) < MIN_TOKENS:
            return None, 0
        votes: dict[int, int] = defaultdict(int)
        for j in range(len(toks) - 3):
            for k in self.grams.get(tuple(toks[j:j + 4]), ()):
                votes[k - j] += 1
        if not votes:
            return None, 0
        shift = max(votes, key=votes.get)
        matched = 0
        anchored = []
        for j, (t, ns, ne) in enumerate(words):
            k = shift + j
            if 0 <= k < len(self.tok) and self.tok[k] == t:
                matched += 1
                a = self.anchor[k]
                if a is not None:
                    anchored.append(a[1] - (ns if a[0] == "start" else ne))
        if matched < max(MIN_TOKENS, MIN_MATCH * len(toks)) or not anchored:
            return None, matched
        return float(np.median(anchored)), matched


# --------------------------------------------------------------------------- probing

class Prober:
    def __init__(self, compute_type: str = "float16"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(MODEL, device="cuda", compute_type=compute_type)

    def words(self, wav: np.ndarray, t0: float, length: float = PROBE_LEN) -> list[tuple[str, float, float]]:
        a, b = int(t0 * SR), int(min(len(wav), (t0 + length) * SR))
        if b - a < SR:
            return []
        segs, _ = self.model.transcribe(wav[a:b], language="en", beam_size=1, word_timestamps=True,
                                        vad_filter=False, condition_on_previous_text=False)
        out = []
        for s in segs:
            for w in (s.words or []):
                for t in norm_tokens(w.word):
                    out.append((t, t0 + float(w.start), t0 + float(w.end)))
        return out


def probe_episode(prober: Prober, wav: np.ndarray, idx: OldIndex) -> list[tuple[float, float | None]]:
    """Returns [(new_time, offset or None)] sorted by new time, after bisection refinement."""
    duration = len(wav) / SR
    times = list(np.arange(5.0, min(EARLY_UNTIL, duration - PROBE_LEN), EARLY_EVERY))
    times += list(np.arange(EARLY_UNTIL, duration - PROBE_LEN, PROBE_EVERY))
    if duration > PROBE_LEN + 5:
        times.append(duration - PROBE_LEN - 2)
    probes: dict[float, float | None] = {}
    for t in times:
        probes[float(t)] = idx.locate(prober.words(wav, float(t)))[0]
    # bisect between neighbouring probes whose offsets differ (a splice lies between them)
    for _ in range(6):
        keys = sorted(probes)
        new = []
        for a, b in zip(keys, keys[1:]):
            oa, ob = probes[a], probes[b]
            if oa is not None and ob is not None and abs(oa - ob) > JUMP and b - a > PROBE_LEN * 1.5:
                new.append((a + b) / 2)
        if not new:
            break
        for t in new:
            probes[float(t)] = idx.locate(prober.words(wav, float(t)))[0]
    return sorted(probes.items())


def build_regions(probes: list[tuple[float, float | None]]) -> list[tuple[float, float, float]]:
    """Group probes into constant-offset regions. Returns [(new_start, new_end, offset)]."""
    regions = []
    cur = None
    for t, off in probes:
        if off is None:
            continue
        if cur is None or abs(off - cur[2]) > JUMP:
            if cur is not None:
                regions.append(cur)
            cur = [t, t + PROBE_LEN, off]
        else:
            cur[1] = t + PROBE_LEN
            cur[2] = float(np.median([cur[2], off])) if abs(cur[2] - off) > 0.3 else cur[2]
    if cur is not None:
        regions.append(cur)
    return [tuple(r) for r in regions]


def apply_regions(old_segments: list[dict], regions, duration: float) -> tuple[list[dict], dict]:
    """Map old segments to new time. Region k covers old times [new_start+off, new_end+off];
    between regions the boundary is placed where the new-time gap splits."""
    out = []
    if not regions:
        return [dict(s, start=None, end=None, matched=False, orig_start=s["start"], orig_end=s["end"]) for s in old_segments], {"n_matched": 0}
    # old-time spans of the regions, extended to meet halfway in new time
    spans = []
    for i, (ns, ne, off) in enumerate(regions):
        lo = ns + off if i > 0 else -1e9
        hi = ne + off if i < len(regions) - 1 else 1e9
        if i > 0:
            pns, pne, poff = regions[i - 1]
            mid_new = (pne + ns) / 2            # splice assumed midway between the two probes
            lo = mid_new + off                  # old time where this region starts
            spans[-1][1] = mid_new + poff       # previous region's old end
        spans.append([lo, hi, off])
    for s in old_segments:
        os_, oe = float(s["start"]), float(s["end"])
        mid = (os_ + oe) / 2
        hit = None
        for lo, hi, off in spans:
            if lo <= mid < hi:
                hit = off
                break
        if hit is None:
            out.append(dict(s, start=None, end=None, matched=False, orig_start=os_, orig_end=oe))
            continue
        ns, ne = os_ - hit, oe - hit
        if ne < 0 or ns > duration:
            out.append(dict(s, start=None, end=None, matched=False, orig_start=os_, orig_end=oe))
            continue
        out.append(dict(s, start=round(max(0.0, ns), 3), end=round(min(duration, ne), 3), matched=True,
                        orig_start=os_, orig_end=oe))
    # old-only spans (removed ads): consecutive regions whose old gap exceeds the new gap -> segments there unmatched
    for i in range(1, len(regions)):
        pns, pne, poff = regions[i - 1]
        ns, ne, off = regions[i]
        old_gap_extra = (off - poff)            # >0 means old audio had extra material (old ad) here
        if old_gap_extra > JUMP:
            mid_new = (pne + ns) / 2
            lo, hi = mid_new + poff, mid_new + off
            for o in out:
                if o["matched"] and lo <= (o["orig_start"] + o["orig_end"]) / 2 < hi:
                    o.update(start=None, end=None, matched=False)
    return out, {"n_matched": sum(o["matched"] for o in out), "n_regions": len(regions),
                 "offsets": [round(r[2], 2) for r in regions]}


def retime_one(prober: Prober, show_id: str, episode_id: str) -> tuple[dict, float] | None:
    hits = sorted((TRANSCRIPTS_DIR / show_id).glob(f"{episode_id}_*.json"))
    audio = find_audio(show_id, episode_id)
    if not hits or audio is None:
        return None
    orig = json.loads(hits[0].read_text())
    t0 = time.monotonic()
    wav = load_waveform(audio)
    duration = len(wav) / SR
    idx = OldIndex(orig["segments"])
    probes = probe_episode(prober, wav, idx)
    regions = build_regions(probes)
    segs, st = apply_regions(orig["segments"], regions, duration)
    wall = time.monotonic() - t0
    st.update({"n_probes": len(probes), "n_probes_matched": sum(o is not None for _, o in probes), "n_segments": len(segs)})
    rec = {**{k: v for k, v in orig.items() if k != "segments"}, "duration": duration, "segments": segs,
           "source": "probe", "probe_model": MODEL, "audio": str(audio.relative_to(PROJECT_ROOT)),
           "original_transcript": str(hits[0].relative_to(PROJECT_ROOT)), "retime_stats": st,
           "realtime_factor": round(duration / wall, 1)}
    return rec, wall


# --------------------------------------------------------------------------- validation

def speaker_agreement(show_id: str, episode_id: str, segs_a: list[dict], segs_b: list[dict]) -> float | None:
    """Agreement of segment speaker labels between two timings of the same episode, by
    time overlap, using the episode's diarization turns."""
    from diarize import assign_segments
    p = DIAR_DIR / show_id / f"{episode_id}.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text())
    turns = rec.get("exclusive_turns") or rec["turns"]
    A = [s for s in assign_segments([s for s in segs_a if s.get("start") is not None], turns)]
    B = assign_segments(segs_b, turns)
    bs = np.array([b["start"] for b in B]); be = np.array([b["end"] for b in B])
    agree = total = 0.0
    for a in A:
        ov = np.clip(np.minimum(be, a["end"]) - np.maximum(bs, a["start"]), 0, None)
        for j in np.nonzero(ov)[0]:
            total += ov[j]
            agree += ov[j] * (B[j]["speaker"] == a["speaker"])
    return agree / total if total else None


def run_validate(n: int, args) -> None:
    truth = []
    for p in sorted(V2_DIR.glob("*/*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if r.get("source") == "archive":
            truth.append((p.parent.name, p.stem, r))
        if len(truth) >= n:
            break
    prober = Prober(args.compute_type)
    errs, rtfs, agree, matched = [], [], [], []
    for show_id, episode_id, tr in truth:
        res = retime_one(prober, show_id, episode_id)
        if res is None:
            continue
        rec, wall = res
        rtfs.append(rec["duration"] / wall)
        segs = rec["segments"]
        matched.append(rec["retime_stats"]["n_matched"] / max(1, len(segs)))
        tsegs = tr["segments"]
        ttok = [set(norm_tokens(s["text"])) for s in tsegs]
        tstart = np.array([s["start"] for s in tsegs])
        e_ep = []
        for s in segs:
            if not s["matched"]:
                continue
            tok = set(norm_tokens(s["text"]))
            if len(tok) < 4:
                continue
            idx = np.where(np.abs(tstart - s["start"]) < 60)[0]
            if not len(idx):
                e_ep.append(60.0)
                continue
            best = max(idx, key=lambda k: len(tok & ttok[k]) / len(tok))
            if len(tok & ttok[best]) / len(tok) < 0.5:
                continue
            e_ep.append(abs(tsegs[best]["start"] - s["start"]))
        errs += e_ep
        ag = speaker_agreement(show_id, episode_id, segs, tsegs)
        if ag is not None:
            agree.append(ag)
        log(f"  {episode_id}: rtf {rtfs[-1]:.0f}x probes {rec['retime_stats']['n_probes']} regions {rec['retime_stats']['n_regions']} "
            f"offsets {rec['retime_stats']['offsets']} matched {matched[-1]:.0%} start-err median {np.median(e_ep) if e_ep else float('nan'):.2f}s "
            f"speaker-agree {ag if ag is not None else float('nan'):.3f}")
    e = np.array(errs)
    log(f"VALIDATION on {len(rtfs)} episodes: mean rtf {np.mean(rtfs):.0f}x | segments re-timed {np.mean(matched):.1%} | "
        f"start error vs full re-transcription: median {np.median(e):.2f}s, 90th {np.percentile(e, 90):.2f}s, within 1 s {np.mean(e <= 1):.1%}, "
        f"within 2 s {np.mean(e <= 2):.1%}, >5 s {np.mean(e > 5):.1%} (n={len(e)}) | segment speaker-label agreement with truth: "
        f"mean {np.mean(agree):.3f}, min {np.min(agree):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", type=int, metavar="N")
    ap.add_argument("--from-diarization", action="store_true")
    ap.add_argument("--episodes-file")
    ap.add_argument("--priority-shows")
    ap.add_argument("--shard")
    ap.add_argument("--tolerance", type=float, default=0.5)
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.validate:
        run_validate(args.validate, args)
        return
    from retranscribe import targets_from_diarization
    pairs = []
    if args.episodes_file:
        for line in Path(args.episodes_file).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                s, e = line.strip().split(",")[:2]
                pairs.append((s.strip(), e.strip()))
    if args.from_diarization:
        pairs += targets_from_diarization(args.tolerance)
    pairs = [(s, e) for s, e in dict.fromkeys(pairs) if not (V2_DIR / s / f"{e}.json").exists()]
    if args.priority_shows:
        first = {l.strip() for l in Path(args.priority_shows).read_text().splitlines() if l.strip()}
        pairs = [p for p in pairs if p[0] in first] + [p for p in pairs if p[0] not in first]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        pairs = pairs[k::n]
    if args.limit:
        pairs = pairs[: args.limit]
    log(f"targets: {len(pairs)}")
    if not pairs:
        return
    prober = Prober(args.compute_type)
    done = failed = 0
    audio_s = wall_s = 0.0
    for i, (show_id, episode_id) in enumerate(pairs, 1):
        out = V2_DIR / show_id / f"{episode_id}.json"
        if out.exists():
            continue
        try:
            res = retime_one(prober, show_id, episode_id)
        except Exception as e:
            failed += 1
            log(f"  FAIL {episode_id} ({show_id}): {type(e).__name__}: {str(e)[:160]}")
            continue
        if res is None:
            continue
        rec, wall = res
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.part")
        tmp.write_text(json.dumps(rec, ensure_ascii=False))
        tmp.replace(out)
        done += 1
        audio_s += rec["duration"]
        wall_s += wall
        st = rec["retime_stats"]
        log(f"  Done {episode_id} ({show_id}) rtf={rec['realtime_factor']}x regions {st['n_regions']} matched {st['n_matched']}/{st['n_segments']}")
        if i % 100 == 0:
            log(f"progress {i}/{len(pairs)} done={done} failed={failed} mean rtf={audio_s / max(1e-6, wall_s):.0f}x")
    log(f"finished: done={done} failed={failed} audio={audio_s / 3600:.1f}h wall={wall_s / 3600:.2f}h mean rtf={audio_s / max(1e-6, wall_s):.0f}x")


if __name__ == "__main__":
    main()
