#!/usr/bin/env python3
"""Re-download the audio of already-transcribed episodes into the Opus archive.

The transcription ladder was run with ARCHIVE=0 for most of the corpus, so the
audio of ~60k transcribed episodes was unlinked after transcription. Speaker
diarization needs it back. This script fetches each episode's audio from its
feed URL and writes a 16 kHz mono Opus copy to data/audio_archive/<show_id>/
<episode_id>.opus — the same layout and encoding step3_transcribe/transcribe.py
uses when ARCHIVE=1, so both halves of the corpus end up identical on disk.

Opus at 24 kb/s VoIP profile is ~11 MB per audio hour; the whole corpus fits in
about 0.6 TB. Speaker-embedding models are trained on 16 kHz speech and are
robust to this codec (pyannote's own archive recommendation).

Targets: every episode with a transcript JSON under data/transcripts/ and no
.opus in the archive. Audio URLs come from the sampling manifests and
corpus_episodes.csv. Permanent HTTP failures (403/404/410) are logged to
data/output/archive_failures.csv and skipped on later runs.

Usage
-----
  .venv/bin/python step4_diarize/archive_download.py --dry-run     # counts only
  .venv/bin/python step4_diarize/archive_download.py --limit 20    # smoke test
  .venv/bin/python step4_diarize/archive_download.py --workers 12 --encoders 8

Run it after the transcription ladder has stopped (see
run_archive_after_ladder.sh): it shares origins with the ladder's downloader
and its ffmpeg encoders are CPU work on the same package as the GPU.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = PROJECT_ROOT / "data/transcripts"
ARCHIVE_DIR = PROJECT_ROOT / "data/audio_archive"
TMP_DIR = ARCHIVE_DIR / "_incoming"
FAILURES_CSV = PROJECT_ROOT / "data/output/archive_failures.csv"
LOG_PATH = PROJECT_ROOT / "step4_diarize/logs/archive_download.log"
MANIFEST_GLOB = str(PROJECT_ROOT / "data/output/sample_out/sample_manifest*.csv")
CORPUS_EPISODES = PROJECT_ROOT / "data/output/sample_out/corpus_episodes.csv"

OPUS_BITRATE = "24k"
PERMANENT_STATUSES = {403, 404, 410}
SHOW_DELAY_SECONDS = float(os.environ.get("PPC_SHOW_DELAY", "1.0"))
HTTP_TIMEOUT = int(os.environ.get("PPC_HTTP_TIMEOUT", "120"))
HTTP_RETRIES = int(os.environ.get("PPC_HTTP_RETRIES", "3"))
CHUNK_SIZE = 1 << 16
USER_AGENT = "political-podcast-corpus/archive (research; contact via GitHub dsyang1219)"

_log_lock = threading.Lock()
_fail_lock = threading.Lock()
_show_lock = threading.Lock()
_last_hit: dict[str, float] = {}
_thread_local = threading.local()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _log_lock:
        print(line, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")


def session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4))
        s.mount("http://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4))
        _thread_local.session = s
    return s


def throttle(show_id: str) -> None:
    """One request per show per SHOW_DELAY_SECONDS; sleep outside the lock."""
    with _show_lock:
        now = time.monotonic()
        wait = max(0.0, _last_hit.get(show_id, 0.0) + SHOW_DELAY_SECONDS - now)
        _last_hit[show_id] = now + wait
    if wait > 0:
        time.sleep(wait)


# --------------------------------------------------------------------------- targets

def transcribed_episodes() -> dict[str, str]:
    """episode_id -> show_id for every transcript JSON on disk."""
    out: dict[str, str] = {}
    for show_dir in TRANSCRIPTS_DIR.iterdir():
        if not show_dir.is_dir():
            continue
        for p in show_dir.glob("*.json"):
            out[p.name.split("_", 1)[0]] = show_dir.name
    return out


def archived_episodes() -> set[str]:
    if not ARCHIVE_DIR.exists():
        return set()
    return {p.stem for p in ARCHIVE_DIR.rglob("*.opus")}


RAW_SUFFIXES = {".mp3", ".m4a", ".wav", ".ogg", ".oga", ".aac", ".flac", ".mp4"}


def pending_raw() -> dict[str, Path]:
    """episode_id -> uncompressed file the ladder archived but never finished
    encoding (its single encoder fell behind the GPU). Encode these in place
    instead of downloading them again."""
    out: dict[str, Path] = {}
    if not ARCHIVE_DIR.exists():
        return out
    for p in ARCHIVE_DIR.rglob("*"):
        if p.is_file() and p.suffix in RAW_SUFFIXES and p.parent.name != TMP_DIR.name:
            out[p.stem] = p
    return out


def permanent_failures() -> set[str]:
    if not FAILURES_CSV.exists():
        return set()
    out = set()
    with open(FAILURES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["reason"].startswith(("HTTP 403", "HTTP 404", "HTTP 410")):
                out.add(row["episode_id"])
    return out


def audio_urls() -> dict[str, dict]:
    """episode_id -> {show_id, audio_url, episode_title} from every manifest."""
    rows: dict[str, dict] = {}
    files = sorted(glob.glob(MANIFEST_GLOB))
    if CORPUS_EPISODES.exists():
        files.append(str(CORPUS_EPISODES))
    for path in files:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "audio_url" not in reader.fieldnames:
                continue
            for r in reader:
                eid = r.get("episode_id", "")
                if eid and r.get("audio_url") and eid not in rows:
                    rows[eid] = {"show_id": r.get("show_id", ""),
                                 "audio_url": r["audio_url"],
                                 "episode_title": r.get("episode_title", "")}
    return rows


def build_targets(shows: set[str] | None) -> list[dict]:
    done = transcribed_episodes()
    have = archived_episodes()
    skip = permanent_failures()
    raw = pending_raw()
    urls = audio_urls()
    targets, missing_url, n_raw = [], 0, 0
    for eid, show_id in done.items():
        if eid in have or eid in skip:
            continue
        if shows and show_id not in shows:
            continue
        if eid in raw:
            targets.append({"show_id": show_id, "episode_id": eid, "audio_url": "",
                            "episode_title": "", "raw": raw[eid]})
            n_raw += 1
            continue
        u = urls.get(eid)
        if u is None:
            missing_url += 1
            continue
        targets.append({"show_id": show_id, "episode_id": eid,
                        "audio_url": u["audio_url"], "episode_title": u["episode_title"]})
    # interleave shows so no single origin sees a burst
    by_show: dict[str, list] = defaultdict(list)
    for t in targets:
        by_show[t["show_id"]].append(t)
    interleaved: list[dict] = []
    queues = list(by_show.values())
    while queues:
        nxt = []
        for q in queues:
            interleaved.append(q.pop(0))
            if q:
                nxt.append(q)
        queues = nxt
    log(f"targets: {len(interleaved):,} ({n_raw:,} already on disk, encode only) | {len(have):,} already archived | "
        f"{len(skip):,} known permanent failures | {missing_url} transcribed without a URL")
    return interleaved


# --------------------------------------------------------------------------- work

def record_failure(t: dict, reason: str) -> None:
    FAILURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with _fail_lock:
        new = not FAILURES_CSV.exists()
        with open(FAILURES_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["show_id", "episode_id", "audio_url", "episode_title", "reason"])
            w.writerow([t["show_id"], t["episode_id"], t["audio_url"], t["episode_title"], reason])


def fetch(t: dict) -> tuple[Path | None, str | None]:
    """Stream the audio to a .part file. Returns (path, None) or (None, reason)."""
    last = "no attempt"
    for attempt in range(HTTP_RETRIES + 1):
        throttle(t["show_id"])
        try:
            resp = session().get(t["audio_url"], stream=True, timeout=HTTP_TIMEOUT,
                                 allow_redirects=True)
        except requests.RequestException as e:
            last = f"request failed: {type(e).__name__}"
            time.sleep(2 ** attempt)
            continue
        if resp.status_code in PERMANENT_STATUSES:
            resp.close()
            return None, f"HTTP {resp.status_code}"
        if resp.status_code != 200:
            last = f"HTTP {resp.status_code}"
            resp.close()
            time.sleep(2 ** attempt)
            continue
        part = TMP_DIR / f"{t['episode_id']}.src.part"
        try:
            with open(part, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
        except (requests.RequestException, OSError) as e:
            part.unlink(missing_ok=True)
            last = f"stream failed: {type(e).__name__}"
            continue
        finally:
            resp.close()
        if part.stat().st_size < 10_000:
            part.unlink(missing_ok=True)
            return None, "tiny response (not audio)"
        return part, None
    return None, last


def encode(src: Path, t: dict, sem: threading.Semaphore) -> str | None:
    dest_dir = ARCHIVE_DIR / t["show_id"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_part = dest_dir / f"{t['episode_id']}.opus.part"
    final = dest_dir / f"{t['episode_id']}.opus"
    with sem:
        try:
            proc = subprocess.run(
                ["nice", "-n", "15", "ffmpeg", "-nostdin", "-y", "-v", "error",
                 "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                 "-c:a", "libopus", "-b:a", OPUS_BITRATE, "-application", "voip",
                 "-f", "opus", str(out_part)],
                capture_output=True, timeout=1800)
        except (subprocess.TimeoutExpired, OSError) as e:
            out_part.unlink(missing_ok=True)
            return f"encode failed: {type(e).__name__}"
    if proc.returncode != 0 or not out_part.exists() or out_part.stat().st_size == 0:
        out_part.unlink(missing_ok=True)
        err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        return f"encode failed (rc={proc.returncode}): {err[-1][:120] if err else 'no output'}"
    out_part.rename(final)
    return None


def one(t: dict, sem: threading.Semaphore) -> str:
    final = ARCHIVE_DIR / t["show_id"] / f"{t['episode_id']}.opus"
    if final.exists():
        return "skip"
    if t.get("raw") is not None:          # the ladder downloaded it; only the encode is missing
        src = t["raw"]
        if not src.exists():
            return "skip"
        if src.stat().st_size < 10_000:   # truncated or empty download: drop it and fetch again below
            src.unlink(missing_ok=True)
            if not t.get("audio_url"):
                t["audio_url"] = audio_urls().get(t["episode_id"], {}).get("audio_url", "")
            if not t["audio_url"]:
                log(f"  FAIL {t['episode_id']} ({t['show_id']}): empty raw file and no URL")
                return "fail"
        else:
            reason = encode(src, t, sem)
            if reason:
                log(f"  FAIL {t['episode_id']} ({t['show_id']}): {reason} (raw file kept)")
                return "fail"
            src.unlink(missing_ok=True)
            return "ok"
    src, reason = fetch(t)
    if src is None:
        record_failure(t, reason or "unknown")
        log(f"  FAIL {t['episode_id']} ({t['show_id']}): {reason}")
        return "fail"
    try:
        reason = encode(src, t, sem)
    finally:
        src.unlink(missing_ok=True)
    if reason:
        record_failure_if_download(t, reason)
        log(f"  FAIL {t['episode_id']} ({t['show_id']}): {reason}")
        return "fail"
    return "ok"


def record_failure_if_download(t: dict, reason: str) -> None:
    if t.get("audio_url"):
        record_failure(t, reason)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8, help="parallel downloads")
    ap.add_argument("--encoders", type=int, default=3, help="concurrent ffmpeg encoders")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shows", default=None, help="comma-separated show_ids to restrict to")
    ap.add_argument("--dry-run", action="store_true", help="count targets and exit")
    ap.add_argument("--encode-only", action="store_true",
                    help="only finish encoding files already on disk; no downloads")
    ap.add_argument("--retry-permanent", action="store_true",
                    help="also retry episodes whose last failure was HTTP 403/404/410 (use with few "
                         "workers and PPC_SHOW_DELAY=5 for hosts that rate-limit)")
    args = ap.parse_args()
    if args.retry_permanent:
        global permanent_failures
        permanent_failures = lambda: set()  # noqa: E731

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for stale in TMP_DIR.glob("*.part"):
        stale.unlink(missing_ok=True)

    shows = set(args.shows.split(",")) if args.shows else None
    targets = build_targets(shows)
    if args.encode_only:
        targets = [t for t in targets if t.get("raw") is not None]
    if args.limit:
        targets = targets[: args.limit]
    if args.dry_run or not targets:
        return

    log(f"=== archive run: {len(targets):,} episodes, workers={args.workers}, encoders={args.encoders} ===")
    sem = threading.Semaphore(args.encoders)
    counts: dict[str, int] = defaultdict(int)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, status in enumerate(ex.map(lambda t: one(t, sem), targets), 1):
            counts[status] += 1
            if i % 200 == 0 or i == len(targets):
                rate = i / max(1e-6, (time.monotonic() - t0)) * 3600
                log(f"  progress {i:,}/{len(targets):,} ok={counts['ok']} fail={counts['fail']} "
                    f"skip={counts['skip']} ({rate:,.0f}/h)")
    log(f"=== done: {dict(counts)} in {(time.monotonic() - t0) / 3600:.1f} h ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
