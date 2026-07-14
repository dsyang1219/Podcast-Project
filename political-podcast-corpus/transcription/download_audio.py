#!/usr/bin/env python3
"""
HTTP downloader for the H25 sample manifest — producer half of a
producer/consumer pair with transcribe.py.

Streams each episode's RSS `audio_url` (podtrac/megaphone/substack/etc.
enclosure links — a direct HTTP GET, no YouTube involved) into
data/audio_queue/, then hands it off to the transcriber via the same
atomic-claim queue protocol used elsewhere in this project:
  - Audio file:  QUEUE_DIR/<episode_id>.<ext>
  - Meta file:   QUEUE_DIR/<episode_id>.meta -> written LAST, once the audio
                 is fully on disk, so the transcriber never claims a partial
                 file. Contents: "show_id|||episode_id|||audio_sha256|||title"
  - Stop signal: QUEUE_DIR/STOP (created after every row has been dispatched)
"""
import argparse
import csv
import hashlib
import mimetypes
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline import config  # noqa: E402  (USER_AGENT, HTTP_TIMEOUT, HTTP_RETRIES)

MANIFEST_DEFAULT = PROJECT_ROOT / "data/output/sample_out/sample_manifest_H25.csv"
QUEUE_DIR = PROJECT_ROOT / "data/audio_queue"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data/transcripts"
FAILURES_CSV = PROJECT_ROOT / "data/output/download_failures.csv"
LOG_PATH = PROJECT_ROOT / "transcription" / "download_audio.log"

PARALLEL = 8
SHOW_DELAY_SECONDS = 1.0  # politeness pacing, keyed by show_id (see plan: NOT
                          # by URL host — most of these wrap through a handful
                          # of redirector hosts like podtrac.com that have
                          # nothing to do with the real per-show CDN).
CHUNK_SIZE = 1 << 16
PERMANENT_STATUSES = {403, 404, 410}
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg", ".oga", ".aac", ".flac", ".opus"}

_log_lock = threading.Lock()
_failures_lock = threading.Lock()
_throttle_lock = threading.Lock()
_last_request_at: dict[str, float] = {}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _log_lock:
        print(line, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")


def throttle(key: str, delay: float = SHOW_DELAY_SECONDS) -> None:
    """Politeness pacing per show_id. Reserves the next slot atomically so
    concurrent workers hitting the same show don't double-book, but releases
    the lock before sleeping so unrelated shows aren't serialized."""
    with _throttle_lock:
        last = _last_request_at.get(key, 0.0)
        now = time.monotonic()
        wait = delay - (now - last)
        _last_request_at[key] = now + max(wait, 0.0)
    if wait > 0:
        time.sleep(wait)


def already_downloaded(episode_id: str) -> bool:
    return any(QUEUE_DIR.glob(f"{episode_id}.*"))


def already_transcribed(show_id: str, episode_id: str) -> bool:
    """Filenames are '<episode_id>_<title>.json' (see transcribe.py's
    find_existing_json) — glob by the episode_id prefix, not an exact name."""
    show_dir = TRANSCRIPTS_DIR / show_id
    return show_dir.exists() and any(show_dir.glob(f"{episode_id}_*.json"))


def guess_extension(resp: requests.Response) -> str:
    """Infer extension from the *final resolved* URL/Content-Type, after
    redirects. The original audio_url is often a wrapper like
    '.../redirect.mp3/pdst.fm/e/traffic.megaphone.fm/XYZ123' — its path is
    not a reliable indicator of the real file type."""
    path = urlparse(resp.url).path
    ext = Path(path).suffix.lower()
    if ext in AUDIO_EXTS:
        return ext
    ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if ctype == "audio/mpeg":
        return ".mp3"
    if ctype in ("audio/mp4", "audio/x-m4a"):
        return ".m4a"
    guessed = mimetypes.guess_extension(ctype) if ctype else None
    if guessed:
        return ".mp3" if guessed == ".mpga" else guessed
    return ".mp3"


def record_failure(row: dict, reason: str) -> None:
    FAILURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with _failures_lock:
        is_new = not FAILURES_CSV.exists()
        with open(FAILURES_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["show_id", "episode_id", "audio_url", "episode_title", "reason"])
            w.writerow([row["show_id"], row["episode_id"], row["audio_url"],
                        row["episode_title"], reason])


def fetch_with_retries(session: requests.Session, url: str, show_id: str):
    """GET with the project's standard retry/backoff shape (pipeline/http_util.py).
    Returns (response, None) on success, or (None, reason) on failure — permanent
    statuses (403/404/410) are not retried."""
    last_reason = "unknown error"
    for attempt in range(1, config.HTTP_RETRIES + 1):
        throttle(show_id)
        try:
            resp = session.get(url, stream=True, timeout=config.HTTP_TIMEOUT,
                                allow_redirects=True)
        except requests.RequestException as e:
            last_reason = f"{type(e).__name__}: {e}"
            if attempt < config.HTTP_RETRIES:
                time.sleep(2 ** attempt)
            continue

        if resp.status_code == 200:
            return resp, None
        if resp.status_code in PERMANENT_STATUSES:
            resp.close()
            return None, f"HTTP {resp.status_code}"
        last_reason = f"HTTP {resp.status_code}"
        resp.close()
        if attempt < config.HTTP_RETRIES:
            time.sleep(2 ** attempt)
    return None, last_reason


def download_one(row: dict, session: requests.Session) -> str:
    show_id = row["show_id"]
    episode_id = row["episode_id"]
    audio_url = row["audio_url"]
    title = row["episode_title"].replace("\n", " ").replace("\r", " ").strip()

    if not audio_url:
        record_failure(row, "empty audio_url")
        return "fail"
    if already_transcribed(show_id, episode_id):
        return "skip-done"
    if already_downloaded(episode_id):
        return "skip-queued"

    resp, reason = fetch_with_retries(session, audio_url, show_id)
    if resp is None:
        log(f"  FAIL {episode_id} ({show_id}): {reason}")
        record_failure(row, reason)
        return "fail"

    ext = guess_extension(resp)
    tmp_path = QUEUE_DIR / f"{episode_id}{ext}.part"
    final_path = QUEUE_DIR / f"{episode_id}{ext}"
    sha256 = hashlib.sha256()
    try:
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    sha256.update(chunk)
    except (requests.RequestException, OSError) as e:
        tmp_path.unlink(missing_ok=True)
        log(f"  ERROR {episode_id} ({show_id}): stream failed: {e}")
        record_failure(row, f"stream failed: {e}")
        return "fail"
    finally:
        resp.close()

    tmp_path.rename(final_path)
    meta_path = QUEUE_DIR / f"{episode_id}.meta"
    meta_path.write_text(f"{show_id}|||{episode_id}|||{sha256.hexdigest()}|||{title}\n")
    log(f"  Queued: {episode_id} ({show_id}) {title!r}")
    return "ok"


def load_manifest(path: Path, limit: int | None) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(MANIFEST_DEFAULT))
    ap.add_argument("--workers", type=int, default=PARALLEL)
    ap.add_argument("--limit", type=int, default=None,
                    help="only dispatch the first N manifest rows (testing)")
    ap.add_argument("--no-stop", action="store_true",
                    help="don't write STOP after dispatching (for incremental runs)")
    args = ap.parse_args()

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    (QUEUE_DIR / "STOP").unlink(missing_ok=True)

    rows = load_manifest(Path(args.manifest), args.limit)
    log(f"=== Starting download run: {len(rows)} episodes, workers={args.workers} ===")

    session = requests.Session()
    session.headers.update({"User-Agent": config.USER_AGENT})

    counts: dict[str, int] = defaultdict(int)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_one, row, session): row for row in rows}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                status = fut.result()
            except Exception as e:
                log(f"  EXCEPTION {row['episode_id']}: {e}")
                status = "fail"
            counts[status] += 1

    log(f"=== Done. {dict(counts)} ===")

    if not args.no_stop:
        (QUEUE_DIR / "STOP").touch()
        log("=== STOP signal written for transcriber ===")


if __name__ == "__main__":
    main()
