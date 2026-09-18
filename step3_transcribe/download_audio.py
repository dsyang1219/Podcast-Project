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
import os
import subprocess
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
from step1_frame import config  # noqa: E402  (USER_AGENT, HTTP_TIMEOUT, HTTP_RETRIES)

MANIFEST_DEFAULT = PROJECT_ROOT / "data/output/sample_out/sample_manifest_H25.csv"
QUEUE_DIR = PROJECT_ROOT / "data/audio_queue"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data/transcripts"
FAILURES_CSV = PROJECT_ROOT / "data/output/download_failures.csv"
LOG_PATH = PROJECT_ROOT / "step3_transcribe" / "logs" / "download_audio.log"

PARALLEL = 16   # safe now that manifests are show-interleaved; see MAX_QUEUED note
SHOW_DELAY_SECONDS = 1.0  # politeness pacing, keyed by show_id (see plan: NOT
                          # by URL host — most of these wrap through a handful
                          # of redirector hosts like podtrac.com that have
                          # nothing to do with the real per-show CDN).
                          # NB: this makes manifest ORDER decide throughput. A
                          # show-grouped manifest pins every worker to one
                          # show_id and caps the whole downloader at ~1 file/s
                          # regardless of workers or bandwidth. sample_ladder.py
                          # round-robins across shows for exactly this reason.
MAX_QUEUED = 250          # backpressure: cap episodes sitting in the queue.
                          # Downloads outrun the GPU by ~20x, so an unbounded
                          # run would need ~4.3 TB of scratch for a 78k-row
                          # manifest instead of the ~15 GB a bounded queue uses.
                          # With pre-decoding on, budget ~2x that: 16 kHz mono
                          # WAV runs 32 kB/s, so a ~44 min episode is ~85 MB
                          # against ~50 MB of mp3. Lower this if disk is tight.
QUEUE_POLL_SECS = 5.0
CHUNK_SIZE = 1 << 16
PERMANENT_STATUSES = {403, 404, 410}
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg", ".oga", ".aac", ".flac", ".opus"}

# --- pre-decoding (see predecode()) ---------------------------------------
DECODED_EXT = ".wav"      # 16 kHz mono s16: exactly what the model consumes.
DECODED_FORMAT = "wav"    # passed as -f: the temp file is named .wav.part, and
                          # ffmpeg can't infer a muxer from the .part suffix.
DECODED_SR = "16000"
PREDECODE_TIMEOUT = 900.0
# Concurrency cap for transcodes, deliberately well below PARALLEL. ffmpeg runs
# ~1 core flat out, so 16 unbounded transcodes would monopolise all 20 cores and
# starve the GPU workers of the CPU they need for VAD — turning a speedup into a
# slowdown. Steady state needs ~1-2 concurrent (one episode consumed per ~20s,
# each taking ~20s of a core), so this is already generous headroom.
PREDECODE_PARALLEL = 6
_predecode_sem = threading.Semaphore(PREDECODE_PARALLEL)
_predecode_enabled = True

# --- per-show circuit breaker ---------------------------------------------
# Origins rate-limit by podcast, not by episode. Observed on The Buck Sexton
# Show: 206 episodes downloaded fine, then every request began returning
# HTTP 500 "Podcast unavailable - try again later" after a ~31s stall. With
# HTTP_RETRIES=3 that is ~97s burned per episode, and because manifests are
# show-interleaved all 16 workers pile onto the same dead show at once — which
# is what stalled the producer for 10 minutes and drained the queue.
#
# So: after a few consecutive failures, stop calling that show entirely for a
# cooldown, and set its rows aside. The block is transient, so the episodes are
# deferred rather than failed — retried in a later pass instead of written off.
CIRCUIT_THRESHOLD = 3
CIRCUIT_COOLDOWN = 1800.0   # 30 min, then one probe request decides
DEFERRED_PASSES = 3         # retry rounds for rows set aside while a show was blocked

_log_lock = threading.Lock()
_failures_lock = threading.Lock()
_throttle_lock = threading.Lock()
_circuit_lock = threading.Lock()
_deferred_lock = threading.Lock()
_circuit_fails: dict[str, int] = {}
_circuit_open_until: dict[str, float] = {}
_deferred: list[dict] = []
_last_request_at: dict[str, float] = {}
_thread_local = threading.local()

# Prebuilt once in main(); per-row globbing over a queue/transcript dir holding
# thousands of files made the skip checks O(n^2) across a large manifest.
_already_queued: set[str] = set()
_already_done: set[str] = set()


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


def build_skip_sets() -> None:
    """Snapshot what is already queued or transcribed, once, at startup.

    A manifest lists each episode at most once, so nothing this run produces can
    make a later row skippable — a snapshot is as correct as a live check and
    turns two directory globs per row into two set lookups.
    """
    global _already_queued, _already_done
    if QUEUE_DIR.exists():
        # .part files are interrupted downloads, not queued work. Counting them
        # as queued silently drops those episodes from every subsequent run —
        # 13 episodes had been stranded this way since July.
        _already_queued = {f.split(".", 1)[0] for f in os.listdir(QUEUE_DIR)
                           if not f.endswith(".part")}
    if TRANSCRIPTS_DIR.exists():
        done = set()
        for show_dir in TRANSCRIPTS_DIR.iterdir():
            if show_dir.is_dir():
                for f in os.listdir(show_dir):
                    if f.endswith(".json"):
                        done.add(f.split("_", 1)[0])
        _already_done = done
    log(f"skip sets: {len(_already_queued):,} queued, {len(_already_done):,} transcribed")


def sweep_stale_parts(max_age_secs: float = 3600.0) -> None:
    """Delete interrupted downloads left by a killed run. Age-gated so a
    concurrently running downloader's in-flight .part files are never touched."""
    if not QUEUE_DIR.exists():
        return
    now = time.time()
    freed = n = 0
    for f in os.listdir(QUEUE_DIR):
        if not f.endswith(".part"):
            continue
        p = QUEUE_DIR / f
        try:
            st = p.stat()
            if now - st.st_mtime > max_age_secs:
                freed += st.st_size
                p.unlink()
                n += 1
        except OSError:
            continue
    if n:
        log(f"swept {n} stale .part files ({freed/1e9:.2f} GB reclaimed)")


def wait_for_queue_space(max_queued: int) -> None:
    """Block while the transcriber's backlog is full.

    Applied before the network fetch so a stalled or stopped transcriber parks
    the downloader instead of filling the disk.
    """
    warned = False
    while True:
        n = sum(1 for f in os.listdir(QUEUE_DIR) if f.endswith(".meta"))
        if n < max_queued:
            return
        if not warned:
            log(f"  queue full ({n} >= {max_queued}) — pausing downloads")
            warned = True
        time.sleep(QUEUE_POLL_SECS)


def get_session() -> requests.Session:
    """One Session per worker thread. requests.Session is not documented as
    thread-safe, and a single shared adapter pools only 10 connections, which
    would serialize workers above that count."""
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": config.USER_AGENT})
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _thread_local.session = s
    return s


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


def circuit_is_open(show_id: str) -> bool:
    """True if this show is in cooldown and should not be called at all.

    Half-open on expiry: the first caller after the cooldown is let through as a
    probe, with the failure count left one short of the threshold so a single
    failed probe re-opens the circuit immediately rather than costing another
    CIRCUIT_THRESHOLD episodes.
    """
    with _circuit_lock:
        until = _circuit_open_until.get(show_id)
        if until is None:
            return False
        if time.time() < until:
            return True
        del _circuit_open_until[show_id]
        _circuit_fails[show_id] = CIRCUIT_THRESHOLD - 1
        log(f"  circuit HALF-OPEN for show {show_id} — probing")
        return False


def circuit_record(show_id: str, ok: bool) -> None:
    """Feed a request outcome to the breaker. Any success closes it: the failures
    we care about are whole-show blocks, not the odd 404 on a single episode."""
    with _circuit_lock:
        if ok:
            _circuit_fails.pop(show_id, None)
            _circuit_open_until.pop(show_id, None)
            return
        n = _circuit_fails.get(show_id, 0) + 1
        _circuit_fails[show_id] = n
        if n >= CIRCUIT_THRESHOLD and show_id not in _circuit_open_until:
            _circuit_open_until[show_id] = time.time() + CIRCUIT_COOLDOWN
            log(f"  circuit OPEN for show {show_id} after {n} consecutive "
                f"failures — deferring its episodes for {CIRCUIT_COOLDOWN/60:.0f} min")


def wait_for_circuits(attempt: int) -> None:
    """Block until the earliest open circuit expires, so a deferred pass doesn't
    burn its retries re-deferring rows against a cooldown that is still running."""
    with _circuit_lock:
        if not _circuit_open_until:
            return
        wait = min(_circuit_open_until.values()) - time.time()
    wait = min(max(wait, 0.0), CIRCUIT_COOLDOWN)
    if wait > 0:
        log(f"  deferred pass {attempt}: waiting {wait/60:.1f} min for cooldowns to expire")
        time.sleep(wait)


def predecode(src: Path, episode_id: str) -> Path | None:
    """Transcode to 16 kHz mono WAV here, on the downloader's idle CPU, instead
    of inside the GPU worker's critical path.

    faster-whisper decodes the container itself at the top of transcribe(), so
    that cost is charged to GPU wall-clock with the device parked. Measured on
    this box: decoding an mp3 to PCM takes ~25s where the same file read from
    pre-decoded WAV takes ~1.35s, and the full pipeline runs at ~106x realtime
    against decode's ~405x — i.e. roughly a quarter of every episode's wall time
    was container decoding. The downloader has 16 workers that spend most of
    their life blocked on queue backpressure, so the work is close to free here.

    Returns the decoded path, or None if ffmpeg failed — the caller then ships
    the original container, which still transcribes correctly, just slower. A
    codec we can't decode must not cost us the episode.
    """
    out_part = QUEUE_DIR / f"{episode_id}{DECODED_EXT}.part"
    out_final = QUEUE_DIR / f"{episode_id}{DECODED_EXT}"
    with _predecode_sem:
        try:
            # nice: the GPU workers' own CPU work (VAD) outranks this.
            proc = subprocess.run(
                ["nice", "-n", "10", "ffmpeg", "-nostdin", "-y", "-v", "error",
                 "-i", str(src), "-ar", DECODED_SR, "-ac", "1",
                 "-f", DECODED_FORMAT, str(out_part)],
                capture_output=True, timeout=PREDECODE_TIMEOUT)
        except (subprocess.TimeoutExpired, OSError) as e:
            out_part.unlink(missing_ok=True)
            log(f"  predecode failed for {episode_id} ({type(e).__name__}) — queueing original")
            return None

    if proc.returncode != 0 or not out_part.exists() or out_part.stat().st_size == 0:
        err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        out_part.unlink(missing_ok=True)
        log(f"  predecode failed for {episode_id} (rc={proc.returncode}: "
            f"{err[-1] if err else 'no output'}) — queueing original")
        return None

    out_part.rename(out_final)
    return out_final


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
    """GET with the project's standard retry/backoff shape (step1_frame/http_util.py).
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


def download_one(row: dict, max_queued: int) -> str:
    show_id = row["show_id"]
    episode_id = row["episode_id"]
    audio_url = row["audio_url"]
    title = row["episode_title"].replace("\n", " ").replace("\r", " ").strip()

    if not audio_url:
        record_failure(row, "empty audio_url")
        return "fail"
    if episode_id in _already_done:
        return "skip-done"
    if episode_id in _already_queued:
        return "skip-queued"

    # Checked before wait_for_queue_space: a deferred row should cost nothing,
    # not park a worker against backpressure first.
    if circuit_is_open(show_id):
        with _deferred_lock:
            _deferred.append(row)
        return "deferred"

    wait_for_queue_space(max_queued)
    resp, reason = fetch_with_retries(get_session(), audio_url, show_id)
    circuit_record(show_id, resp is not None)
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

    # Transcode straight from the .part file, so the original never exists under
    # a name build_skip_sets() counts as queued. A crash mid-transcode then
    # leaves only .part files, which sweep_stale_parts() reclaims; renaming first
    # would strand the episode as "queued" with no .meta — invisible to both
    # halves for the rest of the run.
    if _predecode_enabled and predecode(tmp_path, episode_id) is not None:
        tmp_path.unlink(missing_ok=True)
    else:
        tmp_path.rename(final_path)

    # NB: sha256 is over the *original downloaded bytes*, not the transcode, so
    # resume/dedup against existing transcripts is unaffected by pre-decoding.
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
    ap.add_argument("--max-queued", type=int, default=MAX_QUEUED,
                    help="pause downloading while this many episodes await transcription")
    ap.add_argument("--no-predecode", action="store_true",
                    help="queue the downloaded container as-is instead of "
                         "transcoding to 16 kHz mono WAV first. Costs the GPU "
                         "worker ~25s of container decode per episode, but uses "
                         "roughly a third the queue disk.")
    args = ap.parse_args()

    global _predecode_enabled
    _predecode_enabled = not args.no_predecode

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    (QUEUE_DIR / "STOP").unlink(missing_ok=True)

    rows = load_manifest(Path(args.manifest), args.limit)
    log(f"=== Starting download run: {len(rows)} episodes, workers={args.workers}, "
        f"max_queued={args.max_queued}, predecode={_predecode_enabled} ===")
    sweep_stale_parts()
    build_skip_sets()

    counts: dict[str, int] = defaultdict(int)

    def dispatch(batch: list[dict]) -> None:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(download_one, row, args.max_queued): row for row in batch}
            for fut in as_completed(futures):
                row = futures[fut]
                try:
                    status = fut.result()
                except Exception as e:
                    log(f"  EXCEPTION {row['episode_id']}: {e}")
                    status = "fail"
                counts[status] += 1

    dispatch(rows)

    # Rows the breaker set aside are retried once the main pass is done — by
    # then a transient block has long since lifted. Repeat while each pass is
    # still shrinking the backlog, so a show that blocks a second time in the
    # retry pass still gets another go.
    for attempt in range(1, DEFERRED_PASSES + 1):
        with _deferred_lock:
            batch, _deferred[:] = list(_deferred), []
        if not batch:
            break
        wait_for_circuits(attempt)
        log(f"=== deferred pass {attempt}/{DEFERRED_PASSES}: retrying {len(batch):,} episodes ===")
        dispatch(batch)

    # Whatever is still deferred after the last pass is a real failure: record it
    # so the episodes are accounted for rather than vanishing from the run.
    with _deferred_lock:
        for row in _deferred:
            record_failure(row, "deferred: show blocked for the whole run")
        if _deferred:
            log(f"  {len(_deferred):,} episodes still blocked after "
                f"{DEFERRED_PASSES} passes — recorded as failures")

    log(f"=== Done. {dict(counts)} ===")

    if not args.no_stop:
        (QUEUE_DIR / "STOP").touch()
        log("=== STOP signal written for transcriber ===")


if __name__ == "__main__":
    main()
