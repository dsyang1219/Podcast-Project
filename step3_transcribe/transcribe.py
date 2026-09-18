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
import threading
import time
from collections import deque
from csv import DictWriter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUEUE_DIR = PROJECT_ROOT / "data/audio_queue"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data/transcripts"
# Optional audio archive. Without it every file is unlinked after transcription,
# so any later pass that needs the audio itself — diarization above all — means
# re-downloading the whole corpus from origins that already rate-limit us. The
# move out of QUEUE_DIR is an instant same-filesystem rename on the GPU worker's
# critical path; the ~27s/episode Opus encode happens on background threads.
ARCHIVE_DIR = PROJECT_ROOT / "data/audio_archive"
# 1, not 2: Opus encoding is CPU work on the same package as the GPU, and two
# concurrent encoders raised the GPU temperature reading by ~17F. One keeps up
# comfortably — encoding runs ~100x realtime (~27s per episode) against a
# consumption rate of roughly one episode per 25s.
ARCHIVE_WORKERS = 1
ARCHIVE_BITRATE = "24k"   # 16 kHz mono speech; keeps speaker embeddings intact
STATUS_CSV = PROJECT_ROOT / "data/output/transcribe_status.csv"
STATUS_FIELDS = ["episode_id", "show_id", "engine", "model", "language",
                  "n_segments", "n_words", "realtime_factor", "wall_seconds"]

NO_META_POLL_SECS = 2.0

# --- thermal throttle --------------------------------------------------------
# Holds a target *average* GPU temperature by idling between episodes.
#
# Why in here rather than an external SIGSTOP governor: the die heats ~13F/s and
# nvidia-smi updates at ~1Hz, so anything reacting to instantaneous readings fires
# 10-20F late and then over-cools. Measured, that gave 31% duty at a 129F average
# — worse temperature *and* worse throughput than this approach should manage.
# Modulating at episode granularity (30-60s) instead means the control loop runs
# on a timescale where a 1Hz sensor is plenty, and the pause is spread smoothly
# rather than spent recovering from spikes.
#
# It also cannot wedge the pipeline. An external governor that dies mid-pause
# leaves a SIGSTOPped worker that still answers `kill -0`, so the supervisor sees
# it as healthy and the run stalls silently — which is how it died on Aug 14.
# A sleep inside the worker has no such failure mode.
THROTTLE_MIN_DUTY = 0.05     # never fully stop making progress
THROTTLE_MAX_DUTY = 1.0
# Gain, in duty per degree F of error, with a clamp on per-episode movement.
# Sized against the thermal lag: the package responds over a couple of episodes,
# so correcting most of the error in one step is stable, while the ~100x larger
# gain of a naive controller drives straight to the floor and sticks there.
THROTTLE_GAIN = 0.020
THROTTLE_MAX_STEP = 0.35     # clamp per-episode movement
THROTTLE_SMOOTH_N = 1        # react to the latest window, no extra lag
THROTTLE_MAX_SLEEP = 600.0   # cap any single idle period
# NB: idling does not let the GPU power down. A live CUDA context pins it in P0
# at 2405 MHz / 14W however long it sits unused (measured: 150s of zero work
# never dropped it); only P8 / 208 MHz / 5.5W is meaningfully cooler, and that
# needs the process to exit. Dropping the model reference was tried and does not
# do it — CTranslate2 frees device memory but the context outlives the model, so
# power moved 14.0W -> 13.4W and the P-state did not change. Against a clean
# baseline the context is worth only ~1-2C anyway, so the sleeps below buy
# reduced duty and nothing else.


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


def archive_move(audio_path: str, show_id: str, episode_id: str) -> Path | None:
    """Move the finished audio out of the queue, keeping the extension.

    A rename, not a copy: same filesystem, effectively free, and it leaves no
    window where the file sits in QUEUE_DIR without a .meta — an orphan nothing
    would ever clean up. Compression happens later, in the archive.
    """
    dest_dir = ARCHIVE_DIR / show_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{episode_id}{Path(audio_path).suffix}"
    try:
        os.replace(audio_path, dest)
        return dest
    except OSError as e:
        print(f"  WARNING: could not archive {episode_id}: {e}", flush=True)
        return None


def archive_compress(path: Path) -> None:
    """Opus-encode an archived file in place, then drop the original.

    Runs on a background thread so the GPU worker never waits on it. Failure is
    survivable by design: the uncompressed file simply stays, and the next
    startup sweep retries it. Losing audio is the thing worth avoiding here,
    not losing disk.
    """
    if path.suffix == ".opus" or not path.exists():
        return
    out = path.with_suffix(".opus.part")
    final = path.with_suffix(".opus")
    try:
        proc = subprocess.run(
            ["nice", "-n", "15", "ffmpeg", "-nostdin", "-y", "-v", "error",
             "-i", str(path), "-c:a", "libopus", "-b:a", ARCHIVE_BITRATE,
             "-application", "voip", "-f", "opus", str(out)],
            capture_output=True, timeout=1800)
    except (subprocess.TimeoutExpired, OSError) as e:
        out.unlink(missing_ok=True)
        print(f"  WARNING: archive encode failed for {path.name}: {e}", flush=True)
        return
    if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        out.unlink(missing_ok=True)
        print(f"  WARNING: archive encode failed for {path.name} "
              f"(rc={proc.returncode}) — keeping uncompressed", flush=True)
        return
    out.rename(final)
    path.unlink(missing_ok=True)


def sweep_archive(pool) -> None:
    """Re-queue anything left uncompressed by a killed worker or a failed encode."""
    if pool is None or not ARCHIVE_DIR.exists():
        return
    stale = [p for p in ARCHIVE_DIR.rglob("*")
             if p.is_file() and p.suffix not in (".opus", ".part")]
    for p in stale:
        pool.submit(archive_compress, p)
    for p in ARCHIVE_DIR.rglob("*.opus.part"):
        p.unlink(missing_ok=True)
    if stale:
        print(f"Archive: {len(stale)} file(s) pending compression from a previous run.",
              flush=True)


def gpu_temp_f() -> float | None:
    """Current GPU temperature in F, or None if it can't be read."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.split()
        return float(out[0]) * 9 / 5 + 32
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return None


class Throttle:
    """Idles between episodes to hold a target average GPU temperature.

    Proportional control on duty cycle. Temperature rise above idle is roughly
    proportional to average power, so duty is the right thing to steer: if we are
    hot, do less work per unit wall-clock. Corrections are deliberately small
    (THROTTLE_STEP) because the package has minutes of thermal mass — reacting
    hard to one reading is what makes the external governor oscillate.

    Failure is always toward progress: an unreadable sensor leaves duty alone
    rather than pausing, so a broken nvidia-smi can never stall the run.
    """

    SAMPLE_EVERY_S = 10.0    # background sampling interval
    WINDOW_S = 300.0         # averaging window (matches how the monitor reports)

    def __init__(self, target_f: float):
        self.target_f = target_f
        self.duty = 1.0
        self.samples: list[float] = []
        # Sample on a timer in the background instead of reconstructing the mean
        # from phase endpoints. Every reconstruction attempt was biased: peak-only
        # read ~12F high, equal-weighting peak/trough read high at low duty, and
        # weighting by phase duration still ran ~7F high because the work phase is
        # a ramp, not a step. A timer-based mean is what an external monitor
        # measures, so the controller and the monitor finally agree by construction.
        self._buf: deque = deque(maxlen=int(self.WINDOW_S / self.SAMPLE_EVERY_S))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.SAMPLE_EVERY_S):
            t = gpu_temp_f()
            if t is not None:
                self._buf.append(t)

    def observed_mean(self) -> float | None:
        return (sum(self._buf) / len(self._buf)) if self._buf else None

    def after_episode(self, work_seconds: float) -> float:
        """Sleep to enforce the duty cycle. Returns seconds actually slept."""
        sleep_s = 0.0
        if self.duty < 1.0:
            # work/duty = total cycle; the remainder is idle.
            want = min(work_seconds * (1.0 / self.duty - 1.0), THROTTLE_MAX_SLEEP)
            # Idle in chunks and stop as soon as the measured average is back at
            # target, rather than sleeping out a duration computed from a stale
            # reading. Converges faster and avoids over-sleeping a cold chip.
            while sleep_s < want:
                chunk = min(self.SAMPLE_EVERY_S, want - sleep_s)
                time.sleep(chunk)
                sleep_s += chunk
                obs = self.observed_mean()
                if obs is not None and obs <= self.target_f:
                    break

        observed = self.observed_mean()
        if observed is None:
            return sleep_s          # sensor unreadable: hold duty, keep working

        self.samples.append(observed)
        recent = self.samples[-THROTTLE_SMOOTH_N:]
        err = sum(recent) / len(recent) - self.target_f
        adj = max(-THROTTLE_MAX_STEP, min(THROTTLE_MAX_STEP, -THROTTLE_GAIN * err))
        self.duty = min(THROTTLE_MAX_DUTY, max(THROTTLE_MIN_DUTY, self.duty + adj))
        return sleep_s

    def report(self) -> str:
        if not self.samples:
            return f"duty={self.duty:.0%}"
        recent = self.samples[-20:]
        return (f"duty={self.duty:.0%} temp={sum(recent)/len(recent):.0f}F "
                f"target={self.target_f:.0f}F")


class StartGate:
    """Delays the *start* of each episode until the die is below a temperature.

    Why this rather than Throttle's duty cycle: measured over 2,770 burst onsets
    on this box, the peak a burst reaches is set mostly by where it starts, not
    by how long it runs. Peak rises 0.79F per 1F of start temperature (r=0.63),
    while burst length spans 9F across its whole range. So the lever that moves
    the peak is the starting point, and duty cycling — which only thins bursts
    out after the fact — cannot touch it. Measured starts and the peaks that
    followed:

        start <115F  -> peak 132F      start 135-145F -> peak 152F
        start 115-125F -> peak 137F    start 145F+    -> peak 162F

    Cost, however, is set entirely by where the gate sits relative to the idle
    floor, and the floor is not a constant: measured across one day on this box
    it moved 118F -> 136F as another tenant's CPU load ramped. A gate above the
    floor costs almost nothing; a gate below it can never be satisfied and every
    episode pays MAX_WAIT. Fixed gates of 125F and 130F were both measured
    underwater by afternoon, running 6-16% duty.

    Hence adaptive mode (floor_margin_f): track the floor and sit just above it,
    so the gate follows the box down overnight or when the other tenant stops,
    and cannot be configured below what is reachable. The floor is estimated as
    a low percentile rather than a strict minimum, so one cold sample cannot
    pin the gate somewhere unreachable for the next half hour.

    Progress is guaranteed either way: MAX_WAIT bounds every wait, so even a
    misconfigured gate degrades to a fixed delay rather than stalling the run.
    """

    SAMPLE_EVERY_S = 5.0     # responsive: we act on the current reading
    POLL_S = 5.0
    FLOOR_WINDOW_S = 1800.0  # 30 min: long enough to span a few work/idle cycles,
                             # short enough to track the floor drifting over a day
    FLOOR_PCTL = 0.10        # robust stand-in for "the floor"
    FLOOR_MIN_SAMPLES = 12   # ~1 min before the percentile is trusted

    def __init__(self, start_below_f: float | None = None,
                 max_wait_s: float = 300.0, floor_margin_f: float | None = None):
        self.start_below_f = start_below_f
        self.floor_margin_f = floor_margin_f
        self.max_wait_s = max_wait_s
        self._floor_buf: deque = deque(
            maxlen=int(self.FLOOR_WINDOW_S / self.SAMPLE_EVERY_S))
        # Primed synchronously: the sampler thread's first reading is one
        # SAMPLE_EVERY_S away, and an unprimed `latest` reads as "sensor
        # unreadable", which skips the gate. That would silently exempt the
        # first episode of every run — precisely the one starting from an
        # unknown temperature after a restart.
        self.latest: float | None = gpu_temp_f()
        self.waits: list[float] = []
        self.timeouts = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.SAMPLE_EVERY_S):
            t = gpu_temp_f()
            if t is not None:
                self.latest = t
                self._floor_buf.append(t)

    def floor_estimate(self) -> float | None:
        """Low percentile of the recent window — what this box idles at now."""
        if not self._floor_buf:
            return self.latest
        v = sorted(self._floor_buf)
        if len(v) < self.FLOOR_MIN_SAMPLES:
            return v[0]
        return v[int(self.FLOOR_PCTL * (len(v) - 1))]

    def threshold(self) -> float | None:
        """The temperature to wait for, recomputed each poll so adaptive mode
        tracks the floor even while a single wait is in progress."""
        if self.floor_margin_f is None:
            return self.start_below_f
        floor = self.floor_estimate()
        return None if floor is None else floor + self.floor_margin_f

    def wait_until_cool(self) -> float:
        """Block until the die is at or below the gate. Returns seconds waited."""
        t0 = time.monotonic()
        while True:
            temp = self.latest
            limit = self.threshold()
            if temp is None or limit is None:   # sensor unreadable: never block
                return time.monotonic() - t0
            if temp <= limit:
                break
            waited = time.monotonic() - t0
            if waited >= self.max_wait_s:
                self.timeouts += 1
                break
            time.sleep(min(self.POLL_S, self.max_wait_s - waited))
        waited = time.monotonic() - t0
        if waited > 0:
            self.waits.append(waited)
        return waited

    def report(self) -> str:
        limit = self.threshold()
        if self.floor_margin_f is not None:
            floor = self.floor_estimate()
            desc = (f"gate={limit:.0f}F (floor {floor:.0f}F +{self.floor_margin_f:.0f})"
                    if limit is not None else "gate=adaptive(no reading)")
        else:
            desc = f"gate={self.start_below_f:.0f}F"
        if not self.waits:
            return desc
        recent = self.waits[-20:]
        return (f"{desc} now={self.latest:.0f}F "
                f"mean_wait={sum(recent) / len(recent):.0f}s "
                f"timeouts={self.timeouts}")


def reclaim_stale_claims(max_age_secs: float = 7200.0) -> None:
    """Return claims abandoned by a killed worker to the queue.

    A hard kill mid-episode leaves '<id>.meta.claimed' behind. run() only globs
    '*.meta', so that episode is never retried and its audio is never freed —
    across a multi-week run with restarts these accumulate silently, the same
    way orphaned .part files did on the download side. Age-gated well above the
    worst-case single-episode time so a live worker's claim is never stolen.
    """
    now = time.time()
    reclaimed = 0
    for claimed in glob.glob(str(QUEUE_DIR / "*.meta.claimed")):
        try:
            if now - os.path.getmtime(claimed) > max_age_secs:
                os.rename(claimed, claimed[: -len(".claimed")])
                reclaimed += 1
        except OSError:
            continue
    if reclaimed:
        print(f"Reclaimed {reclaimed} stale claim(s) from a previous run.", flush=True)


def run(engine, model_name: str, archive_pool=None, throttle=None,
        gate=None) -> None:
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
            # Cool BEFORE working, not after. The peak this burst reaches is set
            # by the temperature it starts from, so the wait has to happen here
            # to have any effect on it. Deliberately outside the t_claim window
            # below: this is not work, and counting it as work would make the
            # duty controller (if also enabled) size its own idle against it.
            if gate is not None:
                waited = gate.wait_until_cool()
                if waited > 0:
                    print(f"  gate: waited {waited:.0f}s  {gate.report()}",
                          flush=True)

            # Anchors the throttle's duty calculation: everything from here to
            # the end of the finally block is "work", and the idle period after
            # it is sized against that.
            t_claim = time.monotonic()
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
                if archive_pool is not None:
                    dest = archive_move(audio_path, show_id, episode_id)
                    if dest is not None:
                        archive_pool.submit(archive_compress, dest)
                    else:
                        Path(audio_path).unlink(missing_ok=True)
                else:
                    Path(audio_path).unlink(missing_ok=True)
                if os.path.exists(claimed):
                    os.remove(claimed)
                gc.collect()

            # Idle AFTER releasing the claim and the audio, so a throttled worker
            # never holds an episode hostage — another worker (or a restart) can
            # pick up the queue while this one is cooling.
            if throttle is not None:
                slept = throttle.after_episode(time.monotonic() - t_claim)
                if slept > 0:
                    print(f"  throttle: slept {slept:.0f}s  {throttle.report()}",
                          flush=True)
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
        clip_path = PROJECT_ROOT / "step3_transcribe" / "_smoke_test_tone.wav"
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
    ap.add_argument("--target-temp-f", type=float, default=None,
                    metavar="F",
                    help="hold this AVERAGE GPU temperature by idling between "
                         "episodes. Costs throughput roughly in proportion to "
                         "how far below the uncapped plateau you ask for.")
    ap.add_argument("--start-below-f", type=float, default=None,
                    help="Wait until the GPU is at or below this temperature (F) "
                         "before starting each episode. Caps the PEAK, which the "
                         "duty-cycle throttle cannot: measured, peak rises 0.79F "
                         "per 1F of start temperature. 125 is a good first try "
                         "(~74%% duty, peaks ~137F); 120 is cooler and slower.")
    ap.add_argument("--start-above-floor", type=float, default=None,
                    metavar="MARGIN_F",
                    help="Adaptive gate: wait until the GPU is within MARGIN_F "
                         "degrees of its recent idle floor before starting each "
                         "episode. Preferred over --start-below-f, because the "
                         "floor is not constant — it moved 118F to 136F across "
                         "one day here as another tenant's load ramped, which "
                         "left fixed gates of 125F and 130F underwater and "
                         "running 6-16%% duty. 2 is a good margin.")
    ap.add_argument("--start-wait-max", type=float, default=300.0,
                    help="Cap on any single gate wait, so a gate below what the "
                         "box can reach degrades to a delay, never a stall.")
    ap.add_argument("--no-archive-audio", action="store_true",
                    help="delete audio after transcription instead of keeping an "
                         "Opus copy under data/audio_archive/. Saves ~8 MB per "
                         "episode, at the cost of having to re-download the whole "
                         "corpus for any later pass that needs audio (diarization).")
    args = ap.parse_args()

    language = None if args.auto_lang else args.language

    if args.smoke_test is not None:
        clip = Path(args.smoke_test) if args.smoke_test else None
        smoke_test(args.engine, args.model, args.compute_type, args.batch_size, clip)
        return

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    reclaim_stale_claims()

    archive_pool = None
    if not args.no_archive_audio:
        from concurrent.futures import ThreadPoolExecutor
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        archive_pool = ThreadPoolExecutor(max_workers=ARCHIVE_WORKERS,
                                          thread_name_prefix="archive")
        sweep_archive(archive_pool)
        print(f"Archiving audio to {ARCHIVE_DIR.name}/ "
              f"(opus {ARCHIVE_BITRATE}, {ARCHIVE_WORKERS} threads)", flush=True)

    print(f"Loading {args.engine} model {args.model} ({args.compute_type})...", flush=True)
    engine = ENGINES[args.engine](args.model, args.compute_type, args.batch_size, language)
    print("Model loaded. Watching queue...", flush=True)

    throttle = None
    if args.target_temp_f:
        throttle = Throttle(args.target_temp_f)
        print(f"Thermal throttle: holding ~{args.target_temp_f:.0f}F average "
              f"by idling between episodes", flush=True)

    gate = None
    if args.start_above_floor is not None:
        gate = StartGate(max_wait_s=args.start_wait_max,
                         floor_margin_f=args.start_above_floor)
        print(f"Start gate (adaptive): waiting for the die to come within "
              f"{args.start_above_floor:.0f}F of its recent idle floor before each "
              f"episode (max {args.start_wait_max:.0f}s)", flush=True)
    elif args.start_below_f:
        gate = StartGate(start_below_f=args.start_below_f,
                         max_wait_s=args.start_wait_max)
        print(f"Start gate: waiting for <= {args.start_below_f:.0f}F before each "
              f"episode (max {args.start_wait_max:.0f}s) — caps the peak, not the mean",
              flush=True)

    try:
        run(engine, args.model, archive_pool, throttle, gate)
    finally:
        if archive_pool is not None:
            # Let in-flight encodes finish; anything still queued is picked up by
            # sweep_archive() next start, since the file is already in the archive.
            archive_pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
