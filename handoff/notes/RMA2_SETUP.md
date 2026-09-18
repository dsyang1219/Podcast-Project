# rma2 setup spec — political-podcast-corpus

Written on rma1 (2026-08-10) for a Claude session running on **rma2.caltech.edu**.
Everything below has been verified on both boxes unless explicitly marked as unverified.

## 0. Situation

The corpus was rsynced rma1 → rma2 on 2026-08-10:

- Source: `dsyang@rma1:/home/dsyang/Podcast-Project/political-podcast-corpus/`
- Dest: `dsyang@rma2:~/political-podcast-corpus/`
- 44.2 GiB, 33,534 files, 1,645 dirs. Verified complete by a second dry pass returning
  0 files / 0 bytes to transfer.

The two boxes are **identical**: NVIDIA GB10, aarch64, Python 3.12.3, ffmpeg at
`/usr/bin/ffmpeg`, and the same home path `/home/dsyang`. Accounts are local per machine
(`passwd: files systemd`, no LDAP) and `/home` is local ext4 on each — there is **no shared
filesystem**, so nothing crosses between them implicitly.

Deliberately excluded from the sync, and therefore absent on rma2:

| Path | Size | Why |
|---|---|---|
| `__pycache__/` | 16 KB | Regenerates |
| `.env` | 307 B | Secrets — see §1 |
| `data/audio_queue/*.part` | 0.7 GB | 13 stale partials — see §3 |

`.venv/` and `.venv-tlda/` were excluded from that first pass but **have since been copied
over separately and are working** — see §2. Nothing further is needed for them.

## 1. Create `.env` (required before any LLM-calling script runs)

Path: `~/political-podcast-corpus/.env`. Loaded via `python-dotenv`; two scripts call
`load_dotenv(override=True)`, one calls plain `load_dotenv()`.

```
ANTHROPIC_API_KEY=<fresh key>
OPENAI_API_KEY=<fresh key>
```

Generate **new** keys rather than copying rma1's, so the two boxes can be revoked
independently. Consumers:

- `OPENAI_API_KEY` → `nlp/adspan_phase_a.py:92`, `nlp/adclf_phase_a.py:127`
  (both `OpenAI(api_key=os.environ["OPENAI_API_KEY"])` — a `KeyError` if unset)
- `ANTHROPIC_API_KEY` → `nlp/extract_guests.py`, `nlp/extract_guests_batch.py:159`
  (`anthropic.Anthropic()` with no args — reads the env var implicitly)

**Also note:** there is a *second* `.env` one level up on rma1 at
`~/Podcast-Project/.env` containing `HF_TOKEN` (HuggingFace, for gated pyannote
diarization models). It was outside the synced directory, so it is not on rma2 either.
Only needed if you run diarization — the current `step3_transcribe/transcribe.py` uses
faster-whisper without diarization, so it is probably not required.

Confirm the OpenAI account's rate ceiling before any bulk job: Tier 1 is 10k requests/day,
200k TPM, 2M enqueued batch tokens. If both boxes hit the API under the same org, they
share that budget.

## 2. The two virtualenvs — DONE, copied from rma1

Both envs were copied from rma1 on 2026-08-10 (11.9 GB, 99,684 files) and verified working
on rma2. **You do not need to build or install anything.** Confirmed on rma2:

```
.venv       -> openai 2.53.0 | anthropic 0.120.2 | numpy 2.5.1 | pip 26.1.2   (imports ok)
.venv-tlda  -> torch 2.13.0+cu130 | cuda_available True | device NVIDIA GB10
```

Copying was chosen over rebuilding because the boxes are identical (GB10, aarch64, Python
3.12.3) and a rebuild would have meant fighting aarch64 CUDA wheel resolution —
`.venv-tlda` pins `torch==2.13.0+cu130`, `.venv` pulls a full `nvidia-*` cu13 stack, and
`tlda` installs from a pinned git SHA.

**One caveat that has already been handled — do not undo it.** The corpus lives at a
*different* absolute path on each box:

- rma1: `/home/dsyang/Podcast-Project/political-podcast-corpus/`
- rma2: `/home/dsyang/political-podcast-corpus/`

Virtualenvs are not relocatable: 69 files (console-script shebangs in `bin/`, the `activate`
family, `pyvenv.cfg`) had the rma1 path baked in. These were rewritten in place on rma2 and
verified to leave zero remaining references. The interpreter itself was never affected —
`.venv/bin/python3` is a symlink to `/usr/bin/python3`.

If you ever re-copy either venv from rma1, you must redo that rewrite or every console
script (`pip`, `spacy`, `dotenv`, `torchrun`, …) will fail with a bad interpreter:

```bash
cd ~/political-podcast-corpus
OLD=/home/dsyang/Podcast-Project/political-podcast-corpus
NEW=/home/dsyang/political-podcast-corpus
grep -rlI "$OLD" .venv .venv-tlda | xargs -r sed -i "s|$OLD|$NEW|g"
```

`handoff/requirements-venv.txt` (112 pkgs) and `handoff/requirements-venv-tlda.txt`
(103 pkgs) are retained as the **only dependency record for this repo** — there are no
lockfiles, no `requirements.txt`, no `pyproject.toml`. Keep them for provenance and for any
future rebuild from scratch.

## 3. The 13 missing episodes

### What happened

`step3_transcribe/download_audio.py` streams each episode to
`data/audio_queue/<episode_id>.<ext>.part`, then renames to the final name and writes a
`.meta` file **last**, so the transcriber never claims a partial file. Two download runs were
interrupted mid-stream (file mtimes: 2025-07-13 17:25 and 2025-07-15 15:42), leaving 13
`.part` files with no `.meta`.

Those orphans are **self-perpetuating**, and this is the important part:

```python
def already_downloaded(episode_id: str) -> bool:
    return any(QUEUE_DIR.glob(f"{episode_id}.*"))     # line 77 — matches the .part too
...
    if already_downloaded(episode_id):
        return "skip-queued"                          # line 158
```

The glob is `{episode_id}.*`, which matches `<id>.mp3.part`. So every subsequent run reports
`skip-queued` and never re-fetches, while the transcriber ignores the file for want of a
`.meta`. The episode is invisible in both directions and silently absent from the corpus —
it does not appear in `data/output/download_failures.csv` either (verified: none of the 13
have a failure row).

### State

All 13 resolve against `data/output/sample_out/sample_manifest_H25.csv`, and **none has a
transcript** (checked `data/transcripts/<show_id>/<episode_id>_*.json`). They are genuinely
missing data, not harmless leftovers. They fall in exactly two shows:

| show_id | show | episodes |
|---|---|---|
| 1490993194 | Advisory Opinions | 6 |
| 1669610956 | System Update with Glenn Greenwald | 7 |

Full detail — id, show, pub_date, title, duration, `audio_url` — is in
`handoff/scans/missing_13_episodes.csv` beside this file.

Because the loss is concentrated in two shows rather than spread across the sample, treat
it as a possible per-show coverage bias in anything computed from the H25 sample, not just
13 lost rows.

### Recovery procedure

The `.part` files were **not** copied to rma2, so on rma2 the blocking condition does not
exist and the downloader will fetch these 13 normally. Just run it:

```bash
cd ~/political-podcast-corpus
.venv/bin/python step3_transcribe/download_audio.py --no-stop
```

`--no-stop` matters: without it the script writes a `STOP` sentinel on completion, which
signals the transcriber that the queue is finished. Already-downloaded and
already-transcribed episodes are skipped, so this is safe to run against the full manifest —
it should fetch exactly these 13.

Note the pacing built in: 8 parallel workers, 1.0 s politeness delay keyed by `show_id`
(deliberately not by URL host — most of these wrap through redirectors like podtrac.com that
say nothing about the real per-show CDN). Leave that alone; both shows would otherwise get
hammered, and these are live commercial feeds.

Some of these URLs are ~11 months old (earliest pub_date 2025-05-17). Enclosure links do
expire. `PERMANENT_STATUSES = {403, 404, 410}` are not retried and land in
`data/output/download_failures.csv` — if any of the 13 fail that way, the audio must be
re-resolved from the show's current RSS feed rather than the stored `audio_url`.

### Transcription — blocked on rma2, needs a decision

`step3_transcribe/transcribe.py` uses **faster-whisper / ctranslate2**, and neither is in either
requirements file. On rma1 that stack lives in a separate system-wide env at
**`/opt/whisperx`** — outside the home directory, so it was never part of this sync.

**`/opt/whisperx` does not exist on rma2** (verified). Options:

1. Build a user-local venv on rma2 with `faster-whisper` + `ctranslate2` (no root needed).
2. Have whoever administers rma2 replicate `/opt/whisperx` (needs root).
3. Download the 13 on rma2, copy the audio back to rma1, and transcribe there.

Known quirk to carry over: `/opt/whisperx` on rma1 currently fails to import with
`ImportError: libcudnn.so.9: cannot open shared object file`, and rma1's shell history shows
the workaround was to run with `env LD_LIBRARY_PATH="" whisperx ...`. Expect the same cuDNN
path collision wherever this stack gets rebuilt.

Whatever you build, **run the smoke test before any bulk transcription**:

```bash
python step3_transcribe/transcribe.py --smoke-test
```

It asserts `ctranslate2.get_cuda_device_count() > 0` and fails if throughput is under 0.5×
realtime (`transcribe.py:238-243`). The failure mode it guards against is a ctranslate2 build
that silently falls back to CPU — which still produces correct transcripts, just ~50× slower,
so a bulk run looks like it is working right up until it obviously is not.

## 4. Verification checklist

```bash
cd ~/political-podcast-corpus
test -f .env && grep -c API_KEY .env                  # expect 2 — the only setup step left
.venv/bin/python -c "import openai, anthropic, dotenv; print('nlp deps ok')"   # passes
.venv-tlda/bin/python -c "import torch; print('cuda:', torch.cuda.is_available())"  # True
find data/transcripts -name '*.json' | wc -l          # should match rma1
ls data/audio_queue/*.part 2>/dev/null | wc -l        # expect 0
```

## 5. Do not

- Do not copy `.env` from rma1 — generate fresh keys.
- Do not delete `data/transcripts_orphaned/` without checking; it is a real 62-directory
  artifact of an earlier run, not scratch.
- Do not re-run `download_audio.py` without `--no-stop` unless you intend to signal the
  transcriber that the queue is complete.
- Do not raise `PARALLEL` or lower `SHOW_DELAY_SECONDS` in `download_audio.py`.
