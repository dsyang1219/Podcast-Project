"""Phase B -- detect ad/meta spans across every transcript and excise them.

Runs only after the Phase A gate passes. Five subcommands, each resumable, so
a crash or an interrupted batch never re-spends money already spent:

    segment   spaCy-sentencize all 7,626 episodes -> sentences.jsonl.gz (free,
              ~1 CPU-hour; needed by everything downstream)
    build     windows -> OpenAI Batch API request files (free)
    submit    upload + create batches (spends money)
    poll      report batch status
    collect   download results, merge spans across overlapping windows, excise,
              write cleaned transcripts + the full audit trail

Excision happens on the RAW transcript, upstream of every analysis unit. The
cleaned text is written in exactly the shape nlp/run_chunks.py already consumes
(one joined string per episode), so Phase C re-runs the EXISTING segmentation
logic unchanged -- the only difference is that the ads are gone from its input.

Failure direction is fixed by design: any window whose response is missing,
malformed, or truncated contributes NO spans, so it excises nothing. Errors
leave ads in; they never delete discourse.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from pipeline import config as pipeline_config
from .adspan_common import (
    Span,
    build_windows,
    excise,
    merge_spans,
    segments_to_text,
    sentences_for_texts,
)
from .adspan_detect import DEFAULT_MODEL, PRICING, build_messages, parse_spans

load_dotenv(override=True)

TRANSCRIPTS_DIR = pipeline_config.DATA_DIR / "transcripts"
OUT_DIR = pipeline_config.OUTPUT_DIR / "adspan"
SENTENCES_PATH = OUT_DIR / "sentences.jsonl.gz"
BATCH_DIR = OUT_DIR / "batch"
BATCH_STATE = OUT_DIR / "batch_state.json"
CLEANED_PATH = OUT_DIR / "cleaned_transcripts.jsonl.gz"
AUDIT_PATH = OUT_DIR / "excision_audit.jsonl.gz"

# The binding constraint is NOT the 50k-request / 200MB per-file limit but the
# account's ENQUEUED TOKEN limit: 2,000,000 for gpt-4o-mini at this tier. A
# create() call is accepted and only then fails asynchronously with
# token_limit_exceeded, so oversized files look fine until they don't. Files are
# therefore packed by MEASURED token count with headroom under the limit, and
# run strictly one at a time (see cmd_run_waves).
ENQUEUED_TOKEN_LIMIT = 2_000_000
BATCH_TOKEN_BUDGET = 1_800_000  # 10% headroom for tokenizer estimate drift


# ------------------------------------------------------------- segment ----
def cmd_segment(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if SENTENCES_PATH.exists() and not args.force:
        with gzip.open(SENTENCES_PATH, "rt") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["episode_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[segment] resuming: {len(done)} episodes already segmented")

    paths = sorted(TRANSCRIPTS_DIR.glob("*/*.json"))
    todo = []
    for p in paths:
        d = json.loads(p.read_text())
        if d["episode_id"] in done:
            continue
        text = segments_to_text(d["segments"])
        if not text:
            continue
        todo.append({"episode_id": d["episode_id"], "show_id": str(d["show_id"]),
                      "episode_title": d.get("episode_title", ""), "text": text})
    print(f"[segment] {len(todo)} episodes to segment (of {len(paths)} transcripts)")
    if not todo:
        return

    mode = "at" if (SENTENCES_PATH.exists() and not args.force) else "wt"
    written = 0
    with gzip.open(SENTENCES_PATH, mode) as out:
        # Chunk so partial progress is flushed and resumable, rather than
        # holding 3.7M sentences in memory and losing everything on a crash.
        step = 200
        for i in range(0, len(todo), step):
            block = todo[i:i + step]
            sent_lists = sentences_for_texts([b["text"] for b in block],
                                              n_process=args.n_process)
            for b, sents in zip(block, sent_lists):
                out.write(json.dumps({
                    "episode_id": b["episode_id"], "show_id": b["show_id"],
                    "episode_title": b["episode_title"], "sentences": sents,
                }) + "\n")
                written += 1
            print(f"[segment] {written}/{len(todo)} episodes...", flush=True)
    print(f"[segment] -> {SENTENCES_PATH}")


def load_sentences() -> list[dict]:
    out = []
    with gzip.open(SENTENCES_PATH, "rt") as f:
        for line in f:
            out.append(json.loads(line))
    return out


# --------------------------------------------------------------- build ----
def cmd_build(args) -> None:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    episodes = load_sentences()
    n_sent = sum(len(e["sentences"]) for e in episodes)
    print(f"[build] {len(episodes)} episodes, {n_sent} sentences")

    requests = []
    for e in episodes:
        for w in build_windows(e["sentences"], e["episode_id"], e["show_id"]):
            requests.append({
                "custom_id": f"{e['episode_id']}|{w.window_index}|{w.offset}|{len(w.sentences)}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": build_messages(w.numbered_text()),
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
            })
    print(f"[build] {len(requests)} windows")

    # Measure real token counts so waves pack close to the enqueued-token
    # ceiling without crossing it. Estimating from bytes would either waste a
    # third of each wave or trip the async failure this exists to avoid.
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    print(f"[build] tokenizing {len(requests)} requests to pack waves...", flush=True)
    costs = []
    for i, r in enumerate(requests):
        costs.append(sum(len(enc.encode(m["content"])) + 4 for m in r["body"]["messages"]))
        if (i + 1) % 25_000 == 0:
            print(f"  tokenized {i + 1}/{len(requests)}...", flush=True)
    total_tok = sum(costs)
    print(f"[build] {total_tok / 1e6:.1f}M input tokens total")

    files, cur, cur_tok, idx = [], [], 0, 0

    def flush_wave():
        nonlocal cur, cur_tok, idx
        if not cur:
            return
        path = BATCH_DIR / f"requests_{idx:04d}.jsonl"
        with path.open("w") as f:
            for r in cur:
                f.write(json.dumps(r) + "\n")
        files.append({"path": str(path), "n_requests": len(cur), "tokens": cur_tok})
        idx += 1
        cur, cur_tok = [], 0

    for r, c in zip(requests, costs):
        if cur and cur_tok + c > BATCH_TOKEN_BUDGET:
            flush_wave()
        cur.append(r)
        cur_tok += c
    flush_wave()

    print(f"[build] packed into {len(files)} waves "
          f"(<= {BATCH_TOKEN_BUDGET:,} tokens each, limit {ENQUEUED_TOKEN_LIMIT:,})")
    print(f"[build] mean {sum(f['n_requests'] for f in files) / len(files):.0f} "
          f"requests/wave, mean {sum(f['tokens'] for f in files) / len(files) / 1e6:.2f}M tokens/wave")

    state = {"model": args.model, "n_requests": len(requests),
             "total_input_tokens": total_tok,
             "files": [f["path"] for f in files], "wave_meta": files,
             "batches": {}}
    BATCH_STATE.write_text(json.dumps(state, indent=2))
    print(f"[build] -> {BATCH_STATE}")

    price = PRICING.get(args.model, PRICING["gpt-4o-mini"])
    est_in = len(requests) * 1410
    est_out = len(requests) * 30
    sync = est_in / 1e6 * price["in"] + est_out / 1e6 * price["out"]
    print(f"[build] estimated: {est_in/1e6:.1f}M in / {est_out/1e6:.1f}M out -> "
          f"sync ${sync:.2f}, batch ${sync/2:.2f}")


# -------------------------------------------------------------- submit ----
def _client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def cmd_submit(args) -> None:
    """Submit every request file, in waves if the account's batch queue is full.

    Enqueued-token limits are per-account and not introspectable, so capacity is
    discovered by attempting a submission and reading the rejection. A rejected
    create() costs nothing, so retrying on a timer is safe -- the alternative
    (guessing a wave size) either wastes capacity or stalls on a wrong guess.
    """
    import time as _time

    client = _client()
    state = json.loads(BATCH_STATE.read_text())
    uploaded: dict[str, str] = state.setdefault("uploaded_files", {})

    pending = [p for p in state["files"] if Path(p).name not in state["batches"]]
    print(f"[submit] {len(pending)} file(s) to submit, "
          f"{len(state['batches'])} already in flight")

    while pending:
        stalled = []
        for path in pending:
            name = Path(path).name
            # Re-uploading a 130MB file on every retry wastes minutes; the
            # uploaded file id is reusable across create() attempts.
            if name not in uploaded:
                print(f"[submit] uploading {name} "
                      f"({Path(path).stat().st_size / 1e6:.0f} MB)...", flush=True)
                up = client.files.create(file=Path(path).open("rb"), purpose="batch")
                uploaded[name] = up.id
                state["uploaded_files"] = uploaded
                BATCH_STATE.write_text(json.dumps(state, indent=2))
            try:
                b = client.batches.create(input_file_id=uploaded[name],
                                           endpoint="/v1/chat/completions",
                                           completion_window="24h")
            except Exception as e:  # queue-full is the expected case
                print(f"[submit] {name} deferred: {str(e)[:200]}")
                stalled.append(path)
                continue
            state["batches"][name] = {"id": b.id, "input_file_id": uploaded[name],
                                       "status": b.status}
            BATCH_STATE.write_text(json.dumps(state, indent=2))
            print(f"[submit] {name} -> batch {b.id} ({b.status})")

        pending = stalled
        if pending:
            print(f"[submit] {len(pending)} file(s) waiting on batch queue capacity; "
                  f"retrying in {args.retry_wait}s", flush=True)
            _time.sleep(args.retry_wait)

    print(f"[submit] all {len(state['batches'])} batch(es) in flight")


def cmd_run_waves(args) -> None:
    """Run every wave to completion, one at a time.

    The enqueued-token ceiling means exactly one wave fits at a time, so this is
    inherently sequential: submit -> poll to terminal -> download -> next. Each
    wave's results land on disk before the next is submitted, so an interruption
    costs at most the wave in flight, and a restart resumes from disk.
    """
    import time as _time

    client = _client()
    state = json.loads(BATCH_STATE.read_text())
    raw_dir = BATCH_DIR / "results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta = {m["path"]: m for m in state.get("wave_meta", [])}

    t_start = _time.time()
    for wave_i, path in enumerate(state["files"]):
        name = Path(path).name
        local = raw_dir / f"{name}.out.jsonl"
        if local.exists() and local.stat().st_size > 0:
            continue

        info = state["batches"].get(name)
        # Re-submit a wave that previously failed (e.g. the token-limit
        # rejections) rather than treating it as permanently done.
        if info and info.get("status") in ("failed", "cancelled", "expired"):
            info = None
        if info is None:
            up_id = state.setdefault("uploaded_files", {}).get(name)
            if not up_id:
                up = client.files.create(file=Path(path).open("rb"), purpose="batch")
                up_id = up.id
                state["uploaded_files"][name] = up_id
                BATCH_STATE.write_text(json.dumps(state, indent=2))
            while True:
                try:
                    b = client.batches.create(input_file_id=up_id,
                                               endpoint="/v1/chat/completions",
                                               completion_window="24h")
                    break
                except Exception as e:
                    print(f"[waves] {name} create deferred: {str(e)[:160]}", flush=True)
                    _time.sleep(args.poll_interval)
            info = {"id": b.id, "input_file_id": up_id, "status": b.status}
            state["batches"][name] = info
            BATCH_STATE.write_text(json.dumps(state, indent=2))
            tok = meta.get(path, {}).get("tokens", 0)
            print(f"[waves] {wave_i + 1}/{len(state['files'])} {name} -> {b.id} "
                  f"({tok / 1e6:.2f}M tok)", flush=True)

        while True:
            b = client.batches.retrieve(info["id"])
            if b.status in ("completed", "failed", "cancelled", "expired"):
                break
            _time.sleep(args.poll_interval)

        info["status"] = b.status
        info["output_file_id"] = b.output_file_id
        info["error_file_id"] = b.error_file_id
        BATCH_STATE.write_text(json.dumps(state, indent=2))

        if b.status != "completed" or not b.output_file_id:
            errs = getattr(b, "errors", None)
            print(f"[waves] {name} {b.status}: {errs}", flush=True)
            if not args.keep_going:
                raise SystemExit(f"wave {name} ended {b.status} -- stopping. "
                                  f"Re-run to retry it.")
            continue

        local.write_bytes(client.files.content(b.output_file_id).read())
        rc = b.request_counts
        done = wave_i + 1
        elapsed = _time.time() - t_start
        eta = elapsed / done * (len(state["files"]) - done)
        print(f"[waves] {done}/{len(state['files'])} {name} completed "
              f"(ok={rc.completed} failed={rc.failed}) "
              f"| elapsed {elapsed / 3600:.1f}h ETA {eta / 3600:.1f}h", flush=True)

    print("[waves] all waves complete -- run `collect` next")


def cmd_poll(args) -> None:
    client = _client()
    state = json.loads(BATCH_STATE.read_text())
    all_done = True
    for name, info in state["batches"].items():
        b = client.batches.retrieve(info["id"])
        info["status"] = b.status
        info["output_file_id"] = b.output_file_id
        info["error_file_id"] = b.error_file_id
        rc = b.request_counts
        print(f"[poll] {name}: {b.status}  "
              f"completed={rc.completed} failed={rc.failed} total={rc.total}")
        if b.status not in ("completed", "failed", "cancelled", "expired"):
            all_done = False
    BATCH_STATE.write_text(json.dumps(state, indent=2))
    print(f"[poll] {'ALL TERMINAL' if all_done else 'still running'}")


# ------------------------------------------------------------- collect ----
def cmd_collect(args) -> None:
    client = _client()
    state = json.loads(BATCH_STATE.read_text())

    # ---- download every result line, keyed by custom_id -------------------
    raw_dir = BATCH_DIR / "results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    by_window: dict[str, dict] = {}
    tok_in = tok_out = 0
    n_error_lines = 0

    for name, info in state["batches"].items():
        ofid = info.get("output_file_id")
        if not ofid:
            print(f"[collect] {name}: no output file (status={info.get('status')}) -- skipped")
            continue
        local = raw_dir / f"{name}.out.jsonl"
        if not local.exists():
            print(f"[collect] downloading results for {name}...", flush=True)
            local.write_bytes(client.files.content(ofid).read())
        for line in local.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec.get("custom_id")
            resp = rec.get("response") or {}
            body = resp.get("body") or {}
            if resp.get("status_code") != 200 or "choices" not in body:
                n_error_lines += 1
                continue
            usage = body.get("usage") or {}
            tok_in += usage.get("prompt_tokens", 0)
            tok_out += usage.get("completion_tokens", 0)
            by_window[cid] = body["choices"][0]["message"]["content"]

    print(f"[collect] {len(by_window)} window responses, {n_error_lines} error lines")
    print(f"[collect] tokens: {tok_in/1e6:.1f}M in / {tok_out/1e6:.1f}M out")

    # ---- map window spans back into episode sentence coordinates ----------
    spans_by_episode: dict[str, list[Span]] = defaultdict(list)
    n_parse_errors = 0
    n_spans_raw = 0
    for cid, content in by_window.items():
        try:
            episode_id, _widx, offset, n_win = cid.split("|")
            offset, n_win = int(offset), int(n_win)
        except ValueError:
            n_parse_errors += 1
            continue
        spans, err = parse_spans(content, n_win)
        if err:
            n_parse_errors += 1
        for s in spans:
            n_spans_raw += 1
            spans_by_episode[episode_id].append(
                Span(s.start + offset, s.end + offset, s.label, s.confidence))

    print(f"[collect] {n_spans_raw} raw spans across {len(spans_by_episode)} episodes "
          f"({n_parse_errors} parse errors -> excised nothing)")

    # ---- excise + write cleaned transcripts and the audit trail -----------
    episodes = load_sentences()
    per_show_removed: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # removed, total
    n_words_before = n_words_after = 0
    n_spans_merged = 0
    rows = []

    with gzip.open(CLEANED_PATH, "wt") as cf, gzip.open(AUDIT_PATH, "wt") as af:
        for e in episodes:
            sents = e["sentences"]
            merged = merge_spans(spans_by_episode.get(e["episode_id"], []))
            n_spans_merged += len(merged)
            kept, removed_idx = excise(sents, merged)

            w_before = sum(len(s.split()) for s in sents)
            w_removed = sum(len(sents[i].split()) for i in sorted(removed_idx))
            n_words_before += w_before
            n_words_after += w_before - w_removed

            agg = per_show_removed[e["show_id"]]
            agg[0] += w_removed
            agg[1] += w_before

            cf.write(json.dumps({
                "episode_id": e["episode_id"], "show_id": e["show_id"],
                "episode_title": e["episode_title"],
                "text": " ".join(kept),
            }) + "\n")
            af.write(json.dumps({
                "episode_id": e["episode_id"], "show_id": e["show_id"],
                "n_sentences": len(sents), "n_sentences_removed": len(removed_idx),
                "n_words": w_before, "n_words_removed": w_removed,
                "spans": [s.to_dict() for s in merged],
                # the exact excised text, so no removal is a black box
                "removed_text": [sents[i] for i in sorted(removed_idx)],
            }) + "\n")
            rows.append({
                "episode_id": e["episode_id"], "show_id": e["show_id"],
                "removed_share_words": w_removed / w_before if w_before else 0.0,
                "n_words": w_before,
            })

    price = PRICING.get(state["model"], PRICING["gpt-4o-mini"])
    actual = (tok_in / 1e6 * price["in"] + tok_out / 1e6 * price["out"]) * 0.5

    import pandas as pd
    df = pd.DataFrame(rows)
    show_df = pd.DataFrame([
        {"show_id": s, "removed_words": v[0], "total_words": v[1],
         "removed_share": v[0] / v[1] if v[1] else 0.0}
        for s, v in per_show_removed.items()
    ]).sort_values("removed_share", ascending=False)
    show_df.to_csv(OUT_DIR / "removed_share_per_show.csv", index=False)

    summary = {
        "model": state["model"],
        "n_windows_returned": len(by_window),
        "n_batch_error_lines": n_error_lines,
        "n_parse_errors": n_parse_errors,
        "n_spans_raw": n_spans_raw,
        "n_spans_merged": n_spans_merged,
        "n_episodes": len(episodes),
        "n_words_before": n_words_before,
        "n_words_after": n_words_after,
        "removed_share_overall": (n_words_before - n_words_after) / n_words_before,
        "tokens_in": tok_in, "tokens_out": tok_out,
        "actual_batch_cost_usd": actual,
        "per_show_removed_share_describe": show_df["removed_share"].describe().to_dict(),
        "shows_above_30pct": show_df[show_df["removed_share"] > 0.30]["show_id"].tolist(),
    }
    (OUT_DIR / "phase_b_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n[collect] merged spans: {n_spans_merged}")
    print(f"[collect] words {n_words_before:,} -> {n_words_after:,} "
          f"({summary['removed_share_overall']:.2%} removed)")
    print(f"[collect] actual batch cost ${actual:.2f}")
    print(f"\n[collect] removed-share per show:\n{show_df['removed_share'].describe()}")
    print(f"\n[collect] top 10 shows by removed share:")
    print(show_df.head(10).to_string(index=False))
    flag = show_df[show_df["removed_share"] > 0.30]
    print(f"\n[collect] shows above 30% removed (INSPECT, do not trust blindly): {len(flag)}")
    print(f"[collect] -> {CLEANED_PATH}, {AUDIT_PATH}, phase_b_summary.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("segment"); s.set_defaults(fn=cmd_segment)
    s.add_argument("--n-process", type=int, default=10)
    s.add_argument("--force", action="store_true")

    s = sub.add_parser("build"); s.set_defaults(fn=cmd_build)
    s.add_argument("--model", default=DEFAULT_MODEL)

    s = sub.add_parser("submit"); s.set_defaults(fn=cmd_submit)
    s.add_argument("--retry-wait", type=int, default=600,
                    help="seconds to wait before retrying a queue-full submission")

    s = sub.add_parser("run-waves"); s.set_defaults(fn=cmd_run_waves)
    s.add_argument("--poll-interval", type=int, default=30)
    s.add_argument("--keep-going", action="store_true",
                    help="skip a wave that ends non-completed instead of stopping")

    for name, fn in (("poll", cmd_poll), ("collect", cmd_collect)):
        s = sub.add_parser(name); s.set_defaults(fn=fn)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
