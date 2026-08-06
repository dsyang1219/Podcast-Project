"""Batch-API version of the guest-extraction cascade, for finishing the
remaining corpus at ~50% of the synchronous Haiku cost (the Message Batches
API discount applies to all input and output tokens, no beta header needed).

    .venv/bin/python -m nlp.extract_guests_batch [--poll-interval 60]

Two sequential batch rounds preserve the tier2->tier3 fallback's cost saving
(round 2 only pays for episodes round 1 didn't resolve) at the cost of extra
wall-clock (each batch round can take up to 24h, typically ~1h) versus the
live worker pool in nlp.extract_guests.

Resumable by construction: the live checkpoint (guests_raw_checkpoint.jsonl)
is untouched until BOTH rounds finish, so "todo" (episodes not yet in the
checkpoint) stays fixed for the whole run -- interrupting and re-running this
script just re-polls/re-fetches the same batch IDs from batch_state.json
rather than resubmitting or reprocessing anything. Batch results live on
Anthropic's side for 29 days, so nothing about round-1 classification needs
to be cached locally either.

Episodes whose batch request itself errored/canceled/expired (infra failure,
not a billing rejection) are deliberately EXCLUDED from the checkpoint at the
end -- writing them in as tier="error" is exactly what silently orphaned
65k+ episodes last time (they looked "done" to --resume and were never
retried). They're reported in the summary instead so a rerun of this script
naturally retries them (they're still "todo" next time).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from rapidfuzz import fuzz

from pipeline import config as pipeline_config
from .extract_guests import (
    GUEST_SCHEMA,
    INPUT_PRICE_PER_MTOK,
    MODEL,
    OUTPUT_PRICE_PER_MTOK,
    SYSTEM_PROMPT,
    _load_checkpoint,
    build_user_prompt,
    canonicalize,
    load_approved_patterns,
    strip_hosts,
    tier1_regex_guests,
    verify_evidence,
)
from .guest_common import clean_description, load_episodes, load_hosts, strip_html_urls

BATCH_MAX = 80_000  # requests per Message Batch. Hard API limit is 100,000 or
# 256MB; description-fallback requests run ~194MB estimated at 100k requests
# (long cleaned descriptions), so 80k leaves real headroom under the byte cap
# rather than riding right at the edge of it on a real production run.
BATCH_DISCOUNT = 0.5

OUT = pipeline_config.OUTPUT_DIR
STATE_PATH = OUT / "batch_state.json"
CHECKPOINT_PATH = OUT / "guests_raw_checkpoint.jsonl"


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def _save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_PATH)


def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_request(episode_id: str, kind: str, text: str) -> Request:
    return Request(
        custom_id=episode_id,
        params=MessageCreateParamsNonStreaming(
            model=MODEL,
            max_tokens=1024,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(kind, text)}],
            output_config={"format": {"type": "json_schema", "schema": GUEST_SCHEMA}},
        ),
    )


def submit_batches(client: anthropic.Anthropic, requests: list[Request], label: str) -> list[str]:
    batch_ids = []
    for chunk in _chunk(requests, BATCH_MAX):
        batch = client.messages.batches.create(requests=chunk)
        print(f"[{label}] submitted batch {batch.id} ({len(chunk)} requests)")
        batch_ids.append(batch.id)
    return batch_ids


def poll_batches(client: anthropic.Anthropic, batch_ids: list[str], label: str, poll_interval: int) -> None:
    pending = set(batch_ids)
    while pending:
        for bid in list(pending):
            batch = client.messages.batches.retrieve(bid)
            if batch.processing_status == "ended":
                print(f"[{label}] {bid} ended -- "
                      f"succeeded={batch.request_counts.succeeded} errored={batch.request_counts.errored} "
                      f"canceled={batch.request_counts.canceled} expired={batch.request_counts.expired}")
                pending.discard(bid)
            else:
                print(f"[{label}] {bid} {batch.processing_status} "
                      f"(processing={batch.request_counts.processing})")
        if pending:
            time.sleep(poll_interval)


def fetch_results(client: anthropic.Anthropic, batch_ids: list[str]) -> dict[str, object]:
    """episode_id -> batch result object, across all batches for this round."""
    out = {}
    for bid in batch_ids:
        for result in client.messages.batches.results(bid):
            out[result.custom_id] = result
    return out


def parse_message_result(result) -> tuple[dict | None, int, int]:
    """Returns (parsed_json_or_None, input_tokens, output_tokens). None means
    this episode's request failed -- either at the infra level (not a content
    refusal) or because the response was truncated/malformed JSON (e.g. a
    long guest list hitting max_tokens=1024 mid-string). Real tokens were
    still billed in the latter case, so usage is still counted and returned
    even though the episode itself gets excluded and retried later --
    mirrors nlp.extract_guests.process_one's json.JSONDecodeError handling,
    which exists for exactly this reason (a single bad response must not
    propagate and abort everything else that succeeded)."""
    if result.result.type != "succeeded":
        return None, 0, 0
    msg = result.result.message
    usage = (msg.usage.input_tokens, msg.usage.output_tokens)
    if msg.stop_reason == "refusal":
        return {"has_guests": False, "guests": []}, *usage
    text_block = next((b.text for b in msg.content if b.type == "text"), None)
    if not text_block:
        return {"has_guests": False, "guests": []}, *usage
    try:
        return json.loads(text_block), *usage
    except json.JSONDecodeError:
        return None, *usage


def run(poll_interval: int, limit: int | None, seed: int) -> None:
    client = anthropic.Anthropic()
    state = _load_state()

    eps = load_episodes()
    n_total = len(eps)
    done = _load_checkpoint(CHECKPOINT_PATH)
    todo_ids = [eid for eid in eps["episode_id"] if eid not in done]
    if limit and not state:
        # only subsample on a fresh run -- once batches are submitted, todo_ids
        # must stay whatever was actually sent, or resume logic breaks
        import random
        rng = random.Random(seed)
        todo_ids = rng.sample(todo_ids, min(limit, len(todo_ids)))
    print(f"[data] {len(done)} already done, {len(todo_ids)} todo out of {n_total} total corpus"
          + (f" (--limit {limit} sample)" if limit else ""))

    if not todo_ids:
        print("[data] nothing to do")
        return

    eps_by_id = {row["episode_id"]: row for _, row in eps.iterrows()}
    hosts_by_show = load_hosts()
    approved_patterns = load_approved_patterns()

    # ---- Tier 1 (free regex) -- resolved without any API call ----
    tier1_records: dict[str, dict] = {}
    needs_round1: list[str] = []
    for eid in todo_ids:
        row = eps_by_id[eid]
        pat_entry = approved_patterns.get(row["collection_id"])
        names = tier1_regex_guests(row["episode_title"] or "", pat_entry["best_template"]) if pat_entry else None
        if names:
            tier1_records[eid] = {
                "guests": [{"name": n, "descriptor": "", "topic": "", "evidence": row["episode_title"], "confidence": "high"} for n in names],
                "tier": "tier1_regex",
            }
        else:
            needs_round1.append(eid)
    print(f"[tier1] {len(tier1_records)} resolved by free regex, {len(needs_round1)} need round 1")

    usage = Counter()  # input_tokens, output_tokens across both rounds

    # ---- Round 1: title ----
    if "round1_batch_ids" not in state:
        requests = [build_request(eid, "title", eps_by_id[eid]["episode_title"] or "") for eid in needs_round1]
        state["round1_batch_ids"] = submit_batches(client, requests, "round1")
        state["round1_episode_ids"] = needs_round1
        _save_state(state)
    poll_batches(client, state["round1_batch_ids"], "round1", poll_interval)
    round1_results = fetch_results(client, state["round1_batch_ids"])

    round2_needed: list[str] = []
    round1_records: dict[str, dict] = {}
    round1_failed: set[str] = set()
    for eid in state["round1_episode_ids"]:
        result = round1_results.get(eid)
        if result is None:
            round1_failed.add(eid)
            continue
        parsed, in_tok, out_tok = parse_message_result(result)
        usage["input"] += in_tok
        usage["output"] += out_tok
        if parsed is None:
            round1_failed.add(eid)
            continue
        row = eps_by_id[eid]
        verify_text = f"{row['episode_title'] or ''} {strip_html_urls(row['episode_description'] if isinstance(row['episode_description'], str) else '')}"
        guests = parsed.get("guests", []) if parsed.get("has_guests") else []
        kept, _dropped = verify_evidence(guests, verify_text)
        if kept:
            round1_records[eid] = {"guests": kept, "tier": "tier2_title"}
        else:
            round2_needed.append(eid)
    print(f"[round1] {len(round1_records)} resolved by title, {len(round2_needed)} need round 2, "
          f"{len(round1_failed)} failed at batch level")

    # ---- Round 2: description fallback ----
    if round2_needed and "round2_batch_ids" not in state:
        requests = []
        for eid in round2_needed:
            row = eps_by_id[eid]
            raw_desc = row["episode_description"] if isinstance(row["episode_description"], str) else ""
            requests.append(build_request(eid, "description", clean_description(raw_desc)))
        state["round2_batch_ids"] = submit_batches(client, requests, "round2")
        state["round2_episode_ids"] = round2_needed
        _save_state(state)

    round2_records: dict[str, dict] = {}
    round2_failed: set[str] = set()
    if state.get("round2_batch_ids"):
        poll_batches(client, state["round2_batch_ids"], "round2", poll_interval)
        round2_results = fetch_results(client, state["round2_batch_ids"])
        for eid in state["round2_episode_ids"]:
            result = round2_results.get(eid)
            if result is None:
                round2_failed.add(eid)
                continue
            parsed, in_tok, out_tok = parse_message_result(result)
            usage["input"] += in_tok
            usage["output"] += out_tok
            if parsed is None:
                round2_failed.add(eid)
                continue
            row = eps_by_id[eid]
            verify_text = f"{row['episode_title'] or ''} {strip_html_urls(row['episode_description'] if isinstance(row['episode_description'], str) else '')}"
            guests = parsed.get("guests", []) if parsed.get("has_guests") else []
            kept, _dropped = verify_evidence(guests, verify_text)
            round2_records[eid] = {"guests": kept, "tier": "tier3_description" if kept else "none"}
    print(f"[round2] {len(round2_records)} processed, {len(round2_failed)} failed at batch level")

    # ---- Assemble final records (host-strip, same as the synchronous cascade) ----
    all_failed = round1_failed | round2_failed
    new_records: dict[str, dict] = {}
    for eid in todo_ids:
        if eid in all_failed:
            continue
        row = eps_by_id[eid]
        if eid in tier1_records:
            src = tier1_records[eid]
        elif eid in round1_records:
            src = round1_records[eid]
        elif eid in round2_records:
            src = round2_records[eid]
        else:
            continue  # was never routed anywhere -- shouldn't happen, skip defensively
        guests, n_stripped = strip_hosts(src["guests"], hosts_by_show.get(row["collection_id"], set()))
        new_records[eid] = {
            "episode_id": eid, "collection_id": row["collection_id"], "show_name": row["show_name"],
            "episode_title": row["episode_title"] or "", "tier": src["tier"],
            "n_guests": len(guests), "guests": guests,
            "evidence_dropped": 0, "hosts_stripped": n_stripped,
        }

    print(f"[assemble] {len(new_records)} new records, {len(all_failed)} excluded (will be retried on next run)")

    # ---- Merge with existing checkpoint, canonicalize over the full population, write outputs ----
    merged = {**done, **new_records}
    all_raw_names = [g["name"] for rec in merged.values() for g in rec["guests"]]
    canonical_map = canonicalize(all_raw_names)
    for rec in merged.values():
        for g in rec["guests"]:
            g["guest_canonical"] = canonical_map.get(g["name"], g["name"])

    with CHECKPOINT_PATH.open("a") as f:
        for rec in new_records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_path = OUT / "guests_raw.jsonl"
    with out_path.open("w") as f:
        for eid in eps["episode_id"]:
            if eid in merged:
                f.write(json.dumps(merged[eid], ensure_ascii=False) + "\n")

    cost = (usage["input"] / 1e6 * INPUT_PRICE_PER_MTOK + usage["output"] / 1e6 * OUTPUT_PRICE_PER_MTOK) * BATCH_DISCOUNT
    tier_counts = Counter(rec["tier"] for rec in merged.values())
    summary = {
        "model": MODEL, "batch_discount_applied": BATCH_DISCOUNT,
        "n_episodes_total_corpus": n_total, "n_done_total": len(merged),
        "n_new_this_run": len(new_records), "n_failed_excluded": len(all_failed),
        "tier_counts": dict(tier_counts),
        "input_tokens_this_run": usage["input"], "output_tokens_this_run": usage["output"],
        "cost_this_run_usd": round(cost, 4),
    }
    (OUT / "guests_extraction_batch_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if all_failed:
        print(f"\n[note] {len(all_failed)} episodes failed at the batch-infra level and were "
              f"deliberately left out of the checkpoint -- rerun this script to retry them.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poll-interval", type=int, default=60, help="seconds between batch status checks")
    ap.add_argument("--limit", type=int, default=None, help="sample this many todo episodes instead of all (dry run)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.poll_interval, args.limit, args.seed)


if __name__ == "__main__":
    main()
