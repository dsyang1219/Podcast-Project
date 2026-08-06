"""Guest-extraction cascade (Pew/Stocking & Odabas 2024, DeMets & Spiro 2025
Sec 3.2 methodology), replicated with Claude Haiku instead of GPT-4 for cost.

    .venv/bin/python -m nlp.extract_guests [--limit 200] [--seed 0] [--concurrency 20] [--resume]

Full-corpus runs use a thread pool (--concurrency, default 20 -- I/O-bound
Haiku calls, not CPU-bound) and checkpoint every completed episode to
guests_raw_checkpoint.jsonl as it finishes, so a multi-hour run can be
resumed with --resume after an interruption instead of re-paying for
already-extracted episodes. Canonicalization still runs once at the end
over the full name population (unchanged -- fuzzy clustering needs it all).

Model: claude-haiku-4-5-20251001, temperature 0. Pew/DeMets used GPT-family
models -- comparability to their numbers can't be inferred from "same model
family", so quality is established by nlp/validate_guests.py against a
hand-coded sample, not assumed here.

Three tiers, cheapest first:
  Tier 1 (free, regex) -- ONLY for shows with an approved entry in
    data/output/show_patterns.json (approved:true, set by a human after
    reviewing nlp/discover_patterns.py's output). Currently zero shows are
    approved -- discover_patterns.py deliberately never auto-approves, so
    until a human reviews it, Tier 1 contributes 0 extractions and every
    episode falls through to Tier 2. This is intentional, not a bug.
  Tier 2 (Haiku, title) -- for everything Tier 1 didn't handle.
  Tier 3 (Haiku, cleaned description) -- ONLY when Tier 2 returned no
    guests (has_guests=False, or all its guests failed evidence
    verification), per Pew's title-first fallback order.

Evidence verification (hallucination filter): after every LLM call, each
guest's `evidence` span must fuzzy-match (rapidfuzz partial_ratio >= 88)
somewhere in title + ORIGINAL (HTML-stripped, non-ad-truncated) description.
A guest whose evidence can't be found is dropped -- a hallucinated name
can't quote the source text it was supposedly extracted from.

Host stripping: any surviving guest whose normalized name matches a known
host of that show (data/output/host_dime_lookup_v2.csv, see
nlp/guest_common.py's docstring for why that file and not corpus.csv) is
removed -- a host is not a guest.

Canonicalization runs once, globally, after all episodes are processed
(fuzzy matching needs the full name population, not a per-episode view):
block on last name, rapidfuzz token_set_ratio>=92 within block (first-name
nickname-normalized before comparing), canonical label = most frequent raw
surface form in the resulting cluster.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from rapidfuzz import fuzz

from pipeline import config as pipeline_config
from .discover_patterns import TEMPLATES
from .guest_common import (
    clean_description,
    load_episodes,
    load_hosts,
    nickname_normalize,
    normalize_name,
    strip_html_urls,
)

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
INPUT_PRICE_PER_MTOK = 1.00
OUTPUT_PRICE_PER_MTOK = 5.00

EVIDENCE_MIN_RATIO = 88
CANON_MIN_RATIO = 92

GUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "has_guests": {"type": "boolean"},
        "guests": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "descriptor": {"type": "string"},
                    "topic": {"type": "string"},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["name", "descriptor", "topic", "evidence", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["has_guests", "guests"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are an information extractor for an academic study of political podcasts.

Your ONLY job: identify people who APPEAR AS GUESTS in this episode -- i.e. people who are interviewed, who co-host this specific episode as an invited participant, or who are explicitly described as joining/appearing on the show.

Do NOT extract:
- Hosts (the show's regular, permanent host(s))
- People who are merely DISCUSSED, MENTIONED, or REFERENCED but do not appear (e.g. "Trump announced..." does not make Trump a guest)
- Sponsors, advertisers, or promotional partners
- Producers, editors, or other behind-the-scenes crew not appearing as a speaking guest

Draw only on the text given to you. Do not infer identity, role, or presence from outside knowledge -- if the text doesn't clearly show someone appearing as a guest, don't extract them, even if you recognize the name from elsewhere.

For every guest you extract, `evidence` MUST be a verbatim (exact substring) span copied from the input text that supports them being a guest. Do not paraphrase or summarize the evidence.

If there are no guests, return has_guests=false and an empty guests array."""


def build_user_prompt(kind: str, text: str) -> str:
    label = "episode title" if kind == "title" else "episode description"
    return f"Here is the {label} of a podcast episode. Extract any guests per the system instructions.\n\n{label.upper()}:\n{text}"


class Usage:
    """Thread-safe: extract_for_episode runs concurrently across a thread pool
    for full-corpus runs, and every call hits .add() from a worker thread."""
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def add(self, resp) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += resp.usage.input_tokens
            self.output_tokens += resp.usage.output_tokens

    def cost(self) -> float:
        with self._lock:
            return (self.input_tokens / 1e6) * INPUT_PRICE_PER_MTOK + (self.output_tokens / 1e6) * OUTPUT_PRICE_PER_MTOK


def call_haiku(client: anthropic.Anthropic, kind: str, text: str, usage: Usage) -> dict:
    if not text.strip():
        return {"has_guests": False, "guests": []}
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(kind, text)}],
        output_config={"format": {"type": "json_schema", "schema": GUEST_SCHEMA}},
    )
    usage.add(resp)
    if resp.stop_reason == "refusal":
        return {"has_guests": False, "guests": []}
    text_block = next((b.text for b in resp.content if b.type == "text"), None)
    if not text_block:
        return {"has_guests": False, "guests": []}
    return json.loads(text_block)


def verify_evidence(guests: list[dict], verify_text: str) -> tuple[list[dict], int]:
    kept, dropped = [], 0
    for g in guests:
        ev = g.get("evidence", "")
        if ev and fuzz.partial_ratio(ev, verify_text) >= EVIDENCE_MIN_RATIO:
            kept.append(g)
        else:
            dropped += 1
    return kept, dropped


def tier1_regex_guests(title: str, template: str) -> list[str] | None:
    import re
    pattern = TEMPLATES.get(template)
    if pattern is None:
        return None
    m = pattern.search(title)
    if not m:
        return None
    captured = m.group(1)
    parts = re.split(r"\s*(?:&|,|\band\b)\s*", captured)
    names = [p.strip() for p in parts if p.strip()]
    return names or None


def load_approved_patterns() -> dict[str, dict]:
    path = pipeline_config.OUTPUT_DIR / "show_patterns.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for sid, entry in data.get("shows", {}).items():
        if entry.get("approved") is True:
            out[sid] = entry
    return out


def strip_hosts(guests: list[dict], host_set: set[str]) -> tuple[list[dict], int]:
    kept, stripped = [], 0
    for g in guests:
        if normalize_name(g.get("name", "")) in host_set:
            stripped += 1
        else:
            kept.append(g)
    return kept, stripped


def canonicalize(all_names: list[str]) -> dict[str, str]:
    freq = Counter(all_names)
    unique = list(freq.keys())

    def block_key(name: str) -> str:
        toks = normalize_name(name).split()
        return toks[-1] if toks else normalize_name(name)

    def match_key(name: str) -> str:
        toks = normalize_name(name).split()
        if toks:
            toks[0] = nickname_normalize(toks[0])
        return " ".join(toks)

    blocks: dict[str, list[str]] = defaultdict(list)
    for n in unique:
        blocks[block_key(n)].append(n)

    canonical_map: dict[str, str] = {}
    for _, block_names in blocks.items():
        block_names.sort(key=lambda n: -freq[n])
        clusters: list[list[str]] = []
        for n in block_names:
            mk = match_key(n)
            placed = False
            for cluster in clusters:
                if fuzz.token_set_ratio(mk, match_key(cluster[0])) >= CANON_MIN_RATIO:
                    cluster.append(n)
                    placed = True
                    break
            if not placed:
                clusters.append([n])
        for cluster in clusters:
            canon = max(cluster, key=lambda n: freq[n])
            for n in cluster:
                canonical_map[n] = canon
    return canonical_map


def extract_for_episode(client: anthropic.Anthropic, usage: Usage, row, approved_patterns: dict,
                         hosts_by_show: dict) -> dict:
    """Run the full Tier1->Tier2->Tier3->evidence-verify->host-strip cascade
    for one episode row. Returns tier, guests (pre-canonicalization), and
    per-episode evidence/host-strip counts. Shared by nlp.extract_guests's
    main run() and nlp.validate_guests (so validation predictions are
    produced by the IDENTICAL cascade, not a reimplementation)."""
    title = row["episode_title"] or ""
    raw_desc = row["episode_description"] if isinstance(row["episode_description"], str) else ""
    cleaned_desc = clean_description(raw_desc)
    verify_text = f"{title} {strip_html_urls(raw_desc)}"
    show_id = row["collection_id"]

    tier = "none"
    guests: list[dict] = []
    evidence_dropped = 0

    pat_entry = approved_patterns.get(show_id)
    if pat_entry:
        names = tier1_regex_guests(title, pat_entry["best_template"])
        if names:
            guests = [{"name": n, "descriptor": "", "topic": "", "evidence": title, "confidence": "high"} for n in names]
            tier = "tier1_regex"

    if tier == "none":
        t2_result = call_haiku(client, "title", title, usage)
        t2_guests = t2_result.get("guests", []) if t2_result.get("has_guests") else []
        t2_kept, t2_dropped = verify_evidence(t2_guests, verify_text)
        evidence_dropped += t2_dropped
        if t2_kept:
            guests, tier = t2_kept, "tier2_title"
        else:
            t3_result = call_haiku(client, "description", cleaned_desc, usage)
            t3_guests = t3_result.get("guests", []) if t3_result.get("has_guests") else []
            t3_kept, t3_dropped = verify_evidence(t3_guests, verify_text)
            evidence_dropped += t3_dropped
            if t3_kept:
                guests, tier = t3_kept, "tier3_description"

    guests, n_stripped = strip_hosts(guests, hosts_by_show.get(show_id, set()))

    return {
        "episode_id": row["episode_id"], "collection_id": show_id, "show_name": row["show_name"],
        "episode_title": title, "tier": tier, "guests": guests,
        "evidence_dropped": evidence_dropped, "hosts_stripped": n_stripped,
    }


def _load_checkpoint(path: Path) -> dict[str, dict]:
    """episode_id -> record, for resuming an interrupted full-corpus run
    without re-paying for episodes already extracted."""
    if not path.exists():
        return {}
    done = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[rec["episode_id"]] = rec
    return done


def run(limit: int | None, seed: int, concurrency: int, resume: bool) -> None:
    OUT = pipeline_config.OUTPUT_DIR
    # explicit timeout matters at high concurrency: the SDK default (~10 min)
    # means one abnormally slow request (e.g. a "grammar compilation timed
    # out" case on structured output) can tie up a worker+connection for a
    # very long time with nothing to cut it loose, and enough of those in
    # flight at once can gridlock the whole pool. A short-ish per-request
    # timeout lets max_retries actually retry instead of just waiting.
    client = anthropic.Anthropic(max_retries=6, timeout=45.0)
    usage = Usage()

    eps = load_episodes()
    n_total = len(eps)
    if limit:
        eps = eps.sample(n=min(limit, len(eps)), random_state=seed).reset_index(drop=True)
    print(f"[data] processing {len(eps)} of {n_total} total episodes  (concurrency={concurrency})")

    hosts_by_show = load_hosts()
    approved_patterns = load_approved_patterns()
    print(f"[patterns] {len(approved_patterns)} shows have an APPROVED Tier-1 regex "
          f"(0 expected until a human reviews show_patterns.json)")

    checkpoint_path = OUT / "guests_raw_checkpoint.jsonl"
    done = _load_checkpoint(checkpoint_path) if resume else {}
    if done:
        print(f"[resume] {len(done)} episodes already in checkpoint, skipping")
    todo = [row for _, row in eps.iterrows() if row["episode_id"] not in done]

    tier_counts = Counter(rec["tier"] for rec in done.values())
    hosts_stripped_total = sum(rec.get("hosts_stripped", 0) for rec in done.values())
    evidence_dropped_total = sum(rec.get("evidence_dropped", 0) for rec in done.values())
    n_completed = len(done)

    write_lock = threading.Lock()
    progress_lock = threading.Lock()
    checkpoint_f = checkpoint_path.open("a")

    def process_one(row):
        try:
            return extract_for_episode(client, usage, row, approved_patterns, hosts_by_show)
        except (anthropic.APIError, json.JSONDecodeError) as e:
            # json.JSONDecodeError happens when a response is truncated/malformed
            # (e.g. cut off mid-string) -- without catching it here alongside
            # APIError, a single bad response propagates through fut.result()
            # in run() and kills the entire thread pool, not just this episode.
            print(f"[warn] episode {row['episode_id']} failed ({type(e).__name__}: {e}); marking tier=error")
            return {"episode_id": row["episode_id"], "collection_id": row["collection_id"],
                     "show_name": row["show_name"], "episode_title": row["episode_title"] or "",
                     "tier": "error", "guests": [], "evidence_dropped": 0, "hosts_stripped": 0}

    t0 = time.perf_counter()
    if todo:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(process_one, row) for row in todo]
            for fut in as_completed(futures):
                result = fut.result()
                tier, guests = result["tier"], result["guests"]
                rec = {"episode_id": result["episode_id"], "collection_id": result["collection_id"],
                        "show_name": result["show_name"], "episode_title": result["episode_title"],
                        "tier": tier, "n_guests": len(guests), "guests": guests,
                        "evidence_dropped": result["evidence_dropped"], "hosts_stripped": result["hosts_stripped"]}
                with write_lock:
                    checkpoint_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    checkpoint_f.flush()
                    done[rec["episode_id"]] = rec

                with progress_lock:
                    tier_counts[tier] += 1
                    hosts_stripped_total += result["hosts_stripped"]
                    evidence_dropped_total += result["evidence_dropped"]
                    n_completed += 1
                    if n_completed % 25 == 0 or n_completed == len(eps):
                        elapsed = time.perf_counter() - t0
                        print(f"[progress] {n_completed}/{len(eps)}  tiers={dict(tier_counts)}  "
                              f"cost so far=${usage.cost():.4f}  ({elapsed:.1f}s elapsed this run)")
    checkpoint_f.close()

    # canonicalization needs the full name population -- runs once at the end
    # over the complete checkpoint (this run's new episodes + any resumed ones)
    episode_records = [done[eid] for eid in eps["episode_id"] if eid in done]
    all_raw_names = [g["name"] for rec in episode_records for g in rec["guests"]]

    canonical_map = canonicalize(all_raw_names)
    for rec in episode_records:
        for g in rec["guests"]:
            g["guest_canonical"] = canonical_map.get(g["name"], g["name"])

    out_path = OUT / "guests_raw.jsonl"
    with out_path.open("w") as f:
        for rec in episode_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_with_guests = sum(1 for r in episode_records if r["n_guests"] > 0)
    unique_raw = len(set(all_raw_names))
    unique_canonical = len(set(canonical_map.values()))
    # per-episode rate must come from len(todo) (episodes actually paid for THIS
    # run), not len(eps) -- on a --resume run those differ and len(eps) would
    # understate the rate (checkpoint-resumed episodes cost nothing this run).
    est_full_corpus_cost = usage.cost() / len(todo) * n_total if todo else 0.0

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "n_episodes_processed": len(eps),
        "n_episodes_total_corpus": n_total,
        "pct_with_guests": round(100 * n_with_guests / len(eps), 1) if len(eps) else 0,
        "n_unique_guests_raw": unique_raw,
        "n_unique_guests_canonical": unique_canonical,
        "tier_counts": dict(tier_counts),
        "hosts_stripped": hosts_stripped_total,
        "evidence_dropped": evidence_dropped_total,
        "api_calls": usage.calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_this_run_usd": round(usage.cost(), 4),
        "est_full_corpus_cost_usd": round(est_full_corpus_cost, 2),
        "n_shows_with_approved_tier1_pattern": len(approved_patterns),
    }
    summary_path = OUT / "guests_extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n[output] -> {out_path.name}")
    print(f"[summary] -> {summary_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=20,
                     help="parallel Haiku calls (I/O-bound; not CPU-bound, so this can exceed core count -- "
                          "the SDK's own retry/backoff absorbs rate-limit pressure)")
    ap.add_argument("--resume", action="store_true",
                     help="skip episodes already in guests_raw_checkpoint.jsonl from a prior interrupted run")
    args = ap.parse_args()
    run(args.limit, args.seed, args.concurrency, args.resume)


if __name__ == "__main__":
    main()
