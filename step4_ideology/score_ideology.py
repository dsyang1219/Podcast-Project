#!/usr/bin/env python3
"""Score transcript chunks for political content and ideological lean.

    ./score_ideology.py --pilot                 # the 3,386-chunk pilot set
    ./score_ideology.py --input FILE --out FILE
    ./score_ideology.py --coding-sheet --n 300  # blind sheet for hand validation

Implements Much et al. (2026) Appendix I Tasks 5 and 6 verbatim -- see
step4_ideology/prompts/ideology.md, which is the canonical copy of the wording. The prompts are
duplicated here only so the script runs standalone; if the two ever disagree,
the markdown file is correct.

Two-stage gate
--------------
Stage 1 asks whether a chunk is political at all. Only chunks answering "Yes"
reach stage 2, which scores direction on a 7-point scale. Chunks failing the
gate score 0 and are never sent onward. Much et al. adopted this after finding
a single combined scale made the model overuse "moderate" far more than human
coders did -- the gate forces a commitment.

It also handles advertising for free: a mattress read is not political, fails
stage 1, and never reaches the ideology scorer. Measured on this corpus, 2.2% of
750-char chunks carry ad markers and ~17% of those are politically themed, so
the residue that survives the gate is ~0.4% of all chunks.

Model
-----
Default is a DATED snapshot, never a `-latest` alias: an alias silently becomes
a different model, and for a published measure the model is part of the
instrument. Mini tier is deliberate -- Much et al. used Gemini 3 Flash Preview,
a fast/cheap tier, and reached kappa = 0.834 on directional ideology against a
human agreement ceiling of alpha = 0.556. Their own cross-model check (Gemini 3
Flash vs 2.5 Flash Lite, r = 0.833 at podcast level) shows the task is robust to
tier. Reasoning models are a poor fit: chain-of-thought buys nothing on a fixed
rubric, costs reasoning tokens, and can talk itself out of the straightforward
application the codebook asks for.

The 7-point output is RETAINED even though the published main analysis collapses
to ternary. This design tracks within-show change over time, and shows shift
intensity far more often than they flip sign; a ternary collapse would be blind
to the variation being tested. Both are emitted.
"""
import argparse, csv, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repository root (this script lives one folder down)
MODEL = "gpt-5.4-mini-2026-03-17"

SYSTEM = ("You are a research assistant labeling a podcast transcript segment.\n"
          "Follow instructions carefully. Return only valid JSON. No extra text.")

POLITICS = """Podcast Labelling Instructions

Task: Politics Detector (stage 1)

Decide whether the TARGET segment is about politics, political issues, or
political figures. This Yes/No decision is the detector gate for ideology
stage 2:
- "Yes" segments proceed to ideology direction/intensity scoring.
- "No" segments are not sent to ideology stage 2.

Segments about politics can include discussion about how we govern ourselves,
distribute power, or resolve collective questions. This includes both formal
political institutions and social controversies that have become matters of
public debate.

A segment is political if it addresses things like:
- Government, legislation, public institutions, elections, or political figures.
- Public policies (e.g., taxes, healthcare, civil rights).
- Organized activities by the public to influence politics (e.g., protests,
  boycotts, writing letters to representatives, campaign donations).
- Geopolitical or international conflicts.
- Politically salient social and cultural issues debated in the public sphere,
  such as racism, sexism, LGBTQ rights, immigration, public health mandates,
  religion in public life, regulating education curricula, or free speech.

Context:
{CONTEXT}

TARGET SEGMENT:
{TRANSCRIPT}

Is this about politics? Choose exactly one: Yes, No"""

IDEOLOGY = """Podcast Labelling Instructions

Task: Ideology (stage 2, politics-gated)

This segment has already been labeled political by the Politics Yes/No detector.
This step is only run for segments where politics == "Yes".

Rate the ideological leaning expressed in the TARGET segment.
Use the speaker's stance, not the topic itself.
Do NOT treat quoted, reported, or mocked speech as the speaker's own position
unless the speaker endorses it. A host who ventriloquises or ridicules the other
side's argument is expressing THEIR OWN stance, which is usually the opposite of
the words being voiced.
If the segment is purely descriptive, balanced, mixed between liberal and
conservative cues, or reporting without endorsement (e.g., a straight news
segment about politics), choose "moderate".

Lean cues (U.S. context):
- Liberal: support for redistribution, social safety nets, civil rights
  expansion, regulation, climate action, gun control, reproductive rights.
- Conservative: support for limited government, lower taxes, traditional
  social norms, deregulation, strong law-and-order, gun rights.

Intensity:
- slightly: mild or single cue
- moderately: multiple cues
- strongly: dominant, emphatic stance
- moderate: neither clearly liberal nor clearly conservative, including
  neutral political reporting or mixed cues without a directional stance

Context (optional, only for disambiguation):
{CONTEXT}

TARGET SEGMENT:
{TRANSCRIPT}

Choose exactly one:
strongly_conservative, moderately_conservative, slightly_conservative,
moderate,
slightly_liberal, moderately_liberal, strongly_liberal"""

IDEO_VALUES = ["strongly_conservative", "moderately_conservative", "slightly_conservative",
               "moderate", "slightly_liberal", "moderately_liberal", "strongly_liberal"]
TERNARY = {"strongly_conservative": 1, "moderately_conservative": 1, "slightly_conservative": 1,
           "moderate": 0,
           "slightly_liberal": -1, "moderately_liberal": -1, "strongly_liberal": -1}

_lock = threading.Lock()

# Self-pace INTO the token-per-minute ceiling instead of bouncing off it.
# The account allows 200,000 TPM; each call costs ~900 prompt tokens plus the
# 32 reserved for output, so ~215 calls/min is the hard ceiling. Without pacing,
# every worker hits 429 at once, the SDK backs them all off on the same
# schedule, and when the budget refills they fire simultaneously and exhaust it
# again -- a thundering herd that produced ZERO completed rows in two minutes.
TPM_LIMIT = 200_000
# 560, not 950. Measured over 78 real calls: 517 input + 13 output tokens each
# (40,326 in / 1,048 out across 40 chunks). 950 was a guess made before the
# prompt was final, and it throttled the run to 183 calls/min against a TPM
# limit the run never approached. 560 keeps ~8% headroom over the measurement
# and still sits well under the 500 RPM request ceiling.
TOKENS_PER_CALL = 560
MIN_INTERVAL = 60.0 / (TPM_LIMIT / TOKENS_PER_CALL) * 1.15   # 15% headroom
_pace = threading.Lock()
_last = [0.0]


def pace():
    with _pace:
        dt = time.monotonic() - _last[0]
        if dt < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - dt)
        _last[0] = time.monotonic()


def load_env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            m = re.match(r"^([A-Z_]+)=(.*)$", line.strip())
            if m:
                os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))


def schema(name, values):
    """Structured output. The API enforces the label set, so a malformed or
    out-of-vocabulary answer is impossible rather than something to parse for."""
    return {"type": "json_schema", "json_schema": {
        "name": name, "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "required": [name],
                   "properties": {name: {"type": "string", "enum": values}}}}}


# ── cost guard ─────────────────────────────────────────────────────────────
# Priced from the API's own `usage` field, not from an estimate. A hard ceiling
# matters here because the run is on a fixed weekly budget: if the real rate
# differs from the assumed one, the guard stops the run rather than the invoice
# discovering it. Halting is safe -- cmd_score resumes from the output file, so
# nothing already paid for is rescored.
PRICE_IN, PRICE_OUT = 0.25 / 1e6, 2.00 / 1e6      # $/token, gpt-5.4-mini
_usage = {"in": 0, "out": 0}
_halt = threading.Event()


def cost() -> float:
    return _usage["in"] * PRICE_IN + _usage["out"] * PRICE_OUT


def ask(client, model, prompt, key, values, tries=5):
    if _halt.is_set():
        return None
    backoff = 4.0
    for k in range(tries):
        pace()
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}],
                response_format=schema(key, values),
                # 32, not 2000. OpenAI reserves max_completion_tokens against
                # the TOKENS-per-minute limit, not just what is generated. The
                # output here is a single JSON key -- measured at 13 tokens, with
                # zero reasoning tokens -- so 2000 was billing 125x the quota the
                # call actually needs and throttling the run to a crawl on a
                # limit that was never the real constraint (requests: 499/500
                # remaining; tokens: 7,634/200,000).
                max_completion_tokens=32)
            u = getattr(r, "usage", None)
            if u is not None:
                with _lock:
                    _usage["in"] += getattr(u, "prompt_tokens", 0) or 0
                    _usage["out"] += getattr(u, "completion_tokens", 0) or 0
            return json.loads(r.choices[0].message.content)[key]
        except Exception as e:
            msg = str(e)
            if k == tries - 1:
                with _lock:
                    print(f"    give up: {type(e).__name__} {msg[:110]}", flush=True)
                return None
            # SDK already retried with Retry-After; anything reaching here is a
            # sustained limit, so wait long rather than hammering again.
            time.sleep(backoff if ("429" in msg or "rate" in msg.lower()) else 3)
            backoff = min(backoff * 2, 240)
    return None


def score_one(client, model, row):
    # NaN is TRUTHY, so `row.get("context","") or ""` returns NaN for the first
    # chunk of every episode (which has no preceding chunk) and .replace() dies.
    ctx = row.get("context")
    ctx = "" if ctx is None or (isinstance(ctx, float) and ctx != ctx) else str(ctx)
    txt = row["text"]
    # Context goes to STAGE 2 ONLY. Adding it to both raised directional kappa
    # 0.395 -> 0.556 but DROPPED the gate from 0.526 -> 0.308: a non-political
    # chunk following a political one got flagged political. Much et al. pass the
    # preceding SPEAKER TURN -- a different person, a clean boundary. Ours is the
    # preceding 400 chars of the same speaker mid-thought, so it bleeds far more
    # readily into "is this segment about politics".
    pol = ask(client, model, POLITICS.replace("{CONTEXT}", "").replace("{TRANSCRIPT}", txt),
              "politics", ["Yes", "No"])
    out = {"chunk_id": row["chunk_id"], "politics": pol, "ideology": "", "ternary": ""}
    if pol == "Yes":
        ide = ask(client, model, IDEOLOGY.replace("{CONTEXT}", ctx).replace("{TRANSCRIPT}", txt),
                  "ideology", IDEO_VALUES)
        if ide:
            out["ideology"] = ide
            out["ternary"] = TERNARY[ide]
    return out


def cmd_score(args):
    load_env()
    from openai import OpenAI
    client = OpenAI(max_retries=8, timeout=90.0)
    src = args.input or str(ROOT / "data/output/scoring_chunks_pilot.csv")
    out_path = args.out or str(ROOT / f"data/output/ideology_scored_{args.model.replace('.','_')}.csv")
    chunks = pd.read_csv(src).to_dict("records")
    if args.limit:
        chunks = chunks[:args.limit]
    done = set()
    if os.path.exists(out_path) and not args.overwrite:
        done = {r["chunk_id"] for r in csv.DictReader(open(out_path))}
        print(f"resuming: {len(done):,} already scored")
    todo = [c for c in chunks if c["chunk_id"] not in done]
    print(f"{len(todo):,} chunks to score with {args.model}  (concurrency {args.workers})\n")
    if not todo:
        return
    new = not os.path.exists(out_path) or args.overwrite
    f = open(out_path, "w" if new else "a", newline="")
    w = csv.DictWriter(f, fieldnames=["chunk_id", "politics", "ideology", "ternary"])
    if new:
        w.writeheader()
    n = t0 = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(score_one, client, args.model, c): c for c in todo}
        for fut in as_completed(futs):
            r = fut.result()
            with _lock:
                w.writerow(r); f.flush(); n += 1
                if n % 100 == 0:
                    el = time.time() - t0
                    c = cost()
                    proj = c / n * len(todo)
                    print(f"  {n:,}/{len(todo):,}  {n/el:.1f}/s  "
                          f"eta {(len(todo)-n)/max(n/el,.01)/60:.0f}m  "
                          f"${c:.2f} spent, ${proj:.2f} projected", flush=True)
                    if args.budget and c >= args.budget and not _halt.is_set():
                        _halt.set()
                        print(f"\n  ** BUDGET CEILING ${args.budget:.2f} REACHED at "
                              f"{n:,} chunks -- draining. Rerun the same command to "
                              f"resume; scored chunks are not repeated. **", flush=True)
    f.close()
    print(f"\n{out_path}")
    d = pd.read_csv(out_path)
    print(f"  {len(d):,} scored")
    print(f"  spend this invocation: ${cost():.2f} "
          f"({_usage['in']:,} in / {_usage['out']:,} out tokens)")
    print(f"  political : {(d.politics=='Yes').sum():,} ({(d.politics=='Yes').mean():.1%})")
    v = d[d.ideology.notna() & (d.ideology != "")]
    if len(v):
        print(f"  ideology distribution:")
        for k, c in v.ideology.value_counts().items():
            print(f"    {k:<26} {c:>5} ({c/len(v):.1%})")
        print(f"  ternary mean: {v.ternary.astype(float).mean():+.3f}")


def cmd_sheet(args):
    """Blind coding sheet. Model labels are NOT included -- seeing them would
    anchor the coder, and Much et al.'s RAs were blind to LLM labels throughout.
    Labels live in the scored CSV and are joined on chunk_id afterwards."""
    src = ROOT / "data/output/scoring_chunks_pilot.csv"
    c = pd.read_csv(src)
    random.seed(20260824)
    idx = random.sample(range(len(c)), min(args.n, len(c)))
    s = c.iloc[sorted(idx)][["chunk_id", "show_name", "q", "text"]].copy()
    s["is_political_YN"] = ""
    s["ideology_label"] = ""
    p = ROOT / "data/output/ideology_coding_sheet.csv"
    s.to_csv(p, index=False)
    print(f"{p}\n  {len(s)} chunks, blind (no model labels)")
    print(f"  fill is_political_YN with Yes/No")
    print(f"  fill ideology_label with one of, for political chunks only:")
    print(f"    {', '.join(IDEO_VALUES)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--input"); ap.add_argument("--out")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="hard $ ceiling; run drains and exits when reached")
    ap.add_argument("--coding-sheet", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    if a.coding_sheet: cmd_sheet(a)
    elif a.pilot or a.input: cmd_score(a)
    else: ap.print_help()


if __name__ == "__main__":
    main()
