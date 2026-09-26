#!/usr/bin/env python3
"""Model-coded outrage (Sobieraj & Berry 2011) on the ideology passage sample.

    ./score_outrage_llm.py --pilot 300                 # random pilot, prints base rates + cost
    ./score_outrage_llm.py --run [--budget 40]         # all 152,152 cap-40 passages, resumable
    ./score_outrage_llm.py --run --input X --out Y     # any passage table with chunk_id,text

Why a model and not word lists
------------------------------
Twelve of Sobieraj & Berry's thirteen outrage types are judgments about a
passage (mockery, character assassination, slippery slope ...) that no word
list can make. The thirteenth, emotional display, was coded from the sound of
the voice and is not recoverable from a transcript; it is dropped here and
reported as dropped. The codebook wording is prompts/outrage_sb.md, applied
with a structured-output schema so the model can only return type names from
the list. One call per passage; each type is a yes/no.

Unit: the same 750-character passages, capped at 40 per show-quarter, that the
ideology instrument scored (data/output/scoring_chunks_cap40.csv), so outrage,
side and intensity are measured on identical text.

Output: chunk_id + one 0/1 column per type. Show-level aggregation (share of
political words in passages with each type, z-scored on the 194 reference shows)
is done downstream, not here.
"""
import argparse, csv, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-5.4-mini-2026-03-17"           # same snapshot as the ideology labels
PROMPT_FILE = ROOT / "step6_register/prompts/outrage_sb.md"
DEFAULT_IN = ROOT / "data/output/scoring_chunks_cap40.csv"
DEFAULT_OUT = ROOT / "data/output/outrage_llm_cap40.csv"

# Type 3 (emotional display) is audio-only and excluded; the other twelve keep the codebook's names.
TYPES = ["insult", "namecall", "emotlang", "sparring", "character", "exaggeration",
         "mockery", "conflagration", "extremize", "slippery", "belittle", "obscene"]

PRICE_IN, PRICE_OUT = 0.25 / 1e6, 2.00 / 1e6
_usage = {"in": 0, "out": 0}
_lock = threading.Lock()
_halt = threading.Event()
_pace = threading.Lock()
_last = [0.0]
MIN_INTERVAL = 0.12                          # ~8 requests/s across all threads


def cost():
    return _usage["in"] * PRICE_IN + _usage["out"] * PRICE_OUT


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


def load_prompt():
    """Split prompts/outrage_sb.md into the system prompt and the task text; drop type 3."""
    txt = PROMPT_FILE.read_text()
    system = txt.split("## Task")[0].replace("## System", "").strip()
    task = "## Task" + txt.split("## Task", 1)[1]
    task = task.replace("## Task: Outrage Discourse (Sobieraj & Berry 2011)", "").strip()
    task = "\n".join(l for l in task.splitlines() if not l.startswith("3  emotdisplay"))
    task += ("\n\nScope rule (Sobieraj & Berry): outrage discourse is aimed at political opponents, "
             "public figures, groups, institutions, or their positions. Enthusiasm, praise, advertising, "
             "storytelling, and emotion about non-political matters are never outrage, and emotionally "
             "charged language (type 4) counts only when it colours a claim about such a target.\n"
             "Type 3 (emotional display) is omitted because it cannot be judged from text. "
             "Return the names of the types present, from this list only: " + ", ".join(TYPES) + ".")
    return system, task


SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "outrage", "strict": True,
    "schema": {"type": "object", "additionalProperties": False, "required": ["types"],
               "properties": {"types": {"type": "array", "items": {"type": "string", "enum": TYPES}}}}}}


def ask(client, system, task, text, tries=5):
    if _halt.is_set():
        return None
    backoff = 4.0
    for k in range(tries):
        pace()
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": task.replace("{TRANSCRIPT}", text)}],
                response_format=SCHEMA, max_completion_tokens=80)
            u = getattr(r, "usage", None)
            if u is not None:
                with _lock:
                    _usage["in"] += getattr(u, "prompt_tokens", 0) or 0
                    _usage["out"] += getattr(u, "completion_tokens", 0) or 0
            return json.loads(r.choices[0].message.content)["types"]
        except Exception as e:
            msg = str(e)
            if k == tries - 1:
                with _lock:
                    print(f"    give up: {type(e).__name__} {msg[:110]}", flush=True)
                return None
            time.sleep(backoff if ("429" in msg or "rate" in msg.lower()) else 3)
            backoff = min(backoff * 2, 240)
    return None


def score(rows, out_path, workers, budget, system, task):
    from openai import OpenAI
    client = OpenAI(max_retries=8, timeout=90.0)
    done = set()
    if os.path.exists(out_path):
        done = {r["chunk_id"] for r in csv.DictReader(open(out_path))}
        print(f"resuming: {len(done):,} already scored")
    todo = [r for r in rows if r["chunk_id"] not in done]
    print(f"{len(todo):,} passages to score with {MODEL} (concurrency {workers})", flush=True)
    new = not os.path.exists(out_path)
    f = open(out_path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=["chunk_id", "n_types"] + TYPES)
    if new:
        w.writeheader()
    n, t0 = 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(ask, client, system, task, r["text"]): r for r in todo}
        for fut in as_completed(futs):
            types = fut.result()
            if types is None:
                continue
            row = {"chunk_id": futs[fut]["chunk_id"], "n_types": len(set(types))}
            row.update({t: int(t in types) for t in TYPES})
            with _lock:
                w.writerow(row); f.flush(); n += 1
                if n % 200 == 0:
                    el = time.time() - t0; c = cost()
                    print(f"  {n:,}/{len(todo):,}  {n/el:.1f}/s  eta {(len(todo)-n)/max(n/el,.01)/60:.0f}m  "
                          f"${c:.2f} spent, ${c/n*len(todo):.2f} projected", flush=True)
                    if budget and c >= budget and not _halt.is_set():
                        _halt.set()
                        print(f"\n  ** BUDGET ${budget:.2f} REACHED at {n:,}; draining. Rerun to resume. **", flush=True)
    f.close()
    return n


def report(out_path):
    d = pd.read_csv(out_path)
    print(f"\n{out_path}\n  {len(d):,} scored | any outrage {d.n_types.gt(0).mean():.1%} | mean types/passage {d.n_types.mean():.2f}")
    for t in TYPES:
        print(f"    {t:<14} {d[t].mean():6.1%}")
    print(f"  spend this invocation: ${cost():.2f} ({_usage['in']:,} in / {_usage['out']:,} out tokens)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="score N random passages to a pilot file")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--input", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    load_env()
    system, task = load_prompt()
    df = pd.read_csv(a.input, usecols=["chunk_id", "text"])
    if a.pilot:
        df = df.sample(a.pilot, random_state=a.seed)
        out = a.out or str(ROOT / f"data/output/outrage_llm_pilot{a.pilot}.csv")
    elif a.run:
        out = a.out or str(DEFAULT_OUT)
    else:
        ap.print_help(); return
    rows = df.to_dict("records")
    score(rows, out, a.workers, a.budget, system, task)
    report(out)


if __name__ == "__main__":
    main()
