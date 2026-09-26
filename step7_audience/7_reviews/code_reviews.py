"""Code Apple Podcasts reviews for parasocial bond and related dimensions with the codebook in prompts/parasocial.md.

    ./code_reviews.py --pilot 300            # random pilot -> data/output/reviews_coded_pilot300.csv
    ./code_reviews.py --run [--budget 10]    # all reviews in data/output/apple_reviews.csv, resumable

Same pattern as step6_register/score_outrage_llm.py: one structured-output call per review, category names only.
"""
import argparse, csv, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL = "gpt-5.4-mini-2026-03-17"
PROMPT = ROOT / "step7_audience/7_reviews/prompts/parasocial.md"
IN = ROOT / "data/output/apple_reviews.csv"
CATS = ["connection", "companionship", "direct_address", "emotion", "loyalty", "praise_person", "praise_content",
        "agreement", "anti_media", "hostility", "disagreement"]
PRICE_IN, PRICE_OUT = 0.25 / 1e6, 2.00 / 1e6
_usage = {"in": 0, "out": 0}; _lock = threading.Lock(); _halt = threading.Event(); _pace = threading.Lock(); _last = [0.0]


def cost(): return _usage["in"] * PRICE_IN + _usage["out"] * PRICE_OUT
def pace():
    with _pace:
        dt = time.monotonic() - _last[0]
        if dt < 0.12: time.sleep(0.12 - dt)
        _last[0] = time.monotonic()
def load_env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            m = re.match(r"^([A-Z_]+)=(.*)$", line.strip())
            if m: os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))
def load_prompt():
    txt = PROMPT.read_text(); system = txt.split("## System")[1].split("## Task")[0].strip(); task = txt.split("## Task")[1].strip(); return system, task

SCHEMA = {"type": "json_schema", "json_schema": {"name": "review", "strict": True, "schema": {"type": "object", "additionalProperties": False, "required": ["categories"],
          "properties": {"categories": {"type": "array", "items": {"type": "string", "enum": CATS}}}}}}

def ask(client, system, task, row, tries=5):
    if _halt.is_set(): return None
    text = f"{row['title']}\n{row['text']}"[:2500]
    prompt = task.replace("{RATING}", str(row["rating"])).replace("{TITLE}", str(row["title"])[:120]).replace("{TEXT}", str(row["text"])[:2400])
    backoff = 4.0
    for k in range(tries):
        pace()
        try:
            r = client.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}], response_format=SCHEMA, max_completion_tokens=80)
            u = getattr(r, "usage", None)
            if u is not None:
                with _lock: _usage["in"] += u.prompt_tokens or 0; _usage["out"] += u.completion_tokens or 0
            return json.loads(r.choices[0].message.content)["categories"]
        except Exception as e:
            if k == tries - 1: return None
            time.sleep(backoff if "429" in str(e) or "rate" in str(e).lower() else 3); backoff = min(backoff * 2, 240)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pilot", type=int, default=0); ap.add_argument("--run", action="store_true"); ap.add_argument("--out"); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--budget", type=float, default=0.0); a = ap.parse_args()
    load_env(); system, task = load_prompt(); from openai import OpenAI; client = OpenAI(max_retries=8, timeout=90.0)
    df = pd.read_csv(IN, dtype={"show_id": str}).drop_duplicates(["show_id", "review_id"]); df = df[df.text.astype(str).str.len() > 10]
    if a.pilot: df = df.sample(a.pilot, random_state=5); out = a.out or str(ROOT / f"data/output/reviews_coded_pilot{a.pilot}.csv")
    elif a.run: out = a.out or str(ROOT / "data/output/reviews_coded.csv")
    else: ap.print_help(); return
    done = {r["review_id"] for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
    todo = [r for r in df.to_dict("records") if str(r["review_id"]) not in done]; print(f"{len(todo):,} reviews to code", flush=True)
    new = not os.path.exists(out); f = open(out, "a", newline=""); w = csv.DictWriter(f, fieldnames=["review_id", "show_id", "rating", "n_cats"] + CATS)
    if new: w.writeheader()
    n, t0 = 0, time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(ask, client, system, task, r): r for r in todo}
        for fut in as_completed(futs):
            cats = fut.result()
            if cats is None: continue
            r = futs[fut]; row = {"review_id": r["review_id"], "show_id": r["show_id"], "rating": r["rating"], "n_cats": len(set(cats))}; row.update({c: int(c in cats) for c in CATS})
            with _lock:
                w.writerow(row); f.flush(); n += 1
                if n % 200 == 0:
                    el = time.time() - t0; c = cost(); print(f"  {n:,}/{len(todo):,} {n/el:.1f}/s eta {(len(todo)-n)/max(n/el,.01)/60:.0f}m ${c:.2f} spent ${c/n*len(todo):.2f} projected", flush=True)
                    if a.budget and c >= a.budget and not _halt.is_set(): _halt.set(); print("  ** budget reached; draining **", flush=True)
    f.close(); d = pd.read_csv(out); print(f"\n{out}: {len(d):,} coded | any bond category (connection/companionship/loyalty) {(d[['connection','companionship','loyalty']].sum(axis=1)>0).mean():.1%}")
    for c in CATS: print(f"   {c:<15} {d[c].mean():6.1%}")
    print(f"   spend: ${cost():.2f}")

if __name__ == "__main__": main()
