#!/usr/bin/env python3
"""Pairwise position scaling of shows (pilot on the sandbox).

Instead of absolute 7-point labels, the model compares two speakers' stated
policy positions and says whose are further right; a Bradley-Terry fit turns
thousands of such verdicts into a relative position per show, within a time
period. Design per results/ideology_scaling_notes.md §5:

  prepare    political passages (stage-1 "Yes") from sandbox episodes, up to
             N per show per period
  breakdown  one LLM call per passage: the policy positions the speaker takes
             (concept-guided breakdown; Wu et al. 2025)
  compare    K matchups per show within a period, random opponent shows, one
             random breakdown each, both orderings, A / B / Tie
  fit        Bradley-Terry (ties = half win), bootstrap SEs, z within period;
             validation against side/one-sidedness, DIME, Brookings; order
             consistency; monotonicity against the 7-point gap; cross-period
             agreement -> results/pairwise_pilot.md

    .venv/bin/python step4_ideology/pairwise_scale.py prepare
    .venv/bin/python step4_ideology/pairwise_scale.py breakdown
    .venv/bin/python step4_ideology/pairwise_scale.py compare
    .venv/bin/python step4_ideology/pairwise_scale.py fit

API key from .env (same as score_ideology.py). Outputs under data/output/pairwise/.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "step4_ideology"))
from score_ideology import load_env, pace, _usage, PRICE_IN, PRICE_OUT  # noqa: E402

OUT = ROOT / "data/output/pairwise"
MODEL = "gpt-5.4-mini-2026-03-17"
PERIODS = {"P1": (2019, 2021), "P2": (2024, 2026)}
PER_SHOW = 30          # passages per show per period
MATCHUPS = 20          # opponents per show per period
MIN_PASSAGES = 12
SEED = 20261128
LABEL7 = {"strongly_liberal": -3, "moderately_liberal": -2, "slightly_liberal": -1, "moderate": 0,
          "slightly_conservative": 1, "moderately_conservative": 2, "strongly_conservative": 3}

SYSTEM_BREAK = "You are a research assistant extracting the policy positions a speaker takes in a podcast transcript passage. Return only valid JSON."
PROMPT_BREAK = """Passage from a US political podcast, published {date}:

{text}

List the policy positions the SPEAKER personally takes or endorses in this passage, one per line in the form "issue: stance", at most 6 lines. Include only positions the speaker holds; do not include positions they merely describe, quote or attack. Do not evaluate or rate anything. If the speaker takes no identifiable policy position, write exactly: none"""

SYSTEM_CMP = "You are a research assistant comparing the policy positions of two speakers. Return only valid JSON."
PROMPT_CMP = """Two speakers on US political podcasts, both recorded in {period}. Their stated policy positions, as extracted from one passage each:

Speaker A:
{a}

Speaker B:
{b}

Based only on these stated positions, and judged against the US politics of {period}, whose policy positions are further to the RIGHT (more conservative)? Answer "A", "B", or "Tie" if they are about equally placed or there is too little to compare."""

_lock = threading.Lock()


def client():
    load_env()
    from openai import OpenAI
    return OpenAI()


def jschema(name, props):
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": list(props), "properties": props}}}


def ask(cl, system, prompt, schema, max_tokens, tries=5):
    backoff = 4.0
    for k in range(tries):
        pace()
        try:
            r = cl.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                                           response_format=schema, max_completion_tokens=max_tokens)
            u = getattr(r, "usage", None)
            if u is not None:
                with _lock:
                    _usage["in"] += getattr(u, "prompt_tokens", 0) or 0
                    _usage["out"] += getattr(u, "completion_tokens", 0) or 0
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            if k == tries - 1:
                print(f"    give up: {type(e).__name__} {str(e)[:100]}", flush=True)
                return None
            time.sleep(backoff if "429" in str(e) else 3)
            backoff = min(backoff * 2, 240)


def cost():
    return _usage["in"] * PRICE_IN + _usage["out"] * PRICE_OUT


# ----------------------------------------------------------------------------- prepare

def cmd_prepare(args):
    OUT.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(ROOT / "data/output/scoring_chunks_cap40.csv", dtype=str,
                    usecols=["chunk_id", "episode_id", "collection_id", "show_name", "pub_date", "text", "n_words"])
    lab = pd.read_csv(ROOT / "data/output/ideology_cap40.csv", dtype=str)
    m = c.merge(lab, on="chunk_id")
    sb = set(pd.read_csv(ROOT / "sandbox/episodes.csv", dtype=str).episode_id)
    m = m[m.episode_id.isin(sb) & (m.politics == "Yes") & (pd.to_numeric(m.n_words, errors="coerce") >= 60)].copy()
    m["year"] = pd.to_datetime(m.pub_date, errors="coerce", utc=True).dt.year
    rng = random.Random(SEED)
    rows = []
    for period, (lo, hi) in PERIODS.items():
        mp = m[m.year.between(lo, hi)]
        for sid, g in mp.groupby("collection_id"):
            if len(g) < MIN_PASSAGES:
                continue
            idx = list(g.index); rng.shuffle(idx)
            for i in idx[:PER_SHOW]:
                r = g.loc[i]
                rows.append({"period": period, "show_id": sid, "show_name": r.show_name, "chunk_id": r.chunk_id, "episode_id": r.episode_id,
                             "pub_date": str(r.pub_date)[:10], "label7": r.ideology, "text": r.text})
    p = pd.DataFrame(rows)
    p.to_csv(OUT / "passages.csv", index=False)
    print(f"passages: {len(p):,}; shows per period: {p.groupby('period').show_id.nunique().to_dict()}")


# ----------------------------------------------------------------------------- breakdown

def cmd_breakdown(args):
    p = pd.read_csv(OUT / "passages.csv", dtype=str)
    out = OUT / "breakdowns.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["chunk_id"] for l in open(out)}
    todo = p[~p.chunk_id.isin(done)]
    print(f"breakdown: {len(todo):,} to do ({len(done):,} done)")
    cl = client(); schema = jschema("positions", {"positions": {"type": "string"}})
    f = open(out, "a")
    def one(r):
        res = ask(cl, SYSTEM_BREAK, PROMPT_BREAK.format(date=r.pub_date, text=r.text), schema, 320)
        return r.chunk_id, (res or {}).get("positions")
    n = 0; t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, r) for r in todo.itertuples()]
        for fu in as_completed(futs):
            cid, pos = fu.result()
            if pos is not None:
                with _lock:
                    f.write(json.dumps({"chunk_id": cid, "positions": pos.strip()}) + "\n"); f.flush()
            n += 1
            if n % 500 == 0:
                print(f"  {n:,}/{len(todo):,}  ${cost():.2f}  {n / (time.monotonic() - t0) * 60:.0f}/min", flush=True)
    print(f"breakdown done: ${cost():.2f}")


# ----------------------------------------------------------------------------- compare

def cmd_compare(args):
    p = pd.read_csv(OUT / "passages.csv", dtype=str)
    br = {}
    for l in open(OUT / "breakdowns.jsonl"):
        d = json.loads(l); br[d["chunk_id"]] = d["positions"]
    p["positions"] = p.chunk_id.map(br)
    usable = p[p.positions.notna() & (p.positions.str.strip().str.lower() != "none") & (p.positions.str.len() > 8)]
    print(f"passages with positions: {len(usable):,} of {len(p):,} ({len(usable) / len(p):.0%})")
    rng = random.Random(SEED + 1)
    out = OUT / "comparisons.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["pair_id"] for l in open(out)}
    plan = []
    for period in PERIODS:
        up = usable[usable.period == period]
        shows = sorted(up.show_id.unique())
        by = {s: g.to_dict("records") for s, g in up.groupby("show_id")}
        for s in shows:
            opps = [o for o in shows if o != s]
            for k in range(MATCHUPS):
                o = rng.choice(opps)
                a, b = rng.choice(by[s]), rng.choice(by[o])
                for order in (0, 1):
                    pid = f"{period}:{s}:{o}:{k}:{order}"
                    if pid in done:
                        continue
                    first, second = (a, b) if order == 0 else (b, a)
                    plan.append({"pair_id": pid, "period": period, "show_a": first["show_id"], "show_b": second["show_id"],
                                 "chunk_a": first["chunk_id"], "chunk_b": second["chunk_id"], "order": order,
                                 "pa": first["positions"], "pb": second["positions"],
                                 "l7a": first["label7"], "l7b": second["label7"]})
    print(f"comparisons to run: {len(plan):,} ({len(done):,} done)")
    cl = client(); schema = jschema("answer", {"answer": {"type": "string", "enum": ["A", "B", "Tie"]}})
    f = open(out, "a")
    labels = {"P1": "2019-2021", "P2": "2024-2026"}
    def one(d):
        res = ask(cl, SYSTEM_CMP, PROMPT_CMP.format(period=labels[d["period"]], a=d["pa"], b=d["pb"]), schema, 48)
        return d, (res or {}).get("answer")
    n = 0; t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, d) for d in plan]
        for fu in as_completed(futs):
            d, ans = fu.result()
            if ans is not None:
                rec = {k: v for k, v in d.items() if k not in ("pa", "pb")}; rec["answer"] = ans
                with _lock:
                    f.write(json.dumps(rec) + "\n"); f.flush()
            n += 1
            if n % 1000 == 0:
                print(f"  {n:,}/{len(plan):,}  ${cost():.2f}  {n / (time.monotonic() - t0) * 60:.0f}/min", flush=True)
    print(f"compare done: ${cost():.2f}")


# ----------------------------------------------------------------------------- fit

def bradley_terry(items, wins, ties, iters=2000, tol=1e-9):
    """MM algorithm (Hunter 2004) for Bradley-Terry with ties counted as half a
    win for each side. items: list of ids; wins: dict (i, j) -> count of i over j."""
    idx = {it: k for k, it in enumerate(items)}; n = len(items)
    W = np.zeros((n, n))
    for (i, j), c in wins.items():
        W[idx[i], idx[j]] += c
    for (i, j), c in ties.items():
        W[idx[i], idx[j]] += 0.5 * c; W[idx[j], idx[i]] += 0.5 * c
    N = W + W.T
    p = np.ones(n)
    for _ in range(iters):
        denom = (N / (p[:, None] + p[None, :])).sum(axis=1)
        new = W.sum(axis=1) / np.maximum(denom, 1e-12)
        new = np.maximum(new, 1e-9); new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(np.log(new) - np.log(p))) < tol:
            p = new; break
        p = new
    return dict(zip(items, np.log(p)))


def fit_period(comp: pd.DataFrame, period: str, n_boot: int = 200, seed: int = SEED):
    c = comp[comp.period == period]
    # verdict in terms of "first show more conservative"; use both orderings as independent verdicts
    items = sorted(set(c.show_a) | set(c.show_b))
    def tally(df):
        wins, ties = {}, {}
        for r in df.itertuples():
            if r.answer == "A": wins[(r.show_a, r.show_b)] = wins.get((r.show_a, r.show_b), 0) + 1
            elif r.answer == "B": wins[(r.show_b, r.show_a)] = wins.get((r.show_b, r.show_a), 0) + 1
            else: ties[(r.show_a, r.show_b)] = ties.get((r.show_a, r.show_b), 0) + 1
        return wins, ties
    w, t = tally(c); theta = bradley_terry(items, w, t)
    rng = np.random.default_rng(seed); boots = {it: [] for it in items}
    for _ in range(n_boot):
        bs = c.sample(len(c), replace=True, random_state=int(rng.integers(1e9)))
        w, t = tally(bs); th = bradley_terry(items, w, t)
        for it in items: boots[it].append(th.get(it, np.nan))
    df = pd.DataFrame({"show_id": items, "theta": [theta[i] for i in items],
                       "se": [np.nanstd(boots[i]) for i in items], "n_comparisons": [int(((c.show_a == i) | (c.show_b == i)).sum()) for i in items]})
    df["z"] = (df.theta - df.theta.mean()) / df.theta.std()
    df["period"] = period
    return df


def cmd_fit(args):
    from scipy.stats import spearmanr, pearsonr
    comp = pd.DataFrame([json.loads(l) for l in open(OUT / "comparisons.jsonl")])
    comp["key"] = comp.pair_id.str.rsplit(":", n=1).str[0]
    lines = ["# Pairwise position scaling: sandbox pilot", "", f"Model `{MODEL}`; {len(comp):,} comparisons; periods {PERIODS}; {PER_SHOW} passages and {MATCHUPS} matchups per show per period, both orderings.", ""]
    # order consistency & ties
    def flip(a): return {"A": "B", "B": "A", "Tie": "Tie"}[a]
    g = comp.groupby("key").apply(lambda d: None if len(d) < 2 else (d.sort_values("order").answer.iloc[0], flip(d.sort_values("order").answer.iloc[1])))
    g = g.dropna()
    agree = np.mean([a == b for a, b in g]); ties = (comp.answer == "Tie").mean()
    lines += [f"- Order consistency (verdict unchanged when A/B are swapped): {agree:.1%} of {len(g):,} matchups; tie rate {ties:.1%}.", ""]
    # monotonicity vs 7-point gap
    comp["gap"] = comp.l7a.map(LABEL7) - comp.l7b.map(LABEL7)
    cc = comp.dropna(subset=["gap"]); cc = cc[cc.answer != "Tie"]
    mono = cc.groupby(cc.gap.clip(-4, 4)).apply(lambda d: pd.Series({"p_A_right": (d.answer == "A").mean(), "n": len(d)}))
    lines += ["**Monotonicity against the 7-point labels of the two passages** (P(A judged further right) by the gap in 7-point codes, A minus B; a working scale rises left to right):", "", "| gap | P(A right) | n |", "|---|---|---|"]
    for gp, r in mono.iterrows(): lines.append(f"| {int(gp):+d} | {r.p_A_right:.2f} | {int(r.n)} |")
    lines.append("")
    so = pd.read_csv(ROOT / "data/output/show_level_outcomes.csv", dtype={"show_id": str}).set_index("show_id")
    dime = pd.read_csv(ROOT / "data/output/shows_205_host_dime_scores.csv", dtype={"show_id": str}).set_index("show_id")
    lv = pd.read_csv(ROOT / "data/output/lean_validation.csv", dtype={"collection_id": str}); lv = lv[lv.match_status == "matched"].set_index("collection_id")
    fits = []
    for period in PERIODS:
        if (comp.period == period).sum() == 0: continue
        df = fit_period(comp, period, n_boot=args.boot); fits.append(df)
        df["side"] = df.show_id.map(so.side); df["inten"] = df.show_id.map(so.inten); df["lean"] = df.show_id.map(so.lean)
        df["cfscore"] = pd.to_numeric(df.show_id.map(dime.avg_host_cfscore), errors="coerce")
        df["brookings"] = df.show_id.map(lv.brookings_partisan_leaning)
        lines += [f"## Period {period} ({PERIODS[period][0]}–{PERIODS[period][1]}): {len(df)} shows", "",
                  f"- Bootstrap SE of θ: median {df.se.median():.2f} (θ SD {df.theta.std():.2f}); reliability (1 − median SE²/var) ≈ {1 - (df.se.median() ** 2) / df.theta.var():.2f}.",]
        d = df.dropna(subset=["side"]); r1 = spearmanr(d.z, d.side); r2 = pearsonr(d.z, d.side)
        lines.append(f"- vs current direction score `side` (n={len(d)}): Spearman {r1.statistic:.2f}, Pearson {r2.statistic:.2f}.")
        d = df.dropna(subset=["inten"]); r1 = spearmanr(d.z.abs(), d.inten)
        lines.append(f"- |position| vs one-sidedness `inten` (n={len(d)}): Spearman {r1.statistic:.2f} (should be well below 1 if position ≠ emphasis).")
        d = df.dropna(subset=["cfscore"]); r1 = spearmanr(d.z, d.cfscore) if len(d) > 5 else None
        lines.append(f"- vs hosts' DIME CFscore (n={len(d)}): Spearman {r1.statistic:.2f}." if r1 else f"- DIME: n={len(d)}, too few.")
        d = df.dropna(subset=["lean"]);
        if len(d): lines.append(f"- Known groups: mean z for L shows {d[d.lean == 'L'].z.mean():+.2f} (n={int((d.lean == 'L').sum())}), R shows {d[d.lean == 'R'].z.mean():+.2f} (n={int((d.lean == 'R').sum())}); shows on the wrong side of 0: {int(((d.lean == 'L') & (d.z > 0)).sum() + ((d.lean == 'R') & (d.z < 0)).sum())}.")
        d = df.dropna(subset=["brookings"])
        if len(d) >= 5:
            lines.append(f"- Brookings matched shows (n={len(d)}): mean z by Brookings label {d.groupby('brookings').z.mean().round(2).to_dict()}.")
        lines += ["", "Most left / most right (z):", ", ".join(f"{so.show.get(s, s)} ({z:+.2f})" for s, z in df.nsmallest(5, 'z')[['show_id', 'z']].values), "", ", ".join(f"{so.show.get(s, s)} ({z:+.2f})" for s, z in df.nlargest(5, 'z')[['show_id', 'z']].values), ""]
    allf = pd.concat(fits); allf.to_csv(OUT / "positions.csv", index=False)
    if len(fits) == 2:
        both = fits[0][["show_id", "z"]].merge(fits[1][["show_id", "z"]], on="show_id", suffixes=("_P1", "_P2"))
        r = spearmanr(both.z_P1, both.z_P2)
        lines += [f"## Across periods", "", f"- Shows in both periods: {len(both)}; Spearman of positions P1 vs P2: {r.statistic:.2f} (each period scaled relative to itself; no external bridge yet).", ""]
    lines += [f"API cost so far: ${cost():.2f} (this fit run only counts fit-time usage; see logs for scoring cost).", ""]
    (ROOT / "results/pairwise_pilot.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    b = sub.add_parser("breakdown"); b.add_argument("--workers", type=int, default=4)
    c = sub.add_parser("compare"); c.add_argument("--workers", type=int, default=4)
    f = sub.add_parser("fit"); f.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "breakdown": cmd_breakdown, "compare": cmd_compare, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
