"""Phase A: validate the LLM ad/meta classifier against the 120-passage
hand-labeled gold set BEFORE touching the full corpus.

Gold set (data/output/adclf/gold_120_v2.csv): single-annotator (this session).
The FIRST gold set (gold_120.csv) was labeled from ~220-character excerpts,
not full text, and is unreliable -- ad copy frequently sits later in the
passage than the excerpt reached, so passages containing substantial ad
reads were labeled "content". v2 re-labels all 120 by reading each passage
in full, and additionally records has_ad / ad_words so the predominance
label is derived rather than eyeballed. This single-annotator limitation is
reported explicitly per the task's honesty controls, not hidden.

Passage length: these chunks average ~1206 words (median 6837 chars, max
7935), NOT the ~500 words the pipeline nominally targets. An earlier version
of this script truncated model input at 3000 characters, which silently fed
the classifier only ~44% of the median passage; that truncation is removed.

Model: gpt-4o-mini (cheap, not flagship/reasoning, per task guardrails).
This IS a small enough call volume (120 passages) to run synchronously
rather than via the Batch API -- Batch is for Phase B's ~40k-passage run.

Gate (checked in this script, not just eyeballed): proceed to Phase B only
if ad_sponsor precision >= 0.90 AND zero (or near-zero) false positives on
substantive political/economic content -- the deflation-risk failure mode
this whole task exists to prevent.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv(override=True)

MODEL = "gpt-4o-mini"
GOLD_PATH = Path("data/output/adclf/gold_120_v2.csv")
OUT_DIR = Path("data/output/adclf")

SYSTEM_PROMPT = """You are classifying long (~1200-word) passages transcribed from US political podcasts. \
Each passage may contain a mix of genuine political/economic discussion and inserted advertising \
or show-produced boilerplate. Classify the PASSAGE AS A WHOLE into exactly one label:

- "ad_sponsor": the passage is dominated by a host-read or produced advertisement for a third-party \
product, service, or sponsor (e.g. a skincare brand, a mobile carrier, a gold/investment company, a \
food delivery service). Look for ad-read structure: brand names, "sponsored by", "use code", \
"go to [brand].com", promotional offers, "brought to you by".

- "meta_boilerplate": the passage is dominated by the SHOW'S OWN structural content -- intros \
("welcome to [show name]", host self-introduction), segment transitions ("coming up next..."), \
outros ("thanks for listening", "subscribe", "rate and review") -- NOT a paid third-party ad, and \
NOT substantive political/economic discussion.

- "content": the passage is genuine political, policy, economic, legal, cultural, or social discussion \
-- including passages that DISCUSS a product category, company, or economic topic (e.g. gold as a \
monetary-policy topic, a company's business practices, health insurance policy) as part of the actual \
political/economic conversation, not as a paid placement for it.

CRITICAL DISTINCTION: a host reading a paid sponsorship for a gold-investment company ("Is your \
portfolio built for what's coming next? ... call and mention this show") is ad_sponsor. A host or \
guest discussing gold as a hedge against inflation, monetary policy, or the gold standard as part of \
political/economic commentary is content. The test is not "does this passage mention a product or \
company" -- it's "is this passage functioning as a paid advertisement, or as genuine discourse".

Many passages are long transcript windows (~1200 words) that MIX a short ad-read with substantial \
surrounding political content. A passage containing a 150-word ad read inside 1000 words of political \
discussion is PREDOMINANTLY content, and should be labeled content. Classify based on which one the \
passage is predominantly doing, by volume. If genuinely \
unsure between ad_sponsor and content, choose content -- deleting real political discourse is a much \
costlier error than leaving one ad-adjacent passage in the corpus.

Respond with strict JSON only: {"label": "ad_sponsor" | "meta_boilerplate" | "content", "confidence": 0.0-1.0}
No other text."""


RETRY_WAIT_RE = re.compile(r"try again in (\d+(?:\.\d+)?)(s|m)")


def classify_passage(client: OpenAI, text: str, max_retries: int = 6) -> dict:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            break
        except RateLimitError as e:
            msg = str(e)
            m = RETRY_WAIT_RE.search(msg)
            if m:
                wait = float(m.group(1)) * (60 if m.group(2) == "m" else 1) + 2
            else:
                wait = min(60, 2 ** attempt)
            print(f"    [retry] rate limited, waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})...", flush=True)
            time.sleep(wait)
    else:
        raise RuntimeError(f"exhausted {max_retries} retries on rate limit")

    usage = resp.usage
    content = resp.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"label": "content", "confidence": 0.0, "parse_error": content}
    return {
        "label": parsed.get("label", "content"),
        "confidence": parsed.get("confidence", None),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
    }


CHECKPOINT_PATH = OUT_DIR / "phase_a_results.csv"


def run() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    gold = pd.read_csv(GOLD_PATH, dtype={"chunk_id": str})

    done = {}
    if CHECKPOINT_PATH.exists():
        prev = pd.read_csv(CHECKPOINT_PATH, dtype={"chunk_id": str})
        done = {row["chunk_id"]: row.to_dict() for _, row in prev.iterrows()}
        print(f"[phase A] resuming: {len(done)} already classified from a previous run")

    todo = gold[~gold["chunk_id"].isin(done.keys())]
    print(f"[phase A] classifying {len(todo)} of {len(gold)} gold passages with {MODEL} "
          f"({len(done)} already done)...")

    t0 = time.time()
    for i, (_, row) in enumerate(todo.iterrows()):
        r = classify_passage(client, row["text"])
        r["chunk_id"] = row["chunk_id"]
        r["hand_label"] = row["hand_label"]
        r["stratum"] = row["stratum"]
        done[row["chunk_id"]] = r
        # checkpoint after every call so a crash never loses more than one request
        pd.DataFrame(done.values()).to_csv(CHECKPOINT_PATH, index=False)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(todo)} newly classified ({len(done)}/{len(gold)} total)...", flush=True)
    elapsed = time.time() - t0

    res_df = pd.DataFrame(done.values())
    res_df = res_df.merge(gold[["chunk_id"]], on="chunk_id")  # preserve gold-set membership/order
    res_df.to_csv(OUT_DIR / "phase_a_results.csv", index=False)

    total_prompt_tok = res_df["prompt_tokens"].sum()
    total_completion_tok = res_df["completion_tokens"].sum()
    # gpt-4o-mini pricing: $0.15/1M input, $0.60/1M output (as of this run; verify before Phase B)
    cost = (total_prompt_tok / 1_000_000 * 0.15) + (total_completion_tok / 1_000_000 * 0.60)
    print(f"\n[phase A] done in {elapsed:.1f}s. tokens: {total_prompt_tok} in / {total_completion_tok} out. "
          f"cost=${cost:.4f}")

    per_passage_in = total_prompt_tok / len(gold)
    per_passage_out = total_completion_tok / len(gold)
    full_corpus_n = 40242
    full_cost_sync = (per_passage_in * full_corpus_n / 1_000_000 * 0.15) + \
                      (per_passage_out * full_corpus_n / 1_000_000 * 0.60)
    full_cost_batch = full_cost_sync * 0.5  # batch API 50% discount
    print(f"[phase A] extrapolated full corpus ({full_corpus_n} passages): "
          f"sync=${full_cost_sync:.2f}  batch(50% off)=${full_cost_batch:.2f}")

    # --- confusion matrix ---
    labels = ["ad_sponsor", "meta_boilerplate", "content"]
    confusion = pd.crosstab(res_df["hand_label"], res_df["label"], dropna=False)
    confusion = confusion.reindex(index=labels, columns=labels, fill_value=0)
    print("\n[phase A] confusion matrix (rows=hand label, cols=LLM label):")
    print(confusion)

    # --- ad_sponsor precision/recall/F1 ---
    tp = int(((res_df["label"] == "ad_sponsor") & (res_df["hand_label"] == "ad_sponsor")).sum())
    fp = int(((res_df["label"] == "ad_sponsor") & (res_df["hand_label"] != "ad_sponsor")).sum())
    fn = int(((res_df["label"] != "ad_sponsor") & (res_df["hand_label"] == "ad_sponsor")).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    print(f"\n[phase A] ad_sponsor: precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}  "
          f"(tp={tp} fp={fp} fn={fn})")

    # --- THE KEY NUMBER: content misclassified as ad_sponsor ---
    content_as_ad = res_df[(res_df["hand_label"] == "content") & (res_df["label"] == "ad_sponsor")]
    econ_content_as_ad = content_as_ad[content_as_ad["stratum"].isin(["ambiguous", "substantive_topic"])]
    print(f"\n[phase A] *** DEFLATION RISK *** content passages misclassified as ad_sponsor: "
          f"{len(content_as_ad)} / {len(gold[gold['hand_label'] == 'content'])}")
    if len(content_as_ad) > 0:
        print("[phase A] false positives (verbatim, for inspection):")
        gold_lookup = gold.set_index("chunk_id")
        for _, r in content_as_ad.iterrows():
            txt = gold_lookup.loc[r["chunk_id"], "text"][:300]
            print(f"  [{r['chunk_id']}] stratum={r['stratum']} confidence={r['confidence']}")
            print(f"    {txt!r}")

    gate_precision = precision >= 0.90 if not pd.isna(precision) else False
    gate_fp = len(content_as_ad) <= 2  # near-zero
    gate_pass = gate_precision and gate_fp

    print(f"\n[phase A] GATE: ad_sponsor precision >= 0.90? {gate_precision} ({precision:.3f}). "
          f"content false positives near-zero (<=2)? {gate_fp} ({len(content_as_ad)}).")
    print(f"[phase A] GATE RESULT: {'PASS -- proceed to Phase B' if gate_pass else 'FAIL -- STOP, do not run Phase B'}")

    summary = {
        "model": MODEL,
        "n_gold": len(gold),
        "elapsed_sec": elapsed,
        "tokens_in": int(total_prompt_tok),
        "tokens_out": int(total_completion_tok),
        "pilot_cost_usd": cost,
        "extrapolated_full_corpus_n": full_corpus_n,
        "extrapolated_full_cost_sync_usd": full_cost_sync,
        "extrapolated_full_cost_batch_usd": full_cost_batch,
        "confusion_matrix": confusion.to_dict(),
        "ad_sponsor_precision": precision,
        "ad_sponsor_recall": recall,
        "ad_sponsor_f1": f1,
        "content_false_positives_as_ad": len(content_as_ad),
        "content_false_positive_chunk_ids": content_as_ad["chunk_id"].tolist(),
        "gate_precision_ok": gate_precision,
        "gate_fp_ok": gate_fp,
        "gate_pass": gate_pass,
    }
    (OUT_DIR / "phase_a_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[phase A] summary -> {OUT_DIR / 'phase_a_summary.json'}")


if __name__ == "__main__":
    run()
