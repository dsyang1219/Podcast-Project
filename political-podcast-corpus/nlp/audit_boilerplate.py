"""Measurement-only audit: how much show-specific sponsor/boilerplate copy
survived Stage-1 cleaning and is shaping the embedding matrix used by
nlp/embed_rank.py, nlp/embed_ideology.py, and nlp/embed_dimensions.py.

    python -m nlp.audit_boilerplate [--target-words 500]

Motivation: nlp/clean.py's boilerplate filter (BOILERPLATE_NGRAM_SIZE=8,
BOILERPLATE_MIN_SHOWS=4, BOILERPLATE_COVERAGE=0.5) only catches n-grams
recurring across >=4 DISTINCT shows -- i.e. ad-network copy. A sponsor read
that only ever airs on ONE show (Chevron on POLITICO Energy) never touches
that bar and survives into chunks_{target}_bert.csv's `bert_text` column
(BERT_TEXT_SOURCE = "raw_sentences_boilerplate_removed_no_token_filter" in
nlp/run_chunks.py -- "removed" means cross-show only). This showed up
concretely while blind-reading nlp/embed_dimensions.py's pc_extremes_*.csv:
the identical Chevron/POLITICO Energy ad-read as 4/15 of PC4's low-pole
chunks, a Patreon donor-shoutout block as an extreme on both PC1 and PC5,
Wayfair/Blackout Coffee ad-reads at PC3's low pole.

Does NOT re-run cleaning, re-clean, or re-embed -- this only measures.

Step 1: classify the chunks already sitting in pc_extremes_1..5.csv.
Step 2: build a WITHIN-show analogue of build_boilerplate_index (n-grams
        recurring in >=30% of a show's own transcribed episodes, the
        single-show blind spot in the existing >=4-show filter) and use it
        + the Step-1 sponsor-token list to flag chunks corpus-wide.
Step 3: correlate per-show boilerplate share against avg_host_cfscore
        (the one that would force a re-run of the ideology R^2=0.21 result),
        duration, and episodes_per_week.
Step 4: severity verdict per the thresholds given in the task brief.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline import config as pipeline_config
from .embed_dimensions import load_episode_metadata, load_show_metadata
from .embed_ideology import load_cfscore

TARGET = 500
WORD_RE = re.compile(r"[a-z0-9']+")

# Same n-gram size as the existing cross-show filter (nlp/config.py), so this
# is a genuine "within-show" analogue, not an arbitrary different rule.
NGRAM_SIZE = 8
WITHIN_SHOW_MIN_EPISODE_FRAC = 0.30
MIN_EPISODES_FOR_CHECK = 3  # below this, "recurs in >=30%" isn't a meaningful statement
COVERAGE_FLAG_THRESHOLD = 0.50
NEAR_DUP_COSINE = 0.6

# Step-1 sponsor/boilerplate token list, built from what was actually seen
# in the pc_extremes_*.csv reads (task brief's list, plus obvious variants
# spotted in the transcripts -- e.g. spoken-aloud URLs render as "dot com
# slash" or "<brand>.com forward slash", not literal punctuation).
SPONSOR_PATTERNS = [
    r"\bchevron\b", r"\benergy tailgate\b", r"\bwayfair\b", r"\bblackout coffee\b",
    r"\bpromo code\b", r"\buse code\b", r"\bdiscount code\b",
    r"\bdot com slash\b", r"\bforward slash\b", r"\bthis episode is brought to you\b",
    r"\bbrought to you by\b", r"\bsponsored by\b", r"\b\w+\.com\b",
]
SPONSOR_RE = re.compile("|".join(SPONSOR_PATTERNS), re.I)
PATREON_RE = re.compile(r"\bpatreon\b|\bspecial thanks to\b", re.I)
INTRO_OUTRO_RE = re.compile(
    r"\bwelcome to\b|\bwelcome back to\b|\bthanks for listening\b|\bsee you next time\b|"
    r"\bi'?m your host\b|\bthat'?s our show\b|\bthat'?s it for (?:today|this episode)\b",
    re.I,
)

# For the within-show n-gram coverage check: convert the sponsor phrase list
# into word-tuples of their own length so they contribute to the SAME
# word-level coverage array as the n-gram hits (task: "...covered by these
# n-grams OR by the known-sponsor token list" -- unified as one coverage
# computation rather than two disconnected checks).
SPONSOR_PHRASE_WORDS = [
    tuple(WORD_RE.findall(p.lower())) for p in
    ["chevron", "energy tailgate", "wayfair", "blackout coffee",
     "promo code", "use code", "discount code", "dot com slash",
     "forward slash", "sponsored by", "brought to you by", "patreon"]
    if WORD_RE.findall(p.lower())
]


def normalize_words(text) -> list[str]:
    if not isinstance(text, str):
        return []
    return WORD_RE.findall(text.lower())


def ngrams(words: list[str], n: int):
    for i in range(len(words) - n + 1):
        yield tuple(words[i:i + n])


# --------------------------------------------------------------------- #
# Step 1: classify the already-generated blind-read extremes
# --------------------------------------------------------------------- #

def classify_content(text: str) -> str:
    if PATREON_RE.search(text):
        return "mostly_patreon_shoutout"
    if SPONSOR_RE.search(text):
        return "mostly_ad"
    if INTRO_OUTRO_RE.search(text):
        return "mostly_intro_outro"
    return "substantive"


def step1_classify_extremes(out_dir: Path) -> dict:
    rows = []
    for pc in range(1, 6):
        path = out_dir / f"pc_extremes_{pc}.csv"
        df = pd.read_csv(path)
        df["pc"] = pc
        rows.append(df)
    all_extremes = pd.concat(rows, ignore_index=True)
    all_extremes["content_category"] = all_extremes["bert_text"].map(classify_content)

    # Cross-PC near-duplicate check via TF-IDF cosine, over ALL 150 extreme
    # chunks at once -- this is what catches the Patreon block appearing as
    # an extreme on both PC1 and PC5 (two different pc_extremes_*.csv files).
    vec = TfidfVectorizer(stop_words="english", max_features=20000)
    X = vec.fit_transform(all_extremes["bert_text"].fillna(""))
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0.0)
    # a chunk_id can legitimately appear as an extreme for >1 PC (same chunk,
    # not a duplicate finding) -- exclude same-chunk_id pairs from the check,
    # we only care about DISTINCT chunks that are near-duplicates of content.
    chunk_ids = all_extremes["chunk_id"].to_numpy()
    same_id = chunk_ids[:, None] == chunk_ids[None, :]
    sim_masked = np.where(same_id, 0.0, sim)
    max_sim = sim_masked.max(axis=1)
    best_match_idx = sim_masked.argmax(axis=1)

    all_extremes["max_cosine_to_other_extreme"] = max_sim
    all_extremes["near_duplicate"] = max_sim >= NEAR_DUP_COSINE
    all_extremes["near_duplicate_of_chunk_id"] = [
        chunk_ids[best_match_idx[i]] if all_extremes["near_duplicate"].iloc[i] else None
        for i in range(len(all_extremes))
    ]
    all_extremes["near_duplicate_of_pc"] = [
        int(all_extremes["pc"].iloc[best_match_idx[i]]) if all_extremes["near_duplicate"].iloc[i] else None
        for i in range(len(all_extremes))
    ]

    all_extremes["final_classification"] = np.where(
        all_extremes["near_duplicate"], "near_duplicate_of_another_extreme",
        all_extremes["content_category"],
    )

    per_pc = {}
    for pc in range(1, 6):
        sub = all_extremes[all_extremes["pc"] == pc]
        counts = sub["final_classification"].value_counts().to_dict()
        n_nonsubstantive = int((sub["final_classification"] != "substantive").sum())
        per_pc[f"PC{pc}"] = {
            "n_total": len(sub),
            "counts": counts,
            "n_nonsubstantive": n_nonsubstantive,
            "pct_nonsubstantive": round(100 * n_nonsubstantive / len(sub), 1),
        }

    overall_nonsubstantive = int((all_extremes["final_classification"] != "substantive").sum())
    overall = {
        "n_total_extreme_slots": len(all_extremes),
        "n_nonsubstantive": overall_nonsubstantive,
        "pct_nonsubstantive": round(100 * overall_nonsubstantive / len(all_extremes), 1),
        "cross_pc_near_duplicates_found": int(all_extremes["near_duplicate"].sum()),
    }

    detail_path = out_dir / "boilerplate_audit_extremes_detail.csv"
    all_extremes[["pc", "pole", "chunk_id", "show_name", "content_category",
                  "near_duplicate", "near_duplicate_of_chunk_id", "near_duplicate_of_pc",
                  "max_cosine_to_other_extreme", "final_classification"]].to_csv(detail_path, index=False)

    return {"per_pc": per_pc, "overall": overall, "detail_csv": detail_path.name}


# --------------------------------------------------------------------- #
# Step 2: corpus-wide contamination estimate
# --------------------------------------------------------------------- #

def build_episode_texts(chunks_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    grouped = chunks_df.sort_values(["collection_id", "episode_id", "chunk_index"])
    for (show_id, ep_id), g in grouped.groupby(["collection_id", "episode_id"], sort=False):
        out.setdefault(str(show_id), {})[ep_id] = " ".join(g["bert_text"].fillna("").tolist())
    return out


def build_within_show_index(episode_texts: dict[str, dict[str, str]],
                             n: int = NGRAM_SIZE, min_frac: float = WITHIN_SHOW_MIN_EPISODE_FRAC,
                             min_episodes: int = MIN_EPISODES_FOR_CHECK) -> tuple[dict, dict]:
    show_index: dict[str, set] = {}
    show_meta: dict[str, dict] = {}
    for show_id, eps in episode_texts.items():
        n_eps = len(eps)
        meta = {"n_episodes": n_eps, "checked": n_eps >= min_episodes}
        if n_eps < min_episodes:
            show_index[show_id] = set()
            show_meta[show_id] = meta
            continue
        threshold = max(2, math.ceil(min_frac * n_eps))
        ngram_eps: dict[tuple, set] = defaultdict(set)
        for ep_id, text in eps.items():
            words = normalize_words(text)
            for g in set(ngrams(words, n)):
                ngram_eps[g].add(ep_id)
        idx = {g for g, epset in ngram_eps.items() if len(epset) >= threshold}
        show_index[show_id] = idx
        meta["threshold_episodes"] = threshold
        meta["n_within_show_boilerplate_ngrams"] = len(idx)
        show_meta[show_id] = meta
    return show_index, show_meta


def chunk_coverage(words: list[str], show_ngrams: set, n: int = NGRAM_SIZE) -> tuple[float, bool]:
    L = len(words)
    if L == 0:
        return 0.0, False
    covered = np.zeros(L, dtype=bool)
    if L >= n and show_ngrams:
        for i in range(L - n + 1):
            if tuple(words[i:i + n]) in show_ngrams:
                covered[i:i + n] = True
    for phrase in SPONSOR_PHRASE_WORDS:
        pl = len(phrase)
        if L >= pl:
            for i in range(L - pl + 1):
                if tuple(words[i:i + pl]) == phrase:
                    covered[i:i + pl] = True
    frac = float(covered.mean())
    return frac, frac >= COVERAGE_FLAG_THRESHOLD


def step2_corpus_wide(chunks_df: pd.DataFrame, out_dir: Path,
                       extreme_chunk_ids: set[str]) -> dict:
    episode_texts = build_episode_texts(chunks_df)
    show_index, show_meta = build_within_show_index(episode_texts)

    n_shows_checked = sum(1 for m in show_meta.values() if m["checked"])
    n_shows_total = len(show_meta)
    print(f"[step2] within-show boilerplate index built for {n_shows_checked}/{n_shows_total} "
          f"shows (>= {MIN_EPISODES_FOR_CHECK} transcribed episodes required)")

    coverages = np.empty(len(chunks_df))
    flags = np.empty(len(chunks_df), dtype=bool)
    show_ids = chunks_df["collection_id"].astype(str).to_numpy()
    texts = chunks_df["bert_text"].tolist()
    for i, (sid, text) in enumerate(zip(show_ids, texts)):
        words = normalize_words(text)
        frac, flag = chunk_coverage(words, show_index.get(sid, set()))
        coverages[i] = frac
        flags[i] = flag

    chunks_df = chunks_df.copy()
    chunks_df["boilerplate_coverage_frac"] = coverages
    chunks_df["boilerplate_flagged"] = flags

    overall_frac = float(flags.mean())

    per_show = chunks_df.groupby("collection_id").agg(
        show_name=("show_name", "first"),
        n_chunks=("chunk_id", "size"),
        n_flagged=("boilerplate_flagged", "sum"),
    )
    per_show["pct_flagged"] = 100 * per_show["n_flagged"] / per_show["n_chunks"]
    per_show = per_show.sort_values("pct_flagged", ascending=False)

    hist_edges = [0, 1, 5, 10, 20, 30, 50, 75, 100.0001]
    hist_labels = ["0%", "0-1%", "1-5%", "5-10%", "10-20%", "20-30%", "30-50%", "50-75%", "75-100%"]
    cats = pd.cut(per_show["pct_flagged"], bins=[-0.0001] + hist_edges[1:], labels=hist_labels[1:])
    # separate the true-zero bucket out explicitly since it's the expected mode
    zero_mask = per_show["pct_flagged"] == 0
    hist = {"0%": int(zero_mask.sum())}
    hist.update(cats[~zero_mask].value_counts().reindex(hist_labels[1:], fill_value=0).to_dict())

    tail_mask = chunks_df["chunk_id"].isin(extreme_chunk_ids)
    tail_frac = float(chunks_df.loc[tail_mask, "boilerplate_flagged"].mean()) if tail_mask.sum() else float("nan")

    top_shows = per_show.head(15).reset_index()[
        ["collection_id", "show_name", "n_chunks", "n_flagged", "pct_flagged"]
    ].to_dict("records")

    per_show_path = out_dir / "boilerplate_per_show.csv"
    per_show.reset_index().to_csv(per_show_path, index=False)

    return {
        "n_shows_checked_for_within_show_ngrams": n_shows_checked,
        "n_shows_total": n_shows_total,
        "corpus_wide_flagged_fraction": round(overall_frac, 4),
        "n_chunks_total": len(chunks_df),
        "n_chunks_flagged": int(flags.sum()),
        "tail_fraction_flagged": round(tail_frac, 4) if not np.isnan(tail_frac) else None,
        "n_tail_chunks_checked": int(tail_mask.sum()),
        "tail_vs_corpus_ratio": (round(tail_frac / overall_frac, 2)
                                  if overall_frac > 0 and not np.isnan(tail_frac) else None),
        "histogram_pct_flagged_per_show": hist,
        "top15_most_contaminated_shows": top_shows,
        "per_show_csv": per_show_path.name,
        "_chunks_df_with_flags": chunks_df,  # internal, consumed by step3 -- stripped before json dump
    }


# --------------------------------------------------------------------- #
# Step 3: correlate contamination with things we care about
# --------------------------------------------------------------------- #

def step3_correlations(chunks_df_flagged: pd.DataFrame) -> dict:
    per_show = chunks_df_flagged.groupby("collection_id").agg(
        n_chunks=("chunk_id", "size"), n_flagged=("boilerplate_flagged", "sum"),
    )
    per_show["boilerplate_share"] = per_show["n_flagged"] / per_show["n_chunks"]

    cf = load_cfscore()
    cf = cf.copy()
    cf.index = cf.index.astype(np.int64)
    show_meta = load_show_metadata()["features"]

    joined = per_show.join(cf[["avg_host_cfscore"]], how="left")
    joined = joined.join(show_meta[["avg_episode_minutes", "episodes_per_week"]], how="left")

    results = {}
    for col in ["avg_host_cfscore", "avg_episode_minutes", "episodes_per_week"]:
        sub = joined[["boilerplate_share", col]].dropna()
        n = len(sub)
        if n < 10:
            results[col] = {"n": n, "note": "insufficient coverage for correlation"}
            continue
        r_p, p_p = pearsonr(sub["boilerplate_share"], sub[col])
        r_s, p_s = spearmanr(sub["boilerplate_share"], sub[col])
        results[col] = {"n": n, "pearson_r": float(r_p), "pearson_p": float(p_p),
                         "spearman_r": float(r_s), "spearman_p": float(p_s)}
    return results


# --------------------------------------------------------------------- #
# Step 4: verdict
# --------------------------------------------------------------------- #

def step4_verdict(overall_frac: float, cfscore_r: float | None, tail_frac: float | None) -> dict:
    cfscore_correlated = cfscore_r is not None and abs(cfscore_r) > 0.3
    if overall_frac < 0.02 and not cfscore_correlated:
        severity = "minor"
        recommendation = ("Note as caveat in the dimension writeup; no re-clean/re-embed needed. "
                           "Tail contamination is real but doesn't move the corpus-wide picture.")
    elif overall_frac <= 0.05 or (overall_frac < 0.10 and not cfscore_correlated):
        severity = "moderate"
        recommendation = ("Re-read PC4 (and any other ad-driven pole) after stripping the flagged "
                           "chunks before finalizing its name; the cross-PC meta-pattern and other "
                           "PCs' names still stand as-is.")
    else:
        severity = "material"
        recommendation = ("Run a de-boilerplate pass (extend the boilerplate filter to a within-show "
                           "rule) and re-embed before locking ANY dimension claim or the ideology R^2.")
    if cfscore_correlated:
        recommendation += (" CFscore correlation is non-trivial (|r|>0.3) -- the ideology R^2=0.21 "
                            "result needs a robustness re-run on de-boilerplated text regardless of "
                            "the overall severity bucket.")
    return {
        "severity": severity,
        "corpus_wide_flagged_fraction": overall_frac,
        "cfscore_correlation": cfscore_r,
        "cfscore_correlated_flag": cfscore_correlated,
        "tail_fraction_flagged": tail_frac,
        "recommendation": recommendation,
    }


def run(target: int) -> None:
    OUT = pipeline_config.OUTPUT_DIR
    print(f"[load] chunks_{target}_bert.csv (the CSV actually embedded for target={target})")
    chunks_df = pd.read_csv(OUT / f"chunks_{target}_bert.csv")

    print("\n[step1] classifying pc_extremes_1..5.csv ...")
    step1 = step1_classify_extremes(OUT)
    for pc, r in step1["per_pc"].items():
        print(f"  {pc}: {r['n_nonsubstantive']}/{r['n_total']} non-substantive "
              f"({r['pct_nonsubstantive']}%) -- {r['counts']}")
    print(f"  OVERALL: {step1['overall']['pct_nonsubstantive']}% of extreme slots non-substantive, "
          f"{step1['overall']['cross_pc_near_duplicates_found']} cross-PC near-duplicates found")

    extreme_chunk_ids = set()
    for pc in range(1, 6):
        extreme_chunk_ids.update(pd.read_csv(OUT / f"pc_extremes_{pc}.csv")["chunk_id"])

    print("\n[step2] corpus-wide within-show boilerplate scan (this is slow-ish, ~40k chunks) ...")
    step2 = step2_corpus_wide(chunks_df, OUT, extreme_chunk_ids)
    chunks_df_flagged = step2.pop("_chunks_df_with_flags")
    print(f"  corpus-wide flagged fraction: {step2['corpus_wide_flagged_fraction']:.4f} "
          f"({step2['n_chunks_flagged']}/{step2['n_chunks_total']} chunks)")
    print(f"  tail (PC extreme) flagged fraction: {step2['tail_fraction_flagged']} "
          f"(ratio to corpus: {step2['tail_vs_corpus_ratio']}x)")
    print(f"  per-show histogram: {step2['histogram_pct_flagged_per_show']}")

    print("\n[step3] boilerplate share vs CFscore / duration / episodes_per_week ...")
    step3 = step3_correlations(chunks_df_flagged)
    for k, v in step3.items():
        print(f"  {k}: {v}")

    cfscore_r = step3.get("avg_host_cfscore", {}).get("pearson_r")
    verdict = step4_verdict(step2["corpus_wide_flagged_fraction"], cfscore_r,
                             step2["tail_fraction_flagged"])
    print(f"\n=== VERDICT: {verdict['severity'].upper()} ===")
    print(verdict["recommendation"])

    report = {
        "target": target, "generated_at": datetime.now(timezone.utc).isoformat(),
        "ngram_size": NGRAM_SIZE, "within_show_min_episode_frac": WITHIN_SHOW_MIN_EPISODE_FRAC,
        "min_episodes_for_check": MIN_EPISODES_FOR_CHECK, "coverage_flag_threshold": COVERAGE_FLAG_THRESHOLD,
        "near_dup_cosine_threshold": NEAR_DUP_COSINE,
        "step1_extremes_audit": step1,
        "step2_corpus_wide": step2,
        "step3_correlations": step3,
        "step4_verdict": verdict,
    }
    report_path = OUT / f"boilerplate_audit_report_{target}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[report] -> {report_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-words", type=int, default=TARGET)
    args = ap.parse_args()
    run(args.target_words)


if __name__ == "__main__":
    main()
