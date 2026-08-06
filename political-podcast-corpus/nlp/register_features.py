"""Step 1 of the register-quantification task: compute 6 a-priori "analytical
vs reactive" register features per chunk, on the SAME chunk-level text/target
(target_words=500, data/output/chunks_500_bert.csv, same 40,242-chunk set
used to build data/output/chunk_embeddings_500_weighted.npy and therefore the
same set the chunk-level PCs in nlp/embed_dimensions.py are computed on).

Features are FIXED a priori, before any PC correlation is looked at (no
circularity -- see module docstring in the caller / task brief):

  citation_density              -- rate of citation-like markers (et al.,
                                    "v." case cites, "study/paper/report
                                    found/shows/says", years in parens,
                                    2+-word capitalized proper-noun runs not
                                    at sentence start).
  abstraction                   -- 5 - mean Brysbaert-et-al-2014 concreteness
                                    of content-word lemmas (so HIGHER =
                                    more abstract = hypothesized more
                                    analytical). Lexicon: data/13428_2013_403
                                    _MOESM1_ESM.xlsx (Brysbaert, Warriner &
                                    Kuperman 2014, Behavior Research Methods,
                                    39,954 word lemmas, 1-5 concreteness
                                    scale), shipped as
                                    data/output/brysbaert_concreteness.csv.
  present_tense_news_markers    -- rate of immediacy/news markers ("today",
                                    "breaking", "just", "this morning",
                                    "right now", "developing").
  entity_mix                    -- n_ORG / (n_PERSON + 1), spaCy NER
                                    (en_core_web_sm). Higher = more
                                    institutional/conceptual, lower = more
                                    dense with current political personalities.
  sentence_complexity           -- composite (mean of two globally z-scored
                                    components): mean sentence length (tokens
                                    per sentence) and subordinating-conjunction
                                    rate (SCONJ tokens per sentence, spaCy POS).
  first_vs_third_person         -- first-person pronoun rate minus
                                    third-person pronoun rate (per 100 words).

Run: python -m nlp.register_features [--n-process N] [--batch-size N]
Output: data/output/register_features_500.csv (one row per chunk_id) +
        data/output/register_features_500_report.json (coverage/distributions,
        degeneracy flags).
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import spacy

from pipeline import config as pipeline_config

TARGET = 500
CONCRETENESS_PATH = Path("data/output/brysbaert_concreteness.csv")
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
THIRD_PERSON = {
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
}

NEWS_MARKER_RE = re.compile(
    r"\b(today|breaking|just in|this morning|right now|developing story|developing|"
    r"tonight|this week|as we speak|moments ago|just moments ago|just announced|"
    r"just released|happening now)\b", re.I,
)
ETAL_RE = re.compile(r"\bet al\.?\b", re.I)
CASE_CITE_RE = re.compile(r"\b[A-Z][a-zA-Z.'-]+\s+v\.?\s+[A-Z][a-zA-Z.'-]+")
STUDY_RE = re.compile(
    r"\b(study|paper|report|research|survey|poll|analysis)\b\s+\w*\s*(found|shows?|"
    r"suggests?|argues?|concludes?|reveals?|indicates?)", re.I,
)
YEAR_PAREN_RE = re.compile(r"\(\s*(1[5-9]\d{2}|20\d{2})\s*\)")
# 2+ consecutive capitalized-word runs, used as a cheap proper-noun-sequence
# proxy for citation/named-entity-like density; sentence-initial single caps
# are excluded by requiring >=2 consecutive tokens.
PROPER_RUN_RE = re.compile(r"\b(?:[A-Z][a-z]+\s+){1,}[A-Z][a-z]+\b")


def load_concreteness_lookup() -> dict[str, float]:
    df = pd.read_csv(CONCRETENESS_PATH)
    return dict(zip(df["word"], df["conc_mean"]))


def citation_density(text: str) -> float:
    n_words = max(len(text.split()), 1)
    hits = (
        len(ETAL_RE.findall(text))
        + len(CASE_CITE_RE.findall(text))
        + len(STUDY_RE.findall(text))
        + len(YEAR_PAREN_RE.findall(text))
        + len(PROPER_RUN_RE.findall(text))
    )
    return 1000.0 * hits / n_words


def news_marker_rate(text: str) -> float:
    n_words = max(len(text.split()), 1)
    return 1000.0 * len(NEWS_MARKER_RE.findall(text)) / n_words


def process_chunk(doc, conc_lookup: dict[str, float]) -> dict:
    tokens = [t for t in doc if not t.is_space]
    n_tokens = max(len(tokens), 1)
    sents = list(doc.sents)
    n_sents = max(len(sents), 1)

    conc_vals = [
        conc_lookup[t.lemma_.lower()]
        for t in tokens
        if t.pos_ in CONTENT_POS and t.lemma_.lower() in conc_lookup
    ]
    n_conc_matched = len(conc_vals)
    abstraction = (5.0 - float(np.mean(conc_vals))) if conc_vals else np.nan

    n_person = sum(1 for e in doc.ents if e.label_ == "PERSON")
    n_org = sum(1 for e in doc.ents if e.label_ == "ORG")
    entity_mix = n_org / (n_person + 1.0)

    n_sconj = sum(1 for t in tokens if t.pos_ == "SCONJ")
    mean_sent_len = n_tokens / n_sents
    subord_rate = n_sconj / n_sents

    words_lower = [t.text.lower() for t in tokens if t.is_alpha]
    n_words_lower = max(len(words_lower), 1)
    n_first = sum(1 for w in words_lower if w in FIRST_PERSON)
    n_third = sum(1 for w in words_lower if w in THIRD_PERSON)
    first_rate = 100.0 * n_first / n_words_lower
    third_rate = 100.0 * n_third / n_words_lower

    return {
        "n_tokens": n_tokens,
        "n_sents": n_sents,
        "n_conc_matched": n_conc_matched,
        "abstraction": abstraction,
        "n_person_ents": n_person,
        "n_org_ents": n_org,
        "entity_mix": entity_mix,
        "mean_sent_len_raw": mean_sent_len,
        "subord_rate_raw": subord_rate,
        "first_person_rate": first_rate,
        "third_person_rate": third_rate,
        "first_vs_third_person": first_rate - third_rate,
    }


def run(n_process: int, batch_size: int, corpus: str = "dirty") -> None:
    OUT = pipeline_config.OUTPUT_DIR
    conc_lookup = load_concreteness_lookup()
    print(f"[concreteness] {len(conc_lookup)} word lemmas loaded from Brysbaert et al. 2014")

    if corpus == "clean":
        # The ad-span-CLEANED corpus is RE-CHUNKED after excision (nlp/adspan_rechunk.py),
        # so chunk_id is reused but the underlying text shifts: comparing the two files
        # shows 86% of shared ids carry different text, and cleaned chunks are often
        # LONGER than the dirty chunk of the same id. Reusing register_features_500.csv
        # against the cleaned corpus would therefore attach style features to the wrong
        # text, not merely to stale text. Recomputation is mandatory, not cosmetic.
        # Uses `text` (raw-cased, post-excision) -- NOT `clean_text`, which is lowercased
        # and pruned for LDA/STM and would silently zero out the capitalization-dependent
        # features (case cites, proper-noun runs, NER).
        src = OUT / "adspan" / f"chunks_{TARGET}_nolemma_adclean.csv"
        chunks = pd.read_csv(src, usecols=["chunk_id", "text"]).rename(
            columns={"text": "bert_text"})
        chunks = chunks.sort_values("chunk_id").reset_index(drop=True)
        print(f"[chunks] source={src.name} (ad-span-cleaned, post-excision raw text)")
    else:
        chunks = pd.read_csv(OUT / f"chunks_{TARGET}_bert.csv", usecols=["chunk_id", "bert_text"])
        meta = json.loads((OUT / f"chunk_embeddings_{TARGET}_weighted.meta.json").read_text())
        chunk_id_order = meta["chunk_id_order"]
        chunks = chunks.set_index("chunk_id").loc[chunk_id_order].reset_index()
    chunks["bert_text"] = chunks["bert_text"].fillna("")
    n = len(chunks)
    print(f"[chunks] {n} chunks (corpus={corpus})")

    print("[regex features] computing citation_density, present_tense_news_markers ...")
    chunks["citation_density"] = chunks["bert_text"].map(citation_density)
    chunks["present_tense_news_markers"] = chunks["bert_text"].map(news_marker_rate)

    print(f"[spacy] loading en_core_web_sm, processing {n} chunks "
          f"(n_process={n_process}, batch_size={batch_size}) ...")
    nlp = spacy.load("en_core_web_sm")
    t0 = datetime.now()
    rows = []
    texts = chunks["bert_text"].tolist()
    for i, doc in enumerate(nlp.pipe(texts, n_process=n_process, batch_size=batch_size)):
        rows.append(process_chunk(doc, conc_lookup))
        if (i + 1) % 5000 == 0:
            elapsed = (datetime.now() - t0).total_seconds()
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"  ...{i + 1}/{n} chunks ({elapsed:.0f}s elapsed, {rate:.0f} chunks/s, "
                  f"ETA {eta:.0f}s)", flush=True)
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"[spacy] done in {elapsed:.0f}s")

    spacy_df = pd.DataFrame(rows)
    chunks = pd.concat([chunks.reset_index(drop=True), spacy_df], axis=1)

    # sentence_complexity: composite of two globally z-scored components
    for col in ["mean_sent_len_raw", "subord_rate_raw"]:
        mu, sd = chunks[col].mean(), chunks[col].std()
        chunks[f"z_{col}"] = (chunks[col] - mu) / sd
    chunks["sentence_complexity"] = chunks[["z_mean_sent_len_raw", "z_subord_rate_raw"]].mean(axis=1)

    feature_cols = [
        "citation_density", "abstraction", "present_tense_news_markers",
        "entity_mix", "sentence_complexity", "first_vs_third_person",
    ]

    coverage = {}
    degenerate_flags = []
    for f in feature_cols:
        s = chunks[f]
        n_nonnull = int(s.notna().sum())
        desc = s.describe(percentiles=[.05, .25, .5, .75, .95])
        coverage[f] = {
            "n_nonnull": n_nonnull, "n_total": n, "pct_coverage": round(100 * n_nonnull / n, 2),
            "mean": float(desc["mean"]), "std": float(desc["std"]),
            "min": float(desc["min"]), "p5": float(desc["5%"]), "p25": float(desc["25%"]),
            "median": float(desc["50%"]), "p75": float(desc["75%"]), "p95": float(desc["95%"]),
            "max": float(desc["max"]),
        }
        if s.std(skipna=True) < 1e-6 or s.notna().sum() < 0.5 * n:
            degenerate_flags.append(f)
            print(f"[FLAG] {f} looks degenerate (std={s.std():.6f}, coverage={100 * n_nonnull / n:.1f}%)")

    print("\n[coverage/distribution summary]")
    for f in feature_cols:
        c = coverage[f]
        print(f"  {f:<28} cov={c['pct_coverage']:>5.1f}%  mean={c['mean']:.3f}  std={c['std']:.3f}  "
              f"[p5={c['p5']:.3f}, median={c['median']:.3f}, p95={c['p95']:.3f}]")

    concreteness_coverage = chunks["n_conc_matched"].sum() / chunks["n_tokens"].sum()
    print(f"\n[concreteness match rate] {concreteness_coverage * 100:.1f}% of content-word tokens "
          f"matched the Brysbaert lexicon")

    out_cols = ["chunk_id"] + feature_cols + [
        "n_tokens", "n_sents", "n_conc_matched", "n_person_ents", "n_org_ents",
        "mean_sent_len_raw", "subord_rate_raw", "first_person_rate", "third_person_rate",
    ]
    sfx = "_adclean" if corpus == "clean" else ""
    out_path = OUT / f"register_features_{TARGET}{sfx}.csv"
    chunks[out_cols].to_csv(out_path, index=False)
    print(f"\n[saved] -> {out_path}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_words": TARGET,
        "corpus": corpus,
        "n_chunks": n,
        "concreteness_lexicon": "Brysbaert, Warriner & Kuperman 2014, 39,954 word lemmas",
        "concreteness_content_word_match_rate": float(concreteness_coverage),
        "feature_coverage": coverage,
        "degenerate_features": degenerate_flags,
        "feature_definitions": {
            "citation_density": "per-1000-word rate: et al., case cites (X v. Y), "
                                 "study/paper+found/shows, years-in-parens, "
                                 "2+ consecutive capitalized-word runs",
            "abstraction": "5 - mean Brysbaert concreteness of content-word lemmas "
                           "(higher = more abstract)",
            "present_tense_news_markers": "per-1000-word rate of immediacy/news markers",
            "entity_mix": "n_ORG / (n_PERSON + 1), spaCy en_core_web_sm NER",
            "sentence_complexity": "mean of z(mean sentence length) and z(SCONJ per sentence)",
            "first_vs_third_person": "first-person rate minus third-person rate, per 100 words",
        },
    }
    report_path = OUT / f"register_features_{TARGET}{sfx}_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[report] -> {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-process", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--corpus", default="dirty", choices=["dirty", "clean"],
                    help="clean = ad-span-excised, re-chunked corpus")
    args = ap.parse_args()
    run(args.n_process, args.batch_size, args.corpus)


if __name__ == "__main__":
    main()
