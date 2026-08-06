"""Orchestrator for Stage 1 (cleaning) + Stage 2 (chunking).

    python -m nlp.run_chunks [--target-words 300 500 800] [--min-stub-words 300]
                              [--lemmatize]

Reads every transcript under data/transcripts/<show_id>/<episode_id>_*.json,
joins show_name from data/output/corpus.csv (keyed on collection_id — the
transcript's own show_id/episode_id/episode_title are self-contained, so
this does NOT depend on the current sample_manifest.csv draw), cleans and
chunks each episode's text, and writes one chunks table per target word
count to data/output/chunks_<N>_<lemma|nolemma>.csv.

The lemmatize flag (default off — see nlp/config.py) is baked into every
output filename and into a run manifest so a lemma run and a no-lemma run
never get overwritten or confused with each other; run both and compare
topic coherence + human-read interpretability before picking one.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from pipeline import config as pipeline_config
from . import config as nlp_config
from .clean import CleanStats, build_boilerplate_index, clean_episodes, segments_to_text, load_nlp
from .chunk import chunk_sentences, DEFAULT_MIN_STUB_WORDS
from .emit_sentences import SENTENCE_FIELDS, sentence_rows_for_chunk

TRANSCRIPTS_DIR = pipeline_config.DATA_DIR / "transcripts"

CHUNK_FIELDS = ["show", "collection_id", "episode_id", "episode_title",
                "chunk_id", "text", "n_words", "speakers", "clean_text"]

BERT_FIELDS = ["chunk_id", "collection_id", "show_name", "episode_id",
               "chunk_index", "n_content_words", "bert_text"]

LENGTH_FIELDS = ["chunk_id", "n_content_words", "n_raw_words",
                  "est_wordpieces", "would_truncate_512"]

BERT_TEXT_SOURCE = "raw_sentences_boilerplate_removed_no_token_filter"

# Sentence-transformers truncate at 512 word-pieces; ~1.3 word-pieces per
# raw word is a rough English estimate, good enough to flag which chunk
# sizes are unsafe for the BERT arm before running any embedding model.
WORDPIECES_PER_WORD = 1.3
WORDPIECE_TRUNCATION_LIMIT = 512


def load_show_names() -> dict[str, str]:
    corpus = pd.read_csv(pipeline_config.OUTPUT_DIR / "corpus.csv",
                          usecols=["collection_id", "show_name"])
    return {str(cid): name for cid, name in zip(corpus.collection_id, corpus.show_name)}


def load_transcripts() -> list[dict]:
    """Each entry: show_id, episode_id, episode_title, text (raw joined transcript)."""
    out = []
    for path in sorted(TRANSCRIPTS_DIR.glob("*/*.json")):
        d = json.loads(path.read_text())
        text = segments_to_text(d["segments"])
        if not text:
            continue
        out.append({
            "show_id": str(d["show_id"]),
            "episode_id": d["episode_id"],
            "episode_title": d["episode_title"],
            "text": text,
        })
    return out


def _boilerplate_loss_report(show_names: dict[str, str],
                              show_stats: dict[str, CleanStats]) -> pd.DataFrame:
    rows = []
    for show_id, stats in show_stats.items():
        pct_sentences_dropped = (
            stats.n_sentences_dropped_boilerplate / stats.n_sentences
            if stats.n_sentences else 0.0
        )
        pct_tokens_dropped = (
            stats.n_words_dropped_boilerplate / stats.n_words
            if stats.n_words else 0.0
        )
        flagged = (
            pct_sentences_dropped > nlp_config.BOILERPLATE_LOSS_FLAG_THRESHOLD
            or pct_tokens_dropped > nlp_config.BOILERPLATE_LOSS_FLAG_THRESHOLD
        )
        rows.append({
            "show": show_names.get(show_id, ""),
            "collection_id": show_id,
            "pct_sentences_dropped": pct_sentences_dropped,
            "pct_tokens_dropped": pct_tokens_dropped,
            "flagged": flagged,
        })
    return pd.DataFrame(rows).sort_values("pct_tokens_dropped", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-words", type=int, nargs="+", default=[300, 500, 800],
                     help="chunk sizes (content words) to produce, one output file each")
    ap.add_argument("--min-stub-words", type=int, default=DEFAULT_MIN_STUB_WORDS,
                     help="drop a trailing chunk shorter than this")
    ap.add_argument("--lemmatize", dest="lemmatize", action="store_true",
                     default=nlp_config.DEFAULT_LEMMATIZE,
                     help="use spaCy lemmas instead of surface word forms "
                          f"(default: {nlp_config.DEFAULT_LEMMATIZE})")
    ap.add_argument("--no-lemmatize", dest="lemmatize", action="store_false",
                     help="force surface word forms even if the config default changes")
    ap.add_argument("--min-token-len", type=int, default=nlp_config.DEFAULT_MIN_TOKEN_LEN,
                     help=f"minimum content-token length (default: {nlp_config.DEFAULT_MIN_TOKEN_LEN})")
    ap.add_argument("--n-process", type=int, default=os.cpu_count() or 1,
                     help="spaCy worker processes for Stage 1 cleaning "
                          "(CPU-bound; default: all cores)")
    args = ap.parse_args()

    lemma_tag = "lemma" if args.lemmatize else "nolemma"

    show_names = load_show_names()
    episodes = load_transcripts()
    print(f"[load] {len(episodes)} transcribed episodes across "
          f"{len({e['show_id'] for e in episodes})} shows")

    show_texts: dict[str, list[str]] = {}
    for e in episodes:
        show_texts.setdefault(e["show_id"], []).append(e["text"])
    boilerplate = build_boilerplate_index(show_texts)
    print(f"[boilerplate] {len(boilerplate)} n-grams flagged as cross-show ad copy")

    nlp = load_nlp()

    cleaned: list[tuple[dict, list]] = []
    show_stats: dict[str, CleanStats] = defaultdict(CleanStats)

    n_sentences = 0
    two_char_survivors: set[str] = set()
    print(f"[clean] parsing {len(episodes)} episodes across n_process={args.n_process} "
          f"worker process(es)")
    results = clean_episodes([e["text"] for e in episodes], nlp, boilerplate,
                              lemmatize=args.lemmatize,
                              min_token_len=args.min_token_len,
                              n_process=args.n_process)
    for e, (sentences, stats) in zip(episodes, results):
        cleaned.append((e, sentences))
        show_stats[e["show_id"]] = show_stats[e["show_id"]] + stats
        n_sentences += len(sentences)
        for sent in sentences:
            two_char_survivors.update(t for t in sent.tokens if len(t) == 2)
    print(f"[clean lemmatize={args.lemmatize}] {n_sentences} sentences survived cleaning across all episodes")

    if args.min_token_len <= 2:
        print(f"[clean] {len(two_char_survivors)} distinct 2-char tokens survived "
              f"(min_token_len={args.min_token_len}); eyeball for junk, add genuine "
              f"noise to nlp/config.py:TWO_CHAR_STOPLIST:")
        print(f"  {sorted(two_char_survivors)}")

    loss_report = _boilerplate_loss_report(show_names, show_stats)
    loss_report_path = pipeline_config.OUTPUT_DIR / f"boilerplate_loss_report_{lemma_tag}.csv"
    loss_report.to_csv(loss_report_path, index=False)
    flagged = loss_report[loss_report["flagged"]]
    if len(flagged):
        print(f"[boilerplate] {len(flagged)} show(s) losing >"
              f"{nlp_config.BOILERPLATE_LOSS_FLAG_THRESHOLD:.0%} of sentences or tokens "
              f"to boilerplate stripping — see {loss_report_path.name}:")
        for _, row in flagged.iterrows():
            print(f"  {row['show']!r} ({row['collection_id']}): "
                  f"{row['pct_sentences_dropped']:.1%} sentences, "
                  f"{row['pct_tokens_dropped']:.1%} tokens dropped")
    else:
        print(f"[boilerplate] no show exceeds the "
              f"{nlp_config.BOILERPLATE_LOSS_FLAG_THRESHOLD:.0%} loss threshold "
              f"-> {loss_report_path.name}")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lemmatize": args.lemmatize,
        "min_token_len": args.min_token_len,
        "target_words": args.target_words,
        "min_stub_words": args.min_stub_words,
        "boilerplate_ngram_size": nlp_config.BOILERPLATE_NGRAM_SIZE,
        "boilerplate_min_shows": nlp_config.BOILERPLATE_MIN_SHOWS,
        "boilerplate_coverage": nlp_config.BOILERPLATE_COVERAGE,
        "boilerplate_loss_flag_threshold": nlp_config.BOILERPLATE_LOSS_FLAG_THRESHOLD,
        "filler_words": sorted(nlp_config.FILLER_WORDS),
        "two_char_stoplist": sorted(nlp_config.TWO_CHAR_STOPLIST),
        "stage3_df_no_above": nlp_config.STAGE3_DF_NO_ABOVE,
        "n_episodes": len(episodes),
        "n_shows": len({e["show_id"] for e in episodes}),
        "n_sentences_after_cleaning": n_sentences,
        "chunks_files": {},
    }

    manifest["bert_files"] = {}

    for target in args.target_words:
        rows = []
        bert_rows = []
        length_rows = []
        sentence_rows = []
        for e, sentences in cleaned:
            show_name = show_names.get(e["show_id"], "")
            # Single boundary decision shared by both arms: the LDA row, the
            # bert/length-report rows, and the per-sentence rows below are all
            # derived from the same `ch` objects, so chunk membership can't
            # drift between any of them.
            for ch in chunk_sentences(sentences, target_words=target,
                                       min_stub_words=args.min_stub_words):
                chunk_id = f"{e['episode_id']}_{ch.chunk_id}"
                rows.append({
                    "show": show_name,
                    "collection_id": e["show_id"],
                    "episode_id": e["episode_id"],
                    "episode_title": e["episode_title"],
                    "chunk_id": chunk_id,
                    "text": ch.text,
                    "n_words": ch.n_words,
                    "speakers": ch.speakers,
                    "clean_text": ch.clean_text,
                })
                bert_rows.append({
                    "chunk_id": chunk_id,
                    "collection_id": e["show_id"],
                    "show_name": show_name,
                    "episode_id": e["episode_id"],
                    "chunk_index": ch.chunk_id,
                    "n_content_words": ch.n_words,
                    "bert_text": ch.bert_text,
                })
                # n_raw_words is counted off bert_text itself, not ch.n_words
                # (content-word count) — the whole point of this report is
                # that bert_text reinflates with stopwords/filler that
                # n_words never counted, so it's meaningfully longer.
                n_raw_words = len(ch.bert_text.split())
                est_wordpieces = round(WORDPIECES_PER_WORD * n_raw_words)
                length_rows.append({
                    "chunk_id": chunk_id,
                    "n_content_words": ch.n_words,
                    "n_raw_words": n_raw_words,
                    "est_wordpieces": est_wordpieces,
                    "would_truncate_512": est_wordpieces > WORDPIECE_TRUNCATION_LIMIT,
                })
                sentence_rows.extend(sentence_rows_for_chunk(
                    chunk_id, e["episode_id"], show_name, e["show_id"], ch))
        out = pd.DataFrame(rows, columns=CHUNK_FIELDS)
        out_path = pipeline_config.OUTPUT_DIR / f"chunks_{target}_{lemma_tag}.csv"
        out.to_csv(out_path, index=False)
        words = out["n_words"]
        print(f"[chunk target={target} lemmatize={args.lemmatize}] {len(out)} chunks | "
              f"words: min {words.min()} p50 {words.median():.0f} max {words.max()} "
              f"-> {out_path.name}")
        manifest["chunks_files"][str(target)] = out_path.name

        # bert_text/length report don't depend on --lemmatize (no tokenization
        # on this path) — one file per target size, overwritten by whichever
        # lemma variant runs last (see run docstring / task discussion).
        bert_out = pd.DataFrame(bert_rows, columns=BERT_FIELDS)
        bert_out_path = pipeline_config.OUTPUT_DIR / f"chunks_{target}_bert.csv"
        bert_out.to_csv(bert_out_path, index=False)

        length_out = pd.DataFrame(length_rows, columns=LENGTH_FIELDS)
        length_out_path = pipeline_config.OUTPUT_DIR / f"bert_length_report_{target}.csv"
        length_out.to_csv(length_out_path, index=False)
        pct_truncate = length_out["would_truncate_512"].mean() if len(length_out) else 0.0
        print(f"[bert target={target}] {len(bert_out)} chunks -> {bert_out_path.name} | "
              f"{pct_truncate:.1%} would truncate at {WORDPIECE_TRUNCATION_LIMIT} "
              f"word-pieces -> {length_out_path.name}")

        # Sentence emission also doesn't depend on --lemmatize (raw text
        # only) — one file per target size, same rule as the bert CSV.
        sentence_out = pd.DataFrame(sentence_rows, columns=SENTENCE_FIELDS)
        sentence_out_path = pipeline_config.OUTPUT_DIR / f"sentences_{target}.csv"
        sentence_out.to_csv(sentence_out_path, index=False)
        print(f"[sentences target={target}] {len(sentence_out)} sentences across "
              f"{bert_out['chunk_id'].nunique()} chunks -> {sentence_out_path.name}")

        manifest["bert_files"][str(target)] = {
            "path": bert_out_path.name,
            "n_rows": len(bert_out),
            "bert_text_source": BERT_TEXT_SOURCE,
            "length_report": length_out_path.name,
            "sentences_path": sentence_out_path.name,
            "n_sentences": len(sentence_out),
            # `text` in the LDA CSV and `bert_text` here are the same raw
            # sentence join, intentionally: both trace back to Chunk.text /
            # Chunk.bert_text, which are set to the identical value in
            # nlp/chunk.py's flush(). Not a coincidence, not drift.
            "note": "chunks_<target>_<lemma>.csv's 'text' column and this "
                    "file's 'bert_text' column are the same underlying "
                    "value (Chunk.text == Chunk.bert_text at construction).",
        }

    manifest_path = pipeline_config.OUTPUT_DIR / f"chunks_manifest_{lemma_tag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] -> {manifest_path.name}")


if __name__ == "__main__":
    main()
