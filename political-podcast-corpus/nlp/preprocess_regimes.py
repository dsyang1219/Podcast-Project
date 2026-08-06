"""Toggleable preprocessing pipeline for the ~500-word podcast passages
(chunk-level), Python/spaCy port of the original r/preprocess.R design.

Switched from R/quanteda to Python/spaCy mid-task: R's `preText` package
pulled in `topicmodels` as a build dependency, which requires the GNU
Scientific Library headers (libgsl-dev) at the OS level -- not installed,
and installing it needs sudo the agent doesn't have. spaCy is already a
first-class dependency of this project (nlp/clean.py uses it for the
existing Stage-1 cleaning), so this reimplements the same toggle set
without adding any new dependency. See nlp/pretext_manual.py for the
Denny & Spirling (2018) sensitivity-analysis reimplementation (preText
itself is R/GPL, not pip-installable either).

Operates on the RAW passage text (chunks_500_nolemma.csv's `text` column --
boilerplate-sentence-stripped but otherwise unprocessed) so these toggles
are the ONLY preprocessing applied; nothing is double-cleaned through the
existing nlp/clean.py path. Same passage boundaries as the existing
pipeline -- no re-segmentation.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import spacy
import spacy.lang.en

from .porter_stemmer import PorterStemmer

STOPWORD_FILE = Path("data/spoken_stopwords.txt")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]*")
_STUTTER_RE = re.compile(r"\b(\w+)\b(\s+\1\b)+", re.IGNORECASE)
BACKCHANNEL = {"mm", "mhm", "huh", "hm", "ha", "eh", "oh", "ah"}
KEEP_POS = {"NOUN", "PROPN", "VERB", "ADJ"}


@dataclass
class Toggles:
    lowercase: bool = True
    remove_punct: bool = True
    remove_numbers: bool = True
    remove_symbols: bool = True
    stopwords: bool = True
    ngrams: bool = False
    infrequent_terms: bool = True
    lemmatize: bool = False
    stem: bool = False
    pos_filter: bool = False
    disfluency: bool = True
    min_docfreq: int = 10
    max_docfreq_prop: float = 0.5


def regime_toggles(name: str) -> Toggles:
    base = Toggles()
    if name == "minimal":
        return Toggles(stopwords=False, infrequent_terms=False, disfluency=False)
    if name == "moderate":
        return base
    if name == "aggressive":
        return Toggles(ngrams=True, lemmatize=True)
    if name == "aggressive_pos":
        return Toggles(ngrams=True, lemmatize=True, pos_filter=True)
    if name == "stem_comparison":
        return Toggles(ngrams=True, stem=True, lemmatize=False)
    raise ValueError(f"unknown regime '{name}'")


def load_spoken_stopwords() -> set[str]:
    words = set()
    for line in STOPWORD_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.add(line)
    return words


_SPACY_STOPWORDS = None


def extended_stopwords() -> set[str]:
    global _SPACY_STOPWORDS
    if _SPACY_STOPWORDS is None:
        _SPACY_STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
    return set(_SPACY_STOPWORDS) | load_spoken_stopwords()


def collapse_stutters(text: str) -> str:
    prev = None
    out = text
    while prev != out:
        prev = out
        out = _STUTTER_RE.sub(r"\1", out)
    return out


# ---------------------------------------------------------------- spaCy --
_NLP = None


def load_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _NLP


def _tokenize_regex(text: str, toggles: Toggles) -> list[str]:
    if toggles.disfluency:
        text = collapse_stutters(text)
    if toggles.lowercase:
        text = text.lower()
    toks = _WORD_RE.findall(text) if toggles.remove_punct else text.split()
    if not toggles.remove_numbers:
        # regex already excludes digits; this branch only matters if a
        # future toggle combination wants numbers kept -- flagged, not
        # exercised by the 4 curated regimes (all remove numbers).
        pass
    return [t for t in toks if len(t) >= 2]


def annotate_spacy(texts: list[str], disfluency: bool, n_process: int = 1,
                    batch_size: int = 100, verbose: bool = True) -> list[list[tuple[str, str, str]]]:
    """One shared, expensive spaCy pass producing (text, lemma, pos) per
    token per doc. Cache-and-reuse this across any regimes that both need
    spaCy (aggressive / aggressive_pos differ only in downstream filtering
    of the SAME annotation, not in what gets tagged) -- see
    nlp/export_regime_tokens.py."""
    nlp = load_nlp()
    docs_in = [collapse_stutters(t) if disfluency else t for t in texts]
    out = []
    n = len(docs_in)
    for i, doc in enumerate(nlp.pipe(docs_in, n_process=n_process, batch_size=batch_size)):
        out.append([(tok.text, tok.lemma_, tok.pos_) for tok in doc if tok.is_alpha])
        if verbose and (i + 1) % 5000 == 0:
            print(f"[spacy] annotated {i + 1}/{n}...")
    return out


def tokens_from_annotation(annotated: list[list[tuple[str, str, str]]], toggles: Toggles) -> list[list[str]]:
    out = []
    for doc in annotated:
        toks = []
        for text, lemma, pos in doc:
            if toggles.pos_filter and pos not in KEEP_POS:
                continue
            form = lemma if toggles.lemmatize else text
            form = form.lower() if toggles.lowercase else form
            if len(form) < 2:
                continue
            toks.append(form)
        out.append(toks)
    return out


def _tokenize_spacy_batch(texts: list[str], toggles: Toggles, n_process: int = 1, batch_size: int = 100) -> list[list[str]]:
    ann = annotate_spacy(texts, toggles.disfluency, n_process=n_process, batch_size=batch_size)
    return tokens_from_annotation(ann, toggles)


def build_token_lists(texts: list[str], toggles: Toggles, n_process: int = 1, verbose: bool = True) -> list[list[str]]:
    needs_spacy = toggles.lemmatize or toggles.pos_filter
    if needs_spacy:
        if verbose:
            print(f"[preprocess] spaCy pass over {len(texts)} passages "
                  f"(lemmatize={toggles.lemmatize}, pos_filter={toggles.pos_filter}, n_process={n_process})...")
        token_lists = _tokenize_spacy_batch(texts, toggles, n_process=n_process)
    else:
        token_lists = [_tokenize_regex(t, toggles) for t in texts]

    if toggles.disfluency:
        token_lists = [[t for t in toks if t not in BACKCHANNEL] for toks in token_lists]

    if toggles.stopwords:
        sw = extended_stopwords()
        token_lists = [[t for t in toks if t not in sw] for toks in token_lists]

    stemmer = PorterStemmer() if toggles.stem else None
    if stemmer is not None:
        token_lists = [[stemmer.stem(t) for t in toks] for toks in token_lists]

    if toggles.ngrams:
        token_lists = compound_collocations(token_lists, verbose=verbose)

    return token_lists


# ------------------------------------------------------------ collocations --
def compound_collocations(token_lists: list[list[str]], top_n: int = 200, min_count: int = 30,
                           verbose: bool = True) -> list[list[str]]:
    unigram_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    total_unigrams = 0
    for toks in token_lists:
        unigram_counts.update(toks)
        total_unigrams += len(toks)
        for i in range(len(toks) - 1):
            bigram_counts[(toks[i], toks[i + 1])] += 1

    candidates = {bg: c for bg, c in bigram_counts.items() if c >= min_count}
    # PMI-style score: log( P(w1,w2) / (P(w1) * P(w2)) ), using raw counts
    # (monotonic in the same ranking; avoids float underflow at this N).
    def score(bg):
        w1, w2 = bg
        c = bigram_counts[bg]
        return c * total_unigrams / (unigram_counts[w1] * unigram_counts[w2])

    ranked = sorted(candidates, key=score, reverse=True)[:top_n]
    top_set = set(ranked)
    if verbose:
        print(f"[collocations] compounding top {len(top_set)} of {len(candidates)} candidates "
              f"(min_count={min_count}); examples: {ranked[:8]}")

    out = []
    for toks in token_lists:
        merged = []
        i = 0
        while i < len(toks):
            if i < len(toks) - 1 and (toks[i], toks[i + 1]) in top_set:
                merged.append(f"{toks[i]}_{toks[i + 1]}")
                i += 2
            else:
                merged.append(toks[i])
                i += 1
        out.append(merged)
    return out


# ---------------------------------------------------------- docfreq prune --
def apply_docfreq_prune(token_lists: list[list[str]], toggles: Toggles, verbose: bool = True) -> tuple[list[list[str]], dict]:
    n_docs = len(token_lists)
    vocab_before = set()
    tokens_before = 0
    df_counts: Counter = Counter()
    for toks in token_lists:
        vocab_before.update(toks)
        tokens_before += len(toks)
        df_counts.update(set(toks))

    if not toggles.infrequent_terms:
        acc = {
            "vocab_before": len(vocab_before), "vocab_after": len(vocab_before),
            "tokens_before": tokens_before, "tokens_after": tokens_before,
            "min_docfreq": None, "max_docfreq_prop": None, "n_docs": n_docs,
        }
        return token_lists, acc

    ceiling = toggles.max_docfreq_prop * n_docs
    kept_vocab = {w for w, c in df_counts.items() if toggles.min_docfreq <= c <= ceiling}
    out = [[t for t in toks if t in kept_vocab] for toks in token_lists]
    tokens_after = sum(len(t) for t in out)

    if verbose:
        print(f"[docfreq prune] vocab {len(vocab_before)} -> {len(kept_vocab)} "
              f"(-{100 * (1 - len(kept_vocab) / max(1, len(vocab_before))):.1f}%), "
              f"tokens {tokens_before} -> {tokens_after} "
              f"(-{100 * (1 - tokens_after / max(1, tokens_before)):.1f}%)")

    acc = {
        "vocab_before": len(vocab_before), "vocab_after": len(kept_vocab),
        "tokens_before": tokens_before, "tokens_after": tokens_after,
        "min_docfreq": toggles.min_docfreq, "max_docfreq_prop": toggles.max_docfreq_prop,
        "n_docs": n_docs,
    }
    return out, acc
