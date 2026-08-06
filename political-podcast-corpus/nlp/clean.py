"""Stage 1 — transcript cleaning.

Turns a raw Whisper transcript (segments with only start/end/text — no
speaker labels, see nlp/chunk.py docstring) into an ordered list of
sentences, each carrying both its original text and its cleaned content-word
tokens (surface form or lemma, depending on the `lemmatize` flag — see
nlp/config.py).

Two passes are needed across the whole batch of episodes:
  1. `build_boilerplate_index` — find word n-grams that recur near-verbatim
     across many *different* shows. Organic political commentary doesn't
     repeat that way; dynamic ad-insertion copy ("use code X at checkout")
     does. This catches sponsor boilerplate without a hand-maintained list.
  2. `clean_episode` — sentence-segment + clean one episode's text, dropping
     sentences that are mostly covered by boilerplate n-grams, and reporting
     how much of the episode that drop cost (see `CleanStats`).
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import spacy

from .config import (
    BOILERPLATE_COVERAGE,
    BOILERPLATE_MIN_SHOWS,
    BOILERPLATE_NGRAM_SIZE,
    DEFAULT_LEMMATIZE,
    DEFAULT_MIN_TOKEN_LEN,
    FILLER_WORDS,
    TWO_CHAR_STOPLIST,
)

_WORD_RE = re.compile(r"[a-z0-9']+")


def load_nlp():
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")
    return nlp


@dataclass
class Sentence:
    text: str
    tokens: list[str]   # cleaned content words (surface form or lemma)


@dataclass
class CleanStats:
    """Per-episode bookkeeping for the boilerplate-loss report.

    Sentences/words that are simply empty (segmentation artifacts) are not
    counted here — this tracks only what the boilerplate filter itself
    removes, so a show's loss percentage reflects that filter's behavior on
    that show specifically.
    """
    n_sentences: int = 0
    n_sentences_dropped_boilerplate: int = 0
    n_words: int = 0
    n_words_dropped_boilerplate: int = 0

    def __add__(self, other: "CleanStats") -> "CleanStats":
        return CleanStats(
            n_sentences=self.n_sentences + other.n_sentences,
            n_sentences_dropped_boilerplate=(
                self.n_sentences_dropped_boilerplate
                + other.n_sentences_dropped_boilerplate
            ),
            n_words=self.n_words + other.n_words,
            n_words_dropped_boilerplate=(
                self.n_words_dropped_boilerplate + other.n_words_dropped_boilerplate
            ),
        )


def segments_to_text(segments: list[dict]) -> str:
    segments = sorted(segments, key=lambda s: s["start"])
    return " ".join(s["text"].strip() for s in segments if s.get("text", "").strip())


def _normalize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _ngrams(words: list[str], n: int):
    for i in range(len(words) - n + 1):
        yield tuple(words[i:i + n])


def build_boilerplate_index(show_texts: dict[str, list[str]],
                             n: int = BOILERPLATE_NGRAM_SIZE,
                             min_shows: int = BOILERPLATE_MIN_SHOWS) -> set[tuple[str, ...]]:
    """`show_texts`: show_id -> list of raw episode texts for that show.

    Returns the set of n-grams that occur in at least `min_shows` distinct
    shows (occurrences within the same show only count once, so a show's
    own repeated intro/outro doesn't get flagged — this is specifically
    for boilerplate shared *across* shows, i.e. ad networks).
    """
    ngram_shows: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for show_id, texts in show_texts.items():
        seen_this_show = set()
        for text in texts:
            words = _normalize_words(text)
            seen_this_show.update(_ngrams(words, n))
        for ng in seen_this_show:
            ngram_shows[ng].add(show_id)
    return {ng for ng, shows in ngram_shows.items() if len(shows) >= min_shows}


def _is_boilerplate_sentence(text: str, boilerplate: set[tuple[str, ...]],
                              n: int = BOILERPLATE_NGRAM_SIZE,
                              coverage: float = BOILERPLATE_COVERAGE) -> bool:
    words = _normalize_words(text)
    if len(words) < n:
        return False
    grams = list(_ngrams(words, n))
    flagged = sum(1 for g in grams if g in boilerplate)
    return (flagged / len(grams)) >= coverage


def _process_doc(doc, boilerplate: set[tuple[str, ...]],
                  lemmatize: bool, min_token_len: int,
                  ) -> tuple[list[Sentence], CleanStats]:
    """The actual per-episode cleaning logic, given an already-parsed doc.

    Split out from clean_episode() so a batch of docs can be produced in
    parallel (see clean_episodes()) while this per-doc post-processing —
    which is pure-Python and not the bottleneck — stays single-threaded and
    byte-for-byte identical either way.
    """
    sentences: list[Sentence] = []
    stats = CleanStats()
    for sent in doc.sents:
        raw = sent.text.strip()
        if not raw:
            continue
        n_words_sent = len(_normalize_words(raw))
        stats.n_sentences += 1
        stats.n_words += n_words_sent
        if _is_boilerplate_sentence(raw, boilerplate):
            stats.n_sentences_dropped_boilerplate += 1
            stats.n_words_dropped_boilerplate += n_words_sent
            continue
        tokens = []
        for tok in sent:
            if not tok.is_alpha or tok.is_stop:
                continue
            form = tok.lemma_.lower() if lemmatize else tok.text.lower()
            if len(form) < min_token_len or form in FILLER_WORDS:
                continue
            if len(form) == 2 and form in TWO_CHAR_STOPLIST:
                continue
            tokens.append(form)
        sentences.append(Sentence(text=raw, tokens=tokens))
    return sentences, stats


def clean_episode(text: str, nlp, boilerplate: set[tuple[str, ...]],
                   lemmatize: bool = DEFAULT_LEMMATIZE,
                   min_token_len: int = DEFAULT_MIN_TOKEN_LEN,
                   ) -> tuple[list[Sentence], CleanStats]:
    """Sentence-segment `text` and produce cleaned content-word tokens per
    sentence, dropping sentences that are mostly ad-boilerplate.

    `lemmatize` selects the token form: lemma (spaCy) when True, lowercased
    surface form when False — see nlp/config.py for the rationale on why
    this is a toggle rather than a fixed choice.
    """
    doc = nlp(text)
    return _process_doc(doc, boilerplate, lemmatize, min_token_len)


def clean_episodes(texts: list[str], nlp, boilerplate: set[tuple[str, ...]],
                    lemmatize: bool = DEFAULT_LEMMATIZE,
                    min_token_len: int = DEFAULT_MIN_TOKEN_LEN,
                    n_process: int = 1, batch_size: int = 50,
                    ) -> list[tuple[list[Sentence], CleanStats]]:
    """Same per-episode result as calling clean_episode() on each of `texts`
    in order, but parses docs via nlp.pipe(n_process=n_process) so the CPU-
    bound spaCy parsing is spread across worker processes. n_process=1 is
    equivalent to (and exactly as fast as) a plain loop of clean_episode().

    Parallelizing doc production changes nothing about what gets produced —
    same model, same per-doc logic (_process_doc), just spread across
    processes instead of run serially — so results are identical to a
    serial run for any n_process.
    """
    docs = nlp.pipe(texts, n_process=n_process, batch_size=batch_size)
    return [_process_doc(doc, boilerplate, lemmatize, min_token_len) for doc in docs]
