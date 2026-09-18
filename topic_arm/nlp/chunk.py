"""Stage 2 — chunking.

Fixed-size, non-overlapping windows of ~`target_words` content words
(post-cleaning, see nlp/clean.py), snapped to the nearest sentence boundary
so a chunk never splits a sentence.

There is no real speaker-turn data available for this corpus — Whisper's
segments are ~30s ASR decoding windows, not diarized speaker turns (they
routinely contain several speaker changes; see project notes). Boundaries
are snapped to sentence breaks instead, which is what "don't split
mid-utterance" actually depends on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .clean import Sentence

DEFAULT_TARGET_WORDS = 500
DEFAULT_MIN_STUB_WORDS = 300


@dataclass
class Chunk:
    chunk_id: int
    text: str
    n_words: int
    clean_text: str  # space-joined lemmatized content-word tokens, for Stage 3 BOW
    bert_text: str = ""  # raw sentence text, same join as `text` — named separately
                         # so the BERT arm's input is explicit at the source
    speakers: str = ""  # no diarization available; left blank (see module docstring)
    sentences: list[Sentence] = field(default_factory=list)
    # The exact member Sentence objects, in order — so a downstream sentence
    # emitter (see nlp/emit_sentences.py) can attach chunk_id to the sentences
    # this chunk was already built from, instead of re-splitting bert_text and
    # risking desync from the boundary this chunker just decided.


def chunk_sentences(sentences: list[Sentence],
                     target_words: int = DEFAULT_TARGET_WORDS,
                     min_stub_words: int = DEFAULT_MIN_STUB_WORDS) -> list[Chunk]:
    """Greedily accumulate sentences until the running content-word count
    reaches `target_words`, then snap the boundary to whichever adjacent
    sentence break is numerically closer to the target (i.e. decide whether
    the sentence that crossed the threshold belongs in this chunk or starts
    the next one). Non-overlapping: the next chunk starts fresh from there.
    """
    chunks: list[Chunk] = []
    current: list[Sentence] = []
    current_words = 0

    def flush(sents: list[Sentence]) -> None:
        if not sents:
            return
        text = " ".join(s.text for s in sents)
        tokens = [t for s in sents for t in s.tokens]
        chunks.append(Chunk(
            chunk_id=len(chunks),
            text=text,
            n_words=len(tokens),
            clean_text=" ".join(tokens),
            bert_text=text,
            sentences=list(sents),
        ))

    for sent in sentences:
        n = len(sent.tokens)
        if current and current_words + n >= target_words:
            # decide whether `sent` belongs in the current chunk or the next
            before_gap = abs(target_words - current_words)
            after_gap = abs(target_words - (current_words + n))
            if after_gap <= before_gap:
                current.append(sent)
                current_words += n
                flush(current)
                current, current_words = [], 0
            else:
                flush(current)
                current, current_words = [sent], n
        else:
            current.append(sent)
            current_words += n

    if current_words >= min_stub_words:
        flush(current)

    return chunks
