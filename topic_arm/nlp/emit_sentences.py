"""BERT arm, deliverable 1 — sentence-level emission with parent chunk_id.

Builds one row per surviving Sentence, tagged with the chunk_id of the Chunk
the boundary-decision loop (nlp/chunk.py:chunk_sentences) already grouped it
into. Reads directly from Chunk.sentences — the literal Sentence objects the
chunker accumulated — rather than re-splitting Chunk.bert_text on punctuation,
so sentence_id/chunk_id alignment can't drift from the boundaries that were
already decided (see nlp/chunk.py's `sentences` field docstring).

No boundary logic here; this module only reads what chunk_sentences() already
computed.
"""
from __future__ import annotations

from .chunk import Chunk

SENTENCE_FIELDS = ["sentence_id", "chunk_id", "episode_id", "show_name",
                    "collection_id", "sent_index_in_chunk", "n_raw_words", "text"]


def sentence_rows_for_chunk(chunk_id: str, episode_id: str, show_name: str,
                             collection_id: str, chunk: Chunk) -> list[dict]:
    """One row per member Sentence of `chunk`, in original order.

    Token-empty sentences (all-stopword, e.g. "Yeah.") are included: they
    still carry a raw-text embedding vector and must be poolable into the
    chunk vector like any other sentence.
    """
    rows = []
    for i, sent in enumerate(chunk.sentences):
        rows.append({
            "sentence_id": f"{chunk_id}_s{i}",
            "chunk_id": chunk_id,
            "episode_id": episode_id,
            "show_name": show_name,
            "collection_id": collection_id,
            "sent_index_in_chunk": i,
            "n_raw_words": len(sent.text.split()),
            "text": sent.text,
        })
    return rows
