"""Shared machinery for transcript-level ad/meta SPAN detection and excision.

Why spans and not chunks
------------------------
The earlier attempt (nlp/adclf_phase_a.py) classified whole ~500-content-word
passages as ad / meta / content and failed its gate at precision 0.233. Reading
its "false positives" shows the failure was not the model's: passages holding a
real OneSkin / Wayfair / Mint Mobile / Quince read were hand-labeled `content`
because the ad was a minority of the passage BY VOLUME. Both labels were
defensible, which is the tell that the UNIT was wrong. A ~500-content-word
passage (~1200 raw words here) routinely contains a 120-word sponsor read
wrapped in 1000 words of real political discourse; no whole-passage label can
be right about both halves.

So detection happens on the transcript, at sentence granularity, BEFORE any
analysis unit exists. Excise the ad sentences, then re-segment the cleaned
transcript. That ordering is what the ad-detection literature does (span
detection + excision, not document classification), and it is the only ordering
under which "remove the ads" does not also mean "remove real discourse".

Sentence segmentation
---------------------
Uses the same spaCy sentencizer configuration as nlp/clean.py so a sentence
here is the same object a sentence is downstream. Detection indices therefore
refer to the same units the excision operates on -- there is no second,
divergent segmentation between deciding and cutting.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import spacy

# ------------------------------------------------------------- windowing ----
# A window is the unit sent to the model. It must be long enough that the model
# can see an ad START and END (a sponsor read is typically 8-25 sentences), and
# short enough that sentence indices stay reliable. 36 sentences at this
# corpus's ~15 words/sentence is ~540 words -- comfortably more than one ad
# read, so ads are usually wholly contained rather than clipped by a boundary.
WINDOW_SENTENCES = 36

# Consecutive windows overlap by this many sentences so an ad that straddles a
# window edge is seen intact by at least one window. Merging (merge_spans) then
# unions the duplicate detections.
WINDOW_OVERLAP = 6

MIN_WINDOW_SENTENCES = 4  # don't send a trailing scrap of 1-3 sentences


@dataclass
class Span:
    """A half-open-at-neither-end sentence range: sentences [start, end] inclusive."""
    start: int
    end: int
    label: str          # "ad_sponsor" | "meta_boilerplate"
    confidence: float = 1.0

    def __contains__(self, i: int) -> bool:
        return self.start <= i <= self.end

    def indices(self) -> set[int]:
        return set(range(self.start, self.end + 1))

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end,
                "label": self.label, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, d: dict) -> "Span":
        return cls(int(d["start"]), int(d["end"]), str(d["label"]),
                   float(d.get("confidence", 1.0)))


@dataclass
class Window:
    """A contiguous run of sentences from one episode, sent as one request."""
    episode_id: str
    show_id: str
    window_index: int
    offset: int                 # index in the episode of this window's sentence 0
    sentences: list[str] = field(default_factory=list)

    @property
    def window_id(self) -> str:
        return f"{self.episode_id}::w{self.window_index}"

    def numbered_text(self) -> str:
        """Sentences numbered from 0, one per line -- the exact format the
        prompt tells the model to index against."""
        return "\n".join(f"[{i}] {s}" for i, s in enumerate(self.sentences))


_NLP = None


def load_sentencizer():
    """Same configuration as nlp/clean.py:load_nlp() -- sentencizer only, no
    parser/NER, so segmentation is identical to the downstream pipeline's."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        _NLP.add_pipe("sentencizer")
    return _NLP


def segments_to_text(segments: list[dict]) -> str:
    """Identical to nlp/clean.py:segments_to_text -- duplicated rather than
    imported so this module has no import-time spaCy/config dependency chain."""
    segments = sorted(segments, key=lambda s: s["start"])
    return " ".join(s["text"].strip() for s in segments if s.get("text", "").strip())


def sentences_for_text(text: str, nlp=None) -> list[str]:
    nlp = nlp or load_sentencizer()
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def sentences_for_texts(texts: list[str], nlp=None, n_process: int = 1,
                         batch_size: int = 25) -> list[list[str]]:
    """Batched equivalent of sentences_for_text over many episodes."""
    nlp = nlp or load_sentencizer()
    out = []
    for doc in nlp.pipe(texts, n_process=n_process, batch_size=batch_size):
        out.append([s.text.strip() for s in doc.sents if s.text.strip()])
    return out


def build_windows(sentences: list[str], episode_id: str, show_id: str,
                   window_sentences: int = WINDOW_SENTENCES,
                   overlap: int = WINDOW_OVERLAP,
                   min_window: int = MIN_WINDOW_SENTENCES) -> list[Window]:
    """Split an episode's sentences into overlapping windows.

    The final window is allowed to run short but is dropped entirely if it is
    below `min_window` AND already covered by the previous window's overlap --
    those last few sentences are still analysed, just by the previous request.
    """
    stride = window_sentences - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than window_sentences")

    windows: list[Window] = []
    offset = 0
    wi = 0
    n = len(sentences)
    while offset < n:
        chunk = sentences[offset:offset + window_sentences]
        # A trailing scrap already fully inside the previous window adds cost
        # and no coverage -- skip it.
        if len(chunk) < min_window and windows:
            break
        windows.append(Window(episode_id=episode_id, show_id=show_id,
                               window_index=wi, offset=offset, sentences=chunk))
        wi += 1
        if offset + window_sentences >= n:
            break
        offset += stride
    return windows


# ------------------------------------------------------------- span merge ----
def merge_spans(spans: list[Span], join_gap: int = 1) -> list[Span]:
    """Union overlapping/adjacent spans into maximal spans.

    `join_gap=1` also bridges a single unflagged sentence sitting between two
    flagged ones: a sponsor read broken by one "anyway," or a stray ASR
    fragment is one ad, not two. Larger gaps are NOT bridged -- that would
    start swallowing content between two genuinely separate ads.

    The merged label is "ad_sponsor" if any constituent was an ad (ad is the
    stronger, more actionable claim); confidence is the MINIMUM over
    constituents, so a merged span is only as trustworthy as its weakest part.
    """
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    out = [Span(ordered[0].start, ordered[0].end, ordered[0].label, ordered[0].confidence)]
    for s in ordered[1:]:
        last = out[-1]
        if s.start <= last.end + join_gap:
            last.end = max(last.end, s.end)
            last.label = "ad_sponsor" if "ad_sponsor" in (last.label, s.label) else last.label
            last.confidence = min(last.confidence, s.confidence)
        else:
            out.append(Span(s.start, s.end, s.label, s.confidence))
    return out


def excise(sentences: list[str], spans: list[Span]) -> tuple[list[str], set[int]]:
    """Drop every sentence covered by `spans`. Returns (kept sentences, removed
    index set). The removed set is kept so the audit trail can quote exactly
    what was cut, rather than inferring it from a diff."""
    removed: set[int] = set()
    for s in spans:
        removed |= s.indices()
    kept = [t for i, t in enumerate(sentences) if i not in removed]
    return kept, removed


def clamp_spans(spans: list[Span], n_sentences: int) -> list[Span]:
    """Drop/trim model-returned spans that fall outside the window.

    A model that hallucinates an index past the end of the window would
    otherwise silently excise nothing (harmless) or, once offset into episode
    coordinates, excise the WRONG sentences (not harmless). Clamp explicitly.
    """
    out = []
    for s in spans:
        start = max(0, min(s.start, n_sentences - 1))
        end = max(0, min(s.end, n_sentences - 1))
        if end < start:
            start, end = end, start
        out.append(Span(start, end, s.label, s.confidence))
    return out


# ----------------------------------------------------------- gold set I/O ----
def load_gold(path: Path) -> list[dict]:
    """gold_spans.json: list of {window_id, episode_id, show_id, show_name,
    stratum, sentences: [...], spans: [{start,end,label}], note}."""
    return json.loads(Path(path).read_text())


def gold_ad_indices(entry: dict, labels: tuple[str, ...] = ("ad_sponsor", "meta_boilerplate")) -> set[int]:
    idx: set[int] = set()
    for s in entry.get("spans", []):
        if s["label"] in labels:
            idx |= set(range(int(s["start"]), int(s["end"]) + 1))
    return idx


# --------------------------------------------------------- marker heuristics ----
# Used ONLY to stratify gold-set sampling (find windows likely to contain ads
# so the gold set isn't 95% pure content) and as the documented fallback
# stoplist if the LLM gate fails. Never used to grade the model.
AD_MARKERS = [
    r"\bpromo ?code\b", r"\buse code\b", r"\bcode [A-Z]{3,}\b", r"\bdiscount code\b",
    r"\bsponsored by\b", r"\bbrought to you by\b", r"\bour sponsor\b", r"\bthis episode is sponsored\b",
    r"\bsupport(?:ed)? (?:for|by) this (?:podcast|show|program)\b",
    r"\bgo to [a-z0-9.-]+\.com\b", r"\bvisit [a-z0-9.-]+\.com\b", r"\b[a-z0-9-]+\.com/[a-z]+\b",
    r"\bdot com slash\b", r"\bslash [a-z]+ for\b",
    r"\bfree shipping\b", r"\bfree trial\b", r"\b\d{1,2}% off\b", r"\bterms (?:and|&) conditions apply\b",
    r"\bstart your free\b", r"\bsign up (?:today|now) at\b", r"\blimited time offer\b",
]
META_MARKERS = [
    r"\bwelcome (?:back )?to (?:the )?[A-Z]", r"\bthanks for listening\b",
    r"\brate and review\b", r"\bleave us a review\b", r"\bsubscribe (?:to|on|wherever)\b",
    r"\bfollow us on\b", r"\bcoming up (?:next|after)\b", r"\bwe'll be right back\b",
    r"\bstick around\b", r"\bthat's (?:it|all) for (?:today|this)\b",
    r"\bproduced by\b", r"\bexecutive producer\b", r"\bi'm your host\b",
]
# Product/economics vocabulary that appears in BOTH real ads and real political
# discourse -- the ambiguity stratum the gate is really testing.
AMBIGUOUS_MARKERS = [
    r"\bgold\b", r"\bprecious metals\b", r"\bcrypto\b", r"\bbitcoin\b",
    r"\bhealth insurance\b", r"\bmedicare\b", r"\bretirement\b", r"\bportfolio\b",
    r"\binflation\b", r"\bthe fed\b", r"\bmortgage\b",
]

_AD_RE = re.compile("|".join(AD_MARKERS), re.IGNORECASE)
_META_RE = re.compile("|".join(META_MARKERS), re.IGNORECASE)
_AMBIG_RE = re.compile("|".join(AMBIGUOUS_MARKERS), re.IGNORECASE)


def marker_counts(text: str) -> dict[str, int]:
    return {
        "ad": len(_AD_RE.findall(text)),
        "meta": len(_META_RE.findall(text)),
        "ambiguous": len(_AMBIG_RE.findall(text)),
    }
