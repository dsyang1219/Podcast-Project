"""The span-detection prompt and the single-window call, shared by Phase A
(synchronous, gold set) and Phase B (Batch API, whole corpus).

Both phases MUST send byte-identical prompts or the gate measures a classifier
that never runs. Hence one module: build_messages() is the only place the
prompt exists.
"""
from __future__ import annotations

import json
import re
import time

from .adspan_common import Span, clamp_spans

# Pin the dated snapshot: "gpt-4o-mini" is a moving alias, and a mid-run
# reponse-behavior change would silently invalidate the Phase A gate that
# authorizes Phase B's excision.
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
CHEAPER_MODEL = "gpt-4.1-nano-2025-04-14"

# USD per 1M tokens, verified against the account's published pricing at run
# time. Batch API is half of each.
PRICING = {
    "gpt-4o-mini-2024-07-18": {"in": 0.15, "out": 0.60},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4.1-nano-2025-04-14": {"in": 0.10, "out": 0.40},
    "gpt-4.1-nano": {"in": 0.10, "out": 0.40},
}

SYSTEM_PROMPT = """You find ADVERTISING and SHOW-BOILERPLATE spans inside US political podcast transcripts.

You receive a numbered list of consecutive sentences from one transcript. Return the sentence-index RANGES that are advertising or show boilerplate. Everything you do not return is kept as political discourse.

LABELS

"ad_sponsor" -- a paid placement read into the show: host-read sponsor copy, produced/dynamically-inserted ads, and promos for OTHER organizations' products, services, or podcasts. Typical shape: a brand name, a benefits pitch, a personal testimonial from the host, then a call to action ("go to BRAND.com slash SHOW", "use code X", "free shipping", "30% off").

"meta_boilerplate" -- the show's or its own network's structural content: cold-open intro montages, "welcome to THE SHOW, I'm YOUR HOST", segment teases ("coming up...", "we'll be right back", "after a quick break"), outros, credits ("produced by...", "our engineer is..."), subscribe/rate/review/donate asks, and promos for SIBLING shows on the same network.

Everything else is content -- do not return it.

THE DISTINCTION THAT MATTERS MOST

Naming a product, company, or economic topic does NOT make a sentence an ad. What makes it an ad is that it FUNCTIONS as a paid placement.

  - A sponsor read for a gold or crypto investment firm -> ad_sponsor.
  - A host or guest analyzing gold as a hedge, the gold standard, crypto regulation, a company's business conduct, Medicare/Medicaid, health-insurance premiums, mortgage rates, or inflation -> CONTENT. Never mark these.
  - A guest's book discussed editorially in an interview -> CONTENT. A paid read for a book -> ad_sponsor.
  - Political soundbites inside a produced intro montage -> meta_boilerplate (they are branding, not discussion).

RULES

1. FAVOR PRECISION. When you are not sure, do NOT mark it. Leaving an ad in the corpus is a minor cost; deleting real political discourse corrupts the analysis. Mark only what you would defend as advertising or boilerplate.

2. NEVER mark a sentence that contains BOTH ad copy and genuine discussion. Transcription sometimes merges the end of an ad and the return to the conversation into one long sentence. Marking it would delete real discourse, so leave it unmarked and end your span at the previous sentence.

3. Prefer CONTIGUOUS spans. Ads and boilerplate run in continuous blocks. Do not return many scattered one-sentence spans; if a block is interrupted by one short filler sentence ("Okay.", "Right."), keep the block whole.

4. Span boundaries are the FIRST and LAST sentence of the block, inclusive. Include the segue that starts an ad ("You know, summer routines live or die by whether you can actually do them...") and the final call to action.

5. A window may contain no ads and no boilerplate at all. Returning an empty list is a correct and common answer.

OUTPUT

Strict JSON only, no prose:
{"spans": [{"start": <int>, "end": <int>, "label": "ad_sponsor"|"meta_boilerplate", "confidence": <0.0-1.0>}]}

Use the exact integer indices shown in brackets. Return {"spans": []} if there is nothing to mark."""

USER_TEMPLATE = """Sentences:

{numbered}

Return the ad_sponsor / meta_boilerplate spans as JSON."""


def build_messages(numbered_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(numbered=numbered_text)},
    ]


def parse_spans(content: str, n_sentences: int) -> tuple[list[Span], str | None]:
    """Parse the model's JSON into clamped Spans. Returns (spans, error).

    A parse failure returns NO spans -- i.e. it excises nothing. Failing
    closed is the only safe direction here: a garbled response must never be
    able to delete transcript.
    """
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return [], f"json_decode_error: {content[:200]!r}"

    raw = parsed.get("spans", []) if isinstance(parsed, dict) else parsed
    if not isinstance(raw, list):
        return [], f"spans_not_a_list: {content[:200]!r}"

    spans, bad = [], []
    for s in raw:
        try:
            label = str(s["label"])
            if label not in ("ad_sponsor", "meta_boilerplate"):
                bad.append(f"bad_label:{label}")
                continue
            spans.append(Span(int(s["start"]), int(s["end"]), label,
                               float(s.get("confidence", 1.0))))
        except (KeyError, TypeError, ValueError):
            bad.append(f"malformed_span:{s!r}")
    return clamp_spans(spans, n_sentences), ("; ".join(bad) if bad else None)


_RETRY_WAIT_RE = re.compile(r"try again in (\d+(?:\.\d+)?)(s|m)")


def detect_window(client, numbered_text: str, n_sentences: int,
                   model: str = DEFAULT_MODEL, max_retries: int = 6):
    """Synchronous single-window detection (Phase A). Returns
    (spans, error, prompt_tokens, completion_tokens)."""
    from openai import RateLimitError

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=build_messages(numbered_text),
                temperature=0,
                response_format={"type": "json_object"},
            )
            break
        except RateLimitError as e:
            m = _RETRY_WAIT_RE.search(str(e))
            wait = (float(m.group(1)) * (60 if m.group(2) == "m" else 1) + 2) if m \
                else min(60, 2 ** attempt)
            print(f"    [retry] rate limited, waiting {wait:.0f}s "
                  f"({attempt + 1}/{max_retries})...", flush=True)
            time.sleep(wait)
    else:
        raise RuntimeError(f"exhausted {max_retries} retries on rate limit")

    spans, err = parse_spans(resp.choices[0].message.content, n_sentences)
    return spans, err, resp.usage.prompt_tokens, resp.usage.completion_tokens
