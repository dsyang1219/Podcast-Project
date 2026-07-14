"""Step 3 — explicit inclusion filter.

Rules are applied in a fixed order; the FIRST failing rule is the recorded
exclusion reason (with evidence). A show passes only if no rule fires.

  R1 no_feed_url          Apple lookup had no RSS feed URL
  R2 feed_unreachable     RSS fetch failed after retries
  R3 feed_unparseable     RSS fetched but not parseable XML / no channel
  R4 no_audio             feed has no episodes with audio enclosures
  R5 non_us_politics      publisher blocklist / title-description regex /
                          non-English rss_language / curated manual list
                          (see config), minus exceptions
  R6 insufficient_catalog hours_available < min_hours threshold
"""
from __future__ import annotations

import re

from . import config

RSS_STATUS_RULES = {
    "no_feed_url": "R1",
    "feed_unreachable": "R2",
    "feed_unparseable": "R3",
    "no_audio": "R4",
}


def check_non_us(row: dict, rss: dict) -> str | None:
    """Return evidence string if the show is non-US politics, else None."""
    cid = int(row["collection_id"])
    if cid in config.NON_US_EXCEPTIONS:
        return None
    if cid in config.NON_US_MANUAL:
        return f"curated list: {config.NON_US_MANUAL[cid]}"
    publisher = (row.get("publisher") or "").lower()
    for needle in config.NON_US_PUBLISHERS:
        if re.search(rf"\b{re.escape(needle)}\b", publisher):
            return f"publisher matches blocklist term '{needle}' ({row.get('publisher')})"
    lang = (rss.get("rss_language") or "").strip().lower()
    if lang and not lang.startswith("en"):
        return f"rss_language='{rss.get('rss_language')}' (non-English feed)"
    haystack = " ".join([
        row.get("show_name") or "", rss.get("rss_description") or "",
        rss.get("rss_author") or "",
    ]).lower()
    for pattern in config.NON_US_PATTERNS:
        m = re.search(pattern, haystack)
        if m:
            return f"title/description matches /{pattern}/ ('{m.group(0)}')"
    return None


def apply_filter(row: dict, rss: dict, min_hours: float) -> tuple[bool, str, str]:
    """Return (included, rule, evidence)."""
    status = rss["status"]
    if status in RSS_STATUS_RULES:
        return False, f"{RSS_STATUS_RULES[status]}_{status}", rss.get("error", "")

    evidence = check_non_us(row, rss)
    if evidence is not None:
        return False, "R5_non_us_politics", evidence

    if rss["hours_available"] < min_hours:
        return False, "R6_insufficient_catalog", (
            f"{rss['hours_available']} h available < {min_hours} h threshold")

    return True, "", ""
