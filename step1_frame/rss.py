"""Step 2 — RSS enrichment.

For each show in the frame, fetch its RSS feed (cached) and extract
per-episode and show-level metadata. Duration handling:

  * `duration_sec` comes from <itunes:duration> when present, parsed from
    H:MM:SS / MM:SS / plain-seconds forms.
  * When absent, we estimate from the enclosure byte length assuming 128 kbps
    MP3 and set `duration_source = "estimated_from_bytes"` so estimated hours
    are distinguishable downstream.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from . import config
from .http_util import get

ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"

EPISODE_FIELDS = [
    "collection_id", "show_name", "episode_title", "episode_description",
    "pub_date", "audio_url", "audio_bytes", "duration_sec", "duration_source",
]


def _text(el, tag, ns=None):
    child = el.find(f"{{{ns}}}{tag}" if ns else tag)
    return (child.text or "").strip() if child is not None and child.text else ""


def parse_duration(raw: str) -> float | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", raw):
        return float(raw)
    parts = raw.split(":")
    if all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts) and 2 <= len(parts) <= 3:
        parts = [float(p) for p in parts]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def _parse_pubdate(raw: str) -> str:
    try:
        return parsedate_to_datetime(raw).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return ""


def enrich_show(row: dict, refresh: bool = False) -> dict:
    """Fetch + parse one show's feed. Returns a dict with `status`,
    show-level fields, and an `episodes` list."""
    out = {
        "status": "ok", "error": "",
        "rss_description": "", "rss_language": "", "rss_author": "",
        "rss_owner_name": "", "episode_count": 0, "episodes_with_audio": 0,
        "hours_available": 0.0, "hours_estimated_share": 0.0,
        "avg_episode_minutes": 0.0, "episodes_per_week": 0.0,
        "first_pub_date": "", "last_pub_date": "",
        "episodes": [],
    }
    feed_url = (row.get("feed_url") or "").strip()
    if not feed_url:
        out.update(status="no_feed_url", error="no feedUrl in Apple lookup")
        return out

    body, meta = get(feed_url, config.RSS_CACHE_DIR, config.RSS_DELAY_SECONDS,
                     refresh=refresh)
    if body is None:
        out.update(status="feed_unreachable",
                   error=meta.get("error") or f"HTTP {meta.get('status')}")
        return out

    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        # tolerate stray control bytes, a common feed defect
        try:
            cleaned = re.sub(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]", b"", body)
            root = ET.fromstring(cleaned)
        except ET.ParseError:
            out.update(status="feed_unparseable", error=f"XML parse error: {exc}")
            return out

    channel = root.find("channel")
    if channel is None:
        out.update(status="feed_unparseable", error="no <channel> element")
        return out

    out["rss_description"] = _text(channel, "description")
    out["rss_language"] = _text(channel, "language")
    out["rss_author"] = _text(channel, "author", ITUNES_NS)
    owner = channel.find(f"{{{ITUNES_NS}}}owner")
    if owner is not None:
        out["rss_owner_name"] = _text(owner, "name", ITUNES_NS)

    episodes, pub_dts = [], []
    total_sec, est_sec = 0.0, 0.0
    for item in channel.findall("item"):
        enclosure = item.find("enclosure")
        audio_url = enclosure.get("url", "") if enclosure is not None else ""
        audio_bytes = None
        if enclosure is not None:
            try:
                audio_bytes = int(enclosure.get("length") or 0) or None
            except ValueError:
                audio_bytes = None

        dur = parse_duration(_text(item, "duration", ITUNES_NS))
        source = "itunes:duration"
        if dur is None and audio_bytes and audio_bytes > 100_000:
            dur = audio_bytes * 8 / 128_000  # assume 128 kbps
            source = "estimated_from_bytes"
        if dur is None:
            source = "missing"

        pub_iso = _parse_pubdate(_text(item, "pubDate"))
        if pub_iso:
            pub_dts.append(pub_iso)
        if audio_url and dur:
            total_sec += dur
            if source == "estimated_from_bytes":
                est_sec += dur

        episodes.append({
            "collection_id": row["collection_id"],
            "show_name": row["show_name"],
            "episode_title": _text(item, "title"),
            "episode_description": _text(item, "description")[:2000],
            "pub_date": pub_iso,
            "audio_url": audio_url,
            "audio_bytes": audio_bytes or "",
            "duration_sec": round(dur, 1) if dur else "",
            "duration_source": source,
        })

    with_audio = [e for e in episodes if e["audio_url"]]
    out["episodes"] = episodes
    out["episode_count"] = len(episodes)
    out["episodes_with_audio"] = len(with_audio)
    out["hours_available"] = round(total_sec / 3600, 2)
    out["hours_estimated_share"] = round(est_sec / total_sec, 3) if total_sec else 0.0
    durations = [e["duration_sec"] for e in with_audio if e["duration_sec"] != ""]
    out["avg_episode_minutes"] = round(
        sum(durations) / len(durations) / 60, 1) if durations else 0.0
    if pub_dts:
        pub_dts.sort()
        out["first_pub_date"], out["last_pub_date"] = pub_dts[0], pub_dts[-1]
        first = datetime.fromisoformat(pub_dts[0])
        last = datetime.fromisoformat(pub_dts[-1])
        span_weeks = max((last - first).days / 7, 1 / 7)
        out["episodes_per_week"] = round(len(pub_dts) / span_weeks, 2)

    if not with_audio:
        out.update(status="no_audio", error="no episodes with audio enclosures")
    return out
