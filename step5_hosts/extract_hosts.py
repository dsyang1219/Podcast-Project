"""
Host/author extraction for DIME/NOMINATE lookup prep.

For each show in the final corpus (data/output/corpus.csv), extracts the
host/author name(s), preferring RSS feed metadata over guessing:

    1. <itunes:author> at the channel level (corpus.csv: rss_author)
    2. <itunes:owner><itunes:name>          (corpus.csv: rss_owner_name)
    3. <author> (RSS 2.0, no itunes namespace) -- re-parsed from the cached
       feed body keyed by feed_url; NOT re-fetched.
    4. Parsed from the show title ("X with FIRST LAST", "X w/ FIRST LAST",
       or bare "The FIRST LAST Show/Podcast").
    5. UNKNOWN.

Each candidate string is classified as a person name, a publisher/org, or
self-referential (== the show's own title) before being accepted -- fields
routinely contain the network/publisher instead of the host (e.g.
"iHeartPodcasts", "FOX News Podcasts"), or just repeat the show title
verbatim. NOTE: corpus.csv's `publisher` column is NOT used for this
classification -- it matches rss_author in 219/220 rows (both ultimately
come from the same itunes:author field via Apple's ingestion), so it is not
an independent signal for "org vs person." It's carried into the output only
as context.

Multi-host fields ("Jon Favreau, Jon Lovett ... and Tommy Vietor") are split
and each plausible name emitted as its own row -- this matters for metadata
fields too, not just titles (e.g. Ruthless Podcast's itunes:author lists all
four hosts by name).

itunes:owner is nominally the feed's technical/administrative contact per
the Apple spec, not necessarily the host; a handful of owner values repeat
verbatim across unrelated shows in this corpus (e.g. "Marcus Brown" on both
Hugh Hewitt and Josh Hammer's shows) -- almost certainly a shared
producer/admin contact rather than either show's actual host. Those get
downgraded to medium confidence with a note instead of silently trusting them.

Usage: .venv/bin/python extract_hosts.py
Output: data/output/host_lookup.csv
"""
from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pandas as pd

CORPUS_CSV = Path("data/output/corpus.csv")
RSS_CACHE_DIR = Path("data/cache/rss")
OUT_CSV = Path("data/output/host_lookup.csv")

ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"

# Strings that mean "this is the publisher/platform, not a host." Given list
# from the task, extended with what's actually in this corpus (news orgs,
# think tanks, podcast networks, hosting platforms) so the classifier isn't
# just leaning on the generic suffix-word check below.
PUBLISHER_KEYWORDS = [
    "iheart", "fox news", "fox across", "npr", "msnbc", "ms now",
    "the bulwark", "salem", "premiere networks", "siriusxm", "vox media",
    "slate podcasts", "crooked media", "wnyc", "c-span", "hoover institution",
    "the dispatch", "meidastouch", "meidas media", "audioboom", "podtrac",
    "megaphone", "libsyn", "wondery", "prx", "abc news", "cbs news",
    "nbc news", "politico", "punchbowl", "tyt network", "new york post",
    "the atlantic", "the new yorker", "the economist", "the nation magazine",
    "the washington post", "the daily wire", "the daily signal",
    "the epoch times", "jacobin", "tablet magazine", "foreign affairs",
    "council on foreign relations", "gzero", "aei podcasts",
    "government accountability institute", "the lawfare institute",
    "center for strategic and international studies", "deseret news",
    "star tribune", "journal-constitution", "kuow", "fitsnews",
    "radio america", "realclearpolitics", "narrowcast", "evergreen podcasts",
    "opening arguments media", "hito media", "kayetriot", "scan media",
    "useful idiots", "msw media", "molucca media", "timcast",
    "mary trump media", "fawcett strategies", "democracy docket",
    "talking points memo", "justthenews", "just the news", "al jazeera",
    "monocle", "winkontent", "indivisible", "lemonada", "warroom",
    "trueanon", "sgt report", "x22 report", "politicon", "wanghaf",
    "renegade talk radio", "best of the left", "american conservative university",
    "ktrh", "podcast admin", "podcast wabcradio", "rumble", "spotify",
    "anchor.fm", "buzzsprout", "podbean", "simplecast", "transistor.fm",
    "acast", "image lab media", "snark rocket",
]

# Whole-word org markers -- catches "X Media", "X Network", "X LLC" etc.
# generically even when the exact brand isn't in the keyword list above.
ORG_SUFFIX_WORDS = {
    "media", "network", "networks", "studios", "podcasts", "radio", "news",
    "magazine", "institute", "foundation", "times", "report", "reports",
    "press", "productions", "communications", "group", "company",
    "companies", "enterprises", "llc", "inc", "ltd", "corp", "broadcasting",
    "strategies", "university",
}

# Function words that show up in phrases/brand names but essentially never
# inside a real person's name ("Puzzle in a Thunderstorm", "Best of the Left").
STOPWORDS_MID = {
    "the", "a", "an", "of", "in", "on", "at", "for", "and", "or", "is",
    "are", "with", "from", "to",
}

WITH_RE = re.compile(r"\b(?:with|w/)\s+(.+)$", re.IGNORECASE)
NAME_SHOW_RE = re.compile(
    r"^(?:The\s+)?([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){1,2})\s+(?:Show|Podcast)\b"
)

QUOTE_TRANS = str.maketrans({
    "’": "'", "‘": "'", "“": '"', "”": '"', "`": "'",
})


def normalize(s: object) -> str:
    if pd.isna(s):
        return ""
    s = str(s).translate(QUOTE_TRANS)
    return re.sub(r"\s+", " ", s).strip()


def norm_key(s: object) -> str:
    return normalize(s).lower()


def split_name_list(raw: str) -> list[str]:
    """'Josh Holmes, Comfortably Smug, Michael Duncan and John Ashbrook' -> 4 names."""
    s = normalize(raw)
    if not s:
        return []
    s = re.sub(r"\s+&\s+", ", ", s)
    s = re.sub(r"\s+\+\s+", ", ", s)
    s = re.sub(r"\s+\band\b\s+", ", ", s, flags=re.IGNORECASE)
    parts = [p.strip(" ,") for p in s.split(",")]
    return [p for p in parts if p]


def is_org_or_publisher(token: str) -> bool:
    low = token.lower()
    if any(kw in low for kw in PUBLISHER_KEYWORDS):
        return True
    words = {w.strip(".,").lower() for w in token.split()}
    return bool(words & ORG_SUFFIX_WORDS)


def is_self_title(token: str, show_name: str) -> bool:
    return norm_key(token) == norm_key(show_name)


def is_name_shaped(token: str) -> bool:
    token = token.strip().strip("\"'")
    if not token or any(ch.isdigit() for ch in token):
        return False
    words = token.split()
    if not (1 <= len(words) <= 6):
        return False
    if not token[0].isupper():
        return False
    lower_words = {w.strip(".,").lower() for w in words}
    if lower_words & STOPWORDS_MID:
        return False
    if len(words) == 1 and token.isupper() and len(token) <= 6:
        return False  # bare acronym/initialism, not a person's mononym
    return True


def classify_field(raw: str, show_name: str) -> tuple[list[str], str | None]:
    """Returns (kept_person_names, publisher_token_if_any)."""
    kept, publisher_hit = [], None
    for token in split_name_list(raw):
        if is_org_or_publisher(token):
            publisher_hit = publisher_hit or token
        elif is_self_title(token, show_name):
            continue
        elif is_name_shaped(token):
            kept.append(token)
    return kept, publisher_hit


def rss_author_from_cache(feed_url: str) -> str | None:
    """Re-parse the RSS 2.0 <author> (no itunes namespace) from the cached
    feed body. Read-only -- never fetches. Returns None if not cached or
    the field is absent."""
    if not feed_url or pd.isna(feed_url):
        return None
    key = hashlib.sha256(str(feed_url).encode()).hexdigest()[:24]
    body_path = RSS_CACHE_DIR / f"{key}.body"
    if not body_path.exists():
        return None
    body = body_path.read_bytes()
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        try:
            cleaned = re.sub(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]", b"", body)
            root = ET.fromstring(cleaned)
        except ET.ParseError:
            return None
    channel = root.find("channel")
    if channel is None:
        return None
    el = channel.find("author")
    if el is None or not el.text:
        return None
    return el.text.strip()


def parse_title(show_name: str) -> tuple[list[str], str]:
    """Returns (names, confidence) from title parsing, or ([], '') if none."""
    name = normalize(show_name)
    m = WITH_RE.search(name)
    if m:
        tail = m.group(1)
        tail = re.split(r"\s*[|(]|\s+-\s+", tail)[0].strip()
        names = [n for n in split_name_list(tail) if is_name_shaped(n)]
        if names:
            return names, "medium"
    m = NAME_SHOW_RE.match(name)
    if m:
        cand = m.group(1)
        if is_name_shaped(cand):
            return [cand], "low"
    return [], ""


def main() -> None:
    df = pd.read_csv(CORPUS_CSV)
    print(f"Loaded {len(df)} shows from {CORPUS_CSV}")

    owner_counts = Counter(
        norm_key(v) for v in df["rss_owner_name"] if normalize(v)
    )

    rows = []
    unknown_shows, publisher_only_shows, multi_host_shows = [], [], []

    for _, r in df.iterrows():
        show = normalize(r["show_name"])
        publisher = normalize(r["publisher"])
        found_publisher_tokens: list[str] = []

        names, source, confidence, extra_note = [], "", "", ""

        author_names, pub_hit = classify_field(r.get("rss_author", ""), show)
        if pub_hit:
            found_publisher_tokens.append(pub_hit)
        if author_names:
            names, source, confidence = author_names, "ITUNES_AUTHOR", "high"

        if not names:
            owner_raw = r.get("rss_owner_name", "")
            owner_names, pub_hit = classify_field(owner_raw, show)
            if pub_hit:
                found_publisher_tokens.append(pub_hit)
            if owner_names:
                names, source, confidence = owner_names, "ITUNES_OWNER", "high"
                repeats = owner_counts.get(norm_key(owner_raw), 0)
                if repeats >= 2:
                    confidence = "medium"
                    extra_note = (
                        f"owner value '{normalize(owner_raw)}' repeats across "
                        f"{repeats} shows in corpus -- likely a shared feed "
                        f"admin/producer contact, not verified as this show's host"
                    )

        if not names:
            cached_author = rss_author_from_cache(r.get("feed_url", ""))
            if cached_author:
                rss_names, pub_hit = classify_field(cached_author, show)
                if pub_hit:
                    found_publisher_tokens.append(pub_hit)
                if rss_names:
                    names, source, confidence = rss_names, "RSS_AUTHOR", "high"

        if not names:
            title_names, title_conf = parse_title(show)
            if title_names:
                names, source, confidence = title_names, "PARSED_FROM_TITLE", title_conf

        if not names:
            if found_publisher_tokens:
                source, confidence = "PUBLISHER_ONLY", "none"
                extra_note = extra_note or (
                    f"publisher-only fields: {', '.join(sorted(set(found_publisher_tokens)))}"
                )
                publisher_only_shows.append(show)
            else:
                source, confidence = "UNKNOWN", "none"
                unknown_shows.append(show)
            names = [""]

        if len(names) > 1:
            multi_host_shows.append((show, names))

        for name in names:
            rows.append({
                "show": show,
                "extracted_host": name,
                "host_source": source,
                "match_confidence": confidence,
                "publisher": publisher,
                "notes": extra_note,
            })

    out = pd.DataFrame(rows, columns=[
        "show", "extracted_host", "host_source", "match_confidence",
        "publisher", "notes",
    ])
    out.to_csv(OUT_CSV, index=False)

    print(f"\nWrote {len(out)} host rows for {df['show_name'].nunique()} shows -> {OUT_CSV}")
    print("\n=== breakdown by host_source ===")
    print(out["host_source"].value_counts().to_string())

    print(f"\n=== UNKNOWN shows ({len(unknown_shows)}) ===")
    for s in unknown_shows:
        print(f"  {s}")

    print(f"\n=== PUBLISHER_ONLY shows ({len(publisher_only_shows)}) ===")
    for s in publisher_only_shows:
        print(f"  {s}")

    print(f"\n=== multi-host shows detected ({len(multi_host_shows)}) ===")
    for s, names in multi_host_shows:
        print(f"  {s}: {names}")

    print("\n=== spot check: first 20 (show, extracted_host, host_source) ===")
    for _, r in out.head(20).iterrows():
        print(f"  {r['show']!r:55} {r['extracted_host']!r:30} {r['host_source']}")


if __name__ == "__main__":
    main()
