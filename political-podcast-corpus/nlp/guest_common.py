"""Shared plumbing for the guest-extraction pipeline (Pew/DeMets & Spiro
method replication): episode metadata loading, host-list loading, Tier-0
text cleaning, and name normalization. Used by nlp/discover_patterns.py,
nlp/extract_guests.py, and nlp/validate_guests.py so all three see identical
episode_id/show/text definitions.

Inputs (confirmed against the actual files, not assumed):
  - Episode metadata: data/output/episodes.csv.gz -- columns collection_id,
    show_name, episode_title, episode_description (HTML, truncated to 2000
    chars at scrape time -- verified: max observed length is exactly 2000),
    pub_date, audio_url. NO episode_id column; per nlp/embed_dimensions.py's
    established convention, episode_id = md5(audio_url)[:16] (100% resolved
    there against 7615 chunked episodes). 153202 raw rows / 220 shows; 7
    audio_url collisions -- keep-first, matching embed_dimensions.py.
  - Host list: corpus.csv has NO host column (checked: rank, collection_id,
    show_name, publisher, feed_url, primary_genre, genres, apple_category,
    apple_subcategory, track_count, country, release_date, apple_url,
    artwork_url, lookup_found, rss_description, rss_language, rss_author,
    rss_owner_name, episode_count, episodes_with_audio, hours_available,
    hours_estimated_share, avg_episode_minutes, episodes_per_week,
    first_pub_date, last_pub_date -- no host field). Using
    data/output/host_dime_lookup_v2.csv instead (user-confirmed): 288 rows,
    one per host, columns show_id/show/host_name/alt_names (semicolon-
    separated variants, e.g. "Barbara McQuade; Barbara L. McQuade") among
    others. Covers 197/220 shows in episodes.csv.gz; the other 23 shows get
    no host-stripping (report this gap, not a blocker).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from pipeline import config as pipeline_config

OUT = pipeline_config.OUTPUT_DIR
EPISODES_PATH = OUT / "episodes.csv.gz"
HOSTS_PATH = OUT / "host_dime_lookup_v2.csv"

# ------------------------------------------------------------- episode_id ----

def episode_id_from_audio_url(url) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16] if isinstance(url, str) and url else ""


def load_episodes() -> pd.DataFrame:
    """collection_id, show_name, episode_title, episode_description, pub_date,
    audio_url, episode_id. Deduped on episode_id (keep first), matching
    nlp/embed_dimensions.py's precedent for the same 7 audio_url collisions."""
    eps = pd.read_csv(EPISODES_PATH)
    eps["episode_id"] = eps["audio_url"].map(episode_id_from_audio_url)
    eps = eps[eps["episode_id"] != ""].drop_duplicates(subset="episode_id", keep="first")
    eps["collection_id"] = eps["collection_id"].astype(str)
    return eps.reset_index(drop=True)


# ------------------------------------------------------------------ hosts ----

HONORIFICS = {
    "dr", "sen", "senator", "rep", "representative", "gov", "governor", "president",
    "vice", "mr", "mrs", "ms", "miss", "prof", "professor", "hon", "honorable",
    "father", "rev", "reverend", "general", "gen", "colonel", "col", "captain",
    "secretary", "sec", "ambassador", "amb", "justice", "chief", "judge",
}
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md", "esq"}

_PUNCT_RE = re.compile(r"[.,]")
_WS_RE = re.compile(r"\s+")


@lru_cache(maxsize=2_000_000)
def normalize_name(name: str) -> str:
    """Lowercase, strip accents, honorifics, suffixes, punctuation -- for
    host-match and canonicalization-block comparisons only, never displayed.
    Cached: DIME bulk files repeat the same surname millions of times
    (Zipfian), so memoizing turns the hot path of a 44M-row scan from 44M
    full recomputations into ~O(distinct names)."""
    if not isinstance(name, str) or not name.strip():
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = _PUNCT_RE.sub("", s).lower().strip()
    tokens = [t for t in _WS_RE.split(s) if t]
    tokens = [t for t in tokens if t not in HONORIFICS and t not in NAME_SUFFIXES]
    return " ".join(tokens)


def load_hosts() -> dict[str, set[str]]:
    """show_id (str) -> set of normalized host-name variants (host_name +
    semicolon-split alt_names)."""
    h = pd.read_csv(HOSTS_PATH)
    h["show_id"] = h["show_id"].astype(str)
    out: dict[str, set[str]] = {}
    for _, row in h.iterrows():
        sid = row["show_id"]
        variants = {row["host_name"]}
        if isinstance(row.get("alt_names"), str) and row["alt_names"].strip():
            variants.update(v.strip() for v in row["alt_names"].split(";") if v.strip())
        norm = {normalize_name(v) for v in variants if isinstance(v, str)}
        norm.discard("")
        out.setdefault(sid, set()).update(norm)
    return out


# -------------------------------------------------------------- tier 0 ----

_AD_MARKER_RE = re.compile(
    r"(?:this episode is brought to you by|promo code|use code|"
    r"advertising inquiries|learn more about your ad choices|"
    r"visit\s+\S+\.com/)",
    re.I,
)
_URL_RE = re.compile(r"https?://\S+")
_TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
MIN_HEAD_CHARS = 60


def strip_html_urls(raw: str) -> str:
    """HTML-stripped, URL/timestamp-dropped, but NOT ad-truncated -- the
    'original' text evidence verification checks against (Tier 0's ad-cut is
    the only lossy step; a hallucination-check shouldn't be defeated by it)."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    text = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    text = _URL_RE.sub(" ", text)
    text = _TIMESTAMP_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def clean_description(raw: str) -> str:
    """Tier 0: strip HTML, unescape entities, drop URLs/timestamps, truncate
    trailing ad/sponsor copy. Returns the CLEANED text; caller keeps `raw`
    separately for evidence verification."""
    text = strip_html_urls(raw)
    if not text:
        return ""
    m = _AD_MARKER_RE.search(text)
    if m and m.start() > MIN_HEAD_CHARS:
        text = text[: m.start()]
    return text.strip()


# --------------------------------------------------------- nickname table ----
# Small, hand-maintained -- canonicalization also uses fuzzy token_set_ratio,
# this only catches short-nickname pairs fuzzy matching alone tends to miss
# (e.g. "Liz"/"Elizabeth" score low on naive ratio despite being the same
# person; token_set_ratio on full names with a shared surname recovers most
# of this already, but first-name-only mentions need the explicit map).
NICKNAMES = {
    "liz": "elizabeth", "beth": "elizabeth", "liza": "elizabeth",
    "bill": "william", "will": "william", "billy": "william",
    "bob": "robert", "rob": "robert", "robbie": "robert", "bobby": "robert",
    "dick": "richard", "rich": "richard", "richie": "richard",
    "jim": "james", "jimmy": "james",
    "joe": "joseph", "joey": "joseph",
    "mike": "michael", "mikey": "michael",
    "tom": "thomas", "tommy": "thomas",
    "tony": "anthony",
    "chris": "christopher",
    "dave": "david",
    "steve": "steven", "stevie": "steven",
    "ken": "kenneth", "kenny": "kenneth",
    "ted": "edward", "eddie": "edward", "ed": "edward",
    "sam": "samuel", "sammy": "samuel",
    "alex": "alexander",
    "andy": "andrew", "drew": "andrew",
    "nick": "nicholas",
    "matt": "matthew",
    "dan": "daniel", "danny": "daniel",
    "greg": "gregory",
    "ben": "benjamin", "benny": "benjamin",
    "jon": "jonathan", "johnny": "jonathan",
    "kathy": "katherine", "kate": "katherine", "katie": "katherine", "kathryn": "katherine",
    "peggy": "margaret", "maggie": "margaret", "meg": "margaret",
    "sue": "susan", "suzy": "susan",
    "cindy": "cynthia",
    "debbie": "deborah", "deb": "deborah",
    "jen": "jennifer", "jenny": "jennifer",
    "amy": "amelia",
    "tim": "timothy", "timmy": "timothy",
}


def nickname_normalize(first_name_lower: str) -> str:
    return NICKNAMES.get(first_name_lower, first_name_lower)
