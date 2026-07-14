"""Central configuration for the political podcast corpus pipeline.

Every parameter that affects the sampling frame or the inclusion filter
lives here so a run is fully described by (this file, the frozen chart).
"""
from pathlib import Path

# ---------------------------------------------------------------- paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # frozen chart pulls + raw Apple API responses
CACHE_DIR = DATA_DIR / "cache"      # cached HTTP responses (RSS feeds, lookups)
RSS_CACHE_DIR = CACHE_DIR / "rss"
EXTERNAL_DIR = DATA_DIR / "external"  # Brookings Political Podcast Project export
OUTPUT_DIR = DATA_DIR / "output"

# ---------------------------------------------------------------- chart ----
# Apple Podcasts genre taxonomy: News = 1489, Politics (News subcategory) = 1527.
# https://podcasts.apple.com/us/genre/podcasts-news-politics/id1527
CHART_GENRE_ID = 1527
CHART_GENRE_NAME = "Politics"
CHART_PARENT_GENRE = "News"
CHART_STOREFRONT = "us"
CHART_LIMIT = 250  # as deep as Apple serves for subcategory charts

# Ranked chart of collection ids (undocumented but stable Apple endpoint,
# same source the podcasts.apple.com charts page uses).
CHART_URL = (
    "https://itunes.apple.com/WebObjects/MZStoreServices.woa/ws/charts"
    f"?g={CHART_GENRE_ID}&name=Podcasts&limit={CHART_LIMIT}&cc={CHART_STOREFRONT}"
)
# Documented lookup API used to resolve ids -> metadata (feed URL, artist...).
LOOKUP_URL = "https://itunes.apple.com/lookup"
LOOKUP_BATCH_SIZE = 100  # documented max ids per lookup request is ~200; stay low

# ------------------------------------------------------------ politeness ----
APPLE_DELAY_SECONDS = 1.0   # pause between requests to Apple hosts
RSS_DELAY_SECONDS = 0.25    # pause between RSS fetches (feeds are on many hosts)
HTTP_TIMEOUT = 30
HTTP_RETRIES = 3
USER_AGENT = (
    "political-podcast-corpus/1.0 (academic research; contact via repo issues)"
)

# ------------------------------------------------------ inclusion filter ----
# Threshold for "insufficient back catalog". We sample by hours, so the unit
# is hours of audio available in the show's live RSS feed. Default assumes a
# per-show sampling budget on the order of ~10 h; a show must be able to fill
# at least MIN_HOURS_AVAILABLE of that from its feed. Override with
# --min-hours at run time; the value used is recorded in run_manifest.json.
MIN_HOURS_AVAILABLE = 10.0

# Shows whose frame is non-US politics. The Politics subcategory chart of the
# US storefront still surfaces UK/AU/etc. shows; these rules drop them.
# Non-US outlets are dropped even when the show covers US politics for a
# foreign audience (e.g. BBC "Americast", Guardian "Politics Weekly America").
# Publisher (artist) terms, matched case-insensitively on word boundaries:
NON_US_PUBLISHERS = [
    "bbc",
    "the guardian",
    "novara media",
    "sky news",
    "times radio",
    "the telegraph",
    "the spectator",
    "tortoise media",
    "global player origin",  # The News Agents (UK)
    "australian broadcasting corporation",
    "abc listen",
    "cbc",  # Canadian Broadcasting Corporation
    "rte",  # Irish public broadcaster (word-boundary matched)
    "lbc",
    "goalhanger",  # UK producer; US-frame exceptions listed below
]
# Title/description regexes (case-insensitive) indicating a non-US frame:
NON_US_PATTERNS = [
    r"\buk politics\b",
    r"\bbritish politics\b",
    r"\bwestminster\b",
    r"\bdowning street\b",
    r"\bhouse of commons\b",
    r"\baustralian politic",
    r"\bcanadian politic",
    r"\birish politic",
    r"\bindian politic",
    r"\bcanberra\b",
    r"\bnew zealand politic",
]
# Shows matched by a rule above that are nonetheless US-politics shows.
# (e.g. Goalhanger's "The Rest Is Politics: US" is a US-politics show even
# though its sibling "The Rest Is Politics" is UK.) Keyed by Apple
# collection id -> reason for the exception.
NON_US_EXCEPTIONS = {
    1743030473: "The Rest Is Politics: US — Katty Kay/Anthony Scaramucci, US frame",
}
# Non-US shows that the heuristics cannot catch from metadata alone.
# Curated after manual review of the 2026-07-13 frame; id -> evidence.
# Judgment calls are noted as such — revisit if the frame changes.
NON_US_MANUAL = {
    1611374685: "The Rest Is Politics (Goalhanger) — UK politics",
    1665265193: "The Rest Is Politics: Leading (Goalhanger) — UK-frame interviews",
    1375568988: ("TRIGGERnometry — UK-produced (Kisin/Foster, London); "
                 "judgment call: culture-war interview show, non-US frame"),
}

# ------------------------------------------------------------- lean join ----
# Brookings Political Podcast Project export (Wirtschafter). Place the file
# here or pass --brookings-csv. Expected columns include 'Show Name' and
# 'Partisan Leaning'.
BROOKINGS_CSV_DEFAULT = EXTERNAL_DIR / "full-dataset-2026-07-13.csv"
FUZZY_ACCEPT = 90     # WRatio >= this -> matched
FUZZY_REVIEW = 75     # WRatio in [REVIEW, ACCEPT) -> flag for manual review
FUZZY_AMBIGUOUS_GAP = 5  # best - runner_up < gap -> flag ambiguous
