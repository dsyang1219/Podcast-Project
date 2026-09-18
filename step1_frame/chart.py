"""Step 1 — pull the Apple Podcasts US Politics subcategory chart.

Produces the frozen sampling frame:
  data/raw/chart_ids_YYYYMMDD.json      ranked collection ids, as returned
  data/raw/lookup_YYYYMMDD.json         raw iTunes Lookup responses
  data/raw/raw_chart_YYYYMMDD.csv       one row per ranked show, unmodified

If the frame for `chart_date` already exists it is loaded as-is (frozen);
the network is only touched for a date with no frame on disk.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

from . import config
from .http_util import get

CHART_FIELDS = [
    "rank",
    "collection_id",
    "show_name",
    "publisher",
    "feed_url",
    "primary_genre",
    "genres",
    "apple_category",
    "apple_subcategory",
    "track_count",
    "country",
    "release_date",
    "apple_url",
    "artwork_url",
    "lookup_found",
]


def _chart_paths(chart_date: str):
    return (
        config.RAW_DIR / f"chart_ids_{chart_date}.json",
        config.RAW_DIR / f"lookup_{chart_date}.json",
        config.RAW_DIR / f"raw_chart_{chart_date}.csv",
    )


def pull_chart(chart_date: str | None = None, refresh: bool = False) -> tuple[list[dict], str]:
    """Return (rows, chart_date). Rows are dicts keyed by CHART_FIELDS."""
    chart_date = chart_date or datetime.now(timezone.utc).strftime("%Y%m%d")
    ids_path, lookup_path, csv_path = _chart_paths(chart_date)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and not refresh:
        with csv_path.open() as fh:
            rows = list(csv.DictReader(fh))
        print(f"[chart] frozen frame loaded: {csv_path.name} ({len(rows)} shows)")
        return rows, chart_date

    # -- ranked ids ----------------------------------------------------------
    pulled_at = datetime.now(timezone.utc).isoformat()
    body, meta = get(config.CHART_URL, config.CACHE_DIR / "chart", config.APPLE_DELAY_SECONDS,
                     refresh=refresh)
    if body is None:
        raise RuntimeError(f"chart request failed: {meta}")
    chart_json = json.loads(body)
    ranked_ids = [int(i) for i in chart_json["resultIds"]]
    ids_path.write_text(json.dumps(
        {"pulled_at_utc": pulled_at, "url": config.CHART_URL,
         "genre_id": config.CHART_GENRE_ID, "storefront": config.CHART_STOREFRONT,
         "result_ids": ranked_ids}, indent=2))
    print(f"[chart] {len(ranked_ids)} ranked ids pulled at {pulled_at}")

    # -- resolve metadata via iTunes Lookup ----------------------------------
    lookups: dict[int, dict] = {}
    raw_responses = []
    for start in range(0, len(ranked_ids), config.LOOKUP_BATCH_SIZE):
        batch = ranked_ids[start:start + config.LOOKUP_BATCH_SIZE]
        url = f"{config.LOOKUP_URL}?id={','.join(map(str, batch))}&entity=podcast"
        body, meta = get(url, config.CACHE_DIR / "lookup", config.APPLE_DELAY_SECONDS,
                         refresh=refresh)
        if body is None:
            raise RuntimeError(f"lookup request failed: {meta}")
        payload = json.loads(body)
        raw_responses.append({"url": url, "response": payload})
        for item in payload.get("results", []):
            if item.get("kind") == "podcast" or item.get("wrapperType") == "track":
                lookups[int(item["collectionId"])] = item
    lookup_path.write_text(json.dumps(
        {"pulled_at_utc": pulled_at, "responses": raw_responses}, indent=2))

    # -- assemble frozen frame ------------------------------------------------
    rows = []
    for rank, cid in enumerate(ranked_ids, start=1):
        item = lookups.get(cid, {})
        genre_names = item.get("genres", [])
        genre_names = [g for g in genre_names if isinstance(g, str)]
        rows.append({
            "rank": rank,
            "collection_id": cid,
            "show_name": item.get("collectionName", ""),
            "publisher": item.get("artistName", ""),
            "feed_url": item.get("feedUrl", ""),
            "primary_genre": item.get("primaryGenreName", ""),
            "genres": "|".join(genre_names),
            "apple_category": config.CHART_PARENT_GENRE,
            "apple_subcategory": config.CHART_GENRE_NAME,
            "track_count": item.get("trackCount", ""),
            "country": item.get("country", ""),
            "release_date": item.get("releaseDate", ""),
            "apple_url": item.get("collectionViewUrl", ""),
            "artwork_url": item.get("artworkUrl600", ""),
            "lookup_found": bool(item),
        })

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CHART_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[chart] frozen frame written: {csv_path.name} ({len(rows)} shows, "
          f"{sum(1 for r in rows if r['lookup_found'])} resolved via lookup)")
    return [ {k: str(v) for k, v in r.items()} for r in rows ], chart_date
