"""Pipeline orchestrator.

    python -m pipeline.run [--chart-date YYYYMMDD] [--min-hours H]
                           [--brookings-csv PATH] [--refresh]

Outputs (data/output/):
  raw_chart_YYYYMMDD.csv  frozen sampling frame (also kept in data/raw/)
  corpus.csv              included shows + metadata + RSS stats + hours
  episodes.csv.gz         per-episode records for included shows
  exclusions.csv          every dropped show + rule + evidence
  lean_validation.csv     corpus joined to Brookings lean labels
  run_manifest.json       parameters + counts for the run
  summary.txt             the printed summary
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import shutil
import csv as csv_mod
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config
from .chart import pull_chart
from .rss import enrich_show, EPISODE_FIELDS
from .filter import apply_filter
from .lean_join import load_brookings, join_leans

SHOW_RSS_FIELDS = [
    "rss_description", "rss_language", "rss_author", "rss_owner_name",
    "episode_count", "episodes_with_audio", "hours_available",
    "hours_estimated_share", "avg_episode_minutes", "episodes_per_week",
    "first_pub_date", "last_pub_date",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chart-date", default=None,
                    help="YYYYMMDD of a frozen frame to reuse (default: today UTC)")
    ap.add_argument("--min-hours", type=float, default=config.MIN_HOURS_AVAILABLE,
                    help="min hours of audio in feed to be included")
    ap.add_argument("--brookings-csv", type=Path, default=config.BROOKINGS_CSV_DEFAULT,
                    help="Brookings Political Podcast Project export")
    ap.add_argument("--refresh", action="store_true",
                    help="bypass HTTP caches (re-pulls chart + feeds)")
    args = ap.parse_args()

    started = datetime.now(timezone.utc).isoformat()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- step 1: chart ------------------------------------------------------
    rows, chart_date = pull_chart(args.chart_date, refresh=args.refresh)
    raw_csv = config.RAW_DIR / f"raw_chart_{chart_date}.csv"
    shutil.copy(raw_csv, config.OUTPUT_DIR / raw_csv.name)

    # ---- step 2 + 3: RSS enrichment and filter ------------------------------
    corpus_rows, exclusion_rows, episode_rows = [], [], []
    for i, row in enumerate(rows, 1):
        rss = enrich_show(row, refresh=args.refresh)
        included, rule, evidence = apply_filter(row, rss, args.min_hours)
        if i % 25 == 0 or i == len(rows):
            print(f"[rss] {i}/{len(rows)} feeds processed")
        if included:
            merged = {**row, **{k: rss[k] for k in SHOW_RSS_FIELDS}}
            corpus_rows.append(merged)
            episode_rows.extend(rss["episodes"])
        else:
            exclusion_rows.append({
                "rank": row["rank"], "collection_id": row["collection_id"],
                "show_name": row["show_name"], "publisher": row["publisher"],
                "rule": rule, "evidence": evidence,
                "hours_available": rss["hours_available"],
            })

    corpus = pd.DataFrame(corpus_rows)
    exclusions = pd.DataFrame(exclusion_rows)
    corpus.to_csv(config.OUTPUT_DIR / "corpus.csv", index=False)
    exclusions.to_csv(config.OUTPUT_DIR / "exclusions.csv", index=False)

    with gzip.open(config.OUTPUT_DIR / "episodes.csv.gz", "wt", newline="") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(episode_rows)

    # ---- step 4: lean labels (validation only, never for sampling) ----------
    lean = None
    if args.brookings_csv and Path(args.brookings_csv).exists():
        brookings = load_brookings(args.brookings_csv)
        lean = join_leans(corpus, brookings)
        lean.to_csv(config.OUTPUT_DIR / "lean_validation.csv", index=False)
    else:
        print(f"[lean] Brookings CSV not found at {args.brookings_csv} — "
              "skipping lean join. Re-run with --brookings-csv when available.")

    # ---- summary -------------------------------------------------------------
    buf = io.StringIO()

    def w(line=""):
        print(line)
        buf.write(line + "\n")

    hours = corpus["hours_available"].astype(float) if len(corpus) else pd.Series(dtype=float)
    w("=" * 68)
    w("POLITICAL PODCAST CORPUS — RUN SUMMARY")
    w("=" * 68)
    w(f"chart: Apple Podcasts {config.CHART_STOREFRONT.upper()} / "
      f"{config.CHART_PARENT_GENRE} > {config.CHART_GENRE_NAME} "
      f"(genre {config.CHART_GENRE_ID}), frozen frame {chart_date}")
    w(f"run started: {started}")
    w(f"min-hours threshold: {args.min_hours}")
    w()
    w(f"shows in sampling frame:   {len(rows)}")
    w(f"shows excluded:            {len(exclusions)}")
    if len(exclusions):
        for rule, n in exclusions["rule"].value_counts().items():
            w(f"    {rule:<28s} {n}")
    w(f"shows in final corpus:     {len(corpus)}")
    w()
    if len(corpus):
        w(f"total hours available:     {hours.sum():,.0f} h")
        w("hours per show:")
        w(f"    min {hours.min():.1f} | p25 {hours.quantile(.25):.1f} | "
          f"median {hours.median():.1f} | p75 {hours.quantile(.75):.1f} | "
          f"max {hours.max():.1f}")
        est = corpus["hours_estimated_share"].astype(float)
        w(f"shows with any byte-estimated durations: {(est > 0).sum()} "
          f"(mean estimated share {est.mean():.1%})")
    if lean is not None:
        counts = lean["match_status"].value_counts()
        w()
        w(f"external lean labels (Brookings): "
          f"{int(counts.get('matched', 0))} matched, "
          f"{int(counts.get('review', 0))} for review, "
          f"{int(counts.get('ambiguous', 0))} ambiguous, "
          f"{int(counts.get('none', 0))} unmatched")
        matched = lean[lean.match_status == "matched"]
        if len(matched):
            for lab, n in matched["brookings_partisan_leaning"].value_counts().items():
                w(f"    {lab:<20s} {n}")
    w("=" * 68)
    (config.OUTPUT_DIR / "summary.txt").write_text(buf.getvalue())

    manifest = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "chart_date": chart_date,
        "chart_url": config.CHART_URL,
        "min_hours": args.min_hours,
        "fuzzy_accept": config.FUZZY_ACCEPT,
        "fuzzy_review": config.FUZZY_REVIEW,
        "brookings_csv": str(args.brookings_csv),
        "brookings_joined": lean is not None,
        "n_frame": len(rows),
        "n_excluded": len(exclusions),
        "n_corpus": len(corpus),
        "total_hours": round(float(hours.sum()), 1) if len(corpus) else 0.0,
        "exclusions_by_rule": (exclusions["rule"].value_counts().to_dict()
                               if len(exclusions) else {}),
    }
    (config.OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
