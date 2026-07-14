# Political Podcast Corpus Sampler

Reproducible sampling pipeline that builds a corpus of US political podcasts
from the Apple Podcasts US **Politics** subcategory chart (News > Politics,
genre `1527`), for a computational-social-science study of ideological
dimensionality in political-podcast discourse.

Methodology adapts Wirtschafter 2023 / the Brookings Political Podcast Project,
with two deliberate departures documented in `pipeline/config.py`:

1. Sample from the purpose-built **Politics subcategory** chart, not the general Top 100.
2. **No** algorithmic "you might also like" expansion (it pulls in non-political shows).

## Run

```
pip install pandas rapidfuzz requests
python -m pipeline.run                 # uses today's frozen frame if present
python -m pipeline.run --chart-date 20260713   # reuse a specific frozen frame
python -m pipeline.run --min-hours 10 --brookings-csv data/external/full-dataset-2026-07-13.csv
```

Re-runs against an existing frozen chart make **zero** network requests and are
deterministic. Delete `data/cache/` to force a refresh; `--refresh` bypasses caches.

## Pipeline steps

| Step | Module | Output |
|------|--------|--------|
| 1. Chart scraper | `pipeline/chart.py` | `data/raw/raw_chart_YYYYMMDD.csv` (frozen frame) |
| 2. RSS enrichment | `pipeline/rss.py` | per-episode duration, show-level stats |
| 3. Inclusion filter | `pipeline/filter.py` | `exclusions.csv` (every drop + reason) |
| 4. Lean join (validation only) | `pipeline/lean_join.py` | `lean_validation.csv` |

## Outputs (`data/output/`)

- `raw_chart_YYYYMMDD.csv` — frozen, unmodified sampling frame
- `corpus.csv` — included shows + metadata + RSS stats + hours available
- `episodes.csv.gz` — per-episode records for included shows
- `exclusions.csv` — every dropped show + rule + evidence
- `lean_validation.csv` — corpus fuzzy-joined to Brookings lean labels
- `summary.txt`, `run_manifest.json`

The lean labels are joined for **later axis validation only** and are never used
to filter or stratify the sample.
