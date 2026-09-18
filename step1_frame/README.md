# Step 1 — the sampling frame

**What this step does.** Turns the Apple Podcasts US *Politics* chart into a documented list of shows, with every
exclusion recorded. Everything downstream (sampling, transcription, measurement) works off the files this step
writes. The chart pull is frozen: once `data/raw/raw_chart_20260713.csv` exists, the network is never touched for
that date again, so the frame is reproducible.

**Inputs.** The live Apple chart (only on the first pull), each show's RSS feed (cached under `data/cache/rss/`),
and the Brookings Political Podcast Project export in `data/external/`.

**Outputs (`data/output/`).** `raw_chart_YYYYMMDD.csv` (the frame), `corpus.csv` (included shows with RSS
statistics and available hours), `episodes.csv.gz` (one row per episode of every included show), `exclusions.csv`
(every dropped show, the rule that dropped it, the evidence), `lean_validation.csv` (Brookings lean labels joined
for validation only), `run_manifest.json`, `summary.txt`.

## How the modules relate

`run.py` is the orchestrator; it calls the other modules in this order:

| order | module | role |
|---|---|---|
| 1 | `chart.py` | Pull the ranked chart and the iTunes lookup for each show. Writes the frozen frame to `data/raw/`. |
| 2 | `rss.py` | For each show in the frame, fetch and parse the RSS feed: episode list, durations, audio URLs, feed language. |
| 3 | `filter.py` | Apply the inclusion rules R1 to R6 in fixed order (no feed, unreachable, unparseable, no audio, non-US-politics, under 10 hours available). The first failing rule is the recorded reason. |
| 4 | `lean_join.py` | Fuzzy-join Brookings lean labels onto the surviving shows. Used to validate the ideology axis, never to select shows. |
| — | `config.py` | Every parameter that defines the frame: genre id, chart depth, politeness delays, blocklists, the 10-hour threshold, fuzzy-match cut-offs. A run is fully described by this file plus the frozen chart. |
| — | `http_util.py` | Polite, disk-cached HTTP used by all of the above. |
| — | `budget.py` | Side analysis: how many shows survive at each per-show hours budget. Used once to choose the sampling budget. |

## Running

```
.venv/bin/python -m step1_frame.run --chart-date 20260713
```

`--refresh` clears the cache; `--min-hours` overrides the catalogue threshold. Because this folder is a Python
package (`__init__.py`), other code imports its configuration as `from step1_frame import config`; the topic arm
and the step-3 downloader both do this.
