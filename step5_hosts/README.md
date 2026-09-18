# Step 5 — hosts and DIME ideology scores

**What this step does.** Finds who hosts each show, resolves those people to Wikipedia / Wikidata entries, and
prepares the lookup against the DIME database of campaign-donor ideology scores. The result is an external,
text-independent ideology score per show, which the paper uses to validate the address measure and the LLM
label (the "validation triangle": label, DIME, listeners' party).

**Input.** `data/output/corpus.csv` (step 1) and the cached RSS feeds.

**Outputs (`data/output/`).** `host_lookup.csv`, `host_dime_lookup_v2.csv`, `wikipedia_resolved_v2.csv`, and
the table of record `shows_205_host_dime_scores.csv` (show, host, DIME score), which step 7 reads directly.

## The scripts, in order

| order | script | role |
|---|---|---|
| 1 | `extract_hosts.py` | Extracts host names from the feed metadata in a fixed preference order (itunes:author, itunes:owner, RSS author, the show title), classifies each candidate as person / publisher / self-reference, splits multi-host strings, and flags shared admin contacts. |
| 2 | `resolve_wikipedia_sparql.py` | Resolves shows and hosts to Wikipedia articles through two Wikidata SPARQL queries (exact label match only, type-filtered, evidence recorded) instead of the rate-limited search API. `--audit` prints the acceptance evidence. |

The DIME match itself (name plus occupation against the DIME recipient file) was done in the topic arm's
`topic_arm/nlp/match_guests_dime.py` and `dime_finalize_appearances.py`; `shows_205_host_dime_scores.csv` is the
frozen result.

## Running

```
.venv/bin/python step5_hosts/extract_hosts.py
.venv/bin/python step5_hosts/resolve_wikipedia_sparql.py --run
```

`extract_hosts.py` uses root-relative paths and re-parses cached feeds from `data/cache/rss/`; that cache was
deleted in the September 2026 clean-up and is rebuilt by `step1_frame.run`.
