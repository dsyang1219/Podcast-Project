# Step 2 — which episodes to transcribe

**What this step does.** Chooses episodes from the step-1 episode table and writes a *manifest*: a CSV whose row
order is the transcription priority. The transcription run (step 3) walks the manifest from the top, so it can be
stopped at any point and what has been transcribed is still a balanced, designed sample rather than a half-finished
sweep.

**Input.** `data/output/episodes.csv.gz` from step 1 (frozen alongside the manifest: re-pulling feeds would change
the per-cell ranks).

**Outputs.** `data/output/sample_out/sample_manifest_*.csv`. The manifest of record for the corpus is the ladder
manifest; `sample_manifest_band15.csv` is the extension currently being transcribed.

## The scripts

| script | role |
|---|---|
| `sample_ladder.py` | **The sampler used for the corpus.** Builds the priority ladder: P0 is a census of every episode before 2018 (the thin, irreplaceable tail), then bands P1..Pk take k episodes per show per quarter. Ranks come from a frozen hash of (seed, episode id), so raising k later only appends rows and never reshuffles what was already transcribed. Within a band, rows are round-robined across shows so the downloader is not pinned to one host. |
| `split_manifest.py` | Shards a manifest across N machines by taking row i to shard i mod N. Every shard keeps the same priority and show mix, so a machine that dies early still leaves a valid prefix of the design. This is how the run was split between rma1 and rma2. |
| `sample_episodes.py` | The earlier hours-budget sampler (draw episodes per show until H hours, with nested saturation ladders at 5/10/.../40 h). Kept because the first sample (H25) was drawn with it; the ladder sampler superseded it for the expansion. |

## Running

```
.venv/bin/python step2_sample/sample_ladder.py --max-k 30
.venv/bin/python step2_sample/split_manifest.py --of 2
```

Run from the repository root. The scripts locate `data/` from their own position, so they also work from elsewhere.
