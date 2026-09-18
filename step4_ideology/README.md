# Step 4 — passage-level political content and ideology

**What this step does.** Cuts every transcript into the unit the published ideology instrument was validated on
(750-character passages at sentence boundaries), sends a capped sample of passages through the two-stage LLM
rubric of Much et al. (2026), and checks whether the resulting show-level score can assign each show a side.
The passage table it builds is also the text unit that steps 6 and 7 reuse (political-content density, the
populism lexicons, the holdout scoring).

**Input.** `data/transcripts/` and `data/output/episodes.csv.gz`.

**Outputs (`data/output/`).** `scoring_chunks.csv.gz` (every passage), `scoring_chunks_cap40.csv` (the scored
sample), `ideology_cap40.csv` (LLM labels), `ideology_sideclass.csv`, `ideology_targets_204.csv`.

## The scripts, in the order they run

| order | script | role |
|---|---|---|
| 1 | `build_scoring_chunks.py` | `--build` writes the full passage table; `--pilot N` a stratified validation set. Its `split_750()` and `transcript_text()` are imported by the step-7 scoring scripts so that out-of-frame shows are chunked exactly as the corpus was. |
| 2 | `sample_chunks_for_scoring.py` | Caps the passages per show-quarter (round-robin across episodes, frozen order, monotone in the cap) so the LLM budget buys within-show variation rather than depth on one episode. |
| 3 | `score_ideology.py` | Two-stage rubric: stage 1 asks whether the passage is political; only "yes" passages get the 7-point direction score in stage 2. Dated model snapshot, prompts duplicated from `prompts/ideology.md` (the markdown is canonical). `--coding-sheet` writes a blind sheet for human validation. |
| 4 | `eval_side_classifier.py` | Can the content measure assign a show's side better than DIME? Three leave-one-out rules, plus a party-free-passage check that breaks the circularity between the anchor and the out-group count. |

`prompts/ideology.md` is the instrument wording of record.

## Running

```
.venv/bin/python step4_ideology/build_scoring_chunks.py --build
.venv/bin/python step4_ideology/sample_chunks_for_scoring.py --cap 40 --out data/output/scoring_chunks_cap40.csv
.venv/bin/python step4_ideology/score_ideology.py --input data/output/scoring_chunks_cap40.csv --out data/output/ideology_cap40.csv
.venv/bin/python step4_ideology/eval_side_classifier.py
```

Run from the repository root (`sample_chunks_for_scoring.py` uses root-relative defaults). `score_ideology.py`
reads the API key from `.env`.
