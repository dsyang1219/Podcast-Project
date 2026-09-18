# Step 6 — register: how a show talks, independent of what it talks about

**What this step does.** Measures each episode's *register*, the manner of speech rather than the topic, as rates
per 10,000 words over the full transcript. The paper's key measure, listener-directed address, is the mean of standardised imperative and second-person
rates: the measure of record (`mfte_z`) takes both from the published MFTE tagger; the discovery version (`dir_z`)
used a parse rule, a phrase list and a pronoun list from `measure_register2.py` and its pinned copy. Hostility measures are computed alongside and reported as the null result they were.

**Input.** `data/transcripts/`; `data/output/episodes.csv.gz`; `data/output/lean_validation.csv` for the
side-dependent out-group counts; the Brysbaert concreteness norms in `data/output/`.

**Outputs (`data/output/`).** `register2.csv`, `style_measures.csv`, `outrage_lexical.csv`, one row per episode.
Step 7 aggregates episodes to shows (word-count weighted) and standardises against the 194-show reference scale
frozen in `handoff/prereg/dirz_reference_scale.csv`.

## The scripts

| script | role |
|---|---|
| `measure_register2.py` | **The battery the paper's measure comes from.** Sixteen measures spanning spontaneity (fillers, contractions), complexity, stance (imperatives, superlatives, reported speech), temporal orientation and register marks. A pinned copy of the feature code as it was when the pre-registered tests ran is kept at `step7_audience/inputs/remeasure2.py`. |
| `measure_style.py` | The first, side-agnostic battery: othering pronouns, direct address (you/your), certainty, hedging, questions, negation, first person, concreteness, moral-emotional words. |
| `measure_outrage_lexical.py` | Deterministic lexical outrage measures after Sobieraj & Berry (obscenity, intensifiers, absolutes, insults, out-group terms, speech rate). `--report` gives within- vs between-show variance; `--kwic` shows matches in context. |
| `prompts/` | The LLM incivility and outrage prompts used for the model-based hostility layer. |
| `stage_mfte_corpus.py`, `run_mfte_corpus.sh`, `mfte_supervise.sh`, `mfte_address.py` | **The measure of record (16 Sept 2026):** the address composite rebuilt from the published MFTE tagger (Le Foll 2021): stage transcripts in ≤ 3,000-word pieces, tag them in four GPU shards under a memory-guarding supervisor, then turn MFTE's imperative (VIMP) and second-person (PP2) counts into show scores z-scored on the 194 reference shows (`step7_audience/inputs/mfte_show_scores.csv`). Never run more than four taggers, and none while the transcription worker is up: CPU and GPU share memory on rma2. Results in `handoff/results/mfte_measure.md`. |

## Running

```
.venv/bin/python step6_register/measure_register2.py
.venv/bin/python step6_register/measure_style.py
.venv/bin/python step6_register/measure_outrage_lexical.py --run
```

Run from the repository root: the first two use root-relative paths (`data/transcripts/*/*.txt`).
