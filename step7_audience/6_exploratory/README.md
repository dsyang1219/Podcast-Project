# 7.6 — exploratory work after the confirmatory tests failed

Everything here is labelled exploratory in the paper and is what the Year-2 pre-registration is built on. Two
threads: (a) restricting to shows with enough political content, where address does predict election distrust
and cynicism; (b) checking whether populism or ideological intensity, rather than address, carries the signal.

```
ideology_cap40 labels  --llm_poldensity.py---->  inputs/corpus_poldensity_llm.csv (share of passages the instrument marks political; frame = >= 10th pct)
scoring_chunks.csv.gz --corpus_poldensity.py-->  inputs/corpus_poldensity.csv   (retired dictionary density)
scoring_chunks.csv.gz --populist.py---------->  inputs/show_populist.csv
scoring_chunks.csv.gz --rp_antimedia.py------>  inputs/show_rp_antimedia.csv    (Rooduijn & Pauwels dictionary, anti-media references)
scoring_chunks.csv.gz --horserace.py--------->  inputs/show_gameframe.csv       (strategy / game-frame lexicon)
all show feature tables + corpus and holdout listeners
                      --explore_scan.py------>  outputs/explore_scanB.csv, explore_showlevel.csv, political_frame_address_scan.csv, ...
explore_scan.py's frame block --distill_inten.py-->  intensity classifier for unlabelled shows; intensity vs populism horse race
show_rp_antimedia + address  --rp_test.py------->  joint models (CR2 + wild); --rp_words.py--> which stems carry the populism rate
show_gameframe + address     --hr_infer.py------>  small-sample inference for game frame vs address
```

| script | role |
|---|---|
| `llm_poldensity.py` | **Defines the political-podcast frame (15 Sept 2026 onward):** the share of a show's sampled passages that stage 1 of the Much et al. ideology rubric marks political; cut = corpus 10th percentile (0.644). Published instrument, no word list. Results in `handoff/results/frame_llm_density.md`. |
| `corpus_poldensity.py` | The earlier hand-written political-term dictionary version of density (share of passages with three or more hits). Retired as the frame variable; kept for the record (r = 0.73 with the LLM version). |
| `explore_scan.py` | The pooled 64-show, 1,284-listener scan of every show feature against cynicism, election distrust and exclusivity. The code above its `# ==== SCANS START` marker assembles the pooled frame and is re-used by `distill_inten.py`. The education-composition result and the political-frame results are read from its outputs. |
| `distill_inten.py` | TF-IDF distillation of the LLM intensity labels (grouped cross-validation) to score the 11 unlabelled shows, then intensity against populism. |
| `populist.py`, `rp_antimedia.py`, `rp_words.py`, `rp_test.py` | The populism thread: two lexicons plus the Rooduijn & Pauwels dictionary, scored over the corpus and put in joint models with address. |
| `horserace.py`, `hr_infer.py` | The game-frame thread: strategy / game-frame lexicon against address, with CR2 + wild inference. |

The scan tables the paper cites are copied to `handoff/scans/`; the write-ups are
`handoff/results/kettering_y1_test_results.md` and `handoff/results/exploratory_populism.md`.
