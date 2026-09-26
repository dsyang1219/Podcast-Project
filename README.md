# Political Podcast Corpus

Code and record for *Talking at the Audience: How Political Podcast Hosts Speak, and Who Listens*
(Yang & Alvarez, Linde Center for Science, Society, and Policy, Caltech, 2026).

The paper measures how directly the hosts of 194 US political podcasts talk at their listeners (commands and
second-person address), separately from how one-sided each show's political content is. Address is a stable trait
of a show and is higher on the right. On the audience side the two properties do different work: address marks
which shows reach beyond the college-educated public (the shows that address the listener least have markedly
more educated audiences, replicated in two Kettering–Gallup waves, two Pew surveys and the reading level of the
shows' reviews), while one-sidedness, not address, goes with election distrust, review warmth and felt connection to
the host.

**Status (26 September 2026).** Analysis corpus frozen 18 September at 31,191 episodes / 194 shows (census and
ladder bands 1–5 complete; bands 10 and 15 about half); the ladder finished on 23 September. Address measure of
record: MFTE-based (16 September). Audience results use both Democracy for All waves under one matching rule
(1,247 listeners / 64 shows). The Year 1 and Year 2 pre-registered tests of address on election distrust failed as
frozen and are reported as such; the composition result is exploratory and externally replicated.

## Layout

The repository is **seven numbered steps**, run in order; each folder's `README.md` says what its scripts do.
`topic_arm/` is a side project (topic model / ideology recoverability) that is
not in the paper.

```
step1_frame       Apple Podcasts US Politics chart (frozen 13 July 2026) -> RSS metadata -> six inclusion rules
                  -> data/output/corpus.csv, exclusions.csv, lean_validation.csv          (219 shows)
step2_sample      episode table -> priority-ladder manifest (census before 2018, then k per show per quarter)
                  -> data/output/sample_out/sample_manifest_*.csv
step3_transcribe  manifest -> audio download -> faster-whisper large-v3-turbo on the GPU
                  -> data/transcripts/<show_id>/<episode_id>.json + .txt                    (not in git)
step4_ideology    transcripts -> 750-character passages -> two-stage LLM rubric (Much et al. 2026)
                  -> data/output/ideology_cap40.csv; show-level side, one-sidedness (intensity), political density
step5_hosts       host names -> Wikidata -> DIME donor scores per show
                  -> data/output/shows_205_host_dime_scores.csv
step6_register    transcripts -> listener-directed address: MFTE tagger (measure of record), command types by
                  Biber verb category, the earlier parse/phrase-list version; outrage coding (not in the paper)
                  -> step7_audience/inputs/mfte_show_scores.csv, mfte_episode_rates.csv,
                     mfte_imperative_show_scores.csv, directive_final.csv
step7_audience    show measures + Kettering-Gallup (two waves) + Pew ATP + Apple reviews -> validation,
                  pre-registered tests, audience composition, external checks
                    1_match .. 6_exploratory  as before
                    7_reviews     public Apple Podcasts reviews: collector and parasocial coder (pilot only)
                    8_pew         Pew ATP W165 (named-source audiences) and W118 (main podcast, host connection)
                    9_education_probes  the education-composition analyses behind Results and Figures 4-7
results, scans, figures   result logs and write-ups, scan tables, and the paper's figures (the paper text, slides
                  and Year 1 pre-registration files are kept by the authors, not in this repository)
topic_arm         topic-model arm (progress report RQ1); not used by the paper
data              raw/ (frozen chart, committed)  external/ (restricted, not committed)
                  output/ (small tables of record committed)  transcripts/, audio_queue/ (not committed)
```

## Where the paper's numbers come from

| paper result | where |
|---|---|
| Corpus: 250-show frame, six inclusion rules, 219 -> 204 -> 194 shows; the 15 dropped by recorded decision (14 Aug 2026) | `step1_frame/`, `data/output/exclusions.csv`, `data/output/corpus_shows.csv` (`excluded_reason`) |
| Ladder band at freeze (census + bands 1–5 complete; 6–10 and 11–15 about half) | `data/output/sample_out/corpus_episodes.csv` (priority, done) |
| Ideology instrument (gpt-5.4-mini, Much et al. prompts verbatim), side / one-sidedness / density | `step4_ideology/`, prompts in `step4_ideology/prompts/ideology.md` |
| Passage-level hand coding (36 passages; gate kappa .65, direction .51; Much et al. report kappa .83 vs human ceiling alpha .56) | `data/output/ideology_coding_human_2026-08-26.csv`, `ideology_coding_key_production_2026-09-16.csv` |
| Side-score validation: Brookings (22/24), DIME (r .68, 112 shows; DIME public v4.0), listeners' party (r .92, 27/28 shows, both waves) | `step5_hosts/`, `data/output/lean_validation.csv` (exact matches only), `step7_audience/2_validate/` |
| Address measure of record (MFTE VIMP + PP2, z on the 194-show reference); gap d = .66 (corpus-SD gap .56) | `step6_register/mfte_address.py`, `results/mfte_measure.md`, `data/output/dirz_reference_scale.csv` |
| Stability: ICC, quarter fixed effects, event study with 300 placebo dates, topic decomposition | `results/mfte_stability_checks.md` |
| Address vs one-sidedness: show-level r .35, passage-level r −.01 within shows | `results/address_vs_intensity.md` |
| Command types by Biber (2006) verb category as assigned by MFTE, and the "you know" bigram | `step6_register/mfte_imperative_classes.py`, `step7_audience/inputs/mfte_imperative_show_scores.csv`, `results/imperative_classes_audience.log` |
| Two-wave match (Y1 572/52, Y2 675/49, pooled 1,247/64; 1,200/63 with complete data); Richardson leverage point | `step7_audience/1_match/`, `results/y2_RESULTS.md` (sections "Maximal consistent match" and "Shape of the education association") |
| Year 2 pre-registered test (H1–H3 fail) and errata | `results/prereg_y2/prereg_y2.md` + `.sha256`, `step7_audience/5_prereg/y2_test.py`, `results/y2_RUN_2026-09-23T1710Z.log`, `results/prereg_y2/prereg_y2_ERRATA.md` |
| Education composition: −0.25 alone, −0.21 net of one-sidedness; terciles, credential levels, income held constant; threshold shape; Year 1 alone | `step7_audience/9_education_probes/`, `results/y2_RESULTS.md`, `scans/address_edu_indirect_paths.csv`, `results/address_education_adjacent.log` |
| Attitudes follow one-sidedness (election distrust +0.29 both years), not address; between-show variance ceiling | `scans/scan_maximal_match_verified.csv`, `results/between_show_ceiling.log` |
| Pew replication: W165 named-source audiences (Spearman −.93 over 9), W118 main podcast (education −0.38; connection to host follows one-sidedness) | `step7_audience/8_pew/pew_checks.py`, `data/output/pew_w165_source_audiences.csv`, `results/pew_external_checks.log` |
| Reviews: 16,434 reviews / 141 shows; reading grade −0.35 per SD (116 shows); ratings follow one-sidedness (119 shows); reviews per month | `step7_audience/7_reviews/collect_reviews.py`, `data/output/show_level_outcomes.csv`, `results/y2_RESULTS.md` |
| Multiple comparisons: Benjamini–Hochberg q < .10 within each scan family | `scans/*.csv` (q columns), `results/y2_RESULTS.md` |
| Figures 1–7 | `figures/build_figures.py` -> `figures/fig*.png/.pdf` |

Exploratory results that are recorded but not claimed in the paper (life satisfaction × education, newspaper naming,
the stance index, outrage coding, show-level chart and ad outcomes) are in `results/y2_RESULTS.md`,
`outrage_llm_results.md` and `stance_index_y1_to_y2.log`.

## Data availability

Committed: the frozen Apple chart (`data/raw/`), the inclusion decisions, the sampling manifests, every
show-level table, the LLM passage labels (`ideology_cap40.csv`, labels only), the hand-coding files, the
pre-registration texts with their SHA-256 records, all result logs, scans and write-ups, and the figures.

Not committed, and how to get it:

- **Transcripts and audio.** Regenerable from the frozen chart, the manifests and `step3_transcribe/` (about
  25,900 hours of audio; the audio archive was deleted on 11 September 2026). The transcripts are available from
  the authors on request.
- **Kettering-Gallup *Democracy for All* Year 1 and Year 2 microdata** (`data/external/kettering/`) and every
  respondent-level extract built from them (`step7_audience/inputs/kett.pkl`, `verbatims_long.csv`,
  `match_*.csv`, the frozen alias tables including `prereg_y2_alias_FROZEN.csv`). Obtain
  from the Kettering Foundation under their data agreement; the alias tables' hashes are recorded so a re-built
  copy can be verified.
- **Pew American Trends Panel** waves 118, 141, 144, 150, 155, 165 (`data/external/pew*/`). Public-use files;
  download from Pew Research Center. Only W118 and W165 are used in the paper.
- **Apple Podcasts reviews** (`data/output/apple_reviews.csv`, 16,434 public reviews). Regenerable with
  `step7_audience/7_reviews/collect_reviews.py`; the per-show aggregates are committed.
- **DIME** recipient and contributor files (Bonica, public version 4.0, 1979–2024), the **Brookings Political
  Podcast Project** export, the **Brysbaert** concreteness norms, and the podcastindex database. Third-party;
  drop into `data/external/`.
- `step7_audience/inputs/remeasured.csv` (281 MB passage-level register features) and
  `data/output/scoring_chunks*.csv.gz` (the passage tables): regenerable from the transcripts by
  `step6_register/measure_register2.py` and `step4_ideology/build_scoring_chunks.py`.

## Running

Run every command from the repository root with `.venv/bin/python` (`requirements-venv.txt`; the topic arm uses
`.venv-tlda` from `requirements-venv-tlda.txt`). The transcription worker uses the GPU environment
`../whisperx-env`. Step 1 is a package (`python -m step1_frame.run`); step 7 scripts share `paths.py` and
`common.py` and run as `.venv/bin/python step7_audience/<sub-folder>/<script>.py`. `score_ideology.py` and
`score_outrage_llm.py` read an OpenAI key from `.env` (not committed).

1. `.venv/bin/python -m step1_frame.run --chart-date 20260713`
2. `.venv/bin/python step2_sample/sample_ladder.py --max-k 30`, then `bash step3_transcribe/run_ladder.sh`

3. `.venv/bin/python step4_ideology/build_scoring_chunks.py --build`, `sample_chunks_for_scoring.py --cap 40`,
   `score_ideology.py`
4. `step5_hosts/` then `step6_register/` in the order their READMEs give; the MFTE run is
   `run_mfte_corpus.sh` under `mfte_supervise.sh`, then `mfte_address.py` and `mfte_imperative_classes.py`
5. `step7_audience/` sub-folders 1 to 6 in order; then `5_prereg/y2_test.py`, `9_education_probes/`,
   `8_pew/pew_checks.py`, `7_reviews/collect_reviews.py`
6. `.venv/bin/python figures/build_figures.py` (needs the restricted survey files for Figures 4–7)

## Caveats recorded in the repository

- `data/output/lean_validation.csv` joins Brookings labels by fuzzy title. Only rows with `match_status ==
  "matched"` are correct; the scripts that read it filter on that column (fixed 17 September 2026).
- The stability checks of 17 September (`results/mfte_stability_checks.md`) replace the earlier "flat
  across topics" claim: within a show, address rises with how conversational an episode is, not with its
  political subject.
- The education association is a threshold, not a gradient: it is carried by the lowest-address third of shows
  (`y2_RESULTS.md`, "Shape of the education association"). One matching decision (folding Heather Cox
  Richardson's newsletter readers into the show that reads it aloud) removes it by leverage, not by composition.
- Year 1 alone gives −0.20 (p .02/.04) for address on education and −0.11 (n.s.) net of one-sidedness; Year 2
  gives −0.33 / −0.32. The pooled estimate is −0.25 / −0.21.
- The Year 2 pre-registration text, its SHA-256 record, the alias decisions and the errata are in `results/prereg_y2/`;
  the Year 1 pre-registration files and the paper drafts are held by the authors.
- The Year 2 errata (`results/prereg_y2/prereg_y2_ERRATA.md`): the cynicism composite was mislabelled and item Q34B
  was reworded between waves.

## Use of generative AI

Passage labels were produced by OpenAI gpt-5.4-mini as a research instrument (see step 4). Anthropic's Claude,
through Claude Code, assisted with analysis code, running checks, drafting figures and editing prose. Design,
pre-registrations, hand coding and interpretation are the authors' own.
