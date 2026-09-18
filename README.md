# Political Podcast Corpus

Code and record for *Talking at the Audience: listener-directed address in political podcasts, and the
institutional cynicism of the people who listen* (Yang & Alvarez, Linde Center, Caltech, 2026).

The paper measures how directly the hosts of 194 US political podcasts address their listeners, shows that
right-leaning hosts do so more, that it is a stable trait of a show, and that the audiences of address-heavy
shows are more distrustful of elections, more cynical about institutions, and less formally educated.

**Status (18 September 2026).** Corpus frozen at 31,191 episodes / 194 shows for the analysis; the transcription
ladder continues on rma2 (band 15). The address measure of record is the MFTE-based one from 16 September.
Pre-registered confirmatory tests were run and failed as frozen; the audience results are exploratory.

## Layout

The repository is **seven numbered steps**, run in order; each folder's `README.md` says what its scripts do.
`handoff/` is the paper record. `topic_arm/` is a side project (topic model / ideology recoverability) that is
not in the paper.

```
step1_frame       Apple Podcasts US Politics chart (frozen 13 July 2026) -> RSS metadata -> six inclusion rules
                  -> data/output/corpus.csv, exclusions.csv, lean_validation.csv          (219 shows)
step2_sample      episode table -> priority-ladder manifest (census before 2018, then k per show per quarter)
                  -> data/output/sample_out/sample_manifest_*.csv
step3_transcribe  manifest -> audio download -> faster-whisper large-v3-turbo on the GPU
                  -> data/transcripts/<show_id>/<episode_id>.json + .txt                    (not in git)
step4_ideology    transcripts -> 750-character passages -> two-stage LLM rubric (Much et al. 2026)
                  -> data/output/ideology_cap40.csv; show-level side, intensity, political density
step5_hosts       host names -> Wikidata -> DIME donor scores per show
                  -> data/output/shows_205_host_dime_scores.csv
step6_register    transcripts -> listener-directed address: MFTE tagger (measure of record) and the
                  earlier parse/phrase-list version; hostility batteries alongside
                  -> step7_audience/inputs/mfte_show_scores.csv, mfte_episode_rates.csv, directive_final.csv
step7_audience    show measures + Kettering-Gallup survey + Pew ATP -> validation, population finding,
                  pre-registered tests, exploratory audience results
handoff           paper record: pre-registrations, results logs, scans, paper text, slides, notes
topic_arm         topic-model arm (progress report RQ1); not used by the paper
data              raw/ (frozen chart, committed)  external/ (restricted, not committed)
                  output/ (small tables of record committed)  transcripts/, audio_queue/ (not committed)
```

## Where the paper's numbers come from

| paper result | where |
|---|---|
| Corpus: 250-show frame, six inclusion rules, 219 -> 204 -> 194 shows | `step1_frame/`, `data/output/exclusions.csv`, `handoff/paper/methods_section_v2.md` |
| Ideology instrument (gpt-5.4-mini, Much et al. prompts verbatim), side / intensity / density | `step4_ideology/`, prompts in `step4_ideology/prompts/ideology.md` |
| Passage-level hand coding (36 passages; gate kappa .65, direction .51) | `data/output/ideology_coding_human_2026-08-26.csv`, `ideology_coding_key_production_2026-09-16.csv` |
| Show-level validation of the side score: Brookings (22/24, kappa .83), DIME (r .68, 112 shows), listeners' party (r .92, 25 shows) | `step5_hosts/`, `data/output/lean_validation.csv` (exact matches only), `step7_audience/2_validate/` |
| Address measure of record (MFTE VIMP + PP2, z-scored on the 194-show reference) | `step6_register/mfte_address.py`, `handoff/results/mfte_measure.md`, `handoff/prereg/dirz_reference_scale.csv` |
| Finding 1: right-leaning hosts address listeners more (r .41 with DIME, gap .66 SD) | `handoff/results/mfte_measure.md` |
| Stability: ICC, quarter fixed effects, event study with placebo dates, topic decomposition | `handoff/results/mfte_stability_checks.md` |
| Audience associations (election distrust, cynicism, education) and the intensity rival | `step7_audience/6_exploratory/`, `handoff/results/mfte_measure.md`, `handoff/results/exploratory_populism.md` |
| Three pre-registered tests, all reported as failed | `step7_audience/5_prereg/`, frozen texts and hashes in `handoff/prereg/`, results in `handoff/results/` |
| Population finding: podcast-diet respondents more cynical, both parties, every education level, Pew lagged | `step7_audience/1_match/diet.py`, `step7_audience/3_population/` |

## Data availability

Committed: the frozen Apple chart (`data/raw/`), the inclusion decisions, the sampling manifests, every
show-level table, the LLM passage labels (`ideology_cap40.csv`, labels only), the hand-coding files, the
pre-registration texts with their SHA-256 records, all result logs and write-ups, and the paper text.

Not committed, and how to get it:

- **Transcripts and audio.** Regenerable from the frozen chart, the manifests and `step3_transcribe/` (about
  25,900 hours of audio; the audio archive was deleted on 11 September 2026). The transcripts are available from
  the authors on request.
- **Kettering-Gallup *Democracy for All* Year 1 microdata** (`data/external/kettering/`) and every
  respondent-level extract built from it (`step7_audience/inputs/kett.pkl`, `verbatims_long.csv`,
  `match_*.csv`, the frozen alias tables in `handoff/prereg/`). Obtain from the Kettering Foundation under their
  data agreement; the alias tables' hashes are recorded so a re-built copy can be verified.
- **Pew American Trends Panel** waves 150, 155, 165 (`data/external/pew*/`). Download from Pew Research Center.
- **DIME** recipient and contributor files (Bonica), the **Brookings Political Podcast Project** export, the
  **Brysbaert** concreteness norms, and the podcastindex database. Third-party; drop into `data/external/`.
- `step7_audience/inputs/remeasured.csv` (281 MB passage-level register features) and
  `data/output/scoring_chunks*.csv.gz` (the passage tables): regenerable from the transcripts by
  `step6_register/measure_register2.py` and `step4_ideology/build_scoring_chunks.py`.

## Running

Run every command from the repository root with `.venv/bin/python` (`requirements-venv.txt`; the topic arm uses
`.venv-tlda` from `requirements-venv-tlda.txt`). The transcription worker uses the GPU environment
`../whisperx-env`. Step 1 is a package (`python -m step1_frame.run`); step 7 scripts share `paths.py` and
`common.py` and run as `.venv/bin/python step7_audience/<sub-folder>/<script>.py`. `score_ideology.py` reads an
OpenAI key from `.env` (not committed).

1. `.venv/bin/python -m step1_frame.run --chart-date 20260713`
2. `.venv/bin/python step2_sample/sample_ladder.py --max-k 30`, then `bash step3_transcribe/run_ladder.sh`
   (hardware notes in `handoff/notes/`)
3. `.venv/bin/python step4_ideology/build_scoring_chunks.py --build`, `sample_chunks_for_scoring.py --cap 40`,
   `score_ideology.py`
4. `step5_hosts/` then `step6_register/` in the order their READMEs give; the MFTE run is
   `run_mfte_corpus.sh` under `mfte_supervise.sh`, then `mfte_address.py`
5. `step7_audience/` sub-folders 1 to 6 in order

## Caveats recorded in the repository

- `data/output/lean_validation.csv` joins Brookings labels by fuzzy title. Only rows with `match_status ==
  "matched"` are correct; the scripts that read it filter on that column (fixed 17 September 2026).
- The stability checks of 17 September (`handoff/results/mfte_stability_checks.md`) replace the earlier "flat
  across topics" claim: within a show, address rises with how conversational an episode is, not with its
  political subject.
- The pre-registration texts cite the pre-12-September file paths (`handoff/holdout_alias_table_FROZEN.csv`
  etc.); the files now live in `handoff/prereg/`. The texts were not edited because their hashes are recorded.

## Use of generative AI

Passage labels were produced by OpenAI gpt-5.4-mini as a research instrument (see step 4). Anthropic's Claude,
through Claude Code, assisted with analysis code, running checks, and editing prose. Design, pre-registrations,
hand coding and interpretation are the authors' own.
