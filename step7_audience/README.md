# Step 7 — the audience: linking shows to the people who listen to them

**What this step does.** Joins the show-level measures from steps 4 to 6 to the Kettering-Gallup *Democracy for
All* survey (Year 1, n = 20,338), whose respondents named the news sources they use in their own words. Every
audience-side result in the paper comes from here: the validation of the address measure against listeners'
party, the population finding, the three pre-registered tests, the education-composition result, and the
political-podcast-frame results.

**Inputs (`inputs/`, the inputs of record).** `kett.pkl` (survey with derived controls), `verbatims_long.csv`
(every named source), `match_strict.csv` / `match_expanded.csv` (respondent to show), `directive_final.csv`
(show-level address score `dir_z`, lean and side), `show_measures.csv` and the other `show_*.csv` feature tables,
`remeasure2.py` (pinned feature code), and the holdout scoring outputs. Survey microdata live in
`data/external/kettering/`; Pew waves in `data/external/pew_*`.

**Outputs (`outputs/`).** Result tables and run logs; the ones the paper cites are copied to `handoff/`.

## Shared code

- `paths.py` — the only place paths live (`ROOT`, `DATA`, `HANDOFF`, `INPUTS`, `OUTPUTS`).
- `common.py` — the respondent frames `st` (strict match) and `ex` (expanded match), the control matrix
  (`design()`: party, attention, age, education), and the inference helpers: `clus()` conventional clustered
  OLS, `wild()` wild cluster bootstrap, `showperm()` show-level permutation, `cr_all()` CR1 + CR2
  (Bell-McCaffrey with Satterthwaite df) + wild bootstrap in one call.

Every script begins with `from common import *` after putting this folder on the path, so any script can be run on
its own: `.venv/bin/python step7_audience/4_discovery/cr2.py`.

## The sub-steps, in order

| folder | question it answers | main scripts |
|---|---|---|
| `1_match/` | Which respondents listen to which corpus shows? | `model.py` builds `kett.pkl`; `build.py` the verbatim table and the strict match; `expand.py` the audited host-name expansion; `diet.py` the news-diet segments for the whole sample |
| `2_validate/` | Does `dir_z` measure what we say it does, and is it a stable trait of a show? | `tri_strong.py`, `audience.py` (label vs DIME vs listeners' party), `event_study.py`, `topics_lda.py`, `imp_decomp.py`, `you_split.py` |
| `3_population/` | Are podcast-diet respondents different from mainstream-diet respondents? | `pop_profile.py`, `gradient.py`, `popcontrast.py`, `survive.py` → `perm.py`, `scan.py` |
| `4_discovery/` | In the discovery sample, what does address predict, under strict inference? | `verify.py`, `cr2.py`, `decomp.py`, `excl_strong.py`, `families.py` |
| `5_prereg/` | The three pre-registered confirmatory tests, executed exactly as frozen | `q28.py`; `score_holdout_final.py` → `holdout_test.py`; `score_mr.py` → `withinframe_test.py` |
| `6_exploratory/` | After the confirmatory tests failed: the political-content frame, education composition, populism and intensity | `corpus_poldensity.py`, `explore_scan.py`, `distill_inten.py`, `populist.py`, `rp_antimedia.py`, `rp_test.py`, `rp_words.py`, `horserace.py`, `hr_infer.py` |

Each sub-folder has its own README with the data flow between its scripts.

## Inference conventions used throughout

Shows are the clusters. Every headline estimate reports CR2 (Bell-McCaffrey) standard errors with Satterthwaite
degrees of freedom as the primary p-value, a Rademacher wild cluster bootstrap alongside, and where noted a
show-level permutation test, leave-one-show-out, and Benjamini-Hochberg correction within outcome families.
Composites are standardised on the full survey, never on the listener subsample; address is standardised on the
194-show corpus reference. Pre-registered tests use one-sided alpha = .05 as frozen. Random seed 5 everywhere.
