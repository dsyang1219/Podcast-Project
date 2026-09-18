# Pre-registration ADDENDUM: WITHIN-FRAME holdout test of listener-directed address -> institutional cynicism
STATUS: FROZEN at the timestamp in handoff/prereg_withinframe.sha256. Filed AFTER the out-of-frame test (handoff/holdout_RESULTS.md)
and BEFORE any outcome of any respondent below has been examined. Same hypothesis, outcome, model and inference as
handoff/prereg_holdout.md; only the frame and match differ. Nothing below changes after hashing.

## Why this exists
prereg_holdout.md stated "no within-frame holdout exists in Y1: every corpus-matched respondent was used in discovery."
That was wrong. A systematic re-scan of Q17 verbatims (2026-09-03) found fresh respondents who name CORPUS shows by strings the
discovery matchers missed: misspellings (midas touch), abbreviations (tyt), show-title-only strings for host-mapped shows
(majority report, secular talk), multi-source strings (npr podcasts, pod save america, pbs newshour), and one show never in
either matcher (Timcast). These respondents are INSIDE the corpus sampling frame (Apple Politics chart) and were never analysed.

## Frame and match (FROZEN: handoff/withinframe_alias_table_FROZEN.csv, withinframe_alias_patterns_FROZEN.json)
Rule: title/host/abbreviation string match to a corpus show, reviewed by eye before freezing; respondents in the 385 strict,
544 expanded, or 749 out-of-frame sets are excluded; a show enters with >= 3 fresh respondents.
PRIMARY frame, 10 shows, 150 respondents (4 name two shows -> mean dir_z, cluster = alphabetically first show, as in discovery):
  MeidasTouch 61 (dir_z +0.94, L) | Timcast News 31 (+0.65, R) | Majority Report 28 (see clause) | Young Turks 10 (+0.63, L) |
  Kulinski/Secular Talk 6 (+1.08, L) | Candace 5 (+0.27, R) | Benny Show 4 (+0.63, R) | Destiny 3 (+0.59, L) |
  Pod Save America 3 (-0.12, L) | Tangle 3 (-1.01, L)
Documented decisions: "various social meidas" (typo for media), "pod save the people" (different show), "clay travis twitter"
dropped. Clay & Buck (12 respondents) name a different program from the corpus's Buck Sexton Show feed -> EXCLUDED from the
primary, reported as a robustness inclusion. Timcast strings ("tim pool", "timcast irl") map to the corpus feed Timcast News
(same host, different feed) -> included, with a leave-out reported.
MAJORITY REPORT CLAUSE: the show is in the 205-show corpus list but has no dir_z among the 194 scored shows. It is included
IF AND ONLY IF a dir_z can be computed from ALREADY-DOWNLOADED corpus transcripts with the pinned scoring procedure
(prereg_holdout.md 'Scoring procedure') before any outcome is examined; otherwise it is dropped and reported as dropped.
Its dir_z value, once computed, is recorded in the results file before the test runs.

## Hypothesis, outcome, model, inference: IDENTICAL to prereg_holdout.md
H1 dir_z -> 8-item institutional cynicism (z on full Kettering), positive. Secondary Q33E reversed. H2 exclusivity.
Controls: party (5), Q19 attention, age, education, show ideology (corpus lean L/R -> share_right), right-wing platform index.
OLS, SEs clustered by show; CR2 + Satterthwaite df primary, wild cluster bootstrap (Rademacher, 2000) alongside;
alpha = .05 one-sided; two-sided reported; survey-weighted as robustness.
Success: dir_z coefficient positive with one-sided CR2 p < .05 AND wild p < .05. No re-specification after filing.
Reported regardless: coefficient, CI, both p, df, leave-one-show-out, per-show n / dir_z / audience lean, Clay & Buck inclusion.

## Power, stated in advance
~150 respondents in 10 clusters, one cluster holding ~40% (MeidasTouch). Satterthwaite df will be ~3-5. Minimum detectable
one-sided effect ~0.3 SD/SD against a discovery estimate of +0.22. Prior probability of passing if the discovery effect is
real: roughly 35-40%. A NULL HERE IS WEAK EVIDENCE; a pass is informative. This test is therefore ALSO pre-specified as one
component of a pooled Wave-2 confirmation (within-frame pocket + new out-of-frame political-content shows, criterion
>= 15% political passages measured before outcomes), to be filed separately before those transcripts exist.
Exposure variance is narrow: 8 of 10 shows lie in dir_z [+0.27, +1.08]; the contrast is carried by Tangle (-1.01) and
Pod Save America (-0.12) at the low end, 6 respondents between them. Stated now so it cannot be discovered afterward.
