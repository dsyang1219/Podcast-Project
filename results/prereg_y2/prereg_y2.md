# Pre-registration: YEAR 2 confirmatory test of listener-directed address -> election distrust and institutional cynicism, with ideological intensity as the named rival
STATUS: DRAFT, written 2026-09-23 BEFORE the Year 2 microdata were opened. To be FROZEN (SHA-256 recorded in
prereg_y2.sha256) after two things only: (a) the Year 2 codebook has been read to confirm item numbering, and (b) the
respondent-show match has been built and frozen. No outcome of any Year 2 respondent is examined before the hash.
Nothing below changes after hashing except the bracketed item numbers, which are filled from the codebook.

## Why this exists
Year 1 established, exploratorily, that listeners of address-heavy corpus shows are more distrustful of elections, more
cynical about institutions, and less formally educated (handoff/results/mfte_measure.md). Three Year-1 confirmatory tests
failed as frozen. The 22 Sept 2026 analysis (handoff/results/address_vs_intensity.md) showed that address and ideological
intensity are distinct properties of speech (uncorrelated within shows at the passage level; only address tracks
ideological direction) but cannot be separated as predictors of audience attitudes with 54 shows: each coefficient
roughly halves when the other is included. Year 2 supplies fresh listeners of the same shows. This test asks, in advance,
(1) whether the address associations replicate, and (2) whether address predicts attitudes net of intensity.

## Exposure (FROZEN files)
- Address: `mfte_z` per show from step7_audience/inputs/mfte_show_scores.csv (MFTE VIMP + PP2, z on the 194-show
  reference; measure of record since 16 Sept 2026). SHA-256 of that file recorded at freezing.
- Intensity: `inten` per show from step7_audience/inputs/directive_final.csv (share of political passages labelled
  non-moderate, LLM labels of record), standardised per show-SD on the 194 shows. Shows without an LLM intensity score
  (Majority Report; any out-of-frame show) use `inten_use` from inten_distilled.csv and are flagged.
- A respondent's exposure is the mean over the corpus shows they name; cluster = alphabetically first show named.

## Sample and match
Year 2 respondents who name a corpus show in the open-text news-source items (Y1: Q17_1..3; Y2: [confirm]). Matching
uses, in order: (1) exact corpus title match as in step7_audience/1_match/build.py; (2) the frozen alias tables from Year 1
(handoff/prereg/holdout_alias_table_FROZEN.csv patterns, withinframe_alias_patterns_FROZEN.json) applied unchanged; (3) a
re-scan for new strings, reviewed by eye and frozen as prereg_y2_alias_FROZEN.csv BEFORE any outcome is examined.
Year 2 is a fresh sample (no uig or ENTITY_ID overlaps Year 1), so no panel exclusion is needed. Out-of-frame shows (the ten chart-absent shows) are excluded from the primary test and reported in a
secondary pooled specification, as in Year 1.
A show enters with >= 3 respondents. All per-show counts, exposures and audience leans are recorded in the results file
before the test runs.

## Outcomes (standardised on the FULL Year 2 sample, never on the listener subsample)
Codebook check (23 Sept 2026): Year 2 (n = 23,683, fielded 16 April to 21 May 2026, a fresh sample with no ID overlap
with Year 1) keeps Year 1's numbering for Q17_1-3, Q19, Q31, Q32C, Q33E, Q33H, Q34B, Q41, Q43, EDU, AGE, WEIGHT. It does
NOT carry the Q35 items or the Q20 platform items.
PRIMARY  election distrust: Q33E reversed plus Q32C, each z-scored on the full Year 2 sample, averaged, re-standardised
         (identical construction to Year 1).
SECONDARY institutional cynicism, the 4-item discovery composite (Q35E, Q35C not available): Q31, Q33E, Q33H, Q34B
         reversed, z-scored, averaged (>= 3 answered), re-standardised. This is the composite of step7_audience/common.py
         (addcomp), which correlates .93 with the 8-item scale in Year 1.
TERTIARY respondent education (audience composition), z on the full sample.
Exclusivity (naming no institutional source) is NOT a pre-registered outcome in Year 2.

## Controls
Party identification (5 categories from Q41/Q43, leaners assigned), political attention (Q19), age, education (except
when the outcome), and the share of named shows that lean right. The Year 1 right-wing platform count cannot be built
(Q20 items absent from Year 2) and is dropped; this is the only change from the Year 1 control set. Same construction
otherwise as step7_audience/common.py.

## Hypotheses and specifications
H1 (replication). Address predicts election distrust (primary) and cynicism (secondary): positive coefficient.
H2 (rival). With intensity added as a second show-level covariate on the same per-show-SD scale, the address coefficient
on election distrust remains positive.
H3 (composition). Address predicts lower education, and does so net of intensity.
Specifications, run in this order and all reported: (i) address alone; (ii) address + intensity; (iii) intensity alone.

## Inference (identical to the Year 1 tests)
OLS with show-clustered inference: CR2 (Bell-McCaffrey) with Satterthwaite df as the primary p-value, Rademacher wild
cluster bootstrap (2,000 draws) alongside; one-sided alpha = .05; two-sided reported; survey-weighted as robustness.
Success criteria, fixed now:
- H1 passes if the address coefficient in (i) is positive with one-sided CR2 p < .05 AND wild p < .05 on the PRIMARY
  outcome. The secondary outcome is reported but does not decide H1.
- H2 passes if the address coefficient in (ii) is positive with one-sided CR2 p < .05 AND wild p < .05 on the primary
  outcome. If H1 passes and H2 fails, the paper reports that the association is shared with intensity and cannot be
  attributed to address; if both fail, the paper reports the Year 1 associations as unreplicated.
- H3 passes on the same criterion for education in (ii).
No re-specification after filing. Reported regardless: coefficients, CIs, both p-values, df, leave-one-show-out ranges,
per-show n / address / intensity / audience lean, the pooled specification with out-of-frame shows, and the panel
robustness if applicable.

## Power, stated in advance
If Year 2 matches Year 1 (about 500 corpus listeners in about 50 shows), Satterthwaite df will again be about 4 to 7 and
the minimum detectable one-sided effect about 0.15 to 0.20 SD per SD. The Year 1 estimates are +0.18 (election distrust)
and +0.16 (cynicism) for address alone, and +0.07 to +0.12 for address net of intensity. H1 therefore has roughly even
odds of passing if the Year 1 effect is real; H2 is under-powered unless Year 2 is larger, and a failure of H2 alone is
weak evidence against an independent address effect. This is stated now so it cannot be discovered afterward.
Pooling Year 1 and Year 2 listeners (about 1,000 in about 60 shows) is pre-specified as a secondary analysis for H2
only, with a year indicator, reported after the primary test regardless of its result.

## Asymmetry clause
A pass on H1 replicates the Year 1 audience finding on fresh respondents. A pass on H2 is the first evidence that address
matters over and above one-sidedness. A fail on either is reported as such; neither is re-run under a different frame,
outcome, or exposure.
