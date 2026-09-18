# Pre-registration: Q28 free-text test — written BEFORE any analysis of Q28 against register
Timestamp: 2026-09-02T19:15:13Z  (see file mtime; analysis script runs after this file is written)

## Data
Kettering/Gallup Democracy for All Y1. Q28: "What does democracy mean to you?" free text.
Sample A (show-level test): the 385 respondents exact-title-matched to corpus shows (match_strict.csv).
Sample B (population test): all respondents naming >=1 news source with non-empty Q28.
Q28 has never been examined in relation to register or podcast exposure prior to this file.

## Hypothesis
H1: Higher directive-register exposure (dir_z, mean across named corpus shows) predicts more
    institutionally-cynical framing of democracy in respondents' own words.
H2 (population): respondents naming a corpus political podcast frame democracy more cynically than
    respondents naming only TV/print/web sources.

## Measures on Q28 text (fixed in advance; dictionary-based, no LLM)
PRIMARY   M1  out-group pronoun share = (they|them|their|theirs) / all personal pronouns.
              Chosen because it is the paper's own validated register construct applied to the
              respondent's text. Predicted: positive.
SECONDARY M2  negation rate = (not|no|never|don't|doesn't|isn't|aren't|can't|won't|nothing|none) / words.
              Predicted: positive.
          M3  VADER compound valence. Predicted: negative.
          M4  cynicism lexicon rate = (corrupt|broken|rigged|lie|lies|joke|sham|fake|illusion|
              supposed|used to|no longer|rich|money|elite|elites|control|power|fail|failed|
              doesn't work|not really|in theory) / words. Predicted: positive.
Multiple comparisons: M1 is the confirmatory test at alpha=.05. M2-M4 are BH-corrected as a family.
Responses < 5 words excluded (rates undefined). Word count entered as a control.

## Model (Sample A)
M_k ~ dir_z + party ID + political attention + age + education + show ideology (share_right)
      + right-wing platform index + log(word count).
Inference: wild cluster bootstrap by show (Rademacher, 2000 reps). One-sided in the predicted
direction is pre-specified; two-sided p also reported.
Robustness (reported, not confirmatory): host-name-expanded sample (544); Gallup Panel subsample.

## Model (Sample B)
M_k ~ named_corpus_show + named_other_podcast + party ID + attention + age + education + log(words),
weighted (WEIGHT), HC1 SEs. Reference = named only TV/print/web.

## What counts as failure
M1 one-sided p >= .05 in Sample A. If M1 fails, H1 is not supported by Q28 regardless of M2-M4.
