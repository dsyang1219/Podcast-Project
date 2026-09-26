# Model-coded outrage (Sobieraj & Berry 2011, 12 of 13 types) — results, 22 Sept 2026

Run: step6_register/score_outrage_llm.py, gpt-5.4-mini-2026-03-17, structured output, codebook prompts/outrage_sb.md
plus a political-target scope rule (added after pilot v1 flagged ads/anecdotes as "emotionally charged"). 152,130 of
152,152 cap-40 passages scored (22 failed after retries); $31.70. Emotional display (type 3) dropped: audio-only.
Passage base rates: any outrage 59.9%; emotlang 31%, belittle 19%, exaggeration 15%, mockery 12%, character 12%,
extremize 9%, insult 7%, conflagration 7%, namecall 5%, obscene 4%, slippery 3%, sparring 3%.
Show scores: word-weighted share of sampled passages with each type, z on the 194 reference shows; index = mean of
the 12 z's. Files: step7_audience/inputs/outrage_llm_show_scores.csv, data/output/outrage_llm_show_rates.csv.

## Show level (194 shows)
| measure | R−L gap (SD) | r DIME (112) | r LLM side | AUC | r address |
|---|---|---|---|---|---|
| outrage index | +0.41 (perm p .005) | .10 | .12 | .63 | .37 |
| exaggeration | +0.85 | .28 | .34 | .74 | .25 |
| extremize | +0.65 | .24 | .24 | .69 | .19 |
| namecall / conflagration / belittle / slippery | +0.4 to +0.5 | .10–.27 | .10–.19 | .60–.67 | |
| insult / character / mockery / emotlang / sparring | ≤ +0.15 | ≈ 0 | ≈ 0 | .55–.59 | |
| obscene | −0.38 (left higher) | −.18 | −.21 | .43 | |
| address (mfte_z), for comparison | +0.66 | .41 | .33 | .69 | — |
r(outrage index, intensity) = .72; r(outrage index, address) = .37. Top of the outrage scale is mixed (Necessary
Conversation, Kulinski, Skepticrat, Mary Trump on the left; Wendy Bell, Benny Show on the right); bottom is
institutional/left (POLITICO Energy, KQED Forum, Lectures in History, Volts, War on the Rocks).

## Audience (535 listeners, 53–54 clusters; CR2 / wild p)
| exposure | cynicism | election distrust | education |
|---|---|---|---|
| outrage index | +0.21 (.03/.10) | +0.23 (.14/.10) | −0.37 (.07/.05) |
| exaggeration | +0.18 (.00/.01) | +0.18 (.08/.04) | −0.26 (.05/.02) |
| extremize | +0.16 (.01/.03) | +0.16 (.09/.11) | −0.21 (.03/.02) |
| insult | +0.12 (.07/.13) | +0.16 (.04/.05) | −0.24 (.01/.01) |
| address | +0.16 (.03/.04) | +0.23 (.01/.01) | −0.23 (.03/.01) |
| address ǀ outrage | +0.09 (.14/.17) | +0.18 (.02/.04) | −0.09 (.40/.34) |
| outrage ǀ address | +0.15 (.12/.20) | +0.12 (.32/.10) | −0.31 (.14/.04) |
| outrage ǀ intensity | −0.09 (n.s.) | −0.30 (n.s.) | −0.34 (n.s.) |
| intensity ǀ outrage | +0.29 (.07/.11) | +0.51 (.01/.00) | −0.03 (n.s.) |
| address ǀ outrage + intensity | +0.09 (.08/.09) | +0.12 (.11/.13) | −0.10 (n.s.) |

## Reading
1. On WHICH SIDE does it: address separates left from right far better than outrage (0.66 vs 0.41 SD; r with DIME
   .41 vs .10). Outrage's side difference is carried by exaggeration and extremizing (right) and obscenity (left);
   insults, mockery, character attacks and emotional language do not differ by side. Finding 1 stands with the
   model measure, in a more precise form.
2. On WHO LISTENS: the lexical result ("hostility predicts none of these attitudes") does NOT hold with the model
   measure. Outrage-heavy shows have more cynical and less educated audiences, at about the same strength as
   address-heavy shows; address keeps election distrust net of outrage but not cynicism or education.
3. Outrage and ideological intensity are nearly the same show property (r .72); with both in the model intensity
   dominates and outrage flips sign (collinearity). The three rivals (address, outrage, intensity) cannot be
   separated with 54 shows.
Election-distrust composite here = Q33E reversed + Q32C, z on full survey (gives address +0.23; mfte_measure.md
reported +0.18; still to reconcile).
