# Political-content frame with the published instrument (15 September 2026)

Decision: the frame variable is now the share of a show's sampled passages that stage 1 of the Much et al. (2026)
rubric marks as political (word-weighted), computed by `step7_audience/6_exploratory/llm_poldensity.py`. The
hand-written political-term dictionary in `corpus_poldensity.py` is retired as the frame variable (kept for the
record). Agreement between the two across the 194 shows: r = 0.73, Spearman 0.75; they disagree at the tail,
where the dictionary undercounts shows that discuss politics without naming US figures (Rev Left Radio, War on
the Rocks, Bad Hasbara) and overcounts general news programmes (NBC Nightly News, New Yorker Radio Hour, Slate,
KQED Forum), which the instrument rates as mostly non-political.

Frame cut: corpus 10th percentile of LLM density = 0.644. Below it: The Arena, NBC Nightly News, Maiden Mother
Matriarch, Peter McCormack, Dad Saves America, Timothy Gordon, Michael Berry, New Yorker Radio Hour, KQED Forum,
Hood Politics, Candace, Week In Review, Slate News, Political Orphanage, Newt's World, Countdown with Olbermann,
American Conservative University, ChinaTalk, Dishcast, Volts.

## Corpus listeners only (discovery 385 + within-frame pocket 150; no chart-absent shows)

Exposure in reference-scale SD units; controls party, attention, age, education, right-wing platform count, show
lean; CR2 + wild (1,500 draws). Education model omits education from the controls.

| sample | shows / listeners | election distrust | cynicism (8-item) | education (raw EDU units) |
|---|---|---|---|---|
| all corpus listeners, no frame | 54 / 535 | +0.224 (CR2 .004, wild .011) | +0.206 (.020 / .022) | −0.617 (.004 / .001) |
| dictionary density ≥ .15 (old) | 50 / 502 | +0.230 (.004 / .012) | +0.249 (.005 / .003) | −0.682 (.004 / .005) |
| **LLM density ≥ .644 (new)** | **49 / 456** | **+0.222 (.004 / .023)** | **+0.182 (.023 / .039)** | **−0.604 (.006 / .001)** |

Net of populism (Rooduijn–Pauwels rate) and ideological intensity, corpus only: election distrust +0.060
(p .36 / .35) in the LLM frame, +0.118 (.13 / .16) in the dictionary frame, +0.119 (.12 / .15) unframed;
cynicism +0.139 (.076 / .062), +0.142 (.060 / .051), +0.139 (.063 / .053).

**Which covariate does it (election distrust, competing show features on a per-show-SD scale):**

| model | LLM frame (49 / 456) | unframed (54 / 535) |
|---|---|---|
| address alone | +0.222 (.004 / .013) | +0.224 (.004 / .011) |
| address ǀ populism | +0.216 (.008 / .027) | +0.216 (.008 / .031) |
| address ǀ intensity | +0.061 (.37 / .38) | +0.118 (.13 / .15) |
| address ǀ populism + intensity | +0.060 (.36 / .36) | +0.119 (.12 / .15) |
| intensity alone | +0.321 (<.001 / <.001) | +0.199 (.044 / .017) |
| intensity ǀ address | +0.283 (.001 / .001) | +0.144 (.135 / .005) |
| populism alone | +0.075 (.52 / .11) | +0.084 (.45 / .10) |
| populism ǀ address | +0.034 (.78 / .34) | +0.034 (.72 / .28) |

**The same separation for the other survey outcomes** (LLM frame 49 / 456; unframed 54 / 535 in brackets):

| model | cynicism 8-item | education (SD) | exclusivity |
|---|---|---|---|
| address alone | +0.182 (.023/.045) [+0.206] | −0.297 (.006/.003) [−0.304] | +0.087 (.054/.053) [+0.126] |
| address ǀ populism | +0.159 (.050/.095) [+0.177] | −0.281 (.007/.002) [−0.281] | +0.075 (.072/.057) [+0.107] |
| address ǀ intensity | +0.138 (.072/.043) [+0.141 (.051/.029)] | −0.124 (.22/.24) [−0.178 (.10/.14)] | +0.050 (.35/.33) [+0.024] |
| intensity alone | +0.233 (.002/.007) [+0.201] | −0.403 (.003/.003) [−0.271] | +0.129 (.004/.020) [+0.173] |
| intensity ǀ address | +0.148 (.046/.032) [+0.135 (.010/.022)] | −0.324 (.020/.022) [−0.185 (.12/.02)] | +0.098 (.10/.13) [+0.162 (.015/.037)] |
| populism ǀ address | +0.117 (n.s.) | −0.078 (n.s.) | +0.057 (n.s.) |

Every address association is unchanged by populist vocabulary and roughly halved or worse by ideological
intensity, and intensity is at least as strong a predictor of each outcome. Cynicism is the only outcome where
address keeps a marginal independent coefficient (+0.14, wild p .03–.05). The education-composition result, which
the pooled record called robust, is not separable from intensity inside the corpus either (−0.12 to −0.18, n.s.).

Show-level correlations (49 shows): address–populism +0.30, address–intensity +0.44, populism–intensity +0.44.
Populist vocabulary is not the explanation: address is unchanged when it is added, and populism itself does not
predict election distrust here. Ideological intensity (the share of a show's political passages labelled
anything but moderate) is: it predicts election distrust on its own, and once it is in the model address's
coefficient falls to a quarter (frame) or a half (unframed) of its size. Inside the frame intensity is the
stronger predictor; unframed the two are attenuated together and neither clearly wins.

Cut-off sensitivity (LLM density, election distrust): 5th pct +0.225, 10th +0.222, 15th +0.223, 20th +0.220,
25th +0.222; CR2 p .003–.008, wild p .011–.024.

## What changes in the paper

1. The frame variable is a published instrument; the density dictionary leaves the methods.
2. Inside the corpus the frame does almost no work: the unframed 54-show estimate (+0.224) is the same as the
   framed one. The frame mattered in the pooled analysis only because it removed Rogan, Ryan and Kelly, which
   are outside this paper's scope. Report the frame as a robustness restriction, not as the source of the
   result.
3. The "+0.194 net of populism and intensity" figure was a pooled (56-show) estimate. Within the corpus alone
   the election-distrust association does not survive those two covariates (+0.06 to +0.12, n.s.); cynicism is
   marginal (+0.14, p ≈ .05–.08). The paper must say so: address and populist vocabulary are correlated across
   shows (r ≈ .4–.5 among the features) and the corpus cannot separate them.
4. Education composition is unaffected by any frame choice.
