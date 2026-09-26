# Out-of-frame generalization test — RESULT (confirmatory run)
Run: 2026-09-03T08:12:38Z | prereg sha256 7275fead… (matches last amendment hash) | alias table 4a848b8b… (matches freeze)
Raw log: handoff/results/holdout_RUN_2026-09-03T0812Z.log | show scores: handoff/results/holdout_show_dirz_FINAL.csv
Transcripts: 447 episodes (41–48 per show, all ten ≥ 25 floor), 54,439 passages, pinned pipeline (remeasure2 features, 750-char chunks, nw-weighted rates, z vs 194-show reference).
Respondents: 749 fresh (810 pairs, 57 two-show), all outside the 385/544 discovery sets. Outcome composite check: 8 reversed items all positively inter-correlated, α = .84, r = .93 with discovery 4-item.

## VERDICT: H1 NOT SUPPORTED. H2 NOT SUPPORTED. Secondary NOT SUPPORTED.
| test | b (SD cyn per SD dir_z) | 95% CI | CR2 df | CR2 p (1-sided) | wild p (1-sided) | N |
|---|---|---|---|---|---|---|
| (i) PRIMARY | −0.001 | [−0.49, +0.48] | 3.5 | .501 | .517 | 741 |
| (ii) REQUIRED, Rogan excluded | +0.223 | [−0.17, +0.61] | 3.0 | .084 | .096 | 384 |
| (iii) weights capped 20% | +0.056 | [−0.38, +0.49] | 3.8 | .365 | .359 | 741 |
| (iv) show-level permutation | −0.007 | — | — | .519 | — | 10 shows |
| (v) + INTERVIEW_DOMINANT | +0.060 | [−0.33, +0.45] | 3.8 | .344 | .341 | 741 |
| (v) + INTERVIEW_DOMINANT, no Rogan | +0.219 | [−0.11, +0.55] | 2.9 | .061 | .037 | 384 |
| (vii) + position format score | −0.011 | [−0.39, +0.37] | 3.3 | .532 | .569 | 741 |
| (viii) declarative-you exposure | −0.020 | [−0.48, +0.44] | 3.7 | .546 | .535 | 741 |
| (viii) declarative-you, no Rogan | +0.201 | [−0.19, +0.60] | 3.0 | .101 | .114 | 384 |
| secondary Q33E (i)/(ii) | +0.076 / +0.105 | | | .215 / .140 | .179 / .132 | |
| H2 exclusivity (i)/(ii) | +0.041 / +0.033 | | | .123 / .243 | .167 / .240 | |
Survey-weighted: b=+0.007. Blaze-only dropped: +0.003. Daily-Wire-only dropped: −0.084. Rogan coded right: +0.013.
Leave-one-show-out: every estimate within [−0.15, +0.22]; only "drop Rogan" moves it (+0.22, p=.08).
(vi) Discovery sample with multi-host indicator: coefficient unchanged (+0.219 → +0.219).

## Per-show (reported regardless)
| show | n | dir_z | audience lean (R−D) | mean cyn8 | excl |
|---|---|---|---|---|---|
| Bongino | 11 | +2.35 | +1.00 | +0.04 | .45 |
| Beck | 55 | +1.26 | +0.85 | +0.14 | .45 |
| Rogan | 360 | +1.21 | +0.60 | −0.16 | .54 |
| Ryan | 11 | +0.67 | +0.91 | +0.39 | .55 |
| Kirk | 37 | +0.45 | +0.92 | +0.23 | .54 |
| Tucker | 40 | +0.19 | +0.72 | +0.12 | .55 |
| Parnas | 37 | −0.15 | −0.86 | +1.01 | .51 |
| The Daily | 72 | −0.29 | −0.14 | −0.58 | .26 |
| Kelly | 14 | −0.50 | +0.86 | −0.28 | .29 |
| Shapiro | 112 | −0.52 | +0.94 | −0.26 | .51 |
Show-level corr(dir_z, audience lean) = +0.37 (n=10). Prereg prediction "Rogan highest" was wrong (Bongino highest; Rogan 3rd).

## Reading (per the frozen asymmetry clause)
The association does NOT extend to the largest shows in the ecosystem. Reported as a result, not a caveat. Ambiguous between "no association" and "the largest multi-category shows are a different population"; the paper says a within-frame confirmation requires Kettering Y2.
Two honest observations, neither of which rescues H1: (a) without Rogan the point estimate (+0.22) equals the discovery estimate (+0.22) but with 9 clusters the CI spans zero; (b) Rogan's audience — 48% of the sample — carries high measured address (+1.21, plausibly guest-inflated) but average-to-low cynicism (−0.16), the exact pivot pre-registered in Amendment 1; the format sensitivities did not recover the effect.
Paper consequence: §F.3 (register → cynicism) is exploratory-only (Y1 discovery, ~8 effective clusters) with a FAILED out-of-frame confirmation. It cannot be a headline claim. Levels 1 (measurement/validation) and 2 (population finding: non-institutional diets → cynicism, with Pew lagged replication) are unaffected.
Also notable for the label check: two of the seven right shows (Shapiro −0.52, Kelly −0.50) score BELOW both left shows on address — the partisan register contrast is weaker out-of-frame than in the Politics-chart corpus.

## POST-HOC (not pre-registered; run after the verdict, at the PI's request; label as exploratory in the paper)
Political-content density from the same transcripts (share of 750-char passages with ≥3 political terms; corpus median .32, 10th pct .15):
Parnas .47 · Bongino .29 · Shapiro .22 · Daily .18 · Kirk .18 · Tucker .17 · Beck .15 · Kelly .07 · **Ryan .05 · Rogan .03** (Rogan and Ryan = corpus 2nd percentile). File: handoff/scans/holdout_political_density.csv
| post-hoc model | b | 95% CI | CR2 df | CR2 p1 | wild p1 | G / N |
|---|---|---|---|---|---|---|
| drop Rogan + Ryan (8 shows) | +0.219 | [−0.16, +0.60] | 2.9 | .082 | .083 | 8 / 373 |
| political-content shows only (7; cutoff = corpus 10th pct) | +0.217 | [−0.30, +0.74] | 2.5 | .125 | .217 | 7 / 359 |
| … + INTERVIEW_DOMINANT | +0.221 | [−0.31, +0.76] | 2.3 | .119 | .100 | 7 / 359 |
| all 10 + political-share covariate | +0.203 | [−0.23, +0.64] | 4.1 | .134 | .191 | 10 / 741 |
| 7 shows, secondary Q33E / H2 | +0.118 / +0.016 | | | .168 / .403 | | |
Reading: every specification that removes the non-political shows lands on the discovery magnitude (+0.22 vs +0.219 in discovery) but with 7–8 clusters (df ≈ 3) the smallest detectable one-sided effect is ≈ 0.29, so none reaches .05. Likelihood ratio for the 8-show estimate, discovery effect (0.22) vs zero ≈ 5:1. Status: consistent-but-unconfirmed. Does not change the pre-registered verdict.

## Audience profiles (114 items, controls as in pop_profile; BH within audience) — handoff/scans/holdout_audience_profile.csv
Items distinguishing each audience from the rest of Kettering (q<.05): Parnas 50 · Shapiro 36 · Beck 33 · Kirk 24 · Ryan 23 · **Rogan 11** (with the largest n=360) · Tucker 6 · Daily 5 · Kelly 3 · Bongino 3.
Rogan's audience: young (43 vs 49), 77% male, 71% R/lean-R, lower education, heavier platform use (5.6 vs 4.4; more X and YouTube, less Bluesky), only 46% name a mainstream source (vs 66%); after controls NOT cynical (cyn8 +0.09 ns), NOT strong partisans (−0.01), average civic participation, slightly less confidence in election administration (−0.15). It is the least distinctive audience of the ten — a general (young, male) audience, not a political one.
Adjusted cynicism by audience: Parnas +0.81* · Kirk +0.63* · Ryan +0.56* · Tucker +0.37 · Beck +0.33* · Bongino +0.21 · Rogan +0.09 · Shapiro 0.00 · Kelly −0.01 · Daily −0.25. Within the six right political shows, address and audience cynicism rank-correlate ≈ +0.4 (Shapiro and Kelly lowest on both — the pre-specified pivot).
