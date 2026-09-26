# Year 2 confirmatory test — RESULT (run 2026-09-23T17:10Z, prereg_y2.md frozen 17:09Z, hashes in prereg_y2.sha256)
Log: handoff/results/y2_RUN_2026-09-23T1710Z.log | table: step7_audience/outputs/y2_test_results.csv | script: step7_audience/5_prereg/y2_test.py
Survey: Kettering-Gallup Democracy for All Year 2, n = 23,683, fielded 16 Apr–21 May 2026, fresh sample. Q35 and Q20 items absent:
cynicism is the 4-item composite (Q31, Q33E, Q33H, Q34B); no platform-use control.
Primary frame: 722 corpus listeners, 26 shows (MeidasTouch 192, Candace 79, Letters from an American 70, Pod Save America 63,
Bulwark 51 ...). Satterthwaite df ≈ 1.8: one cluster holds 27% of listeners. Out-of-frame: 823 listeners, 10 shows.

## VERDICT: H1 FAIL. H2 FAIL. H3 FAIL.
| spec | election distrust (primary) | cynicism (4-item) | education |
|---|---|---|---|
| (i) address | +0.10 (1-sided CR2 .077 / wild .161) | −0.00 (n.s.) | −0.16 (.14 / .16 two-sided) |
| (ii) address ǀ intensity | +0.06 (.195 / .267) | −0.00 | −0.14 (n.s.) |
| (iii) intensity | **+0.26 (.013 / .006)** | +0.01 (n.s.) | −0.25 (.09 / .04 two-sided) |
Robustness: outlet strings dropped, Clay & Buck included, survey-weighted: address on election distrust +0.10 to +0.11, one-sided
CR2 .05–.08, wild .13–.18 (never both < .05). MeidasTouch dropped: +0.08 (.22/.26). LOSO: (i) elec b in [+0.03, +0.12],
p < .05 in 2/26 drops. Pooled with out-of-frame (1,443 / 36): address +0.10 (.068/.136); intensity-controlled +0.08 (n.s.);
Rogan excluded, address ǀ intensity +0.11 (.047/.090).

## Pre-specified pooled Y1 + Y2 (1,257 listeners, 58 shows, year indicator)
| exposure | election distrust | cynicism (4-item) | education |
|---|---|---|---|
| address alone | +0.13 (1-sided .014 / .064) | +0.03 (n.s.) | −0.18 (n.s.) |
| address ǀ intensity | +0.07 (.143 / .234) | −0.01 | −0.13 (n.s.) |
| intensity alone | **+0.28 (.006 / .002)** | **+0.14 (.017 / .043)** | **−0.27 (.007 / .003)** |
| intensity ǀ address | +0.28 (.006 / .003) | +0.14 (.017 / .034) | −0.27 (.007 / .005) |

## Reading
Address did not replicate as a predictor of election distrust on fresh respondents at the pre-registered criterion, though the
point estimate is positive and about half the Year 1 size (+0.10 vs +0.18/+0.23), and reaches one-sided CR2 p .05–.08 in every
specification. Address did not predict cynicism (4-item) or education in Year 2. Ideological intensity replicated on election
distrust at full strength (+0.26 vs +0.28) and dominates in the pooled data on all three outcomes; address adds nothing net of
intensity. The Year 2 exposure contrast is carried by two large left-leaning clusters at opposite ends of the address scale
(MeidasTouch +1.47, n=192; Letters from an American −2.90, n=70), whose audiences differ little in election distrust.
Consequence for the paper: Finding 1 (register differs by side; address is a stable trait distinct from intensity) is untouched.
The audience finding must be stated as: one-sided shows have more election-distrustful, less-educated audiences (replicated in
Year 2); address-heavy shows share that audience because they are one-sided, and address shows no independent association.

## ADDENDUM (same day, after the verdict; reported regardless, does not change the frozen verdict)
Correction: the LOSO lines for education in y2_RUN_2026-09-23T1710Z.log used the positive-direction one-sided p; the
hypothesis is negative. Corrected LOSO, address alone on education: b in [−0.36, −0.09], one-sided CR2 p < .05 in 1/26 drops.

Outcome coding checked against party in both years (D/R gaps consistent: elec D −0.16/−0.21, R +0.09/+0.20; cyn4 D +0.02/0.00,
R −0.29/−0.26). Conventional CR1 inference on the primary election-distrust model gives one-sided p = .001 (CR2 .077): the
frozen CR2/wild criterion is much more conservative at df 1.8.

The Year 2 address contrast is carried by two large left-leaning clusters at the ends of the scale: MeidasTouch (n 192,
address +1.47) and Letters from an American (n 70, address −2.90, matched by the host-name string "heather cox richardson",
whose readers may follow the newsletter rather than the podcast).
| election distrust | address alone | address ǀ intensity |
|---|---|---|
| primary frame | +0.10 (.077/.156) | +0.06 (n.s.) |
| drop Letters | +0.03 (n.s.) | −0.04 |
| Y1-comparable match (title + Y1 patterns only, 18 shows / 499) | +0.04 (n.s.) | −0.10 |
| left-leaning shows only (16 / 493), exploratory | +0.12 (.058/.142) | +0.07 (.038/.030); drop Letters +0.09 (.11/.14) |
=> For election distrust, Year 2 is a null for address, not a weak positive; the primary-frame point estimate exists only
because of the Letters cluster.
| education (one-sided negative) | address alone | address ǀ intensity |
|---|---|---|
| primary frame | −0.16 (.143/.169) | −0.14 (.155/.117) |
| drop Letters | −0.36 (.001/.000) | −0.34 (.001/.001) |
| Y1-comparable match | −0.28 (.014/.022) | −0.27 (.021/.015) |
=> The education-composition association replicates in direction and size (Y1 −0.23) and survives intensity, EXCEPT when the
Letters cluster is included: its readers are highly educated but sit below the trend the other 25 shows draw. H3 fails as
frozen because the frozen match included that cluster; the paper reports both facts and does not re-run the verdict.

## Verification of mechanics (23 Sept, after the addendum)
Scales identical across years (EDU 1–8, Q19 1–5, Q33E/Q32C 1–5; Y2 means within 0.15 of Y1). No primary row lacks an
exposure. Letters from an American respondents wrote "heather cox richardson" (68) or the title (2). Independent statsmodels
re-estimate reproduces the primary coefficients exactly (election distrust +0.0971, cluster se .027; education −0.160).
Show-level n-weighted correlation of address with mean listener education: −0.60 (26 shows); −0.62 without Letters; −0.56
without MeidasTouch; −0.68 without both. Respondent-level education slope: −0.36 without Letters, −0.09 without MeidasTouch,
−0.38 without both: the two large clusters lever the slope in opposite directions, but the sign never reverses and the
show-level association is stable. The frozen H3 failure reflects df ≈ 1.8 plus that leverage, not an absent association.

## Cynicism measure: correction and like-for-like comparison (see prereg_y2_ERRATA.md)
Year 2's Q34B is Year 1's Q35B ("Government is sensitive to the interests of people like me"), not Year 1's Q34B. The Year 2
secondary composite as run = Q31 + Q33E + Q33H + Q35B-wording. In Year 1 this composite correlates .91 with the 8-item scale
(inter-item mean r .36). Year 1 results on it, same controls as the Y2 run (535 listeners):
| Year 1 composite | address | intensity | address ǀ intensity |
|---|---|---|---|
| 8-item (paper) | +0.17 (.02/.02) | +0.22 (.00/.01) | +0.08 (n.s.) |
| Y2-equivalent 4-item | +0.14 (.10/.17) | +0.22 (.00/.01) | +0.03 (n.s.) |
| discovery 4-item | +0.18 (.01/.01) | +0.20 (.00/.01) | +0.11 (.08/.07) |
Year 2 on the Y2-equivalent composite: address 0.00, intensity +0.01. So the Year 2 cynicism null is not specific to address:
intensity, which replicated on election distrust, shows nothing on cynicism in Year 2 either. The cynicism secondary is
uninformative about address in Year 2; the election-distrust primary and the education tertiary carry the test.

## Pooled Year 1 + Year 2 (pre-specified secondary; corrected one-sided directions; year indicator; cluster = show)
| | election distrust | education (one-sided negative) |
|---|---|---|
| **Y2 as frozen (1,257 / 58 shows)** | | |
| address alone | +0.13 (.014 / .064; df 2.2) | −0.18 (.075 / .106) |
| address ǀ intensity | +0.07 (.143 / .229) | −0.13 (.090 / .054) |
| intensity alone | +0.28 (.006 / .001; df 8.5) | −0.27 (.007 / .003) |
| **Y2 matched the Year 1 way, no host-name additions (1,036 / 55 shows)** | | |
| address alone | +0.16 (.007 / .015; df 4.6) | −0.25 (.008 / .002) |
| address ǀ intensity | +0.03 (.331 / .310) | −0.17 (.039 / .041) |
| intensity alone | +0.26 (.007 / .003; df 7.3) | −0.26 (.006 / .001) |
Reading: pooled, address alone passes on both outcomes; net of intensity it keeps education (−0.17, both p ≈ .04) and
loses election distrust (+0.03). Intensity keeps both regardless. The Richardson cluster (frozen Y2 match) lowers df from
4.6 to 2.2 and weakens both address estimates.

## Exploratory item scan, pooled Y1+Y2 (37 identically worded items; address ǀ intensity and intensity ǀ address; BH)
File: handoff/scans/y2_pooled_item_scan.csv. No item reaches BH q < .10 for address net of intensity (best raw p: household
income −0.07, p .053; Q31 leaders held accountable +0.07, p .06). Election items separately: address −0.05 / +0.05 (n.s.);
intensity −0.21 (Q33E, p .02/.01) and +0.23 (Q32C, p .02/.00). Education is the only outcome on which address keeps an
association net of intensity (see pooled table above); income points the same way.

## Composition as a pathway (pooled, Y2 matched the Year 1 way, 1,036 / 55)
Education predicts election distrust in both full surveys (−0.16 and −0.15 SD per SD, party/age/attention controlled) and
among listeners (−0.20). Address -> election distrust: with the education control +0.16, without it +0.20 (address alone);
net of intensity +0.03 with, +0.05 without. About 0.03–0.04 SD of the bivariate address association runs through the
less-educated audience address draws; the remainder is one-sidedness. The compositional path is real but small.

## Out-of-sample check of the Year 1 item scan, and Year 2's new civic items (23 Sept)
Year 1 discovery scan (outcome_scan.csv, 385 listeners, dir_z): BH q < .05 for Q21 social media time (+0.28, strongest item),
Q33E, Q32C, Q34B, Q35E, Q12D, Q20K, Q20F, Q35C. Of these, Q21 and Q12D exist unchanged in Year 2. In Year 2 (both matches),
address net of intensity: Q21 +0.04 / +0.03 (n.s.; intensity +0.24 / +0.19), Q12D ≈ 0. Twenty-seven Year 2 items (barriers to
participation, civic participation, local trust/power, voting-intimidation fear, life ladder, income feeling): none reaches
BH q < .70 for address net of intensity on either match; strongest raw p .05–.06 (Q71 local political views −0.20; Q53B joined
a protest +0.16; Q66C fear of confrontation while voting +0.10), all unreplicated across the two matches.
Behavioral traces (pooled, Y1-style): named the show first +0.06 (.13/.03) net of intensity; sole source, number of sources,
loneliness, overload ≈ 0; exclusivity belongs to intensity (+0.16).
Conclusion: across both surveys, the only listener characteristics address predicts net of one-sidedness are compositional
(education, robustly; income, marginally) and, weakly, salience (naming the show first). No attitude item does.

## Full item scans (23 Sept; handoff/scans/address_full_item_scan_y1_y2.csv)
Every numeric survey item, address (mfte_z) alone and net of intensity, controls as in the paper, CR2 p, BH within scan.
Year 1, 535 listeners / 54 shows, 119 items: 0 items at BH q < .10, alone or net (best raw: Q32C +0.24 p .005; Q21 social
media +0.15 p .009). Year 2 Y1-style match, 501 / 18, 105 items: 0 (best raw p .014). Year 2 frozen, 722 / 26, 105 items: 0
(best raw p .041). No item's raw top-10 appears in more than one scan except Q31 and Q32C, and neither survives correction.

## Composition / behaviour scan and pooled subgroup check (23 Sept, final)
Pooled Y1+Y2 (1,036 / 55), outcomes = age, gender, race, LGBT, income, employment, registration, veteran, church, born-again,
rural, community events, social media time, attention; Year 1 adds platform use (8 platforms), free time, loneliness, overload.
Address net of intensity: no outcome at BH q < .27; raw p < .10 only for white (−0.05), employed (−0.06), and in Year 1
LGBT (−0.06, p .017), born-again (+0.05, p .022), free time (+0.16, p .057), YouTube/TikTok (n.s. net; significant alone).
Social media time: address alone +0.15 (p .017) both years, net of intensity +0.07 (n.s.); intensity carries it (+0.19).
Pooled within left-leaning shows / Democrats, address net of intensity on election distrust: +0.03 to +0.05, n.s.; the
Year 2 within-left signal (+0.07) does not survive pooling with the Year 1-style match. Components: neither.

## Confidence in institutions: the ten-item Q33 block (identical wording both years; higher = less confidence)
Pooled (1,257 / 58): address +0.02 (n.s.); address ǀ one-sidedness −0.01; one-sidedness ǀ address +0.11 (.16/.20).
Year 1: address +0.10 (.10/.18), net of one-sidedness +0.00; one-sidedness +0.18 (.01/.01), press freedom +0.20, equal
treatment +0.13, Supreme Court +0.17. Year 2: everything ≈ 0 for both (one-sidedness −0.04). No single institution item
carries an address association in either year, net or alone. Same pattern as the cynicism scale: address never, one-sidedness
in Year 1 only. Election distrust remains the only attitude that replicates across years, and it belongs to one-sidedness.

## Final full-scale search for anything address predicts beyond education (23 Sept)
Block composites (satisfaction, barriers, identity, trust in leaders, democracy support, efficacy, Q36 attitudes, platform
count, participation, voting fears, Christian nationalism, local trust), affective polarization (Y1 Q39A/B: in-party,
out-party, gap, absolute gap), partisan strength, listener-show congruence, free-text register in Q28 (you-rate, we-rate,
imperatives, negative words, length): 31 outcomes; 0 at BH q < .10 for address alone or net of one-sidedness. Raw hints,
Year 1 only and unreplicable: absolute affective gap +0.19 alone (p .02), +0.13 net (p .07); in-party favourability +0.18
(p .06); Year 2 Christian nationalism +0.06 (p .06). Nonlinear: listeners of bottom-quartile-address shows are less
election-distrustful (pooled −0.35, p .03/.08 net of one-sidedness) but the quartile is 66 Letters from an American
respondents plus Tangle; without Letters −0.34 (p .11/.13), Year 1 alone −0.35 (p .25). Top quartile: nothing.
Verdict of the search: across every item, composite, subgroup, component, nonlinear form and free-text feature in two survey
years, address predicts education (and marginally income) and nothing else that one-sidedness does not predict better.

## Maximal consistent match, both years (24 Sept; data/external/kettering/y1_match_maximal.csv, y2/y2_match_maximal.csv)
Rule, applied identically to both years: exact corpus title, reviewed title variants/misspellings, unambiguous host full
names (hosts listed under one corpus show only), and the Year 1 within-frame patterns. Year 1's original analysis sample had
EXCLUDED host-name matches (the 544 "expanded" set was sensitivity-only); the maximal match adds 146 Year 1 listeners (Benny
Show 34, Candace 23, MeidasTouch 21, Timcast 17, Brian Tyler Cohen 13 ...). "heather cox richardson" is ambiguous in the host
table (Letters from an American and Politics Chat) and is excluded by the rule; shown separately.
| pooled sample | listeners / shows | address -> election distrust (alone; ǀ one-sided) | address -> education (alone; ǀ one-sided) | one-sided ǀ address (elec; edu) |
|---|---|---|---|---|
| maximal, all shows | 1,247 / 64 | +0.12 (.09/.13); +0.02 (n.s.) | **−0.25 (.00/.00); −0.20 (.00/.00)**, df 5.5–6.9 | +0.29 (.00/.00); −0.28 (.01/.00) |
| maximal, ≥3 listeners | 1,201 / 33 | +0.13 (.08/.15); +0.03 | −0.27 (.01/.00); −0.22 (.01/.01) | +0.29; −0.28 |
| + Richardson readers -> Letters (117 at address −2.9) | 1,345 / 64 | +0.12 (.03/.20); +0.07 | −0.16 (.15/.32); −0.12 (n.s.), df 1.9 | +0.32; −0.30 |
Year 1 alone (maximal, ≥3): address -> education −0.19 (.03/.04), ǀ one-sided −0.10; Year 2 alone: −0.35 (.00/.00), ǀ −0.34 (.00/.00).
Reading: with a consistent match across years, address predicts a less-educated audience net of one-sidedness at p ≤ .01 on
both tests across 64 shows; the single decision that removes it is folding 117 newsletter readers into the lowest-address
show. Election distrust is unchanged: one-sidedness +0.29 both years; address ≈ 0 net.

## Verified outcome scan on the pooled maximal match (24 Sept; handoff/scans/scan_maximal_match_verified.csv)
1,247 listeners / 64 shows; 45 outcomes (identically worded items, election-distrust and confidence composites, education,
age, gender, race, income, partisan strength); controls party, age, attention, education, show lean, year; CR2 p; BH.
ADDRESS net of one-sidedness: 0 outcomes at BH q < .10. Education is the only outcome close (−0.20, p .005, q .20); the next
best is raw p .09. ONE-SIDEDNESS net of address: 10 outcomes at q < .10, replicating in both years for the core set:
| outcome | pooled | Year 1 | Year 2 |
|---|---|---|---|
| life-ladder satisfaction (Q1) | −0.25 (q .001) | −0.22 (.004) | −0.29 (.000) |
| officials acted improperly when results surprise (Q32C) | +0.26 (q .006) | +0.33 (.000) | +0.20 (.021) |
| financial strain (Q3, higher = worse) | +0.28 (q .006) | +0.30 (.001) | +0.26 (.056) |
| election distrust composite | +0.29 (q .006) | +0.41 (.000) | +0.17 (.072) |
| education | −0.28 (q .07) | −0.32 (.005) | −0.23 (.079) |
| expected ladder in 5 years (Q2) | −0.16 (q .07) | −0.08 (n.s.) | −0.24 (.012) |
| elections administered well (Q33E, reversed sign) | −0.18 (q .07) | −0.30 (.000) | −0.07 (n.s.) |
| white | −0.07 (q .07) | −0.05 | −0.09 (.004) |
| social media hours (Q21) | +0.18 (q .07) | +0.19 (.001) | +0.15 (.14) |
| "people like me are valued" (Q22) | −0.18 (q .07) | −0.09 | −0.25 (.003) |
| age | +0.34 (q .12) | +0.37 (.019) | +0.32 (.052) |
Reading: one-sided shows draw an audience that is older, less educated, more financially strained, less satisfied with life
now and in prospect, more distrustful of elections, and heavier on social media; the core of this (life satisfaction,
financial strain, election items, education) appears in both years. Address adds education only.

## Cynicism on the maximal match (24 Sept)
Address: 8-item scale (Y1) +0.11 (.11/.17) alone, +0.06 net of one-sidedness; 4-item both years pooled +0.03 alone, −0.01
net; 10-institution confidence +0.01 / −0.02; no item significant in either year in the cynical direction. The one item
with any address signal points the other way: in Year 2, address-heavy audiences are MORE likely to trust that leaders will
be held accountable (Q31: −0.14 on the distrust coding, p .02/.00; net of one-sidedness −0.12, .08/.05; Year 1 0.00).
One-sidedness: Y1 8-item +0.19 (.02/.03), 4-item +0.22, confidence +0.19, press freedom +0.25; Year 2 all ≈ 0.
Conclusion: address does not predict cynicism or institutional confidence in any form, in any year, on any match.

## Final per-year scan on the maximal match (24 Sept; handoff/scans/address_final_scan_maximal_by_year.csv)
Year 1: 572 / 52, 123 outcomes; Year 2: 675 / 49, 109 outcomes. Address alone and net of one-sidedness, BH within year.
Survivors at q < .10: none in any cell. Only outcome ranked in the top six in both years: education (Y2 −0.30 net, p .003,
q .31; Y1 −0.09 net, n.s. on this match). Year-1-only leaders alone are the election items (carried by one-sidedness); no
raw hit repeats across years except education. Search closed.

## One-sidedness, per-year full scans on the maximal match (24 Sept; handoff/scans/onesidedness_final_scan_maximal_by_year.csv)
Net of address, BH within year. Year 1 (572 / 52, 123 items): 14 survivors — election distrust composite +0.41, Q33E −0.30,
Q32C +0.33, social media time +0.19, financial strain +0.30, food satisfaction −0.22, life ladder −0.22, Q15C −0.23,
education −0.32, Q32B +0.43, Q38 −0.18, criminal justice confidence −0.17, Q35C −0.14, trust in religious leaders −0.13.
Year 2 (675 / 49, 109 items): 1 survivor — life ladder −0.29 (q .01); election distrust +0.17 (p .07), financial strain
+0.26 (p .06), education −0.23 (p .08) same direction but below the within-year threshold. Pooled (previous section):
10 survivors. The replicated core across years is life satisfaction, financial strain, election distrust, education, age.

## Three further angles for address (24 Sept, maximal match)
1. Discovery measure (dir_z, r .93 with mfte_z): same pattern as the measure of record — education net of one-sidedness
   −0.23 pooled (Y1 −0.13, Y2 −0.33); election distrust +0.02 net; life ladder −0.11 net in Y2 only (p .10/.08).
2. Components net of one-sidedness (pooled): commands carry education (−0.18, .01/.01); second person carries nothing;
   neither predicts election distrust, life ladder, financial strain or social media time.
3. Moderation (9 interactions): address × listener education -> life ladder −0.13 pooled (.01/.01), same sign both years
   but neither alone significant (Y1 −0.14, .10/.07; Y2 −0.12, .55/.08). Simple slopes: among the least-educated tercile
   of listeners, address predicts HIGHER life satisfaction (+0.29, .01/.01); mid −0.01; high −0.10 (n.s.). Address ×
   attention -> election distrust +0.11 pooled (.08/.01) but Y1 n.s. and simple slopes flat: not supported. The
   education-moderated life-satisfaction pattern is exploratory (would be q ≈ .09 across the 9 tests) and is a candidate
   for Year 3 pre-registration, not a finding.

## Address -> institutional outcomes within listener-education terciles (24 Sept, pooled maximal)
13 institutional outcomes × 3 terciles + interactions: 0 low-education slopes and 0 interactions at BH q < .10. One-sidedness
-> election distrust holds in every tercile (low +0.23, mid +0.19, high +0.44). The address × education moderation is
specific to life satisfaction and does not extend to institutions.

## Outrage on the pooled maximal match (24 Sept; handoff/scans/outrage_scan_maximal_match.csv)
1,247 / 64; r(outrage, one-sidedness) .69, r(outrage, address) .49 across these shows. Outrage ALONE: 7 of 45 outcomes at
q < .10 — election distrust +0.27, officials improper +0.26, education −0.40, "people like me valued" −0.26, local-community
identity −0.13, life ladder −0.27, future ladder −0.20; all identical net of address (address adds nothing to outrage).
Net of one-sidedness: 0 survive; one-sidedness net of outrage: 3 (age, financial strain, division-of-power confidence).
Head to head, outrage's residual is community identity (−0.28), trust in religious leaders (−0.25), male (+0.20); one-sidedness'
residual is financial strain (+0.33), officials improper (+0.24), age. Types: exaggeration and extremizing predict election
distrust (+0.24, +0.21), lower education, lower life satisfaction and financial strain; mockery predicts nothing.
Reading: outrage and one-sidedness are one audience-facing property measured two ways; one-sidedness is the marginally
stronger predictor. Address is orthogonal to both at the audience level.

## Survey side, remaining angles for address (24 Sept)
Diet composition (other named sources classified with diet.py: person / institution / platform / podcast; pooled maximal
1,247 / 64): address predicts none of nine diet outcomes (all |b| < .07, q .98), in either year; one-sidedness neither.
Host-naming (listener wrote the host's name rather than the title; 12 shows with both forms, 294 listeners): address
−0.24 (p .10/.18); show-level r .23, driven by person-branded titles (Candace, Benny). Within-show host-namers vs
title-namers: no difference on election distrust.
Full pool incl. out-of-frame shows both years (2,693 listeners / 74 shows, 42 outcomes): address net of one-sidedness 0 at
q<.10; top items age +0.22 (p .02), education −0.15 (p .05/.07), income −0.06 (p .07). One-sidedness net of address holds:
election distrust +0.24, education −0.24, life ladder −0.24, financial strain +0.25 (all p ≤ .03).

## Show-level outcomes for address (24 Sept; data/output/show_level_outcomes.csv) — the first non-compositional results
194 shows; ad-free address (parser components on non-ad passages, r .98 with MFTE); controls one-sidedness, side, solo,
log episodes/week, show age; HC1. Mean chart rank when charted −0.17 SD (p .04; −0.21 dropping the top 10; LOSO
[−0.21, −0.15]; same sign both sides; survives outrage control); chart persistence +0.13 (p .08); rank trajectory −0.13/month
(p .07, censored). Ad-marked share of passages +0.26 (p < .001; ad passages carry 3.5× the commands and 1.8× the "you" of
ordinary talk, but are 1.8% of passages). Review count (log, 128 shows, capped at 500) +0.31 (p .014; left-driven +0.38,
right +0.01; survives ad-share control +0.28). With ad share controlled the chart coefficients fall to −0.11 / −0.08 / +0.09
(n.s.): chart position is shared with commercial load. Review coding pilot (300): direct address to host 35%, emotion 33%,
loyalty 11%, companionship 5%, connection 2%, hostility 25%; full collection relaunched.

## Register in general (24 Sept; step7_audience/inputs/mfte_show_features_all.csv, mfte_show_register_dims.csv)
All 150 MFTE count features aggregated to the 194 shows (44,563 episodes). Biber dimensions built from published loadings
(D1 involved/informational from 24+7 features; D2 narrative; D4 persuasion; D5 abstract, approximated) and a register PCA
(PC1 22% of variance = informational vs involved, r −0.93 with D1). Address sits on the involved pole: r(address, D1) +0.56,
r(address, D5 abstract) −0.58, r(address, PC1) −0.71. But the general register tracks ideology less than address: D1 r .22
with DIME (address .41); R−L gap D1 +0.24 SD (address +0.66). Single features most tied to DIME: conditionals (COND .41),
"have got" (.40), PP2 (.37), VIMP (.35), −VBG, −prepositions, URLs read aloud (.30), prediction modals.
Audience (pooled maximal, 1,200 / 63): no Biber dimension or PC predicts any of election distrust, education, life ladder,
financial strain net of one-sidedness (0 of 36 at BH q<.10); address -> education −0.21 net remains the only signal.
Leave-one-show-out ridge with all 150 features: out-of-sample R2 falls for attitudes (election distrust .218 with
one-sidedness vs .112 with 150 features), rises marginally for education (.087 vs .067 with address alone). Register in
general does not predict audience attitudes; address is the part of it that predicts audience composition.

## Mechanism of address -> education (24 Sept, pooled maximal 1,200 / 63, address net of one-sidedness; baseline −0.212)
Candidate mediators, each added as a show-level covariate: none absorbs the association (range −0.186 to −0.212; all
together −0.206). Lexical complexity (Flesch-type; r −0.67 with address but +0.02 with audience education), type-token
ratio, subordination, Biber D1/D5, conversational topic share, ad share, solo format, episode length/frequency, first
person + hedges, URLs read aloud. Components: commands (VIMP) carry it (−0.186), second-person pronouns do not (−0.078).
Imperative subtypes (from the parsed passages): attention imperatives (look, listen) −0.234 (.00/.00); action imperatives
(go, vote, call, sign up) −0.244 (.00/.00); epistemic imperatives (think, remember, imagine) −0.037 n.s.; deontic "you
should/need to" −0.155 (.02/.01); "you know" filler +0.219 (.05/.04), i.e. the OPPOSITE direction. All survive ad share
and URL controls (attention −0.218, action −0.254, deontic −0.122, "you know" +0.194, ad-free address −0.170). By year:
Y1 attention −0.220 (.00/.01), action −0.164 (.02/.03), "you know" +0.266 (.02/.07); Y2 attention −0.254 (.00/.01),
action −0.327 (.00/.01), deontic −0.220 (.02/.01), "you know" +0.181 n.s. Subtypes are collinear at show level
(attention–action r .69; action r .46 with ad share, .51 with URLs) so no single one dominates head-to-head.
Talk-radio origin (publisher/description regex; 29 commercial-radio shows, mean address +0.48 vs −0.08): commercial-radio
origin raises address by +0.51 SD net of side and one-sidedness, but the education link is unchanged with radio origin
controlled (−0.212 -> −0.212), radio origin itself predicts nothing (−0.011), and the link is identical within non-radio
shows (−0.209, .00/.01). Reading: the education link is carried by the discourse-directing and action-directing command
register of the host (a host telling listeners where to look and what to do), not by simpler language, involvement,
format, advertising or radio genre. Epistemic "think about it" commands do not carry it.

## What follows from the education result (24 Sept; handoff/scans/address_edu_indirect_paths.csv, address_by_education_moderation.csv)
(A) Indirect paths. Education predicts life satisfaction (+0.13), income difficulty (−0.16), community events (+0.18),
election distrust (−0.13), church attendance, in this sample. But address moves education by only −0.21 SD, so every
indirect path address -> education -> outcome is ≤0.04 SD, and address's total association with all 43 pooled items is
null (0 at BH q<.10; also 0 net of education). Address audiences are less educated but do not show the attitudes that
lower education usually brings.
(B) Moderation by listener education (simple slopes, net of one-sidedness). Life ladder: HS-or-less +0.281 (.01/.01,
n 258/33 shows), some college +0.04, BA+ −0.115 (.04/.01); interaction −0.121 (.01/.01). Y1: HS-or-less +0.351 (.01/.01),
interaction −0.096 (.11/.09); Y2: HS-or-less +0.214 (.13/.07), BA+ −0.211 (.04/.02), interaction −0.149 (.02/.01).
Survives household income, income difficulty, social-media time, registration and not-employed controls (+0.28 to +0.30);
LOSO within HS-or-less b in [+0.22, +0.32], CR2 p<.05 in 32/33 drops. One-sidedness runs the other way among HS-or-less
(−0.222, .05/.04). Caveat: 53 interactions tested, none reaches BH q<.10, and this one was seen in an earlier exploratory
pass; it is a candidate, not a finding. Low-education listeners of high-address shows are older (mean 60 vs 45) and more
often not employed (64% vs 36%), but age is controlled and employment does not change the estimate. Nothing else
interacts: attention, sources named, registration, participation (Y1), trust items, composites all flat by group.
(C) Reach. Low-education listeners of corpus shows are far MORE engaged than low-education non-listeners (attention 4.4 vs
3.3 of 5; registered 94–96% vs 80%; 2.9 vs 2.1–2.3 sources named), and within them address predicts none of attention,
sources, mainstream-outlet naming, registration, community events or civic-power beliefs. Address-heavy shows do not pull
in disengaged people.
(D) Audience composition (the plain statement of the education result). Political-podcast listeners are more educated than
the public (BA+ 39% vs 31–34%). By show-address tercile, pooled: low-address shows HS-or-less 12% / BA+ 58%; mid 29% / 24%;
high-address 25% / 33%; the public is 30–34% / 31–34%. Same within left-leaning shows (lower-address half 12% / 58% ->
higher half 24% / 34%) and right-leaning shows (28% / 24% -> 35% / 20%). Net of one-sidedness: address +6.2 points
P(HS or less) per SD pooled (.01/.01; Y1 +2.2 n.s., Y2 +10.1 .02/.01), −9.2 points P(BA+) (.03/.03; Y1 −5.2 n.s., Y2
−13.4 .02/.03). Show level (27 shows with n>=5): address b +0.073 share HS-or-less per SD (p .000), one-sidedness n.s.
Reading: the address-heavy shows are the ones whose audiences look like the general public on education; low-address
shows reach the college-educated. Address is the style of political podcasting that reaches beyond the degree-holding
audience; it does not come with worse attitudes, and among the least-educated listeners it comes with, if anything,
higher life satisfaction.

## Command types by a published classification (24 Sept; step6_register/mfte_imperative_classes.py; step7_audience/inputs/mfte_imperative_show_scores.csv; log handoff/results/imperative_classes_audience.log)
Replaces the hand-sorted attention/action/epistemic verb groups. MFTE assigns Biber's (2006) semantic verb categories to
every verb, including imperatives, so VIMP tokens were split by the tagger's own category: ACT activity (make, take,
check, follow; 20% of imperatives), COMM communication (explain, tell, say; 6%), MENTAL (think, remember; 13%), CAUSE
(let, i.e. "let's"/"let me"; 27%), ASPECT (keep, start, stop; 2%), DOAUX ("do/don't" imperatives; 4%), OCCUR/EXIST (1.5%),
NONE = verb not in Biber's lists (look, go, find, get, subscribe; 27%). "You know" is not tagged as a discourse marker by
MFTE, so it is the PP2 + know bigram (41 per 10k words). All 45,192 tagged episodes; show rates word-weighted; z on 194.
Show level: every command class correlates with address (.5–.8) and with DIME (.2–.4; DOAUX .38, largest R−L gap +0.65);
"you know" is unrelated to address (r .03) and leans LEFT (R−L gap −0.32).
Education, pooled maximal 1,200 / 63, net of one-sidedness (b, CR2/wild):
| class | pooled | Y1 | Y2 | alongside total address (class; address) |
|---|---|---|---|---|
| total address | −0.212 (.00/.00) | −0.107 (.14/.16) | −0.316 (.00/.01) | |
| ACT activity | −0.184 (.01/.02) | −0.089 n.s. | −0.263 (.02/.00) | −0.207 (.08/.10); +0.034 (p .80) |
| COMM | −0.103 (.09/.12) | −0.036 | −0.158 (.07/.03) | −0.025; −0.176 (.10) |
| MENTAL | −0.125 (.42/.23) | −0.096 | −0.150 | −0.043; −0.198 (.01) |
| CAUSE (let's) | −0.141 (.01/.01) | −0.137 (.00/.01) | −0.153 (.04/.02) | −0.097; −0.113 (.14) |
| ASPECT | −0.304 (.01/.00) | −0.140 | −0.444 (.00/.00) | −0.207; −0.091 (.46) |
| DOAUX (do/don't) | −0.199 (.00/.01) | −0.132 (.04/.02) | −0.255 (.00/.00) | −0.154; −0.061 (.61) |
| NONE (look, go...) | −0.150 (.02/.05) | −0.083 | −0.235 (.00/.01) | −0.047; −0.166 (.18) |
| "you know" | +0.214 (.04/.07) | +0.249 (.02/.06) | +0.192 n.s. | +0.271 (.00/.02); −0.250 (.00) |
HS-or-less share, points per SD: ACT +5.2, DOAUX +5.9, ASPECT +10.2, CAUSE +3.4, MENTAL +3.1 n.s., "you know" −5.6 (.02/.04).
All classes together: none individually significant except "you know" (+0.205, .02/.04); classes inter-correlate .4–.8.
Head-to-head: activity, do/don't, aspectual and let's commands each take the total address coefficient to ≈0; mental and
communication commands leave it at −0.18 to −0.20. "You know" net of address AND one-sidedness: +0.271 pooled (.00/.01),
Y1 +0.297 (.01/.02), Y2 +0.247 (.06/.05). Election distrust: no class predicts it (CAUSE +0.06 .06/.04 the closest).
Reading: with Biber's categories the picture is the same as the hand-sorted one. Commands about doing (activity, do/don't,
let's, keep/start/stop) carry the education link; commands about thinking do not; the shared-knowledge marker "you know"
marks the shows with more educated audiences and is the one feature that survives everything.

## What address does for the show and with the listeners (24 Sept; data/output/show_level_outcomes.csv, apple_reviews.csv 16,434 reviews / 141 shows, only 3 shows at the 500 cap)
Reviews per month (119 shows with >=20 reviews; ad-free address; HC1): +0.21 log points per SD (p .002) net of one-sidedness,
side, solo, episodes/week; +0.17 (p .000; ≈ +18% per SD) net of log mean chart rank, ad share and show age; LOSO
[+0.15, +0.19]; within left +0.20 (p .000), within right +0.10 (p .19). Ratings follow ONE-SIDEDNESS, not address: 5-star
share one-sided +0.36 (p .00) vs address +0.04; mean rating +0.30 vs +0.07; 1-star share −0.20 vs −0.08; review length n.s.
Survey listeners (pooled maximal, 24 shows with >=5) vs chart rank: log survey listeners ~ address +0.37 (p .04) net of chart
rank, i.e. the general-population survey finds MORE listeners of address-heavy shows than their Apple rank implies; reviews
per survey listener ~ address −0.03 n.s.
Temporal order (39 uncapped shows with >=10 dated pre-2023 episodes and pre-2023 reviews; only 18,667 episodes are dated):
later reviews (2023–26) ~ early address (<2023) + early reviews + one-sided + side + solo + age: +0.17 (p .38; ≈ +19%);
reverse (later address ~ early reviews) −0.004 (p .92). r(early address, late address) = .90. Same sign, underpowered.
Elections: commands in the 60 days before the 2018/2020/2022/2024 federal elections vs other times, show FE, cluster show
(18,506 dated episodes): doing-commands −0.01/10k (p .97), thinking −0.11 (p .42), all imperatives −0.15 (p .70), "you"
−0.55 (p .77); no extra on the right. Hosts do not mobilise before elections; address is not a campaign behaviour.
News diet (open-text Q17 source types, pooled 1,200 / 63; regex source-type flags, see handoff/results/address_news_diet.log):
listeners of address-heavy shows are NOT more likely to name radio (−2.1 pts, n.s.; Y1 −3.8, Y2 −0.2), not older (+1.6 yrs
n.s.), and the education link is unchanged with radio/cable/YouTube/social flags controlled (−0.196) and within listeners
naming no radio source (−0.207). The talk-radio-succession reading fails. What differs: they name a NEWSPAPER less
(−6.8 pts per SD, .04/.04; Y1 −7.3, Y2 −6.1; base 11%), network TV slightly more (+3.9, .08/.14), YouTube slightly more
(+2.6, .11/.10). One-sidedness: newspaper −8.8 (.03/.01), YouTube +4.5 (.01/.01).
Survey side for voting/opinions: address predicts none of registration, Y1 participation items, volunteering, citizen power
or any opinion item (earlier sections); nothing to build on there.

## Survey angles never scanned before, address × party, multi-show namers, all address factors jointly (24 Sept; handoff/results/survey_new_angles.log)
Never-scanned respondent variables (pooled maximal 1,200 / 63, address net of one-sidedness; 0 at BH q<.10): mail vs web
mode −0.001; opt-in sample +0.03 (Y2 +0.10, .05/.05); Black +0.03; Hispanic +0.03 (.03/.03; Y2 +0.06 .01/.01, Y1 0);
white non-Hispanic −0.06 (.03/.05; Y2 −0.08 .00/.00, Y1 −0.05 n.s.); LGBT −0.02 (Y1 −0.05 .02/.00, Y2 0); female −0.01;
non-metro county (Y1) +0.03 n.s.; rural-urban code +0.18 n.s.; South region (Y1 Census region; Y2 state labels) 0; answered
the open-ended item / its length n.s.; Y2 interview minutes −0.005; share of attitude items skipped 0. Mode and sample
source do not touch the education link (−0.210 with both controlled; −0.221 web-only).
Address × listener's own party (simple slopes): education Dem −0.22 (.02/.02, n 728), Ind −0.20 (.10/.04), Rep −0.17
(.20/.09): same everywhere. Election distrust: Dem +0.145 (.03/.07), Ind −0.17 n.s., Rep −0.10 n.s.; among Democrats Y1
+0.19 (.02/.13), Y2 +0.10 (.24/.25); LOSO [+0.11, +0.17], CR2 p<.05 in 31/36 drops, weakest without MeidasTouch (325 of
the 728 Democrat listeners, address +0.48, audience distrust −0.38 vs Pod Save America −0.81). A lead at most: fails the
wild bootstrap and does not replicate in Y2. Nothing else differs by party (life ladder, attention, valued, democracy).
Cross-pressured listeners (named shows lean against own party): 59; address 0. Multi-show namers 140 (both sides 7):
address predicts neither naming 2+ shows nor both sides; education link among single-show namers −0.198 (.01/.01).
All address factors jointly (education, CR1 cluster p): controls-only R2 .083; composite −0.21 (R2 .102); vimp −0.19 (p .00)
+ pp2 +0.02 (p .70) (R2 .110); + "you know" → vimp −0.15, pp2 −0.08, you know +0.23 (p .00) (R2 .120); all Biber classes +
pp2 + you know (R2 .126): no class individually (ACT −0.12 p .41, CAUSE −0.07 p .08, ASPECT −0.18 p .11), you know +0.20
(p .01). For other outcomes (election distrust, life ladder, income difficulty, social media, attention) no factor is
significant jointly. Conclusion: commands, not pronouns, carry the composition result; "you know" is the one independent
addition; the survey has no remaining untested variable.

## Never-scanned survey variables, party slopes, joint address factors, between-show ceiling, time-matched address (24 Sept; logs handoff/results/survey_new_angles.log, between_show_ceiling.log; handoff/scans/between_show_ceiling.csv)
Never-scanned variables (mode, sample source, race/ethnicity, LGBT, gender, RUCC, region, open-answer given/length, Y2 interview
minutes, item non-response) ~ address net of one-sidedness, pooled 1,200/63: 0 at BH q<.10. Raw: Hispanic +3 pts (.03/.03,
Y2-driven), white non-Hispanic −6 pts (.03/.05; Y2 −8 .00/.00, Y1 n.s.), LGBT Y1 −5 (.02/.00) Y2 0. Mail/opt-in do not
explain education (−0.210 with both; web-only −0.221). Party slopes (address net one-sided): education Dem −0.22 (.02/.02),
Ind −0.20, Rep −0.17 (.20/.09); election distrust Dem +0.15 (.03/.07), Ind −0.17, Rep −0.10; nothing else. Multi-show namers
140, both-sides 7; address predicts neither; education link among single-show namers −0.198 (.01/.01).
Joint factors (education; CR1 cluster p): controls R2 .083; composite −0.21 R2 .102; vimp −0.19 (.00) + pp2 +0.02 (.70)
R2 .110; + you know: vimp −0.15, pp2 −0.08, youknow +0.23 (.00) R2 .120; all 9 factors R2 .126, only youknow (+0.20, .01)
and CAUSE (−0.07, .08) individually. No other outcome responds to vimp/pp2/youknow.
BETWEEN-SHOW CEILING (adjusted R2 of show FE on residuals after party/attention/age/education/year): education 9.7%
(one-sided explains 2.8, address 4.5, both 4.9, all 9 factors 7.0); election distrust 4.4% (one-sided 2.9, address 1.1);
cynicism 2.2%; life ladder 1.0%; income difficulty 3.4%; social media 1.7%; attention 0.7%; democracy doing well 7.2%
(neither measure explains it; 9 factors 3.0 — likely overfit); trust media 2.8%; leaders accountable 0.0%; age 17.5%.
Reading: net of the controls, listeners of different shows barely differ on attitudes; there is almost nothing between shows
for any show-level measure to explain. Education is the outcome with real between-show variance and address explains
about half of it. Time-matched address (episodes in the 12 months before each wave; r .94/.96 with all-time) gives the same
results on the same 1,196 listeners: education −0.200 (.01/.01), election distrust +0.06, cynicism +0.06 (.10/.08), income
difficulty +0.10 (.09/.04), others null. The exposure measure is not the limitation.

## Size benchmark for the education result (24 Sept, pooled maximal 1,200 / 63)
Mean education (1–8): low-address-tercile audiences 5.39 vs high-address 4.42 = 0.97 points = 0.47 SD of the survey
population (terciles are 1.60 SD of address apart). Same survey benchmarks: Democrats vs Republicans 0.14 SD; under-35 vs
65+ −0.15 SD. Bachelor's or higher: low-address 58%, high-address 33%, Democrats 38%, Republicans 31%, public 32%.
The audience education gap across the address range is more than three times the party gap in education.
Vote choice / turnout: neither wave asks how or whether respondents voted (only registration, Y1 participation items,
Y2 Q66/Q67 voting worries); all previously scanned, null for address.

## Education-adjacent angles for address (24 Sept; log handoff/results/address_education_adjacent.log; pooled 1,200 / 63, net of one-sidedness; pooled | Y1 | Y2)
Education vs income: education ~ address −0.212 -> −0.200 with household income added (Y2 −0.275); income ~ address −0.05 n.s.
(Y1 +0.07, Y2 −0.17 .04/.03); income ~ address + education +0.01 (Y1 +0.10 .05/.05, Y2 −0.07). Schooling residualised on
income −0.192 (.00/.00; Y1 −0.135 .05/.08, Y2 −0.245 .00/.00): address audiences are less schooled AT A GIVEN INCOME.
P(no bachelor's AND household income >= $90k) +6.6 pts per SD (.00/.00; Y1 +7.7 .00/.00, Y2 +5.8 .05/.04); terciles 13% ->
17% -> 18% (public 14%). P(bachelor's AND income < $60k) −0.5 pts n.s. The diploma-divide profile (non-college, not poor)
holds in both years.
Education levels (pts per SD): HS diploma/GED +6.0 (.01/.01); vocational/trade +3.1 (.00/.00; base 6%); some college −1.4 n.s.;
associate +1.3 n.s.; bachelor's −4.2 (.16/.18); graduate/professional −5.0 (.02/.03). The shift runs from graduate degrees
to high-school and trade credentials; the middle (some college, associate) is unchanged.
Employment: full time −6.1 pts (.05/.04); not employed/not looking +5.2 (.07/.07; Y1 +9.8 .01/.02, Y2 +0.7); unemployed
+0.9 (Y2 +2.5 .03/.03). Mixed by year.
Civic socialisation (Y1 only): formal civics education −0.07 n.s.; parents/relatives encouraged participation −0.148
(.07/.10; −0.125 net of education); information overload +0.11 n.s.; barriers "not knowing enough" / "unsure how" ≈ 0 both
years; schools satisfaction ≈ 0; free time weekday +0.135 (.14/.03), weekend +0.09 (.13/.05) — p-values disagree.

## Strengthening checks (25 Sept): review readability, exaggeration, stance index
REVIEW READABILITY as an independent audience-education proxy (116 shows with >=20 Apple reviews of >=15 words; 11,870
reviews; Flesch-Kincaid grade of review text; WLS by sqrt(n); HC1; controls one-sidedness, side, solo, log episodes/week):
address −0.35 grade levels per SD (p .000; show SD 1.01), one-sidedness −0.26 (p .001), side n.s.; word length −0.056 (p .000);
words per sentence −0.22 (p .04); review length n.s.; unchanged with mean rating controlled. Proxy check: r(review grade,
survey listener education) = +.57 across the 23 shows with >=5 listeners (word length +.58). Reading: an independent source,
twice the shows, same direction and same specificity (address beats side); the survey composition result is not a survey
artefact.
MISREPRESENTATIVE EXAGGERATION etc. (S&B model-coded; note S&B's 13 types do not include "conspiracy theory"): share of
passages with exaggeration ~ address −0.005 (p .18) | one-sidedness +0.051 (p .000) | right +0.034 (p .000); slippery slope
address n.s., one-sided +.008 (p .000); conflagration address −0.008 (p .03), one-sided +.022 (p .000); any outrage address
n.s., one-sided +.11 (p .000). The guide-stance shows are not the exaggeration shows; one-sided shows are.
STANCE INDEX (guide vs peer), a-priori = z(doing-command z − "you know" z), plus a Y1-fitted linear index (vimp −0.10, pp2
−0.03, you know +0.25), tested on Y2 (log handoff/results/stance_index_y1_to_y2.log). Education: composite Y1 −0.107 n.s.,
Y2 −0.316; a-priori stance Y1 −0.181 (.00/.01), Y2 −0.253 (.00/.00), pooled −0.217; Y1-fitted Y2 −0.228 (.01/.01).
Y2 adj R2: controls .063, composite .102, stance .114, fitted .105. Y2 head-to-head: stance −0.253 (.00/.01), composite
alongside −0.148 (.14/.10). Caveat: stance also predicts Y2 election distrust +0.129 (.02/.02) (cynicism, life ladder,
attention null) — not purely compositional in Y2; check Y1 before using.

## External replication with Pew American Trends Panel microdata (25 Sept; step7_audience/8_pew/pew_checks.py; log handoff/results/pew_external_checks.log; data/output/pew_w165_source_audiences.csv)
W165 (March 2025, n 9,482; SOURCEUSE2 "regularly get news from" 30 named sources; F_EDUCCAT2; WEIGHT_W165). Ten sources map to
scored shows (Rogan, Tucker Carlson Network, Daily Wire -> Ben Shapiro Show, Breitbart -> Breitbart News Daily, NPR -> NPR
Politics Podcast, Fox News -> Fox News Rundown, NBC News -> NBC Nightly News podcast, NYT -> The Daily, Atlantic -> Radio
Atlantic, Politico -> POLITICO Energy [weak proxy]). Weighted BA+ share of each source's regular audience vs our address:
Spearman −0.88 (p .00, n 10); without the Politico proxy −0.93 (p .00, n 9), Pearson −0.71 (p .03); r(address, HS-or-less
share) +0.82. Net of one-sidedness: −19 pts BA+ per SD of address (p .04, n 9); net of the audience's Republican share −11
(p .21, n 9). Extremes: Rogan 27% BA+ (address +1.07), Fox 27% (+0.29), Tucker 28% (+0.28) vs NPR 59% (−0.17), NYT 57%
(−0.49), Atlantic 62% (−0.36). Side check: r(our side score, Pew Republican share) = +0.96 over 6 sources. All adults 34%.
W118 (December 2022, n 5,132; PODMAIN_CODES = podcast listened to most, coded to named shows; PODMHOST = felt connection to
its host; WEIGHT_W118). 206 respondents name one of 10 scored shows (Rogan 84, The Daily 37, Bongino 17, Shapiro 16 + Daily
Wire/Morning Wire 11, Pod Save America 12, Timcast 8, Breaking Points 7, Beck 5, Levin 5, Mea Culpa 4). Controls party,
education, age, gender; cluster = show (G = 10, so CR2 df is tiny):
  felt connection to host (1–5): address −0.03 (n.s.); net one-sided −0.12 (n.s.); ONE-SIDEDNESS net address +0.51 (.11/.08);
  very/extremely connected: address −0.02; one-sided +0.22 (.06/.06). Host shares political opinions: one-sided +0.90 (.02/.03),
  address n.s. Trust in podcast news, discussing, recommending: address n.s.
  Listener education: address −0.38 (.03/.05) on the 1–6 scale, BA+ −13.6 pts per SD (.05/.06); net one-sided −0.35 (.05/.19).
  Show level: Spearman r(address, share BA+) = −0.78 over 10 shows (−0.82 over the 7 with n>=7); r(one-sided, BA+) = −0.31;
  r(address, very connected) +0.14; r(one-sided, very connected) +0.54.
Reading: two Pew probability samples, one from 2022 and one from 2025, neither touched before today, reproduce the
composition result: the more a show talks at the listener, the less educated its regular audience, with our side/one-
sidedness measures also validated (Republican share r .96). The parasocial claim does NOT follow address: felt connection to
the host follows one-sidedness (as do host-opinion frequency and, in the Apple reviews, ratings). Style predicts reach;
stance predicts the bond and the beliefs.

## Shape of the education association and the Richardson leverage point (26 Sept; step7_audience/9_education_probes)
Respondent education by quintile of show address (pooled 1,200): 5.60, 4.74, 4.21, 4.35, 4.00 (BA+ 62%, 41%, 28%, 31%,
27%). Slope net of one-sidedness within subsets: all −0.212 (.00/.00); excluding the lowest-address third −0.002 (n.s.);
excluding the lowest fifth −0.173 (.18/.22); lowest two-thirds only −0.313 (.03/.01). Show level (n>=5), education by
address quartile: 5.49, 5.53, 4.21, 4.04; quadratic term n.s. Reading: a step near the corpus mean, not a gradient —
shows below average on address have professional-class audiences; at or above average, audiences look like the public
and more address changes little. Paper text and abstract must describe a threshold, not a dose-response.
Richardson: 142 respondents name her (61 Y1, 81 Y2); 123 are not matched elsewhere; with the 5 title-namers the Letters
cluster would be 128 at address −2.90 (next lowest −2.27). Her readers are highly educated (+0.64 SD; the linear fit
predicts +0.72), so it is leverage, not composition: including them, net estimate −0.212 -> −0.103, CR2 df 7.0 -> 2.3,
p .21/.16; address alone −0.143 (df 1.8); Letters winsorised to −2.27: −0.127 (.12/.08); all address clipped at −1.5:
−0.166 (.02/.02). Year 1 alone on the 1,200 frame: alone −0.199 (.02/.04), net −0.107 (.14/.16); Year 2 −0.326 / −0.316
(.00/.00). Ladder at freeze (18 Sept): census and bands 1–5 complete (98%), bands 6–10 and 11–15 each about half (53%,
51%). The 15 shows dropped 219 -> 204 were not in the H25 transcript sample; decision recorded 14 Aug 2026
(data/output/corpus_shows.csv, excluded_reason).
