# EXPLORATORY (2026-09-03, after both pre-registered tests failed): what show-level feature tracks audience cynicism across ALL matched podcasts?
Status: exploratory, hypothesis-generating. Discovered by scanning 26 show features × 3 outcomes on the pooled data (BH-corrected within outcome). Needs pre-registered confirmation on Kettering Y2. Not causal: audiences self-select.

## Data
Pooled respondents: 1,284 (discovery 385 + out-of-frame 749 + within-frame 150), 64 shows, all with corpus-scale dir_z. Show features: register (dir_z and components), Rooduijn–Pauwels populism dictionary rate (rp_rate), populist.py elite/people/media-criticism rates, Fawzi-style anti-media, game/strategy frame, Biber D4, political-content density, ideological intensity (LLM |ternary| share; true for 53 corpus shows, distilled TF-IDF classifier for 11 others: grouped AUC .96/.80, out-of-fold show-level r = .93). Files: handoff/scans/show_features_ALL.csv, explore_scanB.csv, explore_showlevel.csv, rp_words_by_show.csv, rp_audience_profile.csv, inten_distilled.csv; scripts in handoff/scripts_exploratory/.
Model: respondent-level OLS, controls party(5)/attention/age/education/right-wing-platform index/show lean; CR2 + Satterthwaite df; wild cluster bootstrap; show clusters.

## Result 1 — Populist vocabulary (R&P: elite*, corrupt*, propagand*, politici*, betray*, scandal*, truth*, establishm*, ruling* …) tracks audience institutional cynicism; listener-directed address does not
| sample | shows | N | b (SD cyn per SD rp_rate) | CR2 p | wild p |
|---|---|---|---|---|---|
| pooled | 64 | 1275 | +0.195 | .001 | .002 |
| FRESH only (never examined for this before today) | 20 | 891 | +0.211 | .028 | .017 |
| discovery only | 51 | 384 | +0.148 | .073 | .038 |
| fresh, political-content shows | 14 | 405 | +0.293 | .024 | .039 |
BH across 26 features: q = .039 (cyn8), .003 (Q33E election distrust), .009 (exclusivity). rp_rate × fresh interaction = −0.011 (p .80): the effect is the SAME SIZE in discovery and fresh samples — the replication signature dir_z lacked (dir_z × fresh = −0.12, p .05).
Robust to: full demographic set (+0.177), survey weights (+0.204), dir_z + political density + format (+0.212), everything at once weighted (+0.225, wild p < .001). Leave-one-show-out (16 largest shows): +0.157 to +0.221, all wild p ≤ .013. Within LEFT shows +0.180 (p .011, 37 shows); within RIGHT shows +0.116 (p .12, 27 shows; Q33E p .05). No party interaction.
Horse race vs dir_z: pooled rp | dir_z +0.212 (p .001), dir_z | rp +0.149 (p .04); FRESH rp | dir_z +0.254 (p .001), dir_z | rp +0.122 (p .27). Show-level corr(rp_rate, dir_z) = +.16 — different constructs.
Horse race vs ideological intensity (r = .53 with rp): cynicism — pooled rp | inten +0.145 (p .013), inten | rp +0.117 (p .085); FRESH rp | inten +0.179 (p .036), inten | rp +0.008. Election distrust: rp wins both samples. Exclusivity: INTENSITY wins pooled (+0.133, p .005; rp → .016), neither in fresh.
Word-level (29 shows, n-weighted r with party-adjusted audience cynicism): truth* .57, corrupt* .43, propaganda .40, betray* .37, ruling .35, scandal .31; elite* carries exclusivity (.49). The signal is the "they are lying to you / corruption" vocabulary, not "the people".
Face validity: highest rp_rate — Young Turks 14.7, Kulinski 14.4, X22 Report 14.3, Majority Report 12.9, Parnas 12.0, Benny 11.0, Bongino 10.1 (both wings); lowest — Rogan 3.6, The Daily 3.7, Kelly 4.0, NBC Nightly 4.1, Shawn Ryan 4.5.
Audience profile of high-rp shows (112 items, BH): 40 items at q < .05 — government does not serve interests (Q35D), elections not administered well (Q33E), REJECTS more presidential power (Q32G) and government direction of media (Q32D), less confidence in courts/Congress/press freedom/equal treatment/division of powers, lower life ladder (Q1), lower trust in accountability (Q31). = the engaged anti-institutional / anti-authoritarian profile from §E.2a, now tied to a show-level content feature.

## Result 2 — Ideological intensity and listener-directed address track abandoning mainstream news (exclusivity), not cynicism
Show-level (29 shows): dir_z ~ exclusivity r_w = .46 (p .046); intensity | rp → exclusivity +0.133 (p .005) pooled. Address and intensity are the "who leaves the mainstream" predictors; populist vocabulary is the "who is cynical" predictor.

## Reconciliation with round 15
Round 15 (strict discovery, 4-item composite): R&P alone +0.08 (ns), address +0.20. Today, discovery-only with the 8-item composite: rp +0.15 (wild .04). The 8-item outcome adds Q33E/Q31/Q35A/Q35D, which are the items most tied to rp. In fresh data address fades and rp holds. Each sample has a different winner, but only rp's effect is sample-invariant.

## Caveats
Exploratory (26 × 3 scan); selection not causation; ecological exposure; 20 fresh clusters; intensity for 11 shows is model-distilled; rp_rate for corpus shows is measured over all corpus years while out-of-frame shows use 50 recent episodes.
## Recommended next
Pre-register on Kettering Y2: H = rp_rate → 8-item cynicism (+), controls as above, CR2+wild; secondary Q33E; with dir_z and intensity as covariates. Also: split-half reliability of rp_rate across episodes; event/topic invariance as done for dir_z.

# EXPLORATORY, continued — what does LISTENER-DIRECTED ADDRESS predict? (119 outcomes × pooled 64 shows and fresh 20 shows; controls party/attn/age/edu/rw_plat/show-lean/gender/race/income/urban-rural; cluster p; BH) — handoff/scans/address_outcome_scan.csv
Address predicts 15 items (pooled) / 16 (fresh) at q<.05; 20 items replicate at q<.15 in BOTH samples. It does NOT predict institutional cynicism once populism/intensity are considered (cyn8 +0.10 pooled, +0.03 fresh). Populism vocabulary predicts 40/50 items — almost all institutional distrust. The three features map onto three outcome families:
  ADDRESS -> economic grievance + social disaffection | POPULIST VOCABULARY -> institutional cynicism & election distrust | INTENSITY -> exclusivity (abandoning mainstream news)
## Address profile (both samples; sign already interpreted)
Economic: inequality is unfair (Q32H +0.18/+0.23*), society should reduce it (Q32I), government should meet basic needs (Q36A), against wealthy influence in politics (Q36I), government should deliver more (Q32J rev), cannot afford necessities (Q15A) — all with INCOME controlled. Populist-vocabulary audiences show the OPPOSITE sign on inequality items.
Disaffection: lower life ladder (Q1), weaker local-community identity (Q14B), less church attendance (Q44), less trust in faith leaders and in family/friends as sources (Q18A/E), contacting officials ineffective (Q24B) but protest effective (Q24C; populism audiences say the reverse), election overseers acted improperly when results surprise (Q32C), would bar radicals from office (Q32B; populism audiences the reverse). More free time (Q5), more social-media hours (Q21), more YouTube/Reddit.
## Composite tests (indices z-scored on full Kettering; formative bundles, r-bar .20 / .12)
| outcome | sample | address alone | address ǀ populism+intensity | populism ǀ … | intensity ǀ … |
|---|---|---|---|---|---|
| ECONOMIC GRIEVANCE (6 items) | pooled 63 shows | +0.242 (CR2 .017, wild .021) | +0.220 (.065/.096) | −0.022 | +0.065 |
| | fresh 20 | +0.274 (.044/.070) | +0.284 (.091/.244) | +0.012 | −0.040 |
| | discovery 50 | +0.171 (.074/.074) | | | |
| DISAFFECTION (8 items) | pooled | +0.178 (.004/.006) | +0.185 (.047/.077) | +0.037 | −0.003 |
| | fresh | +0.200 (.016/.016) | +0.248 (.093/.294) | +0.091 | −0.071 |
| | discovery | +0.127 (.080/.168) | | | |
Leave-one-show-out (econ grievance, pooled): +0.21 to +0.26 for all shows except Rogan (+0.13, wild p .013 — Rogan's young, precarious audience carries part). Within RIGHT shows +0.283 (.033/.069; Bongino/Beck/Kirk audiences vs Shapiro/Kelly); within LEFT +0.087 ns.
Reading: hosts who talk AT the listener draw audiences that are economically aggrieved and socially disaffected — precarious, secular, detached from local community and from institutional participation, believing in protest rather than letters — but not distinctively cynical about institutions. That is the "left-behind / status-anxiety" audience of the political-economy literature (Gidron & Hall 2017; Norris & Inglehart 2019), and it is a within-right contrast. Exploratory; selection not causation; to be pre-registered on Y2 alongside the populism hypothesis.

# CORRECTION (same day, later): the three-way division of labor above is NOT reproduced inside the corpus frame
Corpus frame only (discovery + within-frame pocket; 54 shows, 535 resp), b per SD alone [conditional on the other two], wild p:
| outcome | address | populism | intensity |
|---|---|---|---|
| econ grievance | +0.13 (.14) [+0.07] | +0.15 (.003) [+0.12 (.08)] | +0.15 (.08) [+0.04] |
| disaffection | +0.13 (.07) [+0.09] | +0.03 | +0.09 (.08) [+0.06] |
| cynicism | +0.17 (.03) [+0.11 (.07)] | +0.12 (.05) [+0.07] | +0.17 (.01) [+0.06] |
| election distrust | +0.15 (.003) [+0.07] | +0.11 (.004) [+0.04] | +0.16 (.01) [+0.10] |
| exclusivity | +0.11 (.03) [+0.02] | +0.13 (.05) [+0.06] | +0.16 (.002) [+0.11 (.04)] |
Out-of-frame only (10 shows, 749): address→grievance +0.31 (.11), address→cynicism +0.02; populism→cynicism +0.21 (.07), populism→grievance −0.13. The pooled "address→grievance / populism→cynicism" split is produced by the out-of-frame mega-shows (Rogan above all), not by the corpus.
Standing conclusion: within the corpus, the three content features (inter-r .4–.5) are jointly associated with every disaffection outcome at ≈ +0.15/SD and cannot be adjudicated with 10–54 clusters. The ONE feature-specific link consistent in every cut is ideological intensity → exclusivity. The populism→cynicism and address→grievance sections above are to be read as pooled-sample exploratory patterns that do not replicate inside the frame. Paper language: "jointly associated; feature unresolved; Y2 pre-registered to adjudicate."

# EXPLORATORY, continued — what does ADDRESS predict INSIDE THE CORPUS FRAME? (54 shows, 535 resp; 130 outcomes incl. composites and audience demographics) — handoff/scans/corpus_address_scan.csv
Address alone: 11 outcomes at BH q<.05 (YouTube use, audience education, election suspicion Q32C, exclusivity, Q35E, cyn8, disaffection, social-media hours, Q33E, TikTok). Conditional on populism + intensity only TWO survive BH: audience EDUCATION (−0.27) and "no desire to get involved" (Q12I). Everything attitudinal is shared with the rivals (collinear); the composition result is not.

## RESULT — Listener-directed address draws a LESS FORMALLY EDUCATED audience. Replicates in every sample; address-specific; not income.
| sample | b (SD of education per SD address; party + lean) | CR2 p | wild p | G / N |
|---|---|---|---|---|
| corpus frame | −0.272 | .004 | .002 | 54 / 535 |
| discovery only | −0.285 | .010 | .009 | 51 / 385 |
| within-frame pocket | −0.160 | .042 | .003 | 10 / 150 |
| out-of-frame | −0.180 | .041 | .084 | 10 / 749 |
| pooled | −0.221 | .001 | <.001 | 64 / 1284 |
+ age/gender/income/race/urban/attention: corpus −0.278 (.007/.008); pooled −0.202 (.003/.002). + populism/intensity/political density: corpus −0.119 (.19; df 15, collinear), pooled −0.165 (.012/.004). Within LEFT shows −0.29 (.025/.013); within RIGHT pooled −0.19 (.003/.005) (corpus-right too small: 18 shows, 77 resp, −0.15 ns). Leave-one-show-out (corpus, 12 largest): −0.23 to −0.31, all wild p ≤ .047. Show-level n-weighted r = −0.57 (19 shows ≥5 resp).
Other composition outcomes for address: age, gender, income, attention all NULL; white −0.09 (p .05–.07). So it is education specifically, not SES.
Components: composite carries it; you_rx / imper_syn / imper_rx individually and jointly are collinear and unstable (suppression) — report the composite only.
Interpretation (exploratory): direct, imperative, second-person address is the "involved/oral" register (Biber Dimension 1); it draws listeners with less formal education independent of party, income and age. This is a register–audience fit result (who a style reaches), not an effects result. Consistent with the partisan-register finding: right-leaning shows use more of it AND right-leaning podcast audiences are less formally educated; but the association holds WITHIN left shows too.

## Also examined, NOT robust
Epistemic imperatives (understand/realize/consider…): corpus-frame link to low external efficacy (−0.29, p .01) survives log transform and rivals but weakens without X22 Report (−0.19, p .05–.06), is null out-of-frame (−0.02) and in fresh data (−0.07). Report as corpus-only, leverage-sensitive, not replicated. Component decomposition for the 11 out-of-frame shows: handoff/scans/holdout_imp_decomp.csv.

## Address × EDUCATION (exploratory; corpus frame unless stated)
Address × edu interactions on six composites: cyn8 −0.09 (wild .03), Q33E −0.10 (.07), election-distrust composite −0.04 (.36); others null. Subgroup estimates:
| outcome | LOW-education listeners (edu ≤ some college; 40 shows, 285 resp) | HIGH-education (37 shows, 239) |
|---|---|---|
| election distrust (Q33E rev + Q32C) | **+0.285 (CR2 .001, wild .004)** | +0.163 (.24/.32) |
| cyn8 | +0.175 (.07/.04) | +0.119 (ns) |
| exclusivity | +0.110 (.03/.00) | +0.078 (ns) |
| disaffection | +0.036 | +0.215 (.03/.04) |
Low-edu election distrust: address | populism + intensity +0.163 (.09/.11); populism | address + intensity +0.004; intensity +0.166 (.12/.09). LOSO (8 largest shows): +0.26 to +0.29, all wild p ≤ .03. Item scan among low-edu corpus listeners (118 outcomes): only Q32C (q .001), Q33E (q .04), exclusivity (q .08) survive BH.
Other samples, low-edu: pooled +0.136 (.08/.04); within-frame pocket +0.273 (.15/.07); out-of-frame +0.064 (ns).
Reading: inside the corpus, address's clearest attitudinal correlate is ELECTION DISTRUST, and it is carried by less-educated listeners (where address also beats populism); but the continuous interaction is not significant and the mega-shows do not show it. Report as: "address is associated with election distrust among corpus listeners, most clearly among those with less formal education; the moderation itself is not statistically established." Exploratory.

## Does EDUCATION change what podcast listening / address is associated with? (three designs; answer: essentially no)
A. Address × education interaction scan, 119 items: corpus frame — 0 items at BH q<.10; pooled — 2 (inequality-unfair Q32H, interaction −0.078 q .004: address→economic grievance stronger among less-educated; volunteering Q13 q .04). Composites: no significant interaction on election distrust (−0.04, p .36).
B. Low-education respondents in the FULL sample (12,458 non-podcast baseline; 166 listeners of high-address corpus shows; 30 low-address): listeners of ANY corpus show differ from the low-edu baseline on 43–46 of 112 items (institutional trust items −0.4 to −0.7 SD); high- vs low-address listeners differ on 4 items, all resting on n=30 — not a finding. File: handoff/scans/lowedu_baseline_scan.csv.
C. Action imperatives → participation among low-edu corpus listeners: one survivor, voter registration Q47 (+0.14, q .02; 1=yes → LESS likely registered on shows with more action imperatives), single item, subgroup — not robust. Among HIGH-edu pooled listeners, action imperatives → lower perceived effectiveness of contacting officials/donating/voting (q .02) — noted, unexplained.
D. POPULATION-LEVEL (n=18,893, weighted, full control set), diet segment vs mainstream-only, within education strata:
| outcome | podcast-only, LOW edu | podcast-only, HIGH edu | interaction × edu (p) |
|---|---|---|---|
| cynicism (8-item) | **+0.340** (p<.001) | **+0.348** (p<.001) | −0.005 (.91) |
| election distrust | +0.218 (<.001) | +0.244 (.001) | +0.060 (.20) |
| external efficacy | −0.007 (.89) | −0.172 (.007) | −0.043 (.36) |
| economic grievance | −0.074 (.17) | −0.132 (.03) | −0.066 (.14) |
Podcast+mainstream: +0.216 low-edu vs +0.140 high-edu (interaction −0.094, p .022) — the only education difference, and it is the MIXED diet, not the podcast-only one.
Standing conclusion: the podcast-diet association with institutional cynicism is IDENTICAL across education levels (+0.34 SD in both). Education stratifies levels of cynicism, not the podcast gap. Lower-education podcast listeners are not differentially affected; if anything, podcast-only listening is associated with lower efficacy only among the MORE educated. Address × education: not supported in the corpus.

## Psychological-effect angles for ADDRESS (dose, social embeddedness, within-show timing) — answer: no dose-response; Y1 cannot separate effect from selection
- Interview date: NOT in Kettering Y1 (no date/time/wave field) → within-show time-varying exposure with show fixed effects is impossible in Y1. Request for Y2: interview date.
- DOSE (show named as 1st vs 2nd/3rd news source; 569 first-namers of 1,284): no attitudinal outcome shows a dose gradient — cyn8 address×first −0.04 (corpus) / −0.09 (pooled), election distrust −0.05/−0.03, econ grievance +0.05/+0.12 (p .09), embeddedness +0.17 (p .11)/+0.03; named-later estimates are as large or larger. The ONLY outcome with a dose gradient is EDUCATION composition: first-namers −0.35 (wild .003) vs later −0.20 (.04) in corpus; −0.27 vs −0.18 pooled — the signature of selection (the more central the show, the sharper the audience fit), not of effect.
- SOCIAL EMBEDDEDNESS composite (local identity, trust family/friends as sources, volunteering, church, community events, loneliness): corpus −0.07 (ns); fresh −0.19 (.04/.05); pooled −0.15 (.05/.03); not robust to populism+intensity. Loneliness alone: null. "Beliefs of people like me are valued" (Q22): null.
Standing conclusion: in Y1, address predicts WHO listens (education; sharpest among first-namers) and shares a diffuse disaffection correlate with the other content features; nothing shows a dose-response, and the design cannot distinguish a psychological effect from audience self-selection. A psychological effect of address may well exist; Y1 is not the instrument that can show it.

## Further angles for ADDRESS (response style, thresholds, you/imperative subtypes, election distrust full battery)
- RESPONSE STYLE (29 five-point items; extremity, midpoint use, straightlining, non-response; education controlled): address → all null (extremity +0.07, p .22 pooled). Intensity → more extreme answers (+0.18, p .02/.05) and less straightlining (−0.16, p .005/.013); populism → less straightlining (−0.13, p .006/.023). Attitude-strength markers belong to intensity/populism, not address.
- THRESHOLDS (show-level address quintiles vs middle): no non-monotonic or threshold pattern for cyn8/election distrust/exclusivity; education composition monotone (corpus Q4 −0.61, Q5 −0.55 vs middle).
- SUBTYPES (corpus): attention imperatives ("look/listen") → election distrust +0.19 (CR2 .009, wild .003) and exclusivity +0.09 (.04/.02); action imperatives → election distrust +0.19 (.005/.027); deontic-you → exclusivity +0.12 (.03/.02); addressive-you → election distrust +0.14 (wild .04). Epistemic subtype: see above.
- ELECTION DISTRUST full battery (corpus, 53 shows): address alone +0.247 (CR2 .002, wild .008); + political density + format +0.252 (.001/.002); | populism +0.230 (.006/.033); | intensity +0.130 (.10/.15); | populism + intensity +0.123 (.11/.15) — intensity absorbs it (intensity | address +0.184, wild .008). Both items load (Q33E +0.15 p .009; Q32C +0.24 p .008). Within LEFT +0.25 (.011/.032), within RIGHT +0.26 (.095/.077). LOSO +0.20 to +0.26, all wild p ≤ .05. By sample: discovery +0.264 (.008/.025); within-frame pocket +0.150 (ns); out-of-frame +0.075 (ns); fresh combined +0.064 (ns); pooled +0.129 (.033/.013), pooled | rivals +0.142 (.031/.070). Dose (named first vs later): no gradient.
FINAL STANDING for address in Y1: (1) audience EDUCATION composition — robust, replicated in four samples, dose-consistent → report. (2) Election distrust — the recurring attitudinal correlate in the corpus (+0.25, stable to LOSO, both leans, both items), but shared with ideological intensity and not reproduced in either fresh sample → report as exploratory, corpus-frame, pre-register on Y2 with election distrust as the primary outcome. (3) Cynicism, efficacy, grievance, embeddedness, response style, dose-response, education moderation — not supported for address. Design limit: cross-sectional self-selection; no interview date; psychological effect not identifiable in Y1.

## POLITICAL-CONTENT FRAME (≥ 15% political passages, corpus 10th pct; applied symmetrically): what ADDRESS predicts in BOTH the corpus and the out-of-frame shows — handoff/scans/political_frame_address_scan.csv
Frame: 50 corpus shows (502 resp; Candace, Majority Report, Michael Berry, Volts dropped) + 6 out-of-frame shows (Parnas .47, Bongino .29, Shapiro .22, Daily .18, Kirk .18, Tucker .17; Beck .146 just below; Kelly/Ryan/Rogan out) = 56 shows, 792 resp.
Item scan (128 outcomes; CR1 cluster p within each frame): STRICT survivors (p<.05 in BOTH frames, same sign): 12 — election distrust composite (corpus +0.22, out +0.28), Q32C, Q33E, Q33F criminal justice, Q33G equal treatment under law, Q32J, Q1 life ladder, disaffection (+0.17/+0.27), economic grievance (+0.16/+0.19), education (−0.29/−0.08), white. NOTE: out-of-frame p-values rest on 6 clusters (anti-conservative); the proper inference is the pooled model below.
Pooled political frame, CR2 + wild, + frame indicator (df ≈ 9.4, G=56, N≈776):
| outcome | address alone | address ǀ populism + intensity |
|---|---|---|
| election distrust | **+0.267 (CR2 <.001, wild <.001)** | +0.194 (wild .018) |
| cynicism 8-item | +0.221 (.005 / <.001) | +0.164 (.037) |
| disaffection | +0.159 (<.001 / .003) | +0.139 (.040) |
| economic grievance | +0.182 (.007 / .005) | +0.127 (.095) |
| exclusivity | +0.089 (.066 / .055) | +0.017 |
| efficacy | −0.105 (.079 / .073) | −0.056 |
| education | −0.213 (.006 / <.001) | — |
Cutoff sensitivity (pooled, wild p): election distrust +0.20 to +0.29 at every cutoff from .08 to .30 (all ≤ .047); cyn8 +0.12 to +0.22; education −0.19 to −0.33; disaffection +0.10 to +0.17. The .15 cutoff is not doing the work.
Out-of-frame political shows ALONE (6 clusters, df 1.7): election distrust +0.267 (CR2 .20, wild .077); disaffection +0.214 (.11/.08); cyn8 +0.237 (ns); education −0.082 (.048/.045). Right sign throughout; cannot stand alone.
Leave-one-show-out (pooled political): election distrust +0.25 to +0.30, all wild p ≤ .003, including dropping each out-of-frame show. Corpus-political only (50 shows): election distrust +0.254 (.002/.003), cyn8 +0.208 (.003/.005).
Standing: within the political-content frame, address is associated with election distrust (+0.27), cynicism (+0.22), disaffection and economic grievance, and with a less-educated audience, robustly under proper inference, stable to the cutoff and to every leave-one-out, and surviving populism + intensity for election distrust, cynicism and disaffection. The out-of-frame political shows point the same way but contribute only 6 clusters. Caveats: the content criterion was adopted AFTER the pre-registered full-frame tests failed (symmetric, cutoff-insensitive, but post hoc); selection not causation; to be pre-registered on Y2 with the frame criterion stated in advance.
