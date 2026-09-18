# Stability of listener-directed address (MFTE version), analysis corpus
Episodes 30,915 | shows 194 | dated 30,915 | 2006-03-03 to 2026-07-15

A. ICC(1) composite: 0.490 (share of episode variance that is between shows); reliability of a show mean at the median 124 episodes (ICC(1,k), Spearman-Brown) = 0.992
A. ICC(1) imperatives (VIMP): 0.465 (share of episode variance that is between shows); reliability of a show mean at the median 124 episodes (ICC(1,k), Spearman-Brown) = 0.991
A. ICC(1) second person (PP2): 0.420 (share of episode variance that is between shows); reliability of a show mean at the median 124 episodes (ICC(1,k), Spearman-Brown) = 0.989

B. Two-way fixed effects, 24,829 dated episodes 2018+, 35 quarters: show fixed effects explain 49.2% of variance; adding quarter fixed effects explains a further 0.16% (within-show R2 0.003); cluster-robust joint test of the quarter effects F = 1.52, p = 0.042

C. Event study (show fixed effects, post minus pre within ±70 days, SEs clustered by show); placebo distribution from 300 random dates: sd 0.043, 95% range [-0.094, +0.089] SD
| event | shift (SD of episode address) | cluster p | episodes | placebo rank |
|---|---|---|---|---|
| 2020 election | +0.006 | 0.90 | 539 | 0.86 |
| 6 January | +0.022 | 0.62 | 587 | 0.53 |
| Dobbs | +0.009 | 0.81 | 803 | 0.77 |
| 2022 midterm | +0.048 | 0.13 | 915 | 0.22 |
| assassination attempt | +0.012 | 0.68 | 1,505 | 0.68 |
| Biden withdrawal | +0.024 | 0.38 | 1,567 | 0.51 |
| 2024 election | -0.058 | 0.03 | 1,977 | 0.16 |
| 2025 inauguration | +0.024 | 0.39 | 2,079 | 0.51 |

D. Topic invariance, 7,266 episodes with topic proportions (75-topic LDA), 194 shows: show fixed effects explain 53.8%; the 74 topic proportions add 13.50% (within-show R2 0.292); cluster-robust joint test F = 52.24, p = 0.000
   Between shows, topic mix predicts show-mean address with R2 0.79 (adjusted 0.66, 194 shows)

E. Format control, 194 shows: right minus left gap in show-mean address +0.342 (p 1.1e-05); with a solo-host indicator +0.372 (p 5.6e-06); solo hosts -0.115 (p 0.14); solo share right 84% vs left 58%

Run 17 Sept 2026, scratchpad/stability_checks.py. Episode address = mean of z-scored MFTE VIMP and PP2 rates per 10k words across the analysis episodes. Methods: ICC(1)/ICC(1,k) Shrout & Fleiss 1979, Bliese 2000; two-way FE and cluster-robust inference Bertrand, Duflo & Mullainathan 2004; in-time placebo Abadie, Diamond & Hainmueller 2010; topic model Blei, Ng & Jordan 2003 (K=75, topic arm).


## D2. Subject topics versus style topics (added same run)
Style/housekeeping topics (no subject nouns in top words): T61, T12, T57, T39, T28, T60, T46, T9, T16, T32, T44, T7, T48, T33, T30

| topic set | within-show R2 (share of total variance) | cluster p | between-show 5-fold CV R2 |
|---|---|---|---|
| all 75 topics | 0.292 (13.5% of total) | 0.000 | 0.51 |
| 60 subject topics (politics, policy, world, culture) | 0.197 (9.1% of total) | 0.000 | 0.34 |
| 15 style/housekeeping topics (filler, banter, sign-offs, ad reads) | 0.244 (11.3% of total) | 0.000 | 0.50 |

Within show, +10 points political subject share: address -0.264 SD (p 0.000). Reading: the topic model's conversational-mode topics carry register; political subject matter does not move address.


## D3. Conversational share versus political subject
Within show: conversational share +10 pts -> address +0.264 SD (p 0.000), within R2 0.157; adding the composition of political subjects adds 0.063 (joint p 0.000).
Between shows (194): right-left gap +0.342 -> +0.388 with conversational share controlled (p 8.5e-08) -> +0.395 with solo host too. r(conversational share, address) = 0.40; conversational share right 0.35 vs left 0.37.
