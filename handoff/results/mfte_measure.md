# The address measure from a published tagger (MFTE), 16 September 2026

Run: `step6_register/run_mfte_corpus.sh` (45,570 transcripts cut into 154,417 pieces of ≤ 3,000 words, four GPU
shards) with `mfte_supervise.sh`; scores by `step6_register/mfte_address.py`. MFTE = Multi-Feature Tagger of
English (Le Foll 2021; Python port by Shakir), Stanza part-of-speech layer. Tags used: **VIMP** (imperative verbs)
and **PP2** (second-person pronouns). Per-episode rates per 10,000 words, show means weighted by episode word
count, each z-scored against the 194 reference shows, averaged: `mfte_z`. 45,192 episodes ≥ 500 words, 228 shows
(all 204 corpus shows and the chart-absent shows), 409M words.

## Validation against the current composite and the anchors (194 reference shows)

| | mfte_z | dir_z (current) |
|---|---|---|
| correlation with the other measure | r = 0.93 (Spearman 0.93) | — |
| MFTE imperatives vs paper imperatives (show level) | r = 0.79 | |
| MFTE second person vs paper second person | r = 0.77 | |
| host DIME score (112 shows) | r = 0.41 (Spearman 0.42) | r = 0.43 (0.45) |
| right − left gap (side label) | d = +0.69, p = 1e-5 | d = +0.78, p ≈ 1e-6 |
| correlation with LLM side label | r = 0.33 | r ≈ 0.36 |

Top of the scale: Hood Politics with Prop, UNGOVERNED, Graham Allen, Louder with Crowder, Bannon's War Room;
bottom: Letters from an American, Hope for America, Antiwar News, The President's Inbox, The Dig, Jacobin Radio.
Same ends as the current measure.

## Audience models with mfte_z as exposure (corpus listeners; respondent-level r with dir_z exposure = 0.92)

LLM-density frame, 49 shows / 456 listeners (unframed 54 / 535 in brackets); CR2 / wild p:

| outcome | mfte_z alone | mfte_z ǀ populism | mfte_z ǀ intensity | intensity ǀ mfte_z |
|---|---|---|---|---|
| election distrust | +0.182 (.004/.009) [+0.184 (.004/.005)] | +0.182 (.003/.009) | +0.057 (.22/.25) [+0.098 (.07/.10)] | +0.282 (<.001) [+0.148 (.11/.003)] |
| cynicism 8-item | +0.138 (.020/.037) [+0.158 (.025/.039)] | +0.138 (.022/.052) | +0.085 (.12/.10) [+0.090 (.08/.09)] | +0.175 (.026/.027) [+0.154 (.008/.013)] |
| education (SD) | −0.230 (.041/.016) [−0.233 (.023/.009)] | −0.227 (.040/.016) | −0.082 (.45/.39) [−0.116 (.32/.28)] | −0.345 (.027/.047) [−0.209 (.12/.05)] |
| exclusivity | +0.054 (.16/.14) [+0.082 (.08/.09)] | +0.054 (.14/.11) | +0.011 (n.s.) [−0.010] | +0.121 (.018) [+0.178 (.005/.003)] |

Same pattern as with dir_z, about 80% of the size: election distrust, cynicism and education composition hold;
exclusivity no longer reaches significance; populist vocabulary changes nothing; ideological intensity absorbs the
associations.

## Decision

`mfte_z` becomes the address measure of record for the paper (published instrument, no word list of ours);
`dir_z` is reported in the appendix as the discovery instrument, with the 0.93 correlation. The 10 corpus shows
that lacked a show-level score are on the MFTE scale (see the 204-show validation appended below), so the corpus
can be reported as 204 shows with the measure. Pre-registered results are unchanged: they were run and are
reported with dir_z as frozen.

## 204-show validation (all transcribed corpus shows, including the ten previously without a score)

r(mfte_z, DIME) = +0.44 (Spearman 0.46, n = 120 with a host score); right − left gap d = +0.74 (73 vs 131 shows,
p = 1.5e-6); AUC for side from mfte_z alone 0.71. The ten previously unscored shows fall across the scale (Real
Baron +1.35, Lars Larson +0.82, Mel K +0.41, SGT Report +0.40, Nobody Knows Anything −0.06, Bulwark Takes −0.12,
Majority Report −0.19, Brian Tyler Cohen −0.49, The Globalist −1.20, Washington Today −1.32), consistent with their
side labels. The 204 → 194 gap is closed for the measure of record.
