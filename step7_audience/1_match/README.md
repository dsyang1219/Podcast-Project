# 7.1 — matching respondents to shows

Builds everything the later sub-steps load through `common.py`.

```
data/external/kettering/  --model.py-->  inputs/kett.pkl            (survey + derived party, attention, age, education, weights)
kett.pkl (open-ended items) --build.py-->  inputs/verbatims_long.csv  (one row per named source)
                                           inputs/match_strict.csv    (exact show-title matches)
match_strict + host names   --expand.py--> inputs/match_expanded.csv  (adds host-name matches, with the excluded-host audit)
verbatims_long              --diet.py-->   news-diet segments          (institution / person-branded / platform / podcast / local)
```

| script | role |
|---|---|
| `model.py` | Entry point for the whole arm: reads the Kettering microdata, derives the controls, writes `kett.pkl`, and fits the first respondent-level model of the favourability gap. |
| `build.py` | Splits every open-ended "where do you get news" answer into named sources and matches them to corpus show titles exactly. The strict match (385 respondents) is the sample of record. |
| `expand.py` | Adds matches on host names (544 respondents). Reported only as a sensitivity: it does not reproduce the strict-match results. |
| `diet.py` | Classifies every named source and builds the diet segments (podcast-only, podcast + mainstream, platform-only, mainstream-only) used by the population finding. |
