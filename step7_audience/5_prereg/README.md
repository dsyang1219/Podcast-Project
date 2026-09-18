# 7.5 — the three pre-registered tests

Each test was frozen in `handoff/prereg/` (text plus SHA-256, alias tables, the reference scale) before the data
it needed were touched, and each script executes its pre-registration as written. All three failed on their own
criteria and are reported as failures in the paper.

```
handoff/prereg/prereg_Q28.md                 --q28.py------------------>  Q28 free-text test
data/transcripts/HOLDOUT_*  + dirz_reference_scale.csv
      --score_holdout_final.py-->  inputs/holdout_show_dirz_FINAL.csv --holdout_test.py-->  out-of-frame test (10 chart-absent shows)
corpus transcripts of The Majority Report
      --score_mr.py------------->  inputs/mr_show_dirz.csv  + inputs/WITHINFRAME_SCORES.csv --withinframe_test.py--> within-frame test (150 fresh listeners)
```

| script | role |
|---|---|
| `q28.py` | Pre-registered dictionaries applied to the free-text democracy item. |
| `score_holdout_final.py` | Scores the out-of-frame transcripts with the pinned corpus pipeline: 750-character passages from `step4_ideology/build_scoring_chunks.py`, the `remeasure2` features verbatim, word-count-weighted show rates, z-scores against the 194-show reference. Touches no survey outcome. |
| `holdout_test.py` | Runs tests (i) to (viii), the secondary outcome, H2, robustness and leave-one-show-out. Prints the SHA-256 fingerprints of the frozen files it used at the top of its log. |
| `score_mr.py` | The same pinned scoring for The Majority Report, the one within-frame show without a corpus score. |
| `withinframe_test.py` | The within-frame test on 150 respondents who were not in the discovery sample. |

The recorded runs are `handoff/results/holdout_RUN_2026-09-03T0812Z.log` and
`handoff/results/withinframe_RUN_2026-09-03T0829Z.log`; the write-ups are the `*_RESULTS.md` files beside them.
The scripts change into `inputs/` while they run and log to `outputs/`.
