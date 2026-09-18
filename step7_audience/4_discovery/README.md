# 7.4 — the discovery sample under strict inference

The discovery sample is the 385 strict-match corpus listeners. These scripts estimate what address predicts
among them, with the small-sample cluster corrections the paper reports everywhere.

| script | role |
|---|---|
| `verify.py` | The verification table: every outcome regressed on address with conventional clustered, wild-bootstrap and show-permutation p-values, on both the strict and the expanded match. |
| `cr2.py` | The headline discovery estimates (favourability gap, cynicism composite, exclusivity) under CR1, CR2 with Satterthwaite df, and the wild bootstrap. Reproduces the recorded numbers exactly (cynicism +0.219, CR2 p = .0061, df 7.4). |
| `decomp.py` | Which register measures carry the association with cynicism, with ideology controls and interaction tests. |
| `excl_strong.py` | Exclusivity (naming no institutional source) on address, with show-level and listener-level checks. |
| `families.py` | Benjamini-Hochberg survival of the discovery results under alternative outcome-family definitions (disclosure G.1). |

All of them use `cr_all()` and `design_np()` from `common.py`; none writes an input that another sub-step needs.
