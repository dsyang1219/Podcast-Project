# Pairwise position scaling: sandbox pilot

Model `gpt-5.4-mini-2026-03-17`; 12,120 comparisons; periods {'P1': (2019, 2021), 'P2': (2024, 2026)}; 30 passages and 20 matchups per show per period, both orderings.

- Order consistency (verdict unchanged when A/B are swapped): 72.7% of 6,060 matchups; tie rate 5.0%.

**Monotonicity against the 7-point labels of the two passages** (P(A judged further right) by the gap in 7-point codes, A minus B; a working scale rises left to right):

| gap | P(A right) | n |
|---|---|---|
| -4 | 0.15 | 820 |
| -3 | 0.28 | 1542 |
| -2 | 0.35 | 1030 |
| -1 | 0.41 | 1032 |
| +0 | 0.55 | 2661 |
| +1 | 0.69 | 1024 |
| +2 | 0.76 | 1045 |
| +3 | 0.82 | 1543 |
| +4 | 0.91 | 822 |

## Period P1 (2019–2021): 99 shows

- Bootstrap SE of θ: median 0.26 (θ SD 0.86); reliability (1 − median SE²/var) ≈ 0.91.
- vs current direction score `side` (n=99): Spearman 0.78, Pearson 0.80.
- |position| vs one-sidedness `inten` (n=99): Spearman 0.50 (should be well below 1 if position ≠ emphasis).
- vs hosts' DIME CFscore (n=58): Spearman 0.55.
- Known groups: mean z for L shows -0.45 (n=72), R shows +1.21 (n=27); shows on the wrong side of 0: 20.
- Brookings matched shows (n=17): mean z by Brookings label {'Moderate': -0.35, 'More Conservative': 0.84, 'More Liberal': -0.48}.

Most left / most right (z):
Rev Left Radio (-2.90), Deadline: White House (-2.06), The Beat with Ari Melber (-1.80), The Nation Podcasts (-1.63), The Dig (-1.57)

Renegade Talk Radio (+3.37), Timothy Gordon Rules for Retrogrades Podcast (+2.28), The Breitbart News Daily Podcast (+1.92), Brian Kilmeade Show (+1.91), The Benny Show (+1.82)

## Period P2 (2024–2026): 204 shows

- Bootstrap SE of θ: median 0.25 (θ SD 0.73); reliability (1 − median SE²/var) ≈ 0.89.
- vs current direction score `side` (n=194): Spearman 0.76, Pearson 0.79.
- |position| vs one-sidedness `inten` (n=194): Spearman 0.39 (should be well below 1 if position ≠ emphasis).
- vs hosts' DIME CFscore (n=120): Spearman 0.48.
- Known groups: mean z for L shows -0.53 (n=125), R shows +0.97 (n=69); shows on the wrong side of 0: 32.
- Brookings matched shows (n=28): mean z by Brookings label {'Moderate': -0.3, 'More Conservative': 0.9, 'More Liberal': -0.55, 'Unknown': -1.64}.

Most left / most right (z):
The PoliticsGirl Podcast (-2.69), The Dig (-2.33), No Gods No Mayors (-2.10), Rev Left Radio (-2.02), Antiwar News With Dave DeCamp (-1.92)

The Hugh Hewitt Show: Highly Concentrated (+2.95), Life, Liberty & Levin Podcast (+2.52), John Solomon Reports (+2.34), The 47 Morning Update with Ben Ferguson (+2.12), The Fox News Rundown (+1.98)

## Across periods

- Shows in both periods: 99; Spearman of positions P1 vs P2: 0.69 (each period scaled relative to itself; no external bridge yet).

API cost so far: $0.00 (this fit run only counts fit-time usage; see logs for scoring cost).
