# What organizes the embedding space, if not ideology?

The dominant structure in the corrected chunk-embedding space (target=500,
`centered` anisotropy variant, N=40,242 chunks) is **analytical-vs-reactive
register**, expressed across several different topical domains — not any
single topic, and not a format or length artifact. Every one of the five
PCs named in this analysis splits the same way: a high pole that is
analytical, specialist, and theory-laden (legal doctrine, IR strategy,
political-science frameworks, intellectual history), against a low pole
that is reactive, event-driven, and personality- or news-focused (horse-race
gossip, daily war-briefing recitation, Trump-scandal reporting, short
news-brief clips). PC3 (foreign-policy/war news-briefing vs. domestic
political-process discourse) and PC5 (legal/constitutional case analysis
vs. horse-race campaign punditry) are the cleanest, most broadly distributed
instances of this split — both are genuinely spread across a dozen-plus
shows apiece, not a handful of outliers.

This register axis is the mechanism that explains why ideology is only a
minor signal in this space (the established prior: ~2% of variance, R²=0.21
against DIME CFscore). A wonky-analytical host on the left and a
wonky-analytical host on the right sit close together on this axis — both
score high, both register as "the same kind of show" — while a
reactive-tabloid host of either lean sits near the opposite lean's
reactive-tabloid counterpart. Register cross-cuts ideology instead of
tracking it, so the dimension carrying the most variance in this space is
nearly orthogonal to the one DIME CFscore measures. That is exactly why a
strong ideology signal never surfaces among the top principal components:
it isn't buried, it's riding on a different, much smaller axis entirely
(PC22).

Two of the five dimensions are better described as **detectors** than
graded axes. PC1 identifies a tight five-show specialist-IR-interview
clique (Russian Roulette, Foreign Affairs Interview, Glenn Diesen, Inside
Story, President's Inbox) with high confidence — but its negative pole is
residual heterogeneity, not a coherent second category, so it should be
read as "is this one of that clique," not as a bipolar dimension. PC4
(heterodox intellectual history vs. energy/climate trade press) carries a
double confound and is downgraded to low-moderate confidence: its low pole
conflates topic (energy/climate) with format (short news-brief length,
the strongest length correlation of the top five PCs) and, specifically at
that pole, with a single show's recurring ad-read (POLITICO Energy /
Chevron) — three entangled effects that can't be cleanly separated here.
PC2 (theoretical/academic geopolitical-economic analysis vs. Trump
legal-scandal news) reads cleanly but is the most show-concentrated of the
five (η²=0.415) and should be treated as a real but narrow axis rather than
a general-purpose dimension. PC6–PC10 were not blind-read and are left
honestly unresolved — the quantitative signal alone (no feature correlation
above |r|=0.3, η² in the 0.18–0.26 range) can't distinguish a topic story
from further show-identity fragmentation without reading the extremes.

The magnitude gap to ideology is real but not uniform across the five. PC1
sits a genuine order of magnitude above the ideology axis (9.39% of total
embedding variance vs. PC22's 0.87% — a 10.8x ratio). PC2 through PC5 are
smaller gaps that do not support "order of magnitude": PC2 is 8.5x, PC3 is
7.2x, PC4 is 5.5x, and PC5 is 5.0x the variance share of the ideology axis.
All five nonetheless sit 15–17 PC-ranks above PC22 and carry several times
its variance. Format, length, publication timing, and ad/sponsor
contamination were all explicitly tested and ruled out as drivers of this
structure: no derived metadata feature crosses |r|=0.3 against any top-10
PC (Step 1), and a dedicated audit found ad-boilerplate contamination
corpus-wide is minor (0.35% of chunks; 0% in the PC-extreme tails at a
50%-coverage bar) and uncorrelated with ideology (r=−0.086) — the register
finding is a genuine content signal, not a confound in disguise.
