# Findings (v2, 15 September 2026)

Corpus-only scope. Coefficients are in standard deviations of the outcome (standardised on the full survey) per
reference-scale standard deviation of address; p-values are CR2 with Satterthwaite df and wild cluster bootstrap,
in that order. Items marked pending depend on the MFTE rebuild or the checks queued behind it.

## Finding 1. Left and right political podcasts differ in how they address the listener, not in hostility

Across the 194 corpus shows, listener-directed address (`dir_z`) correlates 0.43 with the host's DIME score
(Spearman 0.45, 112 shows), separates the sides by 0.78 standard deviations (p ≈ 1e-6), and classifies a show's
side with an area under the curve of 0.80. Right-leaning shows talk at their listeners more: more imperatives,
more second person. The hostility measures that the outrage literature would predict to separate the sides do
not: the insult rate correlates 0.00 with DIME, and right-leaning shows use *fewer* emphatics (r = −0.29).
Populist vocabulary (Rooduijn & Pauwels rate) and anti-media references are also unrelated to side (r = −0.04 and
−0.06). The three ideology anchors agree with one another: the LLM side label, the DIME score and the party
identification of a show's own survey listeners.

The result does not depend on our instrument. Rebuilt from the published MFTE tagger's imperative and
second-person tags on the full corpus, the measure correlates 0.93 with the discovery version, 0.41 with DIME,
and separates the sides by 0.69 SD (p = 1e-5), with the same shows at both ends of the scale.

## Finding 2. Address is a trait of the show, not a reaction to events

In a within-show event study on 18,633 dated episodes, address in the 70 days after each of eight political
shocks (the 2020 and 2024 elections, 6 January, Dobbs, the 2022 midterm, the July 2024 assassination attempt,
Biden's withdrawal, the 2025 inauguration) is indistinguishable from the 70 days before, and the observed shifts
fall inside the distribution of placebo dates. Within-show address is flat across the 75 topics of a topic model.
Splitting imperatives into epistemic, attention and action verbs, and second person into guest-directed,
declarative, deontic and audience-plural uses, does not change the ranking of shows.

## Finding 3. The pre-registered confirmatory tests failed

Two tests were frozen in advance on corpus listeners and both failed on their own criteria: the free-text
democracy-item test, and the within-frame test on 150 respondents who named a corpus show but were not in the
discovery sample (address → cynicism b = −0.055, one-sided p .57, 10 shows). A third pre-registered test on ten
chart-absent shows also failed (b = −0.001) and is reported in the disclosure section although those shows lie
outside this paper's frame. Everything in Findings 4 and 5 is therefore exploratory.

## Finding 4. Listeners of address-heavy shows are more distrustful of elections, more cynical, less educated, and more likely to have left institutional news

Among the 535 survey respondents who named a corpus show (54 shows), with party, attention, age, education,
right-wing platform use and show lean controlled:

| outcome | all corpus listeners (54 shows, n = 535) | political-content frame (49 shows, n = 456) |
|---|---|---|
| election distrust | +0.22 (.004 / .011) | +0.22 (.004 / .02) |
| institutional cynicism (8 items) | +0.21 (.020 / .02) | +0.18 (.023 / .04) |
| respondent education (SD) | −0.30 (.004 / .002) | −0.30 (.006 / .003) |
| exclusivity (no institutional source) | +0.13 (.043 / .021) | +0.09 (.054 / .053) |

The political-content frame (shows whose passages the ideology instrument marks political at least 64% of the
time, the corpus 10th percentile) changes nothing: the election-distrust coefficient is +0.22 at every cut-off
from the 5th to the 25th percentile, and leave-one-show-out ranges are narrow. The education result is the same
in the discovery sample and in the later-matched pocket, and it is sharper among respondents who named the show
as their first source. None of these associations is explained by populist vocabulary: adding the Rooduijn &
Pauwels rate leaves each coefficient essentially unchanged (election distrust +0.22, cynicism +0.16 to +0.18,
education −0.28, exclusivity +0.08 to +0.11), and populism predicts none of the outcomes on its own.

## Finding 5. Address and ideological intensity cannot be separated in this corpus

Address-heavy shows are also ideologically intense shows: the share of a show's political passages that the
instrument labels anything other than "moderate" correlates +0.44 with address across shows. Intensity predicts
every outcome in Finding 4 at least as strongly as address does (election distrust +0.32 in the frame, cynicism
+0.23, education −0.40, exclusivity +0.13), and when both are in the model address retains a marginal
coefficient on cynicism only:

| outcome, frame | address ǀ intensity | intensity ǀ address |
|---|---|---|
| election distrust | +0.06 (.37 / .38) | +0.28 (.001 / .001) |
| cynicism | +0.14 (.07 / .04) | +0.15 (.046 / .032) |
| education | −0.12 (.22 / .24) | −0.32 (.020 / .022) |
| exclusivity | +0.05 (.35 / .33) | +0.10 (.10 / .13) |

Unframed the attenuation is milder (election distrust +0.12, p .13; education −0.18, p .10) and neither property
clearly dominates. With 54 shows the design cannot tell a style effect from a content effect, nor either from
selection: what is established is that the audiences of address-heavy, ideologically intense political podcasts
are more election-distrustful, more cynical, less educated and more likely to have abandoned institutional news,
and that populist rhetoric is not what distinguishes those shows.

*(Pending: a within-show check of whether the instrument reads address-heavy passages as more intense, which
would make intensity partly a second measurement of address rather than a rival; and the Year-2 pre-registration
with intensity as the named rival.)*

## What the paper does and does not claim

It claims a measurement result (Finding 1), a stability result (Finding 2), and a set of audience associations
(Finding 4) that survive the frame, the cut-off, leave-one-show-out and the populism rival but not the intensity
rival (Finding 5). It does not claim a causal effect of address on attitudes: the three pre-registered tests that
would have supported one failed, and the audience associations are consistent with selection into shows as much
as with influence by them.
