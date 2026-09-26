# Kettering Y1 — results of the requested tests (Sept 2, 2026)

Scripts: build.py, expand.py, tests.py, diag.py, rest.py in the session scratchpad.
Exposure = mean dir_z (directive register z-score) across the corpus shows a respondent named.
Baseline model: ap_gap ~ dir_z + party ID + political attention + age + education.

## 0. Baseline reproduction — close but not exact

    reported earlier   b=+0.657  t=+3.16  p=.0017
    my reproduction    b=+0.590  t=+2.94  p=.0035   n=384

Same sign and magnitude; small differences almost certainly from party-ID coding or control
construction, which I rebuilt from the codebook rather than recovering the original script.
One difference matters: in my spec share_right is SIGNIFICANT (p=.043), not marginal (p=.073).
So "register survives, ideology fades" is weaker than reported — both survive, register more so.

## 1. HOST-NAME EXPANSION — THE HEADLINE RESULT, AND IT IS NEGATIVE

Audited alias table: 31 hosts whose name IS the podcast brand, 43 verbatim strings, adding 159
respondents (385 -> 544, 51 -> 59 shows). 14 hosts excluded with documented reasons (TV simulcast:
Hayes, Reid, Acosta, Melber, Wallace, Phang, Cuomo, Dickerson, Failla; person != show: Ted Cruz,
Michael Cohen, Galloway, Baker, Vance). Tables saved: alias_table_hosts.csv, alias_excluded.csv.

    strict                n=384   b=+0.590   p=.0035
    EXPANDED              n=543   b=+0.100   p=.346    <-- effect gone
    expanded, HCR removed n=494   b=+0.434   p=.023

**A defensible expansion of the matching rule eliminates the finding.**

### Why: one show is a massive leverage point
Letters from an American (Heather Cox Richardson): dir_z = **-2.80** against a corpus SD of 0.82 —
a 3.4-sigma outlier — with 55 matched listeners whose mean AP gap (6.29) is ABOVE the sample mean
(5.45). Low directive score, high polarization: it single-handedly flattens the slope.

Leave-one-SHOW-out on the expanded sample:
    drop Letters from an American   -> b=+0.434  p=.023
    drop any other show (n>=15)     -> b between +0.036 and +0.149, all n.s.

Note also that HCR's dominant medium is a Substack newsletter that shares the podcast's name, so
many of those 55 may not be podcast listeners at all. Either way the point stands: the estimate
is hostage to which shows happen to land in the sample.

### The earlier leave-one-out was the wrong LOO
The previously reported LOO interval [+0.601, +0.704] was leave-one-RESPONDENT-out. Exposure is
assigned at SHOW level, so leave-one-SHOW-out is the correct check — and it is wildly unstable.
Respondent-level LOO badly understates uncertainty under clustered exposure.

## 2. Inference is not the problem — leverage is

    naive OLS        b=+0.590  se=.201  p=.0035
    show-clustered   b=+0.590  se=.177  p=.0009   (51 clusters)

Clustering on show slightly TIGHTENS it. So the SEs were never the weak point.

## 3. Weighting and sample type

    weighted (WEIGHT)   strict n=384  b=+0.806  t=+4.04  p=.0001
                        expanded      b=+0.231  t=+2.14  p=.033
    Gallup Panel only   strict n=262  b=+0.560  p=.019    <-- survives in the probability half
    opt-in only         strict n=122  b=+0.600  p=.137    (same b, underpowered)
    Gallup Panel only   expanded n=369 b=+0.036 p=.763

Good news: on the strict sample the effect holds in the probability subsample with essentially the
same coefficient as the opt-in half, which answers the non-probability objection. The paper should
report the WEIGHTED estimate or justify not doing so — weighting materially changes the number.

## 4. Q20 PLATFORM CONTROLS — this genuinely strengthens the result

Right-wing platform index = regular/infrequent use of Truth Social, Rumble, Parler, Gab.

    baseline                       b=+0.590  p=.0009
    + rw platform index            b=+0.484  p=.0022
    + rw platform + share_right    b=+0.414  p=.0141

The association is not membership in the right-wing platform ecosystem. This is a better confound
control than share_right alone and should go in the paper.

## 5. TWENTY-THREE OTHER OUTCOMES — ALL NULL AFTER CORRECTION

Show-clustered SEs, strict sample. Two of 23 reach p<.05 (expected by chance: 1.15), and
**nothing survives Benjamini-Hochberg** (smallest q = 0.129).

    born-again (Q46)              b=-0.083  p=.0071   q=.129
    perspectives diversity (Q38)  b=-0.131  p=.0112   q=.129
    everything else               p = .10 to .998

Specifically dead:
  - MOBILIZATION. Q24A-F perceived effectiveness of voting/contacting/protest/campaigning/
    donating/town halls: all null (p=.10 to .93). Q13 volunteering p=.72, Q10 community events
    p=.53, Q47 registration p=.12. The imperative-to-action hypothesis is NOT supported.
    (Correction to my earlier note: Q24 measures perceived EFFECTIVENESS, not behavior.
    The behavioral items are Q13/Q10/Q47.)
  - PARASOCIAL. Q4 loneliness b=-0.009, p=.88 — a precise null, not an underpowered one.
    The withdrawn parasocial claim gets no support here.
  - RELIGIOSITY VALIDATION. Q44 attendance p=.16, Q46 born-again p=.007 but NEGATIVE and
    non-surviving. Register does not validate against listener religiosity.
  - DEMOCRATIC NORMS. Q32F protest rights (Gallup's own item) p=.47; Q32B/D/G all null;
    Q36K political violence, Q31 accountability, Q29 democracy health all null.

## What this means

The discriminant story survives and is now much better documented: across 23 additional outcomes
the directive-exposure association appears ONLY on the party favorability gap. That is real
specificity and argues against a response-style artifact.

But the cost is severe. The single surviving result is fragile to a defensible change in the
matching rule, and the fragility traces to one high-leverage show among only ~51-59 shows. With
23 further tests all null, the honest reading is that this is one significant finding from a
large search, resting on a small and unstable show-level sample.

Recommendations:
1. Report the expansion. Hiding it would be the forking-paths problem in its purest form.
   "b=+0.590 strict, +0.100 expanded, +0.434 expanded-minus-one-outlier" is the honest interval.
2. Use leave-one-SHOW-out as the robustness standard everywhere, not leave-one-respondent-out.
3. Keep the Q20 platform control — it is a real improvement.
4. Report weighted estimates.
5. This strengthens rather than weakens the case for the Y2 panel request: only a within-person
   design escapes the show-leverage problem, because exposure change is identified within
   respondent rather than across 51 shows.

---

# AUDIT ROUND 2 — corrections to the above

## Data-integrity check: clean
85 records have MODE 3/4/5 (undocumented; codebook lists only 1=Web, 2=Mail). In those rows AGE is
a category (1-5) not years, Q24C holds weight-like garbage (up to 46,339), and WEIGHT is NaN.
**Zero of them are in the matched sample** (strict or expanded). AGE in the matched sample runs
21-88. Missing code is -98 and was correctly filtered. The nulls are not a coding artifact.

## CORRECTION 1 — the analytic clustered SE is anti-conservative

    analytic cluster-robust p = .0009
    WILD CLUSTER BOOTSTRAP p  = .047     (2000 Rademacher reps, 51 clusters)
    bootstrap t distribution sd = 1.53   (analytic inference assumes ~1.0)

Cluster sizes are severely unbalanced: largest cluster 73 respondents, top 3 clusters = 52.6% of
the sample, and 24 of 51 clusters have n = 1. That is exactly the regime where analytic
cluster-robust SEs fail and the wild bootstrap is required (Cameron, Gelbach & Miller 2008).

**The honest p-value on the headline result is ~.05, not .0017 and not .0009.**

## CORRECTION 2 — my "show-level r = +0.062" was misleading

That was the UNWEIGHTED correlation across 51 shows, and 24 of those shows have a single listener,
so their show-mean residual is one person's noise. Precision-weighted by listener count (correct),
with controls residualized, the coefficient is stable:

    shows n>=1  (51 shows)  b=+0.432  permutation p=.069
    shows n>=2  (27 shows)  b=+0.532  permutation p=.036
    shows n>=3  (21 shows)  b=+0.531  permutation p=.024
    shows n>=5  (14 shows)  b=+0.485  permutation p=.074

b sits at +0.43 to +0.53 regardless of threshold. The show-level signal is real; it is just
marginal. This is a fully cluster-immune test and it agrees with the wild bootstrap.

## Popularity / audience-size controls — the effect survives cleanly

dir_z is near-orthogonal to popularity: r = .037 with in-sample listener count, .035 with charted
share, .115 with corpus hours.

    baseline                      b=+0.590  p=.0009
    + log(in-sample listeners)    b=+0.610  p=.0009
    + charted share               b=+0.599  p=.0008
    + best chart rank             b=+0.610  p=.0007
    + log(corpus hours)           b=+0.564  p=.0016
    + ALL popularity              b=+0.586  p=.0013
    + ALL popularity + share_right b=+0.438 p=.0152

Audience size is not driving it. Worth reporting — it closes an obvious referee question.

## The expansion result holds up, and is not merely outlier leverage

Show-level, precision-weighted, residualized, n>=2 shows:

    strict     27 shows  b=+0.532  perm p=.032   Huber b=+0.713
    expanded   35 shows  b=+0.099  perm p=.663   Huber b=+0.223

Huber regression downweights outliers, so this is not just Letters from an American exerting
leverage — the shows added by host-name matching genuinely do not fit the pattern.

## Revised bottom line

There IS a signal. It is not nothing, and my previous summary understated it.

  - Effect size b ~ +0.43 to +0.59, stable across popularity controls, platform controls,
    weighting, the probability subsample, and every show-level threshold.
  - Correct inference puts it at p ~ .02 to .07 rather than p ~ .001.
  - It does NOT survive a defensible broadening of the matching rule.

So: a marginal, specification-dependent association, reasonably robust to confounds but not to the
question of which respondents count as exposed. That is publishable as a carefully-hedged
association in a larger paper. It is not publishable as a standalone claim, and the previously
reported precision (p = .0017, LOO [+0.601, +0.704]) should not be repeated.

---

# ROUND 3 — exhaustive outcome scan (113 variables)

Every usable outcome in the survey (numeric, 2-12 levels, n>=300), not a hand-picked subset.
Controls: party ID, attention, age, education. Strict sample, n=384, 51 shows.

## Screen
    113 tests | p<.05 (analytic clustered): 20 | chance expects 5.7 | BH survivors (analytic): 9

## After correcting the anti-conservative SE (1.53x t-deflation, = wild bootstrap)
    BH across all 113:                     0 survive q<.05   (1 at q<.10: Q21 social-media time)
    BH across 98 non-media-use outcomes:   0 survive         min q = 0.34

**Under correct inference AND correct correction for the search, no single outcome survives.**
That is the headline and it should be stated first.

## But the nominal hits are not scattered — they cluster on one construct

Analytic hits (cleaned single items, wild-bootstrap p, then with ideology controls = share_right +
right-wing platform index, then show-level permutation, then the host-name-expanded sample):

    item                                    b     WILD p | +ideo b  +ideo WILD | show perm p | EXPANDED wild p
    Q33E elections administered well     -.336   .0013  |  -.266    .0113      |   .021      |   .141
    Q34B know how to reach officials     -.205   .0033  |  -.257    .0027      |   .029      |   .106
    Q35E laws uphold freedom/justice     -.275   .0080  |  -.299    .0020      |   .026      |   .924
    Q33H freedom of press working        -.255   .0260  |  -.227    .0200      |   .161      |   .085
    Q35C govt includes people like me    -.179   .0313  |  -.179    .0813      |   .059      |   .311
    Q32C assume fraud when surprised     +.287   .0213  |  +.259    .1467      |   .032      |   .373
    Q12D unsure how to get involved      -.095   .0240  |  -.094    .0327      |   .135      |   .147

All seven point the same way: directive exposure -> lower institutional confidence, lower
political efficacy, more election suspicion. Noise does not usually produce seven same-direction
hits on one construct.

## Composite: INSTITUTIONAL CYNICISM (reverse-scored Q35E, Q35C, Q34B, Q33H)

    Cronbach alpha = 0.52, mean inter-item r = .22   <-- WEAK. Four loosely related items, not a scale.

    strict, base controls        b=+.207   WILD p=.0027
    + ideology controls          b=+.219   WILD p=.0007    <-- gets STRONGER, not weaker
    show-level permutation       b=+.167   p=.0035  (27 shows)
    leave-one-show-out (n>=15)   b in [+.196, +.231], every drop p<.002
    Gallup Panel only (n=259)    b=+.213   WILD p=.025
    weighted                     b=+.222   cluster p=.0006
    effect size                  +0.24 SD cynicism per 1 SD directive register
    EXPANDED sample              b=+.078   WILD p=.165     <-- fails, again

## Three things that must be said about this

1. THE COMPOSITE IS POST-HOC. It was assembled from the items that came up significant. Its
   p-value is therefore not a valid test of anything — it is the Texas sharpshooter. The
   robustness table above shows the pattern is STABLE (not driven by one show, present in the
   probability half, survives weighting), which is worth knowing; it does not show the pattern is
   REAL in the sense a pre-specified test would.

2. THE SUPPRESSION EFFECT IS THE INTERESTING PART. Ideology controls strengthen the coefficient.
   Plausible mechanism: in 2025 with Republicans in government, right-leaning respondents rate
   responsiveness / justice / access HIGHER; directive shows skew right (r=.35); so the ideology
   channel drags the raw directive->cynicism association toward zero, and controlling for it
   reveals it. That is the OPPOSITE of "it's just conservative media." Worth stating, with the
   caveat that it is one plausible story, not a tested one.

3. THE EXPANSION HAS NOW KILLED THREE FINDINGS (AP gap, election distrust, institutional
   cynicism). That is no longer "one outlier show." The host-name-matched respondents are
   systematically different. One substantive hypothesis: people who write "benny johnson" or
   "brian tyler cohen" rather than a show title may be consuming YouTube/TikTok clips, not podcast
   episodes — a different exposure modality than the full transcripts the register measure is
   built on. If so the strict match is the CORRECT operationalization, not merely the convenient
   one. That is a hypothesis; the fact is that "who counts as exposed" is the single unresolved
   decision and it must be pre-registered, not chosen.

## What this is, honestly

Not a finding from Y1. A well-characterized, ideology-independent, show-level-robust PATTERN that
does not survive correction for the search that found it. That is exactly what a pre-registration
candidate looks like:

    H1 (pre-registered for Kettering Y2): Among respondents naming a corpus show by exact title,
    directive register exposure predicts lower institutional confidence / political efficacy
    (Q34B, Q35E, Q33E, Q33H), controlling party ID, attention, age, education, show ideology, and
    right-wing platform use. One test, one pre-specified composite, one pre-specified match rule.

If it replicates on fresh data under those constraints, that is a paper. It is a stronger
candidate than the AP gap: bigger effect, cleaner confound story, tighter theory (directive
address -> learned passivity / cynicism about ordinary channels is a coherent mechanism).

## Y1 is now burned for confirmatory use

Across this session: 23 + 113 outcomes, multiple composites, multiple match rules, multiple
control sets. Any further "discovery" in Y1 is uninterpretable. Everything from here is Y2 —
which makes the Kettering data request the whole ballgame.

---

# ROUND 4 — "are you sure they don't survive?" Real bootstrap on the whole family.

## Correction to my earlier claim
I applied a blanket 1.53x t-deflation to all 113 items, taken from the AP-gap model. Running the
REAL wild cluster bootstrap (1000 reps each) on all 47 democratic-attitude items shows the actual
inflation is milder: median bootstrap t-sd = **1.20** (range 0.98-1.65). So my earlier "nothing
survives" used an approximation that was too harsh for most items. The user was right to push.

## Real wild-bootstrap p-values, top of the democratic-attitude family (47 items)
    Q33E elections administered well   t=-4.47  wild p=.0020
    Q34B know how to reach officials   t=-3.72  wild p=.0050
    Q35E laws uphold freedom/justice   t=-3.30  wild p=.0080
    Q32C assume fraud when surprised   t=+3.88  wild p=.0170
    Q33H freedom of press working      t=-2.74  wild p=.0270
    Q32A right to vote even if radical t=-2.61  wild p=.0420
    Q35C govt includes people like me  t=-2.95  wild p=.0490

## Survival under BH depends ENTIRELY on the family you correct across
    family                                              k    rank-1 threshold  survivors        min q
    A. all 113 screened                                113   .00044            none*            --
    B. 98 non-media-use outcomes                        98   .00051            NONE             .196
    C. all democratic-attitude items (Q22,23,29-36)     47   .00106            NONE             .094
    F. democracy batteries (Q29,31,32-36)               42   .00119            NONE             .084
    D. institutional confidence + efficacy (Q33-35)     19   .00263            Q33E, Q34B       .038
    E. institutional confidence only (Q33)              10   .00500            Q33E             .020
    (*A: only Q21 social-media time survives, a media-use variable not an outcome)

## Why they don't survive — the actual arithmetic
BH's first-rank threshold is .05/k. Q33E's real p is .0020. That clears any family of k <= 25 and
fails any family of k >= 26. The most defensible A-PRIORI family — defined by question block, not
by result — is C (all 47 democratic-attitude items). Under C, Q33E fails by about 2x
(.0020 vs .00106) and the whole family bottoms out at q = .094.

The 19-item family D where two items survive is NOT the family I would have named before seeing
results; I would have named C. Choosing D now because it works is forking paths.

## What this means, precisely
Not "there's nothing." Not "it survives." It sits AT the boundary, and which side of the boundary
it lands on is decided by a denominator that was never fixed in advance. That is the entire
argument for pre-registration in one table: on Y2, with ONE pre-specified test (k=1, threshold
.05), Q33E at p ~ .002 is not close — it is a clean, unambiguous result. The only thing standing
between this pattern and a publishable finding is having named the test before looking.

---

# ROUND 5 — structurally different tests, and the first CROSS-SURVEY REPLICATION

Instead of scanning more outcomes (burned), three things that are different in kind.

## A. Which register dimension carries institutional cynicism? (14 measures, one outcome, ideology controls, wild p)
    imperatives     b=+.118  p=.019
    out_group_pron  b=+.100  p=.037
    america         b=-.080  p=.045
    second_person   b=+.119  p=.053
    insult .46, profanity .54, hedging .58, assertion .65, religious .78, fillers .99  -> all null
    [dir_z composite b=+.219 p=.001 — stronger than any component, as a real aggregate should be]
Nothing survives BH across 14 (min q=.185), but the loading is exactly the directive/adversarial-
address family and NOT the incivility family. Same story as the DIME validation: it is how the host
addresses the listener, not how nasty the host is.

## B. Party heterogeneity — this one is NOT a subgroup artifact
    Democrats (D+lean)     n=267  b=+.164  p=.007
    Independents           n= 36  b=+.356  p=.008
    Republicans (R+lean)   n= 75  b=+.273  p=.0003
    interaction F p=.0003 (slopes differ; independents steepest) — but present in ALL THREE.
Contrast with the AP-gap result, which lived entirely in independents. This is a materially more
credible pattern.

## C. POPULATION CONTRAST — the corpus match is not needed for this, and it is not a boundary result
Full Kettering sample naming >=1 news source (n ~ 18,800), weighted, controls party/attention/age/edu.
Reference group = named only TV / print / web. Coefficients in SD units of the outcome.

    outcome                         named corpus show (385)     named any pol. podcast (1,337)
    leaders held accountable        -0.38  q<.001               -0.20  q<.001
    laws uphold freedom/justice     -0.34  q<.001               -0.13  q<.001
    govt includes people like me    -0.29  q<.001               -0.08  q=.05
    freedom of press working        -0.23  q<.001               -0.12  q=.008
    elections administered well     -0.21  q=.005               -0.14  q<.001
    democracy doing                 -0.19  q<.001               -0.00  ns
    radicals may protest            +0.23  q<.001               +0.21  q<.001   <-- Gallup's own published item, replicated
    assume fraud when surprised     -0.26  q<.001               -0.02  ns       <-- note: LESS suspicious overall
    political violence              -0.02  ns                   -0.06  ns       <-- discriminant null
    citizen power                   +0.03  ns                   +0.06  ns

Political-podcast news consumers are 0.2-0.4 SD more institutionally cynical than people who get
news only from TV/print/web, and the corpus (mid-tail political) shows are a stronger version of
the general podcast pattern. Every main item q<.01. This is descriptive — selection into podcasts is
obvious — but it is the contextual foundation the register finding sits inside, and it replicates
Gallup's own analysis of this dataset.

## PEW REPLICATION — lagged, independent panel, same pattern
ATP W150 (Jul-Aug 2024): "how often do you get news from podcasts" (1 Never - 4 Often), n ~ 10,000.
Linked by QKEY to trust items asked 2-8 MONTHS LATER. Weighted, controls party-ideology / age /
education / internet use. b = SD change in trust per one step of podcast-news frequency.

                            W155 (Sep 2024)            W165 (Mar 2025)
    trust national news     -0.066  p<.0001  n=9,147   -0.083  p<.0001  n=8,982
    trust local news        -0.067  p<.0001            -0.083  p<.0001
    trust social media      +0.140  p<.0001            +0.154  p<.0001
    trust friends/family    +0.011  p=.49              +0.024  p=.13     <-- discriminant null

Across the full 1-4 range that is ~0.2-0.25 SD less trust in news institutions — the same magnitude
as the Kettering corpus-show effect — with exposure measured BEFORE the outcome, in a different
probability panel, with a different instrument. And the discriminant is exactly right: informal
trust (friends/family) does not move; institutional trust falls; trust in the informal channel
(social media) rises. A general negativity / response-style artifact would not produce that shape.

## What this adds up to
The register-specific result (directive -> cynicism among corpus-show listeners) remains a Y2
pre-registration candidate. But the POPULATION-LEVEL claim — political-podcast news consumers hold
markedly lower institutional trust and efficacy than other news consumers, independent of party —
is now supported in two independent probability-panel datasets, one of them lagged, at n ~ 9,000-
18,800, with clean discriminant validity. That is a result you can write today.

Structure it suggests for the paper:
  1. Population level (Kettering + Pew): podcast news consumers are more institutionally cynical.
  2. Show level (corpus register x Kettering): within them, DIRECTIVE shows carry the association,
     not incivility — exploratory in Y1, pre-registered for Y2.
  3. Mechanism (DIME + register decomposition): imperatives / second person / out-group address.

## Online replication targets for (1), in priority order
  - Reuters Digital News Report 2023-2026: podcast news use + trust-in-news battery, ~2,000 US/yr,
    academic access on request. Four cumulative years.
  - Knight/Gallup American Views 2017-2020: trust in institutions + news sources, n ~ 20,000,
    address-based, "public release data" published.
  - ANES 2024: no podcast item (see below); radio-program battery only.

## PEW ROBUSTNESS — stricter exposure, larger effect (dose-response)
Exposure: W155 (Sep 2024) "most common way you get election news" = Podcasts (455 of 9,633).
Outcome: W165 (Mar 2025) trust battery, 6 months later. Weighted, same controls.

    trust national news     b=-0.340 SD  p<.0001   n=8,934
    trust local news        b=-0.242 SD  p=.0008
    trust friends/family    b=-0.044 SD  p=.57     <-- discriminant null, again
    trust social media      b=+0.323 SD  p<.0001

The stricter definition ("podcasts are my MAIN source") gives effects 3-4x larger than the
frequency measure, i.e. dose-response. Same discriminant shape. Same direction. Same panel,
different exposure item, different outcome wave.

## ANES 2024 — cannot carry a podcast test
Named-program batteries are TV (V241601-602), Spanish-language TV (V241603), radio (V241604).
No podcast item. A radio-battery test would answer "are talk-radio listeners cynical," which is a
different and long-established literature; not run.

## Knight/Gallup American Views — request target, not confirmed downloadable
"Public release data" is referenced on the report page but no respondent-level file surfaced.
Ask Knight directly; foundation, same play as Kettering.

## STATUS OF THE CROSS-SURVEY POPULATION CLAIM
"Political-podcast news consumers hold lower institutional trust / efficacy than other news
consumers, independent of party; informal trust unaffected; trust in informal channels higher."
    Kettering Y1   n~18,800   cross-sectional   0.2-0.4 SD   all q<.01   (democracy + trust items)
    Pew ATP        n~9,000    LAGGED 2-8 mo     0.2-0.34 SD  all p<.001  (trust-in-news items)
Two probability panels, two instruments, one lagged, consistent discriminant. Writable now.

---

# ROUND 6 — Q28 PRE-REGISTERED TEST: FAILED (primary), directionally consistent

Pre-registration written to handoff/prereg/prereg_Q28.md at 2026-09-02T19:15:13Z; analysis ran 2s later.
Q28 had never been examined against register before this. Failure criterion was stated in advance:
"M1 one-sided p >= .05 in Sample A -> H1 not supported by Q28."

## Sample A (385 corpus-matched; ideology controls + log words; wild cluster bootstrap)
    measure                 b (SD)   1-sided p (predicted dir)   2-sided p   n
    M1 out-group pronouns   +0.12    .125                         .247        198  <-- PRIMARY: FAIL
    M2 negation rate        +0.13    .097                         .206        343
    M3 VADER valence        +0.06    .862 (wrong direction)       .292        343
    M4 cynicism lexicon     +0.13    .062                         .110        343
    secondary family BH: all q = .29

    expanded (robustness only): M1 +0.10 SD, 1-sided p=.054.  Gallup Panel only: M1 p=.21, n=126.

## Sample B (population, n up to 15,372; weighted)
    corpus-show namers vs TV/print/web: out-group +0.18 SD (p=.10), valence -0.12 (p=.07),
    cynicism +0.12 (p=.08), negation 0. Nothing survives BH (q=.13).

## Reading
By its own pre-specified criterion, H1 is NOT supported by Q28. Report it that way.
What the numbers also show: every register-relevant measure points in the predicted direction at
roughly +0.12 SD — about half the Likert-scale effect (+0.24 SD), which is what a noisier
measure of the same construct would produce. Consistent with H1; does not confirm it. The test was
underpowered: median 14-word responses give noisy rates, and only ~half contain any pronoun
(n=198 for the primary).
This is what pre-registration is for: a directionally-consistent marginal result gets reported as
a failed confirmation, not spun into support.

---

# REUTERS DNR — no-request route checked and closed

Only public redistribution of DNR microdata found: the **useNews** dataset (Puschmann & Haim 2020),
OSF uzca3, CC-licensed, direct download. Contains DNR 2019 and 2020 raw survey data, 12 countries,
US n = 2,012 (2019) / 2,055 (2020). Downloaded and inspected both years.

    28 core variables: demographics, weight, general/main news use, avoidance, worn-out, 13 source-
    TYPE yes/no flags (tv, radio, print, online, sns, blogs...), interest in news/politics,
    left-right self-placement, plus ~150 named-brand weekly-use flags (MediaCloud IDs).
    PODCAST variable: none.   TRUST variable: none.

The extract stripped exactly the two variables the test needs. ORA records for 2024/2025/2026 are
PDF-only. Chart-level "get data" gives aggregates without controls. No third-party deposit of later
waves found. Conclusion: DNR microdata with Q1_podcast_open + trust items is request-only.
Request drafted: handoff/reuters_dnr_data_request_email.md. Contact: reuters.institute@politics.ox.ac.uk
(generic) or the DNR contact form; ask for the DNR data team.
Files kept in scratchpad/usenews/ in case the source-type flags are ever useful (they are not for this).

---

# ROUND 7 — descriptive exploration of the FULL sample (labeled exploratory / descriptive throughout)

## B. AUDIENCE-COMPOSITION VALIDATION OF SHOW LABELS  <-- the important one; different logical category
Question fixed in advance: do the paper's show-level labels match who actually listens?
Expanded match used (544 resp / 59 shows) because this validates LABELS, not the outcome.

    shows with >=10 Kettering listeners (14 shows):
      audience partisan lean (R% - D%) vs LLM side label      r = +0.978  p < .0001
      audience partisan lean vs DIME host cfscore             r = +0.767  p = .010  (n=10 with DIME)
      L-labeled shows: 76% Democratic audiences.  R-labeled: 91% Republican.  Zero mismatches.
      show RELIGIOUS-language score vs audience weekly church r = +0.820  p = .0003
      show RELIGIOUS-language score vs born-again share       r = +0.862  p = .0001
    shows with >=5 listeners (22): label r = +0.913; religious/born-again r = +0.672.

This is a THIRD, fully independent validation of the ideology labels — not LLM, not FEC, but real
audiences. It partially closes the "no human validation" gap at the show level. And the religious-
language measure, which showed no association with listeners' religiosity in the OUTCOME test,
predicts audience religiosity at r = .82-.86 as a COMPOSITION test. Both are true and both belong in
the paper: register predicts who listens; no evidence it changes them.
Face-validity bonus: the two mixed-audience shows are Tangle (D 33% / R 17%) and Breaking Points
(D 38% / R 21%) — exactly the two that market themselves as nonpartisan / heterodox.

## A. THE MEDIA-DIET MAP (51,082 top-3 mentions from 20,464 adults) — descriptive, publishable as-is
    institution 47.7% | platform/generic (facebook, youtube...) 20.3% | other 22.4% | local 5.3%
    person-branded 3.5% | unnamed "podcast" 0.7%
    concentration: top 10 strings = 36% of mentions; top 50 = 59%; 9,731 distinct strings.
    7.8% of Americans name a PERSON as a top-3 news source (1,495 of 19,085).
      by party: leanR 11.7 > leanD 8.9 ~ R 8.8 > D 7.3 > I 5.1.   by age: peaks 30-44 (8.9%), 65+ lowest (6.4%).
    top persons: Rogan (~334 across spellings), Tucker Carlson 64, MeidasTouch 59, Heather Cox
    Richardson 49, Aaron Parnas 44, Charlie Kirk 42, Benny Johnson 32, Rachel Maddow 29, Hasan Piker 28.

## C. "PERSON-BRANDED" NEWS DIET vs democracy attitudes — exploratory, n ~ 18,800, weighted, controls
Same cynicism syndrome as the podcast-namer result, reframed as Pew's "news influencer" construct:
    leaders accountable -0.35 | laws uphold justice -0.31 | elections run well -0.24 | press freedom -0.24
    govt includes me -0.22 | democracy doing -0.14 | feel valued -0.11 | AP gap +0.16 | protest rights +0.17
    all q < .002.   NULL: violence, loneliness, overload, citizen power, reach officials.
Overlaps heavily with podcast-namers (most named persons are podcasters), so NOT independent of the
earlier finding — but "personalized news diet" is the cleaner, literature-connected framing.

## D. DIET EXCLUSIVITY GRADIENT — exploratory, weighted
    weighted shares: mainstream-only 61% | neither (platforms/other only) 29% | podcast+mainstream 5.3% | podcast-ONLY 4.5%
    podcast-ONLY: 30% D / 59% R, mean age 43.  podcast+mainstream: 45/49, age 47.  mainstream-only: 48/40, age 52.
    vs mainstream-only (SD units, controls):        podcast-ONLY    podcast+mainstream
      leaders accountable                            -0.33 ***       -0.18 ***
      elections run well                             -0.28 ***       -0.13 **
      laws uphold justice                            -0.26 ***       -0.14 ***
      press freedom working                          -0.23 ***       -0.12 *
      radicals may protest                           +0.18 ***       +0.21 ***
      assume fraud if surprised                       +0.07 ns        -0.10 *
      political violence / loneliness / overload      ns              ns
Monotone: podcast-only > podcast+mainstream > mainstream-only on every cynicism item. Adding a
mainstream source alongside podcasts roughly halves the gap. Descriptive; natural reading is that
mainstream exposure buffers, but selection explains it equally well.

## E. Quick descriptives
    Rumble users 8.2% of adults (17% name a podcast vs 9%); Truth Social 11.0% (15% vs 9%);
    Bluesky 11.2% (13% vs 9%); TikTok 41.7% (9% vs 11% — no podcast link).
    Info overload flat across diet breadth (3.63-3.75 / 5).  Urban-rural: flat 8-10%, most-rural 6%.

## Standing
B is a validation result and goes in the paper's measurement section as-is.
A is descriptive and goes in the data section as-is.
C and D are exploratory population-level patterns: report as descriptive context with the selection
caveat stated, or hold for Y2. Neither touches the register->outcome question, so neither is
compromised by the earlier search — but neither is a confirmed effect either.

---

# ROUND 8 — how robust is the population finding, and what exactly is it?

Outcome: 4-item institutional-cynicism scale (Q35E, Q35C, Q34B, Q33H), alpha = .66 at n = 18,703;
8-item broad scale (+Q31, Q33E, Q35A, Q35D) alpha = .84. Weighted WLS, HC1.

## Survival battery — the effect does not move
    control ladder (podcast-namer vs all, 4-item):
      party+attention+age+edu                      +0.162 SD  p=8e-7
      + gender, race, income                       +0.171
      + social-media hours, # platforms            +0.163
      + urban-rural, survey mode                   +0.162
      + diet breadth (# sources named)  [FULL]     +0.157 SD  p=4e-7     attenuation: 3%
    8-item scale, full controls                    +0.218 SD  p=5e-13
    within party (full):  D +0.188 p=1e-4 | R +0.152 p=6e-4 | I +0.097 ns (n=3,195)
    item-by-item (full): accountable -.22, reflects majority -.20, serves citizens -.21, justice -.16,
      elections -.16, press -.15, includes me -.12 (all p<2e-4) | reach officials -.02 ns |
      violence -.05 ns | loneliness +.06 (p=.06; coded so + = LESS lonely)
    alternative exposure definitions: corpus-only +0.296 | broad +0.157 | wrote "podcast" +0.171 — all p<2e-3

## THE GRADIENT — reframes the finding
    4-way segments vs mainstream-only (full controls, n=18,450):
      podcast-ONLY            +0.269  p=3e-9      (4.3% of adults)
      platform-ONLY           +0.178  p=8e-11     (16.6%)  <-- facebook/youtube/x-only diets
      podcast + mainstream    +0.132  p=1e-3      (5.5%)
      other-only              +0.028  ns          (11.3%)
      pod_only = platform_only?  p=.070   (not distinguishable)
      pod_only = pod_plus?       p=.019   (mainstream buffer is real)
    continuous dose: each podcast named +0.079 SD (p=5e-6); EACH MAINSTREAM SOURCE NAMED -0.096 SD
      (p=2e-21). Additive (interaction p=.20). Cell means: 0pod/0main +.05, 2+pod/0main +.25,
      0pod/2+main -.16, 2+pod/2+main +.09.
    within party: D pod_only +.35, R pod_only +.27; platform_only ~+.18 in both.

    => The single strongest, most precise coefficient in the whole analysis is the NEGATIVE effect
       of naming mainstream sources. Platform-only diets are about as cynical as podcast-only diets.
       The defensible headline is therefore NOT "podcast listeners are cynical" but:
       "News diets with no institutional source are ~0.2 SD more institutionally cynical, whether
        the non-institutional source is a podcast or a platform feed; each institutional source
        named is worth ~-0.06 to -0.10 SD; podcasts sit at the far end of that gradient."

## The crack, and its resolution
    Gallup Panel (probability) only, full controls: podcast-namer +0.065 p=.08 (opt-in +0.142 p=.009).
    BUT: pod x panel interaction p=.50 — the halves do not differ detectably.
    Within panel: pod_only +0.182 p=.002 | platform_only +0.199 p=1e-7 | per-mainstream -0.063 p=1e-5
      -> the DIET-EXCLUSIVITY finding holds in the probability half.
    Within panel: per-podcast +0.027 ns | corpus-namers +0.074 ns (n=262)
      -> the PODCAST-SPECIFIC increment is noisy in the probability half.
    Pew ATP (probability, lagged) shows the podcast-specific effect clearly (-0.083/level; main-source
    -0.34 SD). So: exclusivity effect robust in both samples; podcast increment robust in Pew, noisy in
    Kettering-panel, not detectably different across Kettering halves.
    Note the samples differ a lot at baseline: panel +0.22 SD cynicism / 12.5% podcast-namers vs
    opt-in -0.17 / 6.4% — the OPT-IN half is the odd one (less cynical, fewer podcasts), the reverse
    of the usual worry.

## THE TRIANGLE — three independent measures of show ideology agree (answering the user's question)
    shows with >=10 listeners (14):        >=5 listeners (22):
      audience lean vs LLM label  r=+.978    +.913
      audience lean vs DIME       r=+.767    +.759
      LLM label vs DIME           r=+.770    +.747
    Text, donations, and actual listeners agree at r ~ .75-.98. This is convergent validity of the
    ideology labels from three fully independent sources, and it is the measurement section's
    strongest sentence. It does NOT substitute for passage-level human coding (it validates SHOW
    labels, not passage labels) — but it is exactly what a reviewer asks for when they ask "how do
    you know the LLM got it right."
    REGISTER vs audience lean (n=22): directive composite r=+.40 (p=.06), imperatives +.41,
    out-group +.39; insult -.16, profanity -.22, hedging -.14 (all ns).
    Same magnitude and SAME PATTERN as register vs DIME (imperatives +.39, insult .00). The "directive
    not hostile" result now replicates against a third ideology measure — real audiences — though at
    n=14-22 shows it is consistent rather than independently significant.

---

# ROUND 9 — AUDIENCE DATA THAT TIES TO THE MECHANISM

Question: does directive register leave a footprint in HOW audiences use media (not what they believe)?
Predicted footprint, stated before running: more EXCLUSIVE audiences (name no institutional source).
Nine audience-structure measures x four register measures screened at the show level; exclusivity
was the one that came through cleanly and discriminated directive from hostile register.

## A. Show-level: register -> audience EXCLUSIVITY (share of listeners naming NO mainstream source)
NBC Nightly News excluded (it is itself a mainstream source, so its 0% is definitional).
    13 shows (>=10 listeners), 406 listeners:
      directive composite   r=+0.65 (p=.016)  rho=+0.63   partial r | show ideology = +0.62 (p=.025)
      imperatives           r=+0.65 (p=.016)  rho=+0.75   partial r = +0.64 (p=.020)
      second-person         r=+0.57 (p=.043)               partial r = +0.53 (p=.063)
      out-group pronouns    r=+0.27 ns
      INSULT                r=-0.20 ns       <-- hostility does not predict exclusivity
    21 shows (>=5), 457 listeners: directive r=+0.52 (p=.016), rho=+0.63 (p=.002), partial +0.46 (p=.034); insult +0.14 ns.
    Within-ideology, L-labeled shows: more-directive half 59% exclusive (MeidasTouch, TYT, Breaking
      Points, Pakman) vs less-directive half 35% (BTC, Pod Save America, Bulwark, Tangle, Letters).
      A 24-point gap inside one ideology. R-labeled: 62% vs 60% — flat, but only 4 shows.
    LISTENER-LEVEL (521 listeners, 58 shows, clustered by show; controls show ideology, party, age,
      attention, education): dir_z b=+0.064 per SD, p=.0009 | show ideology b=+0.017, p=.84.
      Register predicts exclusivity; ideology does not.
    Other audience measures: diet breadth mirrors exclusivity (directive -> fewer sources, r~-0.4).
      Social-media hours and # platforms correlate with EVERY register measure including insult
      (~+0.5) — that is age, not mechanism. Q38 one-sidedness, Q11 disrespect, Q12D unsure: noise.

## C. Pew ATP W150 — podcast-news frequency -> mechanism items, within influencer-news consumers
n = 2,012 (probability panel), weighted, controls party-ideology / age / education / internet use.
    feel PERSONAL CONNECTION to an influencer   +4.1 pp per level   p=.001    q=.002
      raw gradient: never 26% -> rarely 31% -> sometimes 32% -> often 39%
    influencers' opinions mostly AGREE with me  +2.6 pp              p=.031    q=.043
    get OPINIONS / TAKES from influencers        +3.5 pp              p<.0001
    influencers HELPED me understand             +5.8 pp              p<.0001
    FOLLOW / subscribe                           +6.0 pp              p<.0001
    news from influencers is DIFFERENT           null
    get BASIC FACTS from influencers             null
The relational / opinion dimension moves; the informational dimension does not. That is the
audience-side signature of directive, personal address — and the parasocial item is the one the
paper had withdrawn for lack of evidence.

## What this does for the paper
The three levels now form a chain, each link with its own evidence:
    directive register  -->  exclusive (non-institutional) diets  -->  institutional cynicism
    [Round 9: r~.65,           [Round 8: platform-only ~ podcast-only;   [Rounds 5-8: two panels,
     ideology-partialled,       each mainstream source -0.096 SD,         one lagged, 3% attenuation]
     listener-level p=.0009]    p=2e-21]
    plus Pew: podcast use -> parasocial connection / agreement / opinion-seeking (probability panel).

Caveats, stated plainly:
  - Exploratory on Y1: 9 audience measures x 4 register measures were screened; exclusivity was the
    pre-stated prediction and the only clean survivor, but it is still a Y1 result. It is the natural
    second pre-registered hypothesis for the holdout / Y2 alongside cynicism.
  - Cross-sectional: "directive shows attract exclusive listeners" and "directive shows make listeners
    exclusive" are not separable here.
  - Within-R is flat on 4 shows; the within-L gap carries the within-ideology claim.
  - Pew items are population-level (podcast frequency), not show-linked.

---

# ROUND 10 — STRENGTHENING PASS: what survived, what I have to correct

## 1. EXCLUSIVITY — stands on the STRICT sample only; my expanded-sample p was wrong to cite
Listener-level P(no institutional source) ~ dir_z + show ideology + party + age + attention + edu,
clustered by show, NBC excluded. Analytic clustered p vs WILD cluster bootstrap p:

    sample     definition                     b/SD    analytic p   WILD p    n    shows
    expanded   no mainstream outlet          +0.064   .0009        .17       521   58    <-- FAILS
    expanded   no institutional (broad)      +0.072   .0001        .15       521   58    <-- FAILS
    STRICT     no mainstream outlet          +0.126   .0011        .0085     339   50    <-- holds
    STRICT     no TV/print outlet            +0.127   .0021        .011      339   50
    STRICT     no institutional (broad)      +0.147   <.0001       .0025     339   50

CORRECTION: I previously quoted the expanded-sample p=.0009. Under wild bootstrap it is .17. The
analytic clustered SE is off by ~200x there (58 clusters, MeidasTouch 82 / NBC 73 dominating). The
finding is real on the pre-specified strict match — wild p=.0085, +0.13 SD per SD of register,
doubling to +0.15 on the broad definition — and that is the number to report.
This is the FOURTH result the host-name expansion has killed. The expansion is not a robustness
check anymore; it is a different population, and the strict match is the operationalization.
Tercile gradient (expanded): low-directive 36% exclusive -> mid 58% -> high 60%. The step is
between low and mid.

## 2. TRIANGLE — with bootstrap CIs (resampling shows)
    audience lean vs LLM label   >=10: r=.978 [.948, .992] n=14 | >=5: r=.913 [.773, .981] n=22
    audience lean vs DIME        >=10: r=.767 [.464, .994] n=10 | >=5: r=.759 [.232, .968] n=13
    LLM label vs DIME            >=10: r=.770 [.487, .989] n=10 | >=5: r=.747 [.124, .976] n=13
    all 59 shows, sqrt(n)-weighted: label r=.881; DIME (38 shows) r=.656
    listener-level: 594 listeners / 59 shows, R2=.62, p=6e-98
    label predicts the audience's majority party in 21 of 22 shows (95%)
The LABEL leg is tight and should be the headline. The DIME leg is real but imprecise (CI reaches
.23 at 13 shows) — report the interval, don't lead with the point estimate.

## 3. MEDIATION — the chain is NOT supported; CORRECTION to my "one story" framing
Strict, NBC excluded, ideology controls, clustered; n=312.
    a:  register -> exclusivity            +0.127  p=.003
    c:  register -> cynicism (total)       +0.199  p=.0001
    b:  exclusivity -> cynicism | register +0.145  p=.061
    c': register -> cynicism | exclusivity +0.180  p=.0003
    indirect a*b = +0.018 = 9% of total.  Cluster-bootstrap 95% CI [-0.003, +0.037] — includes 0.
Register's association with cynicism is DIRECT (c' ~ c). Exclusivity does not carry it. The b-path
is underpowered here (n=312 vs 18k at population level), but even the point estimate is 9%
mediated. So: three associations that cohere, NOT a demonstrated chain. Last round's
"directive -> exclusive diets -> cynicism" framing overreached; the honest version is
"directive register is associated with both exclusive diets and cynicism, and the cynicism
association is not explained by exclusivity."

## 4. GRADIENT — CORRECTION in the other direction: podcast-only IS beyond platform-only
TOST equivalence test, podcast-only vs platform-only, 8-item cynicism, full controls, n=3,974:
    difference = +0.144 SD (se .052).  Not equivalent within +/-0.10, +/-0.15, or +/-0.20 SD.
I previously said "platform-only ~ podcast-only (p=.07)". A non-significant difference is not
equivalence. On the better scale with full controls the difference is +0.14 SD, z=2.8. Corrected
headline: non-institutional diets carry most of the gap; podcast-only diets sit a further 0.14 SD
beyond platform-only. The podcast-specific increment is real.

## 5. PEW parasocial item — holds within party; driven by older listeners
    Rep/lean R  b=+.051/level p=.003 | Dem/lean D +.037 p=.038
    age 18-29 +.034 ns (n=419) | 30-49 +.032 ns (856) | 50+ +.072 p=.0001 (697)

## Publishable-strength ledger after this pass
    Triangle (label leg)           r=.91-.98 with tight CIs          -> report
    Population cynicism/gradient   +0.22 SD, 3% attenuation, Pew lagged, pod-only > platform-only
                                                                       -> report as association
    Register -> exclusivity        strict: +0.13 SD/SD, wild p=.0085  -> report; strict rule only
    Register -> cynicism           strict: +0.20, wild p~.003, but post-hoc composite, expansion-fragile,
                                   NOT mediated by exclusivity        -> report as exploratory, disclosed
    Pew parasocial/agreement       n=2,012, BH-corrected, within-party -> report

---

# ROUND 11 — analytic clustered inference: CR1 vs CR2+Satterthwaite vs wild bootstrap

    estimate                                          b     CR1 p    CR2 p   Satt df  WILD p   G    N
    Favourability gap, strict, base controls       +0.590   .0016    .0154     8.4    .0405   51   384
    Cynicism composite, strict, +ideology          +0.219   .0002    .0061     7.4    .0005   49   378
    Exclusivity, strict, NBC excl., +ideology      +0.124   .0023    .0249     7.1    .0140   50   312
    Exclusivity, EXPANDED, NBC excl., +ideology    +0.062   .0000    .0581     2.6    .0930   57   471
    Cynicism composite, EXPANDED, +ideology        +0.078   .0255    .2808     2.3    .1615   56   537

CR1 = conventional cluster-robust SE, t on G-1 df (what statsmodels cov_type="cluster" and Stata
vce(cluster) report). CR2 = Bell-McCaffrey (2002) small-sample correction with Satterthwaite
degrees of freedom (Pustejovsky & Tipton 2018).

## The diagnosis in one number
Satterthwaite df — the EFFECTIVE number of clusters the coefficient is identified from — is 7-8 on
the strict sample and 2-3 on the expanded, against 49-57 nominal clusters. With 24 of 51 clusters
at n=1 and the top three clusters holding 53% of respondents, the register coefficient is really
being estimated from a handful of big shows. CR1 assumes ~50 effective clusters; that is why it is
off by one to three orders of magnitude on the expanded sample.

## What to report
CR2 + Satterthwaite agrees with the wild bootstrap on every row and reaches the same conclusion
every time: strict results hold, expanded results fail. It IS an analytic clustered p, it is
standard in the clustered-inference literature, and a methods reviewer will accept it. Report CR2
p (with its Satterthwaite df) as the primary inference and wild-bootstrap p alongside; two
independent small-sample methods agreeing is stronger than either alone.
Never report CR1 in this paper. Correct §H step 5 accordingly: "CR2 with Satterthwaite df, or wild
cluster bootstrap, on every show-clustered estimate — never conventional CR1."

---

# ROUND 12 — is directive register horse-race / strategy framing? (the Cappella & Jamieson challenge)

Why it matters: the spiral-of-cynicism mechanism runs through STRATEGY / GAME framing (politics as who
is winning, polls, tactics). If directive shows are also the game-frame shows, that established
mechanism absorbs the mode-of-address result. Lexicon after Aalberg, Strömbäck & de Vreese (2012),
two subframes, scored over all 1,366,988 passages; show-level per-10k rate (median 28.4, IQR 22-36).

## A. Not the same construct — if anything, opposed
    dir_z vs horse-race subframe   r=+0.09 (ns)
    dir_z vs strategy subframe     r=-0.26 (p=.0002)    <-- directive shows use LESS strategy talk
    dir_z vs combined game frame   r=-0.04 (ns)
## B. Game framing is not partisan
    vs DIME r=-0.002 | vs LLM side r=-0.09. Highest game-frame shows are the insider/process shows —
    Hacks on Tap, Playbook, Politically Georgia, Perino on Politics; lowest are culture-war shows.
## C. Joint models, corpus listeners (strict, NBC excl., ideology controls), CR2 + wild inference
    CYNICISM      b       CR2 p   Satt df  WILD p
      directive alone            +0.202   .008     8.1     .0025
      game frame alone           -0.130   .031     4.4     .089
      both: directive            +0.167   .013     9.4     .0025   <-- SURVIVES
      both: game frame           -0.074   .069     6.1     .099
    EXCLUSIVITY
      directive alone            +0.126   .037     8.3     .024
      game frame alone           -0.132   .007     4.5     .038
      both: directive            +0.076   .11      9.6     .073    <-- attenuates to marginal
      both: game frame           -0.106   .015     6.2     .035    <-- game frame carries it (negatively)

## Reading
1. Directive address is NOT horse-race journalism. The constructs are orthogonal at the show level and
   the strategy subframe runs the other way. The paper's mechanism is distinct from Cappella & Jamieson's.
2. In this population, game framing goes with LESS institutional cynicism and LESS exclusive audiences —
   the opposite of the spiral-of-cynicism prediction. Coherent reading: strategy/process talk is the
   register of mainstream-adjacent political journalism, and its audiences also consume mainstream
   sources. That is consistent with the diet-exclusivity gradient (round 8), not a contradiction of it.
3. For CYNICISM, directive survives the game-frame control with the coefficient barely moved. For
   EXCLUSIVITY, game framing (negative) is the stronger predictor and directive falls to marginal:
   directive shows' audiences are exclusive in part because directive shows are not the insider shows.
   §F.1 should be restated with that qualification.
Caveats: exploratory, Y1, n=306-312; lexicon modelled on a published frame taxonomy but built here;
Satterthwaite df 4-10. As with everything on the strict sample: the holdout is the test.

---

# ROUND 13 — what "directive" IS, tested with literature instruments (three alternative mechanisms ruled out)

## A. Decomposing the composite (spaCy dependency parse, 122,444-passage stratified sample, 600/show)
What the syntactic imperatives are: "let" 12.8% (let me tell you / let's), action verbs 45% (go, get, make,
take, do, give, visit), attention-getters 13% (look, listen, see, wait), EPISTEMIC only 6% (understand,
realize, remember, think). What the second person is: generic "you" 72%, "you know" (filler) 19%,
deontic "you need to / should / have to" 7%, addressive "I'm telling you" 1.6%.
Which subtype tracks ideology (show level, vs DIME, n=100):
    you_know (filler)         r=+0.00  (p=.97)   <-- the construct is NOT conversational informality
    you_deontic               r=+0.44  (p<.001)  <-- strongest single carrier
    you_other (generic)       r=+0.38
    you_addressive            r=+0.36
    imp_action                r=+0.37
    imp_attention             r=+0.31
    imp_epistemic             r=+0.15  (ns)      <-- NOT epistemic instruction
Against Kettering outcomes (strict, NBC excl., ideology controls, CR2 / wild):
    CYNICISM:    composite +0.202 (wild .0025) > deontic+action +0.132 (.043) > generic-you +0.148 (.036)
                 > deontic-you +0.119 (.087) ~ action +0.102 (.077); filler -0.009 (.88); epistemic df=2.9
    EXCLUSIVITY: deontic-you +0.106 (CR2 .025, wild .023) is the cleanest single carrier; composite +0.126 (.025)
    (subtypes come from a 122k sample and are noisier than the full-corpus composite, which is part of why
     the composite wins on cynicism)
    Deontic-address composite vs DIME r=+0.46 (reference dir_z +0.43); vs dir_z r=+0.83.

## B. Biber Dimension 4 "overt expression of persuasion" from MFTE (600 episodes / 164 shows / 100 with DIME)
    D4 (MDNE+MDWS+COND+SPLIT)      vs dir_z r=-0.12 (ns)   vs DIME +0.04 (ns)
    necessity modals (MDNE) alone  vs dir_z r=-0.03         vs DIME -0.02       <-- NULL
    MFTE imperatives (VIMP)        vs dir_z r=+0.51 ***     vs DIME +0.19 (p=.053)
    MFTE 2nd person (PP2)          vs dir_z r=+0.43 ***     vs DIME +0.13
    emphatics (EMPH)               vs DIME r=-0.29 (p=.004) — right-leaning shows use FEWER emphatics
=> The construct is NOT Biber's overt persuasion. Necessity modals are null; but deontic-YOU is the
   strongest subtype. The modal is not what carries it — the SECOND-PERSON SUBJECT is. MFTE's own
   imperative and 2nd-person tags confirm the spaCy measure. It is address, not modality.
   (MFTE sample is small per show; DIME correlations attenuated.)

## C. Rooduijn & Pauwels (2011) populism dictionary and Fawzi-style anti-media (full corpus, 194 shows)
    R&P populism      vs dir_z r=+0.11 (ns)  vs DIME -0.04  vs INSULT +0.41
    people-centrism   vs dir_z r=+0.01       vs DIME -0.01  vs insult +0.22
    anti-media        vs dir_z r=+0.01       vs DIME -0.06  vs insult +0.38
=> Validated populist content is orthogonal to directive register and belongs to the HOSTILITY cluster.
   It is not partisan. Highest R&P: Kulinski (L), Owen Report (R), TYT (L). Highest anti-media: System
   Update (R), Timcast (R), Olbermann (L), On the Media (L).
   Joint models (strict, ideology controls, CR2 / wild):
    CYNICISM     directive alone +0.202 (.0025) | R&P alone +0.081 (.32) | anti-media alone +0.007 (.87)
                 directive | R&P +0.192 (.004) | directive | anti-media +0.202 (.0045)
                 directive | R&P + people + anti-media +0.190 (CR2 .024, wild .019)   <-- SURVIVES ALL THREE
    EXCLUSIVITY  directive alone +0.126 (.025) | R&P alone +0.043 (.48) | anti-media +0.010 (.80)
                 directive | R&P +0.122 (.025) | directive | anti-media +0.127 (.021)
                 directive | all three +0.102 (CR2 .080, wild .106)   <-- marginal under the triple control
   (My earlier ad hoc "anti-elite" lexicon correlated +0.19-0.31 with dir_z; the validated R&P does not.
    The ad hoc list was picking up something broader. Use R&P.)

## Standing after round 13
Three content-based alternative mechanisms tested against mode of address — strategy/game framing
(Aalberg et al.), populist anti-elitism (Rooduijn & Pauwels), anti-media populism (Fawzi) — two with
validated instruments. None correlates with directive register above r=0.11; none predicts listener
cynicism on its own; directive -> cynicism survives each and all of them together. Exclusivity is
more fragile: game framing (negatively) is the stronger predictor there.
What "directive" is: SECOND-PERSON ADDRESS — the host speaking TO the listener — with action
imperatives; not filler, not epistemic instruction, not necessity modality, not Biber D4 persuasion,
not populist or anti-media content. The literature that fits is Horton & Wohl para-social address,
not Cappella & Jamieson frames and not the populism-style literature.
Rename in the paper: "directive" -> "listener-directed address" (or "second-person address").

---

# ROUND 14 — is there a stronger EVENT? No: listener-directed address is event-invariant

Within-show event study, episode-level LDA composite (corpus-standardised), ±60 days, show fixed
effects, clustered by show; 18,633 dated episodes / 194 shows.
    event                        eps   shows   post b (SD)    p     L / R shift     post×side p
    2024 general election       1206   159      -0.030      .32    -0.01 / -0.08      .999
    2025 inauguration           1180   168      +0.003      .94    +0.03 / -0.05      .46
    Trump assassination attempt  995   155      +0.002      .95    +0.01 / -0.01      .78
    Biden withdraws             1032   155      +0.037      .29    +0.05 / +0.02      .71
    2022 midterm                 695   116      -0.012      .82    +0.02 / -0.09      .27
    2020 general election        419    75      -0.018      .78    -0.08 / +0.14      .08
    January 6                    450    78      +0.084      .29    +0.11 / +0.02      .58
    Dobbs                        587   106      +0.048      .35    +0.08 / -0.01      .45
Placebo-date inference (2024 election vs 60 random non-election dates, same design): real shift
-0.030 SD, placebo sd 0.062, two-sided placebo p = .57.
Event-time profile (weekly, show-demeaned): noise around zero both sides; pre-trend slopes
-0.004 and +0.007 SD/week. Run-up: final 4 weeks before the election vs prior 4 weeks: 2024 +0.055
(p=.24), 2022 +0.048 (p=.55), 2020 -0.175 (p=.12) — no mobilisation spike, no side difference.

## Reading
No political event in eight years moves how hosts address their listeners, within show, on either
side. This is the level-dissociation result (AUC .80 between shows / .57 within; §6 of the record)
confirmed with an event design: listener-directed address is a TRAIT of the host, not a reaction
to the calendar. It cuts against any mobilisation reading ("hosts ramp up 'you need to' before
elections") and for the para-social reading (it is how the host talks, full stop). It also means
there is no event-based quasi-experiment to be had on the text side; the corpus's leverage is
between shows, and the audience link is where fresh data matter.
Consistent with the earlier record: the 2024 election shifted the AP gap by only +0.016 (p=.061).

---

# ROUND 15 — topics, and a full survey profile of the diet segments

## A. Is listener-directed address higher for certain TOPICS? No.
LDA k=75 (existing, 40,242 chunks; 20,573 joinable at passage level across 204 shows), address demeaned
within show. Spread of within-show topic means = 0.042 SD vs 0.23 SD across shows; topic adds R2 = 0.005
over show (0.105 -> 0.109). Highest-address "topics" are AD READS (promo codes, T9 +0.06) and sign-offs
(T60 +0.06), then tariffs/economy (+0.04); lowest are Supreme Court (-0.10), immigration (-0.07),
constitutional law (-0.05), Ukraine (-0.05). Whole range +/-0.1 SD. Left-right gap largest on economy
(+0.15), reversed on family/personal (-0.13) and Iran (-0.10); interaction negligible.
=> Third confirmation of the trait story: not events (round 14), not topics, SHOWS. Keep the ad-stripped
robustness (already in the record) prominent: ads are the most second-person passages in the corpus.

## B. Population profile: diet segments x every survey item (115 items, n~18,900, weighted, full controls, BH)
Survivors at q<.05: podcast-only 46 | podcast+mainstream 25 | platform-only 67.
### Podcast-only vs mainstream-only (SD units)
Institutional performance (Q33): Congress -0.32, elections -0.28, criminal justice -0.24, people's role
  -0.23, press -0.21, division of powers -0.21, judiciary and others also negative.
Government responsiveness (Q35): serves citizens -0.33, sensitive to people like me -0.30, reflects
  majority -0.27, upholds justice -0.26, includes people like me -0.21.  <-- this cluster DIVERGES from
  platform-only (-0.08 to -0.10): podcast-specific.
System support: leaders committed to democracy (Q30C) -0.32; DEMOCRACY IS THE BEST FORM OF GOVERNMENT
  (Q30A) -0.22 (platform-only -0.21 too); accountability (Q31) -0.31.
CIVIL-LIBERTARIAN NORMS — new, coherent, podcast-specific (platform-only ~0 on all four):
  radicals should be able to protest (Q32F)             +0.17   (platform-only +0.00)
  bar radicals from office (Q32B)                       -0.18   (platform-only +0.01)
  give presidents more power (Q32G)                     -0.17   (platform-only -0.02)
  media should take more government direction (Q32D)   -0.35   (platform-only -0.01)  <-- largest divergence
Other: less concerned about crime (Q16 -0.17); lower current life-ladder (Q1 -0.13); see LESS community
  disrespect (Q11 -0.22, coded 1=never..4=frequently); feel MORE welcome participating (Q12H); economic
  differences "unfair" -0.16 (less egalitarian, party controlled). Platform use: more YouTube/Rumble, less Facebook.
NULLS worth stating: partisan STRENGTH 0.00 (q=.999) — podcast-only diets are not more strongly partisan;
  American identity -0.01; local identity -0.05; loneliness +0.08 ns (direction = less lonely); registered
  to vote -0.03 ns; volunteering, community events ~ns; political violence +0.04 ns; AP gap +0.05 ns.
### Platform-only vs mainstream-only — a DIFFERENT population
Weaker American identity (-0.11) and local identity (-0.11); lower life-ladder now (-0.10) and in 5 years
(-0.08); more income strain (+0.10); more concentration difficulty (+0.13); more economic worry; LESS
registered to vote (+0.14 = "no"); less civic education (-0.10) and less political upbringing (-0.13);
MORE political-violence tolerance (+0.09, q=.007); lower AP gap (-0.08); none of the civil-libertarian items.
=> "Non-institutional diet" is two populations. Platform-only = disengagement and precarity. Podcast-only =
   ENGAGED anti-institutionalism: distrust the institutions, defend the liberties, not more partisan, not
   disengaged. That is the profile the paper should describe, and it is consistent with Gallup's own
   published finding on protest rights, extended across the norms battery.
Caveats: descriptive, cross-sectional, selection; 115-item family with BH; segment definitions from
verbatim classification (round 8). Signs read from codebook labels (recorded in pop_profile.csv).

---

# ROUND 16 — holdout pipeline test + face-validity ranking (2026-09-03T00:24:21Z), no outcomes touched
Pipeline end-to-end on the first 19 holdout episodes (2,014 passages): same 750-char chunker
(build_scoring_chunks.split_750), same remeasure2 features, nw-weighted show rates, z against the
194-show reference. Works.
    show            eps  passages  imper_syn  imper_rx  you_rx  dir_z
    Rogan             6      1428        71        32     287   +0.86   <-- top; dyadic guest-you
    Shapiro           1        27       118        16     206   +0.45   (1 episode — meaningless yet)
    The Daily         7       341        67        22     196   -0.29
    Parnas            3        75        81        17     197   -0.31
    Tucker            2       131        33        12     351   -0.44   (very high you, very low imperatives — dialogue)
Reading: the dyadic-format confound is real and material. Amendment 1 filed to the prereg: a
pre-specified, publicly-coded INTERVIEW_DOMINANT indicator as a labeled sensitivity; primary test
unchanged. Ranking will be re-run when all shows clear 25 episodes; The Daily / Parnas / Tucker
values are provisional at 2-7 episodes.

---

# ROUND 17 — can the measure be improved? (2026-09-03T00:29:16Z)
Sentence-type split (questions / person-names / declarative / deontic / plural), holdout + 50-show corpus sample:
  - Rogan: you 286/10k, 69% declarative, 13% in questions (corpus median 11%), deontic 22.6 (Shapiro 26, Beck 26,
    Daily 14, Parnas 10). Text heuristics do NOT separate guest-you from listener-you; conversational
    "you" to a co-present interlocutor is declarative too.
  - Declarative-you is a marginally cleaner instrument on the corpus (DIME r=.50 vs .46; vs dir_z .79 vs .77).
Position in episode (holdout): monologue shows spike in the open (Shapiro open 565 you/10k, 350 imperatives;
  body 170/90), interview shows flat (Rogan 247/287/293, Tucker 320/337/360). => a text-only format
  fingerprint; stable once shows have 25-50 episodes. Filed as Amendment 2 (vii).
Corpus: interrogative-you vs dir_z r=+.67 (address LEVEL conflates dialogue) but vs DIME r=+.03 and vs
  side +.11; partialling q_share out of dir_z leaves DIME r=.49 (raw .47). => The left-right register
  contrast is robust to the dialogue component; the level is not. State this in the measurement section.
Clean fix: diarization (host-turn address). rma2 archives Rogan/Tucker/Beck/Bongino audio; rma1 discards.
~15-20 GPU-hours for the archived shard; a later sensitivity, not tonight.

# ROUND 18 — holdout ranking at real sample sizes (153 eps, no outcomes touched), 02:58Z
    Bongino +2.38 (34 eps; you 372/10k, imp 110) | Rogan +1.24 (29) | Beck +1.06 (30) | Tucker +0.16 (28)
    Parnas -0.27 (12) | The Daily -0.28 (13) | Shapiro -0.73 (7, provisional) | Kirk/Kelly/Ryan pending
    Corpus landmarks: Crowder +2.01, MeidasTouch +0.94, PSA -0.12, Letters -2.80.
Amendment 2 (vii) position-based format score: DOES NOT DISCRIMINATE — median open/body you ratio 0.9-1.0
for monologue (Bongino 1.00, Beck 0.97) and interview (Rogan 0.91, Tucker 0.89) alike; Shapiro's opening
spike was one episode (median 0.81, mean 1.44); Parnas reversed (0.39). Reported as pre-specified; it
will not carry format control — Amendment 1's hand-coded INTERVIEW_DOMINANT remains the format sensitivity.
Ops: rma2 took over the rma1 manifest at 02:56Z; rma2-shard leftovers (eps 29-50) parked so both machines
work the six under-25 shows first.

## Round 19 — 2026-09-03T08:12Z — CONFIRMATORY out-of-frame generalization test: H1 FAILED
Full record: handoff/results/holdout_RESULTS.md; raw log handoff/results/holdout_RUN_2026-09-03T0812Z.log. Primary b = −0.001 (CR2 p1 .50, wild .52, df 3.5). Rogan-excluded b = +0.223 (p1 .084/.096). All sensitivities (iii)–(viii), secondary Q33E, H2 exclusivity: null. Outcome composite verified (α .84). Register→cynicism stays exploratory with a failed pre-registered confirmation; population finding and measurement/validation levels unaffected.

## Round 20 — 2026-09-03T08:29Z — WITHIN-FRAME pre-registered test (150 fresh corpus-show respondents, 10 shows): H1 FAILED (b = −0.055, p .57). Pooled exploratory across 17 fresh political-content shows: +0.169, p ≈ .10. Full record: handoff/results/withinframe_RESULTS.md. Register→cynicism now has two failed pre-registered confirmations.

## Round 21 — 2026-09-03 ~09:00Z — EXPLORATORY pooled scan (64 shows, 1,284 resp): R&P populist-vocabulary rate tracks audience cynicism (+0.195, wild p .002; fresh-only +0.211, p .017; same size in both samples; survives everything; beats dir_z and ideological intensity in fresh data). Ideological intensity / address track exclusivity instead. Full record: handoff/results/exploratory_populism.md.

## Round 22 — 2026-09-03 ~10:30Z — EXPLORATORY within corpus frame: address → audience EDUCATION composition (−0.27 SD/SD, CR2 p .004, wild .002; replicates in all four samples; age/income/gender null; survives full demographics). Address's attitudinal associations are shared with populism/intensity (collinear). Epistemic-imperative → efficacy corpus-only, leverage-sensitive. Record: handoff/results/exploratory_populism.md.

## Round 23 — 2026-09-03 ~11:15Z — Education as moderator: NOT supported. Population-level podcast-only → cynicism +0.34 SD in both low- and high-education strata (interaction p .91, n=18,893). Address × education null in corpus (0/119 items). Record: handoff/results/exploratory_populism.md.

## Round 24 — 2026-09-03 ~12:00Z — Address, final exploratory ledger: education composition (robust) + election distrust (corpus-only, shared with intensity, not in fresh samples); response style, thresholds, dose, embeddedness null. Record: handoff/results/exploratory_populism.md.

## Round 25 — 2026-09-03 ~12:45Z — Political-content frame (≥15% political passages, symmetric): address → election distrust +0.27 (wild <.001, 56 shows), cynicism +0.22, disaffection +0.16, education −0.21; stable across cutoffs .08–.30 and all leave-one-out; survives populism+intensity for election distrust/cynicism/disaffection. Out-of-frame political shows alone: same sign, 6 clusters, wild p .08. Post-hoc frame → pre-register on Y2. Record: handoff/results/exploratory_populism.md.
