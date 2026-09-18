# Methods (v2, 15 September 2026)

Scope: the address measure and its validation, and the Kettering-Gallup findings for listeners of the political
corpus. Chart-absent shows, the population news-diet finding and the topic arm are out of scope. Items still
pending are marked.

## 1. Corpus

The sampling frame is the Apple Podcasts US chart for the *Politics* subcategory of *News* (250 shows), frozen on
13 July 2026, a public list rather than a curated one. Each show's RSS feed supplied its episode catalogue. Six
inclusion rules were applied in fixed order with the first failure recorded: no feed URL (2), feed unreachable
(2), unparseable (1), no audio enclosures (0), not US politics by publisher blocklist, title pattern, feed language
or curated list (14), under ten hours of audio available (12). 219 shows remained.

Three further counts follow from the transcription history and are reported as such. Fifteen of the 219 were not
in the initial 25-hour-per-show transcription sample and were kept out of the corpus by decision on 14 August
2026, leaving 204 transcribed shows. Ten of those 204 had too few transcribed episodes (10 to 18) when the
show-level table was built in August to receive a show-level ideology label, and the address reference set was
fixed on the remaining 194. *(Pending: score the ten against the frozen reference and report the validation with
and without them.)*

Episodes were chosen by a priority ladder: a census of every episode before 1 January 2018, then bands taking k
episodes per show per calendar quarter with ranks from a frozen hash, so later bands only append. Audio was
transcribed with faster-whisper (large-v3-turbo, English, voice-activity detection, no diarisation). Advertisements
were retained, following the ideology instrument's own practice (Much et al. 2026); about 0.4% of passages are
politically themed ad content. Transcripts were cut into 750-character passages at sentence boundaries, the unit on
which that instrument was validated; 1.38 million passages.

Two external anchors were attached for validation only: the host's DIME campaign-finance ideology score, obtained
by resolving host names through Wikidata and matching to the DIME 1979–2024 files (112 of 194 shows), and the
Brookings Political Podcast Project lean labels (Wirtschafter 2023).

## 2. The ideology instrument and what is built from it

A capped sample of passages (40 per show-quarter; 152,152 in all) was labelled with the two-stage rubric of Much
et al. (2026), prompts verbatim, model `gpt-5.4-mini-2026-03-17`. Stage 1 asks whether a passage is political;
stage 2 places political passages on a seven-point scale from strongly liberal to strongly conservative. Three
show-level quantities come from these labels, all weighted by passage word count:

- **side**: share of political passages labelled conservative minus share labelled liberal;
- **ideological intensity**: share of political passages labelled anything other than "moderate";
- **political-content density**: share of all sampled passages that stage 1 marks political (corpus median 0.85).

Density defines the political-podcast frame used in §6; intensity is the rival explanation tested in §7.

## 3. Measuring listener-directed address

Register, the manner of speech rather than its topic (Biber 1988), was measured on the same passages as counts per
10,000 words, aggregated to shows by word-count-weighted mean. The address score, `dir_z`, averages three
standardised rates: **imperatives identified from a dependency parse** (a bare-form verb heading a clause with no
subject or auxiliary, not preceded by *to*, *not* or *never*; spaCy), **a ten-phrase imperative dictionary**
(look, listen, think about it, consider, remember, imagine, ask yourself, understand, realize, let me tell you;
written for this study), and **second-person pronouns** (you, your, yours, yourself). Each rate is z-scored
against the 194-show corpus and the three z-scores averaged; the reference means and standard deviations are
frozen, so any show can be scored on the same scale.

**The measure of record** is the same construct rebuilt from a published tagger: the Multi-Feature Tagger of
English (MFTE; Le Foll 2021), whose imperative-verb (VIMP) and second-person (PP2) tags were applied to every
transcript (45,192 episodes, 409M words), converted to rates per 10,000 words, aggregated to shows by word-weighted
mean, z-scored against the 194 reference shows and averaged (`mfte_z`). It correlates 0.93 with the discovery
version above, 0.41 with the host's DIME score (discovery version 0.43), and separates the sides by 0.69 SD
(discovery 0.78). The discovery version is reported in the appendix. No word list written for this study enters
the measure of record.

The motivation is the broadcast-talk literature: mass media address each listener as an individual (Fairclough
1989; Scannell 1996), direct address and imperatives are the marks of intimate radio talk (Montgomery 1986;
Hutchby 1996), the relationship this builds is parasocial (Horton & Wohl 1956; Dibble, Hartmann & Rosaen 2016)
and audio alone sustains it (Schlütz & Hedder 2022; Euritt 2023). Second person carries normative force (Orvell,
Kross & Gelman 2017) and raises involvement independently of content (Cruz, Leonhardt & Pezzuti 2017).

Address was the survivor of a wider search over three feature batteries on full episodes: a style battery
(certainty, hedging, questions, negation, pronoun othering, Brysbaert concreteness, moral-emotional words after
Brady et al. 2017), a wider register battery (fillers, contractions, complexity, stance, temporal orientation), and
a lexical outrage battery operationalised from the Sobieraj and Berry (2011) categories with out-group references
after Rathje, Van Bavel and van der Linden (2021). Apart from the concreteness norms and the moral-foundations
dictionaries, those word lists were written for this study and are given in full in the appendix; the hostility
null below should be read with that in mind.

## 4. Validation of the measure

Address separates the sides: r = 0.43 with the host's DIME score (Spearman 0.45), a between-side gap of 0.78 SD,
AUC 0.80. The lexical hostility measures do not: insult rate r = 0.00 with DIME, and right-leaning shows use fewer
emphatics. The LLM side label, the DIME score and the party identification of a show's own survey listeners agree
pairwise. Address is a stable property of a show: a within-show event study around eight political shocks from
November 2020 to January 2025, with placebo dates, shows no post-shock shift, and within-show address is flat
across the 75 topics of a topic model. Decomposing imperatives into epistemic, attention and action verbs, and
second person into guest-directed, declarative, deontic and audience-plural uses, leaves the show ranking
unchanged.

## 5. Survey and linkage

The Kettering Foundation / Gallup *Democracy for All* survey, Year 1 (n = 20,338), asks respondents to "name the
top three programs, channels, podcasts, websites, influencers, or other outlets from which you get news and
information." Mentions that are the exact title of a corpus show, checked by hand, form the strict match: 385
respondents naming 51 shows in the discovery sample, plus 150 later-matched respondents (the within-frame
pocket), 535 corpus listeners of 54 shows in all. Exposure is the mean `dir_z` of the shows a respondent named;
the cluster is the first show named.

Controls in every model: party identification (Republican, Democrat, Independent, leaners assigned), political
attention, age, education, a count of right-wing platforms used (Truth Social, Rumble, Parler, Gab), and the
show's lean. Outcomes are standardised on the full survey, never on the listener subsample: an eight-item
institutional cynicism scale (α = .84: confidence in election administration and press freedom, laws uphold
freedom and justice, government includes people like me, decisions reflect the majority and serve citizens,
leaders held accountable, I know how to reach officials; Cappella & Jamieson 1997; Hetherington 2005; Citrin &
Stoker 2018); election distrust (confidence in election administration reversed, plus agreement that election
officials acted improperly when results surprise; Berlinski et al. 2023; Clayton et al. 2021); exclusivity
(naming no institutional news outlet; Ladd 2012); and respondent education as an outcome (audience composition;
Zaller 1992).

## 6. The political-content frame

Shows whose LLM density is at or above the corpus 10th percentile (0.644): 49 of the 54 matched shows, 456
listeners. The five matched shows below the cut are The Arena, Candace, Michael Berry, New Yorker Radio Hour and
Volts. The frame is reported as a robustness restriction: inside the corpus the framed and unframed estimates are
the same (§7), and the estimate is unchanged at every cut-off from the 5th to the 25th percentile.

## 7. Models and inference

Each outcome is regressed on exposure and the controls. A second specification adds two show-level rivals on a
per-show-SD scale: the Rooduijn and Pauwels (2011) populism dictionary rate (with Fawzi-style 2019 anti-media
references), and ideological intensity from §2. Because respondents cluster in few shows, every estimate reports
CR2 standard errors with Satterthwaite degrees of freedom (Pustejovsky & Tipton 2018) as the primary p-value and
a Rademacher wild cluster bootstrap with 1,000–2,000 draws alongside (Cameron, Gelbach & Miller 2008), with
show-level permutation, leave-one-show-out and Benjamini-Hochberg correction where noted. Pre-registered tests
used one-sided α = .05 and required both p-values below it.

## 8. Pre-registration and status of the findings

Confirmatory tests were frozen before the data were touched (Nosek et al. 2018). The two corpus-based tests
failed on their own criteria: the free-text democracy item, and the within-frame test on the 150 pocket
respondents (b = −0.055, one-sided p .57). A third pre-registered test on chart-absent shows also failed
(b = −0.001); it is outside this paper's frame but is reported in the disclosure section.

What follows is exploratory. Among corpus listeners, address predicts election distrust (+0.22 SD per reference
SD; CR2 p .004, wild .011; +0.22 inside the frame), institutional cynicism (+0.21; .020 / .026), a less-educated
audience (−0.30 SD of the survey's education distribution; .004 / .002) and exclusivity (+0.13; .043 / .021).
Populist vocabulary explains none of this: every coefficient is unchanged when the populism rate is added, and
populism predicts nothing on its own. **Ideological intensity does.** It correlates +0.44 with address across
shows, predicts each outcome at least as strongly, and once it is in the model address retains only a marginal
coefficient on cynicism (+0.14, wild p .03–.05) and none on election distrust (+0.06), education (−0.12) or
exclusivity (+0.05). With 54 shows the two properties cannot be separated, and the paper says so: what is
established is that listeners of address-heavy, ideologically intense shows are more election-distrustful, more
cynical, less educated and more likely to have abandoned institutional news; which of the two show properties
carries that, and whether either causes it, is left to the Year-2 pre-registration (LLM-density frame, election
distrust primary, intensity as the named rival, interview date requested) and to a within-show check of whether
the instrument reads address-heavy passages as more intense. *(Pending: that within-show check.)*
