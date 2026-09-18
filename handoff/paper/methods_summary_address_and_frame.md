# Methods (high-level draft): the address measure and the political-corpus survey findings

Scope of this draft: the measurement and validation of listener-directed address, and the Kettering-Gallup
findings for listeners of the political corpus. The population (news-diet) finding, the out-of-frame test on
chart-absent shows, and the topic arm are left out; see the note at the end on what still has to be disclosed.

## 1. Corpus

The sampling frame is the Apple Podcasts US *Politics* chart, frozen on 13 July 2026 (250 shows), a public and
reproducible list rather than a curated one. Each show's RSS feed supplied its episode catalogue. Six inclusion
rules were applied in a fixed order with the first failing rule recorded as the reason (no feed, unreachable,
unparseable, no audio, not US politics, under ten hours available), leaving 219 shows. Episodes were chosen by a
priority ladder: a census of every pre-2018 episode, then bands of k episodes per show per quarter, with ranks
from a frozen hash so that later bands only append. Audio was transcribed with faster-whisper (large-v3-turbo,
English, voice-activity detection, no diarisation). Advertisements were retained, following the practice of the
ideology instrument the corpus adopts (Much et al. 2026). The result is roughly 30,000 episodes across 194 shows
with usable transcripts.

Two external anchors were attached to shows and used only for validation: the host's DIME campaign-finance
ideology score, obtained by resolving host names through Wikidata, and the Brookings Political Podcast Project
lean labels (Wirtschafter 2023).

## 2. Measuring listener-directed address

Register, the manner of speech as distinct from its topic (Biber 1988), was measured on 750-character passages,
the unit on which the corpus's ideology instrument was validated (Much et al. 2026), and expressed as counts per
10,000 words. The address measure, `dir_z`, averages three standardised rates:

1. **Syntactic imperatives.** From a dependency parse (spaCy): a bare-form verb heading a clause with no subject
   or auxiliary and not preceded by *to*, *not* or *never* ("look at what they did", "go read the bill").
2. **Dictionary imperatives.** Ten attention-directing phrases (look, listen, think about it, consider, remember,
   imagine, ask yourself, understand, realize, let me tell you).
3. **Second person.** you, your, yours, yourself.

Passage rates are averaged to the show level weighted by passage length, standardised against the mean and
standard deviation of the 194-show corpus (a frozen reference table), and averaged. The theoretical motivation is
the broadcast-talk literature: mass media address each listener as an individual (Fairclough 1989, "synthetic
personalisation"; Scannell 1996), direct address and imperatives are the marks of intimate radio talk
(Montgomery 1986; Hutchby 1996), and the resulting relationship is parasocial (Horton & Wohl 1956; Dibble,
Hartmann & Rosaen 2016), which audio alone can sustain (Schlütz & Hedder 2022; Euritt 2023). Second person is
not a neutral pronoun count: it carries normative force (Orvell, Kross & Gelman 2017) and raises involvement
independently of content (Cruz, Leonhardt & Pezzuti 2017).

The composite was one outcome of a wider search. Three batteries were run on full episodes: a style battery
(othering pronouns, certainty, hedging, questions, negation, first person, concreteness from Brysbaert norms,
moral-emotional words after Brady et al. 2017), a wider register battery (fillers, contractions, complexity,
stance, temporal orientation), and a lexical outrage battery operationalised from the Sobieraj and Berry (2011)
categories (vulgarity, intensifiers, absolutes, contempt, insults, catastrophe) plus out-group references
following Rathje, Van Bavel and van der Linden (2021). All word lists other than the Brysbaert norms, the
moral-foundations dictionaries and the LDNOOBW profanity list were written for this project and are given in
full in the appendix.

## 3. Validation of the measure

*Discriminant.* Address separates left- and right-leaning shows: correlation with the host's DIME score 0.43
(Spearman 0.45), between-side difference 0.78 standard deviations (p ≈ 1e-6), area under the curve 0.80. The
hostility measures show no left-right difference (insults r = 0.00 with ideology), so the sides differ in how
they address the listener rather than in outrage (contrast Berry & Sobieraj 2014; Young 2020).

*Convergent.* The parser-based and dictionary imperative counts agree, and both agree with an independent
register tagger (MFTE imperatives r = 0.51, second person r = 0.43 with `dir_z`). The show's LLM ideology label,
its host's DIME score and the party identification of its own survey listeners form a consistent triangle.

*Stability.* A within-show event study around eight political shocks in 2020–2025 with placebo dates shows no
movement in address; within-show address is flat across the 75 topics of a topic model. Address is a trait of a
show, not a reaction to the news.

*Robustness of construction.* Imperatives were decomposed into epistemic, attention and action subtypes and
second person into deontic, addressive and filler uses; guest-directed "you" was separated from
listener-directed "you" and an alternative declarative-only score computed; an opening-versus-body ratio checks
for scripted intros. None changes the ranking materially.

## 4. Linking shows to listeners

The Kettering-Gallup *Democracy for All* survey (Year 1, n = 20,338) asks respondents to name, in their own
words, where they get news. Verbatim answers were split into named sources and matched exactly to corpus show
titles; a host-name expansion is reported only as a sensitivity. Exposure is the mean `dir_z` of the corpus
shows a respondent named. Outcomes are composites standardised on the full survey, never on the listener
subsample: an eight-item institutional cynicism scale (α = .84), following the trust and cynicism literature
(Cappella & Jamieson 1997; Hetherington 2005; Citrin & Stoker 2018); election distrust (Q33E reversed plus Q32C),
motivated by the malleability of election confidence (Berlinski et al. 2023; Clayton et al. 2021); efficacy,
economic grievance and disaffection composites; and education.

## 5. The political-content frame

Not every chart show is a political show. A show's political-content density is the share of its passages that
the ideology instrument's first stage marks as political with three or more political terms. The frame is
density ≥ 15%, the corpus's tenth percentile, which excludes three general-interest shows (Rogan 3%, Ryan 5%,
Kelly 7%). Within the frame there are 56 shows and 792 matched listeners. Models regress each outcome on
exposure with party, political attention, age and education as controls. Because populism and address could be
confounded (Mudde 2004; Krämer 2014), the populism dictionary of Rooduijn and Pauwels (2011), Fawzi-style
(2019) anti-media references, and passage-level ideological intensity are added as show-level covariates in a
second specification. Results are checked across cutoffs from 8% to 30% and with each show left out in turn.

## 6. Inference

Respondents cluster within shows and there are few shows, so every estimate reports CR2 standard errors with
Satterthwaite degrees of freedom (Pustejovsky & Tipton 2018) as the primary p-value and a Rademacher wild cluster
bootstrap with 1,500–2,000 draws alongside (Cameron, Gelbach & Miller 2008), plus a show-level permutation test
as a low-power third check and Benjamini-Hochberg correction within outcome families. Seeds are fixed.

## 7. Pre-registration and the status of the findings

Following Nosek et al. (2018), the confirmatory tests were frozen (text hashed, alias tables locked) before the
relevant data were touched. The corpus-based confirmatory tests failed on their own criteria: the free-text
democracy item, and the within-frame test on 150 fresh corpus listeners (b = −0.055). The political-content
frame was defined afterwards and its results are therefore exploratory: within the frame, address predicts
election distrust (+0.27 SD per SD, CR2 and wild p < .001; +0.19, p = .018 net of populism and intensity),
institutional cynicism (+0.22), disaffection (+0.16) and economic grievance (+0.18), and address-heavy shows
reach less-educated audiences (−0.21 SD in education, replicated in four samples). These are the pre-registered
hypotheses for Year 2, with the content frame, election distrust as primary outcome, and populism and intensity
as covariates specified in advance.

---
**Disclosure note.** Dropping the out-of-frame test from the main text is defensible because those ten shows are
outside the political corpus, but it was pre-registered and it failed; it has to appear in the disclosure section
or supplement, alongside the other two failures. Otherwise the paper reports the exploratory frame result without
the confirmatory test that motivated the frame.

**References cited here** (full list in `lit_review_background.md`): Berlinski et al. 2023; Berry & Sobieraj
2014; Biber 1988; Brady et al. 2017; Cameron, Gelbach & Miller 2008; Cappella & Jamieson 1997; Citrin & Stoker
2018; Clayton et al. 2021; Cruz, Leonhardt & Pezzuti 2017; Dibble, Hartmann & Rosaen 2016; Euritt 2023;
Fairclough 1989; Fawzi 2019; Hetherington 2005; Horton & Wohl 1956; Hutchby 1996; Krämer 2014; Montgomery 1986;
Much et al. 2026; Mudde 2004; Nosek et al. 2018; Orvell, Kross & Gelman 2017; Pustejovsky & Tipton 2018; Rathje,
Van Bavel & van der Linden 2021; Rooduijn & Pauwels 2011; Scannell 1996; Schlütz & Hedder 2022; Sobieraj & Berry
2011; Wirtschafter 2023; Young 2020.
