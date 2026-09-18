# Methods, fully specified (address measure and political-corpus survey findings)

Every list, threshold and formula below is taken from the code as run. Word lists marked **(ours)** were written
for this project; everything else is a published instrument.

## 1. Corpus

**Sampling frame.** The Apple Podcasts US chart for the *Politics* subcategory of *News* (genre id 1527,
storefront `us`, 250 ranks, the deepest Apple serves), pulled and frozen on 13 July 2026. Each chart entry was
resolved through the iTunes Lookup API to its RSS feed, and every feed was fetched and cached so that a re-run
touches no network.

**Inclusion rules**, applied in this order, first failure recorded with evidence (counts in parentheses):
R1 no feed URL (2); R2 feed unreachable after retries (2); R3 feed unparseable (1); R4 no audio enclosures (0);
R5 not US politics: publisher blocklist, title/description patterns, non-English feed language, curated manual
list, minus exceptions (14); R6 under 10 hours of audio available (12). 31 excluded, 219 retained.

**Episode selection.** A priority ladder over the episode table: P0 is a census of every episode published before
1 January 2018; bands P1..Pk then take k episodes per show per calendar quarter, k up to 30. Within a band,
episodes are ranked by a frozen hash of (seed, episode id), so raising k appends and never reshuffles; within a
band rows are round-robined across shows. The analysis corpus is the transcripts on disk at the time the
register measures were frozen: 194 shows with at least one scored passage (204 shows had transcripts; the
reference set is the 194 present in the passage table), 31,191 episodes, about 25,900 hours of audio.

**Transcription.** faster-whisper `BatchedInferencePipeline` on the CTranslate2 conversion of Whisper
large-v3-turbo, float16, batch size 64, English forced, Silero voice-activity detection, no diarisation and no
alignment pass. Output per episode: timestamped segments (JSON) and plain text. Advertisements were retained:
the ideology instrument the corpus adopts scores whole episodes without sponsor removal, and in this corpus
2.2% of passages carry ad markers, of which about 17% are politically themed, so the residue is about 0.4% of
passages.

**Passages.** Each transcript is cut into windows of at most 750 characters at sentence boundaries (sentences
longer than the cap are hard-split); fragments under 120 characters are dropped. This is the unit on which the
ideology instrument was validated. The full passage table has 1,378,017 passages.

**External ideology anchors, for validation only.**
- *DIME.* Host names were extracted from feed metadata (itunes:author, itunes:owner, RSS author, then the show
  title), classified person/publisher/self-reference, resolved to Wikidata entities by exact-label SPARQL
  match with type and occupation filters, and matched to the DIME recipient and contributor files (1979–2024) on
  name plus occupation/employer evidence. A show's `avg_host_cfscore` is the mean CFscore of its matched hosts;
  112 of the 194 shows have one.
- *Brookings.* The Political Podcast Project's lean labels (Wirtschafter 2023), fuzzy-joined on show name
  (rapidfuzz WRatio ≥ 90 accepted; 75–89 reviewed by hand).

## 2. Passage-level political content and ideology (the LLM instrument)

Every passage in a capped sample (up to 40 passages per show-quarter, drawn round-robin across episodes in a
frozen order; 152,152 passages scored) was labelled with the two-stage rubric of Much et al. (2026), prompts
copied verbatim (`step4_ideology/prompts/ideology.md`), model `gpt-5.4-mini-2026-03-17` (a dated snapshot, never
an alias). Stage 1: is the passage political (yes/no). Stage 2, for "yes" only: strongly / moderately / slightly
liberal, moderate, slightly / moderately / strongly conservative. Three show-level quantities come from this:

- **Side** = (share of political passages labelled conservative) − (share labelled liberal), each weighted by
  passage word count. Range −1 to +1. This is the "LLM side label".
- **Ideological intensity** (`inten`) = share of a show's political passages labelled anything other than
  *moderate*. Used as a covariate in the frame models.
- **Political-content density** (the frame variable) = the word-count-weighted share of a show's sampled
  passages that stage 1 marks political. Corpus median 0.85; the frame cut is the corpus 10th percentile, 0.644.
  (An earlier hand-written political-term dictionary, retired on 15 September 2026, correlates 0.73 with this;
  see `handoff/results/frame_llm_density.md`.)

## 3. Register measurement

All register features are computed on the same 750-character passages (passages under 40 word tokens skipped;
tokens are runs of letters and apostrophes in lower-cased text) and expressed as **counts per 10,000 words**.
Show-level rates are passage rates averaged with passage word count as the weight, which equals the show's
overall rate. Three batteries were run; the address measure is built from the first.

### 3.1 The pinned feature set (the one dir_z uses)

Computed once with spaCy `en_core_web_sm` (parser on, NER and lemmatiser off); the exact script is frozen as
`step7_audience/inputs/remeasure2.py`.

- **Syntactic imperatives** (`imper_syn`; rule ours, no word list). A token counts if all hold: part of speech
  VERB with fine tag VB (bare form); dependency label ROOT, conj, ccomp or advcl; no child with label nsubj,
  nsubjpass, expl, aux or auxpass; and the preceding token in the sentence is not *to*, *not* or *never*.
- **Dictionary imperatives** (`imper_rx`) **(ours)**: look, listen, think about it, consider, remember,
  imagine, ask yourself, understand, realize, let me tell you. Whole-word, case-insensitive.
- **Second person** (`you_rx`): you, your, yours, yourself. (A parse-based version, pronouns carrying the
  morphological feature Person=2, was computed alongside; correlation with the dictionary version across shows
  is above 0.9.)
- Also in the pinned set, used in the hostility comparison and decompositions: possibility modals (might, may,
  could, would, should, can, tag MD); extended Moral Foundations Dictionary scores (eMFD: total, care,
  authority, sanctity) and MFD 2.0 hits (published dictionaries); vulgarity from the LDNOOBW list (published)
  and from a short list **(ours)**: damn, hell, shit, fuck, fucking, crap, ass, bastard, bullshit; religious
  words **(ours)**: god, lord, jesus, christ, faith, pray, church, evil, moral, sin; hedges **(ours)**: maybe,
  perhaps, might, possibly, probably, seems, apparently, sort of, kind of, i think, i guess, arguably, somewhat;
  certainty **(ours)**: definitely, certainly, obviously, clearly, undeniably, absolutely, unquestionably, of
  course, literally, no question.

### 3.2 The address score

For each of `imper_syn`, `imper_rx`, `you_rx`: show rate → z-score against the mean and population standard
deviation (ddof = 0) of the 194 corpus shows (frozen in `handoff/prereg/dirz_reference_scale.csv`; reference
means are roughly 45 imperatives, 20 dictionary imperatives and 190 second-person forms per 10,000 words).
`dir_z` is the unweighted mean of the three z-scores. Any new show, including the out-of-frame shows, is scored
against the same frozen reference. The three components correlate 0.34–0.39 with DIME individually; the
composite 0.43.

### 3.3 The wider register battery (`measure_register2.py`, per episode, episodes under 500 words skipped; all lists ours)

fillers: um, uh, erm, hmm, like, y'know, basically, literally, actually, right · discourse markers: so, well,
now, anyway, look, listen, okay, alright, see · imperative phrases: the ten above plus picture this, here's the
thing · superlatives/absolutes: best, worst, greatest, biggest, most, least, never, always, every, everyone,
nobody, all, none, huge, massive, enormous · quantification: a number optionally followed by percent, %,
million, billion, trillion, thousand · reported speech: said, says, claimed, claims, told, argued, argues,
wrote, stated, according to, reportedly, allegedly · religious: god, lord, jesus, christ, faith, pray, prayer,
church, biblical, scripture, sin, soul, holy, blessed, evil, moral · institutional: congress, senate, house,
court, agency, department, administration, committee, federal, legislation, bill, policy, regulation, statute
· contractions: any word ending in 's, 't, 're, 've, 'll, 'd, 'm · false starts: a word immediately repeated ·
root type-token ratio (types / √tokens) · mean word length · share of words with 7+ letters · words per sentence
· past/future orientation: log((was, were, had, did, went, came, took, made, got, said, thought + 1) / (will,
going to, gonna, shall, would, could, might, may + 1)).

### 3.4 The style battery (`measure_style.py`, per episode)

othering = log((they, them, their, theirs, themselves + 1) / (we, us, our, ours, ourselves + 1)) · direct
address: you, your, yours, yourself, you're · first person: i, me, my, mine, myself · certainty: definitely,
certainly, obviously, clearly, undeniably, absolutely, unquestionably, without a doubt, no question, of course,
literally · hedges: as in 3.1 · question rate: question marks per 10,000 words · negation: not, never, no,
nothing, nobody, none, cannot, can't, won't, don't · moral-emotional **(ours, after Brady et al. 2017)**: evil,
wrong, right, wicked, corrupt, betray, betrayal, shame, shameful, disgrace, disgraceful, outrage, outrageous,
immoral, injustice, unjust, hypocrite, hypocrisy, coward, cowardly, destroy, attack, fight, war, hate, hatred,
greed, greedy, liar, lie, lies, fraud, criminal, abuse · concreteness: mean Brysbaert et al. (2014) rating over
covered tokens (published norms).

### 3.5 The outrage battery (`measure_outrage_lexical.py`, categories from Sobieraj & Berry 2011; regexes ours)

vulgarity: fuck*, shit*, goddamn, damn, asshole*, bastard*, bitch*, piss(ed/ing), bullshit, dumbass, jackass ·
intensifier: absolutely, completely, utterly, totally, literally, insane, insanity, outrageous, outrage,
ridiculous, absurd, disgusting, disgraceful, shameful, pathetic, appalling, unbelievable, horrific · absolute:
never, always, every single, nobody, no one ever, everyone knows, not one, zero chance, complete(ly) failure ·
contempt: so-called, supposedly, allegedly, apparently, if you can call, quote unquote, whatever that means ·
insult: idiot*, moron*, clown*, liar*, lying, corrupt, crook*, thug*, coward*, grifter*, hack*, stupid, fraud*,
traitor*, scum* · catastrophe: destroy*, destruction, collapse, end of democracy/america/the republic,
existential threat, civil war, dictatorship, tyranny, authoritarian · out-group references (after Rathje et al.
2021), with the show's side fixed externally from the Brookings label: left terms democrat(s), democratic
party, liberal(s), progressives, biden, kamala, pelosi, schumer, aoc, squad; right terms republican(s), gop,
conservative(s), trump, maga, desantis, mcconnell, maga republicans; bare "left"/"right" deliberately excluded
· speech rate in words per second from segment timestamps, as a partial stand-in for the vocal-escalation
category Sobieraj and Berry coded from audio.

## 4. Validation of the address measure

- **Against the host's DIME score** (112 shows): Pearson r = 0.43, Spearman 0.45. Between-side gap
  (side > 0 vs side < 0): d = +0.78, p ≈ 1e-6; classifying side from dir_z alone gives AUC 0.80.
- **Hostility does not separate the sides**: insult rate r = 0.00 with DIME; emphatics r = −0.29 (right-leaning
  shows use fewer); the populism and anti-media rates r = −0.04 and −0.06 with DIME.
- **Independent tagger.** MFTE (Le Foll 2021), run on 600 episodes from 164 shows: its imperative tag (VIMP)
  correlates 0.51 and its second-person tag (PP2) 0.43 with dir_z. A full-corpus MFTE run is in progress to
  rebuild the measure entirely from that published tagger.
- **Validation triangle at the listener level**: the LLM side label, the host DIME score and the party
  identification of a show's own survey listeners agree pairwise across matched shows.
- **Stability.** Event study on 18,633 dated episodes: address in the 70 days after each of eight shocks (2020
  election 3 Nov 2020; 6 Jan 2021; Dobbs 24 Jun 2022; 2022 midterm 8 Nov 2022; Trump assassination attempt 13
  Jul 2024; Biden withdrawal 21 Jul 2024; 2024 election 5 Nov 2024; inauguration 20 Jan 2025) versus the 70 days
  before, with show fixed effects, compared with the same statistic at every placebo date from March 2019 to
  May 2026; the real shifts sit inside the placebo distribution. Topic invariance: within-show address is flat
  across the 75 topics of the topic model.
- **Decompositions (all lists ours).** Imperative verbs split into *epistemic* (understand, realize, know,
  think, remember, consider, imagine, notice, recognize, believe, ask, wonder, forget, bear, keep, trust,
  picture, assume, note, realise), *attention* (look, listen, see, watch, hear, check, wait, hold, stop, hang,
  guess, mark, behold) and *action* (go, get, call, vote, donate, subscribe, share, sign, join, buy, support,
  fight, stand, take, make, do, give, send, read, visit, …). Second person split, mutually exclusively, into
  *you know* filler (dropped), *question* (sentence ends in ?), *named-person* (sentence contains a PERSON
  entity, the guest-directed cue) and *declarative*; *deontic you* is "you" followed by need, have, should,
  must, gotta, got, ought, better, want, can't, cannot; *explicit audience address* is you guys, you all,
  y'all, everybody, everyone, folks, listener(s), audience, if you're listening/watching, those of you. An
  alternative score with declarative-only "you" (`dir_z_declar`) and an opening-versus-body ratio (second
  person in the first 10% of an episode over the 30–100% body) are reported; neither changes the show ranking.

## 5. Survey and linkage

**Data.** Kettering Foundation / Gallup *Democracy for All*, Year 1, n = 20,338 US adults, weighted file
`KETTERING_DATA_Y1_WEIGHTED`.

**Naming sources.** Q17_1–Q17_3: "Please name the top three programs, channels, podcasts, websites,
influencers, or other outlets from which you get news and information." Answers were lower-cased and trimmed
and stacked to one row per mention. **Strict match** (the sample of record): a mention that is the exact title
of a corpus show, checked by hand (`kettering_strict.csv`), mapped to the show id case-insensitively: 385
respondents, 412 mentions, 51 shows. A host-name expansion (544 respondents) is reported only as a sensitivity.

**Exposure.** The mean dir_z of the corpus shows a respondent named; cluster = the alphabetically first show
named.

**Controls** (every model): party identification (Q41 Republican/Democrat/Independent, with Q43 leaners
assigned to a party; dummies), political attention (Q19, "How much attention do you pay to government and
political matters?"), age, education (EDU, highest level completed), and in the frame models a count of
right-wing platforms used (Truth Social, Rumble, Parler, Gab: Q20C/G/E/H at "often" or "sometimes") plus the
show's lean (right = 1).

**Outcomes**, every composite z-scored on the full 20,338, never on the listener subsample; survey codes ≤ 0
treated as missing:
- *Institutional cynicism, 8 items* (α = .84), each reversed so higher = more cynical, averaged: Q35E "Our
  government's laws and policies are careful to uphold freedom and justice for all"; Q35C "Government includes
  many people who have similar backgrounds and experiences to my own"; Q34B "If I had a concern, I know how to
  share it with my elected officials"; Q33H confidence in freedom of the press; Q31 "I trust that U.S. political
  leaders will be held accountable to our nation's laws and constitution"; Q33E confidence in how elections are
  administered; Q35A "Government decisions generally reflect what the majority of people want done"; Q35D
  "Government decisions usually attempt to serve the best interests of citizens, even when I disagree".
- *Election distrust*: mean of z(6 − Q33E) and z(Q32C "It is reasonable to assume that people who oversee
  elections acted improperly when election outcomes are surprising").
- *Economic grievance (6 items)*: inequality is unfair (Q32H), society should reduce it (Q32I), government
  should meet basic needs (Q36A), against wealthy influence in politics (Q36I), government should deliver more
  (Q32J reversed), cannot afford necessities (Q15A); income controlled.
- *Disaffection (8 items)*: life ladder (Q1, reversed), local-community identity (Q14B, reversed), church
  attendance (Q44, reversed), trust in faith leaders and in family/friends as sources (Q18A, Q18E, reversed),
  contacting officials is ineffective (Q24B), protest is effective (Q24C), election overseers acted improperly
  (Q32C). Formative bundle, mean inter-item r ≈ .12.
- *External efficacy*: the Q34 items on being able to reach and influence officials (Q34A–D).
- *Exclusivity*: names no institutional outlet (regex over the three verbatims: cnn, fox, nbc, abc, cbs, msnbc,
  npr, pbs, bbc, new york times, nyt, washington post, wall street, wsj, reuters, associated press, ap, usa
  today, local news, newspaper, the hill, politico, axios, bloomberg, cnbc, newsnation, newsmax).
- *Education* as an outcome (audience composition).

## 6. The political-content frame

Shows whose LLM political density is at or above the corpus 10th percentile (0.644). Among shows with matched
listeners this leaves **49 corpus shows and 456 listeners** (out of 54 shows and 535 listeners unframed);
the excluded matched shows are The Arena, Candace, Michael Berry, New Yorker Radio Hour and Volts. Because the
frame changes the estimates very little inside the corpus (see §8), it is reported as a robustness restriction.
The pooled analysis in the record, which added six chart-absent shows, is outside this paper's scope.

**Model.** Outcome on dir_z (in reference-scale SD units) plus the controls above. **Second specification**
adds three show-level covariates: the Rooduijn & Pauwels (2011) populism rate (stems elit*, consensus*,
undemocratic*, referend*, corrupt*, propagand*, politici*, *deceit*, *deceiv*, *betray*, shame*, scandal*,
truth*, dishonest*, establishm*, ruling*, per 10,000 words), anti-media references (the media, mainstream
media, news media, the press, journalist(s), reporter(s), msm, legacy media, corporate media, fake news, and
named outlets, weighted by VADER negativity of the passage; after Fawzi 2019), and ideological intensity as
defined in §2. **Sensitivity**: density cut-offs from .08 to .30; leave-one-show-out.

## 7. Inference

Respondents are clustered in shows; the number of shows is small. For every estimate: (a) CR2 (Bell–McCaffrey
bias-reduced) cluster-robust standard error with Satterthwaite degrees of freedom (Pustejovsky & Tipton 2018),
the primary p; (b) wild cluster bootstrap under the null with Rademacher weights, one draw per show,
1,000–2,000 replications (Cameron, Gelbach & Miller 2008); (c) where noted, a show-level precision-weighted
permutation test (4,000 permutations of the exposure across shows) and Benjamini–Hochberg correction within
outcome families. NumPy seed 5 for the pre-registered scripts, 7 for the scans. Pre-registered tests use
one-sided α = .05 and require both CR2 and wild p < .05.

## 8. Pre-registration and status

- **Q28 test**, frozen 2 September 2026 19:15 UTC. Free-text item "What does democracy mean to you?"; M1 =
  out-group pronouns (they, them, their, theirs) as a share of all personal pronouns; M2 = negation words per
  word; M3 = VADER compound valence; M4 = a cynicism lexicon (corrupt*, broken, rigged, lie/lies/lying, joke,
  sham, fake, illusion, supposed, elite(s), control(led), power, fail*, used to, no longer, rich, money,
  doesn't work, not really, in theory) **(ours)**. Answers under five words dropped. **Failed** on the primary
  criterion (M1 one-sided p ≥ .05).
- **Out-of-frame test**, frozen 2 September 2026 21:08 UTC before any audio existed (10 chart-absent shows, 749
  respondents; alias table and patterns frozen with SHA-256). **Failed**: b = −0.001, CR2 df 3.5. Outside the
  corpus, so outside this paper's frame, but it must appear in the disclosure section.
- **Within-frame test**, frozen 3 September 2026 08:28 UTC: 150 respondents naming a corpus show who were not in
  the 385 discovery respondents, 10 shows. **Failed**: b = −0.055, CR2 df 3.0, one-sided p .57.
- **Exploratory, post hoc**, corpus listeners only (discovery sample plus the within-frame pocket, 535
  listeners of 54 shows). Election distrust on address: +0.224 SD per reference SD (CR2 p .004, wild .011)
  unframed; +0.222 (.004 / .023) inside the LLM-density frame (49 shows, 456 listeners); the same +0.22 at
  every frame cut-off from the 5th to the 25th percentile. Cynicism +0.206 (.020 / .022) unframed, +0.182
  (.023 / .039) framed. Education (audience composition): −0.30 SD of the survey's education distribution per SD of
  address (.006 / .003), the same in every sample and sharper among respondents who named the show first, but
  see below: it too is not separable from intensity inside the corpus (−0.12, n.s., net of intensity). **The election-distrust association survives populist vocabulary but not ideological intensity**:
  adding the Rooduijn–Pauwels rate leaves it at +0.216 (p .008 / .027) and populism itself predicts nothing;
  adding intensity (the share of a show's political passages labelled anything but moderate, r = +0.44 with
  address across shows) cuts it to +0.06 in the frame (n.s.) and +0.12 unframed (p .13), while intensity
  predicts election distrust on its own (+0.32 in the frame, +0.28 net of address). Cynicism behaves the same
  way (+0.14, p .06–.08 net of both). The paper reports address and intensity as inseparable in this corpus and
  names intensity, not populism, as the rival. These are the Year-2
  pre-registered hypotheses: LLM-density frame fixed in advance, election distrust primary, populism and
  intensity as covariates, interview date requested.
