"""Plain-language speaker notes for Talking_at_the_Audience_slides.pptx.
Bullet points, numerals for numbers, acronyms spelled out on first use, methods described in ordinary words."""
from pptx import Presentation

N = {}

N[1] = """SAY
• Political podcasts: a huge news source we have barely measured.
• 3 claims, in order:
  1. Left and right differ in HOW hosts address listeners, not in hostility.
  2. People who get political news from podcasts are distinctively cynical about institutions.
  3. The way a host talks to you predicts who you are.
• Everything that follows is evidence for those 3 claims."""

N[2] = """SAY
• The largest political shows out-draw cable news programs.
• Nearly all research on partisan media is about television and print: incivility, outrage, horse-race coverage.
• Podcasts are a different form: long, intimate, 1 voice speaking directly to you for hours a week.
• Common assumption: right-leaning shows are angrier. Never measured at scale.
• 2 gaps:
  – Content: no transcribed, validated collection of the field.
  – Audience: nobody could link a specific show's language to that show's own listeners.
• The survey linkage is what makes this paper different."""

N[3] = """SAY
• Question 1: how do left- and right-leaning hosts differ in how they talk?
• Question 2: who listens to political podcasts, compared with other news consumers?
• Question 3: does a show's way of addressing the listener predict who its audience is and what they believe?
• Flag now: question 3 is where the work is exploratory. Questions 1 and 2 are on firmer ground."""

N[4] = """SAY (optional if short on time)
• Media malaise / spiral of cynicism (Robinson 1976; Cappella & Jamieson 1997; Ladd 2012): news diets outside institutional journalism go with lower trust. We find that.
• Incivility and outrage (Mutz & Reeves 2005; Sobieraj & Berry 2011): the right is more hostile. We do NOT find that.
• Para-social interaction (Horton & Wohl 1956): direct personal address builds a 1-sided relationship with a performer. Fits what we see.
• Populist communication (Rooduijn & Pauwels 2011; Fawzi 2019) enters as a competing explanation we test alongside, not one we rule out."""

N[5] = """SAY
• Starting point: the Apple Podcasts US Politics chart, ranks 1–250, frozen on 13 July 2026.
• Filter applied in a fixed order: 250 → 219 (14 not US politics, 12 with under 10 hours of audio, 5 with no usable feed) → 204 in the final set.
• 29,421 episodes transcribed; 1.38 million passages of 750 characters; median episode 47 minutes.
• Scope: the biggest political shows (Shapiro, The Daily, Tucker) are filed under other Apple categories, so they are not in the collection. We bring 6 of them back later as an outside test.

WHAT I DID, IN PLAIN TERMS
• Transcription: an open-source speech-to-text model (Whisper, the large-v3-turbo version) running on 2 university GPU machines at about 100 times real time; roughly 260 GPU-hours in total. Audio only, English only, no attempt to tell speakers apart.
• Which episodes: every episode before 2018 (there are few, and old feeds disappear), then a fixed number of episodes per show per quarter from 2018 on, chosen by a fixed random rule so the sample can be extended without reshuffling.
• Ideology labels: a language-model classifier read 152,152 passages and labelled each as left, right or neither; a show's score is the balance across its passages. Hosts were also looked up in a database of political donations (DIME) to get an independent ideology score for 112 shows."""

N[6] = """SAY
• Kettering–Gallup Democracy for All, Year 1 (July–August 2025): 20,338 respondents; 45% from a probability-based panel; weighted to the population.
  – Asked people to type their top 3 news sources → 51,082 typed answers. People typed the names of shows. That is the key asset.
  – Matched: 385 respondents by exact show title, 150 more by catching misspellings and abbreviations, and 290 listeners of 6 large political podcasts outside the collection = 792 listeners across 56 political podcasts.
• Pew American Trends Panel, 3 waves (July 2024 – March 2025): about 9,000 people per wave. Podcast use asked in mid-2024, trust in institutions asked 2 to 8 months later. No show names, but time order.

WHAT I DID, IN PLAIN TERMS
• Outcome in Kettering: 8 survey questions about institutions (do leaders get held accountable, does government serve people like me, are elections run well, is press freedom working, and so on), each put on a common scale, averaged into one cynicism score. The 8 items hang together well (reliability 0.84).
• News-diet groups for everyone in the survey: sorted each typed source into podcast, mainstream outlet, or social platform, then grouped people as podcast-only, podcast plus mainstream, mainstream-only, or platform-only."""

N[7] = """SAY
• Listener-directed address = average of 3 measures, each a rate per 10,000 words, put on a common scale across shows:
  1. grammatical commands: a bare verb starting a clause with no subject ("Look at this", "Remember that"), found with a sentence parser
  2. a fixed list of instruction words: look, listen, remember, understand, think about it
  3. the words you, your, yourself
• Read the example on the slide.
• Every measure existed before this study; we wrote no word list ourselves.
• Breaking the measure apart: "you need to / you should" is the piece most tied to host ideology (correlation +0.44); the filler "you know" is unrelated (0.00); a standard persuasion measure from linguistics is also unrelated (−0.12). It is talking TO the listener that carries the signal.

WHAT I DID, IN PLAIN TERMS
• Scored every 750-character passage, then averaged up to the show, weighting by passage length, and expressed each show relative to the other 194 shows.
• Advertising was flagged but kept in for these measures (removing it is only needed for the topic model).
• Known limit: in interview shows, "you" is often the guest. This collection is mostly monologue and co-host shows, and we add a format control where it matters."""

N[8] = """SAY
• 4 checks on the ideology label:
  – Against hosts' political donations: correlation 0.68 (112 shows).
  – Against editorial ratings: agreement 0.75 (25 shows); against expert ratings: 0.77 (26 shows).
  – Against the party of each show's own listeners in the survey: correlation 0.91 across 22 shows; the label picks the audience's majority party correctly in 21 of 22.
• Left-labelled shows average 76% Democratic audiences; right-labelled shows 91% Republican.
• Text, money and listeners agree.

WHAT I DID, IN PLAIN TERMS
• For each show with at least 5 listeners in the survey, took the share of its listeners who are Republican minus the share who are Democratic, and correlated that with the show's label.
• Not yet done: having 2 human coders check 200 individual passages against the language model. Planned before submission."""

N[9] = """SAY (optional; keep for a methods-minded room)
• The problem: the exposure is a property of the SHOW, and a few shows hold most of the listeners (the 4 largest = 47% of the 792). Treating 792 listeners as 792 independent observations overstates how much we know; really we have closer to 9 independent units.
• What we do about it, 3 ways:
  1. Clustered standard errors with a small-sample correction. "Standard error" = how much an estimate would wobble on a fresh sample. Clustering means listeners of the same show are treated as one group. The correction reports the honest effective number of groups (about 9), and p-values are computed against that.
  2. A bootstrap check. In plain terms: pretend address has no effect; take each show's leftover errors from that model and flip their signs with a coin toss, one coin per show; rebuild a fake outcome; refit; repeat 2,000 times. That shows what results look like when nothing is going on, respecting that listeners of the same show move together. The p-value is how often a fake result beats the real one.
  3. A shuffle test at the show level: randomly reassign address scores across shows and re-run.
• The first two must agree; they do on every estimate.
• The outside test was written down and time-stamped before any audio was fetched."""

N[10] = """SAY
• Chart: how strongly each speech measure correlates with the host's donation-based ideology.
  – Combined address measure +0.43; grammatical commands +0.39; you/your +0.39; "they/them" out-group pronouns +0.29
  – Insult 0.00; hedging −0.13; profanity −0.17
• Right-leaning hosts speak to the audience in the second person and issue more instructions. They are not more insulting.
• Size of the left–right gap on address: 0.78 standard deviations (a standard deviation = the typical spread among shows), after controlling for format. Very unlikely by chance (p < 0.0001).
• Insult tracks how EXTREME a show is (correlation +0.55) but not which side it is on.
• A classifier using only speech-style features tells left from right shows 80% of the time between shows, and barely above chance within a show → style is a property of the show.

WHAT I DID, IN PLAIN TERMS
• Correlated each show-level measure with the host's donation score (112 shows with a score).
• Compared left shows (125) with right shows (69) on each measure; to get p-values, shuffled the left/right labels across shows 10,000 times and asked how often a gap as big as the real one appears. Corrected for testing many measures at once.
• Format control: right shows are 84% solo-hosted, left shows 58%, so every comparison was re-run holding solo vs multi-host constant.
• Classifier: a simple logistic regression on the style measures, always tested on a show it had not seen."""

N[11] = """SAY
• 8 political shocks 2020–2025: 2 presidential elections, 1 midterm, the inauguration, the assassination attempt, Biden's withdrawal, January 6, the Dobbs decision.
• Within each show, address does not shift around any of them, on either side; every change is at most 0.08 standard deviations.
• The 2024 election "shift" is no bigger than what we see at 60 randomly chosen dates (p = 0.57).
• Across 75 topics, a show's address barely varies by subject (spread of 0.04 standard deviations); the most "you"-heavy passages are ad reads.
• Hosts do not ramp it up before elections and do not change it by subject. It is how they talk.
• Why it matters: a stable trait can steer who chooses a show; a reaction to the news could not.

WHAT I DID, IN PLAIN TERMS
• Event study: for each show, compared address in the 8 weeks before and after each event, with each show as its own control; ran the same comparison at 60 random dates to see what "nothing happened" looks like.
• Topics: a standard topic model (75 topics) on the ad-cleaned passages; adding topic shares to a model of address explains an extra 0.5% of variation."""

N[12] = """SAY
• People who name a political podcast as a news source score 0.22 standard deviations higher on the cynicism scale than other news consumers (18,257 people; p = 5 × 10⁻¹³).
• Controls: party, political attention, age, education, gender, race, income, social-media hours, platforms used, urban vs rural, survey mode, number of sources named. The estimate moves 3% across the whole list.
• Diet groups versus mainstream-only: podcast-only +0.27; platform-only +0.18; podcast plus mainstream +0.13.
• Within Democrats +0.19; within Republicans +0.15. Lower education +0.34; higher education +0.35 (no difference, p = 0.91) → not "podcasts prey on the less educated".
• The gradient: each additional mainstream source a person names is worth −0.10 standard deviations; each additional podcast +0.08.
• Podcast-only exceeds platform-only by +0.14 standard deviations, give or take 0.05; a formal test rejects "these two are the same".

WHAT I DID, IN PLAIN TERMS
• Ordinary linear regression of the cynicism score on the diet group plus the controls, using the survey weights, with standard errors that do not assume equal spread across people.
• "Mainstream" = a fixed list of 27 outlets (CNN, Fox, NPR, New York Times, and so on).
• Checked that unrelated items (tolerance of political violence, loneliness) do not move, which they do not."""

N[13] = """SAY
• Pew: each step up in how often someone got news from podcasts in mid-2024 goes with, months later:
  – trust in national news −0.066 (September 2024) and −0.083 (March 2025)
  – trust in local news −0.067 and −0.083
  – trust in social media +0.140 and +0.154
  – trust in friends and family +0.011 and +0.024 (no real change)
• People who said podcasts were their MAIN election-news source: trust in national news −0.34 standard deviations in March 2025.
• The SHAPE is the point: institutions down, informal channels up, personal relationships flat. A general "grumpy respondent" pattern could not do that.
• This is time order, not proof of cause. People choose their podcasts.

WHAT I DID, IN PLAIN TERMS
• Linked the same people across the 3 waves; regressed each later trust item on earlier podcast use with Pew's controls and weights.
• The stricter measure of exposure (podcasts as main source) gives effects 3 to 4 times larger."""

N[14] = """SAY
• Podcast-only listeners versus mainstream-only, on questions about democratic norms:
  – media should take direction from government: −0.35 standard deviations (they disagree more); expand presidential power −0.17; bar radicals from office −0.18; radicals should be allowed to protest +0.17
  – no more strongly partisan (0.00); not lonelier; not less likely to be registered
• Platform-only listeners: none of that civil-liberties pattern; instead weaker national identity, money strain, lower registration, more tolerance of political violence.
• Podcast-only = engaged and anti-institutional. Platform-only = disengaged.

WHAT I DID, IN PLAIN TERMS
• Ran the same weighted regression for all 115 usable survey questions, one at a time, for each diet group; corrected for the number of tests so that chance findings are filtered out (40 questions still differ for the podcast-only group)."""

N[15] = """SAY
• How much less educated is a show's audience, per 1 standard deviation more address, with party and show ideology held constant:
  – main collection −0.27 (54 shows, 535 listeners; p = 0.004 by clustered standard errors, 0.002 by bootstrap)
  – original discovery sample −0.29; newly matched listeners −0.16; the 6 outside political podcasts −0.08; everything together −0.21 (p = 0.006 / below 0.001)
• Unchanged when age, gender, income, race, urban–rural and attention are added (−0.28).
• Holds within left shows (−0.29) and within right shows (−0.19); holds when any of the 12 largest shows is dropped; at the show level the correlation is −0.57.
• It is education specifically: income, age and gender show nothing.
• People who name the show as their FIRST source show it more strongly (−0.35) than people who name it second or third (−0.20). That is what audience selection looks like.
• Reading: a fit between style and audience. Direct, instructive, second-person speech is the "involved" spoken register, and it reaches people with less schooling regardless of party. This is about who a style reaches, not what it does to them.

WHAT I DID, IN PLAIN TERMS
• Regression of each listener's education level on their show's address score, plus party and show ideology, with listeners grouped by show for the standard errors (small-sample corrected) and the coin-flip bootstrap (1,000 rounds) as a second check; repeated dropping one show at a time."""

N[16] = """SAY (optional)
• Pew, among people who regularly get news from online personalities (2,012 people), percentage-point change per step of podcast-news frequency:
  – feel a personal connection +4.1; follow or subscribe +6.0; "helped me understand" +5.8; get opinions from them +3.5; "their opinions agree with mine" +2.6
  – get basic facts from them +1.2 (not significant); "their news is different" no change
• "Personal connection" rises 26% → 31% → 32% → 39% across the frequency levels; same in both parties.
• The relationship and opinion items move; the information items do not. That is the listener-side fingerprint of direct address.

WHAT I DID, IN PLAIN TERMS
• 7 yes/no items, each regressed on podcast-news frequency with weights and controls; corrected for running 7 tests."""

N[17] = """SAY — say "exploratory" BEFORE the numbers
• Population here: political podcasts, defined by content = at least 15% of a show's passages are about politics (the collection's own 10th percentile), applied the same way to shows inside and outside the collection → 56 shows, 792 listeners.
• Per 1 standard deviation more address: election distrust +0.27 standard deviations (p below 0.001 both ways); institutional cynicism +0.22 (p = 0.005 / below 0.001); disaffection +0.16; economic grievance +0.18; relying on no institutional source +0.09 (not significant).
• Chart: blue = the 50 shows in the collection, red = the 6 large outside shows; they agree. Grey = the pooled estimate after also controlling for populist vocabulary and ideological extremity: election distrust +0.19 (p = 0.018) and cynicism +0.16 (p = 0.037) survive.
• Robust: between +0.20 and +0.29 at every content threshold from 8% to 30%; dropping any single show leaves +0.25 to +0.30. The 6 outside shows alone point the same way but are too few to stand on their own.
• Why exploratory: the outside test was written down in advance for 10 shows chosen by chart absence, and it did NOT pass on those 10. 2 of the 10 turned out not to be political podcasts by content (3% and 5% political passages). The content definition came AFTER that. Year 2 of the Kettering study will test it with the definition fixed in advance.

WHAT I DID, IN PLAIN TERMS
• Regression of each listener's outcome on their show's address score, controlling for party, attention, age, education, gender, race, income, urban–rural, use of right-wing platforms, the show's ideology, and whether the show is inside or outside the collection; listeners grouped by show for standard errors; bootstrap with 2,000 rounds.
• Election distrust = average of 2 items: "elections are administered well" (reversed) and "officials acted improperly when results were surprising".
• The 2 competing explanations: a published populism dictionary (words like corrupt, elite, betrayed) counted per 10,000 words; and ideological extremity = the share of a show's passages the language model labelled as clearly left or clearly right.
• Political content = share of 750-character passages containing 3 or more political terms; the collection's median show is 32%."""

N[18] = """SAY
• 3 tests were written down, with hypothesis, data, model and pass/fail rule, and time-stamped before their data were examined. All 3 failed on their own rules:
  1. Outside test: address → cynicism among 749 new listeners of 10 large shows absent from the chart. Estimate −0.001. Leaving out Rogan: +0.22, p = 0.08. Did not pass.
  2. Inside test: 150 new listeners of shows in the collection, found by catching misspellings. Estimate −0.06. Did not pass.
  3. A free-text question ("what does democracy mean to you"). p = 0.125. Did not pass.
• Everything after those tests (the content-based definition, the education result, the comparison with populist content) was found by searching, is labelled exploratory, and the search is disclosed in the paper.
• The 3 speech features (address, populist vocabulary, extremity) are correlated with each other at 0.4 to 0.5; with about 56 shows we can rank them only as far as the previous slide shows.
• Why this slide: a reviewer will find the time-stamped files. Better they hear it from us.
• The framing: an idea generated in the first sample → tested → not confirmed on the pre-set frame → recovered on a content-based frame → to be confirmed in Year 2."""

N[19] = """SAY
• Association, not effect: people choose their podcasts. The Pew time order is necessary but not sufficient. No sign that more listening goes with stronger attitudes; it does go with a sharper education pattern, which is what selection predicts.
• Scope: shows on the Politics chart plus the 6 largest political podcasts outside it; the content-based definition of "political podcast" was set after the fact and will be fixed in advance for Year 2.
• Address is sensitive to format: in interview shows "you" is often the guest. The left–right gap survives removing question-form "you"; comparing LEVELS across formats needs speaker separation, which the archived audio allows.
• The 3 speech features are correlated (0.4–0.5).
• Validation still open: passage-level ideology labels not yet human-checked; the advertising-detection test set (886 sentences) was labelled by a language model, human relabelling pending.
• Few effective clusters, reported honestly."""

N[20] = """SAY — same 3 claims as the opening
1. How the sides differ: in how hosts address the listener, not in hostility (gap of 0.78 standard deviations versus a correlation of 0.00); a stable trait of the show; validated against donations, experts and listeners' own party.
2. Who listens: +0.22 standard deviations of institutional cynicism, in 2 national panels, 1 with time order, in both parties, at every education level; civil-libertarian, not disengaged.
3. What address predicts: within political podcasts, hosts who talk AT the listener reach less-educated audiences (robust, 4 samples) who distrust elections and institutions (exploratory, to be confirmed in Year 2)."""

N[21] = """SAY
• Before submission: 2 human coders check 200 passages; human relabelling of the 886-sentence advertising test set; disclosure section; freeze the episode counts.
• Year 2 pre-registration: content-based definition fixed in advance; election distrust as the main outcome; populist vocabulary and extremity as controls; ask for interview dates so we can compare the same show across weeks.
• Measurement: separate speakers in the archived audio for interview shows; extend the collection to 10 episodes per show per quarter (running now).
• The causal study: a survey experiment. Play people the same segment with and without the second-person, instructive language, at random, then ask the trust and efficacy questions. Cheap, uses the measure this paper validates, and the only design that says whether address does anything to a listener rather than who it attracts."""

N[22] = """SAY
• Thank you. Questions.

LIKELY QUESTIONS
• Isn't "you" just the guest in interview shows?
  – Real limit, stated in the paper. The collection is mostly monologue and co-host shows. The question-form "you" (the part most likely aimed at a guest) is unrelated to host ideology (correlation +0.03), and removing it leaves the validation unchanged. Comparing levels across formats needs speaker separation; the archived audio allows it.
• Isn't the population finding just selection?
  – We never claim otherwise. But it survives 12 controls (moves 3%), holds within parties and within education levels, replicates with podcast use measured 2–8 months before the outcome, and has a shape (institutions down, informal up, personal flat) a general-distrust story cannot produce.
• You changed the population after the pre-set test failed.
  – Yes. The rule is symmetric, objective, and works at any threshold from 8% to 30%, and a 3%-political show is not a political podcast. But it came after → exploratory → Year 2 fixes it in advance. That is why slide 18 exists.
• Why not populist content or extremity instead?
  – We cannot rule them in or out: the 3 features are correlated 0.4–0.5. Within political podcasts, address is the one whose link to election distrust and cynicism survives the other 2 in the same model, but with 56 shows that is a ranking, not a verdict. Year 2 carries all 3.
• What do the effect sizes mean in plain terms?
  – 0.22 standard deviations of cynicism ≈ the gap between Democrats and Republicans on the same scale in this period. 0.27 standard deviations of education per unit of address ≈ a third of the way from "some college" to "bachelor's degree".
• Why language-model labels rather than human coders?
  – Show-level labels are checked 4 ways, including against the party of real listeners, which the model never saw. Passage-level human coding (200 passages, 2 coders) is scheduled before submission.
• Is 204 shows enough?
  – For the left–right style gap, yes: 0.78 standard deviations, shuffle-test p below 1 in 100,000. For the audience link the limit is listeners per show, which is why we report the honest cluster count and why Year 2 matters.
• What is the bootstrap, simply?
  – Pretend address has no effect, create 2,000 fake datasets by flipping each show's errors as a block with a coin toss, refit each time, and count how often a fake result beats the real one. It respects that listeners of the same show move together."""

N[23] = """BACKUP — only if asked
• Starting list: Apple US Politics chart, ranks 1–250, frozen 13 July 2026 (250 is as deep as Apple's chart goes).
• Filter: 2 with no feed address, 2 unreachable, 1 unreadable, 14 not US politics, 12 under 10 hours of audio → 219; 15 not in the topic-model sample → 204.
• Which episodes: every episode before 2018; then a fixed number per show per quarter, chosen by a fixed random rule.
• Transcription: Whisper large-v3-turbo, batches of 64 audio segments (batches of 128 gave identical text 1.24× faster; a lower-precision setting changed 0.76% of words and was rejected), English only, silence trimmed automatically, no speaker separation; 2 university GPU machines.
• Style measures: every 750-character chunk, split at sentence ends, nothing dropped; ads flagged. Topic model: advertising and boilerplate sentences removed first (5.9% of words; the detector had to reach 90% precision on ads and under 2% false alarms on real content before it was allowed to run; it hit 0.7%), then passages of about 500 content words (37,942 of them).
• Ideology: language-model classifier in 2 stages (is this political? if so, which side?); hosts matched to the donation database in 3 stages, with every manual decision logged."""

N[24] = """BACKUP — only if asked
• On the 56-show frame, 57% of shows have fewer than 5 listeners and the 4 largest (MeidasTouch 134, Shapiro 94, NBC Nightly News 72, The Daily 71) hold 47% of all listeners.
• Standard clustered errors are too optimistic here: 1 estimate went from p = 0.0009 the standard way to p = 0.17 under the bootstrap.
• So every show-linked estimate uses:
  1. Small-sample-corrected clustered standard errors (the Bell–McCaffrey correction), with the degrees of freedom computed honestly, about 9, meaning the estimate is really based on about 9 independent groups.
  2. The coin-flip bootstrap, step by step:
     a. Fit the model without address; keep each listener's leftover error.
     b. In each of 2,000 rounds, toss 1 coin per SHOW; on tails, flip the sign of every error in that show.
     c. Add the flipped errors back to the no-address predictions → a fake outcome in which address truly does nothing.
     d. Refit the full model on the fake outcome; record how big the address effect looks.
     e. p = share of the 2,000 fake effects at least as extreme as the real one.
  3. A show-level shuffle test as a third check; plus dropping each show in turn, and varying the political-content threshold.
• Scales: the cynicism score is standardized on the whole survey, never on the listener subset; address is standardized on the 194 shows in the collection, and the 6 outside shows are scored against that same yardstick."""

if __name__ == "__main__":
    path = str(__import__("pathlib").Path(__file__).resolve().with_name("Talking_at_the_Audience_slides.pptx"))  # next to this script
    prs = Presentation(path)
    for i, slide in enumerate(prs.slides, 1):
        slide.notes_slide.notes_text_frame.text = N[i]
    prs.save(path); print("notes written to", len(prs.slides), "slides")
