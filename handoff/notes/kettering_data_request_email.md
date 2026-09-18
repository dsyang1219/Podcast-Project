# Kettering-Gallup Y2 data access request

## Recipients

**To:** Chris Miljanich, Justin Lall (Gallup) - authors of "What Podcaster Audiences Want From
U.S. Democracy" (June 15, 2026)
**Cc:** Ellyn Maese, PhD - Senior Research Consultant, Gallup; lead author on the Kettering-Gallup
media/democracy analyses
**Cc:** Brad Rourke - Chief External Affairs Officer, Kettering Foundation
**Cc:** Prof. R. Michael Alvarez (faculty sponsor) - ASK HIM FIRST before adding him

## Addresses — what is VERIFIED vs INFERRED

VERIFIED (from the organizations' own pages):
- communications@gallup.com  / +1-202-715-3030
    Gallup's stated channel for questions about "a survey, survey findings, or historical data."
    Closest thing to an official research/data inbox. Source: news.gallup.com/poll/128246
- mediainquiry@gallup.com
    Gallup Media Inquiries page. Press-oriented; use only as backup.
- Kettering Foundation: contact form at https://kettering.org/contact-us/
    Phone listings conflict across sources (937-434-7300 vs 937-299-3600) - confirm before calling.
    Address: 200 Commons Rd, Dayton, OH 45459.

NOT VERIFIED - individual addresses could not be confirmed. Every directory that lists them masks
the local part: Miljanich c***@gallup.com, Lall j***@gallup.com, Maese e***@gallup.com. All three
masks are consistent with first_last@gallup.com (reported as ~58% of Gallup addresses), but that
is an inference, not a confirmation. Do not treat a guessed address as reliable delivery.

## Recommended send order

1. CHECK YOUR OWN INBOX FIRST. The Y1 microdata came through a registration form on
   gallup.com/analytics/697985. The confirmation/download email is very likely from a monitored
   address, and it is the only address in this list confirmed to already handle Democracy for All
   data requests. Reply to that thread if it exists.
2. ASK ALVAREZ. He may know people in Gallup's research organization directly. Separately,
   Chris Miljanich is a recent UC Santa Barbara political science PhD - Alvarez plausibly knows
   his committee. A warm intro beats every address below.
3. communications@gallup.com, with Miljanich, Lall and Maese named in the first line so it routes.
4. Guessed individual addresses (first_last@gallup.com), accepting bounce risk, ideally alongside 3.
5. LinkedIn message to Ellyn Maese or Justin Lall - identity is verified there even though it is a
   worse medium.

TO CONFIRM BEFORE SENDING:
- Ask Alvarez before cc'ing him, and ask whether he'd rather make an introduction directly -
  a warm intro from him almost certainly beats this email.
- Citadel fellowship deliberately omitted from signature - see notes.
- Alvarez's address: Caltech directory UID is "rma" (directory.caltech.edu/personnel/rma), so
  likely rma@caltech.edu or rma@hss.caltech.edu - just confirm with him.

---

**Subject:** Democracy for All Y2 — Q17 verbatims and Gallup Panel linkage

Dear Dr. Miljanich and Mr. Lall,

I'm an undergraduate researcher at the Linde Center for Science, Society, and Policy at Caltech,
working with Professor R. Michael Alvarez on political podcasts as a form of political
communication.

The project starts from a corpus of roughly 200 U.S. political podcasts — about 1.4 million
transcript passages. Rather than measuring what hosts talk about, I measure how they address their
audiences: how often they issue direct instructions, speak to listeners in the second person, and
draw in-group versus out-group distinctions. Three findings motivate the request.

**The left-right difference is in manner of address, not hostility.** Validating show-level register
against host campaign contribution ideology scores (DIME, n = 112 shows), the strongest correlates
of a show's ideological position are imperatives (r = +0.39, p = 2e-5) and second-person address
(r = +0.39, p = 3e-5) — right-leaning shows talk *at* their listeners more. Insult and invective, by
contrast, are uncorrelated with ideological direction almost exactly (r = +0.0005, p = .996), even
though they track ideological *intensity* strongly (r = +0.55). The familiar incivility story does
not distinguish left from right in this corpus; the mode of address does.

**Register is a stable property of shows, not of episodes.** A classifier using register features
separates left- from right-leaning shows at AUC = 0.80 between shows but only 0.57 within them, with
non-overlapping confidence intervals. That dissociation is what makes a listener's named shows a
meaningful exposure measure rather than noise.

**Register tracks a political outcome in your Y1 data.** The Q17 verbatims let me match 51 corpus
shows to 385 respondents — an order of magnitude better named-source coverage than I found in Pew's
American Trends Panel or the ANES, both of which I tried first. (Pew withholds its open-ended
responses entirely.) Respondents naming more directive-register shows report a wider party
favorability gap on Q39A/Q39B: b = +0.66, t = 3.16, p = .002, controlling party identification,
political attention, age, and education. It is not simply conservative-media exposure — directive
exposure and right-leaning exposure correlate at only r = 0.36, and with both in the model directive
register holds (p = .018) while show ideology falls to marginal (p = .073). The association is also
specific: it is null for political violence attitudes, trust in leaders, feeling respected, and
sense of citizen power, which argues against a general response-style artifact.

The clear limitation is that Y1 is cross-sectional, so media effects cannot be separated from
self-selection. Three questions about Year 2:

1. **Release timing.** Is respondent-level Y2 microdata (fielded April 24–June 10, 2026) expected to
   be posted through the same registration form as Y1, and if so, roughly when?

2. **Q17 verbatims.** Were the open-ended "top three programs, channels, podcasts, websites,
   influencers" responses retained as raw text in Y2, as in Y1? This is what makes the study
   uniquely suited to the question — most surveys release only coded categories, which collapse
   precisely the mid-tail shows that make up the corpus.

3. **Panel linkage.** Y1 includes 9,177 Gallup Panel respondents (SAMP_TYPE 1–2) alongside the
   opt-in sample, and 263 of my 385 matched respondents are panel members. If any portion of the
   panel was re-interviewed in Y2, is there a stable respondent identifier permitting a Y1→Y2 link?
   Even a few hundred linked cases would support a cross-lagged model — prior-wave exposure
   predicting change in the favorability gap, controlling prior-wave polarization — which is a
   materially stronger design than the cross-section allows.

If a linked file requires a data use agreement or IRB review, I am glad to complete whatever process
applies; Professor Alvarez (Flintridge Foundation Professor of Political and Computational Social
Science, and co-director of the Linde Center) would serve as institutional signatory, and he is
copied here. I would also be happy to preregister the analysis, including the show-name matching
procedure, before Y2 data is in hand.

In return I can share what I've built: a hand-audited alias table mapping Q17 verbatim spellings to
canonical show identities (this proved essential — naive fuzzy matching produced badly wrong
matches, e.g. "social media" to *On the Media*, "news nation" to *Face the Nation*), and show-level
register measures for the ~200 podcasts in the corpus, which may be useful for future Democracy for
All media analyses. I would cite the study per your preferred citation and share any resulting
manuscript before submission.

Thank you for releasing the Y1 verbatims rather than suppressing them — it is unusual, and it is the
reason this analysis was possible at all.

Best regards,

Danielle Yang
Undergraduate Researcher, Linde Center for Science, Society, and Policy
California Institute of Technology
dsyang@caltech.edu

(Faculty sponsor: Prof. R. Michael Alvarez, Flintridge Foundation Professor of Political and
Computational Social Science, Caltech)
