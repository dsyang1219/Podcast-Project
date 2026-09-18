# Pre-registration: OUT-OF-FRAME GENERALIZATION TEST of directive register -> institutional cynicism
(previously drafted as a 'holdout'; renamed — see 'What this test is')
STATUS: FROZEN at 2026-09-02T21:08:19Z (UTC). No audio had been fetched at this timestamp. File this text on OSF
(osf.io/prereg) as-is; nothing below changes after this line.

## Hypothesis
H1. Among respondents who name a large political podcast outside the corpus, exposure to more
    directive register (dir_z of the named show) predicts higher institutional cynicism,
    controlling party identification, political attention, age, education, show ideology, and
    right-wing platform use. Predicted direction: positive.
H2 (secondary). The same exposure predicts a higher probability of naming no institutional news
    source (audience exclusivity). Predicted direction: positive.

## What this test is — and is not
The corpus sampling frame is the Apple Podcasts US POLITICS subcategory chart (pipeline/chart.py).
None of the ten shows below has ever appeared on that chart (chart_panel.csv, 15 dates, 302 shows);
they are filed under News, Comedy, or Society & Culture. They are therefore OUTSIDE the corpus's
sampling frame. Their Kettering listeners are independent respondents whose outcomes have never
been examined against register — but they are not a random holdout from the discovery population.

This is an OUT-OF-FRAME GENERALIZATION TEST. The asymmetry is stated now:
  - If H1 holds: the association extends beyond the Politics subcategory into the largest shows
    in the ecosystem. That is STRONGER evidence than a within-frame holdout would give, and it
    closes the coverage limitation (§11) in the same move.
  - If H1 fails: it is ambiguous between "no association" and "the largest, multi-category shows
    are a different population." The paper will report the failure as a scope boundary on §F.3,
    not as a refutation of it, and will say that a within-frame confirmation requires Kettering Y2.
No within-frame holdout exists in Y1: every corpus-matched respondent was used in discovery, and
the ~145 corpus shows with no Kettering listeners cannot supply one.

## Shows (fixed now; cutoff = >= 15 Kettering respondents naming the show; must have a podcast RSS)
    The Joe Rogan Experience         feeds.megaphone.fm/GLT1412515089      ~368 respondents
    The Ben Shapiro Show / Daily Wire feeds.megaphone.fm/BVDWV5370667266    ~114
    The Glenn Beck Program           feeds.megaphone.fm/BMDC3567910388       ~69
    The Tucker Carlson Show          feeds.megaphone.fm/RSV1597324942        ~64
    The Daily (New York Times)       feeds.simplecast.com/54nAGcIl           ~73
    The Parnas Perspective           feeds.megaphone.fm/MTH4368798504        ~51
    The Charlie Kirk Show            omnycontent.com/d/playlist/5e27a451...  ~48
    The Megyn Kelly Show             feeds.simplecast.com/RV1USAfC           ~26
    The Shawn Ryan Show              feeds.megaphone.fm/WWO7410387571        ~21
    The Dan Bongino Show             feeds.megaphone.fm/WWO3519750118        ~17
Excluded: Hasan Piker (no podcast RSS; Twitch-native). Timcast IRL (charted on the Politics
subcategory 15 times, best rank 42, and is in the corpus frame — not out-of-frame; its 33
respondents are discovery-territory and are excluded). Any show whose feed fails to yield >= 15
episodes is dropped and reported as dropped.

## Episode sample (fixed now)
The 50 most recent episodes in each feed as of 2026-09-02, excluding episodes under 15
minutes; 25 is the minimum for a show to be retained. Rationale: single-episode register
consistency within show is 0.176 (§11), so by Spearman-Brown a 10-episode mean has reliability
0.68, 25 episodes 0.84, 50 episodes 0.91. Transcription runs at ~110x real time, so 50 costs
nothing that matters; the binding constraint is download bandwidth, not compute. Same 750-character passage windows, same register pipeline (remeasure2.py), same
show-level aggregation and z-scoring against the ORIGINAL corpus distribution (so dir_z is on the
same scale as the discovery sample).

## Respondent match (FROZEN — handoff/holdout_alias_table_FROZEN.csv, holdout_alias_patterns_FROZEN.json)
749 fresh respondents, 810 respondent-show pairs (57 name two holdout shows), 10 shows:
  Rogan 370 | Shapiro/Daily Wire 112 | The Daily 73 | Tucker 64 | Beck/Blaze 59 | Kirk 39 |
  Parnas 37 | Kelly 23 | Shawn Ryan 21 | Bongino 12
Documented mapping decisions, made before any outcome was examined:
  - "the daily" (73) is mapped to the NYT show; in a top-3-news-sources context the NYT Daily is
    the dominant referent, but the string is generic and this is the least certain mapping.
  - "blaze" / "blaze tv" / "blaze media" are mapped to Glenn Beck; BlazeTV is his network and
    carries other hosts. Robustness: re-run with Blaze-only strings dropped (reported).
  - "daily wire" / "dailywire" are mapped to Shapiro; the Daily Wire carries other hosts.
    Robustness: re-run with Daily-Wire-only strings dropped (reported).
Respondents in the 385 strict or 544 expanded discovery sets are excluded (done).
## Respondent match (fixed now)
Host-name and show-title matching on Q17_1-3 verbatims, using an alias table written BEFORE any
outcome is computed and frozen at filing. For personality-branded shows the host name IS the
title; the title/host ambiguity that invalidated the corpus expansion does not arise. A respondent
naming two holdout shows gets the mean dir_z. Respondents already in the 385 discovery sample or
the 544 expanded sample are EXCLUDED.

## Outcome (fixed now)
PRIMARY   institutional cynicism, 8-item: reverse-scored Q35E, Q35C, Q34B, Q33H, Q31, Q33E, Q35A,
          Q35D, each z-scored on the full Kettering sample, averaged, re-standardised.
SECONDARY Q33E alone (elections administered well, reversed).
H2        exclusivity = 1 if no Q17 verbatim matches the mainstream-outlet list used in round 8.

## Model and inference (fixed now)
OLS, respondent level. Controls: party ID (5 categories), Q19 attention, age, education,
show ideology (LLM side label; DIME where available), right-wing platform index (Q20C/E/G/H).
SEs clustered by show. Inference: CR2 (Bell-McCaffrey) with Satterthwaite df as primary;
wild cluster bootstrap (Rademacher, 2000 reps) alongside. alpha = .05, one-sided in the predicted
direction; two-sided also reported. Weighted (WEIGHT) as a reported robustness, unweighted primary.

## Success / failure criteria (fixed now)
H1 supported: dir_z coefficient positive with one-sided CR2 p < .05 AND wild p < .05.
H1 not supported: either p >= .05. No re-specification, no alternative composite, no alternative
match rule after filing. If H1 fails, the paper reports "the association does not extend to
the largest shows in the ecosystem" as a result, not as a caveat.
Expected effect if H1 holds, from discovery: ~+0.2 SD per SD of dir_z.

## Leverage, pre-specified
Rogan alone is ~368 of ~840 respondents — one cluster holding ~44% of the sample — and is expected
to sit far below the corpus's directive range. With 10 clusters and that imbalance, Satterthwaite
df will be roughly 3-4. The test is, in effect, "do Rogan's listeners differ from Kirk/Beck/
Carlson/Shapiro's." Stated in advance so it cannot be discovered afterward. Required, pre-specified:
  (i)  PRIMARY: respondent-level model as specified, CR2 + wild p, Satterthwaite df reported.
  (ii) REQUIRED ROBUSTNESS: the same model with Rogan excluded. If (i) succeeds and (ii) fails,
       H1 is reported as "driven by the Rogan contrast," not as supported.
  (iii) SENSITIVITY: per-show weights capped so no show exceeds 20% of the effective sample.
  (iv) Show-level precision-weighted permutation test on the 10 show means (very low power;
       reported, not decisive).
Success requires (i) AND (ii).

## What will be reported regardless
The coefficient, CIs, both p-values, the Satterthwaite df, a leave-one-show-out table, the
per-show dir_z scores, and the per-show audience partisan lean (as a further check on the label).

## Scoring procedure — pinned after the freeze (computational specification only; no change to hypothesis, outcome, match, or inference)
The prereg says "same register pipeline, same aggregation, z-scored against the original corpus distribution."
That is now concretely: (1) chunk holdout transcripts with build_scoring_chunks.py (750-char cap, identical to
the corpus); (2) extract per-passage features with /tmp/remeasure2.py (spaCy syntactic imperatives, second-
person pronouns, regex imperatives); (3) show-level value = nw-weighted mean of the per-10k-word passage rate
for imper_syn, imper_rx, you_rx; (4) z-score each against the 194-show corpus distribution of the same
quantity (handoff/dirz_reference_scale.csv, population SD); (5) dir_z = mean of the three z's.
Verification, done before any holdout audio existed: this rate-based dir_z correlates r = 0.991 with the
reference dir_z used in every discovery analysis, and reproduces its validation (vs DIME r = +0.45 vs +0.43
reference; vs LLM side +0.36 vs +0.35). The reference file's show-level columns were the same rates on a
constant scale factor (corr 0.979 with H4 above).
Pipeline launched 2026-09-02 ~21:20 UTC, chained after the in-flight recovery run: manifest
data/output/sample_out/sample_manifest_holdout.csv, 10 shows x 50 episodes, 732 audio-hours.

## AMENDMENT 1 — 2026-09-03T00:24:21Z — filed BEFORE any holdout respondent outcome has been examined
Trigger: the pre-specified face-validity check (show-level dir_z ranking on the first 19 transcribed
episodes, no survey data touched) puts The Joe Rogan Experience at +0.86 on the corpus scale — above
MeidasTouch (+0.94 is the corpus's 2nd-highest) — with you_rx 287/10k and imper_syn 71/10k over 1,434
passages. Tucker Carlson (2 eps) shows you_rx 351 with imper_syn 33. This is the dyadic-format
confound: in a long-form host+guest interview the second person is addressed to the GUEST, and the
instrument cannot distinguish that from address to the listener. The corpus is monologue/co-host
dominated (its "multi-host" shows are co-host panels, not interviews), so this did not arise there;
it is an out-of-frame measurement issue, and Rogan is ~49% of the holdout sample.

The PRIMARY test is unchanged and will be run and reported exactly as frozen.

Added, pre-specified now, as a labeled SENSITIVITY (not a replacement):
  (v) FORMAT-ADJUSTED model: primary model + a show-level indicator INTERVIEW_DOMINANT, coded now from
      public knowledge of each show's format (not from the data):
        interview-dominant = 1: Joe Rogan Experience, Tucker Carlson Show, Shawn Ryan Show
        interview-dominant = 0: Ben Shapiro Show, Glenn Beck Program, Charlie Kirk Show, Dan Bongino
                                Show, Megyn Kelly Show (monologue + interview mix; coded 0 because the
                                monologue open dominates), Parnas Perspective (solo to camera),
                                The Daily (scripted narrative with reporter dialogue; coded 0)
      Reported alongside (i)-(iv). If (i) fails and (v) passes, H1 is reported as "supported once
      interview format is held constant," clearly labeled as a post-freeze sensitivity.
  (vi) The same INTERVIEW_DOMINANT indicator will be applied retrospectively to the DISCOVERY sample
       (corpus shows) to check whether it changes the discovery coefficient; the corpus's existing
       solo/multi control already showed the register differences survive format (§C.3).
Prediction registered now: Rogan will remain the highest-dir_z show; (i) is more likely to fail than
pass (~25%); if (v) diverges from (i) it will be because Rogan's high measured address does not carry
a correspondingly cynical audience.

## AMENDMENT 2 — 2026-09-03T00:29:16Z — filed BEFORE any holdout respondent outcome has been examined
Tested whether text-only cues separate guest-directed from listener-directed second person. Sentence type
does NOT: Rogan's "you" is 69% declarative, 13% interrogative (corpus median 11%); on declarative and
deontic "you" alike Rogan sits with the monologue shows. Position in the episode DOES carry a format
fingerprint: monologue shows concentrate address in the opening (Shapiro open 565 you/10k, body 170;
imperatives 350 vs 90), interview shows are flat (Rogan 247/287/293; Tucker 320/337/360).
Added, pre-specified now, as labeled SENSITIVITIES (primary unchanged):
  (vii) DATA-DERIVED FORMAT SCORE: per show, ratio of you_rx in the opening 10% of each episode to
        you_rx in the body (30-100%), nw-weighted, averaged over episodes; log-transformed. Computed
        from register only, no outcomes. Enters the primary model as a covariate in place of the
        hand-coded INTERVIEW_DOMINANT of Amendment 1 (both reported).
  (viii) DECLARATIVE-ADDRESS variant of the exposure: dir_z rebuilt with you_rx restricted to second
        person in non-interrogative sentences not containing a PERSON entity and not "you know".
        On a 50-show corpus sample this variant validates slightly better against DIME (r=.50 vs .46)
        and tracks the reference dir_z at r=.79. Reported as an alternative exposure, not primary.
Corpus-side check recorded for the paper: interrogative-you rate correlates with the address LEVEL
(r=+.67 with dir_z) but not with ideology (r=+.03 vs DIME, +.11 vs LLM side); partialling it out of
dir_z leaves the DIME validation intact (r=.49 vs .47). The partisan register contrast is not a
dialogue artifact; the level measure partly is.
Diarization (host-turn-only address) remains the clean fix; rma2 archives audio for Rogan, Tucker,
Beck, Bongino and could be diarized later as a further sensitivity. Not part of tonight's test.
