# Incivility scoring prompts

Follows the gated two-stage structure of Much et al. (2026), Appendix I: a binary
detection gate, then a second call that scores only the chunks that passed. Each
call returns exactly ONE key, and each dimension is a SEPARATE call.

Why one key and one dimension per call: Much et al. score ideology and gender
independently "so that the ideology label is not influenced by the gender label
and vice versa." That matters more here than it did there — the correlation
BETWEEN incivility and ideology is a finding we intend to report, so asking a
model for both in one response would let it reconcile them and manufacture the
result at scoring time, undetectably.

Construct: incivility as operationalised in political communication research
(Coe, Kenski & Rains 2014; Mutz 2015; Brooks & Geer 2007) — a violation of
interpersonal norms of political discussion. NOT "toxicity" as content-moderation
systems define it: those flag identity-term mentions and non-standard dialect
(Sap et al. 2019; Davidson et al. 2019), which in this corpus correlates with
topic and would confound the measure with what a show talks about.

Unlike ideology and gender, incivility is UNIPOLAR — absent to extreme, not a
direction between two poles. So stage 2 scores intensity, and a separate gated
call scores type. Intensity permits no "none" value: absence is already handled
by the gate.

---

## Shared system prompt

    You are a research assistant labeling a podcast transcript segment.
    Follow instructions carefully. Return only valid JSON. No extra text.

---

## Task 1: Incivility Detection (Stage 1)

    Podcast Labelling Instructions

    Task: Incivility Detector (stage 1)

    Goal: Detect whether the TARGET segment contains uncivil political speech.
    This is stage 1 only: do NOT rate intensity or type here.
    Intensity and type are handled in later stages.

    Incivility is speech that violates norms of interpersonal political
    discussion. It is about HOW something is said, not WHICH side is taken.
    Criticism can be forceful, detailed, and harsh while remaining civil.

    Count as uncivil:
    - Name-calling: mean-spirited or disparaging words aimed at a person or group
      ("clown," "traitor," "scum")
    - Aspersion: mean-spirited or disparaging words aimed at an idea, plan,
      policy, or behavior ("insane policy," "garbage bill")
    - Accusations of deliberate deception: calling someone a liar or claiming
      they are knowingly lying (as distinct from saying they are mistaken)
    - Vulgarity: profanity and crude language
    - Pejoratives about how someone communicates: mocking a person's speech,
      voice, intelligence, or manner of argument
    - Mockery, ridicule, or contemptuous impersonation of a person or group

    Do NOT count as uncivil:
    - Strong disagreement, forceful argument, or blunt criticism stated without
      disparagement
    - Reporting or quoting someone ELSE's uncivil speech, unless the speaker
      endorses or amplifies it
    - Vulgarity used for emphasis with no target ("this is so damn hard")
      when nothing and no one is being disparaged
    - Discussion OF incivility as a topic

    Decision rules:
    - Choose "Yes" if any of the above is present, even if brief or mild.
    - Choose "No" when the segment is merely critical, forceful, or negative
      without disparagement.
    - Political talk is often heated but civil; do not treat heat as incivility.

    Context (optional, only for understanding references):
    {CONTEXT}

    TARGET SEGMENT:
    {TRANSCRIPT}

    Choose exactly one: Yes, No

    Return ONLY this JSON object with exactly one key:

    {
      "incivility": "Yes_or_No"
    }

    Rules:
    - Return only JSON.
    - The value must be exactly "Yes" or "No".

---

## Task 2: Incivility Intensity (Stage 2, gated)

Run only where Task 1 returned "Yes". "None" is not permitted — absence was
already decided at the gate, and allowing it here reintroduces the hedging
Much et al. found when a single scale carries both decisions.

    Podcast Labelling Instructions

    Task: Incivility Intensity (stage 2) -- NON-NEUTRAL VERSION

    This is a SECOND-STAGE prompt after a detector already found the segment to
    contain uncivil speech. Your job is to rate HOW uncivil it is.
    Do NOT use "none" -- the presence of incivility is already established.

    Rate the intensity of the uncivil speech in the TARGET segment.
    Judge the severity of the norm violation, not whether you agree with the
    speaker and not how strongly they feel.

    Strength guidance:
    - slight:   a single mild instance; passing snark, light mockery, or a
                lone mild vulgarity aimed at a target
    - moderate: repeated disparagement, or one clearly contemptuous attack;
                sustained ridicule; explicit accusation of lying
    - strong:   dominant and sustained contempt; dehumanising language;
                stacked slurs or insults; the segment's main purpose is attack

    Context:
    {CONTEXT}

    TARGET SEGMENT:
    {TRANSCRIPT}

    Choose exactly one (none is NOT allowed):
    slight, moderate, strong

    Return ONLY this JSON object with exactly one key:

    {
      "incivility_intensity": "one_choice_from_list"
    }

    Rules:
    - Return only JSON.
    - The value must be exactly one of the allowed strings.

---

## Task 3: Incivility Target (Stage 2, gated) -- OPTIONAL

Run only where Task 1 returned "Yes". This is what separates generic nastiness
from AFFECTIVE POLARIZATION: hostility aimed at the political out-group is a
different construct from hostility in general, and the distinction is the whole
question in the Iyengar/Finkel literature. Skip this task if only the aggregate
incivility rate is needed; it is a third call per flagged chunk.

    Podcast Labelling Instructions

    Task: Incivility Target (stage 2)

    This segment has already been labeled as containing uncivil speech.
    Identify WHO or WHAT the uncivil speech is aimed at.

    Judge the target of the disparagement itself, not the topic of the segment.

    Categories:
    - political_opponents: the opposing party, its politicians, its voters, or
      its ideological camp, from the speaker's evident position
    - political_allies: the speaker's own side -- own party, own camp, or
      figures within it
    - media: journalists, news outlets, other commentators or platforms
    - institutions: government agencies, courts, universities, corporations,
      international bodies
    - individual_nonpolitical: a specific person attacked in a non-political
      capacity
    - diffuse: no identifiable target; general venting or ambient vulgarity

    Decision rules:
    - Choose the PRIMARY target if several are attacked.
    - "political_opponents" vs "political_allies" is judged relative to the
      speaker's own apparent position in the segment, not to any fixed party.
    - If the speaker's own side cannot be determined, and the target is a
      political actor, use "political_opponents" only when the disparagement
      clearly runs across a partisan line; otherwise use "diffuse".

    Context:
    {CONTEXT}

    TARGET SEGMENT:
    {TRANSCRIPT}

    Choose exactly one:
    political_opponents, political_allies, media, institutions,
    individual_nonpolitical, diffuse

    Return ONLY this JSON object with exactly one key:

    {
      "incivility_target": "one_choice_from_list"
    }

    Rules:
    - Return only JSON.
    - The value must be exactly one of the allowed strings.

---

## Aggregation

Mirrors Much et al. exactly so the three dimensions stay comparable:

  episode incivility rate = word-count-weighted share of chunks flagged uncivil
  episode incivility intensity = word-count-weighted mean of intensity
      (slight=1, moderate=2, strong=3) over FLAGGED chunks only; undefined
      when an episode has no uncivil chunks
  episode outgroup animosity = word-count-weighted share of chunks that are
      uncivil AND targeted at political_opponents

  show-quarter = word-count-weighted mean over episodes, weighting each episode
      by its total word count (rate) or by its uncivil-chunk word count
      (intensity), following her treatment of gender vs ideology weights.

Report the RATE as primary. It is the analogue of her directional share, it is
defined for every episode, and it does not inherit stage-2 measurement error.
