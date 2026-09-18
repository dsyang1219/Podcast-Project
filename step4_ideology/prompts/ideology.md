# Ideology scoring prompts — Much et al. (2026), Appendix I, Tasks 5 and 6

Transcribed verbatim so the measure stays comparable to their published
show-level scores. Do not paraphrase: the wording IS the instrument, and their
validation (kappa = 0.834 on directional ideology against a human ceiling of
alpha = 0.556) attaches to these exact strings.

Two deviations from their setup, both deliberate and both recorded here:

  MODEL     They used Gemini 3 Flash Preview via Vertex batch. We use an OpenAI
            model. Their validation does NOT transfer across models — but they
            measured cross-model stability themselves (Gemini 3 Flash vs Gemini
            2.5 Flash Lite, r = 0.833 at podcast level), so drift is expected to
            be modest rather than absent. Treat agreement with their published
            show scores on overlapping shows as a cross-model, cross-lab
            replication, which is a stronger claim than inheriting their numbers.

  CHUNKING  Their unit is a diarized SPEAKER TURN capped at 750 chars. This
            corpus has no diarization, so we cut fixed 750-char windows at
            sentence boundaries. Untested assumption; the 3,581 episodes with
            archived audio can be diarized and scored both ways to check it.

Stage 2 runs ONLY on chunks where stage 1 returned "Yes". Chunks that fail the
gate score 0 and are never sent for direction. That gate is also what makes ad
reads harmless: a mattress commercial is not political, fails stage 1, and never
reaches the ideology scorer.

---

## Shared system prompt

    You are a research assistant labeling a podcast transcript segment.
    Follow instructions carefully. Return only valid JSON. No extra text.

---

## Task 5: Politics Detection (stage 1)

    Podcast Labelling Instructions

    Task: Politics Detector (stage 1)

    Decide whether the TARGET segment is about politics, political issues, or
    political figures. This Yes/No decision is the detector gate for ideology
    stage 2:
    - "Yes" segments proceed to ideology direction/intensity scoring.
    - "No" segments are not sent to ideology stage 2.

    Segments about politics can include discussion about how we govern ourselves,
    distribute power, or resolve collective questions. This includes both formal
    political institutions and social controversies that have become matters of
    public debate.

    A segment is political if it addresses things like:
    - Government, legislation, public institutions, elections, or political figures.
    - Public policies (e.g., taxes, healthcare, civil rights).
    - Organized activities by the public to influence politics (e.g., protests,
      boycotts, writing letters to representatives, campaign donations).
    - Geopolitical or international conflicts.
    - Politically salient social and cultural issues debated in the public sphere,
      such as racism, sexism, LGBTQ rights, immigration, public health mandates,
      religion in public life, regulating education curricula, or free speech.

    Context:
    {CONTEXT}

    TARGET SEGMENT:
    {TRANSCRIPT}

    Is this about politics? Choose exactly one: Yes, No

    Return ONLY this JSON object with exactly this key:

    {
      "politics": "Yes_or_No"
    }

    Rules:
    Return only JSON.
    politics must be exactly "Yes" or "No".

---

## Task 6: Ideology Direction (stage 2, politics-gated)

Unlike the gender tasks, "moderate" IS permitted here — it absorbs straight news
delivery and segments that mix liberal and conservative cues without a net stance.

    Podcast Labelling Instructions

    Task: Ideology (stage 2, politics-gated)

    This segment has already been labeled political by the Politics Yes/No detector.
    This step is only run for segments where politics == "Yes".

    Rate the ideological leaning expressed in the TARGET segment.
    Use the speaker's stance, not the topic itself.
    If the segment is purely descriptive, balanced, mixed between liberal and
    conservative cues, or reporting without endorsement (e.g., a straight news
    segment about politics), choose "moderate".

    Lean cues (U.S. context):
    - Liberal: support for redistribution, social safety nets, civil rights
      expansion, regulation, climate action, gun control, reproductive rights.
    - Conservative: support for limited government, lower taxes, traditional
      social norms, deregulation, strong law-and-order, gun rights.

    Intensity:
    - slightly: mild or single cue
    - moderately: multiple cues
    - strongly: dominant, emphatic stance
    - moderate: neither clearly liberal nor clearly conservative, including
      neutral political reporting or mixed cues without a directional stance

    Context (optional, only for disambiguation):
    {CONTEXT}

    TARGET SEGMENT:
    {TRANSCRIPT}

    Choose exactly one:
    strongly_conservative, moderately_conservative, slightly_conservative,
    moderate,
    slightly_liberal, moderately_liberal, strongly_liberal

    Return ONLY this JSON object with exactly one key:

    {
      "ideology": "one_choice_from_list"
    }

    Rules:
    Return only JSON.
    The value must be exactly one of the allowed strings.

---

## Aggregation (Much et al. §4.2)

Retain the full 7-point output. Their main analysis collapses to ternary
(-1 liberal / 0 moderate / +1 conservative) because human-LLM agreement is
substantially higher on direction than on intensity, and disagreements are almost
entirely ordinal-neighbour rather than sign flips. Keep BOTH:

    net_ideology(episode) = word-count-weighted share of CONSERVATIVE chunks
                          - word-count-weighted share of LIBERAL chunks
                            (among politically-flagged chunks only)
                            bounded [-1, 1]; undefined if no political content

    podcast level = weighted mean of episode indices, each episode weighted by
                    its total POLITICAL-chunk word count

The ternary is their published measure and keeps this corpus comparable to it.
The 7-point matters here for a reason it did not there: this design tracks
WITHIN-SHOW change over time, and shows rarely flip sign — they shift intensity.
A ternary collapse would be blind to exactly the variation being tested.
