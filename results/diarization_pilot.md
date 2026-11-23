# Diarization pilot — reference `s1act`

Reference variant: `s1act`. DER is computed treating the reference variant's output as ground truth, so it measures how far each acceleration moves the result, not absolute accuracy. 0.25 s collar, overlapping speech included.

| variant | episodes | hours | DER vs reference | speakers (mean) | realtime factor | GPU-hours for 75,359 h |
|---|---|---|---|---|---|---|
| s1act | 405 | 381.4 | 0.000 | 8.50 | 102x | 737 |
| s3act | 405 | 381.4 | 0.024 | 6.16 | 295x | 255 |
| s3act31 | 405 | 381.4 | 0.034 | 5.31 | 308x | 245 |
| s3dom | 405 | 381.4 | 0.032 | 5.84 | 355x | 213 |
| s3full | 2 | 2.4 | 0.003 | 2.50 | 57x | 1333 |
| stock | 40 | 38.2 | 0.002 | 8.95 | 22x | 3362 |

Model: `pyannote/speaker-diarization-community-1`, clustering `VBxClustering`, device cuda.

- `s1act`: step 1.0 s, batch 32, embed active, mcs 0.01
- `s3act`: step 3.0 s, batch 32, embed active, mcs 0.01
- `s3act31`: step 3.0 s, batch 32, embed active, mcs 0.01
- `s3dom`: step 3.0 s, batch 32, embed dominant, mcs 0.01
- `s3full`: step 3.0 s, batch 32, embed full, mcs 0.01
- `stock`: step 1.0 s, batch 32, embed full, mcs stock

# Diarization pilot — reference `stock`

Reference variant: `stock`. DER is computed treating the reference variant's output as ground truth, so it measures how far each acceleration moves the result, not absolute accuracy. 0.25 s collar, overlapping speech included.

| variant | episodes | hours | DER vs reference | speakers (mean) | realtime factor | GPU-hours for 75,359 h |
|---|---|---|---|---|---|---|
| s1act | 39 | 37.4 | 0.002 | 9.13 | 104x | 722 |
| s3act | 39 | 37.4 | 0.017 | 6.38 | 297x | 253 |
| s3act31 | 39 | 37.4 | 0.024 | 5.72 | 309x | 244 |
| s3dom | 39 | 37.4 | 0.024 | 6.15 | 370x | 203 |
| s3full | 1 | 1.6 | 0.002 | 3.00 | 56x | 1343 |
| stock | 39 | 37.4 | 0.000 | 9.10 | 22x | 3357 |

Model: `pyannote/speaker-diarization-community-1`, clustering `VBxClustering`, device cuda.

- `s1act`: step 1.0 s, batch 32, embed active, mcs 0.01
- `s3act`: step 3.0 s, batch 32, embed active, mcs 0.01
- `s3act31`: step 3.0 s, batch 32, embed active, mcs 0.01
- `s3dom`: step 3.0 s, batch 32, embed dominant, mcs 0.01
- `s3full`: step 3.0 s, batch 32, embed full, mcs 0.01
- `stock`: step 1.0 s, batch 32, embed full, mcs stock

## Reading the pilot (8 Nov 2026)

404 episodes, two per show from 202 shows, 381 hours. `s1act` (stride 1 with the
two embedding changes) reproduces stock pyannote to within fp16 noise (DER 0.002
on the 39 stock episodes), so it stands in for stock on the full sample.

- **Accuracy cost of stride 3:** DER 0.017 against stock on 39 episodes, 0.024
  against the stride-1 reference on all 405. Almost all of it is speaker
  confusion (0.014 of the 0.017); missed speech and false alarms are unchanged,
  so segmentation is intact and the difference is in clustering.
- **Where the confusion is:** stock finds 8.9 speakers per episode, stride 3
  finds 6.3, but both find the same number of speakers with at least 5% of the
  speech (3.1 vs 3.2), and the top speaker's share differs by a median of
  0.001. The merged clusters are voices under 2% of an episode (clips, ad
  reads, stingers), which hold 2.8% of speech under stock and 1.9% under
  stride 3. For host identification and host-only text this is immaterial.
- **Variants:** the 3.1 pipeline with the relative minimum cluster size
  (`s3act31`, DER 0.024) and the dominant-slot mode (`s3dom`, 0.024) are both
  worse than the default; the dominant mode's extra 20% speed is not worth it.
- **Speed:** default variant 295× realtime over 381 hours, 255 GPU-hours for
  the 75,359-hour corpus, against 3,360 for stock.
- **Thermal:** the adaptive floor gate is inert for a GPU-bound run (the floor
  drifts up with the work); episodes started at 138–156 °F, median 147 °F, for
  13 hours. The corpus run uses a fixed cap of 145 °F (63 °C) instead
  (`--gate-below-f`).
- **Host identification** on two episodes per show is not informative:
  recurrence needs several episodes, so every show fell back to the dominant
  voice and the name check agreed on 34 of 53 self-introductions. It is
  re-run on the full corpus; the 20 hand-labelled episodes score that run.

**Decision:** corpus run with `community-1`, stride 3 s, active-slot
embeddings, fp16 ResNet, 145 °F start cap (`step4_diarize/run_corpus.sh`,
launched 8 Nov 2026 20:05 UTC on the 50,395 episodes archived so far; it
repeats until the archive re-download finishes).

## Host identification vs hand labels (default)

8 episodes, duration-weighted: precision 0.929, recall 0.878.

## Host identification vs hand labels (22 Nov 2026, all 20 episodes)

Twenty episodes from twenty shows labelled by hand at the Whisper-segment level
(H host, G other voice, A ad/clip, X unsure; 859 rows, 305 minutes of H/G
speech). Scored against the corpus-run host labels.

**By speech duration (segment level)**

| scoring set | precision | recall | F1 |
|---|---|---|---|
| all labelled segments | 0.915 | 0.857 | 0.885 |
| segments with purity ≥ 0.8 (the host-only-text filter) | 0.979 | 0.921 | 0.949 |

**By speaker (is each voice the labeller called host flagged as host?)**
precision 1.00 (20 of 20 flagged hosts are hosts), recall 0.69 (9 of 29 voices
the labeller called host are not flagged). None of the nine is a split host
voice (cosine to the host voice ≤ 0.47, most under 0.25), so lowering the
similarity threshold does not recover them and would add 2,700–30,000 false
host slots corpus-wide; the threshold stays at 0.6. The nine are:

- rotating or fill-in presenters (NPR Politics ×2, The 11th Hour): recur in
  under 50% of the show's episodes, so they are "regular" voices, not hosts,
  by construction;
- recurring low-share voices (Josh Hammer SPEAKER_03, recurrence 0.88; Rest Is
  Politics SPEAKER_05, 0.56) under the 10%-of-show-speech rule: announcers or
  quiet co-presenters;
- slivers of 30–60 s in one episode each (Kaitlan Collins ×2, Mark Kaye, Rest
  Is Politics), different voices the labeller marked H.

The gap between speaker-level precision (1.00) and duration-level precision
(0.915) is Whisper-segment granularity: a 5–30 s segment often holds a host
question and a guest answer, and the majority-overlap rule assigns it to one
speaker. The purity filter removes most of that, and word-level alignment
(WhisperX) would remove the rest if turn-level text is wanted later.
