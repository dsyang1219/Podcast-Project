# Speaker diarization of the corpus: approach, sources, output

*Record as of 7 November 2026. The pilot (404 episodes) is running; its numbers go into `results/diarization_pilot.md`.*

## What we are adding and why

The corpus is 84,879 transcribed episodes from 194 political podcasts, 75,359 hours of audio. The transcripts have timestamps but no speakers. We are adding a speaker layer to every episode: who spoke when, which Whisper segment belongs to which voice, and which voice is the host. That lets us rebuild the listener-directed-address measure on host speech alone, measure host speaking share and turn-taking, and release the corpus with a speaker layer that others can use. No podcast corpus of this size has one: SPoRC, the largest to date, diarized a random third of its episodes because stock diarization was too slow.

## The pipeline

1. **Audio.** Every episode's audio is re-fetched from its feed and stored as 16 kHz mono Opus at 24 kb/s (`data/audio_archive/<show_id>/<episode_id>.opus`, about 11 MB per hour, 0.8 TB for the corpus). Speaker-embedding models are trained on 16 kHz speech and are insensitive to this codec. About 1% of episodes no longer have retrievable audio and stay text-only.

2. **Diarization: pyannote, with three speed changes.** We use `pyannote/speaker-diarization-community-1` (pyannote.audio 4.0), which chains a neural segmentation model (`pyannote/segmentation-3.0`, powerset multi-class, 10 s windows) that finds up to three simultaneous speakers per window, a speaker-embedding model (WeSpeaker ResNet34-LM, 256-d) applied per window and speaker, and VBx clustering with a PLDA back end that groups the embeddings into speakers. We do not change any model or any clustering parameter. We change how often the embedding model is called:
   - **Segmentation step 1 s → 3 s.** Each second of audio is then covered by about three windows instead of ten.
   - **Embed only the speaker slots that contain speech.** Stock pyannote embeds every slot of every window, including silent ones that it later discards. On a two-speaker episode this is 2,955 passes for 1,129 that matter. The output is unchanged.
   - **Half precision for the ResNet.** The log-mel filterbank stays in fp32 (it overflows in fp16); only the ResNet runs under fp16 autocast.

   Measured on this machine on a 49-minute episode: stock 118 s (25× realtime, 96% of it embeddings); with all three changes 10 s (300×). The stride-1 version with the two embedding changes alone reproduces stock output exactly (DER 0.000) at 110×, so those two are pure speed. Stride 3 moved DER by 0.003 against stock on the two test episodes. For the corpus this is about 250 GPU-hours instead of about 3,500.

3. **Attaching speakers to the transcript.** Each Whisper segment gets the speaker whose turns cover most of it (using pyannote's exclusive, non-overlapping diarization) and a *purity*, the share of the segment that speaker covers. Segments that straddle a speaker change have low purity and can be dropped or split downstream; host-only text uses purity ≥ 0.8.

4. **Host identification, two routes.**
   - *Voice recurrence (primary).* Within a show, the per-episode speaker centroids are clustered across episodes (cosine distance, average linkage). A guest's voice appears in one episode; a host's appears in nearly all. A voice is a host if it recurs in at least half of the show's diarized episodes and carries at least 10% of its speech, up to three hosts for co-hosted shows. Shows with fewer than four diarized episodes are flagged low-confidence.
   - *Self-introduction by name (check).* Host names come from the show's feed author field and the host lookup built for the DIME match (122 shows have a person's name). In the first five minutes, a segment matching "I'm Candace", "this is Daniel", "my name is …" marks its speaker as host. Agreement between the two routes is reported per show; disagreements are flagged for review.

5. **Validation (the pilot).** Two episodes from each of 202 shows, run through stock pyannote (first 40), the stride-1 reference, and the accelerated variants, compared by diarization error rate. Twenty episodes are hand-labelled at the segment level (host / guest / ad / unsure) to score host identification by precision and recall of host speech.

## Where each piece comes from

| piece | source |
|---|---|
| Segmentation, embedding, clustering, reconstruction | pyannote.audio (Bredin 2023, Interspeech; Plaquet & Bredin 2023, Interspeech, powerset segmentation); WeSpeaker (Wang et al. 2024, *Speech Communication*); VBx (Landini et al. 2022, *Computer Speech & Language*) |
| Coarser segmentation stride, fp16 + batching, and the diagnosis that coarse stride under-counts speakers | Yamaguchi 2026, *Fast and Robust On-Device Speaker Diarization*, arXiv:2606.08505. Reports ~10× over a batched baseline with DER flat on AMI; its relative-minimum-cluster-size fix applies to agglomerative clustering (pyannote 3.1), which we also run in the pilot |
| Per-chunk embedding and the observation that stock pyannote embeds 630 s of audio per 30 s of input | SDBench / SpeakerKit, Pacheco et al. 2025, Interspeech, arXiv:2507.16136 |
| Embedding only active speaker slots | this project; follows from how pyannote's clustering filters and discards inactive slots |
| Choice of pyannote over Sortformer (4-speaker cap) and DiariZen (5× slower) | Lanzendörfer et al. 2025, *Benchmarking Diarization Models*, arXiv:2509.26177 (RTX A6000: pyannote 3.1 45×, DiariZen 20×, Sortformer v2 214×; 2-speaker DER 19.9 / 11.4 / 10.1) |
| Diarizing a podcast corpus at scale; host-name heuristic | SPoRC: Litterer, Jurgens & Card 2025, ACL (pyannote 2.1 on 370k episodes / 220k hours, ~22,000 GPU-hours; host = first voice that says the host's name) |
| Host as the voice that recurs across a show's episodes | arXiv:2206.03400 (2022, dysfluency detection on podcast data; per-show host centroid from the dominant cluster across episodes, matched by cosine distance) |

## What the output looks like

One JSON per episode, `data/diarization/<show_id>/<episode_id>.json` (pilot variants carry a suffix, e.g. `.s3act.json`). Example, *The Dig*, "AMLO Shatters Mexican Establishment", 49 minutes, two speakers, 249 turns, 118 Whisper segments:

```json
{
  "show_id": "1043245989", "episode_id": "0f0caa45c5cf553c",
  "episode_title": "AMLO Shatters Mexican Establishment",
  "duration": 2961.0, "n_speakers": 2,
  "turns": [
    {"start": 0.217, "end": 4.216, "speaker": "SPEAKER_00"},
    {"start": 4.503, "end": 5.937, "speaker": "SPEAKER_00"},
    ...
  ],
  "speakers": {
    "SPEAKER_00": {"duration": 611.2,  "share": 0.216, "centroid": [256 floats]},
    "SPEAKER_01": {"duration": 2220.6, "share": 0.784, "centroid": [256 floats]}
  },
  "segments": [
    {"start": 93.74, "end": 123.52,
     "text": "My name is Daniel Denver, and I'm broadcasting from Providence, Rhode Island ...",
     "speaker": "SPEAKER_00", "purity": 0.879},
    ...
  ],
  "provenance": {
    "model": "pyannote/speaker-diarization-community-1",
    "segmentation_window_s": 10.0, "segmentation_step_s": 3.0,
    "embed_mode": "active", "half": true, "clustering": "VBxClustering",
    "audio": "data/audio_archive/1043245989/0f0caa45c5cf553c.opus",
    "wall_seconds": 10.1, "realtime_factor": 293.8
  }
}
```

- `turns` are pyannote's speaker turns; overlapping speech appears as overlapping turns.
- `speakers` gives each voice's total time, share of the episode's speech, and its centroid embedding. The centroids are what the host identification clusters across episodes, and they also allow matching the same voice across shows (a guest circuit) without names.
- `segments` are the Whisper segments from the transcript with the speaker label and purity added, so text analyses can be restricted to one speaker.
- `provenance` records exactly how the episode was processed.

Host identification writes two tables and, optionally, host-only transcripts:

- `data/output/diarization/episode_speakers.csv`: one row per (episode, speaker) with duration, share, the show-level voice cluster, its recurrence across the show's episodes, and the flags `is_host_centroid`, `is_host_name`, `is_host`.
- `data/output/diarization/show_hosts.csv`: one row per show with the number of hosts found, their recurrence and speaking share, the host names known from metadata, and how often the name check agreed with the voice check.
- `data/transcripts_host/<show_id>/<episode_id>.txt`: the host's segments only (purity ≥ 0.8), the input for rebuilding the register measure on host speech.

## Limits to state

- Stride 3 can merge a voice that has very few embeddings in an episode (a short interjection, or a panel with many speakers). The pilot measures how much this costs on our audio; the 3.1 pipeline with the relative minimum cluster size is the fallback if it matters.
- Host identification by recurrence needs several diarized episodes per show; early in the run, shows with few episodes are flagged low-confidence and fall back to the dominant voice.
- About 1% of episodes have no retrievable audio and remain text-only.
- Speaker labels are per episode; cross-episode identity comes from the centroids, not from pyannote.

## Code

`step4_diarize/` in the repository: `archive_download.py` (audio), `accelerated_pipeline.py` (the three changes as patches on a pretrained pipeline instance), `diarize.py` (run, thermal gate, segment labelling), `identify_hosts.py`, `pilot_eval.py`, `profile_stages.py`, with a README.

## Addendum, 20 November 2026: corpus run and host status

Diarization of all 81,882 episodes with audio finished 20 Nov 2026 (12 days,
no failures). Host identification now distinguishes three cases per episode
(`data/output/diarization/episode_hosts.csv`, column `host_status`):

| status | episodes | meaning |
|---|---|---|
| host_found | 72,076 (88%) | a recurring host voice is present |
| host_absent | 1,828 (2%) | the show has a fixed host (top voice recurs in ≥ 70% of episodes) but this episode has none of it: guest-hosted, re-run or special |
| no_fixed_host | 7,977 (10%) | the show has no voice recurring in 70% of episodes (28 of 202 shows: compilation and rotating-presenter feeds such as Slate News, Politico Playbook, Jacobin Radio) |

Every speaker also carries a `role`: host, regular (a non-host voice recurring
in ≥ 10% of the show's episodes, 125,721 speaker slots) or guest (301,773).
A speaker is a host if it belongs to a host cluster or its cosine similarity
to the show's host voice is ≥ 0.6 (self-introduced hosts sit at 0.93, other
speakers at 0.13, 99th percentile 0.52). Voice and self-introduction by name
agree on 92% of 1,877 cases across 68 shows. Host-only text is written only
for host_found episodes. These numbers are from the first pass; 48% of
episodes have transcripts being re-aligned to the archived audio
(`retranscribe.py`, finishing ~28 Nov), after which the tables are regenerated.
