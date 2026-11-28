# The political podcast corpus: a data overview

**Code, tables of record and this document:** https://github.com/dsyang1219/Podcast-Project (branch `main`). Bulk data (audio, transcripts, speaker-layer files) stays on the lab machine; the tables and scripts that produce every number here are in the repository.

*State of the data on 28 November 2026. Numbers come from the files named in each section; the episode-level table behind most of them is `data/output/corpus_episode_table.csv`, rebuilt from the transcripts on this date. The frame is 219 shows, all transcribed; the fifteen reinstated on 23 November 2026 were transcribed the next day and are included in every count. The ten out-of-frame shows that were transcribed for a separate generalization test (section 4) are not counted anywhere in this document.*

## 1. What it is, in one paragraph

A transcribed and speaker-labelled archive of the US political podcast chart. It starts from the 250 shows on the Apple Podcasts US Politics chart on 13 July 2026, keeps the 219 with a working feed, US political content and a usable catalogue, and samples their back catalogues by a fixed rule: every episode published before 2018 and up to 30 episodes per show per quarter after that. The result is 85,127 transcribed episodes, 75,029 hours of audio, 749 million words, covering March 2006 to July 2026, with 2026 alone contributing 12,181 episodes in six and a half months. Every episode with retrievable audio (82,129, 96.5%) also carries speaker turns, per-speaker voice embeddings and a host label. Alongside the text sit show-level measures (listener-directed address, one-sidedness, outrage, register), host identities and donations, Brookings partisan labels, two waves of a national survey matched to the shows by respondents' own answers, Pew microdata and 16,434 Apple reviews.

## 2. The frame: which shows

**Principle.** The population is defined by a public, external classification and a fixed set of rules, not by a judgment of which shows matter. Every exclusion is recorded with its evidence, and the frame can be rebuilt from one frozen file.

**Source.** The Apple Podcasts US chart for the *Politics* subcategory (genre 1527, under News), pulled once on 13 July 2026 to its maximum depth of 250 and frozen (`data/output/raw_chart_20260713.csv`). A daily panel of the same chart was then collected from 13 July to 11 September 2026: 24 snapshots, 302 distinct shows ever charted (`chart_panel.csv`), which is what the chart-rank and charting-persistence variables draw on.

**Rules**, applied in fixed order; the first rule a show fails is its recorded reason (`exclusions.csv`, 31 shows):

| rule | shows | examples |
|---|---|---|
| R1 no feed URL | 2 | |
| R2 feed unreachable | 2 | |
| R3 feed unparseable | 1 | |
| R5 not US politics | 14 | The Rest Is Politics (UK), Americast (BBC), TRIGGERnometry (UK), NARITAS (French), El hilo (Spanish) |
| R6 under 10 hours of audio in the catalogue | 12 | The Five, Fox News Sunday Audio, PBS News Hour Brooks and Capehart, The Economist's Tocqueville Road Trip: clip feeds |

That leaves 219 shows. Fifteen of them were set aside by a recorded decision on 14 August 2026, because they were not in the first transcript set when the measures were built and the frame was fixed (Candace Owens, HasanAbi, Autocracy in America, Checks and Balance from The Economist, Deseret Voices, Essential Minnesota, Extremely American, Get Angry, It's That Part, Make It Make Sense, Mueller She Wrote, The Loren Piretra Show, The Nerd Reich, Uncertain with DL Hughley, Wired In). On 23 November 2026 they were reinstated for the rebuild of the analysis on the finished corpus, so the frame is exactly the **219 rule-passing shows**, listed in Appendix A (`data/output/frame_decisions.md`). Fourteen of the fifteen are at the catalogue floor (10–55 hours); the fifteenth is the Daily Wire-era *Candace Owens* feed (302 hours), a second feed of a host whose current show *Candace* was already in. Their ladder episodes were transcribed on 24 November 2026 (748 of 750; 412 hours) and are included throughout. `corpus.csv` carries each show's chart rank, publisher, feed URL, Apple genre fields, feed author and owner, feed episode count (153,102 episodes listed across the 219 feeds, median 416 per show) and hours available (132,163). Publishers are fragmented: iHeartPodcasts (9 shows), The Bulwark (6), Fox News Podcasts (5), NPR, Premiere Networks and MSW Media (3 each) are the largest. Brookings' partisan labels were joined afterwards to validate the one-sidedness measure and never used to select shows.

**The analysis set.** The paper's models use **194 of the 204** (`show_level_outcomes.csv`): the frame frozen on 18 September 2026 as a table of 31,191 episodes (census and bands 1–5 complete, bands 6–15 partial at the freeze), with the 204 → 194 step recorded in `results/mfte_measure.md`. The 194 hold 83,689 of the 85,127 episodes (73,958 hours, 738 million words); 125 lean left and 69 right by the one-sidedness measure. The full corpus described here is the superset that finished transcribing on 6 November 2026; the paper's frozen table does not change.

## 3. The sampling design: which episodes

Catalogues run from 21 episodes to many thousands, so a census would be a corpus of daily news feeds. Episodes were drawn instead by a priority ladder (`step2_sample/sample_ladder.py`) that gives every show-quarter equal weight up to a cap:

| stratum | rule | episodes |
|---|---|---|
| census | every episode published before 1 January 2018: the thin, irreplaceable tail | 6,207 |
| ladder, bands 1–15 | the k-th random draw per show-quarter, k = 1 … 15 | 49,325 |
| ladder, bands 16–30 | k = 16 … 30 | 24,878 |
| pre-ladder draw (H25) | the first sample, up to 25 hours per show from 2024–2026, drawn before the ladder was adopted; its episodes outside the bands are kept | 3,945 |
| reinstated shows | the same ladder drawn on 23 November 2026 for the fifteen reinstated shows (plus 24 earlier transcripts) | 772 |

**A worked example.** Pod Save America has 1,200 episodes in its feed and 1,089 in the corpus. Its quarters before 2018 are in the census in full (24 episodes in 2017Q1, 29 in 2017Q4). From 2018 on, each quarter contributes its first five random draws to bands 1–5, the next ten to bands 6–15, and up to fifteen more to bands 16–30, so a quarter with 31 episodes (2025Q1) gives 5 + 10 + 15 = 30 and a quarter with 25 (2021Q1) gives 5 + 10 + 10 = 25, all of it. A small show is a census throughout: The Moment with Jorge Ramos and Paola Ramos has 43 episodes in its feed and all 43 in the corpus. Across the 3,826 show-quarters with ladder draws, the 30-episode cap binds in 37%, the median show-quarter contributes 19 episodes, and 7% contribute five or fewer. The band-1 slice alone (one episode per show-quarter) is 3,820 episodes; bands 1–5 are 18,504.

Within a show-quarter the draw order is random, so any prefix of bands is itself a stratified random sample: band 1 alone gives one episode per show-quarter, bands 1–5 give five. That is what let the corpus grow from 15 bands to 30 without redrawing anything. Bands fill unevenly because many show-quarters have fewer than 30 episodes: band 16 has 1,958 episodes and band 30 has 1,330. The design manifests are in `data/output/sample_out/`. The ladder finished on 6 November 2026; about 1,800 manifest rows never transcribed because their audio was gone or blocked (HTTP 403/404/410), concentrated in a handful of shows. The cost of the design is that episode counts are not proportional to a show's output, so everything is analysed as per-show or per-word rates, never as totals.

## 4. Why some well-known shows are not in it

The Joe Rogan Experience, The Ben Shapiro Show, The Charlie Kirk Show, The Dan Bongino Show, The Glenn Beck Program, The Megyn Kelly Show, The Tucker Carlson Show, The Shawn Ryan Show and The Daily are absent because none of them is on the Politics chart. Apple files Rogan and Shawn Ryan under Comedy and Society & Culture, and the others under News, and the shows classify themselves that way. The frame's definition of a political podcast is the category as Apple and the show itself define it; adding shows by hand would replace a rule with an opinion, and the same opinion would then be owed to every talk show that discusses politics some of the time.

The classification was checked against content rather than taken on trust. Because survey respondents named these shows, 50 episodes of each were transcribed and scored with the same LLM passage rubric used for the corpus, then set aside. The share of passages that are political confirms the category boundary:

| show | political passages per 10,000 words | share of passages political | percentile within the 204-show frame |
|---|---|---|---|
| The Joe Rogan Experience | 19 | 0.03 | 1st |
| The Shawn Ryan Show | 31 | 0.05 | 2nd |
| The Megyn Kelly Show | 45 | 0.07 | 3rd |
| The Glenn Beck Program | 82 | 0.15 | 11th |
| The Tucker Carlson Show | 91 | 0.17 | 15th |
| The Charlie Kirk Show | 93 | 0.18 | 16th |
| The Daily | 95 | 0.18 | 16th |
| The Ben Shapiro Show | 122 | 0.22 | 31st |
| The Dan Bongino Show | 143 | 0.29 | 38th |
| *frame median* | *157* | *0.31* | *50th* |

Rogan talks about politics in about one passage in thirty, Shawn Ryan in one in twenty; both sit below the 2nd percentile of the frame. Kelly is at the 3rd. Shapiro and Bongino would fit by content but are not in the category chart, and the rule is the category, applied uniformly. These ten shows were used once, as a pre-registered test of whether the address-to-attitude association generalizes outside the frame; it did not, and the single estimate that moved was the one that dropped Rogan, whose audience resembles no other. They are excluded from every number in this document and from the released corpus; the record of that test is `results/holdout_RESULTS.md`.

## 5. The transcripts

**Engine.** faster-whisper, CTranslate2, fp16, batched, on one GB10 GPU. 77,501 episodes with `deepdml/faster-whisper-large-v3-turbo-ct2`; the first 7,626 (the earliest runs) with `large-v3`. Every episode is tagged `en`. Throughput was about 106× realtime under a start-temperature gate.

**Format.** One JSON and one TXT per episode under `data/transcripts/<show_id>/<episode_id>_<title>.json`. The JSON holds `episode_id`, `show_id`, `episode_title`, `engine`, `model`, `language`, `duration`, `audio_sha256` and `segments`, each segment `{start, end, text}` in seconds. There are no word timestamps. The TXT is the segment texts joined by newlines.

**Size and shape.**

| | |
|---|---|
| shows | 219 |
| episodes | 85,127 |
| audio hours | 75,029 |
| words | 749.3 million |
| Whisper segments | 10.16 million; median 115 per episode; mean 26.6 s |
| episode length | mean 52.9 min, median 50.5, 10th–90th percentile 15–90 min, longest 12.8 h; 1,339 under 5 min; 490 over 3 h |
| speech rate | median 167 words per minute (10th–90th: 143–189) |
| publication dates | all known; 3 March 2006 to 15 July 2026 |

**Time coverage.** Heavily recent, by design (the pre-2018 census is small because the shows were) and by the medium's growth:

| year | episodes | hours | | year | episodes | hours |
|---|---|---|---|---|---|---|
| 2006–2010 | 498 | 322 | | 2019 | 3,336 | 3,158 |
| 2011 | 756 | 643 | | 2020 | 4,891 | 4,483 |
| 2012–2014 | 1,091 | 838 | | 2021 | 6,543 | 5,898 |
| 2015 | 665 | 584 | | 2022 | 8,801 | 7,675 |
| 2016 | 942 | 702 | | 2023 | 10,556 | 9,332 |
| 2017 | 2,255 | 2,144 | | 2024 | 13,110 | 11,328 |
| 2018 | 2,652 | 2,377 | | 2025 | 16,826 | 14,878 |
| | | | | 2026 (to 15 July) | 12,181 | 10,650 |

**Show coverage.** Episodes per show: median 295, from 18 to 2,476 (Renegade Talk Radio). Hours per show: median 240, from 10 (the reinstated shows sit at the catalogue floor) to 2,580. Concentration is moderate: the ten largest shows hold 19% of the hours, the thirty largest 43%. The largest by hours are Renegade Talk Radio (2,580 h), Slate News (1,453), The Buck Sexton Show (1,400), On the Media (1,358), The Peter McCormack Show (1,354), Jacobin Radio (1,348), Pod Save America (1,282) and Opening Arguments (1,253).

**Alignment caveat.** For 48% of episodes the archived audio (section 6) differs in length from the file that was transcribed, because the shows' ad servers splice different ads into each download. The text of those transcripts is right; their timestamps are right for a file that no longer exists. They are being re-transcribed from the archived audio into `data/transcripts_v2/` (same schema plus `source: "archive"`); three-quarters of the corpus is aligned as of 28 November and the rest finishes about 3 December. Episodes archived during transcription itself, the whole of bands 16–30, are unaffected.

## 6. The audio archive

Audio was not kept during the first transcription runs. It was re-fetched from the feeds in November 2026 and stored as 16 kHz mono Opus at 24 kb/s, `data/audio_archive/<show_id>/<episode_id>.opus`, about 9.7 MB per audio hour and 730 GB for the corpus. 2,998 episodes (3.5%) have no retrievable audio: 2,790 HTTP 403, 133 HTTP 404, 53 HTTP 410. The blocks are real rather than rate-induced (a gentle retry recovered 3 of 2,976) and are concentrated in six shows: Jacobin Radio (732 episodes), Timothy Gordon (416), Human Events Daily (336), Antiwar News (283), The Kyle Kulinski Show (263) and Week In Review (215). Those episodes stay text-only. `data/output/archive_failures.csv` lists them.

## 7. The speaker layer

**Method.** pyannote `speaker-diarization-community-1` (powerset segmentation, WeSpeaker ResNet34 embeddings, VBx clustering), unchanged in models and parameters, with three speed changes that leave output within a DER of 0.017–0.024 of stock pyannote: a 3 s segmentation step, embedding only speaker slots that contain speech, and fp16 for the ResNet. 300× realtime; the corpus took 12 days (8–20 November 2026). Full account and pilot in `results/diarization_methods.md` and `results/diarization_pilot.md`.

**Format.** `data/diarization/<show_id>/<episode_id>.json`: `turns` (start, end, speaker; overlaps allowed), `exclusive_turns` (non-overlapping), `speakers` (per voice: duration, share, 256-d centroid), `segments` (the Whisper segments with `speaker` and `purity`, the share of the segment the winning speaker covers), and `provenance` (configuration, audio and transcript paths, whether the transcript is aligned to the audio, timing).

**What it contains.**

| | |
|---|---|
| episodes | 82,129 of 85,127 (96.5%, every episode with audio), in 207 shows |
| speaker slots | 519,136; median 5 per episode (10th–90th: 2–12; max 79); median 3 with ≥ 5% of the speech |
| turns | median 554 per hour (10th–90th: 280–949); median length 2.4 s, mean 5.3 s, 90th percentile 12.8 s; 45% under 2 s |
| overlapping speech | median 0% of speech |
| segment purity | median 0.89; 65% of segments ≥ 0.8; 15% below 0.6 |

**Hosts.** Identified per show by recurrence: a voice is a host if it appears in at least half the show's diarized episodes and holds at least 10% of its speech (up to three hosts), or if its cosine similarity to the show's host voice is ≥ 0.6. Every speaker slot has a `role`: host (92,188 slots), regular (a non-host voice recurring in ≥ 10% of the show's episodes; 125,545), or guest (301,403). Every episode has a `host_status` (`data/output/diarization/episode_hosts.csv`):

| status | episodes | meaning |
|---|---|---|
| host_found | 72,313 (88%) | a recurring host voice is present; host share of speech median 0.52 (10th–90th: 0.19–0.98); 23% of these episodes have two or more hosts |
| host_absent | 1,837 (2%) | the show has a fixed host but this episode has none of it: guest-hosted, re-run, special |
| no_fixed_host | 7,979 (10%) | one of 29 shows with no voice recurring in 70% of episodes: compilation and rotating-presenter feeds (Slate News, the Politico feeds, Jacobin Radio, …) |

178 of the 207 diarized shows have a fixed host (152 with one host, 43 with two, 12 with three); the 12 shows without any diarized episode are those whose audio is entirely blocked. Show-level tables: `show_hosts.csv` (hosts, recurrence, share, name agreement) and `episode_speakers.csv` (one row per slot with share, role, recurrence, host similarity, flags).

**Validation.** Against stock pyannote on 404 pilot episodes: DER 0.024, all of it confusion among voices under 2% of an episode. Against 20 hand-labelled episodes from 20 shows (859 segments): host speech identified with precision 0.915 and recall 0.857 at the segment level, 0.979 and 0.921 on segments with purity ≥ 0.8; at the speaker level precision 1.00 and recall 0.69, the misses being rotating or fill-in presenters and low-share recurring voices rather than acoustic errors. Voice and self-introduction by name agree in 93% of 2,673 cases across 72 shows.

## 8. Derived measures

All measures use published tools or LLM rubrics; none uses a hand-made word list.

| measure | what | where | coverage |
|---|---|---|---|
| **Listener-directed address** | MFTE (Le Foll 2021, Stanza-based Biber tagger): imperatives (VIMP) and second-person pronouns (PP2) per 10,000 words; word-weighted show means; z on the 194 | `step7_audience/inputs/mfte_show_scores.csv`, `mfte_episode_rates.csv` | 45,192 episodes, 409 million words; VIMP median 30.1 per 10k (10th–90th: 17.5–52.6), PP2 median 215 (161–286) |
| Command types | VIMP split by Biber 2006 verb class (activity, communication, mental, causative, aspectual, do-auxiliary), plus "you know" and "let's" | `mfte_imperative_episode_counts.csv`, `mfte_imperative_show_scores.csv` | same 45,192 episodes |
| Full register | 40,242 chunks on Biber-style dimensions (citation density, abstraction, news markers, entity mix, sentence complexity, person) | `data/output/register_features_500.csv`, `mfte_show_register_dims.csv` | |
| Style measures | othering, we/they, direct address, first person, certainty, hedging, questions, negation, moral-emotional language, concreteness (Brysbaert norms) | `data/output/style_measures.csv` | 29,161 episodes |
| Political density | share of passages that are political, from the same rubric | `step7_audience/inputs/corpus_poldensity.csv` | 219 frame shows; median 157 per 10k words |
| **One-sidedness / side** | Much et al. (2026) two-stage LLM rubric (gpt-5.4-mini) on 40-passage caps per show | `ideology_cap40.csv`, `ideology_sideclass.csv`, `show_level_outcomes.csv` (`inten`, `side`, `lean`) | 152,152 passages; 125 L / 69 R among the 194; validated against Brookings (κ .83 on 24 matched shows), host donations (DIME, r .68) and listeners' party (r .92) |
| Outrage | Sobieraj & Berry categories by LLM (insult, name-calling, mockery, …) | `outrage_llm_show_rates.csv`, `outrage_llm_show_scores.csv` | 219 shows |
| Hosts and donations | host names from feed metadata and Wikipedia, DIME CFscores | `host_lookup.csv`, `host_dime_lookup_v2.csv`, `shows_205_host_dime_scores.csv` | 122 shows with a person-named host |
| Partisan labels | Brookings podcast list, fuzzy-matched | `lean_validation.csv` (use `match_status == "matched"` only: 29 shows) | |
| Show outcomes | chart rank, episodes per week, episode length, ads, review counts | `show_level_outcomes.csv` | 194 shows |

## 9. Audience and external data

- **Kettering Foundation / Gallup "Democracy for All" surveys**, two waves: Year 1 (n = 20,338) and Year 2 (n = 23,683). Respondents named their own news sources in open text; those answers were hand-matched to the shows (alias tables, frozen and excluded from the public repository). The pooled match is 1,247 listeners of 64 shows (`step7_audience/inputs/match_strict.csv`, `match_expanded.csv`). Respondent-level microdata are restricted and not in the repository.
- **Pew American Trends Panel**, waves 165 and 118: education, party and age profiles of the audiences of 30 named sources (`pew_w165_source_audiences.csv`) and the main-podcast and host-connection items of W118. Microdata restricted.
- **Apple Podcasts reviews**: 16,434 reviews across 141 shows with date, rating, title, text and a hashed author id (`apple_reviews.csv`, raw file not public); readability and a parasocial-language pilot coding derive from it.

## 10. Metadata: a data dictionary

Everything below joins on `show_id` (Apple collection id) and, where episode-level, `episode_id` (16-hex hash of the feed item). Coverage is given where it is not complete.

**Show level, from the chart and the feeds** (`data/output/corpus.csv`, 219 rows; `corpus_show_list.csv` for the 204)

| field | meaning |
|---|---|
| `rank` | position on the Apple US Politics chart, 13 July 2026 |
| `collection_id`, `show_name`, `publisher`, `apple_url`, `artwork_url` | Apple catalogue identity |
| `primary_genre`, `genres`, `apple_category`, `apple_subcategory` | Apple's classification (all Politics under News) |
| `feed_url`, `rss_description`, `rss_language`, `rss_author`, `rss_owner_name` | the RSS feed and its self-description (owner name missing for 4%) |
| `track_count`, `episode_count`, `episodes_with_audio` | catalogue size as listed by Apple and by the feed |
| `hours_available`, `hours_estimated_share`, `avg_episode_minutes`, `episodes_per_week` | catalogue volume and cadence from feed durations (share of durations estimated rather than declared) |
| `first_pub_date`, `last_pub_date`, `release_date` | span of the feed; Apple's last-release stamp |
| `lookup_found`, `country` | Apple lookup succeeded; storefront |

**Show level, chart dynamics** (`chart_panel.csv`, 6,713 rows: one per show per snapshot, 24 snapshots 13 July – 11 September 2026): `chart_date`, `show_id`, `in_corpus`, `charted`, `rank`, `publisher`. Aggregated into `show_level_outcomes.csv` as `n_dates_charted`, `mean_rank`, `best_rank`, `first_rank`, `last_rank`, `rank_sd`, `persistence`, `slope_charted`, `slope_censored`, `charted_last`.

**Show level, hosts and donations** (`host_lookup.csv`: show, extracted host, source of the name [feed author, owner, title parse], confidence; `host_dime_lookup_v2.csv`, 288 host rows: `host_name`, `final_source`, `confidence`, `final_cfscore` (DIME CFscore, 153 hosts), `real_occupation` (108), `matched_bonica_cids` (67), alternative names, notes; `shows_205_host_dime_scores.csv`: per show the number of hosts, how many scored, coverage, `avg_host_cfscore`).

**Show level, reviews** (`apple_reviews.csv`, 16,434 rows: `show_id`, `review_id`, `date`, `rating`, `title`, `text`, `author_hash`; aggregated in `show_level_outcomes.csv` as `n_reviews`, `mean_rating`, `share5`, `share1`, `first`, `last`, `rev_len`, `span_days`, `reviews_per_month`, `capped`; 66% of the 194 shows have reviews).

**Show level, measures** (`show_level_outcomes.csv`, 194 rows): `inten` (one-sidedness), `side` (signed left–right), `lean` (L/R), `mfte_z`, `vimp_z`, `pp2_z` (address), `addr_full`, `addr_noad` (address with and without ad passages), `outrage_z`, `ad_share`, `solo`, `n_hosts`, `ep_len`, `age_years`, plus the chart and review aggregates above and log transforms. Full feature sets: `mfte_show_features_all.csv` (155 columns: every MFTE / Biber feature per 10,000 words at show level, e.g. `PP1all`, `PASSall`, `NNP`, `ThVSTNCall`, type–token ratio `TTR`, average word length `AWL`), `mfte_show_register_dims.csv` (194 shows × the tag set), `mfte_imperative_show_scores.csv` (command types), `outrage_llm_show_scores.csv` (219 shows: 12 outrage categories, `any`, `n_types`, `passages`, `outrage_z`), `corpus_poldensity.csv` (political density), `lean_validation.csv` (Brookings join with `match_status`).

**Episode level, from the feeds and the design** (`data/output/sample_out/corpus_episodes.csv` and the ladder manifests): `episode_id`, `show_id`, `episode_title`, `pub_date`, `duration_sec`, `duration_source` (declared in the feed or estimated), `audio_url`, `quarter`, `priority` (band: 0 = census, 1–30 = ladder band), `draw_order`, `stratum`, `done`. Consolidated for the whole corpus in `corpus_episode_table.csv` (84,879 rows): plus `duration` (measured from the audio), `n_segments`, `n_words`, `language`, `model`, `design`, `band`.

**Episode level, transcript** (`data/transcripts/<show_id>/<episode_id>_<title>.json`): `episode_id`, `show_id`, `episode_title`, `engine`, `model`, `language`, `duration`, `audio_sha256`, `segments[{start, end, text}]`. Re-transcribed episodes (`data/transcripts_v2/`) add `source`, `audio`, `original_transcript`, `realtime_factor`.

**Episode level, measures** (`mfte_episode_rates.csv`, 45,192 rows: `words`, `vimp`, `pp2`; `mfte_imperative_episode_counts.csv`: command types and "you know" / "let's" counts; `style_measures.csv`, 29,161 rows: `othering`, `they_per10k`, `we_per10k`, `direct_address`, `first_person`, `certainty`, `hedge`, `question_rate`, `negation`, `moral_emo`, `concreteness`; `ep_dates.csv`).

**Passage level** (`ideology_cap40.csv`, 152,152 rows: `chunk_id`, `politics` [is the passage political], `ideology` [left–right score], `ternary`; `scoring_chunks_*.csv` carry the passage text, `show_id`, position and word count; `register_features_500.csv`, 40,242 chunks: `citation_density`, `abstraction`, `present_tense_news_markers`, `entity_mix`, `sentence_complexity`, `first_vs_third_person`, token, sentence and entity counts).

**Episode level, speaker layer** (`data/diarization/<show_id>/<episode_id>.json`): `turns[{start, end, speaker}]`, `exclusive_turns`, `speakers{label: {duration, share, centroid[256]}}`, `segments[{start, end, text, speaker, purity}]`, `provenance{model, segmentation_window_s, segmentation_step_s, batch_size, embed_mode, half, clustering, audio, transcript, transcript_duration, transcript_source, aligned, wall_seconds, decode_seconds, realtime_factor}`. Tables derived from it:

| table | one row per | fields |
|---|---|---|
| `episode_hosts.csv` | episode | `n_speakers`, `host_share`, `n_hosts`, `n_regular`, `show_fixed_host`, `aligned`, `host_status` |
| `episode_speakers.csv` | episode × speaker | `duration`, `share`, `role` (host / regular / guest), `global_cluster` (voice cluster within the show), `recurrence` (share of the show's episodes the voice appears in), `host_sim` (cosine to the show's host voice), `is_host_centroid`, `is_host_name`, `is_host`, `low_confidence`, `show_fixed_host`, `aligned` |
| `show_hosts.csv` | show | `n_episodes`, `n_global_clusters`, `fixed_host`, `n_hosts`, `n_regular_voices`, `host_recurrence`, `host_share`, `host_names_known`, `name_intro_episodes`, `name_agree`, `name_disagree`, `low_confidence`, `n_unaligned` |

**Survey and external** (restricted microdata, not in the repository): Kettering Year 1 and Year 2 respondent files with the open-text news-source answers, the frozen alias tables that map those answers to `show_id`, and the matched-listener files (`match_strict.csv`, `match_expanded.csv`: respondent id, wave, `show_id`, match type); Pew ATP W165 and W118 microdata; the public aggregates are `pew_w165_source_audiences.csv` (`source`, `n`, weighted shares with a BA, with high school or less, Republican, aged 65+).

## 11. Where things live

```
step1_frame/        chart scrape, feed parsing, exclusion rules, Brookings join
step2_sample/       the priority ladder and manifests
step3_transcribe/   downloader, transcriber, supervisor with thermal gate
step4_diarize/      audio archive, accelerated pyannote, host identification,
                    re-transcription, hand-label kit, pilot evaluation
step4_ideology/     passage sampling and LLM scoring of side / one-sidedness
step5_hosts/        host extraction and Wikipedia / DIME resolution
step6_register/     MFTE address, command classes, register, style, outrage
step7_audience/     survey match, validation, population, pre-registration,
                    exploratory, reviews, Pew, education probes
data/transcripts/   episodes (json + txt)                                 8.9 GB
data/transcripts_v2 audio-aligned re-transcriptions, in progress
data/audio_archive/ Opus audio                                            725 GB
data/diarization/   speaker-layer JSONs                                   9.6 GB
data/output/        show and episode tables, measures, failure logs
results/            the written record of every analysis and decision
scans/, figures/    result scans and paper figures
```

Identifiers: `show_id` is the Apple collection id; `episode_id` is a 16-hex-character hash of the feed item, stable across manifests and renames. Every table joins on these two. `data/output/corpus_show_list.csv` is the machine-readable form of Appendix A.

## 12. Known limits

- **Not all of the chart.** 31 of 250 chart shows are excluded by rule; the 194 analysis shows used in the paper are a frozen subset of the 219-show frame; and the large cross-genre talk shows are outside it by construction (section 4).
- **Sampling, not census, after 2017.** Prolific shows are capped at 30 episodes per quarter, so episode counts across shows are not proportional to output.
- **Audio gaps.** 3.5% of episodes have no audio and therefore no speaker layer; six shows account for most of it.
- **Timestamp alignment.** About a quarter of transcripts are still being re-aligned to the archived audio (finishing about 3 December); until then those episodes are flagged `aligned: false` and excluded from host-only text. The sandbox is fully aligned.
- **Segment granularity.** Whisper segments (mean 26.6 s) can hold two speakers; the purity score marks them, and word-level alignment would be needed for turn-level text.
- **Hosts.** Rotating-presenter feeds have no fixed host by construction; fill-in presenters and quiet co-presenters under 10% of a show's speech are not labelled host.
- **Transcription errors.** No systematic WER measurement on this corpus; Whisper large-v3 class models run around 5–10% WER on clean English podcast speech in published evaluations, higher on crosstalk, phone-quality guests and names.
- **Copyright.** Full transcripts of commercial podcasts are not redistributable as such; the release plan is the derived layers (timestamps, speakers, embeddings, measures) with scripts that regenerate text from the public feeds.

## 13. The sandbox: a slice to explore

`sandbox/` (5.4 GB; `step2_sample/build_sandbox.py`) is the k = 5 design cut out of the corpus as a self-contained folder: the 6,207 census episodes plus ladder bands 1–5, 18,754 episodes, for all 219 shows, rebuilt on 27 November 2026 with every episode's transcript aligned to its audio. Because the ladder's draw order is random within a show-quarter, the slice is a stratified random sample of the frame, and its distributions estimate the corpus's. It holds the transcripts (the audio-aligned version where one exists), the speaker-layer JSONs, the host tables, the address measures, a one-row-per-episode table and a show table, plus the plots in `sandbox/figures/`.

| | sandbox | full corpus |
|---|---|---|
| episodes | 24,961 | 85,127 |
| audio hours | 22,058 | 75,029 |
| words | 216.2 million | 749.3 million |
| episodes with a speaker layer | 23,896 (95.7%) | 82,129 (96.5%) |
| transcripts aligned to the archived audio | 100% | 76% (rest by ~3 Dec) |
| episode length, median (10th–90th pct) | 50.9 min (18.3–87.6) | 50.5 min (15.1–90.4) |
| episodes per show, median (min–max) | 84 (6–1,591) | 295 (18–2,476) |
| speakers per episode, median | 5 | 5 |
| host found / absent / no fixed host | 88.0% / 1.8% / 10.1% | 88.0% / 2.2% / 9.7% |
| host share of speech, median | 0.52 | 0.52 |
| speech rate, median | 167 wpm | 167 wpm |
| 2024 / 2025 / 2026 episodes | 2,991 / 3,556 / 2,848 | 13,110 / 16,826 / 12,181 |

The per-show maximum is the census: a show with a large pre-2018 catalogue contributes all of it, while post-2018 contributions are capped at five per quarter. `sandbox/README.md` documents the files and the distributions; `--bands 1` or `--bands 15` rebuild the slice at other depths.

## Appendix A. The 219 shows in the corpus

Chart rank on 13 July 2026; episodes and hours transcribed; lean from the one-sidedness measure (L/R, the 194 analysis shows only); fixed host = a voice recurs in ≥ 70% of diarized episodes; political density = political passages per 10,000 words (median 157 across the frame). Shows outside the 194-show analysis set are starred; the fifteen reinstated on 23 November 2026 are marked †.

| rank | show | publisher | episodes | hours | years | lean | fixed host | pol./10k |
|---|---|---|---|---|---|---|---|---|
| 1 | Pod Save America | Pod Save America | 1,089 | 1,283 | 2017–2026 | L | yes | 228 |
| 2 | The MeidasTouch Podcast | MeidasTouch Network | 735 | 512 | 2020–2026 | L | yes | 205 |
| 3 | Bulwark Takes* | The Bulwark | 141 | 72 | 2026–2026 |  | no | 153 |
| 4 | The Bulwark Podcast | The Bulwark | 264 | 268 | 2024–2026 | L | yes | 173 |
| 5 | Letters from an American | Heather Cox Richardson | 389 | 75 | 2022–2026 | L | yes | 421 |
| 6 | Candace | Candace Owens | 271 | 254 | 2024–2026 | R | yes | 58 |
| 7 | The President's Daily Brief | The First TV | 551 | 189 | 2022–2026 | R | yes | 147 |
| 8 | The New Yorker Radio Hour | WNYC Studios and The New Yor | 1,046 | 590 | 2015–2026 | L | yes | 85 |
| 9 | The NPR Politics Podcast | NPR | 832 | 274 | 2020–2026 | L | yes | 298 |
| 10 | Real Baron Podcast* | Baron Coleman | 38 | 129 | 2026–2026 |  |  | 46 |
| 11 | Verdict with Ted Cruz | Premiere Networks | 638 | 457 | 2020–2026 | R | yes | 237 |
| 12 | Fast Politics with Molly Jong-Fast | iHeartPodcasts | 471 | 448 | 2022–2026 | L | yes | 204 |
| 13 | The Necessary Conversation | The Necessary Conversation | 192 | 188 | 2022–2026 | L |  | 179 |
| 15 | The Next Level | The Bulwark | 344 | 280 | 2022–2026 | L | yes | 183 |
| 16 | Radio Atlantic | The Atlantic | 370 | 213 | 2017–2026 | L | no | 148 |
| 17 | The Political Scene / The New Yorker | The New Yorker | 137 | 101 | 2025–2026 | L | no | 165 |
| 18 | Ruthless Podcast | Josh Holmes, Comfortably Smu | 661 | 844 | 2020–2026 | R | yes | 149 |
| 19 | Runaway Country with Alex Wagner | Runaway Country with Alex Wa | 76 | 75 | 2018–2026 | L | no | 224 |
| 20 | Raging Moderates with Scott Galloway and Jessica Tarlov | Vox Media Podcast Network | 179 | 137 | 2024–2026 | L | yes | 178 |
| 21 | Breaking Points with Krystal and Saagar | iHeartPodcasts | 653 | 881 | 2021–2026 | L | yes | 155 |
| 22 | UnJustified | MSW Media | 452 | 515 | 2017–2026 | L | yes | 192 |
| 23 | On the Media | WNYC Studios | 1,892 | 1,358 | 2011–2026 | L | yes | 115 |
| 24 | Pod Save the World | Pod Save the World | 537 | 635 | 2017–2026 | L | yes | 148 |
| 25 | The Briefing with Jen Psaki | MS NOW, Jen Psaki | 324 | 250 | 2023–2026 | L | yes | 265 |
| 26 | #SistersInLaw | Politicon | 316 | 341 | 2021–2026 | L | yes | 178 |
| 27 | Bannon`s War Room | WarRoom.org | 527 | 492 | 2022–2026 | R | yes | 154 |
| 28 | X22 Report | X22 Report | 167 | 232 | 2025–2026 | R | yes | 211 |
| 29 | All In with Chris Hayes | MS NOW, Chris Hayes | 175 | 137 | 2025–2026 | L | yes | 248 |
| 30 | The Jim Acosta Show | Jim Acosta | 154 | 136 | 2025–2026 | L | yes | 189 |
| 31 | Deadline: White House | Nicolle Wallace, MS NOW | 177 | 146 | 2018–2026 | L | yes | 265 |
| 32 | NBC Nightly News with Tom Llamas | NBC News | 437 | 166 | 2018–2026 | L | no | 115 |
| 33 | The Wright Report | Bryan Dean Wright | 428 | 219 | 2023–2026 | R | yes | 182 |
| 34 | Talking Feds | Harry Litman | 518 | 440 | 2019–2026 | L | yes | 220 |
| 35 | Breaking News from Pod Save America | Crooked Media | 171 | 61 | 2025–2026 | L | no | 196 |
| 36 | The Lincoln Project | The Lincoln Project | 583 | 434 | 2021–2026 | L | no | 162 |
| 37 | Political Gabfest | Slate Podcasts | 1,112 | 1,032 | 2006–2026 | L | yes | 162 |
| 38 | The Fox News Rundown | FOX News Podcasts | 265 | 105 | 2025–2026 | R | yes | 187 |
| 39 | The Rest Is Politics: US | Goalhanger | 222 | 155 | 2024–2026 | L | yes | 220 |
| 40 | Advisory Opinions | The Dispatch | 657 | 754 | 2019–2026 | R | yes | 140 |
| 41 | The Focus Group Podcast | The Bulwark | 162 | 149 | 2022–2026 | L | yes | 243 |
| 42 | This Week with George Stephanopoulos | ABC News | 60 | 50 | 2025–2026 | R | yes | 275 |
| 43 | Impolitic with John Heilemann | Audacy / Puck | 382 | 472 | 2020–2026 | L | yes | 146 |
| 44 | It’s That Part† | Jesse Lee Hammonds | 28 | 10 | 2025–2026 |  | yes | 0 |
| 46 | No Lie with Brian Tyler Cohen | Brian Tyler Cohen | 353 | 256 | 2020–2026 | L | yes | 262 |
| 47 | The Playbook Podcast | POLITICO | 1,318 | 163 | 2016–2026 | L | no | 349 |
| 48 | John Solomon Reports | John Solomon | 811 | 640 | 2020–2026 | R | yes | 189 |
| 50 | The Daily Punch | Punchbowl News | 737 | 146 | 2021–2026 | L | yes | 309 |
| 51 | The Remnant with Jonah Goldberg | The Dispatch | 949 | 1,155 | 2017–2026 | R | yes | 115 |
| 52 | The Illegal News with Sarah Longwell | The Bulwark | 120 | 108 | 2024–2026 | L | yes | 176 |
| 53 | Next Up with Mark Halperin | MK Media | 123 | 148 | 2025–2026 | R | yes | 189 |
| 54 | Face the Nation with Margaret Brennan | CBS News | 505 | 395 | 2018–2026 | L | yes | 252 |
| 55 | The Young Turks | TYT Network | 1,068 | 1,141 | 2017–2026 | L | yes | 194 |
| 56 | The 11th Hour with Ali Velshi | MS NOW, Ali Velshi | 99 | 75 | 2019–2026 | L | no | 226 |
| 57 | Talkin‘ Politics & Religion Without Killin‘ Each Other | Scan Media, LLC | 439 | 464 | 2020–2026 | L | yes | 113 |
| 58 | Louder with Crowder | Louder with Crowder | 893 | 1,174 | 2014–2026 | R | yes | 84 |
| 60 | The Ben Ferguson Podcast | iHeartPodcasts | 821 | 505 | 2020–2026 | R | yes | 216 |
| 61 | Pod Force One | New York Post | 110 | 68 | 2025–2026 | R | yes | 174 |
| 62 | The David Pakman Show | David Pakman | 488 | 509 | 2022–2026 | L | yes | 258 |
| 63 | Brian Kilmeade Show | FOX News Podcasts | 797 | 1,073 | 2020–2026 | R | yes | 176 |
| 64 | Stay Tuned with Preet | Preet Bharara | 918 | 642 | 2017–2026 | L | yes | 157 |
| 65 | TRUE ANON TRUTH FEED | TrueAnon | 21 | 26 | 2024–2026 | L |  | 73 |
| 66 | Making the Argument with Nick Freitas | Nick Freitas | 240 | 321 | 2024–2026 | R | yes | 96 |
| 67 | The Warning with Steve Schmidt | iHeartPodcasts | 446 | 183 | 2023–2026 | L | yes | 155 |
| 68 | Politics Chat | Heather Cox Richardson | 48 | 38 | 2026–2026 | L | yes | 231 |
| 69 | Main Justice | MS NOW, Andrew Weissmann, Ma | 232 | 202 | 2023–2026 | L | yes | 188 |
| 70 | Liberty Lockdown | Clint Russell | 533 | 612 | 2020–2026 | R | yes | 88 |
| 71 | The Majority Report with Sam Seder* | Sam Seder | 25 | 33 | 2026–2026 |  | yes | 163 |
| 73 | The Beat with Ari Melber | Ari Melber, MS NOW | 597 | 440 | 2018–2026 | L | yes | 201 |
| 75 | Renegade Talk Radio | Renegade Talk Radio | 2,476 | 2,580 | 2009–2026 | R | no | 110 |
| 76 | The PoliticsGirl Podcast | Meidas Media Network, Leigh  | 249 | 207 | 2021–2026 | L | yes | 177 |
| 77 | Hacks On Tap | Vox Media | 383 | 374 | 2019–2026 | L | yes | 248 |
| 78 | THE DAILY BLAST with Greg Sargent | Greg Sargent | 335 | 133 | 2024–2026 | L | yes | 295 |
| 81 | Politics War Room with James Carville & Al Hunt | Politicon | 379 | 418 | 2019–2026 | L | yes | 194 |
| 82 | Newt's World | Gingrich 360 | 830 | 568 | 2019–2026 | R | yes | 96 |
| 83 | The 47 Morning Update with Ben Ferguson | Premiere Networks | 664 | 414 | 2020–2026 | R | yes | 248 |
| 84 | Know Your Enemy | Matthew Sitman | 275 | 240 | 2019–2026 | L | yes | 121 |
| 86 | Sidebar with Katie Phang | Katie Phang | 94 | 38 | 2026–2026 | L | yes | 203 |
| 87 | The Benny Show | Benny Johnson | 595 | 824 | 2021–2026 | R | yes | 218 |
| 88 | The Hugh Hewitt Show: Highly Concentrated | Salem Podcast Network | 73 | 86 | 2026–2026 | R | yes | 138 |
| 89 | Straight White American Jesus | Bradley Onishi + Daniel Mill | 777 | 367 | 2018–2026 | L | yes | 96 |
| 90 | The Intercept Briefing | The Intercept | 404 | 381 | 2017–2026 | L | no | 152 |
| 91 | The Loren Piretra Show† | Loren Piretra | 43 | 10 | 2026–2026 |  | yes | 59 |
| 92 | Tangle | Isaac Saul | 637 | 400 | 2020–2026 | L | yes | 211 |
| 93 | GZERO World with Ian Bremmer | GZERO Media | 437 | 199 | 2017–2026 | L | yes | 143 |
| 94 | Human Events Daily with Jack Posobiec | Human Events with Jack Posob | 336 | 246 | 2021–2026 | R |  | 125 |
| 95 | The Michael Berry Show | KTRH (KTRH-AM) | 315 | 194 | 2024–2026 | R | yes | 71 |
| 96 | The Chris Cuomo Project | Chris Cuomo | 439 | 305 | 2022–2026 | L | yes | 115 |
| 97 | The Skepticrat | Puzzle in a Thunderstorm, LL | 273 | 198 | 2015–2026 | L | yes | 122 |
| 98 | The Foreign Affairs Interview | Foreign Affairs Magazine | 100 | 87 | 2024–2026 | L | yes | 148 |
| 99 | The Brian Lehrer Show | WNYC | 320 | 161 | 2024–2026 | L | yes | 147 |
| 100 | Glenn Diesen - Greater Eurasia Podcast | Glenn Diesen | 230 | 176 | 2024–2026 | L | yes | 128 |
| 101 | Fox Across America w/ Jimmy Failla | FOX News Podcasts | 788 | 1,199 | 2020–2026 | R | yes | 161 |
| 102 | Rev Left Radio | Revolutionary Left Radio | 674 | 969 | 2017–2026 | L | yes | 47 |
| 103 | The Graham Allen Show | Graham Allen | 810 | 784 | 2019–2026 | R | yes | 88 |
| 104 | Going Rogue with Lara Logan | Lara Logan | 88 | 130 | 2025–2026 | R | yes | 102 |
| 107 | RealClearPolitics Podcast | RealClearPolitics, Carl Cann | 317 | 264 | 2024–2026 | R | yes | 231 |
| 108 | The Tom Woods Show | Tom Woods | 538 | 403 | 2021–2026 | R | yes | 91 |
| 109 | The Joy Reid Show | Joy-Ann Reid | 144 | 248 | 2025–2026 | L | yes | 149 |
| 111 | Nobody Knows Anything* | Narrowcast Media | 26 | 26 | 2026–2026 |  |  | 159 |
| 112 | The Smerconish Podcast | SiriusXM | 679 | 278 | 2021–2026 | L | yes | 145 |
| 113 | The Source with Kaitlan Collins | CNN Podcasts | 383 | 288 | 2023–2026 | L | yes | 252 |
| 114 | American Prestige | Daniel Bessner & Derek Davis | 76 | 51 | 2026–2026 | L | yes | 115 |
| 115 | System Update with Glenn Greenwald | Rumble | 405 | 521 | 2023–2026 | R | yes | 210 |
| 116 | Volts | David Roberts | 432 | 426 | 2020–2026 | L | yes | 65 |
| 118 | Wendy Bell Radio Podcast | Wendy Bell | 109 | 69 | 2026–2026 | R | yes | 192 |
| 119 | Uncommon Knowledge | Hoover Institution | 100 | 104 | 2022–2026 | R | yes | 101 |
| 120 | Brian Tyler Cohen* | Brian Tyler Cohen Network | 123 | 27 | 2026–2026 |  | yes | 283 |
| 121 | Checks and Balance from The Economist† | The Economist | 19 | 11 | 2023–2026 |  | yes | 289 |
| 123 | Trump's Terms | NPR | 384 | 58 | 2023–2026 | L | yes | 350 |
| 124 | The Mel K Show* | Mel K | 62 | 90 | 2026–2026 |  | yes | 149 |
| 125 | Defending Democracy with Marc Elias | Democracy Docket | 240 | 153 | 2022–2026 | L | yes | 298 |
| 126 | The Coffee Klatch with Robert Reich | Robert Reich | 359 | 124 | 2021–2026 | L | yes | 206 |
| 127 | The Breakdown | Alberto fo sho | 314 | 216 | 2024–2026 | L | no | 208 |
| 128 | Lectures in History | C-SPAN | 416 | 464 | 2014–2026 | L | no | 91 |
| 130 | Opening Arguments | Opening Arguments Media LLC | 1,116 | 1,253 | 2016–2026 | L | yes | 128 |
| 131 | What Really Matters with Walter Russell Mead | Tablet Magazine | 136 | 78 | 2023–2026 | R | yes | 156 |
| 134 | The Pete Quiñones Show | Peter R Quiñones | 181 | 829 | 2025–2026 | R | yes | 53 |
| 135 | Mission Implausible | iHeartPodcasts | 124 | 94 | 2024–2026 | L | yes | 96 |
| 136 | Dad Saves America | John Papola | 216 | 339 | 2022–2026 | R | yes | 47 |
| 137 | 10 Minute Drill | Matt Whitlock | 183 | 39 | 2025–2026 | R | yes | 309 |
| 138 | American Thought Leaders | The Epoch Times | 172 | 150 | 2025–2026 | R | yes | 58 |
| 139 | The Tony Kinnett Cast | The Daily Signal | 347 | 347 | 2023–2026 | R | yes | 181 |
| 140 | Head in the Office | HITO Media LLC | 332 | 474 | 2021–2026 | L | yes | 180 |
| 141 | Slate News | Slate Podcasts | 1,930 | 1,453 | 2006–2026 | L | no | 116 |
| 142 | Conversations with Bill Kristol | Bill Kristol | 322 | 387 | 2014–2026 | L | yes | 179 |
| 143 | The Al Franken Podcast | The Al Franken Podcast | 391 | 354 | 2019–2026 | L | yes | 159 |
| 144 | American Conservative University | American Conservative Univer | 275 | 214 | 2024–2026 | R | no | 113 |
| 145 | HasanAbi† | HasanAbi | 32 | 16 | 2026–2026 |  | yes | 129 |
| 146 | What the Hell Is Going On | AEI Podcasts | 200 | 180 | 2023–2026 | R | yes | 190 |
| 147 | Paul Krugman Podcast | Paul Krugman | 76 | 31 | 2025–2026 | L | yes | 133 |
| 148 | Deseret Voices† | Deseret News | 41 | 23 | 2025–2026 |  | no | 60 |
| 149 | UNGOVERNED | Shawn Farash | 76 | 78 | 2026–2026 | R | yes | 153 |
| 150 | KQED's Forum | KQED | 795 | 658 | 2020–2026 | L | no | 89 |
| 151 | The Mary Trump Podcast | Mary Trump Media | 94 | 30 | 2024–2026 | L |  | 166 |
| 152 | The Damage Report with John Iadarola | TYT Network | 985 | 877 | 2018–2026 | L | yes | 152 |
| 153 | Wired In† | The Daily Wire | 18 | 19 | 2026–2026 |  | yes | 213 |
| 155 | Jacobin Radio | Jacobin | 1,182 | 1,349 | 2017–2026 | L | no | 111 |
| 156 | Owen Report | Owen Shroyer | 103 | 319 | 2025–2026 | R | yes | 164 |
| 157 | Bad Faith | Briahna Joy Gray | 100 | 87 | 2025–2026 | L | yes | 111 |
| 160 | Perino on Politics | FOX News Podcasts | 163 | 78 | 2023–2026 | R | yes | 274 |
| 161 | MOATS with George Galloway | Molucca Media Ltd | 483 | 740 | 2021–2026 | L | yes | 139 |
| 162 | The Bob Cesca Show / Politics Podcast, News & Commentary | Bob Cesca | 876 | 1,001 | 2019–2026 | L | yes | 159 |
| 163 | The President’s Inbox | Council on Foreign Relations | 99 | 59 | 2024–2026 | L | yes | 164 |
| 164 | The Dig | Daniel Denvir | 547 | 873 | 2015–2026 | L | yes | 100 |
| 165 | Unf*cking The Republic | UNFTR Media | 422 | 369 | 2020–2026 | L | yes | 98 |
| 166 | Washington Today* | C-SPAN | 50 | 41 | 2026–2026 |  | no | 327 |
| 167 | War on the Rocks | Ryan Evans | 337 | 221 | 2013–2026 | L | yes | 79 |
| 168 | Federalist Radio Hour | Radio America | 557 | 459 | 2022–2026 | R | no | 134 |
| 169 | The Conversation with Dasha Burns | POLITICO | 546 | 316 | 2016–2026 | L | no | 249 |
| 170 | Useful Idiots with Katie Halper and Aaron Maté | Useful Idiots, LLC | 413 | 453 | 2019–2026 | L | yes | 149 |
| 171 | Amarica's Constitution | Akhil Reed Amar | 286 | 417 | 2021–2026 | L | yes | 197 |
| 172 | ChinaTalk | Jordan Schneider | 535 | 525 | 2017–2026 | L | yes | 63 |
| 173 | The Victor Davis Hanson Show | Victor Davis Hanson and Jack | 532 | 573 | 2021–2025 | R | yes | 131 |
| 174 | Politically Georgia | The Atlanta Journal-Constitu | 757 | 462 | 2018–2026 | L | yes | 278 |
| 175 | Hawk Podcasts | Hawk | 153 | 86 | 2024–2026 | L | yes | 223 |
| 176 | Bad Hasbara - The World's Most Moral Podcast | Matt Lieb | 202 | 276 | 2023–2026 | L | yes | 73 |
| 177 | The Political Orphanage | Andrew Heaton | 580 | 588 | 2018–2026 | R | yes | 89 |
| 178 | Maiden Mother Matriarch with Louise Perry | Louise Perry | 270 | 226 | 2023–2026 | R | yes | 34 |
| 179 | SGT Report’s The Propaganda Antidote* | SGT Report | 77 | 71 | 2026–2026 |  | yes | 111 |
| 180 | Make It Make Sense† | The Washington Post | 42 | 24 | 2026–2026 |  | no | 133 |
| 181 | Mark Kaye Saves The Republic | Kayetriot Media | 265 | 302 | 2024–2026 | R | yes | 242 |
| 182 | Scott Horton Show - Just the Interviews | Scott Horton | 509 | 367 | 2021–2026 | L | yes | 113 |
| 183 | The Peter McCormack Show | Peter McCormack | 944 | 1,355 | 2017–2026 | R | yes | 44 |
| 184 | Firing Line with Margaret Hoover | Firing Line With Margaret Ho | 211 | 160 | 2021–2026 | L | yes | 181 |
| 185 | Life, Liberty & Levin Podcast | FOX News Podcasts | 113 | 68 | 2025–2026 | R | yes | 235 |
| 186 | Antiwar News With Dave DeCamp | Dave DeCamp | 283 | 142 | 2022–2026 | L |  | 229 |
| 187 | The Globalist* | Monocle | 80 | 81 | 2026–2026 |  | yes | 93 |
| 188 | The Harry Sisson Show | Harry Sisson | 258 | 51 | 2025–2026 | L | yes | 302 |
| 189 | It's A Numbers Game with Ryan Girdusky | iHeartPodcasts | 162 | 124 | 2025–2026 | R | yes | 215 |
| 190 | The Kyle Seraphin Show | Kyle Seraphin | 449 | 610 | 2022–2026 | R | yes | 120 |
| 191 | POLITICO Energy | POLITICO | 868 | 138 | 2019–2026 | L | no | 167 |
| 192 | Autocracy in America† | The Atlantic | 29 | 15 | 2024–2026 |  | yes | 152 |
| 193 | Stay Free with Russell Brand | Stay Free with Russell Brand | 436 | 433 | 2022–2026 | R | yes | 88 |
| 194 | THE WEEKEND SHOW | Evergreen Podcasts | 233 | 270 | 2022–2026 | L | yes | 181 |
| 195 | The Kyle Kulinski Show | Kyle Kulinski | 263 | 362 | 2022–2026 | L |  | 207 |
| 197 | The Dishcast with Andrew Sullivan | Andrew Sullivan | 274 | 286 | 2020–2026 | R | yes | 78 |
| 198 | Majority 54 | Majority 54 | 380 | 347 | 2017–2026 | L | yes | 145 |
| 199 | Timcast News | Timcast Media | 1,011 | 1,205 | 2018–2026 | R | yes | 193 |
| 200 | Hope For America with Heather Delaney Reese | Heather Delaney Reese | 105 | 32 | 2026–2026 | L |  | 262 |
| 201 | Best of the Left - Leftist Perspectives on Progressive Politics, News, Culture, Economics and Democracy | Best of the Left: Perspectiv | 497 | 762 | 2022–2026 | L | yes | 134 |
| 202 | The Inside Story Podcast | Al Jazeera | 227 | 93 | 2025–2026 | L | no | 181 |
| 204 | The Great America Show | Fawcett Strategies | 615 | 325 | 2021–2026 | R | no | 260 |
| 205 | No Gods No Mayors | The Office of the Mayor | 98 | 70 | 2024–2026 | L | yes | 90 |
| 206 | Cleanup on Aisle 45 with AG & Harry Dunn | MSW Media | 287 | 274 | 2021–2026 | L | yes | 190 |
| 207 | Rational Security | The Lawfare Institute | 570 | 591 | 2015–2026 | L | no | 161 |
| 209 | The Arena | Father Josiah Trenham | 874 | 369 | 2012–2026 | R | yes | 13 |
| 210 | The Reason Roundtable | The Reason Roundtable | 100 | 113 | 2024–2026 | R | yes | 159 |
| 211 | Batya! | Batya | 51 | 33 | 2025–2026 | R | yes | 232 |
| 212 | The Breitbart News Daily Podcast | SiriusXM | 598 | 511 | 2021–2026 | R | no | 157 |
| 213 | The Keith Edwards Show | Keith Edwards | 294 | 55 | 2025–2026 | L | yes | 221 |
| 214 | The Nation Podcasts | The Nation Magazine | 803 | 587 | 2015–2026 | L | no | 147 |
| 215 | Week In Review | KUOW News and Information | 203 | 169 | 2022–2026 | L | yes | 107 |
| 217 | Hood Politics with Prop | Cool Zone Media and iHeartPo | 356 | 276 | 2020–2026 | L | yes | 57 |
| 219 | Timothy Gordon Rules for Retrogrades Podcast | Timothy Gordon | 416 | 467 | 2019–2026 | R |  | 31 |
| 221 | The Drill Down with Peter Schweizer | Government Accountability In | 267 | 119 | 2021–2026 | R | yes | 197 |
| 222 | What's The Plan? with Leah Greenberg and Ezra Levin | Indivisible | 26 | 26 | 2026–2026 | L | yes | 188 |
| 223 | Extremely American† | NPR | 25 | 15 | 2021–2025 |  | yes | 86 |
| 224 | Week In Review | FITSNews | 215 | 172 | 2022–2026 | R |  | 103 |
| 225 | Brad vs Everyone | iHeartPodcasts | 295 | 178 | 2024–2026 | R | yes | 114 |
| 226 | Essential Minnesota† | Minnesota Star Tribune | 30 | 11 | 2026–2026 |  | yes | 140 |
| 229 | Seattle Nice | David Hyde, Erica Barnett, a | 198 | 118 | 2021–2026 | L |  | 81 |
| 230 | The Lars Larson Show Pacific Northwest Podcast* | Lars Larson | 44 | 72 | 2026–2026 |  | yes | 156 |
| 231 | Destiny | Destiny | 191 | 298 | 2025–2026 | L | yes | 141 |
| 232 | The Nerd Reich† | nerdreich | 20 | 16 | 2025–2026 |  | yes | 116 |
| 233 | The Buck Sexton Show | Premiere Networks | 1,249 | 1,400 | 2017–2026 | R | yes | 160 |
| 234 | The Reason Interview With Nick Gillespie | The Reason Interview With Ni | 99 | 95 | 2024–2026 | R | yes | 101 |
| 235 | Get Angry† | Bryan Andrews, ItsLuke | 18 | 11 | 2026–2026 |  | yes | 96 |
| 236 | The Josh Hammer Show | The Josh Hammer Show | 76 | 57 | 2026–2026 | R | yes | 163 |
| 237 | Mea Culpa | Michael Cohen | 272 | 343 | 2020–2026 | L | yes | 202 |
| 238 | The Moment with Jorge Ramos and Paola Ramos | iHeartPodcasts | 43 | 43 | 2025–2026 | L | yes | 106 |
| 240 | Mueller, She Wrote - When the Facts Were Fresh† | MSW Media | 51 | 53 | 2018–2018 |  | yes | 174 |
| 241 | I Don't Speak German | IDSG | 238 | 290 | 2019–2026 | L | yes | 50 |
| 242 | Candace Owens† | The Daily Wire | 350 | 177 | 2021–2024 |  | yes | 14 |
| 244 | Countdown with Keith Olbermann | iHeartPodcasts | 457 | 433 | 2022–2026 | L | yes | 152 |
| 245 | The Mona Charen Show | The Bulwark | 287 | 280 | 2019–2026 | L | yes | 205 |
| 246 | Russian Roulette | Center for Strategic and Int | 163 | 120 | 2018–2026 | L | no | 110 |
| 248 | Uncertain with DL Hughley† | DL Hughley | 26 | 19 | 2026–2026 |  | yes | 103 |
| 249 | Politix | Politix | 134 | 87 | 2023–2026 | L | yes | 292 |
| 250 | The TPM Show with Josh and Kate | Talking Points Memo | 473 | 433 | 2018–2026 | L | yes | 169 |

## Appendix B. Chart shows not in the corpus

**Excluded by rule** (31 of the 250 chart shows; `data/output/exclusions.csv`):

| rank | show | rule | evidence |
|---|---|---|---|
| 14 | TRIGGERnometry | R5_non_us_politics | curated list: TRIGGERnometry — UK-produced (Kisin/Foster, London); judgment call: culture- |
| 45 | The Dispatch Podcast | R3_feed_unparseable | XML parse error: not well-formed (invalid token): line 5158, column 441 |
| 49 | The Five | R6_insufficient_catalog | 2.31 h available < 10.0 h threshold |
| 59 | The Rest Is Politics | R5_non_us_politics | curated list: The Rest Is Politics (Goalhanger) — UK politics |
| 72 | Sources & Methods | R6_insufficient_catalog | 2.6 h available < 10.0 h threshold |
| 74 | America First with Nicholas J. Fuentes Repost | R2_feed_unreachable | feed.podbean.com/wanghaf/feed.xml now returns HTTP 404 (entire show removed from Podbean,  |
| 79 | This Morning With Gordon Deal | R6_insufficient_catalog | 0.0 h available < 10.0 h threshold |
| 80 | Fox News Sunday Audio | R6_insufficient_catalog | 2.63 h available < 10.0 h threshold |
| 85 | The Rest Is Politics: Leading | R5_non_us_politics | curated list: The Rest Is Politics: Leading (Goalhanger) — UK-frame interviews |
| 105 | NARITAS | R5_non_us_politics | rss_language='fr' (non-English feed) |
| 106 | Tocqueville Road Trip from The Economist | R6_insufficient_catalog | 1.61 h available < 10.0 h threshold |
| 110 | PBS News Hour - Brooks and Capehart | R6_insufficient_catalog | 4.3 h available < 10.0 h threshold |
| 117 | Common Sense with Dan Carlin | R6_insufficient_catalog | 8.22 h available < 10.0 h threshold |
| 122 | El hilo | R5_non_us_politics | rss_language='es' (non-English feed) |
| 129 | Americast | R5_non_us_politics | publisher matches blocklist term 'bbc' (BBC News) |
| 132 | nan | R1_no_feed_url | no feedUrl in Apple lookup |
| 133 | SPN LIVE - The Stew Peters Network | R2_feed_unreachable | HTTP 404 |
| 154 | Super Tuesdays with Chris Cillizza and Chuck Todd | R6_insufficient_catalog | 2.48 h available < 10.0 h threshold |
| 158 | 不明白播客 | R5_non_us_politics | rss_language='zh' (non-English feed) |
| 159 | The Sunday Story | R6_insufficient_catalog | 1.85 h available < 10.0 h threshold |
| 196 | Novara Media | R5_non_us_politics | publisher matches blocklist term 'novara media' (Novara Media) |
| 203 | Politics Weekly America | R5_non_us_politics | publisher matches blocklist term 'the guardian' (The Guardian) |
| 208 | DW News Brief | R6_insufficient_catalog | 0.13 h available < 10.0 h threshold |
| 216 | Frontline: The War in Ukraine and Global Security | R5_non_us_politics | publisher matches blocklist term 'times radio' (Times Radio) |
| 218 | nan | R1_no_feed_url | no feedUrl in Apple lookup |
| 220 | The Rest Is Money | R5_non_us_politics | publisher matches blocklist term 'goalhanger' (Goalhanger) |
| 227 | Es la Mañana de Federico | R5_non_us_politics | rss_language='es-ES' (non-English feed) |
| 228 | Accountability | R6_insufficient_catalog | 1.24 h available < 10.0 h threshold |
| 239 | On Couch with Kelly & Clint | R6_insufficient_catalog | 2.85 h available < 10.0 h threshold |
| 243 | RONZHEIMER. | R5_non_us_politics | rss_language='de' (non-English feed) |
| 247 | Azad / آزاد | R5_non_us_politics | rss_language='fa' (non-English feed) |

The fifteen shows set aside by decision on 14 August 2026 were reinstated on 23 November 2026 (Appendix A, †; `data/output/frame_decisions.md`).
