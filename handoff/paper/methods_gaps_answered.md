# Methods gaps — answered from the pipeline code and run records (3 Sept 2026)

Sources: step3_transcribe/transcribe.py, run_ladder.sh, step1_frame/{chart,filter,config}.py, data/output/exclusions.csv, build_scoring_chunks.py, nlp/adspan_*.py + data/output/adspan/*.json, nlp/run_chunks.py / adspan_rechunk.py, nlp/lda_*.py + data/output/lda_kcurve_report.json + adspan/phase_c_report.json, extract_hosts.py, nlp/match_guests_dime.py, data/output/host_dime_lookup_v2.csv, ideology_targets_204.csv, lda_ideology_204_report.json, nlp/embed_ideology_nonlinear.py.

## 1. Transcription
- Engine: **faster-whisper** (CTranslate2), `BatchedInferencePipeline` — NOT WhisperX; no diarization (the docstring says so explicitly; diarization is listed as future work on the archived audio).
- Model: **deepdml/faster-whisper-large-v3-turbo-ct2** (the CT2 conversion of Whisper large-v3-turbo). The argparse default is `large-v3`, but every run went through run_ladder.sh, which sets the turbo-ct2 model.
- Settings: compute_type **float16**, batch size **64** (validated against batch 16: 0.00% WER difference on the check episodes), language fixed to **"en"** (no auto-detect), VAD = faster-whisper's built-in Silero VAD inside the batched pipeline, one transcriber worker per GPU (~10 GB VRAM at batch 64), throughput ~100–110× real time.
- Hardware: two **NVIDIA GB10** machines (rma1, rma2; DGX-Spark class), one transcriber each; the ladder throttles on GPU temperature.
- Output: word-timestamped JSON + plain text per episode, keyed by md5(audio_url)[:16].

## 2. Show selection and sampling frame
- Frame: the **Apple Podcasts US "Politics" subcategory chart** pulled 2026-07-13, ranks 1–250 (250 is as deep as Apple serves); 248 of 250 carry primary_genre = Politics. The chart file is frozen on disk (raw_chart_20260713.csv).
- Inclusion filter (step1_frame/filter.py), rules applied in fixed order, first failure recorded with evidence: R1 no RSS feed URL (2), R2 feed unreachable (2), R3 feed unparseable (1), R4 no audio enclosures (0), R5 non-US-politics — publisher blocklist, title/description regex, non-English feed language, curated manual list, minus exceptions (14), R6 insufficient catalogue < 10 hours available (12). 31 excluded → **219** shows.
- 219 → **204**: 15 shows were not in the H25 transcript sample (the stratified ~7,600-episode sample used for the topic/ideology arm) and were kept out of the corpus by a recorded decision on 2026-08-14 (corpus_shows.csv, `excluded_reason`). Say this plainly; it is the step a reader will ask about.
- Consequence for scope: the population is "political podcasts that attained and held chart presence in the Politics subcategory," which is mid-tail; the largest shows are filed under other categories and enter only through the out-of-frame test.

## 3. Ad cleaning
Two different arms, and the paper must say which applies where.
- **Register/ideology scoring arm (1.38M chunks):** NO removal. Following Much et al., the whole episode is scored; each 750-char chunk carries a `has_ad_marker` flag (regex: promo code, use code, dot com slash, percent off, sponsored by, brought to you by, this episode is sponsored, free shipping, free trial, go to X.com) so every analysis can be re-run without flagged chunks. Politically-branded sponsors are rare and the register measures are computed on all chunks.
- **Topic/LDA arm (H25 sample, 7,626 episodes):** sentence-level **ad/meta span detection and excision** before segmentation. Detector: gpt-4o-mini-2024-07-18 via the OpenAI Batch API, one pinned prompt (nlp/adspan_detect.py), windows of sentences; 131,463 windows, 51,675 raw spans merged to 35,500; **5.9% of words removed** (53.04M → 49.89M); any malformed/missing response excises nothing (failure direction = leave ads in). Cost $14.52.
- **Validation (Phase A gate, before any excision):** 26 windows / 886 sentences / 538 content sentences hand-annotated at sentence level into ad_sponsor / meta_boilerplate / content, stratified to contain ads. Gate: ad-sponsor precision ≥ .90 AND content false-positive rate ≤ 2%. gpt-4o-mini: content FP rate 0.7% (4 sentences, 0 substantive), 130 false negatives; gate passed. gpt-4.1-nano: FP 0.2%, 279 FNs; passed but under-detects. **Caveat to state:** the gold set was annotated by an LLM agent (Claude) reading the full windows, not by a human; it is a different model from the detector, every window is stored with full text for audit, and a human re-annotation is on the pre-submission list.
- A cross-show boilerplate n-gram filter (n-grams recurring in ≥ 4 shows) is additionally applied at segmentation in the LDA arm and was rebuilt on the cleaned corpus.

## 4. Segmentation
- **Register/ideology arm:** `split_750` — exhaustive, non-overlapping chunks of ≤ **750 characters** (Much et al. footnote 19), preferring sentence boundaries (split on [.!?]+space); a sentence longer than 750 chars is hard-split so no text is dropped; no minimum length at build time (0.2% of chunks are under 120 chars). Register scoring then drops passages under 40 words. Result: 1.38M passages.
- **LDA arm:** ~**500 content-word passages** (target 500, minimum stub 300, min token length 2, no lemmatization) built from sentencized, boilerplate-filtered, ad-excised text with the same chunker as the contaminated run. 40,242 passages before excision; **37,942** after — the unit the STM/topic and passage-level effect estimates use.

## 5. K = 75 and what else was tried
- Pilot sweep (nlp/sweep_k.py, sklearn LDA, UMass coherence + perplexity, K = 5–20+) and a full sweep at scale with tomotopy (TermWeight.ONE, seed 0, 500 Gibbs iterations, c_v coherence @ top-10): **K ∈ {20, 30, 50, 75, 100, 150}**.
- Ideology recoverability (ridge, LOSO-CV R², N = 119 with one documented exclusion) by K: 20 → .21, 30 → .25, 50 → .33, **75 → .36**, 100 → .41, 150 → .38; permutation null ≈ −0.02 throughout. Rule recorded: K = 75 is the first K in the plateau (R² stops rising meaningfully through 150), so it was preferred over the noisy peak at 100.
- **Honesty point for the text:** that choice used the ideology outcome. Phase C re-derived K on the cleaned corpus with an outcome-independent rule (smallest K within one SE of peak c_v): peak c_v is at K = 75 (.673, SE .013) but the rule selects **K = 30**, where LOSO R² is .27 versus .45 at K = 75. The paper should report K = 75 as the primary representation with the coherence-selected K = 30 result alongside, and say that the two criteria disagree.

## 6. DIME matching and the three target sets
- Hosts: extracted from RSS (itunes:author → itunes:owner name → RSS author → parsed from title), classified person vs. organization vs. self-reference; multi-host strings split. Matched to DIME 1979–2024 recipients and contributors (bonica.cid) by surname + first-word streaming match; buckets 0 / 1 / 2+ candidates; 2+ resolved by DIME attributes (state, party, seat; occupation/employer) against show/host context, then a logged manual web-search stage (IRB-approved, every query logged); merged multi-record donors; confidence recorded per host (host_dime_lookup_v2.csv: final_source, confidence, matched_bonica_cids).
- Guests: same three-stage cascade on 109,653 guest appearances (single_recipient 16,043; single_donor 14,419; description-resolved 25,689; manual_search 9,346; ambiguous residual 11,182; dropped as too common 5,115; unmatched 27,859).
- Target sets (ideology_targets_204.csv; lda_ideology_204_report.json), all against the same K = 75 CLR topic vectors:
  - **host_clean, n = 84** — shows with full host coverage; target = mean host cfscore; LOSO R² = .36, r = .60.
  - **extended_all, n = 204** — host cfscore where coverage is full, else guest-weighted mean cfscore; LOSO R² = .43, r = .66 (the headline).
  - well_supported, n = 188 — extended_all with ≥ 20 DIME-matched guests for guest-primary rows; R² = .43.
  - **guest_uniform, n = 202** — guest-only ideology for every show with a guest score; R² = .61, r = .79.
  - extended_weighted, n = 204 — inverse-variance-weighted ridge; R² = .43.
  Host coverage in the 204: full 84, partial 36, none 77, no host listed 7.

## 7. The regression behind R² ≈ 0.46
- Estimator: **ridge regression** on CLR-transformed show-level topic proportions (K = 75), StandardScaler → Ridge in a sklearn Pipeline; alpha chosen by nested CV inside each training fold (small fixed grid; best_alpha = 100 in the synthesis run).
- Validation: **leave-one-show-out** CV; scaler and alpha fit on the 203 training shows only. **All reported R² are held-out (LOSO-CV)**; in-sample R² is computed only to show the overfit gap and is never reported. Permutation null (shuffled targets) ≈ −0.01 to −0.03.
- The figures: 0.43 (extended_all, n = 204, contaminated K = 75 fit); 0.452 at K = 75 on the ad-cleaned corpus; 0.478 in the topic-PCA write-up (same harness, N = 119 host-matched); 0.27 at the coherence-selected K = 30. "≈ 0.46" should be replaced by whichever of these the sentence refers to, with n and corpus stated.
