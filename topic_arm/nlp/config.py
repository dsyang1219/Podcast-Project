"""Stage 1/2 preprocessing parameters (and the documented Stage 3 convention).

Every knob that affects token identity or topic-model inputs lives here so a
run is fully described by (this file, the transcript set, the CLI flags
passed to run_chunks). See nlp/clean.py and nlp/chunk.py for what each stage
does with these.
"""
from __future__ import annotations

# ---------------------------------------------------------- boilerplate ----
BOILERPLATE_NGRAM_SIZE = 8
BOILERPLATE_MIN_SHOWS = 4        # n-gram must recur across >= this many shows
BOILERPLATE_COVERAGE = 0.5       # drop sentence if >= this fraction of its
                                  # word n-grams are flagged boilerplate

# Per-show boilerplate token/sentence loss above this fraction gets flagged —
# uneven stripping across shows breaks the equal-hours-per-show density
# assumption the sampling design relies on.
BOILERPLATE_LOSS_FLAG_THRESHOLD = 0.15

# ----------------------------------------------------------- tokenizing ----
# Field convention is drop <2, keep 2+ (a 3-char floor drops real political
# content: gop, fbi, cia, doj, ice, tax, gun, war, oil, law, eu, un).
DEFAULT_MIN_TOKEN_LEN = 2

# Pure interjection / backchannel filler — near-zero topic signal in any
# context. Deliberately does NOT include "know"/"think"/"mean"/"right"/"left":
# those are borderline and left to Stage 3 document-frequency pruning rather
# than a hardcoded drop, since they carry real political-discourse meaning in
# some contexts (e.g. "right wing") and DF pruning is the more principled
# place to catch the ones that turn out to be pure filler in this corpus.
FILLER_WORDS = {
    "um", "uh", "umm", "uhh", "erm", "uh-huh", "mm-hmm", "mhm",
    "hmm", "huh", "yeah", "yep", "yup", "nah", "okay", "ok", "alright",
    "like",  # 93-98% DF, pure discourse filler, no political sense
}

# Known-junk 2-char tokens once the length floor drops to 2. Kept
# deliberately tiny and reviewed by hand (see the one-time survivor print in
# run_chunks.py) — do not add abbreviations/acronyms here.
TWO_CHAR_STOPLIST = {"ha", "eh", "hm", "oh"}

# ------------------------------------------------------------ lemmatize ----
# Schofield & Mimno (2016) and downstream work find lemmatization often
# REDUCES topic interpretability, and it's corpus-dependent rather than a
# fixed rule — hence a toggle, defaulted to the safer (off) side of the
# evidence, not hardcoded either way. Run the whole pipeline both ways and
# compare topic coherence + human-read interpretability before picking one
# for the final analysis.
DEFAULT_LEMMATIZE = False

# --------------------------------------------------- stage 3 (documented) ----
# Not yet implemented in this repo, but the threshold is fixed here so it
# travels with the rest of the preprocessing decisions. Grimmer & Stewart
# 2013 / Maier et al.: prune extreme document frequency, not a low ceiling —
# 0.99 keeps a term appearing in most episodes (e.g. "trump" at ~57% DF)
# while still dropping near-universal terms. Treat as a tunable robustness
# parameter (report results are stable across nearby values), not a fixed
# constant to bury in model code.
STAGE3_DF_NO_ABOVE = 0.99

# ------------------------------------------------------ embedding (arm 2) ----
# Prefix-free 768d model by default: E5/BGE require a passage/document prefix
# applied uniformly to every input or embeddings land off-distribution — an
# easy silent error. Defaulting to a model with no prefix convention avoids
# that failure mode entirely; swap to an E5/BGE model + its prefix for a
# robustness check, never leave EMBED_PREFIX empty for those families.
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBED_PREFIX = ""

# Measured on this machine's GPU (NVIDIA GB10): batch_size=96 in fp16 beat
# larger batches (256/512/1024 were all slower here, likely padding/memory-
# bandwidth bound on variable-length sentence batches) -- ~2250 sentences/sec
# vs ~40/sec on CPU. fp16 is compute-only; embeddings are still stored as
# float32 (cast after encode), so this doesn't change output precision.
EMBED_BATCH_SIZE = 96
EMBED_FP16 = True  # only applied when running on CUDA; ignored on CPU
