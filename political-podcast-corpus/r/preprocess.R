#!/usr/bin/env Rscript
# Toggleable quanteda preprocessing pipeline for the ~500-word podcast
# passages (chunk-level). Operates on the RAW passage text (chunks_500_
# nolemma.csv's `text` column -- boilerplate-sentence-stripped but
# otherwise unprocessed) so the toggles here are the ONLY preprocessing
# applied; nothing is double-cleaned through the existing Python
# nlp/clean.py path. Same passage boundaries as the existing pipeline --
# this script does not re-segment anything.
#
# Usage (library, sourced by run_regime.R / preText driver scripts):
#   source("r/preprocess.R")
#   toks <- build_tokens(corpus_df, toggles = regime_toggles("moderate"))
#
# Each toggle is independently on/off so Part B (preText factorial design
# and the curated regime comparison) can vary them.

suppressMessages({
  library(quanteda)
  library(quanteda.textstats)
})

STOPWORD_FILE <- "data/spoken_stopwords.txt"

# ---------------------------------------------------------------- toggles --
# The 6 binary steps matching the preText-style factorial design, plus a
# couple of extra knobs (min/max docfreq, disfluency) that are always-on
# unless explicitly disabled -- preText's own factorial only varies the 6
# canonical steps; the docfreq/disfluency knobs are varied separately in
# the curated-regime sweep (Part B substantive checks), not blown up into
# the 64-cell factorial (that would make it 2^8 = 256 and preText's method
# specifically targets the 6 steps Denny & Spirling enumerate).
default_toggles <- list(
  lowercase        = TRUE,
  remove_punct     = TRUE,
  remove_numbers   = TRUE,
  remove_symbols   = TRUE,
  stopwords        = TRUE,   # extended (base quanteda + spoken filler list)
  ngrams           = FALSE,  # collocation detection + compounding
  infrequent_terms = TRUE,   # min/max docfreq trimming
  lemmatize        = FALSE,  # udpipe lemma (mutually exclusive with stem)
  stem             = FALSE,  # Porter stem -- comparison-only, not a default
  pos_filter       = FALSE,  # keep noun/proper-noun/verb/adj only (udpipe)
  disfluency       = TRUE    # strip stutters ("the the") + backchannel tokens
)

# Named regimes used throughout Part A/B. "minimal"/"moderate"/"aggressive"
# mirror the standard preText curated-regime convention; "aggressive_pos"
# adds POS-filtering on top of aggressive as the interpretability toggle.
regime_toggles <- function(name) {
  base <- default_toggles
  switch(name,
    minimal = modifyList(base, list(
      stopwords = FALSE, infrequent_terms = FALSE, disfluency = FALSE
    )),
    moderate = base,  # = default_toggles: standard + extended stopwords + docfreq + disfluency
    aggressive = modifyList(base, list(
      ngrams = TRUE, lemmatize = TRUE
    )),
    aggressive_pos = modifyList(base, list(
      ngrams = TRUE, lemmatize = TRUE, pos_filter = TRUE
    )),
    stem_comparison = modifyList(base, list(
      ngrams = TRUE, stem = TRUE, lemmatize = FALSE
    )),
    stop(sprintf("unknown regime '%s'", name))
  )
}

load_spoken_stopwords <- function(path = STOPWORD_FILE) {
  lines <- readLines(path, warn = FALSE)
  lines <- trimws(lines)
  lines <- lines[!grepl("^#", lines) & nchar(lines) > 0]
  unique(lines)
}

extended_stopwords <- function() {
  unique(c(stopwords("en"), load_spoken_stopwords()))
}

# ------------------------------------------------------------ disfluency --
# Collapse immediate word-repeat stutters ("the the", "I I", "know know")
# down to one occurrence, and drop standalone one-word backchannel lines
# that are pure interjection (already mostly covered by the stopword list,
# this catches the ones that show up as a token by themselves with no
# neighboring content, e.g. a lone "right." reaction line).
collapse_stutters <- function(char_vec) {
  # matches a token immediately followed by itself (case-insensitive),
  # collapses to a single instance; iterated once (double stutters like
  # "the the the" reduce to "the the" -> run twice to be safe).
  x <- char_vec
  pat <- "(?i)\\b(\\w+)\\b(\\s+\\1\\b)+"
  x <- gsub(pat, "\\1", x, perl = TRUE)
  x <- gsub(pat, "\\1", x, perl = TRUE)
  x
}

# ------------------------------------------------------------- pipeline --
# `corpus_df` must have columns: doc_id, text (raw passage text).
# Returns a quanteda tokens object with all requested toggles applied.
build_tokens <- function(corpus_df, toggles, udmodel = NULL, verbose = TRUE) {
  stopifnot(all(c("doc_id", "text") %in% names(corpus_df)))

  txt <- corpus_df$text
  if (isTRUE(toggles$disfluency)) {
    txt <- collapse_stutters(txt)
  }

  corp <- corpus(txt, docnames = corpus_df$doc_id)

  if (isTRUE(toggles$pos_filter) || isTRUE(toggles$lemmatize)) {
    if (is.null(udmodel)) stop("pos_filter/lemmatize requires udmodel (see r/udpipe_setup.R)")
    return(build_tokens_udpipe(corp, toggles, udmodel, verbose))
  }

  toks <- tokens(
    corp,
    remove_punct   = isTRUE(toggles$remove_punct),
    remove_numbers = isTRUE(toggles$remove_numbers),
    remove_symbols = isTRUE(toggles$remove_symbols),
    remove_url     = TRUE
  )

  if (isTRUE(toggles$lowercase)) toks <- tokens_tolower(toks)

  if (isTRUE(toggles$disfluency)) {
    # standalone backchannel tokens not already in the stopword list
    backchannel <- c("mm", "mhm", "huh", "hm", "ha", "eh", "oh", "ah")
    toks <- tokens_remove(toks, pattern = backchannel)
  }

  if (isTRUE(toggles$stopwords)) {
    toks <- tokens_remove(toks, pattern = extended_stopwords())
  }

  if (isTRUE(toggles$ngrams)) {
    toks <- compound_collocations(toks, verbose = verbose)
  }

  if (isTRUE(toggles$stem)) {
    toks <- tokens_wordstem(toks, language = "en")
  }

  toks <- tokens_select(toks, min_nchar = 2)
  toks
}

# udpipe path: annotate raw corpus, then keep either lemma or surface form,
# optionally filtered to a content-word POS whitelist. Runs stopword/
# ngram/docfreq toggles AFTER annotation so they see the same downstream
# treatment as the non-udpipe path.
#
# Split into (1) annotate raw text -> data.frame, (2) turn an already-built
# annotation data.frame into tokens for a given toggle set -- so a single
# expensive udpipe_annotate() pass (r/udpipe_annotate_parallel.R runs this
# in parallel across cores and caches the result) can be reused for BOTH
# the aggressive and aggressive_pos regimes without re-tagging the corpus.
build_tokens_udpipe <- function(corp, toggles, udmodel, verbose = TRUE) {
  suppressMessages(library(udpipe))
  txt <- as.character(corp)
  if (verbose) cat(sprintf("[udpipe] annotating %d passages (pos_filter=%s, lemmatize=%s)...\n",
                            length(txt), toggles$pos_filter, toggles$lemmatize))
  ann <- udpipe_annotate(udmodel, x = txt, doc_id = names(txt),
                          tagger = "default", parser = "none")
  ann <- as.data.frame(ann)
  tokens_from_annotation_df(ann, names(txt), toggles, verbose)
}

# `ann`: udpipe annotation data.frame (columns doc_id, token, lemma, upos).
# `doc_order`: character vector of doc_ids in the desired output order
# (docs with zero surviving tokens still appear, as an empty token vector).
tokens_from_annotation_df <- function(ann, doc_order, toggles, verbose = TRUE) {
  keep_pos <- c("NOUN", "PROPN", "VERB", "ADJ")
  if (isTRUE(toggles$pos_filter)) {
    ann <- ann[ann$upos %in% keep_pos, ]
  } else {
    # still drop pure punctuation/symbol/number UPOS tags even when not
    # doing full content-word filtering, matching remove_punct/numbers
    ann <- ann[!(ann$upos %in% c("PUNCT", "SYM", "NUM", "X")), ]
  }

  form <- if (isTRUE(toggles$lemmatize)) ann$lemma else ann$token
  form <- if (isTRUE(toggles$lowercase)) tolower(form) else form
  ann$form <- form
  ann$form[is.na(ann$form)] <- ""

  doc_tokens <- split(ann$form, ann$doc_id)
  missing <- setdiff(doc_order, names(doc_tokens))
  if (length(missing) > 0) {
    filler <- setNames(replicate(length(missing), character(0), simplify = FALSE), missing)
    doc_tokens <- c(doc_tokens, filler)
  }
  doc_tokens <- doc_tokens[doc_order]  # preserve original doc order
  toks <- as.tokens(doc_tokens)

  if (isTRUE(toggles$disfluency)) {
    toks <- tokens_remove(toks, pattern = c("mm", "mhm", "huh", "hm", "ha", "eh", "oh", "ah"))
  }
  if (isTRUE(toggles$stopwords)) {
    toks <- tokens_remove(toks, pattern = extended_stopwords())
  }
  if (isTRUE(toggles$ngrams)) {
    toks <- compound_collocations(toks, verbose = verbose)
  }
  toks <- tokens_select(toks, min_nchar = 2)
  toks
}

# ------------------------------------------------------------ collocations --
compound_collocations <- function(toks, top_n = 200, min_count = 30, verbose = TRUE) {
  colls <- textstat_collocations(toks, size = 2, min_count = min_count)
  colls <- colls[order(-colls$lambda), ]
  top <- head(colls, top_n)
  if (verbose) cat(sprintf("[collocations] compounding top %d of %d candidates (min_count=%d)\n",
                            nrow(top), nrow(colls), min_count))
  if (nrow(top) == 0) return(toks)
  tokens_compound(toks, pattern = phrase(top$collocation), concatenator = "_")
}

# ---------------------------------------------------------- docfreq prune --
# Applied to a DFM (post min/max docfreq trim), reported alongside the
# vocab accounting the task asks for (before/after size, tokens dropped).
apply_docfreq_prune <- function(dfm_obj, min_docfreq = 10, max_docfreq_prop = 0.5, verbose = TRUE) {
  vocab_before <- nfeat(dfm_obj)
  tokens_before <- sum(dfm_obj)
  out <- dfm_trim(dfm_obj, min_docfreq = min_docfreq, max_docfreq = max_docfreq_prop,
                   docfreq_type = "count", max_docfreq_type = "prop")
  vocab_after <- nfeat(out)
  tokens_after <- sum(out)
  if (verbose) {
    cat(sprintf("[docfreq prune] vocab %d -> %d (-%.1f%%), tokens %d -> %d (-%.1f%%)\n",
                vocab_before, vocab_after, 100 * (1 - vocab_after / vocab_before),
                tokens_before, tokens_after, 100 * (1 - tokens_after / tokens_before)))
  }
  attr(out, "vocab_accounting") <- list(
    vocab_before = vocab_before, vocab_after = vocab_after,
    tokens_before = tokens_before, tokens_after = tokens_after,
    min_docfreq = min_docfreq, max_docfreq_prop = max_docfreq_prop
  )
  out
}

build_dfm <- function(corpus_df, toggles, udmodel = NULL, verbose = TRUE) {
  toks <- build_tokens(corpus_df, toggles, udmodel = udmodel, verbose = verbose)
  d <- dfm(toks)
  if (isTRUE(toggles$infrequent_terms)) {
    d <- apply_docfreq_prune(d, verbose = verbose)
  } else {
    attr(d, "vocab_accounting") <- list(
      vocab_before = nfeat(d), vocab_after = nfeat(d),
      tokens_before = sum(d), tokens_after = sum(d),
      min_docfreq = NA, max_docfreq_prop = NA
    )
  }
  d
}
