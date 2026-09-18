#!/usr/bin/env Rscript
# Build one curated regime's full-corpus token export for the Python LDA
# refit (nlp/lda_regime_refit.py). Reads the raw passage text (same
# boundaries as the existing pipeline, no re-segmentation), applies
# r/preprocess.R's toggles for the named regime, and writes:
#   data/output/regimes/tokens_<regime>.csv   (chunk_id, collection_id, clean_text)
#   data/output/regimes/vocab_accounting_<regime>.json
#
# Usage: Rscript r/export_regime_tokens.R <regime_name>
#   regime_name in {minimal, moderate, aggressive, aggressive_pos, stem_comparison}

.libPaths(c("r/lib", .libPaths()))
suppressMessages({
  library(quanteda)
  library(jsonlite)
})
source("r/preprocess.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript r/export_regime_tokens.R <regime_name>")
regime_name <- args[1]

OUT_DIR <- "data/output/regimes"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

CHUNKS_PATH <- "data/output/chunks_500_nolemma.csv"
CACHE_DIR <- "data/output/regimes/udpipe_cache"
dir.create(CACHE_DIR, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("[%s] loading raw passages from %s...\n", regime_name, CHUNKS_PATH))
raw <- read.csv(CHUNKS_PATH, stringsAsFactors = FALSE,
                 colClasses = c(chunk_id = "character", collection_id = "character"))
raw <- raw[, c("chunk_id", "collection_id", "text")]
raw$text[is.na(raw$text)] <- ""
names(raw)[names(raw) == "chunk_id"] <- "doc_id"
cat(sprintf("[%s] %d passages loaded\n", regime_name, nrow(raw)))

toggles <- regime_toggles(regime_name)
needs_udpipe <- isTRUE(toggles$pos_filter) || isTRUE(toggles$lemmatize)

udmodel <- NULL
if (needs_udpipe) {
  source("r/udpipe_annotate_parallel.R")
  ann_cache <- file.path(CACHE_DIR, "full_corpus_annotated.rds")
  ann <- get_or_build_annotation(raw, ann_cache)
  toks <- tokens_from_annotation(ann, raw$doc_id, toggles)
} else {
  corpus_df <- raw[, c("doc_id", "text")]
  toks <- build_tokens(corpus_df, toggles, verbose = TRUE)
}

d <- dfm(toks)
if (isTRUE(toggles$infrequent_terms)) {
  d <- apply_docfreq_prune(d, verbose = TRUE)
} else {
  attr(d, "vocab_accounting") <- list(
    vocab_before = nfeat(d), vocab_after = nfeat(d),
    tokens_before = sum(d), tokens_after = sum(d),
    min_docfreq = NA, max_docfreq_prop = NA
  )
}

kept_vocab <- featnames(d)
toks_pruned <- tokens_select(toks, pattern = kept_vocab, selection = "keep")

clean_text <- vapply(as.list(toks_pruned), paste, character(1), collapse = " ")
out_df <- data.frame(
  chunk_id = raw$doc_id,
  collection_id = raw$collection_id,
  clean_text = clean_text[raw$doc_id],
  stringsAsFactors = FALSE
)
out_path <- file.path(OUT_DIR, sprintf("tokens_%s.csv", regime_name))
write.csv(out_df, out_path, row.names = FALSE)
cat(sprintf("[%s] wrote %s (%d rows, %d empty)\n", regime_name, out_path,
            nrow(out_df), sum(out_df$clean_text == "")))

acc <- attr(d, "vocab_accounting")
acc$regime <- regime_name
acc$n_docs <- nrow(out_df)
acc$n_empty_docs <- sum(out_df$clean_text == "")
acc$toggles <- toggles
write_json(acc, file.path(OUT_DIR, sprintf("vocab_accounting_%s.json", regime_name)),
           auto_unbox = TRUE, pretty = TRUE)
cat(sprintf("[%s] vocab accounting: %d -> %d vocab, %d -> %d tokens\n",
            regime_name, acc$vocab_before, acc$vocab_after, acc$tokens_before, acc$tokens_after))
cat(sprintf("[%s] DONE\n", regime_name))
