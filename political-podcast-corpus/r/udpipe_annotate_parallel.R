#!/usr/bin/env Rscript
# Parallel udpipe annotation across the full ~40k-passage corpus, with a
# cache so the (expensive) tagging pass is run ONCE and shared by both the
# aggressive (lemmatize) and aggressive_pos (lemmatize + POS-filter)
# regimes -- they differ only in downstream filtering
# (tokens_from_annotation_df in r/preprocess.R), not in what gets tagged.
#
# udpipe::udpipe_annotate() is single-threaded per call; this splits the
# corpus into one chunk per core and runs parallel::mclapply, each worker
# loading its own (small, ~15-20MB) model instance.

.libPaths(c("r/lib", .libPaths()))
suppressMessages({
  library(udpipe)
  library(parallel)
})

UDPIPE_MODEL_DIR <- "r/udpipe_models"

get_udpipe_model_path <- function() {
  dir.create(UDPIPE_MODEL_DIR, showWarnings = FALSE, recursive = TRUE)
  existing <- list.files(UDPIPE_MODEL_DIR, pattern = "english.*\\.udpipe$", full.names = TRUE)
  if (length(existing) > 0) return(existing[1])
  cat("[udpipe] downloading English model (one-time)...\n")
  m <- udpipe_download_model(language = "english-ewt", model_dir = UDPIPE_MODEL_DIR)
  m$file_model
}

annotate_chunk <- function(model_path, ids, texts) {
  udmodel <- udpipe_load_model(model_path)
  ann <- udpipe_annotate(udmodel, x = texts, doc_id = ids, tagger = "default", parser = "none")
  as.data.frame(ann)[, c("doc_id", "token", "lemma", "upos")]
}

# raw: data.frame with doc_id, text. cache_path: .rds to read/write.
get_or_build_annotation <- function(raw, cache_path, n_cores = NULL) {
  if (file.exists(cache_path)) {
    cat(sprintf("[udpipe] loading cached annotation from %s\n", cache_path))
    return(readRDS(cache_path))
  }

  model_path <- get_udpipe_model_path()
  n_cores <- if (is.null(n_cores)) max(1, detectCores() - 2) else n_cores
  n <- nrow(raw)
  cat(sprintf("[udpipe] annotating %d passages across %d cores (model=%s)...\n",
              n, n_cores, basename(model_path)))

  split_idx <- split(seq_len(n), cut(seq_len(n), n_cores, labels = FALSE))
  t0 <- Sys.time()
  chunks <- mclapply(split_idx, function(idx) {
    annotate_chunk(model_path, raw$doc_id[idx], raw$text[idx])
  }, mc.cores = n_cores)
  elapsed <- as.numeric(Sys.time() - t0, units = "secs")

  failed <- vapply(chunks, function(x) inherits(x, "try-error") || is.null(x), logical(1))
  if (any(failed)) stop(sprintf("[udpipe] %d/%d worker chunks failed", sum(failed), length(chunks)))

  ann <- do.call(rbind, chunks)
  cat(sprintf("[udpipe] annotation done in %.1fs (%.0f passages/sec), %d tokens total\n",
              elapsed, n / elapsed, nrow(ann)))
  saveRDS(ann, cache_path)
  ann
}

tokens_from_annotation <- function(ann, doc_order, toggles) {
  tokens_from_annotation_df(ann, doc_order, toggles)
}
