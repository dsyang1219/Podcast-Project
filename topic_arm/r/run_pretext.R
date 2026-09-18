#!/usr/bin/env Rscript
# Part B, standard preText output (Denny & Spirling 2018).
#
# preText's own procedure is O(n^2) pairwise document-distance comparisons
# per specification, so it is run on a SAMPLE of passages, not the full
# ~40k-passage corpus -- this matches preText's own documented usage (the
# package's examples and the original paper's applications use samples on
# the order of a few hundred documents, precisely because of this cost).
# The curated-regime substantive checks (r/export_regime_tokens.R +
# nlp/lda_regime_refit.py) use the FULL corpus; only this sensitivity
# scan is sampled.
#
# Sampling: stratified by show (collection_id) so the sample isn't
# dominated by whichever shows happen to have the most chunks, capped at
# SAMPLE_N passages total.

.libPaths(c("r/lib", .libPaths()))
suppressMessages({
  library(quanteda)
  library(preText)
  library(jsonlite)
})

set.seed(0)
SAMPLE_N <- 300
OUT_DIR <- "data/output/regimes"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("[pretext] loading passages...\n")
raw <- read.csv("data/output/chunks_500_nolemma.csv", stringsAsFactors = FALSE,
                 colClasses = c(chunk_id = "character", collection_id = "character"))
raw <- raw[nchar(trimws(raw$text)) > 0, ]

# stratified sample: proportional-ish allocation capped per show so no
# single show dominates, then trimmed/padded to exactly SAMPLE_N.
by_show <- split(seq_len(nrow(raw)), raw$collection_id)
n_shows <- length(by_show)
per_show_cap <- max(1, ceiling(SAMPLE_N / n_shows) + 2)
sampled_idx <- unlist(lapply(by_show, function(idx) {
  sample(idx, size = min(length(idx), per_show_cap))
}))
if (length(sampled_idx) > SAMPLE_N) sampled_idx <- sample(sampled_idx, SAMPLE_N)
sample_df <- raw[sampled_idx, ]
cat(sprintf("[pretext] sampled %d passages from %d shows (stratified)\n",
            nrow(sample_df), length(unique(sample_df$collection_id))))

docs <- sample_df$text
names(docs) <- sample_df$chunk_id

cat("[pretext] generating factorial preprocessing specifications...\n")
preprocessed_documents <- factorial_preprocessing(
  docs,
  use_ngrams = TRUE,
  infrequent_term_threshold = 0.2,
  verbose = TRUE,
  parallel = TRUE,
  cores = max(1, parallel::detectCores() - 2)
)

n_specs <- length(preprocessed_documents$dfm_list)
cat(sprintf("[pretext] %d specifications generated\n", n_specs))
saveRDS(preprocessed_documents, file.path(OUT_DIR, "pretext_factorial_docs.rds"))

cat("[pretext] running preText scoring (bootstrapped document distances)...\n")
pretext_results <- preText(
  preprocessed_documents,
  dataset_name = "podcast_passages",
  distance_method = "cosine",
  num_comparisons = 20,
  verbose = TRUE,
  parallel = TRUE,
  cores = max(1, parallel::detectCores() - 2)
)
saveRDS(pretext_results, file.path(OUT_DIR, "pretext_results.rds"))

cat("[pretext] regression: which preprocessing steps drive score variation...\n")
reg_plot <- tryCatch(
  regression_coefficient_plot(pretext_results, remove_intercept = TRUE),
  error = function(e) { cat("[pretext] regression_coefficient_plot failed:", conditionMessage(e), "\n"); NULL }
)
if (!is.null(reg_plot)) {
  ggplot2::ggsave(file.path(OUT_DIR, "pretext_regression_coefficients.png"), reg_plot,
                   width = 8, height = 6, dpi = 150)
}

score_plot <- tryCatch(
  preText_score_plot(pretext_results),
  error = function(e) { cat("[pretext] preText_score_plot failed:", conditionMessage(e), "\n"); NULL }
)
if (!is.null(score_plot)) {
  ggplot2::ggsave(file.path(OUT_DIR, "pretext_scores.png"), score_plot, width = 10, height = 8, dpi = 150)
}

scores_df <- pretext_results$preText_scores
write.csv(scores_df, file.path(OUT_DIR, "pretext_scores.csv"), row.names = FALSE)

# regression coefficients as a plain table too (not just the plot), so the
# "which steps drive the most variation" answer is machine-readable.
reg_summary <- tryCatch({
  form <- as.formula(paste("preText_score ~", paste(setdiff(
    names(pretext_results$choice_data), c("preText_score")
  ), collapse = " + ")))
  fit <- lm(form, data = cbind(preText_score = scores_df$preText_score, pretext_results$choice_data))
  as.data.frame(summary(fit)$coefficients)
}, error = function(e) {
  cat("[pretext] manual regression fallback failed:", conditionMessage(e), "\n")
  NULL
})
if (!is.null(reg_summary)) {
  write.csv(reg_summary, file.path(OUT_DIR, "pretext_regression_table.csv"), row.names = TRUE)
}

summary_out <- list(
  n_passages_sampled = nrow(sample_df),
  n_shows_sampled = length(unique(sample_df$collection_id)),
  n_specifications = n_specs,
  mean_score = mean(scores_df$preText_score),
  sd_score = sd(scores_df$preText_score),
  top5_most_unusual = head(scores_df[order(-scores_df$preText_score), ], 5),
  top5_least_unusual = head(scores_df[order(scores_df$preText_score), ], 5)
)
write_json(summary_out, file.path(OUT_DIR, "pretext_summary.json"), auto_unbox = TRUE, pretty = TRUE, force = TRUE)

cat("\n[pretext] top 5 most unusual specifications (highest preText score = most different from consensus):\n")
print(head(scores_df[order(-scores_df$preText_score), c("preText_score")], 5))
cat("\n[pretext] DONE\n")
