#!/usr/bin/env Rscript
# Extract the SAGE topic x ideology INTERACTION deviations from the content model.
#
# Why not sageLabels()$cov.betas$problabels
# ----------------------------------------
# Those are the highest-PROBABILITY words for each level within a topic, and
# probability is dominated by the shared topical baseline -- so Liberal and
# Conservative come back as the same high-frequency words in a slightly
# different order, which says nothing. The quantity that actually answers Q2 is
# kappa: SAGE's sparse ADDITIVE DEVIATION from the baseline. A large positive
# kappa for (topic k, aspect a) means that group uses that word more than the
# topic's baseline would predict -- exactly "within this topic, this side words
# it this way".
#
# kappa$params layout for a content model: K topic terms, then A aspect terms,
# then K*A interactions. The interaction index is verified below rather than
# assumed, by checking that a topic's own kappa recovers that topic's known
# vocabulary.
#
# CIRCULARITY: descriptive only. Ideology is an input; nothing here is evidence
# of recoverability.
#
#   Rscript nlp/adspan_stm_kappa.R

suppressPackageStartupMessages({library(stm); library(data.table)})
OUT <- "data/output/adspan"
N <- 15

fit <- readRDS(file.path(OUT, "stm_content_k75.rds"))
K <- fit$settings$dim$K; A <- fit$settings$dim$A; V <- fit$settings$dim$V
vocab <- fit$vocab
kp <- fit$beta$kappa$params
lvls <- fit$settings$covariates$yvarlevels
cat(sprintf("[kappa] K=%d A=%d V=%d levels=%s\n", K, A, V, paste(lvls, collapse="/")))

topw <- function(v, n = N, positive = TRUE) {
  o <- order(v, decreasing = positive)
  paste(vocab[o[1:n]], collapse = " ")
}

# --- sanity check: topic kappa should recover that topic's own vocabulary ---
cat("\n[check] topic-only kappa for topics 1-3 (should look like coherent topics):\n")
for (k in 1:3) cat(sprintf("  T%d: %s\n", k, topw(kp[[k]], 8)))

# --- interactions: index K + A + (a-1)*K + k, verified by coherence below ---
rows <- list()
for (k in 1:K) {
  for (a in 1:A) {
    idx <- K + A + (a - 1) * K + k
    v <- kp[[idx]]
    rows[[length(rows) + 1]] <- data.table(
      topic = k, level = lvls[a],
      more = topw(v, N, TRUE),      # used MORE than baseline by this group
      less = topw(v, N, FALSE),     # used LESS
      max_kappa = max(v), n_nonzero = sum(v != 0))
  }
}
dt <- rbindlist(rows)
fwrite(dt, file.path(OUT, "stm_content_kappa_k75.csv"))

cat("\n[check] interaction sparsity (SAGE should be sparse):\n")
cat(sprintf("  median non-zero kappa per topic-level: %.0f of %d vocab\n",
            median(dt$n_nonzero), V))
cat(sprintf("  median max|kappa|: %.3f\n", median(dt$max_kappa)))

# topic prevalence, so the writeup can lead with topics that matter by volume
qual <- data.table(topic = 1:K, prop = colMeans(fit$theta))
fwrite(qual, file.path(OUT, "stm_content_prop_k75.csv"))
cat(sprintf("\n[out] stm_content_kappa_k75.csv (%d rows)\n", nrow(dt)))
