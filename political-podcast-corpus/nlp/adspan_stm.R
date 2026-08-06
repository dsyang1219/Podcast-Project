#!/usr/bin/env Rscript
# STM arm: DESCRIBE how ideology shapes discourse. Not evidence of recoverability.
#
# THE CIRCULARITY BOUNDARY -- read before using any number this writes
# -------------------------------------------------------------------
# Ideology is an INPUT to this model. Nothing here can therefore be evidence
# that ideology is recoverable from discourse, or that topics predict ideology.
# That claim comes only from the unsupervised LDA/PCA arm, which never saw the
# target. What this model can say is DESCRIPTIVE: taking the ideology-discourse
# link as already established (externally via DIME, and internally via the
# unsupervised result), it describes the SHAPE of that link -- which topics are
# more prevalent on each side (Q1), and whether the same topic is worded
# differently by side (Q2).
#
# Every output file here is named and phrased for descriptive use. If a number
# from this script ends up supporting a sentence of the form "this shows
# ideology can be recovered", that sentence is wrong.
#
# K
# -
# K=75 is inherited from the LDA arm, where it was chosen as the c_v coherence
# PEAK -- a criterion computed from the corpus alone that never sees ideology.
# So K is independent of the outcome here too. (The LDA arm's pre-registered
# 1SE rule would instead pick K=30, and the c_v peak is within 1SE of K=30, 50,
# 90 and 100 -- coherence cannot really discriminate in that range. K=75 is used
# for comparability with the rest of the project, not because it is sharply
# optimal.)
#
# Unit: PASSAGE (37,942), nested within 204 shows. See adspan_stm_export.py --
# estimateEffect treats passages as independent, so its CIs are
# ANTI-CONSERVATIVE. A show-clustered comparison is written alongside.
#
#   Rscript nlp/adspan_stm.R prevalence   # Q1
#   Rscript nlp/adspan_stm.R content      # Q2

suppressPackageStartupMessages({
  library(stm); library(quanteda); library(data.table); library(jsonlite)
})

args  <- commandArgs(trailingOnly = TRUE)
phase <- if (length(args) >= 1) args[1] else "prevalence"
K     <- if (length(args) >= 2) as.integer(args[2]) else 75L
SEED  <- 12345L
EMITS <- if (length(args) >= 3) as.integer(args[3]) else 75L
NGRP  <- if (length(args) >= 4) as.integer(args[4]) else 1L
EMTOL <- if (length(args) >= 5) as.numeric(args[5]) else 1e-5
OUT   <- "data/output/adspan"

cat(sprintf("[stm] phase=%s K=%d\n", phase, K))

# ---------------------------------------------------------------- data ----
dt <- fread(file.path(OUT, "stm_input.csv"), colClasses = c(collection_id = "character"))
dt[, ideo_group := factor(ideo_group, levels = c("Liberal", "Moderate", "Conservative"))]
cat(sprintf("[data] %d passages, %d shows\n", nrow(dt), uniqueN(dt$collection_id)))

# clean_text is ALREADY tokenized and pruned by the Python pipeline. Splitting
# on whitespace and tallying is the only step allowed -- any tokenizer cleaning
# here would hand STM a different vocabulary than LDA saw.
toks <- quanteda::tokens(dt$clean_text, what = "fastestword")
dfmat <- quanteda::dfm(toks, tolower = FALSE)
cat(sprintf("[dfm] %d docs x %d features\n", ndoc(dfmat), nfeat(dfmat)))

conv <- quanteda::convert(dfmat, to = "stm",
                          docvars = dt[, .(collection_id, ideology, ideo_group,
                                           is_host_target)])
docs <- conv$documents; vocab <- conv$vocab
# MUST be a plain data.frame: stm's estimateEffect indexes metadata with
# `metadata[, varlist, drop=FALSE]`, which a data.table interprets as a symbol
# lookup and errors on. quanteda::convert hands back a data.table here.
meta <- as.data.frame(conv$meta)
cat(sprintf("[stm] vocab %d (LDA arm used 38008)\n", length(vocab)))

# --------------------------------------------------- register flagging ----
# Same rule the Python side uses: >=2 spoken-stopword/profanity terms in the top
# words, or any profanity. The project's own is_contaminant flag (n_filler>=5)
# is too permissive and misses the topics that actually anchor components.
PROFANITY <- c("fucking","fuck","shit","shitty","damn","hell","ass","bitch","goddamn")
sw_path <- "data/spoken_stopwords.txt"
FILLER <- if (file.exists(sw_path)) readLines(sw_path, warn = FALSE) else character(0)
is_register <- function(words) {
  nf <- sum(words %in% FILLER | words %in% PROFANITY)
  np <- sum(words %in% PROFANITY)
  nf >= 2 || np >= 1
}

# --------------------------------------------------------------- fits ----
if (phase == "prevalence") {
  t0 <- Sys.time()
  fit <- stm(docs, vocab, K = K, prevalence = ~ s(ideology), data = meta,
             init.type = "Spectral", seed = SEED, verbose = TRUE,
             max.em.its = EMITS, reportevery = 5)
  cat(sprintf("[fit] %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  saveRDS(fit, file.path(OUT, sprintf("stm_prevalence_k%d.rds", K)))

  lab <- labelTopics(fit, n = 12)
  qual <- data.table(topic = 1:K,
                     coherence = semanticCoherence(fit, docs),
                     exclusivity = exclusivity(fit),
                     prop = colMeans(fit$theta))
  qual[, top_words := apply(lab$prob, 1, paste, collapse = " ")]
  qual[, frex_words := apply(lab$frex, 1, paste, collapse = " ")]
  qual[, register := sapply(1:K, function(i) is_register(lab$prob[i, ]))]

  # A LINEAR ideology term for the reported effect, even though the model used a
  # spline: a spline coefficient is not interpretable as "per unit of DIME",
  # and the deliverable asks for an effect with a CI. The spline stays in the
  # model so prevalence is not forced linear during fitting.
  ee <- estimateEffect(1:K ~ ideology, fit, metadata = meta, uncertainty = "Global")
  s <- summary(ee)
  eff <- rbindlist(lapply(1:K, function(i) {
    co <- s$tables[[i]]
    data.table(topic = i, estimate = co["ideology", 1], se = co["ideology", 2],
               t = co["ideology", 3], p = co["ideology", 4])
  }))
  eff[, `:=`(ci_low = estimate - 1.96 * se, ci_high = estimate + 1.96 * se)]

  # Show-clustered check: collapse theta to show means and regress on show-level
  # ideology (n=204). This is the honest effective sample size. The passage-level
  # SEs above will be far smaller; the DIRECTION and RANKING should agree, and
  # where they do not, trust this one.
  th <- as.data.table(fit$theta); setnames(th, paste0("T", 1:K))
  th[, collection_id := meta$collection_id]
  showmeans <- th[, lapply(.SD, mean), by = collection_id]
  sid <- unique(dt[, .(collection_id, ideology)])
  showmeans <- merge(showmeans, sid, by = "collection_id")
  clust <- rbindlist(lapply(1:K, function(i) {
    m <- lm(showmeans[[paste0("T", i)]] ~ showmeans$ideology)
    cs <- summary(m)$coefficients
    data.table(topic = i, est_show = cs[2, 1], se_show = cs[2, 2], p_show = cs[2, 4])
  }))

  res <- merge(merge(qual, eff, by = "topic"), clust, by = "topic")
  fwrite(res, file.path(OUT, sprintf("stm_prevalence_effects_k%d.csv", K)))
  cat(sprintf("[out] stm_prevalence_effects_k%d.csv\n", K))

} else if (phase == "content") {
  t0 <- Sys.time()
  # content must be a single categorical variable -- stm cannot take a
  # continuous content covariate, which is why ideo_group (terciles) exists.
  # SPEED: a BINARY content covariate, not the terciles. SAGE estimates a
  # kappa deviation per level per topic over the whole vocabulary, so cost
  # scales with the number of levels; 3 -> 2 removes a third of that work. It
  # also sharpens the question actually being asked -- Q2 is "do liberal and
  # conservative shows word this topic differently", and the Moderate tercile
  # was never part of that contrast. Every DOCUMENT is retained, so the corpus
  # and vocabulary are unchanged and the topic space stays comparable.
  sid <- unique(dt[, .(collection_id, ideology)])
  med <- median(sid$ideology)
  meta$ideo_bin <- factor(ifelse(meta$ideology > med, "Conservative", "Liberal"),
                          levels = c("Liberal", "Conservative"))
  cat(sprintf("[content] binary split at show-level median %.4f: %s\n", med,
              paste(names(table(meta$ideo_bin)), table(meta$ideo_bin),
                    sep = "=", collapse = " ")))
  fit <- stm(docs, vocab, K = K, prevalence = ~ s(ideology),
             content = ~ ideo_bin, data = meta,
             init.type = "Spectral", seed = SEED, verbose = TRUE,
             max.em.its = EMITS, reportevery = 5,
             ngroups = NGRP, emtol = EMTOL)
  cat(sprintf("[fit] %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  saveRDS(fit, file.path(OUT, sprintf("stm_content_k%d.rds", K)))

  sl <- sageLabels(fit, n = 15)
  lvls <- levels(meta$ideo_bin)
  rows <- list()
  for (i in 1:K) {
    base <- sl$marginal$prob[i, ]
    rows[[length(rows) + 1]] <- data.table(
      topic = i, level = "MARGINAL", words = paste(base, collapse = " "),
      register = is_register(base))
    for (j in seq_along(lvls)) {
      w <- sl$cov.betas[[j]]$problabels[i, ]
      rows[[length(rows) + 1]] <- data.table(
        topic = i, level = lvls[j], words = paste(w, collapse = " "),
        register = is_register(w))
    }
  }
  fwrite(rbindlist(rows), file.path(OUT, sprintf("stm_content_words_k%d.csv", K)))
  qual <- data.table(topic = 1:K, prop = colMeans(fit$theta),
                     coherence = semanticCoherence(fit, docs))
  fwrite(qual, file.path(OUT, sprintf("stm_content_quality_k%d.csv", K)))
  cat(sprintf("[out] stm_content_words_k%d.csv\n", K))
}
cat("[done]\n")
