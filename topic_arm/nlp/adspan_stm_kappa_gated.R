#!/usr/bin/env Rscript
# Frequency- and show-dispersion-gated readout of the SAGE topic x ideology
# interaction kappas from the content model.
#
# Why the ungated readout (adspan_stm_kappa.R) is not usable for the writeup
# ------------------------------------------------------------------------
# Two separate problems, both diagnosed on the K=75 fit:
#
#   1. RARE-WORD DOMINANCE. kappa is an unweighted additive deviation, so its
#      largest values land on terms with almost no corpus support. Median corpus
#      count of the ungated top-15 was 23 occurrences; 78% occurred <50 times.
#
#   2. SHOW-IDENTITY LEAKAGE. The content covariate is a SHOW-LEVEL median split,
#      so the binary aspect is partly just a partition of shows. Any term that
#      identifies a show -- host names, guest names, segment names -- separates
#      the two levels perfectly and absorbs kappa mass. That is show identity,
#      not ideology, and it dominated the frequency-gated top-10.
#
# Corpus-wide dispersion does NOT fix (2) -- measured, not assumed: gating at
# >=20 shows removed 20 of 8669 terms and left the readout unchanged. The
# leaking tokens are common GIVEN names (eli, todd, jake, matt, sarah, ari)
# that many shows say, so they are widely dispersed and dispersion cannot see
# them. A gazetteer is required.
#
# The gazetteer is built from the project's own resources rather than a generic
# name list, and deliberately does NOT drop every person-name:
#   - HOSTS (host_dime_lookup_v2.csv, host_name + alt_names) are always dropped.
#     A host name is show identity by construction.
#   - GUESTS (guests_dime.csv) are dropped only when the guest appears on <=2
#     distinct shows. A guest carried across many shows is a public figure, and
#     which side talks about them is exactly the signal Q2 is asking about.
# So mamdani / fetterman / greene / soros survive; joyce and jill (the
# #SistersInLaw hosts, which were leading two Liberal lists) do not.
#
# A /usr/share/dict case heuristic was tried and rejected: it flags any term the
# dictionary only lists capitalized, which includes medicare and antifa.
#
# Also note the `less` column of the ungated CSV is meaningless: kappa is
# overwhelmingly one-sided (median 1352 positive / 68 negative / 36576 zero per
# topic-level), so an ascending sort returns a 36k-way tie at zero broken by
# vocab index -- i.e. alphabetical order. Zeros are dropped here.
#
# CIRCULARITY: descriptive only. Ideology is an input; nothing here is evidence
# of recoverability.
#
#   Rscript nlp/adspan_stm_kappa_gated.R
suppressPackageStartupMessages({library(stm); library(quanteda); library(data.table)})
OUT     <- "data/output/adspan"
N       <- 15
MINC    <- 200L   # min corpus occurrences
MINSHOW <- 20L    # min distinct shows overall (of 204)
MINSIDE <- 10L    # min distinct shows on the attributed side

fit   <- readRDS(file.path(OUT, "stm_content_k75.rds"))
K <- fit$settings$dim$K; A <- fit$settings$dim$A; V <- fit$settings$dim$V
vocab <- fit$vocab
kp    <- fit$beta$kappa$params
lvls  <- fit$settings$covariates$yvarlevels
wc    <- fit$settings$dim$wcounts$x
cat(sprintf("[kappa] K=%d A=%d V=%d levels=%s\n", K, A, V, paste(lvls, collapse="/")))

# ---- show-level incidence, rebuilt exactly as the fit saw the corpus ----
dt <- fread(file.path(OUT, "stm_input.csv"), colClasses = c(collection_id = "character"))
sid <- unique(dt[, .(collection_id, ideology)])
med <- median(sid$ideology)                       # same rule as adspan_stm.R
dt[, ideo_bin := factor(ifelse(ideology > med, "Conservative", "Liberal"),
                        levels = c("Liberal", "Conservative"))]
cat(sprintf("[split] show-level median %.4f (must match the fit log)\n", med))

dfmat <- quanteda::dfm(quanteda::tokens(dt$clean_text, what = "fastestword"),
                       tolower = FALSE)
stopifnot(identical(sort(featnames(dfmat)), sort(vocab)))
byshow <- quanteda::dfm_group(dfmat, groups = dt$collection_id)
byshow <- byshow[, match(vocab, featnames(byshow))]      # align to fit's vocab
shows_tot <- Matrix::colSums(byshow > 0)

side <- unique(dt[, .(collection_id, ideo_bin)])
side <- side[match(rownames(byshow), collection_id)]
shows_side <- sapply(lvls, function(l) Matrix::colSums(byshow[side$ideo_bin == l, ] > 0))

# ------------------------------------------------- name gazetteer ----
MAXGUESTSHOWS <- 2L
name_toks <- function(x) {
  x <- unlist(strsplit(tolower(x[!is.na(x)]), "[^a-z'-]+"))
  x <- gsub("'s$", "", x)
  unique(x[nchar(x) >= 2])
}
hosts <- fread("data/output/host_dime_lookup_v2.csv", colClasses = "character")
host_names <- c(hosts$host_name, unlist(strsplit(hosts$alt_names, ";")))
host_tok <- name_toks(host_names)

gu <- fread("data/output/guests_dime.csv", colClasses = c(collection_id = "character"))
gshows <- gu[, .(n_shows = uniqueN(collection_id)), by = guest_name]

# Per TOKEN, not per person: how many DISTINCT people carry it, and what is the
# widest show reach of any of them. A token borne by many different people is a
# generic given name (janet, abby, jasmine) and is low-content whoever says it;
# a token borne by one person with wide reach is a public figure (fetterman,
# greene) and is exactly the signal Q2 wants. Doing this per person instead --
# subtracting the token sets of multi-show guests -- silently rescues every
# generic given name that any wide-reach guest happens to share, which is the
# bug this replaces.
MAXPEOPLE <- 5L
tk <- gshows[, .(tok = name_toks(guest_name)), by = .(guest_name, n_shows)]
tstat <- tk[, .(n_people = uniqueN(guest_name), max_shows = max(n_shows)), by = tok]
guest_tok <- tstat[n_people >= MAXPEOPLE | max_shows <= MAXGUESTSHOWS, tok]
NAMES <- union(host_tok, guest_tok)
cat(sprintf("[names] %d host + %d guest tokens (generic: >=%d people; show-local: <=%d shows) = %d unique\n",
            length(host_tok), length(guest_tok), MAXPEOPLE, MAXGUESTSHOWS, length(NAMES)))
cat(sprintf("[names] kept as public figures: %s ...\n",
            paste(head(intersect(tstat[n_people < MAXPEOPLE & max_shows > MAXGUESTSHOWS, tok],
                                 tolower(vocab)), 12), collapse = " ")))

is_name <- gsub("'s$", "", tolower(vocab)) %in% NAMES
cat(sprintf("[names] %d of %d vocab terms flagged as names (%d of them pass the count gate)\n",
            sum(is_name), V, sum(is_name & wc >= MINC)))
cat(sprintf("[gate] count>=%d: %d | +shows>=%d: %d | +not-a-name: %d terms\n",
            MINC, sum(wc >= MINC), MINSHOW, sum(wc >= MINC & shows_tot >= MINSHOW),
            sum(wc >= MINC & shows_tot >= MINSHOW & !is_name)))

# ---- gated readout; interaction index verified against mnreg's design ----
idx <- function(k, a) K + A + (a - 1) * K + k
topw <- function(v, a, n = N, names_only = FALSE) {
  ok <- which(wc >= MINC & shows_tot >= MINSHOW & shows_side[, a] >= MINSIDE &
              v > 0 & (is_name == names_only))
  ok <- ok[order(v[ok], decreasing = TRUE)][1:n]
  paste(vocab[ok[!is.na(ok)]], collapse = " ")
}
prop <- colMeans(fit$theta)
# dropped_names is kept alongside so the filter is auditable rather than silent
res <- rbindlist(lapply(1:K, function(k) rbindlist(lapply(1:A, function(a) {
  data.table(topic = k, prop = prop[k], level = lvls[a],
             more = topw(kp[[idx(k, a)]], a),
             dropped_names = topw(kp[[idx(k, a)]], a, names_only = TRUE))
}))))
fwrite(res, file.path(OUT, "stm_content_kappa_k75_gated.csv"))

cat("\n=== top 12 topics by prevalence ===\n")
for (k in order(prop, decreasing = TRUE)[1:12]) {
  cat(sprintf("\nT%d (prop %.3f)\n", k, prop[k]))
  for (a in 1:A) cat(sprintf("  %-12s %s\n", lvls[a], res[topic == k & level == lvls[a], more]))
}
cat(sprintf("\n[out] stm_content_kappa_k75_gated.csv (%d rows)\n", nrow(res)))
