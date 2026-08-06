lib <- "r/lib"
dir.create(lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))

cran_pkgs <- c("quanteda", "quanteda.textstats", "quanteda.textplots",
               "udpipe", "remotes", "jsonlite", "dplyr", "purrr", "SnowballC")
for (p in cran_pkgs) {
  if (!requireNamespace(p, quietly = TRUE, lib.loc = lib)) {
    cat("Installing", p, "...\n")
    install.packages(p, lib = lib, Ncpus = 8)
  } else {
    cat(p, "already installed\n")
  }
}

if (!requireNamespace("preText", quietly = TRUE, lib.loc = lib)) {
  cat("Installing preText from GitHub...\n")
  remotes::install_github("matthewjdenny/preText", lib = lib, upgrade = "never")
} else {
  cat("preText already installed\n")
}

cat("DONE\n")
