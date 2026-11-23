# Riboflavin gene expression (normal regression, p >> n): dissertation Section 5.1.2.
# Run from the repository root:  Rscript run_real_data_riboflavin.R
#
# Design as in the dissertation: one 80/20 train/test split (seed 1), X and y
# standardised on the training set, a single run of 13 models (the slow full
# hierarchies are replaced by their faster variants where those exist).
# Expect a few hours: keep the Mac awake and plugged in.
#
# Change since submission: the original code ran scale() on X *before* ranking
# genes by variance. After scaling every gene has variance 1, so the "1,000 most
# variable genes" were in effect the first 1,000 columns. Genes are now ranked
# by variance on the raw training rows, then standardised.
# Predictions also include the intercept (negligible here, since y_train has
# mean 0, but it keeps every real-data script the same).

if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source("R/models.R")
source("R/evaluate.R")
source("R/real_data.R")

options(mc.cores = parallel::detectCores())
dir.create("results", showWarnings = FALSE)

# The data ship with the hdi package, which is archived on CRAN and hard to
# install on new R versions (its dependency lars needs a Fortran compiler).
# If hdi isn't installed, download its last source release and load just the
# data file from it. No compiling needed.
load_riboflavin <- function() {
  if (requireNamespace("hdi", quietly = TRUE)) {
    e <- new.env(); data("riboflavin", package = "hdi", envir = e)
    return(e$riboflavin)
  }
  tarball <- remotes:::download_version("hdi")
  dir <- tempfile(); untar(tarball, exdir = dir)
  e <- new.env()
  load(list.files(file.path(dir, "hdi", "data"), pattern = "^riboflavin", full.names = TRUE)[1], envir = e)
  e$riboflavin
}
riboflavin <- load_riboflavin()
stopifnot(nrow(riboflavin$x) == 71, ncol(riboflavin$x) == 4088)
X <- as.matrix(riboflavin$x)   # 71 samples x 4,088 genes (log expression)
y <- riboflavin$y              # log riboflavin production rate

split <- split_and_standardise(X, y, train_frac = 0.8, seed = 1,
                               standardise_y = TRUE, top_var = 1000)

ribo_models <- c(normal_models[c("SSVS_fixed", "SSVS_prior", "BLASSO_fixed", "BLASSO_prior")],
                 faster_models[c("BLASSO_faster_prior", "BLASSO_fasterfaster_prior",
                                 "Horseshoe_fasterfaster",
                                 "HorseshoePlus_faster_fixed", "HorseshoePlus_faster_prior",
                                 "GDP_faster_fixed", "GDP_faster_prior")],
                 normal_models[c("DirichletLaplace_fixed", "DirichletLaplace_prior")])

res <- run_real_data(ribo_models, split, family = "normal",
                     n_rep = 1, out_file = "results/real_riboflavin.rds")

print(summarise_real(res))   # test/train MSE, number of genes selected, time

# Reference point: y is standardised, so predicting the training mean (0) for
# every test sample gives this test MSE. A useful model should beat it.
cat("Intercept-only test MSE:", round(mean(split$y_test^2), 3), "\n")
