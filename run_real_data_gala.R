# Galapagos species counts (Poisson regression): dissertation Section 5.3.1.
# Run from the repository root:  Rscript run_real_data_gala.R
#
# Design as in the dissertation: one 80/20 train/test split (seed 1), then each
# of the 12 Poisson models is fitted 20 times on that split.
#
# Changes since submission:
#  1. Predictions include the intercept beta0. Without it, lambda = exp(X beta)
#     is centred on 1 while species counts run into the hundreds, so every
#     log-likelihood was far too low (worst of all when nothing was selected).
#  2. `Endemics` is no longer a predictor. It counts the endemic species on each
#     island, which is part of the response (Species), so it leaks the answer.
#     The original code used every column except Species, so it was included
#     (it is V1 in Appendix C.8, the variable SSVS, BLASSO and GDP always chose).
#     Set INCLUDE_ENDEMICS <- TRUE to reproduce the original design.
#  3. Optional (LOG_PREDICTORS): use log(Area), log(Elevation), log(Adjacent).
#     These three are very skewed (a few huge islands), so on the raw scale one
#     extreme test island can make exp(eta) explode. Logging them is the usual
#     analysis of this dataset (e.g. Faraway). Nearest and Scruz stay raw
#     (Scruz is 0 for Santa Cruz itself). FALSE reproduces the original scale.

if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source("R/models.R")
source("R/evaluate.R")
source("R/real_data.R")

options(mc.cores = parallel::detectCores())
dir.create("results", showWarnings = FALSE)

INCLUDE_ENDEMICS <- FALSE
LOG_PREDICTORS   <- TRUE
N_REP <- 20   # set to 1 for a quick check

data("gala", package = "faraway")
# Columns: Species (response), Endemics, Area, Elevation, Nearest, Scruz, Adjacent
predictors <- c("Area", "Elevation", "Nearest", "Scruz", "Adjacent")
if (INCLUDE_ENDEMICS) predictors <- c("Endemics", predictors)

y <- gala$Species
X <- gala[, predictors]
if (LOG_PREDICTORS) for (v in c("Area", "Elevation", "Adjacent")) X[[v]] <- log(X[[v]])

split <- split_and_standardise(X, y, train_frac = 0.8, seed = 1)

out_file <- paste0("results/real_gala",
                   if (INCLUDE_ENDEMICS) "_with_endemics",
                   if (LOG_PREDICTORS) "_log", ".rds")
res <- run_real_data(glm_models("poisson"), split, family = "poisson",
                     n_rep = N_REP, out_file = out_file)

print(summarise_real(res))           # mean test/train log-likelihood, d_hat, time
print(round(res$selection_props, 2)) # share of runs selecting each variable

# Reference point: predicting the training mean count for every test island.
cat("Intercept-only test LL:",
    round(sum(dpois(split$y_test, mean(split$y_train), log = TRUE)), 1), "\n")
