# Boston Housing (normal regression): dissertation Section 5.1.1.
# Run from the repository root:  Rscript run_real_data_boston.R
#
# Two designs, as in the dissertation, each with the 12 normal models:
#  1. Fixed split: one 80/20 split (seed 1), each model fitted 20 times on it.
#     This shows how stable selection is across MCMC runs.
#  2. Random splits: 20 different 80/20 splits (seeds 1 to 20), one fit each.
#     This also shows how much the results depend on the split.
# X and y are standardised with the training means and sds each time.
#
# Change since submission: predictions include the intercept beta0 (negligible
# here, since y_train has mean 0, but it keeps every real-data script the same).
#
# Change since submission: the variable `B` (column `black`) is dropped. It is
# derived from the proportion of Black residents by town and is widely
# criticised for that reason (scikit-learn removed the dataset). Set
# DROP_B <- FALSE to reproduce the dissertation's 13-variable analysis.

if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source("R/models.R")
source("R/evaluate.R")
source("R/real_data.R")

options(mc.cores = parallel::detectCores())
dir.create("results", showWarnings = FALSE)

DROP_B <- TRUE
N_REP  <- 20   # set to 1 for a quick check

data("Boston", package = "MASS")
y <- Boston$medv                               # median home value ($1000s)
X <- Boston[, setdiff(names(Boston), c("medv", if (DROP_B) "black"))]
suffix <- if (DROP_B) "_noB" else ""

# 1. Fixed split
split <- split_and_standardise(X, y, train_frac = 0.8, seed = 1, standardise_y = TRUE)
res_fixed <- run_real_data(normal_models, split, family = "normal", n_rep = N_REP,
                           out_file = paste0("results/real_boston_fixed", suffix, ".rds"))
print(summarise_real(res_fixed))
print(round(res_fixed$selection_props, 2))

# 2. Random splits
res_random <- run_real_data_random_splits(normal_models, X, y, family = "normal",
                                          n_rep = N_REP, standardise_y = TRUE,
                                          out_file = paste0("results/real_boston_random", suffix, ".rds"))
print(summarise_real(res_random))
print(round(res_random$selection_props, 2))
