# Pima Indians Diabetes (logistic regression): dissertation Section 5.2.1.
# Run from the repository root:  Rscript run_real_data_pima.R
#
# Design as in the dissertation: one 80/20 train/test split (seed 1), then each
# of the 12 logistic models is fitted 20 times on that split.
# Change since submission: predictions now include the intercept beta0 (the
# original script left it out), so test/train log-likelihoods differ from
# Appendix C.2. Variable selection is unaffected.

if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source("R/models.R")
source("R/evaluate.R")
source("R/real_data.R")

options(mc.cores = parallel::detectCores())
dir.create("results", showWarnings = FALSE)

data("PimaIndiansDiabetes", package = "mlbench")
y <- as.integer(PimaIndiansDiabetes$diabetes == "pos")
X <- PimaIndiansDiabetes[, setdiff(names(PimaIndiansDiabetes), "diabetes")]

split <- split_and_standardise(X, y, train_frac = 0.8, seed = 1)

N_REP <- 20   # set to 1 for a quick check
res <- run_real_data(glm_models("logistic"), split, family = "logistic",
                     n_rep = N_REP, out_file = "results/real_pima.rds")

print(summarise_real(res))          # mean test/train log-likelihood, d_hat, time
print(round(res$selection_props, 2)) # share of runs selecting each variable
