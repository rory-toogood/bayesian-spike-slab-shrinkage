# Reproduce the simulation study (Chapter 4).
# Run from the repository root:  Rscript run_simulations.R
#
# The full study is many hours of MCMC. Leave QUICK <- TRUE first to check
# everything works on a few small settings.

QUICK <- TRUE

# Work from the repository root so the R/ and stan/ paths resolve. In RStudio
# this switches to the folder this script is in; otherwise run setwd() yourself.
if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source("R/simulate_data.R")
source("R/simulate_glm_data.R")
source("R/models.R")
source("R/evaluate.R")

options(mc.cores = parallel::detectCores())
dir.create("results", showWarnings = FALSE)
take <- function(s, k = 3) if (QUICK) s[seq_len(min(k, length(s)))] else s

# 1. Normal regression, uncorrelated predictors: 252 settings x 12 models
scenarios <- build_scenarios()
main <- run_grid(take(scenarios), normal_models, out_file = "results/sim_normal.rds")
print(summarise_results(main))

# 2. Faster variants on the mixed-signal ("bimodal") settings (Table 4.9)
mixed <- scenarios[grepl("_bimodal$", names(scenarios))]
fast <- run_grid(take(mixed), c(normal_models["SSVS_fixed"], faster_models),
                 out_file = "results/sim_faster.rds")
print(summarise_results(fast))

# 3. Correlated predictors (inverse-Wishart covariance): 54 settings
corr <- run_grid(take(build_correlated_scenarios()), normal_models,
                 out_file = "results/sim_correlated.rds")
print(summarise_results(corr))

# 4. Logistic and Poisson regression: 24 settings each
for (family in c("logistic", "poisson")) {
  res <- run_grid(take(build_glm_scenarios(family)), glm_models(family),
                  out_file = sprintf("results/sim_%s.rds", family))
  print(summarise_results(res))
}

# Real-data analyses (Boston, Riboflavin, Pima, Gala) are in run_real_data_*.R
