# Re-time the BLASSO faster variants on the 63 mixed-signal ("bimodal") settings
# used for Table 4.9. Run from the repository root:  Rscript retime_blasso_faster.R
#
# Each fit is timed separately (fit_and_score() reads the time from the fit it
# just ran). BLASSO faster is included as a control: if its mean is close to
# the 32.5 s from the dissertation, your machine runs at about the same speed
# as in 2025, so the faster-faster time can be compared directly with 102.6 s.

# Work from the repository root so the R/ and stan/ paths resolve. In RStudio
# this switches to the folder this script is in; otherwise run setwd() yourself.
if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source("R/simulate_data.R")
source("R/models.R")
source("R/evaluate.R")

options(mc.cores = parallel::detectCores())
dir.create("results", showWarnings = FALSE)

scenarios <- build_scenarios()
mixed <- scenarios[grepl("_bimodal$", names(scenarios))]
stopifnot(length(mixed) == 63)

res <- run_grid(mixed,
                faster_models[c("BLASSO_faster_prior", "BLASSO_fasterfaster_prior")],
                out_file = "results/retime_blasso_faster.rds")

print(summarise_results(res))   # mean TPR, TNR, Time per method

# Sanity check: every method should have many different run times
print(tapply(res$Time, res$method, function(t) length(unique(t))))

# ---------------------------------------------------------------------------
# Re-run only the slowest settings (keep the Mac awake and plugged in: in
# Terminal run `caffeinate -i` while this runs, or System Settings > Battery >
# prevent automatic sleeping).
# heavy <- mixed[grepl("^n1000_p200|^n30_p200_d10", names(mixed))]
# res_heavy <- run_grid(heavy, faster_models[c("BLASSO_faster_prior", "BLASSO_fasterfaster_prior")],
#                       out_file = "results/retime_blasso_heavy.rds")
# print(res_heavy)
