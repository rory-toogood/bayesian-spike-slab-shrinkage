# Fit one Stan model to one dataset and score its variable selection.

library(rstan)
rstan_options(auto_write = TRUE)   # cache compiled models between runs

#' Posterior inclusion probability (PIP) for each predictor.
#'
#' As in the dissertation, PIP_j is the share of posterior draws in which the
#' marginalised inclusion probability pi_j is at least 0.5.
inclusion_probability <- function(fit, incl = "gamma_prob") {
  draws <- rstan::extract(fit, pars = incl)[[1]]   # draws x p
  colMeans(draws >= 0.5)
}

#' Fit `model` to `dataset` and return TPR, TNR and run time.
#'
#' @param model An entry from models.R: list(file, incl).
#' @param dataset Output of generate_bvs_data(), or any list with X, y, beta.
#' @param threshold Select predictor j when PIP_j >= threshold (0.5 = median
#'   probability model).
fit_and_score <- function(model, dataset, threshold = 0.5, seed = 1, ...) {
  X <- dataset$X
  fit <- rstan::stan(file = model$file,
                     data = list(x = X, y = c(dataset$y), n = nrow(X), p = ncol(X)),
                     refresh = 0, seed = seed, ...)   # fixed seed: reruns give identical draws

  selected <- inclusion_probability(fit, model$incl) >= threshold
  relevant <- abs(dataset$beta) > 0

  data.frame(
    TPR  = mean(selected[relevant]),          # share of true signals found
    TNR  = 1 - mean(selected[!relevant]),     # share of noise variables excluded
    Time = sum(rstan::get_elapsed_time(fit))  # warm-up + sampling, all chains (s)
  )
}

#' Run every model in `models` on every dataset in `scenarios`.
#' Returns one long data frame (scenario x method). Results are saved after each
#' scenario so a long run can be resumed.
run_grid <- function(scenarios, models, out_file = NULL, ...) {
  rows <- list()
  for (i in seq_along(scenarios)) {
    name <- names(scenarios)[i]
    for (m in names(models)) {
      res <- fit_and_score(models[[m]], scenarios[[name]], ...)
      rows[[length(rows) + 1]] <- cbind(scenario = name, method = m, res)
      message(sprintf("[%d/%d] %s  %s  TPR=%.2f TNR=%.2f  %.1fs",
                      i, length(scenarios), name, m, res$TPR, res$TNR, res$Time))
    }
    if (!is.null(out_file)) saveRDS(do.call(rbind, rows), out_file)
  }
  do.call(rbind, rows)
}

#' Mean TPR / TNR / Time per method, as in the dissertation's summary tables.
summarise_results <- function(results) {
  aggregate(cbind(TPR, TNR, Time) ~ method, data = results, FUN = mean)
}
