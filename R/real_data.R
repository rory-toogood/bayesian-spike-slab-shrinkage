# Real-data evaluation: fit each model on a training set, select variables,
# and score predictions on the held-out test set.
#
# Used by run_real_data_pima.R (logistic), run_real_data_gala.R (Poisson) and
# run_real_data_riboflavin.R and run_real_data_boston.R (normal), through `family`.

# One random train/test split. X (and, if asked, y) is standardised with the
# training means and standard deviations only, so no test information leaks
# into the fit. `top_var` keeps only the columns with the largest variance,
# ranked on the raw training rows (used for Riboflavin).
split_and_standardise <- function(X, y, train_frac = 0.8, seed = 1,
                                  standardise_y = FALSE, top_var = NULL) {
  set.seed(seed)
  X <- as.matrix(X)
  n <- nrow(X)
  train <- sample(seq_len(n), floor(train_frac * n))
  test  <- setdiff(seq_len(n), train)

  if (!is.null(top_var)) {
    v <- apply(X[train, , drop = FALSE], 2, var)
    X <- X[, order(v, decreasing = TRUE)[seq_len(top_var)], drop = FALSE]
  }

  centre <- colMeans(X[train, , drop = FALSE])
  spread <- apply(X[train, , drop = FALSE], 2, sd)
  y_train <- y[train]; y_test <- y[test]
  if (standardise_y) {
    m <- mean(y_train); s <- sd(y_train)
    y_train <- (y_train - m) / s
    y_test  <- (y_test  - m) / s
  }
  list(X_train = scale(X[train, , drop = FALSE], centre, spread),
       X_test  = scale(X[test,  , drop = FALSE], centre, spread),
       y_train = y_train, y_test = y_test)
}

# Log-likelihood of y given the linear predictor eta.
glm_loglik <- function(y, eta, family = c("logistic", "poisson")) {
  family <- match.arg(family)
  if (family == "logistic") sum(dbinom(y, 1, plogis(eta), log = TRUE))
  else                      sum(dpois(y, exp(eta), log = TRUE))
}

# Fit one model to one split and score it.
# Prediction uses the posterior means of the intercept and of the selected
# coefficients: eta = beta0_hat + X[, selected] %*% beta_hat[selected].
# If nothing is selected, the prediction is the intercept alone.
fit_and_score_real <- function(model, split, family, threshold = 0.5, seed = 1, ...) {
  fit <- rstan::stan(file = model$file,
                     data = list(x = split$X_train, y = c(split$y_train),
                                 n = nrow(split$X_train), p = ncol(split$X_train)),
                     refresh = 0, seed = seed, ...)
  draws    <- rstan::extract(fit, pars = c("beta0", "beta", model$incl))
  selected <- colMeans(draws[[model$incl]] >= 0.5) >= threshold

  beta0_hat <- mean(draws$beta0)
  beta_hat  <- colMeans(draws$beta)
  beta_hat[!selected] <- 0          # unselected variables drop out

  eta_train <- beta0_hat + c(split$X_train %*% beta_hat)
  eta_test  <- beta0_hat + c(split$X_test  %*% beta_hat)

  time <- sum(rstan::get_elapsed_time(fit))
  # Convergence checks, so unreliable fits are visible in the results
  diag <- data.frame(max_Rhat     = max(summary(fit)$summary[, "Rhat"], na.rm = TRUE),
                     divergent    = rstan::get_num_divergent(fit),
                     max_treedepth_hits = rstan::get_num_max_treedepth(fit))
  scores <- if (family == "normal") {
    data.frame(test_MSE  = mean((split$y_test  - eta_test)^2),
               train_MSE = mean((split$y_train - eta_train)^2),
               d_hat = sum(selected), Time = time)
  } else {
    data.frame(test_ll  = glm_loglik(split$y_test,  eta_test,  family),
               train_ll = glm_loglik(split$y_train, eta_train, family),
               d_hat = sum(selected), Time = time)
  }
  list(scores = cbind(scores, diag), selected = selected)
}

# Repeat every model `n_rep` times on the same split (as in the dissertation:
# the split is fixed, only the MCMC run changes). Returns per-run scores and,
# for each model, the share of runs that selected each variable.
run_real_data <- function(models, split, family, n_rep = 20, out_file = NULL, ...) {
  var_names <- colnames(split$X_train)
  scores <- list()
  selection <- lapply(models, function(m)
    matrix(0L, n_rep, length(var_names), dimnames = list(NULL, var_names)))

  save_progress <- function(r) {
    done <- seq_len(r)
    out <- list(scores = do.call(rbind, scores),
                selection_props = t(sapply(selection, function(s) colMeans(s[done, , drop = FALSE]))))
    if (!is.null(out_file)) saveRDS(out, out_file)
    out
  }
  for (r in seq_len(n_rep)) {
    for (nm in names(models)) {
      res <- fit_and_score_real(models[[nm]], split, family, seed = r, ...)
      scores[[length(scores) + 1]] <- cbind(method = nm, run = r, res$scores)
      selection[[nm]][r, ] <- as.integer(res$selected)
      message("Repeat ", r, ": ", nm, " done (", round(res$scores$Time), " s, max R-hat ",
              round(res$scores$max_Rhat, 2), ")")
      if (n_rep == 1) out <- save_progress(r)   # slow single runs: save after every model
    }
    out <- save_progress(r)
  }
  out
}

# A new random train/test split for every repeat (split seed = repeat number),
# so the results also reflect how much they depend on the split.
run_real_data_random_splits <- function(models, X, y, family, n_rep = 20, out_file = NULL,
                                        train_frac = 0.8, standardise_y = FALSE, ...) {
  var_names <- colnames(as.matrix(X))
  scores <- list()
  selection <- lapply(models, function(m)
    matrix(0L, n_rep, length(var_names), dimnames = list(NULL, var_names)))
  out <- NULL
  for (r in seq_len(n_rep)) {
    split <- split_and_standardise(X, y, train_frac = train_frac, seed = r,
                                   standardise_y = standardise_y)
    for (nm in names(models)) {
      res <- fit_and_score_real(models[[nm]], split, family, seed = r, ...)
      scores[[length(scores) + 1]] <- cbind(method = nm, run = r, res$scores)
      selection[[nm]][r, ] <- as.integer(res$selected)
      message("Split ", r, ": ", nm, " done (", round(res$scores$Time), " s, max R-hat ",
              round(res$scores$max_Rhat, 2), ")")
    }
    out <- list(scores = do.call(rbind, scores),
                selection_props = t(sapply(selection, function(s) colMeans(s[seq_len(r), , drop = FALSE]))))
    if (!is.null(out_file)) saveRDS(out, out_file)
  }
  out
}

summarise_real <- function(results) {
  sc <- results$scores
  cols <- setdiff(names(sc), c("method", "run"))
  out <- aggregate(sc[cols], by = list(method = sc$method), FUN = mean)
  out[match(unique(sc$method), out$method), ]   # keep the models' order
}
