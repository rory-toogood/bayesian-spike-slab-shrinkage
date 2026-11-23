# Simulated data for Bayesian variable selection experiments.
# generate_bvs_data() is unchanged from the dissertation code; the scenario grid
# below reproduces the 252 uncorrelated normal-regression settings (Table 4.1).

library(MASS)

centre <- function(v) as.numeric(scale(v, center = TRUE, scale = FALSE))

#' Simulate one regression dataset with a known sparse coefficient vector.
#'
#' @param n Sample size.
#' @param p Number of predictors.
#' @param d Number of truly relevant predictors (non-zero betas).
#' @param signal_type One of "bimodal", "gaussian", "half_positive_negative",
#'   "extreme_sparse" (see the dissertation, Section 4.1).
#' @param snr Signal-to-noise ratio, var(X beta) / var(noise).
#' @param rho Equicorrelation between predictors (0 = uncorrelated).
#' @param seed Optional RNG seed.
generate_bvs_data <- function(n, p, d, signal_type = "bimodal", snr = 5,
                              rho = 0, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  # 1. Predictors: equicorrelated Gaussian, then standardised
  Sigma <- matrix(rho, nrow = p, ncol = p) + diag(1 - rho, p)
  X <- apply(MASS::mvrnorm(n = n, mu = rep(0, p), Sigma = Sigma), 2, scale)

  # 2. True coefficients: d active predictors chosen at random
  beta <- rep(0, p)
  active_idx <- sample(1:p, d, replace = FALSE)

  if (signal_type == "bimodal") {
    # Mixed strength: half weak (|beta| ~ 1), half strong (|beta| ~ 3), random signs
    beta[active_idx] <- ifelse(runif(d) < 0.5,
                               ifelse(runif(d) < 0.5, rnorm(d, 1, 0.3), rnorm(d, -1, 0.3)),
                               ifelse(runif(d) < 0.5, rnorm(d, 3, 1),   rnorm(d, -3, 1)))
  } else if (signal_type == "gaussian") {
    beta[active_idx] <- rnorm(d, mean = 0, sd = 3)
  } else if (signal_type == "half_positive_negative") {
    beta[active_idx] <- rnorm(d, mean = 3, sd = 1)
    beta[sample(active_idx, d %/% 2)] <- -beta[sample(active_idx, d %/% 2)]
  } else if (signal_type == "extreme_sparse") {
    beta[active_idx] <- ifelse(runif(d) < 0.5, rnorm(d, 5, 2), rnorm(d, -5, 2))
  } else {
    stop("Invalid signal_type. Choose from: 'bimodal', 'gaussian', ",
         "'half_positive_negative', 'extreme_sparse'.")
  }

  # 3. Response with noise scaled to hit the target SNR, then centred
  signal_variance <- var(X %*% beta)
  sigma_epsilon <- sqrt(signal_variance / snr)
  y <- centre(X %*% beta + rnorm(n, 0, sigma_epsilon))

  list(X = X, y = y, beta = beta, snr = snr, signal_type = signal_type, rho = rho)
}

# (n, p, d) settings used in the dissertation
simulation_setups <- list(
  list(n = 50,   p = 10,  d_values = c(3, 5)),
  list(n = 50,   p = 50,  d_values = c(3, 5, 8)),
  list(n = 100,  p = 50,  d_values = c(5, 8, 10)),
  list(n = 100,  p = 100, d_values = c(5, 10, 15)),
  list(n = 500,  p = 100, d_values = c(10, 15, 20)),
  list(n = 1000, p = 200, d_values = c(15, 20)),
  list(n = 30,   p = 200, d_values = c(10, 20)),
  list(n = 30,   p = 10,  d_values = c(3, 5, 8))
)
signal_types <- c("bimodal", "gaussian", "half_positive_negative", "extreme_sparse")
snrs <- c(1, 5, 10)

#' Build the full list of simulated datasets, named
#' "n{n}_p{p}_d{d}_snr{snr}_{signal}". 21 (n, p, d) x 4 signals x 3 SNRs = 252.
build_scenarios <- function(setups = simulation_setups, signals = signal_types,
                            snr_values = snrs, rho = 0, seed = 42) {
  scenarios <- list()
  for (setup in setups) {
    for (d in setup$d_values) {
      for (signal in signals) {
        for (snr in snr_values) {
          name <- paste0("n", setup$n, "_p", setup$p, "_d", d, "_snr", snr, "_", signal)
          scenarios[[name]] <- generate_bvs_data(setup$n, setup$p, d,
                                                 signal_type = signal, snr = snr,
                                                 rho = rho, seed = seed)
        }
      }
    }
  }
  scenarios
}

# ---------------------------------------------------------------------------
# Correlated predictors (Section 4.1.2): covariance drawn from an
# inverse-Wishart, with the degrees of freedom nu controlling how strong the
# correlation is (nu = p + 2 strong, p + 15 medium, p + 50 weak).

generate_bvs_inverse_wishart <- function(n, p, d, signal_type = "bimodal",
                                         snr = 5, nu = NULL, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  if (is.null(nu)) nu <- p + 2
  Sigma <- MCMCpack::riwish(v = nu, S = diag(p))
  X <- apply(MASS::mvrnorm(n = n, mu = rep(0, p), Sigma = Sigma), 2, scale)

  beta <- rep(0, p)
  active_idx <- sample(1:p, d, replace = FALSE)
  if (signal_type == "bimodal") {
    beta[active_idx] <- ifelse(runif(d) < 0.5,
                               ifelse(runif(d) < 0.5, rnorm(d, 1, 0.3), rnorm(d, -1, 0.3)),
                               ifelse(runif(d) < 0.5, rnorm(d, 3, 1),   rnorm(d, -3, 1)))
  } else {
    stop("Only 'bimodal' signals were used for the correlated simulations.")
  }

  sigma_epsilon <- sqrt(var(X %*% beta) / snr)
  y <- centre(X %*% beta + rnorm(n, 0, sigma_epsilon))
  list(X = X, y = y, beta = beta, snr = snr, Sigma = Sigma, signal_type = signal_type)
}

correlated_setups <- list(
  list(n = 50, p = 10,  d_values = c(3, 5, 8)),
  list(n = 50, p = 50,  d_values = c(5, 10, 15)),
  list(n = 30, p = 200, d_values = c(5, 10, 20))
)
correlation_levels <- list(high = function(p) p + 2,
                           medium = function(p) p + 15,
                           low = function(p) p + 50)

#' 3 settings x 3 d x 3 correlation levels x 2 SNRs = 54 datasets.
build_correlated_scenarios <- function(snr_values = c(2, 5)) {
  scenarios <- list(); id <- 1
  for (s in correlated_setups) for (d in s$d_values) for (lvl in names(correlation_levels)) {
    for (snr in snr_values) {
      name <- paste0("n", s$n, "_p", s$p, "_d", d, "_", lvl, "_snr", snr)
      scenarios[[name]] <- generate_bvs_inverse_wishart(
        s$n, s$p, d, snr = snr, nu = correlation_levels[[lvl]](s$p), seed = 1000 + id)
      id <- id + 1
    }
  }
  scenarios
}
