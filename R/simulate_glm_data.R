# Simulated logistic and Poisson regression data (Section 4.2-4.3).
# The linear predictor is rescaled to have sd = eta_sd, which controls how
# strong the signal is on the link scale.

library(MASS)

bimodal_beta <- function(p, d) {
  beta <- rep(0, p)
  active_idx <- sample(1:p, d, replace = FALSE)
  beta[active_idx] <- ifelse(runif(d) < 0.5,
                             ifelse(runif(d) < 0.5, rnorm(d, 1, 0.3), rnorm(d, -1, 0.3)),
                             ifelse(runif(d) < 0.5, rnorm(d, 3, 1),   rnorm(d, -3, 1)))
  beta
}

generate_glm_bvs_data <- function(n, p, d, family = c("logistic", "poisson"),
                                  eta_sd = 1.5, seed = NULL) {
  family <- match.arg(family)
  if (!is.null(seed)) set.seed(seed)
  X <- apply(MASS::mvrnorm(n = n, mu = rep(0, p), Sigma = diag(p)), 2, scale)
  beta <- bimodal_beta(p, d)
  eta <- as.numeric(scale(X %*% beta)) * eta_sd

  y <- if (family == "logistic") {
    rbinom(n, 1, plogis(eta))
  } else {
    rpois(n, exp(pmin(pmax(eta, -4), 4)))   # clip eta so lambda can't explode
  }
  list(X = X, y = y, beta = beta, eta_sd = eta_sd)
}

glm_setups <- list(
  list(n = 50,  p = 10, d = 3), list(n = 50,  p = 10, d = 5),
  list(n = 100, p = 10, d = 3), list(n = 100, p = 10, d = 5),
  list(n = 50,  p = 20, d = 5), list(n = 50,  p = 20, d = 10),
  list(n = 100, p = 20, d = 5), list(n = 100, p = 20, d = 10)
)

#' 8 (n, p, d) settings x 3 signal strengths = 24 datasets per family.
#' Seeds match the dissertation runs (1000 + j logistic, 2000 + j Poisson).
build_glm_scenarios <- function(family = c("logistic", "poisson"),
                                eta_sds = c(0.5, 1.5, 2.5)) {
  family <- match.arg(family)
  base_seed <- if (family == "logistic") 1000 else 2000
  scenarios <- list(); j <- 1
  for (s in glm_setups) for (eta_sd in eta_sds) {
    name <- paste0("n", s$n, "_p", s$p, "_d", s$d, "_etaSD", eta_sd)
    scenarios[[name]] <- generate_glm_bvs_data(s$n, s$p, s$d, family, eta_sd,
                                               seed = base_seed + j)
    j <- j + 1
  }
  scenarios
}
