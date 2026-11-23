# Registry of the Stan models used in the dissertation.
# Every model takes data list(n, p, x, y). `incl` is the name of the vector of
# inclusion probabilities (pi_j) in that model.

stan_dir <- function(...) file.path("stan", ...)

normal_models <- list(
  SSVS_fixed               = list(file = stan_dir("normal", "ssvs_fixed.stan"),               incl = "gamma_prob"),
  SSVS_prior               = list(file = stan_dir("normal", "ssvs_prior.stan"),               incl = "gamma_prob"),
  BLASSO_fixed             = list(file = stan_dir("normal", "blasso_fixed.stan"),             incl = "gamma_prob"),
  BLASSO_prior             = list(file = stan_dir("normal", "blasso_prior.stan"),             incl = "gamma_prob"),
  Horseshoe_fixed          = list(file = stan_dir("normal", "horseshoe_fixed.stan"),          incl = "gamma_prob"),
  Horseshoe_prior          = list(file = stan_dir("normal", "horseshoe_prior.stan"),          incl = "gamma_prob"),
  HorseshoePlus_fixed      = list(file = stan_dir("normal", "horseshoe_plus_fixed.stan"),     incl = "gamma_prob"),
  HorseshoePlus_prior      = list(file = stan_dir("normal", "horseshoe_plus_prior.stan"),     incl = "gamma_prob"),
  DirichletLaplace_fixed   = list(file = stan_dir("normal", "dirichlet_laplace_fixed.stan"),  incl = "gamma_prob"),
  DirichletLaplace_prior   = list(file = stan_dir("normal", "dirichlet_laplace_prior.stan"),  incl = "gamma_prob"),
  GDP_fixed                = list(file = stan_dir("normal", "gdp_fixed.stan"),                incl = "gamma_prob"),
  GDP_prior                = list(file = stan_dir("normal", "gdp_prior.stan"),                incl = "gamma_prob")
)

faster_models <- list(
  BLASSO_faster_prior        = list(file = stan_dir("faster", "blasso_faster_prior.stan"),        incl = "gamma_prob"),
  BLASSO_fasterfaster_prior  = list(file = stan_dir("faster", "blasso_fasterfaster_prior.stan"),  incl = "gamma_prob"),
  Horseshoe_fasterfaster     = list(file = stan_dir("faster", "horseshoe_fasterfaster_prior.stan"), incl = "gamma_prob"),
  HorseshoePlus_faster_fixed = list(file = stan_dir("faster", "horseshoe_plus_faster_fixed.stan"), incl = "pi"),
  HorseshoePlus_faster_prior = list(file = stan_dir("faster", "horseshoe_plus_faster_prior.stan"), incl = "pi"),
  GDP_faster_fixed           = list(file = stan_dir("faster", "gdp_faster_fixed.stan"),           incl = "gamma_prob"),
  GDP_faster_prior           = list(file = stan_dir("faster", "gdp_faster_prior.stan"),           incl = "gamma_prob")
)

glm_models <- function(family = c("logistic", "poisson")) {
  family <- match.arg(family)
  stems <- c(SSVS = "ssvs", BLASSO = "blasso", Horseshoe = "horseshoe",
             HorseshoePlus = "horseshoe_plus", DirichletLaplace = "dirichlet_laplace",
             GDP = "gdp")
  out <- list()
  for (nm in names(stems)) for (setup in c("fixed", "prior")) {
    out[[paste0(nm, "_", setup)]] <- list(
      file = stan_dir(family, paste0(stems[[nm]], "_", setup, ".stan")),
      incl = "gamma_prob")
  }
  out
}
