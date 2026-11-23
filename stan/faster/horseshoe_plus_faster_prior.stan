data {
  int<lower=0> n;              // Number of observations
  int<lower=0> p;              // Number of predictors
  matrix[n, p] x;              // Design matrix
  vector[n] y;                 // Response vector
}

parameters {
  real beta0;                           // Intercept
  vector[p] beta;                       // Regression coefficients
  real<lower=0> sigma;                  // Residual SD

  // Hyper-local scalings
  vector<lower=0>[p] eta_spike;
  vector<lower=0>[p] eta_slab;

  // Global scales
  real<lower=0> tau_spike;
  real<lower=0> tau_slab;

  // Inclusion probabilities
  vector<lower=0, upper=1>[p] pi;

  // Shared local standard deviation (mixture)
  vector<lower=0>[p] lambda;
}

transformed parameters {
  real<lower=0> sigma2 = square(sigma);
}

model {
  // Global priors
  beta0 ~ normal(0, 10);
  sigma2 ~ inv_gamma(0.5, 0.5);
  tau_spike ~ cauchy(0, 1);
  tau_slab ~ cauchy(0, 10);

  for (j in 1:p) {
    // Hyper-local priors
    eta_spike[j] ~ cauchy(0, 1);
    eta_slab[j] ~ cauchy(0, 1);
    pi[j] ~ beta(1, 1);

    // Spike-and-slab prior on λ_j
    target += log_mix(pi[j],
                      cauchy_lpdf(lambda[j] | 0, tau_slab * eta_slab[j]),
                      cauchy_lpdf(lambda[j] | 0, tau_spike * eta_spike[j]));

    // Prior on β_j
    beta[j] ~ normal(0, lambda[j]);
  }

  // Likelihood
  y ~ normal(beta0 + x * beta, sigma);
}
