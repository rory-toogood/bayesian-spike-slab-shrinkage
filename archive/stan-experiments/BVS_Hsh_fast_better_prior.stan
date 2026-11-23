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

  vector<lower=0>[p] lambda;            // Local shrinkage λ_j
  vector<lower=0>[p] tau_slab;          // Slab scale τ_j,slab
  vector<lower=0>[p] tau_spike;         // Spike scale τ_j,spike

  vector<lower=0, upper=1>[p] gamma_prob;       // Inclusion probabilities π_j
}

transformed parameters {
  real<lower=0> sigma2 = square(sigma);  // Residual variance
}

model {
  // Priors on global quantities
  beta0 ~ normal(0, 10);
  sigma2 ~ inv_gamma(0.5, 0.5);

  // Priors on tau_slab and tau_spike: half-Cauchy(0, sigma)
  for (j in 1:p) {
    tau_slab[j] ~ cauchy(0, 10*sigma);
    tau_spike[j] ~ cauchy(0, sigma/10);
  }

  // Priors on lambda_j: mixture of half-Cauchys
  for (j in 1:p) {
    target += log_mix(gamma_prob[j],
                      cauchy_lpdf(lambda[j] | 0, tau_slab[j]),
                      cauchy_lpdf(lambda[j] | 0, tau_spike[j]));
  }

  // Priors on pi_j
  gamma_prob ~ beta(1, 1);

  // Prior on beta_j | lambda_j
  for (j in 1:p) {
    beta[j] ~ normal(0, lambda[j]);
  }

  // Likelihood
  y ~ normal(beta0 + x * beta, sigma);
}
