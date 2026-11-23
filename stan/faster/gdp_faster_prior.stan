data {
  int<lower=0> n;              // Number of observations
  int<lower=0> p;              // Number of predictors
  matrix[n, p] x;              // Design matrix
  vector[n] y;                 // Response vector
}

parameters {
  real beta0;                  // Intercept
  vector[p] beta;              // Regression coefficients
  vector<lower=0>[p] tau;      // Variance scale for beta_j: tau_j
  vector<lower=0>[p] lambda_slab;  // Slab scale (from Gamma)
  vector<lower=0>[p] lambda_spike; // Spike scale (from Gamma)
  real<lower=0> sigma;         // Residual SD

  real<lower=0> alpha_slab;
  real<lower=0> alpha_spike;
  real<lower=0> eta_slab;
  real<lower=0> eta_spike;

  vector<lower=0, upper=1>[p] gamma_prob;  // Inclusion probabilities π_j
}

transformed parameters {
  real<lower=0> sigma2 = square(sigma);
}

model {
  // Priors
  beta0 ~ normal(0, 10);
  sigma2 ~ inv_gamma(0.5, 0.5);

  alpha_slab ~ gamma(2, 2);
  eta_slab ~ gamma(3, 0.5);
  alpha_spike ~ gamma(3, 0.5);
  eta_spike ~ gamma(2, 2);

  for (j in 1:p) {
    // Sample slab/spike lambda values
    lambda_slab[j] ~ gamma(alpha_slab, eta_slab);
    lambda_spike[j] ~ gamma(alpha_spike, eta_spike);
    gamma_prob[j] ~ beta(1, 1);

    // Prior on τ_j using log_mix over exponential(lambda²/2)
    target += log_mix(gamma_prob[j],
                      exponential_lpdf(tau[j] | square(lambda_slab[j]) / 2),
                      exponential_lpdf(tau[j] | square(lambda_spike[j]) / 2));

    // Then, prior on beta_j given τ_j
    beta[j] ~ normal(0, sqrt(tau[j]));
  }

  // Likelihood
  y ~ normal(beta0 + x * beta, sigma);
}
