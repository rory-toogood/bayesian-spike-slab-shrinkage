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

  vector<lower=0>[p] lambda;            // Local scale λ_j
  vector<lower=0>[p] tau;               // τ_j
  vector<lower=0, upper=1>[p] gamma_prob;       // Inclusion probabilities π_j
}

transformed parameters {
  real<lower=0> sigma2 = square(sigma);  // Residual variance
}

model {
  // Global priors
  beta0 ~ normal(0, 10);
  sigma2 ~ inv_gamma(0.5, 0.5);

  for (j in 1:p) {
    // Mixture prior on τ_j
    target += log_mix(gamma_prob[j],
                      cauchy_lpdf(tau[j] | 0, 10 * sigma),
                      cauchy_lpdf(tau[j] | 0, sigma / 10));

    // Prior on λ_j | τ_j
    lambda[j] ~ cauchy(0, tau[j]);

    // Prior on β_j | λ_j
    beta[j] ~ normal(0, lambda[j]);
  }

  // Prior on inclusion probs
  gamma_prob ~ beta(1, 1);

  // Likelihood
  y ~ normal(beta0 + x * beta, sigma);
}
