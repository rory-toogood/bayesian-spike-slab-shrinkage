data {
  int<lower=0> n;              // Number of observations
  int<lower=0> p;              // Number of predictors
  matrix[n, p] x;              // Design matrix
  vector[n] y;                 // Response vector
}

parameters {
  real beta0;                       // Intercept
  vector[p] beta;                   // Regression coefficients
  real<lower=0> sigma;              // Residual SD

  vector<lower=0>[p] tau2;          // Local variances (τ_j²)
  vector<lower=0>[p] lambda2;       // λ_j²
  vector<lower=0, upper=1>[p] gamma_prob;   // π_j
}

transformed parameters {
  real<lower=0> sigma2 = square(sigma);
}

model {
  // Priors
  beta0 ~ normal(0, 10);
  sigma2 ~ inv_gamma(0.5, 0.5);

  for (j in 1:p) {
    // Prior on tau²_j | lambda²_j
    tau2[j] ~ exponential(lambda2[j] / 2);

    // Mixture prior on lambda²_j
    target += log_mix(gamma_prob[j],
                      gamma_lpdf(lambda2[j] | 0.5, 0.5),
                      gamma_lpdf(lambda2[j] | 5.0, 0.5));

    // Prior on beta_j | tau²_j
    beta[j] ~ normal(0, sqrt(tau2[j]));

    // Prior on π_j
    gamma_prob[j] ~ beta(1, 1);
  }

  // Likelihood
  y ~ normal(beta0 + x * beta, sigma);
}
