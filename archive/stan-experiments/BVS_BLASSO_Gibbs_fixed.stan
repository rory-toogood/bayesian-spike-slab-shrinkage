data {
  int<lower=0> n;              // Number of observations
  int<lower=0> p;              // Number of predictors
  matrix[n, p] x;              // Design matrix
  vector[n] y;                 // Response vector
}

parameters {
  real beta0;                  // Intercept
  vector[p] beta;              // Regression coefficients
  real<lower=0> sigma;         // Residual SD
  
  // Spike and slab scales
  vector<lower=0>[p] tau_slab2;   // slab variances
  vector<lower=0>[p] tau_spike2;  // spike variances

  // Mixing probabilities
  vector<lower=0, upper=1>[p] gamma_prob; // π_j
}

transformed parameters {
  real<lower=0> sigma2 = square(sigma);
}

model {
  // Priors
  beta0 ~ normal(0, 10);
  sigma2 ~ inv_gamma(0.5, 0.5);
  
  for (j in 1:p) {
    tau_slab2[j] ~ exponential(1 / 4);
    tau_spike2[j] ~ exponential(10 / 2);
    gamma_prob[j] ~ beta(1, 1);
  }

  // Spike-and-slab prior for beta_j as mixture of Normals
  for (j in 1:p) {
    target += log_mix(gamma_prob[j],
                      normal_lpdf(beta[j] | 0, sqrt(tau_slab2[j])),
                      normal_lpdf(beta[j] | 0, sqrt(tau_spike2[j])));
  }

  // Likelihood
  y ~ normal(beta0 + x * beta, sigma);
}