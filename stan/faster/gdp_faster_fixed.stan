data {
  int<lower=0> n;                  // number of observations
  int<lower=0> p;                  // number of predictors
  matrix[n, p] x;                  // design matrix
  vector[n] y;                     // response vector
}

parameters {
  real beta0;                      // intercept
  vector[p] beta;                  // regression coefficients
  vector<lower=0>[p] tau;          // local scale (standard deviation)
  vector<lower=0>[p] lambda;       // rate parameter for exponential prior on tau
  real<lower=0> sigma;             // residual standard deviation
  vector<lower=0, upper=1>[p] gamma_prob; // inclusion probabilities π_j
}

transformed parameters {
  real<lower=0> sigma2 = square(sigma);
}

model {
  // Hyperpriors
  beta0 ~ normal(0, 10);
  sigma2 ~ inv_gamma(0.5, 0.5);

  for (j in 1:p) {
    // Mixture prior for λ_j
    target += log_mix(gamma_prob[j],
                      gamma_lpdf(lambda[j] | 1.0, 10.0),   // slab component
                      gamma_lpdf(lambda[j] | 10.0, 1.0));  // spike component

    // τ_j ~ Exponential(λ_j^2 / 2)
    tau[j] ~ exponential(square(lambda[j]) / 2);

    // β_j ~ N(0, τ_j)
    beta[j] ~ normal(0, tau[j]);

    // Prior for π_j
    gamma_prob[j] ~ beta(1, 1);
  }

  // Likelihood
  y ~ normal(beta0 + x * beta, sigma);
}
