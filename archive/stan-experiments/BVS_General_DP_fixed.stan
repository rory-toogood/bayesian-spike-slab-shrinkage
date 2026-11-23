data {
  int<lower=0> N;                 // Number of observations
  int<lower=0> p;                 // Number of predictors
  matrix[N, p] X;                 // Design matrix
  vector[N] y;                    // Response variable
  real<lower=0> alpha;            // Shape parameter for GDP (controls tail behavior)
  real<lower=0> eta;              // Scale parameter for GDP    
}

parameters {
  real beta0;           // Intercept
  vector[p] beta;                 // Regression coefficients
  vector<lower=0>[p] lambda;      // Local shrinkage parameters
  vector<lower=0>[p] tau;         // Auxiliary parameters
  real<lower=0> sigma;            // Residual standard deviation
  real<lower=0, upper=1> gamma_prob[p]; // Probability that gamma[j] = 1
}

transformed parameters{
  real<lower=0> sigma2;
  real<lower=0> lambda0_2[p];
  real<lower = 0> lambda1_2[p];
  sigma2 = sigma^2;
  lambda0_2 = lambda0^2;
  lambda1_2 = lambda1^2;
}


model {
  // Priors
  // GDP prior parameterization
  lambda ~ exponential(tau .* tau / 2);
  tau ~ gamma(alpha, eta);
  
  // Prior for intercept and residual standard deviation
  alpha_intercept ~ normal(0, 10);
  sigma ~ cauchy(0, 5);
  
  for (j in 1:p) {
    gamma_prob[j] ~ beta(1,1); 
  }

  // Prior for beta (spike-and-slab)
  for (j in 1:p) {
    target += log_mix(gamma_prob[j],
                      normal_lpdf(beta[j] | 0, lambda1[j]), // slab (when beta_j included)
                      normal_lpdf(beta[j] | 0, lambda0[j])); // spike
  }

  // Likelihood
  for (i in 1:n) {
    y[i] ~ normal(beta0 + dot_product(beta, x[i, :]), sigma);
  }
}



