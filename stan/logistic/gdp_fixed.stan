data {
  int<lower=0> n;  
  int<lower=0> p;  
  matrix[n, p] x; 
  array[n] int<lower=0,upper=1> y;
  
}

parameters {
  real beta0;           // Intercept
  vector[p] beta;                 // Regression coefficients
  vector<lower=0>[p] lambda0;      // Local shrinkage parameters
  vector<lower=0>[p] lambda1;
  real<lower=0> tau0;         // Auxiliary parameters
  real<lower=0> tau1;
  real<lower=0> sigma;            // Residual standard deviation
  array[p] real<lower=0, upper=1> gamma_prob; // Probability that gamma[j] = 1
}

transformed parameters{
  real<lower=0> sigma2;
  real<lower = 0> tau0_2;
  real<lower = 0> tau1_2;
  sigma2 = sigma^2;
  tau0_2 = tau0^2;
  tau1_2 = tau1^2;
}


model {
  // Priors
  // GDP prior parameterization
  tau0 ~ gamma(10 , 1);
  tau1 ~ gamma(1 , 10);
  // Prior for intercept and residual standard deviation
  beta0 ~ normal(0, 10);
  sigma2 ~ inv_gamma(0.5, 0.5);
  
  for (j in 1:p) {
    lambda0[j] ~ exponential(tau0_2 / 2);
    lambda1[j] ~ exponential(tau1_2 / 2);
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
    y[i] ~ bernoulli_logit(beta0 + dot_product(beta, x[i,]));
  }
}



