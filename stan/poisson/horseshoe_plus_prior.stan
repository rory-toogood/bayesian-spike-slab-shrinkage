data {
  int<lower=0> n;  
  int<lower=0> p;  
  matrix[n, p] x; 
  array[n] int<lower=0> y;
}

parameters {
  vector[p] beta;      
  real beta0;          
  real<lower=0> sigma;  
  array[p] real <lower = 0> lambda0;
  array[p] real <lower = 0> lambda1;
  array[p] real <lower = 0> eta0;
  array[p] real <lower = 0> eta1;
  real <lower = 0> tau0;
  real <lower = 0> tau1;
  array[p] real<lower=0, upper=1> gamma_prob; // Probability that gamma[j] = 1
}

transformed parameters{
  real<lower=0> sigma2;
  sigma2 = sigma^2;
}


model {
  // Priors
  beta0 ~ normal(0,10);
  sigma2 ~ inv_gamma(0.5, 0.5);
  tau0 ~ cauchy(0, 1);  // Varying this multiplication factor varies how much the 
  tau1 ~ cauchy(0, 10);
  for (j in 1:p) {
    eta0[j] ~ cauchy(0,1);
    eta1[j] ~ cauchy(0,1);
    lambda0[j] ~ cauchy(0, tau0*eta0[j]);
    lambda1[j] ~ cauchy(0, tau1*eta1[j]);
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
    y[i] ~ poisson_log(beta0 + dot_product(beta, x[i,]));
  }
}


