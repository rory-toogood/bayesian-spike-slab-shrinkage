data {
  int<lower=0> n;  
  int<lower=0> p;  
  matrix[n, p] x; 
  vector[n] y;     
}

parameters {
  vector[p] beta;      
  real beta0;
  real<lower=0> tau[p]; // Prior scale for gamma[j] == 1
  real<lower=1> c[p]; // Prior scale (when multiplied by tau) for gamma[j] == 0, lower = 1 as want slab fatter than spike
  real<lower=0> sigma;  
  vector<lower=0, upper=1>[p] gamma_prob; // Probability that gamma[j] = 1
  real<lower=0, upper=1> gamma_prob_global; // Global inclusion probability
}

transformed parameters {
  real<lower= 0> sigma2;
  sigma2 = sigma^2;
}
model {
  // Priors
  sigma2 ~ inv_gamma(0.5, 0.5);
  gamma_prob_global ~ beta(1, 1); // Global inclusion probability
  
  for (j in 1:p) {
    gamma_prob[j] ~ beta(10* gamma_prob_global, 10 * (1 - gamma_prob_global)); // theta j 
  }

  // Prior for beta (spike-and-slab)
  for (j in 1:p) {
    target += log_mix(gamma_prob[j],
                      normal_lpdf(beta[j] | 0, c[j] * tau[j]),
                      normal_lpdf(beta[j] | 0, tau[j]));
  }

  // Likelihood
  for (i in 1:n) {
    y[i] ~ normal(beta0 + dot_product(beta, x[i, :]), sigma);
  }
}
