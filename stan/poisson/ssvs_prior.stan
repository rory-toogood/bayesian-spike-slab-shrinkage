data {
  int<lower=0> n;  
  int<lower=0> p;  
  matrix[n, p] x; 
  array[n] int<lower=0> y;
}

parameters {
  vector[p] beta;      
  real beta0;          
  array[p] real<lower=0> tau; // Prior scale for gamma[j] == 1
  array[p] real<lower=1> c; // Prior scale (when multiplied by tau) for gamma[j] == 0, lower = 1 as want slab fatter than spike
  real<lower=0> sigma;  
  array[p] real<lower=0, upper=1> gamma_prob; // Probability that gamma[j] = 1
}

transformed parameters {
  real<lower= 0> sigma2;
  sigma2 = sigma^2;

}

model {
  // Priors
  beta0 ~ normal(0,10);
  sigma2 ~ inv_gamma(0.5, 0.5);
  for (j in 1:p) {
    tau[j] ~ inv_gamma(3, 1);
    c[j] ~ inv_gamma(0.5, 10);
    gamma_prob[j] ~ beta(1,1); 
  }

  // Prior for beta (spike-and-slab)
  for (j in 1:p) {
    target += log_mix(gamma_prob[j],
                      normal_lpdf(beta[j] | 0, c[j] * tau[j]),
                      normal_lpdf(beta[j] | 0, tau[j]));
  }

  // Likelihood
  for (i in 1:n) {
    y[i] ~ poisson_log(beta0 + dot_product(beta, x[i,]));
  }
}
