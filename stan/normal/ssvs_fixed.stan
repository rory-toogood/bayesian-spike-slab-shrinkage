data {
  int<lower=0> n;  
  int<lower=0> p;  
  matrix[n, p] x; 
  vector[n] y;     
}

parameters {
  vector[p] beta;      
  real beta0;          
  real<lower=0> sigma;  
  array[p] real<lower=0, upper=1> gamma_prob; // Probability that gamma[j] = 1
}

transformed parameters{
  real<lower= 0> sigma2;
  sigma2 = sigma^2;
}


model {
  // Priors
  beta0 ~ normal(0, 10);
  sigma2 ~ inv_gamma(0.01,0.01);
  for (j in 1:p) {
    gamma_prob[j] ~ beta(1,1); 
  }

  // Prior for beta (spike-and-slab)
  for (j in 1:p) {
    target += log_mix(gamma_prob[j],
                      normal_lpdf(beta[j] | 0, 10), // slab tau^2 * c^2
                      normal_lpdf(beta[j] | 0, 0.1)); // spike tau^2 * c^2
  }

  // Likelihood
  for (i in 1:n) {
    y[i] ~ normal(beta0 + dot_product(beta, x[i, :]), sigma);
  }
}
