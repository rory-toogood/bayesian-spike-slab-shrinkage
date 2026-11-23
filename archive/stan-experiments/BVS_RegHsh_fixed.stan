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
  real <lower = 0> lambda0[p];
  real <lower = 0> lambda1[p];
  real<lower=0, upper=1> gamma_prob[p]; // Probability that gamma[j] = 1
  real<lower = 0> c;
}

transformed parameters{
  real< lower = 0 > sigma2;
  real<lower = 0> lambda0_2[p];
  real<lower = 0> lambda1_2[p];
  real<lower = 0> c2;
  real<lower = 0> lambda_tilda0_2[p];
  real<lower = 0> lambda_tilda1_2[p];
  sigma2 = sigma^2;
  lambda0_2 = lambda0^2;
  lambda1_2 = lambda1^2;
  c2 = c^2;
  for (j in 1:p){
    lambda_tilda0_2[j] = c2*lambda0_2[j]/(c2 + lambda0_2[j]);
    lambda_tilda1_2[j] = c2*lambda1_2[j]/(c2 + 100 * lambda1_2[j]);
  }
}


model {
  // Priors
  sigma2 ~ inv_gamma(0.5, 0.5);
  c2 ~ inv_gamma(p/2, 100*p/2);
  for (j in 1:p) {
    lambda0[j] ~ cauchy(0, 1);
    lambda1[j] ~ cauchy(0, 1);
    gamma_prob[j] ~ beta(1,1); 
  }

  // Prior for beta (spike-and-slab)
  for (j in 1:p) {
    target += log_mix(gamma_prob[j],
                      normal_lpdf(beta[j] | 0, 100*sqrt(lambda_tilda1_2[j]), // slab (when beta_j included)
                      normal_lpdf(beta[j] | 0, sqrt(lambda_tilda0_2[j]))); // spike
  }

  // Likelihood
  for (i in 1:n) {
    y[i] ~ normal(beta0 + dot_product(beta, x[i, :]), sigma);
  }
}



