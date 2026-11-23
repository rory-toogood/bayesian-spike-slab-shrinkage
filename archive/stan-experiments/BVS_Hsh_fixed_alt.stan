// As in the regularised horseshoe prior paper

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
  real <lower = 0> tau0[p];
  real <lower = 0> tau1[p];
  real<lower=0, upper=1> gamma_prob[p]; // Probability that gamma[j] = 1
}

transformed parameters{
  real<lower=0> sigma2;
  sigma2 = sigma^2;
}


model {
  // Priors
  sigma2 ~ inv_gamma(0.5, 0.5);
  for (j in 1:p) {
    tau0[j] ~ cauchy(0, 1);
    tau1[j] ~ cauchy(0, 1);
    gamma_prob[j] ~ beta(1,1); 
  }

  // Prior for beta (spike-and-slab)
  for (j in 1:p) {
    target += log_mix(gamma_prob[j],
                      normal_lpdf(beta[j] | 0, 100*tau1[j]), // slab (when beta_j included)
                      normal_lpdf(beta[j] | 0, tau0[j])); // spike
  }

  // Likelihood
  for (i in 1:n) {
    y[i] ~ normal(beta0 + dot_product(beta, x[i, :]), sigma);
  }
}


