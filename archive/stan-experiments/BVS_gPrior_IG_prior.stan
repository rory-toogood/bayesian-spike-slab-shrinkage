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
  vector<lower=0, upper=1>[p] gamma_prob; // Probability that gamma[j] = 1
  real<lower=0> g1;
  real<lower=0> g2;
}

transformed parameters{
  real<lower=0> sigma2;
  sigma2 = sigma^2;
}

model {
  // Priors
  sigma2 ~ inv_gamma(0.5, 0.5);
  g1 ~ inv_gamma(0.5, 0.5);
  g2 ~ inv_gamma(0.5, 0.5);
  for (j in 1:p) {
    gamma_prob[j] ~ beta(1,1); 
  }

  // Prior for beta (spike-and-slab)
//  for (j in 1:p) {
//    target += log_mix(gamma_prob[j],
//                      normal_lpdf(beta[j] | 0, g1 * sigma2 * inverse(x' * x)), // slab (when beta_j included)
//                      normal_lpdf(beta[j] | 0, g2 * sigma2 * inverse(x' * x))); // spike
//  }

  for (j in 1:p) {
    target += log_sum_exp(
      log(gamma_prob) + multi_normal_lpdf(beta | rep_vector(0, p), sigma^2 * inverse(x' * x)),
      log((1-gamma_prob)) + multi_normal_lpdf(beta | rep_vector(0, p), sigma^2 * inverse(x' * x))
    );
  }

  // Likelihood
  for (i in 1:n) {
    y[i] ~ normal(beta0 + dot_product(beta, x[i, :]), sigma);
  }
}

