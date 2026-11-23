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
  array[p] real<lower=0> lambda0_sq;   // Slab lambda^2 (low shrinkage)
  array[p] real<lower=0> lambda1_sq;   // Spike lambda^2 (high shrinkage)
  vector<lower=0>[p] tau_sq;     // Local scale parameters (mixing variables)
  array[p] real<lower=0, upper=1> gamma_prob; // Bernoulli probs for spike/slab
}

transformed parameters {
  real<lower=0> sigma2 = square(sigma);
}

model {
  // Hyperpriors
  beta0 ~ normal(0, 10);
  sigma2 ~ inv_gamma(0.5, 0.5);
  
  for (j in 1:p) {
    lambda0_sq[j] ~ gamma(0.5, 0.5);   // slab
    lambda1_sq[j] ~ gamma(5, 0.5);     // spike
    gamma_prob[j] ~ beta(1, 1);        // prior inclusion probability
    
    // Prior on tau_sq (scale mixture of normals)
    target += log_mix(gamma_prob[j],
                      exponential_lpdf(tau_sq[j] | 0.5 * lambda0_sq[j]),
                      exponential_lpdf(tau_sq[j] | 0.5 * lambda1_sq[j]));
    
    // Prior on beta given tau_sq
    beta[j] ~ normal(0, sqrt(tau_sq[j]));
  }

  // Likelihood
  y ~ normal(beta0 + x * beta, sigma);
}

