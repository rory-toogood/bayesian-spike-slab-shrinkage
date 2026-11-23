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
  real<lower=0> sigma2;
  sigma2 = sigma^2;
}

model {
  // Priors
  beta0 ~ normal(0, 10);
  sigma2 ~ inv_gamma(0.5, 0.5);
  for (j in 1:p) {
    gamma_prob[j] ~ beta(1,1); 
  }

  // Prior for beta (spike-and-slab)
  for (j in 1:p) {
    target += log_mix(gamma_prob[j],
                      double_exponential_lpdf(beta[j] | 0, 20*sigma), // input param is sigma = 1/lambda_0 ie more shrinkage (as not included)
                      double_exponential_lpdf(beta[j] | 0, sigma/10)); /// input param is sigma = 1/lambda_1 so more shrinkage 
  }

  // Likelihood
  for (i in 1:n) {
    y[i] ~ normal(beta0 + dot_product(beta, x[i, :]), sigma);
  }
}

