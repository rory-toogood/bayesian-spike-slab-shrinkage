data{
  int <lower = 0> n;
  int <lower = 0> p;
  matrix[n,p] x;
  vector[n] y;
}

parameters{
  vector[p] beta;
  real beta0;
  real <lower = 0> tau[p];
  real <lower = 0> phi[p];
  vector[p] gamma_prob; // Probability that gamma[j] = 1 (logit scale)

  real <lower = 0 > sigma;
}

transformed parameters {
  real<lower=0> tau2[p];
  real<lower=0> phi2[p];
  vector<lower=0, upper=1>[p] gamma; // Probability that gamma[j] = 1

  real<lower= 0> sigma2;
  sigma2 = sigma^2;

  for (j in 1:p){
    tau2[j] = (tau[j])^2;
    phi2[j] = (phi[j])^2;
    gamma[j] = inv_logit(gamma_prob[j]); // Transform to probability
  }
}

model{
  // Priors (Ridge equivalent)
  sigma2 ~ inv_gamma(0.5, 0.5);
  for(j in 1:p) gamma_prob[j] ~ normal(0, 1); // Prior for gamma_prob (logit scale)
  
  for(j in 1:p) tau[j] ~ inv_gamma(0.5, 0.5);
  
  for (j in 1:p) {
    beta[j] ~ normal(0, gamma[j] * tau2[j] + (1 - gamma[j]) * phi2[j]);
  }
  
  for (i in 1:n){ 
    y[i] ~ normal(beta0 + dot_product(beta, x[i]), sigma);
  }
  
}



