data {
  int<lower=0> n;  
  int<lower=0> p;  
  matrix[n, p] x; 
  array[n] int<lower=0> y;
}

parameters {
  real beta0;                     // Intercept
  vector[p] beta_raw;             // Raw regression coefficients
  vector<lower=0>[p] psi_spike;   // Local shrinkage exponential parameters for spike
  vector<lower=0>[p] psi_slab;    // Local shrinkage exponential parameters for slab
  simplex[p] phi_spike;           // Dirichlet local shrinkage for spike
  simplex[p] phi_slab;            // Dirichlet local shrinkage for slab
  real<lower=0> tau_spike;        // Global shrinkage for spike
  real<lower=0> tau_slab;         // Global shrinkage for slab
  real<lower=0> a_spike;  // Dirichlet parameter for spike
  real<lower=0> a_slab;   // Dirichlet parameter for slab
  real<lower=0> sigma;            // Residual standard deviation
  vector<lower=0, upper=1>[p] gamma_prob; // Inclusion probabilities
}

transformed parameters {
  vector[p] beta;                 // Actual regression coefficients
  vector<lower=0>[p] lambda_spike;
  vector<lower=0>[p] lambda_slab;
  real<lower=0> sigma2 = square(sigma);
  
  // Compute local shrinkage parameters
  for (j in 1:p) {
    lambda_spike[j] = sqrt(psi_spike[j]) * phi_spike[j] * tau_spike;
    lambda_slab[j] = sqrt(psi_slab[j]) * phi_slab[j] * tau_slab;
    
    // Compute coefficients using spike-and-slab mixture
    beta[j] = beta_raw[j] * sqrt(gamma_prob[j] * square(lambda_slab[j]) + 
                                (1 - gamma_prob[j]) * square(lambda_spike[j]));
  }
}

model {
  a_spike~gamma(1,4);  
  a_slab~gamma(4,1);
 
  // Priors for global shrinkage
  tau_spike ~ gamma(n*a_spike, 0.5);
  tau_slab ~ gamma(n*a_slab, 0.5);
  
  // Dirichlet priors for local shrinkage
  phi_spike ~ dirichlet(rep_vector(a_spike, p));
  phi_slab ~ dirichlet(rep_vector(a_slab, p));
  
  // Exponential priors for psi
  psi_spike ~ exponential(0.5);
  psi_slab ~ exponential(0.5);
  
  // Prior for inclusion probabilities
  gamma_prob ~ beta(1, 1);
  
  // Prior for intercept and noise
  beta0 ~ normal(0, 10);
  sigma2 ~ inv_gamma(0.5, 0.5);
  
  // Prior for raw coefficients
  beta_raw ~ std_normal();
  
  // Likelihood
  y ~ poisson_log(beta0 + x * beta);
}

generated quantities {
  vector[p] inclusion_prob = gamma_prob;
  vector[p] lambda_effective;
  
  // Compute effective shrinkage and posterior predictive checks
  for (j in 1:p) {
    lambda_effective[j] = sqrt(gamma_prob[j] * square(lambda_slab[j]) + 
                              (1 - gamma_prob[j]) * square(lambda_spike[j]));
  }
  
}


