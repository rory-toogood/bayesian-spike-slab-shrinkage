# Adapting Shrinkage Priors to the Spike and Slab

Bayesian variable selection for regression models, including high-dimensional settings. This repository holds the Stan and R code for my MMath dissertation at Durham University (2025; dissertation graded 82%, project 81% overall). The dissertation itself is in [`docs/dissertation.pdf`](docs/dissertation.pdf), with a one-page summary poster in [`docs/poster.pdf`](docs/poster.pdf); see [Corrections since submission](#corrections-since-submission) for what has changed since.

The idea: take five popular continuous shrinkage priors (Bayesian Lasso, Horseshoe, Horseshoe+, Dirichlet–Laplace and Generalised Double Pareto), build each one into a **spike-and-slab** prior, and test whether the hybrid keeps the adaptive shrinkage of the original prior while also giving explicit include/exclude decisions. Every method is compared with stochastic search variable selection (SSVS; George & McCulloch, 1993) as a strong baseline.

## Why this approach

| Decision | Why |
|---|---|
| **Combine shrinkage priors with spike-and-slab** | Continuous shrinkage priors (Bayesian Lasso, Horseshoe and others) adapt well to sparse signals and are widely used for variable selection, but because the prior is continuous, the posterior puts zero probability on βⱼ = 0 exactly: draws and posterior means are never exactly zero, so selection needs an extra thresholding rule. (The Bayesian Lasso's posterior *mode* is the lasso estimate, which can be exactly zero, but the full posterior is not.) Spike-and-slab priors give an explicit include/exclude probability, but they can be slow to mix, for example under collinearity. A hybrid aims to get both. |
| **Marginalise out the binary inclusion indicators** | Stan's sampler (HMC/NUTS) can't sample discrete parameters. Summing over γⱼ ∈ {0,1} gives the mixture prior βⱼ ~ πⱼ·p_slab + (1−πⱼ)·p_spike, which keeps the selection structure and is fully continuous. |
| **πⱼ ~ Beta(1, 1)** | This encodes no prior belief about how sparse the model is. It is easy to change if prior knowledge exists. |
| **"Fixed" and "prior" version of each model** | Fixed hyperparameters are faster and often mix better. Hyperpriors adapt to unknown sparsity but cost more computation. Running both shows how sensitive each method is to tuning. |
| **"Faster" variants** | The full hierarchies were slow, with long warm-ups and divergent transitions. Moving the spike-and-slab mixture lower in the hierarchy (onto a scale or rate parameter instead of βⱼ) gives fewer mixture evaluations and a narrower posterior funnel. The cost is less adaptivity and more sensitivity to hyperparameters. |

## Simulation design

Rather than testing settings by hand, `R/simulate_data.R` generates datasets with a **known** sparse β, and `R/evaluate.R` loops the models over the settings and returns summary metrics.

- **Grid:** 21 (n, p, d) combinations × 4 signal structures × 3 signal-to-noise ratios = **252 datasets**, from n = 30, p = 10 up to n = 1000, p = 200. It includes p > n settings (n = 30, p = 200).
- **Signal structures:** mixed weak/strong ("bimodal"), Gaussian, half positive/half negative, and extreme sparse.
- **Also tested:** correlated predictors (inverse-Wishart covariance at three correlation strengths, 54 datasets), and logistic and Poisson regression (24 datasets each).
- **Metrics:** true positive rate (signals found), true negative rate (noise variables excluded), and run time.
- **Selection rule:** variable j is selected if πⱼ ≥ 0.5 in at least half of the posterior draws. This is exactly the *median probability model* (Barbieri & Berger, 2004), which selects every variable with posterior inclusion probability PIPⱼ = P(γⱼ = 1 | y) ≥ 0.5. With a Beta(1, 1) prior, πⱼ | γⱼ ~ Beta(1 + γⱼ, 2 − γⱼ), so P(πⱼ ≥ 0.5 | y) = 0.25 + 0.5·PIPⱼ, which is at least 0.5 exactly when PIPⱼ ≥ 0.5. (This holds for every model that mixes two densities with `log_mix`; the Dirichlet–Laplace models blend variances instead, so for them the rule is simply "posterior median of πⱼ ≥ 0.5".)
- **Time:** warm-up plus sampling, summed over the 4 chains (so roughly 4× the wall-clock time when chains run in parallel).

## Key results

### Simulations

Faster variants against the baseline, averaged over the 63 mixed-signal ("bimodal") settings (dissertation Table 4.9):

| Method | Mean TPR | Mean TNR | Mean time (s) |
|---|---|---|---|
| SSVS (fixed), baseline | 0.704 | 0.979 | 56.1 |
| Adapted Bayesian Lasso (prior) | 0.637 | 0.960 | 102.6 |
| **Bayesian Lasso, faster (prior)** | **0.719** | **0.957** | **32.5** |
| Bayesian Lasso, faster-faster (prior) | 0.706 | 0.958 | see note |
| GDP, faster (prior) | 0.809 | 0.847 | 67.7 |

- Moving the spike-and-slab split down the hierarchy cut the adapted Bayesian Lasso's mean run time from 102.6 s to 32.5 s (about 3× faster), while its TPR rose from 0.637 to 0.719 and its TNR stayed level.
- *Note:* the dissertation reports 5.9 s for the faster-faster variant. That figure is wrong (see [Corrections](#corrections-since-submission)). Re-timed in 2026 on the same 63 settings, faster and faster-faster take about the same time (means of 38.2 s and 40.1 s on today's machine, median ratio 1.02), so both give roughly a 3× speed-up, not 17×.
- The adapted Bayesian Lasso and adapted Generalised Double Pareto were the strongest adapted priors. They competed with SSVS, especially in harder settings, and beat it in several.
- Horseshoe and Horseshoe+ adaptations improved a lot with the faster variants, but they still trailed the baseline.

The dissertation tables for these runs are in [`results/faster_methods_combined.csv`](results/faster_methods_combined.csv). That file still shows the 5.9 s timing as originally reported.

### Real data (re-run in 2026 with the corrected code)

Summaries are in [`results/real_data_summary.csv`](results/real_data_summary.csv) and [`results/real_data_selection.csv`](results/real_data_selection.csv); the 2026 re-timing is in [`results/retime_blasso_2026.csv`](results/retime_blasso_2026.csv). Each real-data script prints test and training performance, the number of variables selected, run time and convergence diagnostics (largest R-hat, divergent transitions, tree-depth hits).

- **Pima Indians Diabetes (logistic, n = 768, p = 8; 20 runs on one split).** SSVS (fixed) and both GDP setups select pregnancies, glucose and BMI in every run (GDP sometimes adds the pedigree function) and give the best test log-likelihoods (about −67 on 154 test cases, against −98 for an intercept-only model). BLASSO (prior) is close behind (−70.6) with glucose and BMI. Six setups select nothing in every run (SSVS prior, BLASSO fixed, both Horseshoe setups, Horseshoe+ fixed and DL fixed), because their hyperparameters were not tuned for GLMs, and both DL setups had many divergent transitions.
- **Galápagos species (Poisson, n = 30, p = 5), with log(Area), log(Elevation), log(Adjacent).** SSVS, BLASSO and GDP select log(Area) in every run, matching the species–area relationship in the literature. GDP (fixed) also picks Scruz and Adjacent and has the best test log-likelihood (−32, against −517 for predicting the training mean).
- **Riboflavin (normal, n = 71, the 1,000 most variable of 4,088 genes).** This p ≫ n problem is beyond what these HMC samplers handle reliably: only DL (fixed) reached R-hat ≈ 1.01, and the other models reached R-hat between 1.2 and 19.9. DL (prior) (test MSE 1.13), GDP faster (prior) (1.43), BLASSO faster (prior) (1.66) and DL (fixed) (1.89) beat an intercept-only prediction (2.44); the others did not. Treat these numbers as indicative only.
- **Boston Housing (normal, n = 506, p = 12 after dropping `B`).** Over 20 random train/test splits, GDP (prior) is clearly best: it selects rm, dis and lstat in at least 95% of splits, plus ptratio and nox in most, with mean test MSE 0.39 against 1.04 for an intercept-only prediction (y standardised). SSVS (fixed) and BLASSO (fixed) select lstat every time, sometimes with rm and dis (test MSE about 0.58). Six setups select nothing in almost every split, including BLASSO (prior), which was competitive in the dissertation (its model code is unchanged, so the difference comes from the data or splits), and Dirichlet–Laplace again had many divergent transitions.

## Repository layout

```
stan/
  normal/     12 models: 6 methods (SSVS + 5 adapted priors) x {fixed, prior}
  faster/     7 reparameterised "faster" variants
  logistic/   the same 12 models with a Bernoulli-logit likelihood
  poisson/    the same 12 models with a Poisson-log likelihood
R/
  simulate_data.R       normal-regression generators: 252 uncorrelated + 54 correlated settings
  simulate_glm_data.R   logistic and Poisson generators (24 settings each)
  models.R          registry of Stan models
  evaluate.R        fit a model, select variables, TPR / TNR / time; loop over a grid
  real_data.R       train/test split, fit, predict, score; repeat over runs or splits
run_simulations.R          reproduces the simulation study (QUICK mode for a smoke test)
retime_blasso_faster.R     re-times the two faster Bayesian Lasso variants (see Corrections)
run_real_data_boston.R     Boston Housing (normal)
run_real_data_riboflavin.R Riboflavin (normal, p >> n)
run_real_data_pima.R       Pima Indians Diabetes (logistic)
run_real_data_gala.R       Galapagos species counts (Poisson)
results/                   summary results
docs/dissertation.pdf      the dissertation as submitted (May 2025, unchanged)
docs/poster.pdf            one-page summary poster (2026, with the corrected results)
archive/
  stan-experiments/   earlier models (SSLASSO, g-priors, other parameterisations)
```

## How to reproduce

```r
# R >= 4.2 with a working C++ toolchain
source("install.R")            # rstan, MASS, MCMCpack, mlbench, faraway, remotes
# then, from the repository root:
source("run_simulations.R")    # QUICK <- TRUE runs 3 small settings first
```

The full grid is several hours of MCMC. Set `QUICK <- FALSE` in `run_simulations.R` to run all of it. The real-data scripts run on their own (`run_real_data_*.R`); Riboflavin takes several hours.

## Corrections since submission

I re-checked all the code in 2026, before publishing it. The [dissertation PDF](docs/dissertation.pdf) is unchanged from the version submitted in May 2025; these are the differences between it and this repository.

**Changes that affect reported numbers**

1. **Faster-faster Bayesian Lasso run time.** Table 4.9 reports 5.9 s. The saved time was identical for all 63 runs, because the timer read a fit object left over from an earlier run. Re-timed, it takes about as long as the faster variant (see Key results).
2. **Intercept left out of GLM predictions (Pima, Galápagos).** Test and training log-likelihoods were computed from Xβ without β₀, so every predicted probability was centred on 0.5 and every Poisson mean on 1. Variable selection was not affected. Predictions now include β₀, and a model that selects nothing predicts from the intercept alone.
3. **Riboflavin gene subset.** The genes were standardised *before* being ranked by variance, so every gene had variance 1 and the "1,000 most variable genes" were in effect the first 1,000 columns. Genes are now ranked by variance on the raw training data, then standardised.
4. **Galápagos used `Endemics` as a predictor.** Endemics counts the endemic species on each island, which is part of the response (Species), so the model was partly given the answer. It is now excluded. The analysis also now uses log(Area), log(Elevation) and log(Adjacent), as is standard for this dataset; on the raw scale a single extreme test island dominated the log-likelihood.
5. **Boston Housing no longer uses `B`**, following current practice for this dataset (scikit-learn removed it over this race-derived variable). `DROP_B <- FALSE` in `run_real_data_boston.R` reproduces the dissertation's analysis.
6. **Riboflavin Table 5.2.** The last four rows (Dirichlet–Laplace and GDP) appear to have their labels swapped. Several methods had far lower test errors than SSVS, but none beat an intercept-only prediction (test MSE 2.44 on this split), a baseline the dissertation did not report.

**Where the code and the write-up disagree.** The code produced the dissertation's results, so these are corrections to the description, not to the models:

7. **SSVS hyperparameters.** The code uses spike and slab *standard deviations* of 0.1 and 10 (variances 0.01 and 100), not variances of 0.1 and 10. In the prior setup, the inverse-gamma priors are on τ and c, not on τ² and c².
8. **SSVS (fixed) residual variance** uses σ² ~ IG(0.01, 0.01). All other models use IG(0.5, 0.5).
9. **Horseshoe+ (prior)** uses a slab scale τ_slab ~ C⁺(0, 10), not C⁺(0, 100).
10. **GDP faster (fixed) and GDP faster (prior) are different hierarchies.** The fixed file has the faster-faster structure (mixture on the rate λ, with τ as a standard deviation); the prior file has the faster structure (mixture on τ, with τ as a variance). Their comparison is therefore not purely fixed against learned hyperparameters.
11. **Dirichlet–Laplace.** The global scale uses Gamma(n·a, 1/2) with n the sample size, while Bhattacharya et al. use the number of coefficients (p here). Also note (as Section 3.6.4 describes) that DL blends the spike and slab variances, βⱼ = zⱼ·√(πⱼλ²_slab + (1−πⱼ)λ²_spike), to keep a non-centred parameterisation that samples well. So πⱼ acts as a weight between two scales rather than the probability of coming from the slab, and DL is not a two-component spike and slab like the other models. Putting a `log_mix` on the local scale ψⱼ instead would restore that while keeping the non-centred form. Two unused parameters with no prior have been removed from the DL (prior) files.

**Design choices worth knowing**

- Each simulation setting is one dataset (fixed seed), so settings with the same (n, p) share X and the active set. Comparisons between methods are paired, but there are no replicates, so no standard errors.
- The models sample σ but state the prior as σ² ~ IG(a, b) without a Jacobian adjustment, so the prior actually used is σ² ~ IG(a + ½, b), e.g. IG(1, 0.5) rather than IG(0.5, 0.5). With moderate n the effect on the results is small.
- Dirichlet–Laplace was the hardest model for the sampler throughout: many divergent transitions and tree-depth warnings, even on the small Galápagos data.

## Limitations and next steps

- In p ≫ n settings (Riboflavin) most models did not converge: the spike-and-slab posterior is multimodal, and NUTS, which moves along smooth trajectories, rarely crosses the low-probability regions between modes. Gibbs samplers with conjugate updates, tempering, or variational inference are natural next steps.
- Most full hierarchical models are computationally expensive under HMC. The faster variants help, and other inference methods (e.g. variational inference) are a natural next step.
- Results are sensitive to hyperparameters. A principled tuning strategy (empirical Bayes or cross-validation) would make them more stable.
- Evaluation could be extended to coefficient-estimation accuracy, uncertainty calibration and Bayesian model averaging.

## References

Main sources: Ročková & George (2018), *The Spike-and-Slab LASSO*; Park & Casella (2008), *The Bayesian Lasso*; Carvalho, Polson & Scott (2010), *The Horseshoe Estimator for Sparse Signals*; Bhadra et al. (2017), *Horseshoe+*; Bhattacharya et al. (2015), *Dirichlet–Laplace Priors*; Armagan, Dunson & Lee (2013), *Generalized Double Pareto Shrinkage*; George & McCulloch (1993), *Variable Selection via Gibbs Sampling*. The full bibliography is in the dissertation.

## Licence

Code is released under the MIT licence (see `LICENSE`).
