# Install the R packages this repository needs (Stan files use the current
# `array[p] real` syntax, so any recent rstan works).
install.packages(c("rstan", "MASS", "MCMCpack", "rstudioapi",
                   "mlbench", "faraway", "remotes"))
# The Riboflavin data come from the hdi package, which is archived on CRAN.
# You don't need to install it: run_real_data_riboflavin.R downloads the last
# hdi release with `remotes` and loads just the data file.
# mlbench (Pima) and faraway (Gala) are for the real-data examples.
