# Certifying clusters from sum-of-norms clustering
Sum-of-norms clustering is a clustering formulation based on convex optimization that automatically induces hierarchy. We present a clustering test for sum-of-norms clustering that identifies and certifies the correct cluster assignment from an approximate solution yielded by any primal-dual algorithm. Numerical experiments are conducted on Gaussian mixture and half-moon data.  Relying on the fact that our certification extends to multiplicative weighting, our numerical experiments illustrate a technique to strengthen the recovery power of sum-of-norms clustering applied to data sampled from distributions with peaks such as a Gaussian mixture.

This repository contains the numerical implementation for the clustering test in \[1\] and the ADMM algorithm presented in \[2\]. 

### Dependencies
The code has been tested in `Julia 1.3` and depends on the packages `Distributions`, `LinearAlgebra`, `Random`, `Distributions`, `LightGraphs`.

### Quick Tour
- `cluster_test.jl` implements ADMM in \[2\] to solve SON with both uniform and multiplicative weights, apply the clustering test in \[1\] to certify cluster assignments and evaluate the performance of the clustering.
- `data_generation.jl` generates half-moon and a mixture of Gaussians datasets and call `cluster_test.jl` to find certified clusters for these datasets.
- `examples.jl` is a sample script that one should follow in order to use the library.



\[1\]: Tao Jiang, Stephen Vavasis. *Certifying clusters from sum-of-norms clustering.*
Available [here](https://arxiv.org/pdf/2006.11355).

\[2\]: Eric C. Chi, Kenneth Lange. *Splitting Methods for Convex Clustering.*
Available [here](https://arxiv.org/abs/1304.0499).
