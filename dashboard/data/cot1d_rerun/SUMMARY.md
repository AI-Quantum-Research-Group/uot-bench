# Vanilla Sinkhorn (scaling form) re-run — 1D benchmark

Solver `uot.solvers.sinkhorn.SinkhornTwoMarginalSolver` (`_sinkhorn_plain`), tol = 1e-6, `maxiter = 100_000`, `JAX_ENABLE_X64=True`, one dedicated H100 per job.

Regenerates the `sinkhorn` rows in `dashboard/data/cot1d/`, which were produced by the superseded `_sinkhorn` log-increment path and are NaN throughout (`cost`, `error` = NaN, `iterations` = 11, `status` = success).

**12684 runs across 7 generator configs. NaN count: 0. Distinct iteration counts: 522 (range 11–100000).**

## Per distribution and regularization

`distribution` follows the dashboard's own parsing of the dataset name (`1D-<distribution>-<n>p`), so multi-component families appear as separate rows.

| distribution | reg | runs | NaN | frac hit maxiter | median final error |
|---|---|---|---|---|---|
| cauchy | 0.001 | 210 | 0 | 1.000 | 4.599e-02 |
| cauchy | 0.01 | 210 | 0 | 0.367 | 9.977e-07 |
| cauchy | 0.1 | 210 | 0 | 0.000 | 2.125e-07 |
| cauchy | 1 | 210 | 0 | 0.000 | 3.271e-14 |
| cauchy-vs-gaussian | 0.001 | 217 | 0 | 1.000 | 1.438e-01 |
| cauchy-vs-gaussian | 0.01 | 217 | 0 | 0.608 | 3.334e-03 |
| cauchy-vs-gaussian | 0.1 | 217 | 0 | 0.000 | 2.105e-11 |
| cauchy-vs-gaussian | 1 | 217 | 0 | 0.000 | 2.277e-17 |
| exp-vs-cauchy | 0.001 | 217 | 0 | 1.000 | 6.456e-02 |
| exp-vs-cauchy | 0.01 | 217 | 0 | 0.995 | 6.427e-03 |
| exp-vs-cauchy | 0.1 | 217 | 0 | 0.000 | 3.589e-08 |
| exp-vs-cauchy | 1 | 217 | 0 | 0.000 | 2.225e-17 |
| exp-vs-gaussian | 0.001 | 217 | 0 | 1.000 | 1.111e-01 |
| exp-vs-gaussian | 0.01 | 217 | 0 | 0.447 | 5.901e-07 |
| exp-vs-gaussian | 0.1 | 217 | 0 | 0.000 | 6.360e-17 |
| exp-vs-gaussian | 1 | 217 | 0 | 0.000 | 2.668e-17 |
| exponential | 0.001 | 210 | 0 | 0.724 | 1.859e-02 |
| exponential | 0.01 | 210 | 0 | 0.000 | 4.772e-07 |
| exponential | 0.1 | 210 | 0 | 0.000 | 1.221e-09 |
| exponential | 1 | 210 | 0 | 0.000 | 2.327e-17 |
| gaussian-1c | 0.001 | 210 | 0 | 0.933 | 8.234e-02 |
| gaussian-1c | 0.01 | 210 | 0 | 0.200 | 2.485e-09 |
| gaussian-1c | 0.1 | 210 | 0 | 0.000 | 3.392e-17 |
| gaussian-1c | 1 | 210 | 0 | 0.000 | 3.362e-17 |
| gaussian-2c | 0.001 | 210 | 0 | 0.962 | 4.930e-02 |
| gaussian-2c | 0.01 | 210 | 0 | 0.319 | 9.389e-07 |
| gaussian-2c | 0.1 | 210 | 0 | 0.000 | 5.588e-09 |
| gaussian-2c | 1 | 210 | 0 | 0.000 | 2.925e-17 |
| gaussian-4c | 0.001 | 210 | 0 | 0.995 | 4.854e-02 |
| gaussian-4c | 0.01 | 210 | 0 | 0.152 | 9.399e-07 |
| gaussian-4c | 0.1 | 210 | 0 | 0.000 | 7.367e-08 |
| gaussian-4c | 1 | 210 | 0 | 0.000 | 3.056e-17 |
| gaussian-6c | 0.001 | 210 | 0 | 1.000 | 2.660e-02 |
| gaussian-6c | 0.01 | 210 | 0 | 0.005 | 8.854e-07 |
| gaussian-6c | 0.1 | 210 | 0 | 0.000 | 2.097e-07 |
| gaussian-6c | 1 | 210 | 0 | 0.000 | 4.301e-17 |
| gen-hyperb-mixture-2c | 0.001 | 210 | 0 | 1.000 | 3.714e-02 |
| gen-hyperb-mixture-2c | 0.01 | 210 | 0 | 0.357 | 9.966e-07 |
| gen-hyperb-mixture-2c | 0.1 | 210 | 0 | 0.000 | 4.061e-08 |
| gen-hyperb-mixture-2c | 1 | 210 | 0 | 0.000 | 2.440e-17 |
| gen-hyperb-mixture-4c | 0.001 | 210 | 0 | 0.967 | 2.686e-02 |
| gen-hyperb-mixture-4c | 0.01 | 210 | 0 | 0.162 | 8.895e-07 |
| gen-hyperb-mixture-4c | 0.1 | 210 | 0 | 0.000 | 9.386e-08 |
| gen-hyperb-mixture-4c | 1 | 210 | 0 | 0.000 | 3.681e-17 |
| gen-hyperb-mixture-6c | 0.001 | 210 | 0 | 1.000 | 3.221e-02 |
| gen-hyperb-mixture-6c | 0.01 | 210 | 0 | 0.000 | 8.132e-07 |
| gen-hyperb-mixture-6c | 0.1 | 210 | 0 | 0.000 | 1.056e-07 |
| gen-hyperb-mixture-6c | 1 | 210 | 0 | 0.000 | 5.587e-17 |
| students-2df | 0.001 | 210 | 0 | 0.867 | 1.255e-02 |
| students-2df | 0.01 | 210 | 0 | 0.167 | 9.395e-07 |
| students-2df | 0.1 | 210 | 0 | 0.000 | 4.202e-07 |
| students-2df | 1 | 210 | 0 | 0.000 | 6.187e-10 |
| students-4df | 0.001 | 210 | 0 | 0.867 | 1.228e-02 |
| students-4df | 0.01 | 210 | 0 | 0.167 | 9.432e-07 |
| students-4df | 0.1 | 210 | 0 | 0.000 | 4.222e-07 |
| students-4df | 1 | 210 | 0 | 0.000 | 5.794e-10 |
| students-6df | 0.001 | 210 | 0 | 0.867 | 1.224e-02 |
| students-6df | 0.01 | 210 | 0 | 0.167 | 9.411e-07 |
| students-6df | 0.1 | 210 | 0 | 0.000 | 4.048e-07 |
| students-6df | 1 | 210 | 0 | 0.000 | 5.711e-10 |

## Rolled up to the seven generator configs

| generator config | reg | runs | NaN | frac hit maxiter | median final error |
|---|---|---|---|---|---|
| 1d_heavy_tailed_cauchy | 0.001 | 210 | 0 | 1.000 | 4.599e-02 |
| 1d_heavy_tailed_cauchy | 0.01 | 210 | 0 | 0.367 | 9.977e-07 |
| 1d_heavy_tailed_cauchy | 0.1 | 210 | 0 | 0.000 | 2.125e-07 |
| 1d_heavy_tailed_cauchy | 1 | 210 | 0 | 0.000 | 3.271e-14 |
| 1d_heavy_tailed_student | 0.001 | 630 | 0 | 0.867 | 1.239e-02 |
| 1d_heavy_tailed_student | 0.01 | 630 | 0 | 0.167 | 9.415e-07 |
| 1d_heavy_tailed_student | 0.1 | 630 | 0 | 0.000 | 4.170e-07 |
| 1d_heavy_tailed_student | 1 | 630 | 0 | 0.000 | 5.950e-10 |
| 1d_light_tailed_exponential | 0.001 | 210 | 0 | 0.724 | 1.859e-02 |
| 1d_light_tailed_exponential | 0.01 | 210 | 0 | 0.000 | 4.772e-07 |
| 1d_light_tailed_exponential | 0.1 | 210 | 0 | 0.000 | 1.221e-09 |
| 1d_light_tailed_exponential | 1 | 210 | 0 | 0.000 | 2.327e-17 |
| 1d_light_tailed_gaussian | 0.001 | 210 | 0 | 0.933 | 8.234e-02 |
| 1d_light_tailed_gaussian | 0.01 | 210 | 0 | 0.200 | 2.485e-09 |
| 1d_light_tailed_gaussian | 0.1 | 210 | 0 | 0.000 | 3.392e-17 |
| 1d_light_tailed_gaussian | 1 | 210 | 0 | 0.000 | 3.362e-17 |
| 1d_multimodal_gaussians | 0.001 | 630 | 0 | 0.986 | 4.131e-02 |
| 1d_multimodal_gaussians | 0.01 | 630 | 0 | 0.159 | 9.118e-07 |
| 1d_multimodal_gaussians | 0.1 | 630 | 0 | 0.000 | 7.832e-08 |
| 1d_multimodal_gaussians | 1 | 630 | 0 | 0.000 | 3.298e-17 |
| 1d_multimodal_gh | 0.001 | 630 | 0 | 0.989 | 3.193e-02 |
| 1d_multimodal_gh | 0.01 | 630 | 0 | 0.173 | 8.962e-07 |
| 1d_multimodal_gh | 0.1 | 630 | 0 | 0.000 | 7.466e-08 |
| 1d_multimodal_gh | 1 | 630 | 0 | 0.000 | 3.442e-17 |
| 1d_paired | 0.001 | 651 | 0 | 1.000 | 1.008e-01 |
| 1d_paired | 0.01 | 651 | 0 | 0.684 | 3.246e-03 |
| 1d_paired | 0.1 | 651 | 0 | 0.000 | 7.490e-12 |
| 1d_paired | 1 | 651 | 0 | 0.000 | 2.393e-17 |

## reg = 1e-2 by grid size

reg = 1e-2 is not a clean convergence row: about 28% of runs exhaust the iteration cap. The fraction is essentially flat in n, so this is a property of the problem instances, not of grid resolution.

| n | runs | frac hit maxiter | median final error |
|---|---|---|---|
| 32 | 453 | 0.263 | 9.384e-07 |
| 64 | 453 | 0.278 | 9.391e-07 |
| 128 | 453 | 0.283 | 9.442e-07 |
| 256 | 453 | 0.278 | 9.390e-07 |
| 512 | 453 | 0.280 | 9.463e-07 |
| 1024 | 453 | 0.278 | 9.377e-07 |
| 2048 | 453 | 0.278 | 9.458e-07 |
