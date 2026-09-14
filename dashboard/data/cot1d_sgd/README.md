# SGD (`GradientAscentMultiMarginalSGD`) — 1D benchmark

Recovered results for the solver labelled **SGD** in the thesis figures. These
are the runs the printed Figure 2 / Figure 3 SGD panels were made from; they had
never been copied into this repository.

## Provenance

Copied from `~/ot-algorithm-comparison/results/synthetic/cot/sgd_1d_*.csv`
(October 2025), which is also the directory `dashboard/data/cot1d/` was copied
from — every file the two directories share is byte-identical. Eleven files in
that source directory were never brought across, including all seven `sgd_*`.

Produced by `configs/runners/cot/sgd.yaml`:
`uot.solvers.gradient_ascent.GradientAscentMultiMarginalSGD`,
`learning_rate = 0.002`, `momentum = 0.9`, `maxiter = 1e6`, `tol = 1e-6`,
reg in {1.0, 0.1, 0.01, 0.001}. The data carries those values in its own
`learning_rate` / `momentum` columns, which is what identifies it.

## Two things to know before plotting

**1. `sgd` and `gradient` are different solvers that share one label.**
`dashboard/dataset.py::SOLVERS_MAP` maps *both* raw names to `"SGD"`:

    "gradient": "SGD",   # gradient.yaml -> GradientAscentTwoMarginalSolver, lr 1e-3, no momentum
    "sgd":      "SGD",   # sgd.yaml      -> GradientAscentMultiMarginalSGD,  lr 2e-3, momentum 0.9

`gradient` rows live in `dashboard/data/cot1d/gradient_1d_*.csv`. If both
directories are loaded together the two solvers are averaged into one panel and
every cell roughly halves (gaussian-1c max-iter hit rate at eps=1e-3 goes
100% -> 50%), silently. Only the `sgd` rows reproduce the printed figure:
`gradient` has a 0.1% hit rate and would render the panel almost entirely dark.
Exclude `gradient`, or give it its own label, when building the SGD panel.

**2. The three exponential rows are failed runs, not stable runs.**

| distribution        | runs | failed |
|---------------------|------|--------|
| exponential         |  868 |  100%  |
| exp-vs-gaussian     |  868 |  100%  |
| exp-vs-cauchy       |  868 |  100%  |
| all others          | 10388|    0%  |

Every one raised `Cost shape (n, n) incompatible with marginals (k, n)` with
k < n: the exponential generators drop zero-weight support points (scipy's
`expon.pdf` is exactly 0 left of `loc`), so the marginal no longer matches the
n x n cost matrix. The solver never ran.

These rows have `status = failed` and no `iterations`. A heatmap that counts
`iterations >= maxiter` scores them as 0 and paints them as *perfectly stable*,
which is how they appear in the printed figure. **Filter on
`status == "success"` before computing hit rates**, and show the exponential
rows as missing data rather than as zeros.

## Schema

Normalised to the `dashboard/data/cot1d/` column order, plus
`tol`, `learning_rate`, `momentum`, `residual_l2`, `exception`.

- `error` is a copy of `residual_l2`, which is `max_i ||marginal_i - a_i||_2` —
  the same quantity the current schema calls `error`
  (cf. `uot/solvers/sinkhorn/sinkhorn.py::_compute_error`). The originals had no
  `error` column, so without this the rows would read as NaN and be counted as
  numerical instability.
- `cost_rerr` is NaN; the dashboard derives it from the `lp` reference rows.
- The original `potentials` column was dropped: a stringified array repr,
  truncated with `...` at larger n, and ~90% of the 30 MB bulk. The untouched
  originals remain in the source directory above.
