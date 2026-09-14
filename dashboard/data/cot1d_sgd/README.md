# SGD (`GradientAscentMultiMarginalSGD`) — 1D benchmark

Recovered results for the solver labelled **SGD** in the thesis figures. These
are the runs the printed Figure 2 / Figure 3 SGD panels were made from; they had
never been copied into this repository.

## Provenance

Copied from `~/ot-algorithm-comparison/results/synthetic/cot/sgd_1d_*.csv`
(October 2025), which is also the directory `dashboard/data/cot1d/` was copied
from — every file the two directories share is byte-identical. Eighteen files in
that source directory were never brought across (1 back-n-forth, 1 empty lbfgs,
1 duplicate lp, 7 pdlp, 7 sgd, 1 sinkhorn-normed); all seven `sgd_*` are among
them. (An earlier revision of this file said eleven; that number wrongly
excluded the seven `pdlp_*` files.)

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

**2. The exponential rows come from a separate December rerun, at eps=1e-3 only.**

The October run failed outright on every exponential-family problem
(2604 rows, `status = failed`):
`GradientAscentMultiMarginalSGD` extracts marginals with
`as_point_cloud(include_zeros=False)`, which drops the zero-weight support
points the exponential generators produce (scipy's `expon.pdf` is exactly 0
left of `loc`), while the cost matrix stays n x n, giving
`Cost shape (n, n) incompatible with marginals (k, n)`. The solver never ran.
Those failed rows are still present in
`sgd_1d_light_tailed_exponential_14-15-56.csv` and
`sgd_1d_paired_16-20-41.csv`; they are kept so the failure stays visible.

A targeted rerun in December 2025 fixed those distributions and is the source
of the printed figures. It is imported here as:

| file | rows | from |
|---|---|---|
| `sgd_1d_light_tailed_exponential_rerun_12-16-26.csv` | 210 | `sgd-reg-0.001_1d_light_tailed_exponential_2025-12-12-16-26.csv` |
| `sgd_1d_paired_exponential_rerun_12-16-29.csv` | 434 | `sgd-reg-0.001_1d_paired_2025-12-12-16-29.csv` |

both from `~/ot-algorithm-comparison/results/synthetic/cot1d_sgd/`, 100%
success, same solver and hyper-parameters (lr 0.002, momentum 0.9,
maxiter 1e6). With them the reg=1e-3 hit rates reproduce the printed figure to
within one dataset (~3 points at 30-31 datasets per cell).

**The paired import is filtered.** The December paired file also contains
`cauchy-vs-gaussian`, which October already covers at all four regs; importing
it whole would have double-counted that distribution at reg=1e-3. Only the
`exp-vs-cauchy` and `exp-vs-gaussian` rows were taken (434 of 651). After the
filter no (dataset, reg) cell is duplicated anywhere in this directory.

**Coverage gap — the December rerun used `sgd-reg-0.001.yaml`, which has the
other three regs commented out.** So:

| distribution | reg=0.001 | reg=0.01 | reg=0.1 | reg=1.0 |
|---|---|---|---|---|
| exponential, exp-vs-cauchy, exp-vs-gaussian | yes | **none** | **none** | **none** |
| the other 12 distributions | yes | yes | yes | yes |

A figure spanning all four regs (e.g. hit rate vs eps) will have no SGD data
for the three exponential rows outside eps=1e-3. Closing that gap needs a
rerun, and the solver must be fixed first: `include_zeros=False` is still on
`uot/solvers/gradient_ascent/gradient_ascent.py:44` (and in `SAGASolver`), so
the October failure still reproduces exactly on current code. The flag is a
no-op for the other 12 distributions, which have no exact-zero weights.

Still filter on `status == "success"` before computing any hit rate: the
retained failed rows have no `iterations`, and a heatmap counting
`iterations >= maxiter` would score them 0 and paint them as perfectly stable.

## Schema

Normalised to the `dashboard/data/cot1d/` column order, plus
`tol`, `learning_rate`, `momentum`, `residual_l2`, `exception`.

- **`error` is RECONSTRUCTED, not original data.** The source files have no
  `error` column at all. It was added here as a verbatim copy of the source
  `residual_l2` column, which is `max_i ||marginal_i - a_i||_2` — the same
  quantity the current schema calls `error`
  (cf. `uot/solvers/sinkhorn/sinkhorn.py::_compute_error`). Without it these
  rows would read as NaN and be miscounted as numerical instability. The
  original `residual_l2` column is retained alongside it so the substitution
  stays auditable and reversible.
- `cost_rerr` is NaN; the dashboard derives it from the `lp` reference rows.
- The original `potentials` column was dropped: a stringified array repr,
  truncated with `...` at larger n, and ~90% of the 30 MB bulk. The untouched
  originals remain in the source directory above.
