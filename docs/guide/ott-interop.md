# OTT-JAX interoperability

`uot-bench` ships an **optional** adapter that exposes
[OTT-JAX](https://ott-jax.readthedocs.io) solvers as native
`BaseSolver` subclasses. Once installed, OTT solvers run inside the same
`Generator → Problem → BaseSolver → Experiment → run_pipeline → DataFrame`
pipeline as the built-in solvers — same YAML configs, SLURM scripts and Dash
dashboard.

The adapter lives under `uot.interop.ott` and is gated behind a clean
`ImportError`: nothing in `uot.interop.ott` is imported by core `uot`, so
`uot-bench` still installs and runs end-to-end without `ott-jax`.

## Install

```bash
pip install "uot-bench[ott]"   # adds ott-jax>=0.4.7
```

Importing the package without `ott-jax` raises a clear, actionable error at the
`uot.interop.ott` boundary (not at top-level `import uot`).

## Available solvers

| Wrapper | Wraps | `input_kind` |
|---|---|---|
| `OTTSinkhornSolver` | `ott.solvers.linear.sinkhorn.Sinkhorn` | `ott_linear` |
| `OTTLRSinkhornSolver` | `ott.solvers.linear.sinkhorn_lr.LRSinkhorn` | `ott_linear` |
| `OTTGromovWassersteinSolver` | `ott.solvers.quadratic.gromov_wasserstein.GromovWasserstein` | `ott_quadratic` |
| `OTTLRGromovWassersteinSolver` | `ott.solvers.quadratic.gromov_wasserstein_lr.LRGromovWasserstein` | `ott_quadratic` |
| `OTTSinkhornDivergence` | `ott.tools.sinkhorn_divergence` | `marginals_costs` |
| `OTTDiscreteBarycenterSolver` | `ott.solvers.linear.continuous_barycenter` | `ott_barycenter` |
| `OTTGWBarycenterSolver` | `ott.solvers.quadratic.gw_barycenter` | `ott_gw_barycenter` |
| `OTTUnivariateSolver` | `ott.solvers.linear.univariate` | `marginals_costs` |

```python
from uot.interop.ott import (
    OTTSinkhornSolver,
    OTTLRSinkhornSolver,
    OTTGromovWassersteinSolver,
    OTTLRGromovWassersteinSolver,
    OTTSinkhornDivergence,
    OTTDiscreteBarycenterSolver,
    OTTGWBarycenterSolver,
    OTTUnivariateSolver,
)
```

## Calling convention — pre-built OTT problems

OTT linear/quadratic/barycenter wrappers do **not** take `(marginals, costs)`.
They declare a non-default
[`input_kind`](custom-solver.md#representation-negotiation-input_kind) so the
runner builds the OTT problem object **once, outside the timed solve region**,
and passes it as the first positional argument to `solve`. This keeps the timed
section free of translation overhead.

When calling a wrapper directly (outside the harness), build the OTT problem
first via the `to_ott_*` helpers:

```python
from uot.interop.ott import OTTSinkhornSolver

linear_problem = problem.to_ott_linear_problem(epsilon=1e-2)
out = OTTSinkhornSolver(max_iterations=4000, threshold=1e-6).solve(
    linear_problem, epsilon=1e-2,
)
print(float(out["cost"]), out["converged"])
```

`OTTSinkhornDivergence` and `OTTUnivariateSolver` keep `input_kind =
"marginals_costs"` (they need raw point arrays), so they are called the usual
way: `solver.solve(marginals, costs, epsilon=...)`.

## Problem / measure → OTT helpers

These additive, lazily-imported helpers turn `uot` objects into first-class OTT
objects (useful even without the harness):

| Method | Returns |
|---|---|
| `TwoMarginalProblem.to_ott_linear_problem(tau_a, tau_b, scale_cost, batch_size, epsilon)` | `ott …LinearProblem` |
| `TwoMarginalProblem.to_ott_quadratic_problem(fused_penalty, tau_a, tau_b, …)` | `ott …QuadraticProblem` (GW / fused-GW) |
| `BarycenterProblem.to_ott_barycenter_problem(epsilon, cost_name, …)` | `ott …FreeBarycenterProblem` |
| `PointCloudMeasure.to_ott_geometry(other=None, cost_name, scale_cost, batch_size, epsilon)` | `ott PointCloud` geometry |
| `GridMeasure.to_ott_geometry(other=None, epsilon)` | `ott Grid` geometry |

```python
# Direct OTT usage, no harness:
from ott.solvers.linear.sinkhorn import Sinkhorn

lin = problem.to_ott_linear_problem(epsilon=0.1, tau_a=0.8, tau_b=0.8)  # unbalanced
out = Sinkhorn(max_iterations=1000)(lin)
```

Unbalanced OT is configured through `tau_a` / `tau_b`, which live on the OTT
`LinearProblem` — pass them to `to_ott_linear_problem(...)`, or, under the
harness, put them in the solver's `param_grid` (the representation builder threads
them into the problem). Setting them only on the wrapper's `__init__` has no
effect on the pre-built-problem path.

## YAML configs

Drop-in runner configs ship under `configs/runners/cot/`:
`ott_sinkhorn.yaml`, `ott_lr_sinkhorn.yaml`, `ott_gw.yaml`.

```yaml
param-grids:
  sinkhorn:
    - epsilon: !!float 1e-1
    - epsilon: !!float 1e-2

solvers:
  ott-sinkhorn:
    solver: uot.interop.ott.OTTSinkhornSolver
    jit: false
    use-cost-matrix: false
    param-grid: sinkhorn

experiment:
  name: OTT Sinkhorn Solver
  function: uot.experiments.measurement.measure_with_gpu_tracker
```

Set `use-cost-matrix: false` — OTT wrappers build their own geometry and do not
need the eager `(n×m)` cost matrix. Per-run `epsilon` is baked into the OTT
geometry by the runner; structural hyperparameters (`lse_mode`, `threshold`,
`max_iterations`, `rank`, `gamma`, …) are forwarded to the wrapper's `__init__`
by `instantiate_solver` (extra keys are ignored).

## Output translation

OTT output pytrees are translated to the `SolverOutput` `TypedDict` in
`uot.interop.ott._outputs`:

- `cost` uses OTT's `primal_cost` (`<C, P>`) for Sinkhorn / LR-Sinkhorn, and the
  final outer cost for GW — comparable to what native solvers store in `cost`.
- Low-rank Sinkhorn stores its factors in `low_rank_plan = (Q, R, g)` instead of
  materialising the dense `Q diag(1/g) Rᵀ` coupling.
- GW's `costs` array is padded with `-1.0` for outer steps that never ran; the
  adapter filters those sentinels before reporting the final cost and iteration
  count.
- `error` is the last finite entry of OTT's `errors` array.

## Notes & caveats

- **`scale_cost`** defaults to `1.0` to match `uot`'s eager-matrix semantics.
  OTT's `"mean"` / `"max_cost"` rescale the cost matrix and shift the effective
  `epsilon`; opt in explicitly if you want them.
- **dtype**: set `JAX_ENABLE_X64=True` (and `jax.config.update("jax_enable_x64",
  True)`) for fp64 parity with `uot`'s SLURM defaults; most published OTT numbers
  are fp32.
- **Low-rank plans**: downstream code that expects a dense `transport_plan` must
  handle the `low_rank_plan` factors instead.

## Worked example

See `notebooks/ott_interop_benchmark.ipynb` for an end-to-end comparison of
native vs OTT Sinkhorn, low-rank Sinkhorn, Gromov–Wasserstein and Sinkhorn
divergence, including transport-plan visualizations.

For the full design rationale and a `uot-bench` ↔ OTT-JAX feature comparison, see
`COMPARISON_OTT_JAX.md` at the repository root.
