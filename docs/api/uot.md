# `uot` — top-level API

All primary classes are importable directly from `uot`:

```python
from uot import (
    # Core abstractions — subclass these
    Problem,
    Generator,

    # Concrete problem types
    TwoMarginalProblem,
    BarycenterProblem,
    MultiMarginalProblem,

    # Input/output dataclasses
    SolverInputs,
    PointCloudInputs,
    GridInputs,

    # Solver infrastructure
    BaseSolver,
    SolverConfig,

    # Experiment runner
    Experiment,
    run_pipeline,

    # Measures
    BaseMeasure,
    PointCloudMeasure,
    GridMeasure,
)
```

| Symbol | Module | Description |
|---|---|---|
| `Problem` | `uot.problems.base_problem` | Abstract base for OT problems. Subclass this. |
| `Generator` | `uot.problems.problem_generator` | Abstract base for problem factories. Subclass this. |
| `TwoMarginalProblem` | `uot.problems.two_marginal` | Standard 2-marginal OT: `(name, mu, nu, cost_fn)`. |
| `BarycenterProblem` | `uot.problems.barycenter_problem` | N-marginal barycenter with `lambdas` weights. |
| `MultiMarginalProblem` | `uot.problems.multi_marginal` | General N-marginal OT. |
| `SolverInputs` | `uot.problems.base_problem` | Frozen dataclass: `marginals`, `costs`, `lambdas`, `cost_fn`, `cost_name`, `is_squared_euclidean`. |
| `PointCloudInputs` | `uot.problems.base_problem` | Inputs pre-projected to a shared point cloud support. |
| `GridInputs` | `uot.problems.base_problem` | Inputs as a regular grid for grid-based solvers. |
| `BaseSolver` | `uot.solvers.base_solver` | Abstract solver. Implement `solve(marginals, costs, **kwargs) -> SolverOutput`. |
| `SolverConfig` | `uot.solvers.solver_config` | Bundles a solver class + param grid for `run_pipeline`. |
| `Experiment` | `uot.experiments.experiment` | Pairs a `solve_fn` (measurement function) with solver calls. |
| `run_pipeline` | `uot.experiments.runner` | Sweeps an `Experiment` over all `(problem, solver, param)` combinations. Returns `pd.DataFrame`. |
| `BaseMeasure` | `uot.data.measure` | Abstract base for measures. |
| `PointCloudMeasure` | `uot.data.measure` | Scattered points + weights. |
| `GridMeasure` | `uot.data.measure` | Regular grid + N-D weight tensor. |

For detailed API documentation see the module pages below.
