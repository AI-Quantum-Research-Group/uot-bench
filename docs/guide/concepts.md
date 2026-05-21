# Core concepts

uot-bench is built around four abstractions that map onto the typical
optimal-transport workflow.

## The four objects

```
Generator  ──generates──▶  Problem  ──solver_inputs()──▶  BaseSolver
                                                               │
                                              Experiment ◀────┘ (wraps measurement)
                                                  │
                                           run_pipeline
                                                  │
                                           pandas.DataFrame
```

| Class | Role |
|---|---|
| `Problem` | Holds one OT problem: two or more marginal `BaseMeasure`s and cost function(s). Computes cost matrices on demand. |
| `Generator` | A factory that yields `Problem` instances. All hyper-parameters live in `__init__`; `generate()` takes nothing. |
| `BaseSolver` | Accepts marginals and cost arrays, returns a `SolverOutput` dict. |
| `Experiment` | Pairs a measurement function with a solver call. Used to abstract *what to measure* (time, precision, GPU usage) from *how to run it*. |
| `run_pipeline` | Sweeps an `Experiment` over every `(problem, solver, param_set, fold)` combination and returns a `pd.DataFrame`. |

## Canonical imports

```python
from uot import (
    Problem, Generator,           # base classes — subclass these
    TwoMarginalProblem,           # concrete problem (2 marginals)
    BarycenterProblem,            # barycenter (N marginals + lambdas)
    BaseSolver, SolverConfig,     # solver infrastructure
    Experiment, run_pipeline,     # experiment runner
    BaseMeasure,                  # measure base class
    PointCloudMeasure,            # scattered points + weights
    GridMeasure,                  # regular grid + weights
)
```

## Where to go next

- [Writing a custom Problem](custom-problem.md)
- [Writing a custom Generator](custom-generator.md)
- [Writing a custom Solver](custom-solver.md)
- [Running an Experiment in Python](experiments.md)
- [Measures and cost functions](measures.md)
- [CLI & YAML pipeline](../cli/index.md)
