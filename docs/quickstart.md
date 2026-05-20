# Quickstart

All primary classes are available directly from `uot`:

```python
from uot import (
    Problem, Generator,          # base classes to subclass
    TwoMarginalProblem,          # concrete problem types
    BarycenterProblem,
    BaseSolver, SolverConfig,    # solver infrastructure
    Experiment, run_pipeline,    # experiment runner
)
```

## Two-marginal Sinkhorn (minimal)

```python
import numpy as np
from uot import TwoMarginalProblem
from uot.data import PointCloudMeasure
from uot.solvers import SinkhornTwoMarginalSolver
from uot.utils.costs import cost_euclid_squared

x = np.linspace(0.0, 1.0, 64).reshape(-1, 1)
y = np.linspace(0.0, 1.0, 64).reshape(-1, 1)
a = np.exp(-((x - 0.3) ** 2) / 0.01).reshape(-1); a /= a.sum()
b = np.exp(-((y - 0.7) ** 2) / 0.02).reshape(-1); b /= b.sum()

mu = PointCloudMeasure(x, a, name="mu")
nu = PointCloudMeasure(y, b, name="nu")

problem = TwoMarginalProblem("toy", mu, nu, cost_euclid_squared)
inputs = problem.solver_inputs()

result = SinkhornTwoMarginalSolver().solve(
    marginals=inputs.marginals,
    costs=inputs.costs,
    reg=1e-2,
)
print("cost:", float(result["cost"]))
```

## Running multiple solvers in a pipeline

```python
from uot import Experiment, SolverConfig, run_pipeline
from uot.solvers import SinkhornTwoMarginalSolver, LBFGSTwoMarginalSolver
from uot.experiments.measurement import measure_time_and_output

experiment = Experiment("comparison", measure_time_and_output)

solvers = [
    SolverConfig(
        name="Sinkhorn",
        solver=SinkhornTwoMarginalSolver,
        param_grid=[{"reg": 0.01, "maxiter": 500}],
    ),
    SolverConfig(
        name="LBFGS",
        solver=LBFGSTwoMarginalSolver,
        param_grid=[{"reg": 0.01, "maxiter": 200}],
    ),
]

# problems is a list of Problem instances
df = run_pipeline(experiment, solvers, [problems], folds=1)
print(df[["name", "reg", "time", "cost"]].to_string())
```

## Writing your own Problem or Generator

See [Writing custom Problem/Generator](extending.md) for full worked examples.
