# Writing a custom Problem

Subclass `uot.Problem` to define a custom OT problem type and plug it into
`Experiment` or `run_pipeline`.

## Abstract methods

You must implement four methods:

| Method | Return type | Description |
|---|---|---|
| `get_marginals()` | `list[BaseMeasure]` | Return the marginal measures. |
| `get_costs()` | `list[jax.Array]` | Return cost matrices (one per marginal pair). Compute lazily and cache. |
| `to_dict()` | `dict` | Metadata that becomes columns in the result DataFrame. |
| `free_memory()` | `None` | Drop any cached arrays (called by the runner to control memory). |

`get_lambdas()` is optional — override it for barycenter problems to return
the `lambdas` array.

## Minimal example

```python
# my_problem.py
from __future__ import annotations
from collections.abc import Callable

import jax
import jax.numpy as jnp

from uot import Problem
from uot.data import BaseMeasure, PointCloudMeasure
from uot.utils.types import ArrayLike


class MyTwoMarginalProblem(Problem):
    def __init__(
        self,
        name: str,
        mu: PointCloudMeasure,
        nu: PointCloudMeasure,
        cost_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    ) -> None:
        super().__init__(name, [mu, nu], [cost_fn])
        self._cost_fn = cost_fn
        self._C: list[jax.Array] | None = None

    def get_marginals(self) -> list[BaseMeasure]:
        return self.measures

    def get_costs(self) -> list[jax.Array]:
        if self._C is None:
            X, _ = self.measures[0].as_point_cloud()
            Y, _ = self.measures[1].as_point_cloud()
            self._C = [self._cost_fn(X, Y)]
        return self._C

    def to_dict(self) -> dict:
        n_mu = self.measures[0].as_point_cloud()[1].shape[0]
        n_nu = self.measures[1].as_point_cloud()[1].shape[0]
        return {
            "dataset": self.name,
            "type": "my_two_marginal",
            "n_mu": n_mu,
            "n_nu": n_nu,
            "cost": self.cost_name,
        }

    def free_memory(self) -> None:
        self._C = None
```

## Getting data into a solver

`Problem` provides three input-bundle methods. Call one to build a dataclass
that groups everything a solver needs:

| Method | Returns | Use when |
|---|---|---|
| `solver_inputs()` | `SolverInputs` | Generic — marginals + costs list + metadata. Most solvers use this. |
| `point_cloud_inputs()` | `PointCloudInputs` | When all marginals share the same support (aligned point clouds). |
| `grid_inputs()` | `GridInputs` | When all marginals are `GridMeasure` instances on the same grid. |

```python
inputs = problem.solver_inputs()
result = solver.solve(marginals=inputs.marginals, costs=inputs.costs, reg=0.01)
```

## `free_memory()` semantics

The runner calls `problem.free_memory()` after each solver call to release
cached cost matrices. If your problem caches large arrays (e.g. a pre-computed
`n × n` cost matrix), ensure `free_memory()` sets the cache to `None`.
Failing to do this causes memory to accumulate across all problems in a fold.

## `to_dict()` semantics

The dict returned by `to_dict()` is merged with solver metadata and measurement
results to form one DataFrame row. Choose keys that identify the problem
unambiguously (distribution family, dimension, number of points, etc.).
Avoid returning large arrays — only scalar metadata.

## Built-in concrete problem types

| Class | Description |
|---|---|
| `TwoMarginalProblem` | Standard 2-marginal OT. Constructor: `(name, mu, nu, cost_fn)`. |
| `BarycenterProblem` | N-marginal barycenter with weights `lambdas`. |
| `MultiMarginalProblem` | General N-marginal OT. |

## Using the custom problem in an experiment

```python
import numpy as np
from uot import Experiment
from uot.data import PointCloudMeasure
from uot.solvers import SinkhornTwoMarginalSolver
from uot.experiments.measurement import measure_time_and_output
from uot.utils.costs import cost_euclid_squared

from my_problem import MyTwoMarginalProblem

rng = np.random.default_rng(0)
mu = PointCloudMeasure(rng.standard_normal((100, 2)), np.ones(100) / 100)
nu = PointCloudMeasure(rng.standard_normal((100, 2)), np.ones(100) / 100)

problem = MyTwoMarginalProblem("demo", mu, nu, cost_euclid_squared)
experiment = Experiment("demo", measure_time_and_output)
result = experiment.run_single(problem, SinkhornTwoMarginalSolver(), reg=0.01)
print(result)
```
