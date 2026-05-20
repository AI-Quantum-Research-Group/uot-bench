# Writing your own Problem and Generator

`uot-bench` is designed to be extended.  After `pip install uot-bench` you can
define custom problem types and generators using only the classes exported from
the top-level `uot` package, then plug them directly into the experiment
infrastructure.

---

## 1. Extending `Problem`

Subclass `uot.Problem` (alias for `uot.problems.base_problem.Problem`) and
implement the four abstract methods:

| Method | Return type | Description |
|---|---|---|
| `get_marginals()` | `list[BaseMeasure]` | The marginal measures |
| `get_costs()` | `list[jax.Array]` | Cost matrices as JAX arrays |
| `to_dict()` | `dict` | Metadata stored in result DataFrames |
| `free_memory()` | `None` | Release any cached arrays |

```python
# my_problem.py
from __future__ import annotations
from collections.abc import Callable

import jax
import jax.numpy as jnp

from uot import Problem
from uot.data import BaseMeasure, PointCloudMeasure


class MyTwoMarginalProblem(Problem):
    """A minimal two-marginal OT problem for a custom data source."""

    def __init__(
        self,
        name: str,
        mu: PointCloudMeasure,
        nu: PointCloudMeasure,
        cost_fn: Callable,
    ) -> None:
        super().__init__(name, [mu, nu], [cost_fn])
        self._cost_fn = cost_fn
        self._C: list[jax.Array] | None = None

    # --- required abstract methods ---

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

### Using it in an Experiment

```python
import numpy as np
import jax.numpy as jnp

from uot import Experiment, SolverConfig
from uot.data import PointCloudMeasure
from uot.solvers import SinkhornTwoMarginalSolver
from uot.experiments.measurement import measure_time_and_output
from uot.utils.costs import cost_euclid_squared

from my_problem import MyTwoMarginalProblem

rng = np.random.default_rng(0)
X = rng.standard_normal((100, 2))
Y = rng.standard_normal((100, 2))
mu = PointCloudMeasure(X, np.ones(100) / 100)
nu = PointCloudMeasure(Y, np.ones(100) / 100)

problem = MyTwoMarginalProblem("demo", mu, nu, cost_euclid_squared)

experiment = Experiment("demo", measure_time_and_output)
solver = SinkhornTwoMarginalSolver()
result = experiment.run_single(problem, solver, reg=0.01, maxiter=500)
print(result)
```

---

## 2. Extending `Generator`

Subclass `uot.Generator` (alias for `uot.problems.problem_generator.Generator`)
and implement the single abstract method `generate(self) -> Iterator[Problem]`.
Store all hyper-parameters in `__init__` — `generate()` takes no arguments.

```python
# my_generator.py
from __future__ import annotations
from collections.abc import Iterator

import numpy as np

from uot import Generator, TwoMarginalProblem
from uot.data import PointCloudMeasure
from uot.utils.costs import cost_euclid_squared


class GaussianPairGenerator(Generator):
    """Yields pairs of random Gaussian point clouds."""

    def __init__(self, n: int = 200, dim: int = 2, seed: int = 0) -> None:
        self.n = n
        self.dim = dim
        self._rng = np.random.default_rng(seed)

    def generate(self) -> Iterator[TwoMarginalProblem]:
        for i in range(10):
            X = self._rng.standard_normal((self.n, self.dim))
            Y = self._rng.standard_normal((self.n, self.dim))
            w = np.ones(self.n) / self.n
            mu = PointCloudMeasure(X, w)
            nu = PointCloudMeasure(Y, w)
            yield TwoMarginalProblem(
                f"gaussian_pair_{i}",
                mu,
                nu,
                cost_euclid_squared,
            )
```

### Using it in `run_pipeline`

```python
from uot import Experiment, SolverConfig, run_pipeline
from uot.solvers import SinkhornTwoMarginalSolver, LBFGSTwoMarginalSolver
from uot.problems.iterator import OnlineProblemIterator
from uot.experiments.measurement import measure_time_and_output

from my_generator import GaussianPairGenerator

experiment = Experiment("comparison", measure_time_and_output)

gen = GaussianPairGenerator(n=200, dim=2, seed=42)
problems = list(gen.generate())   # or use OnlineProblemIterator(gen)

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

df = run_pipeline(experiment, solvers, [problems], folds=1)
print(df[["name", "reg", "time", "cost"]].to_string())
```

---

## 3. Tips

- **Cost functions** live in `uot.utils.costs` (`cost_euclid_squared`, etc.).
- **Measures**: `PointCloudMeasure(points, weights)` for scattered points;
  `GridMeasure(axes, weights_nd)` for regular grids.
- `Generator.one()` returns the first generated problem — useful for quick
  inspection without iterating.
- Generators are used for lazy evaluation. Yield rather than returning a list
  to keep memory usage low for large datasets.
- `Problem.key()` returns a SHA-1 hash of the problem state — it is used by the
  storage layer to deduplicate saved problems.
