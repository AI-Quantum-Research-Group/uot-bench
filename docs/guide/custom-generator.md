# Writing a custom Generator

Subclass `uot.Generator` to produce a sequence of `Problem` instances.
Generators are used both programmatically (`run_pipeline`) and in YAML
configs via the `generator:` key.

## Contract

```python
class Generator(ABC):
    @abstractmethod
    def generate(self) -> Iterator[Problem]: ...
```

- All hyper-parameters go in `__init__`.
- `generate()` takes **no arguments** — it uses only `self`.
- Yield problems lazily to keep memory low for large datasets.

## Minimal example

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

    def __init__(self, n: int = 200, dim: int = 2, num_datasets: int = 10, seed: int = 0) -> None:
        self.n = n
        self.dim = dim
        self._num_datasets = num_datasets   # used by OnlineProblemIterator
        self._rng = np.random.default_rng(seed)

    def generate(self) -> Iterator[TwoMarginalProblem]:
        for i in range(self._num_datasets):
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

## Helper methods on `Generator`

| Method | Description |
|---|---|
| `one()` | Returns the first problem from `generate()`. Useful for quick inspection. |
| `solver_inputs()` | Calls `one().solver_inputs()`. |
| `point_cloud_inputs()` | Calls `one().point_cloud_inputs()`. |
| `grid_inputs()` | Calls `one().grid_inputs()`. |

## Using with `run_pipeline`

```python
from uot import Experiment, SolverConfig, run_pipeline
from uot.solvers import SinkhornTwoMarginalSolver, LBFGSTwoMarginalSolver
from uot.experiments.measurement import measure_time_and_output

from my_generator import GaussianPairGenerator

gen = GaussianPairGenerator(n=200, dim=2, num_datasets=10, seed=42)
problems = list(gen.generate())

experiment = Experiment("comparison", measure_time_and_output)
solvers = [
    SolverConfig("Sinkhorn", SinkhornTwoMarginalSolver,
                 param_grid=[{"reg": 0.01, "maxiter": 500}]),
    SolverConfig("LBFGS", LBFGSTwoMarginalSolver,
                 param_grid=[{"reg": 0.01, "maxiter": 200}]),
]

df = run_pipeline(experiment, solvers, [problems], folds=1)
print(df[["dataset", "solver", "reg", "time"]].to_string())
```

## Online vs serialized

| Approach | When to use |
|---|---|
| `list(gen.generate())` | Small datasets or one-off runs. All problems are held in memory. |
| `OnlineProblemIterator(gen)` | Large datasets. Problems are generated on demand and discarded after use. |
| `uot-serialize` → `ProblemStore` / `HDF5ProblemStore` | When you want to reuse the same dataset across many benchmark runs without regenerating. |

```python
from uot.problems.iterator import OnlineProblemIterator
problems_iter = OnlineProblemIterator(gen, num_datasets=10, cache_gt=False)
df = run_pipeline(experiment, solvers, [problems_iter], folds=1)
```

## Using in YAML configs

Any `Generator` subclass can be referenced from a generator config by its fully
qualified class name. All `__init__` parameters become YAML keys:

```yaml
generators:
  my-dataset:
    generator: mypackage.generators.GaussianPairGenerator
    n: 128
    dim: 2
    num_datasets: 30
    seed: 7
```

Then serialize:

```bash
uot-serialize --config my_generators.yaml --export-dir datasets/my-dataset
```

See [Generating datasets](../cli/serialize.md) for the full YAML schema.
