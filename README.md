# uot-bench

[![PyPI](https://img.shields.io/pypi/v/uot-bench)](https://pypi.org/project/uot-bench/)
[![Python](https://img.shields.io/pypi/pyversions/uot-bench)](https://pypi.org/project/uot-bench/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**uot-bench** is a Python toolkit for optimal transport solvers and benchmarking.
It provides JAX-first implementations of common OT methods, utilities for generating
problems and measures, and a configurable pipeline for running experiments at scale.

> **Package name vs import name**: `pip install uot-bench`, then `import uot`.

Full documentation: [docs/](docs/index.md)

## Install

```bash
pip install uot-bench
```

Optional extras:

```bash
pip install "uot-bench[viz,color-transfer,gurobi]"
pip install "uot-bench[storage]"     # HDF5 problem store
pip install "uot-bench[profiling]"   # GPU resource tracking
pip install "uot-bench[mnist]"       # MNIST classification experiment
pip install "uot-bench[cuda12]"      # JAX with CUDA 12
pip install "uot-bench[all]"         # All optional extras
```

## 60-second example

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

See [docs/quickstart.md](docs/quickstart.md) for more examples, and
[docs/guide/custom-solver.md](docs/guide/custom-solver.md) to write your own solver.

## CLI cheatsheet

After `pip install uot-bench` the following console scripts are available.
Each is equivalent to the `python -m <module>` form shown alongside it.

| Console script | `python -m` equivalent | What it does | Schema |
|---|---|---|---|
| `uot-serialize --config X --export-dir Y` | `python -m uot.problems.problem_serializer` | Generate + persist problems to disk | [cli/serialize](docs/cli/serialize.md) |
| `uot-benchmark --config X --export results.csv` | `python -m uot.experiments.synthetic.benchmark` | Run experiment over problems × solvers, write CSV | [cli/benchmark](docs/cli/benchmark.md) |
| `uot-color-transfer --config X` | `python -m uot.experiments.real_data.color_transfer.color_transfer` | Color transfer experiment | [cli/color-transfer](docs/cli/color-transfer.md) |
| `uot-color-transfer-viz --origin_folder X --results_folder Y` | `python -m uot.experiments.real_data.color_transfer.visualization` | Launch visualization dashboard | |
| `uot-mnist-distances --config X` | `python -m uot.experiments.real_data.mnist_classification.count_pairwise_distances` | Step 1 of MNIST: pairwise OT distances | [cli/mnist](docs/cli/mnist.md) |
| `uot-mnist-classification --config X` | `python -m uot.experiments.real_data.mnist_classification.mnist_classification` | Step 2 of MNIST: KNN classification | [cli/mnist](docs/cli/mnist.md) |
| `uot-inspect-store --dataset X --outdir Y` | `python -m uot.problems.inspect_store` | Visualize a serialized problem dataset | |

## Writing your own Problem / Generator / Solver

Subclass `uot.Problem`, `uot.Generator`, or `uot.BaseSolver` and plug them
directly into `Experiment` and `run_pipeline`.

- [Writing a custom Problem](docs/guide/custom-problem.md)
- [Writing a custom Generator](docs/guide/custom-generator.md)
- [Writing a custom Solver](docs/guide/custom-solver.md)
- [Running experiments in Python](docs/guide/experiments.md)

## Linting

```bash
ruff check .
```
