# uot-bench

uot-bench is a Python toolkit for optimal transport solvers and benchmarking.
It provides JAX-first implementations of common OT methods, utilities for
generating problems and measures, and a configurable benchmarking pipeline for
running experiments at scale.

- Documentation index: [docs/index.md](docs/index.md)
- Problems module: [docs/problems.md](docs/problems.md)
- Problem generators: [docs/generators.md](docs/generators.md)
- SLURM guide: [docs/slurm.md](docs/slurm.md)
- Color transfer experiment: [docs/color_transfer.md](docs/color_transfer.md)
- MNIST classification experiment: [docs/mnist.md](docs/mnist.md)

## Install

Core install:
```
pip install uot-bench
```

Extras (optional):
```
pip install "uot-bench[viz,image-analysis,color-transfer,gurobi]"
```

CUDA (optional, JAX):
```
pip install "uot-bench[cuda12]"
```

## Quickstart (Python API)

A minimal two-marginal Sinkhorn example:

```python
import numpy as np

from uot.data.measure import DiscreteMeasure
from uot.problems.two_marginal import TwoMarginalProblem
from uot.solvers.sinkhorn import SinkhornTwoMarginalSolver
from uot.utils.costs import cost_euclid_squared

# Create two 1D discrete measures
x = np.linspace(0.0, 1.0, 64).reshape(-1, 1)
y = np.linspace(0.0, 1.0, 64).reshape(-1, 1)

a = np.exp(-((x - 0.3) ** 2) / 0.01).reshape(-1)
a = a / a.sum()
b = np.exp(-((y - 0.7) ** 2) / 0.02).reshape(-1)
b = b / b.sum()

mu = DiscreteMeasure(x, a, name="mu")
nu = DiscreteMeasure(y, b, name="nu")

problem = TwoMarginalProblem("toy", mu, nu, cost_euclid_squared)
solver = SinkhornTwoMarginalSolver()

result = solver.solve(
    marginals=problem.get_marginals(),
    costs=problem.get_costs(),
    reg=1e-2,
)

print("cost:", float(result["cost"]))
```

## Benchmarking CLI tools (Pixi)

This project uses [Pixi](https://prefix.dev/docs/pixi/) to manage
benchmarking dependencies and tasks.

### Installing Pixi

- [Linux](https://prefix.dev/docs/pixi/install#linux)
- [macOS](https://prefix.dev/docs/pixi/install#macos)
- [Windows](https://prefix.dev/docs/pixi/install#windows)

After installation run `pixi install` to set up the environment. Available
commands can be invoked with `pixi run <task>`.

### Common commands

- `pixi run serialize --config <config.yaml> --export-dir <directory>`
- `pixi run benchmark --config <config.yaml> --folds <n> --export <file>`
- `pixi run lint` or `ruff check .`

## Slurm

### On Compute Canada clusters

The generic script for both SLURM and local runs is `scripts/run_benchmark.sh`.
Example:
```
sbatch scripts/run_benchmark.sh configs/generators/gaussians.yaml configs/runners/gaussians.yaml
```

Monitor GPU usage with:
```
srun --jobid 123456 --pty watch -n 30 nvidia-smi
```

## Synthetic datasets

Create a generator config like:

```yaml
generators:
  1D-gaussians-64:
    generator: uot.problems.generators.GaussianMixtureGenerator
    dim: 1
    num_components: 1
    n_points: 64
    num_datasets: 30
    borders: (-6, 6)
    cost_fn: uot.utils.costs.cost_euclid_squared
    use_jax: true
    seed: 42
```

Then serialize:
```
pixi run serialize --config configs/generators/gaussians.yaml --export-dir datasets/synthetic
```

## Running experiments

Example config:

```yaml
param-grids:
  regularizations:
    - reg: 1
    - reg: 0.001

solvers:
  sinkhorn:
    solver: uot.solvers.sinkhorn.SinkhornTwoMarginalSolver
    jit: true
    param-grid: regularizations

problems:
  dir: datasets/synthetic
  names:
    - 1D-gaussians-64
  
experiment: 
  name: Measure on Gaussians
  function: uot.experiments.measurement.measure_time
```

Run the benchmark:
```
pixi run benchmark --config configs/runners/gaussians.yaml --folds 1 --export results/raw/gaussians.csv
```

## Color Transfer

Example config:

```yaml
param-grids:
  epsilons:
    - reg: 1
    - reg: 0.01

solvers:
  sinkhorn:
    solver: uot.solvers.sinkhorn.SinkhornTwoMarginalSolver
    param-grid: epsilons
    jit: true

bin-number:
  - 16
  - 32
soft-extension:
  - no
  - yes
displacement-interpolation:
  - 0.0
  - 1.0
color-space: rgb
# active-channels: [r, g]
batch-size: 100000
pair-number: 3
images-dir: ./datasets/images
rng-seed: 42

drop-columns:
  - transport_plan
  - monge_map
  - u_final
  - v_final

experiment: 
  name: Time and test
  output-dir: ./outputs/color_transfer
```

For detailed explanation of parameters, see [docs/color_transfer.md](docs/color_transfer.md).
Notably, `soft-extension` and `displacement-interpolation` can be single values
or lists, and the pipeline will run once per option.

Example: Lab space with selected channels.
```yaml
color-space: lab
active-channels: [l, a]
```

Run the experiment:
```
pixi run color-transfer --config ./configs/color_transfer/example.yaml
```

Dashboard visualization:
```
pixi run color-transfer-visualization --origin_folder <path_to_input_images> --results_folder <path_to_resulting_images>
```

## MNIST Classification

The MNIST experiment is performed in two steps:

- Distance matrix calculation
- Classification

See [docs/mnist.md](docs/mnist.md) for configs.

Step 1:
```
pixi run mnist_distances --config ./configs/mnist_dist_example.yaml
```

Step 2:
```
pixi run mnist_classification --config ./configs/mnist_classification_example.yaml
```

## Linting

Run `pixi run lint` or `ruff check .`.
