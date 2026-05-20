# Running benchmarks

The benchmark command sweeps a set of solvers over a set of problems, repeating
for every parameter combination, and writes the results as a CSV file.

## Command

```bash
uot-benchmark --config configs/runners/cot/sinkhorn.yaml \
              --dataset datasets/synthetic \
              --folds 3 \
              --export results/sinkhorn_run.csv

# or equivalently:
python -m uot.experiments.synthetic.benchmark \
    --config configs/runners/cot/sinkhorn.yaml \
    --dataset datasets/synthetic \
    --folds 3 \
    --export results/sinkhorn_run.csv
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--config <path>` | required | Path to a runner YAML config. |
| `--export <path>` | `gaussian_toy_results.csv` | Full path (including filename) for the output CSV. Parent directories are created automatically. |
| `--folds <n>` | `1` | Number of times to repeat the experiment (e.g. for timing variance). |
| `--dataset <path>` | — | Path to a serialized dataset directory or `.h5` file. Overrides the `problems:` block in the config. |
| `--generators-config <path>` | — | Path to a generator config. Problems are generated online and added to the problem set. |
| `--progress` | `False` | Show a progress bar. |

## Three ways to source problems

| Method | When to use |
|---|---|
| `problems:` block in the runner config | Dataset was pre-serialized and lives in a known location. |
| `--dataset <dir\|.h5>` CLI flag | Override or supplement the config's `problems:` block at runtime. |
| `--generators-config <path>` | Generate problems on-the-fly; no serialization step needed. |

The three methods are additive — you can combine `--dataset` and
`--generators-config` in the same run.

## Runner config schema

```yaml
param-grids:
  <grid-name>:
    - <param-key>: <value>
      <param-key>: <value>
    - <param-key>: <value>

solvers:
  <solver-name>:
    solver: <fully.qualified.SolverClass>
    jit: <bool>
    use-cost-matrix: <bool>   # optional, default true
    param-grid: <grid-name>   # optional

problems:                     # optional — may be supplied via --dataset instead
  dir: <path>
  names:
    - <dataset-name>

experiment:
  name: <string>
  function: <fully.qualified.measure_function>
```

### `param-grids`

A named collection of parameter dictionaries. Each entry becomes one solver run.
Reference a grid from a `solvers:` entry via `param-grid: <grid-name>`.

```yaml
param-grids:
  regularizations:
    - reg: 1.0
      maxiter: 1000000
    - reg: 0.01
      maxiter: 1000000
    - reg: 0.001
      maxiter: 1000000
```

### `solvers`

Each entry names a solver and links it to a param grid. `solver:` must be the
fully qualified class name of a `BaseSolver` subclass — either a built-in
or your own:

```yaml
solvers:
  sinkhorn-log:
    solver: uot.solvers.sinkhorn.SinkhornTwoMarginalLogJaxSolver
    jit: true
    param-grid: regularizations

  my-solver:
    solver: mypackage.solvers.MySolver
    jit: false
    param-grid: regularizations
```

`jit: true` wraps the solver in a JIT-compiled call. `use-cost-matrix: false`
tells the runner not to pre-compute the cost matrix (useful for solvers that
compute it internally, e.g. `BackNForthSqEuclideanSolver`).

### `experiment.function`

One of the built-in measurement functions from `uot.experiments.measurement`:

| Function | Columns added to the output CSV |
|---|---|
| `uot.experiments.measurement.measure_time` | `time` |
| `uot.experiments.measurement.measure_time_and_output` | `time` + all solver output keys (e.g. `cost`, `iterations`, `error`) |
| `uot.experiments.measurement.measure_solution_precision` | `cost_rerr` (relative error vs exact LP cost) |
| `uot.experiments.measurement.measure_with_gpu_tracker` | `time`, `peak_gpu_mem`, `avg_gpu_mem`, `peak_cpu_mem`, and more |

### YAML anchors

Anchors avoid repeating common solver settings:

```yaml
defaults: &run
  jit: true

solvers:
  sinkhorn:
    <<: *run
    solver: uot.solvers.sinkhorn.SinkhornTwoMarginalSolver
    param-grid: regularizations

  lbfgs:
    <<: *run
    solver: uot.solvers.lbfgs.LBFGSTwoMarginalSolver
    param-grid: regularizations
```

## Output CSV schema

The output file has one row per `(problem, solver, param_set, fold)` combination.
Columns are the union of:

1. **Problem metadata** — keys returned by `problem.to_dict()`.
   For built-in problems this includes `dataset`, `type`, `n_mu`, `n_nu`, `cost`.
2. **Solver/run metadata** — `solver` (name from config), `fold`, plus all
   parameter keys from the matching param-grid entry (e.g. `reg`, `maxiter`).
3. **Measurement output** — keys produced by the chosen `experiment.function`
   (e.g. `time`, `cost`, `iterations`).

**Where results land:** `--export` is the full path. If the parent directory
does not exist it is created automatically. There is no timestamping —
use a meaningful filename like `results/sinkhorn_$(date +%Y%m%d).csv`.
To post-process or timestamp programmatically, call `run_pipeline` from Python
and write the returned `DataFrame` yourself.

## Running from Python

```python
from uot import Experiment, SolverConfig, run_pipeline
from uot.solvers import SinkhornTwoMarginalSolver, LBFGSTwoMarginalSolver
from uot.experiments.measurement import measure_time_and_output
from uot.problems.generators import GaussianMixtureGenerator

gen = GaussianMixtureGenerator(dim=1, num_components=1, n_points=64,
                                num_datasets=10, seed=42)
problems = list(gen.generate())

experiment = Experiment("timing", measure_time_and_output)
solvers = [
    SolverConfig("Sinkhorn", SinkhornTwoMarginalSolver,
                 param_grid=[{"reg": 0.01, "maxiter": 500}]),
    SolverConfig("LBFGS", LBFGSTwoMarginalSolver,
                 param_grid=[{"reg": 0.01, "maxiter": 200}]),
]

df = run_pipeline(experiment, solvers, [problems], folds=1)
df.to_csv("results.csv", index=False)
```

See [Running an Experiment in Python](../guide/experiments.md) for details.
