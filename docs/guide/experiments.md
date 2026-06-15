# Running experiments in Python

The experiment infrastructure has three layers of increasing scope.
Choose the one that matches what you're doing.

## `experiment.run_single` — one problem, one solver call

```python
from uot import Experiment
from uot.experiments.measurement import measure_time_and_output

experiment = Experiment("timing", measure_time_and_output)
result = experiment.run_single(problem, solver, reg=0.01, maxiter=500)
# result is a dict — keys depend on the measurement function
print(result)
```

Use this for interactive exploration or debugging a single case.

## `experiment.run_on_problems` — one solver over many problems

```python
df = experiment.run_on_problems(problems, solver, reg=0.01, maxiter=500)
# df is a pandas.DataFrame with one row per problem
```

## `run_pipeline` — full sweep (recommended for benchmarking)

```python
from uot import Experiment, SolverConfig, run_pipeline
from uot.solvers import SinkhornTwoMarginalSolver, LBFGSTwoMarginalSolver
from uot.experiments.measurement import measure_time_and_output

experiment = Experiment(
    name="comparison",
    solve_fn=measure_time_and_output,
    # hooks=[MyPostSolveHook()],   # optional; see guide/hooks.md
)

solvers = [
    SolverConfig(
        name="Sinkhorn",
        solver=SinkhornTwoMarginalSolver,
        param_grid=[{"reg": 0.01, "maxiter": 500},
                    {"reg": 0.001, "maxiter": 1000}],
    ),
    SolverConfig(
        name="LBFGS",
        solver=LBFGSTwoMarginalSolver,
        param_grid=[{"reg": 0.01}],
    ),
]

df = run_pipeline(experiment, solvers, [problems], folds=3)
df.to_csv("results.csv", index=False)
```

`run_pipeline` sweeps every `(problem, solver, param_set)` combination,
repeats `folds` times, and returns a single `pd.DataFrame`.

### `SolverConfig` fields

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Label used in the `solver` column of the output CSV. |
| `solver` | `type[BaseSolver]` | The solver **class** (not an instance). |
| `param_grid` | `list[dict[str, Any]]` | One dict per param combination. Each becomes a separate run. |
| `is_jit` | `bool` | Wrap the solver in a JIT call. Default `False`. |
| `use_cost_matrix` | `bool` | Pass a pre-computed cost matrix. Default `True`. |

## Measurement functions

The `solve_fn` argument to `Experiment` must match the `SolveFn` Protocol:

```python
class SolveFn(Protocol):
    def __call__(self, prob, instance, view, **kwargs) -> dict[str, Any]: ...
```

`view` is the prepared representation the runner builds before the timed solve.
For the default `input_kind = "marginals_costs"` it is a `SolverInputs` dataclass
(`view.marginals`, `view.costs`); other solver kinds receive a backend-specific
pre-built problem object. The built-in measurement functions handle both via
`uot.experiments.measurement.invoke_solver`, so you rarely touch `view` directly.
See [Writing a custom Solver → Representation negotiation](custom-solver.md#representation-negotiation-input_kind).

The built-in options are in `uot.experiments.measurement`:

| Function | Added columns |
|---|---|
| `measure_time` | `time` (milliseconds) |
| `measure_time_and_output` | `time` + all solver output keys (`cost`, `transport_plan`, `iterations`, `error`, etc.) |
| `measure_solution_precision` | `cost_rerr` — relative error vs. the exact LP cost. Requires `TwoMarginalProblem` (calls `problem.get_exact_cost()`). |
| `measure_with_gpu_tracker` | `time`, `time_unit`, `time_counter`, `peak_gpu_mem`, `combined_peak_gpu_ram`, `gpu_mem_unit`, `peak_gpu_util_pct`, `mean_gpu_util_pct`, `peak_ram_MiB`, `combined_peak_ram_MiB`, `max_cpu_util_pct`, `mean_cpu_util_pct`. Requires `pip install "uot-bench[profiling]"`. |

You can also pass a custom callable that matches `SolveFn`.

## DataFrame schema

Every row in the output `pd.DataFrame` is the union of:

1. **Problem metadata** — keys from `problem.to_dict()`.
   For `TwoMarginalProblem` these are `dataset`, `type`, `n_mu`, `n_nu`, `cost`.
2. **Run metadata** — `solver` (name), `fold`, `status` (`"success"` or
   `"failed"`), `problem_index`, plus all parameter keys from the matching
   `param_grid` entry (e.g. `reg`, `maxiter`).
3. **Measurement output** — keys from the chosen `measure_*` function.

Failed solver calls produce a row with `status="failed"` and `exception=<message>`;
all measurement columns are absent (NaN after concat).

!!! note "Post-solve hooks can add or fan out rows"
    If the `Experiment` (or the problem, via `Problem.post_solve_hooks()`) has
    hooks, each successful solve may produce **more than one** row — e.g. the
    colour-transfer hook emits one row per post-processing mode. Hooks can also
    add extra metric columns. See [Post-solve hooks](hooks.md).

## YAML alternative

The same sweep can be driven from a YAML config without writing Python.
See [Running benchmarks](../cli/benchmark.md).
