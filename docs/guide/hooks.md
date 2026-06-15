# Post-solve hooks

Hooks let you attach domain-specific post-processing to an `Experiment` without
forking the generic runner. A hook runs **after** every `solve_fn` call and can
add metrics to the result row, or fan it out into multiple rows (e.g. one per
post-processing mode).

## The `PostSolveHook` protocol

```python
from uot.experiments.hooks import PostSolveHook

class PostSolveHook(Protocol):
    def __call__(
        self,
        problem: Problem,
        view: Any,
        metrics: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any] | list[dict[str, Any]] | None: ...
```

Return values:

| Return value | Effect |
|---|---|
| `None` | No change; the current row is kept as-is. |
| `dict` | Merged into the current row (keys override existing values). |
| `list[dict]` | **Replaces** the current row with one row per list element (fan-out). |

`view` is the representation object the runner built before the timed solve (a
`SolverInputs` for native solvers, or a pre-built OTT problem for OTT wrappers).
`context` carries `problem_index`, `solver_name`, and `solver_kwargs`.

## Registering hooks

### Experiment-level hooks

Pass a list to `Experiment(…, hooks=[…])`:

```python
from uot.experiments import Experiment
from uot.experiments.measurement import measure_time_and_output
from uot.experiments.real_data.color_transfer.hooks import ColorTransferHook

hook = ColorTransferHook(output_dir="output/color_transfer")
experiment = Experiment(name="CT", solve_fn=measure_time_and_output, hooks=[hook])
```

Experiment-level hooks run after problem-level hooks (see below) on every problem.

### Problem-level hooks

Override `Problem.post_solve_hooks()` to attach hooks scoped to a specific
problem type:

```python
class MyProblem(TwoMarginalProblem):
    def post_solve_hooks(self) -> list:
        return [MyMetricHook()]
```

Problem-level hooks are prepended to any experiment-level hooks automatically.

## Hook chaining and fan-out

`apply_hooks` applies hooks in order. Each hook receives the **current row**, not
the original base metrics — so fan-out composes correctly:

- A first hook that returns `[row_a, row_b]` (fan-out) causes the second hook to
  be applied separately to both `row_a` and `row_b`.
- A second hook that returns a `dict` merges into every current row.

## Built-in hook: `ColorTransferHook`

`uot.experiments.real_data.color_transfer.hooks.ColorTransferHook` reconstructs
transported images and computes domain metrics. It replaces the old
`ColorTransferExperiment` class.

```python
from uot.experiments.real_data.color_transfer.hooks import ColorTransferHook

hook = ColorTransferHook(
    output_dir="output/color_transfer",
    soft_extension_modes=[False, True],    # one result row per mode
    displacement_alphas=[1.0, 0.5],        # × one row per alpha
    drop_columns=["transport_plan"],
)
```

It fans out into `len(soft_extension_modes) × len(displacement_alphas)` result
rows, saving a transported image per row and computing distribution and image
quality metrics.

## API reference

::: uot.experiments.hooks.PostSolveHook
    options:
      show_root_heading: true
      show_source: false

::: uot.experiments.hooks.apply_hooks
    options:
      show_root_heading: true
      show_source: false
