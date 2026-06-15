# Writing a custom Solver

Subclass `uot.BaseSolver` to implement a new OT algorithm and plug it into
`Experiment`, `run_pipeline`, or a YAML runner config.

## The contract

```python
class BaseSolver(ABC):
    requires_squared_euclidean: bool = False   # set True if your solver needs ‖x-y‖²
    input_kind: str = "marginals_costs"        # representation requested from the runner

    @abstractmethod
    def solve(
        self,
        marginals: Sequence[BaseMeasure],
        costs: Sequence[ArrayLike],
        *args,
        **kwargs,
    ) -> SolverOutput: ...
```

The `solve` signature above is the default `input_kind = "marginals_costs"`
contract. A solver can request a different prepared representation instead — see
[Representation negotiation](#representation-negotiation-input_kind) below.

**`SolverOutput`** is a `TypedDict` (from `uot.solvers.base_solver`).
Only `cost` is required; all other keys are `NotRequired`:

```python
class SolverOutput(TypedDict):
    cost: jax.Array | float
    transport_plan: NotRequired[jax.Array]
    coupling: NotRequired[jax.Array]        # alias kept for back-compat
    iterations: NotRequired[int]
    converged: NotRequired[bool]
    error: NotRequired[float | jax.Array]
    u_final: NotRequired[jax.Array]
    v_final: NotRequired[jax.Array]
    potentials: NotRequired[tuple[jax.Array, jax.Array]]
    monge_map: NotRequired[jax.Array]
    # Low-rank transport plan factors (Q, R, g); dense plan = Q @ diag(1/g) @ R.T
    low_rank_plan: NotRequired[tuple[jax.Array, jax.Array, jax.Array]]
    time: NotRequired[float]
```

Return only the keys your solver actually computes. Extra keys not in
`SolverOutput` are also fine — they appear as columns in the result CSV.

## Minimal example

```python
from collections.abc import Sequence

import jax.numpy as jnp

from uot import BaseSolver
from uot.data import BaseMeasure
from uot.solvers.base_solver import SolverOutput
from uot.utils.types import ArrayLike


class MeanCostSolver(BaseSolver):
    """Trivial baseline: returns the mean of the cost matrix as 'cost'."""

    def solve(
        self,
        marginals: Sequence[BaseMeasure],
        costs: Sequence[ArrayLike],
        **kwargs,
    ) -> SolverOutput:
        cost = jnp.mean(costs[0])
        return {"cost": cost}
```

## JAX-friendly example

A sketch of a Sinkhorn-style solver (see `uot/solvers/sinkhorn/sinkhorn.py`
for the full real implementation):

```python
from collections.abc import Sequence

import jax
import jax.numpy as jnp

from uot import BaseSolver
from uot.data import BaseMeasure, PointCloudMeasure
from uot.solvers.base_solver import SolverOutput
from uot.utils.types import ArrayLike


class MySinkhornSolver(BaseSolver):

    def solve(
        self,
        marginals: Sequence[BaseMeasure],
        costs: Sequence[ArrayLike],
        reg: float = 1e-2,
        maxiter: int = 1000,
        tol: float = 1e-6,
        **kwargs,
    ) -> SolverOutput:
        mu, nu = marginals[0], marginals[1]
        _, a = mu.as_point_cloud()
        _, b = nu.as_point_cloud()
        C = jnp.asarray(costs[0])

        K = jnp.exp(-C / reg)
        u = jnp.ones_like(a)
        for _ in range(maxiter):
            v = b / (K.T @ u)
            u = a / (K @ v)
        transport_plan = jnp.diag(u) @ K @ jnp.diag(v)
        cost = jnp.sum(transport_plan * C)

        return {
            "cost": cost,
            "transport_plan": transport_plan,
            "u_final": u,
            "v_final": v,
        }
```

## Per-solver TypedDict (optional)

If you want stronger typing for your solver's output, subclass `SolverOutput`:

```python
from typing import NotRequired
from uot.solvers.base_solver import SolverOutput

class MySolverOutput(SolverOutput, total=False):
    custom_metric: float
```

## Representation negotiation (`input_kind`)

By default the runner builds a `SolverInputs` view from each problem and calls
`solve(marginals=…, costs=…, **kwargs)`. A solver can instead request a
different *representation* by setting the class attribute `input_kind` to a kind
registered in `uot.experiments.representations`:

| `input_kind` | View passed to `solve` |
|---|---|
| `"marginals_costs"` (default) | `SolverInputs` → unpacked as `marginals=`, `costs=` |
| `"point_cloud"` | `PointCloudInputs` (shared-support aligned) |
| `"grid"` | `GridInputs` |
| `"ott_linear"`, `"ott_quadratic"`, `"ott_barycenter"`, `"ott_gw_barycenter"` | pre-built OTT problem object (registered by `uot.interop.ott`) |

For any non-default kind the prepared view is passed as the **first positional
argument** to `solve` (not unpacked). The view is built **outside** the timed
solve region and cached on the problem, so heavy translation never counts toward
the measured solve time.

```python
class MyGeometrySolver(BaseSolver):
    input_kind = "point_cloud"

    def solve(self, view, **kwargs) -> SolverOutput:
        support = view.support          # PointCloudInputs fields
        a, b = view.weights
        ...
```

Register your own kind once at import time:

```python
from uot.experiments.representations import register_representation

register_representation("my_backend", lambda problem, **opts: _build(problem, **opts))
```

A builder has signature `(problem: Problem, **opts) -> view` and must accept and
ignore unknown `**opts` (use `**_`).

## Plugging into an `Experiment`

```python
from uot import Experiment
from uot.experiments.measurement import measure_time_and_output

from my_solver import MySinkhornSolver
from my_problem import MyTwoMarginalProblem   # or any Problem subclass

problem = MyTwoMarginalProblem(...)
experiment = Experiment("demo", measure_time_and_output)
solver = MySinkhornSolver()

result = experiment.run_single(problem, solver, reg=0.01, maxiter=500)
print(result)   # dict with time + solver output keys
```

For sweeping multiple problems and solvers, use `run_pipeline`:

```python
from uot import SolverConfig, run_pipeline

solvers = [
    SolverConfig("my-sinkhorn", MySinkhornSolver,
                 param_grid=[{"reg": 0.1}, {"reg": 0.01}]),
]
df = run_pipeline(experiment, solvers, [problems], folds=3)
df.to_csv("results.csv", index=False)
```

## Plugging into the YAML pipeline

Reference your solver by fully qualified class name in a runner config.
Constructor keyword arguments go at the solver entry level;
per-call arguments go in `param-grids`:

```yaml
param-grids:
  my-params:
    - reg: 0.1
      maxiter: 1000
    - reg: 0.01
      maxiter: 5000

solvers:
  my-solver:
    solver: mypackage.solvers.MySinkhornSolver
    jit: true
    param-grid: my-params
```

The `solver:` key is resolved at runtime by `uot.utils.instantiate_solver`
which imports the class and instantiates it. Constructor kwargs (e.g. a
`device` parameter that doesn't change per-run) can be added to the solver
entry alongside `solver:`, `jit:`, etc.

See [Running benchmarks](../cli/benchmark.md) for the full runner YAML schema.

## Built-in solvers

| Class | Description |
|---|---|
| `uot.solvers.SinkhornTwoMarginalSolver` | Sinkhorn algorithm (plain / log-domain). |
| `uot.solvers.LBFGSTwoMarginalSolver` | Quasi-Newton via jaxopt. |
| `uot.solvers.GradientAscentTwoMarginalSolver` | First-order gradient ascent. |
| `uot.solvers.LinearProgrammingTwoMarginalSolver` | Exact LP via the `ot` package. |
| `uot.solvers.BackNForthSqEuclideanSolver` | Back-and-forth method (squared Euclidean). |

## OTT-JAX solvers (`uot.interop.ott`)

Requires `pip install "uot-bench[ott]"`. See [OTT-JAX interoperability](ott-interop.md) for the full guide.

| Class | Description |
|---|---|
| `uot.interop.ott.OTTSinkhornSolver` | OTT Sinkhorn (LSE mode, full-rank). |
| `uot.interop.ott.OTTLRSinkhornSolver` | OTT low-rank Sinkhorn; returns `low_rank_plan` factors. |
| `uot.interop.ott.OTTGromovWassersteinSolver` | Entropic Gromov–Wasserstein (supports fused-GW). |
| `uot.interop.ott.OTTLRGromovWassersteinSolver` | Low-rank GW. |
| `uot.interop.ott.OTTSinkhornDivergence` | Debiased Sinkhorn divergence. |
| `uot.interop.ott.OTTDiscreteBarycenterSolver` | Free-support Wasserstein barycenter. |
| `uot.interop.ott.OTTGWBarycenterSolver` | Gromov–Wasserstein barycenter. |
| `uot.interop.ott.OTTUnivariateSolver` | Closed-form 1D OT. |
