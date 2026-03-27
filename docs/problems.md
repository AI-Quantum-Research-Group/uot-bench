# Problems Module

This module defines OT problem containers, storage backends, and iterators.
It is the core interface between generated data and solvers/experiments.

## Core abstractions

### MarginalProblem
File: `uot/problems/base_problem.py`

`MarginalProblem` is the base class for problems with two or more marginals.
It stores a list of measures and cost functions and provides:

- `solver_inputs(...)`, `point_cloud_inputs(...)`, and `grid_inputs(...)`
- `shared_support(...)` and `weights_on_shared_support(...)` to align measures
- `key()`/`__hash__()` based on pickled content for reproducible storage

Shared-support utilities are useful for barycenter solvers and grid-based methods.

## Problem types

### TwoMarginalProblem
File: `uot/problems/two_marginal.py`

Represents two measures `(mu, nu)` and a single cost function. It lazily
computes and caches the cost matrix and (optionally) the exact optimal transport
solution via POT (`ot.emd`) for benchmarking.

Key methods:
- `solver_inputs()` returns marginals, costs, and cost metadata
- `point_cloud_inputs()` returns aligned support, stacked weights, and optional cost
- `grid_inputs()` returns shared axes, stacked grid weights, and optional cost
- `get_exact_cost()` / `get_exact_coupling()` compute ground truth

### BarycenterProblem
File: `uot/problems/barycenter_problem.py`

Represents a barycenter problem over multiple measures with weights `lambdas`.
The cost matrix is computed once from the first marginal's support.

Key methods:
- `lambdas()` returns weights
- `point_cloud_inputs(...)` returns `(support, weights, cost, lambdas)` via a dataclass

### MultiMarginalProblem
File: `uot/problems/multi_marginal.py`

A thin placeholder subclass of `MarginalProblem`. It currently provides no
additional functionality beyond the base constructor.

## Storage backends

### ProblemStore (pickle)
File: `uot/problems/store.py`

Pickle-based store for generated problems. Uses a content hash as filename.
Typical flow:

```python
from uot.problems.store import ProblemStore

store = ProblemStore("datasets/synthetic/gaussians")
store.save(problem)
problem = store.load(next(iter(store.all_problems())))
```

### HDF5ProblemStore
File: `uot/problems/hdf5_store.py`

Stores problems in a structured HDF5 file. Supports saving and loading of
`TwoMarginalProblem` instances (loading for other problem types is not
implemented yet). Uses gzip compression and chunking for efficiency.

## Iterators

### ProblemIterator
File: `uot/problems/iterator.py`

Lazy iterator over a `ProblemStore` directory. Loads one problem at a time,
which is helpful for large datasets.

### OnlineProblemIterator
File: `uot/problems/iterator.py`

Generates problems on the fly from a `ProblemGenerator` without serialization.
Useful for quick tests and JIT warm-ups. Set `cache_gt=True` to precompute
costs and ground truth.

## Serialization (CLI)

File: `uot/problems/problem_serializer.py`

The serializer reads a YAML config, instantiates generators, and saves the
result either as pickles or into an HDF5 file.

CLI usage:
```
pixi run serialize --config configs/generators/gaussians.yaml --export-dir datasets/synthetic
```

Internally, the serializer resolves:
- `generator`, `class`, `cost_fn` as import paths
- any nested generator configs that end with `_cfg` (used by `PairedGenerator`)

## Utility helpers

`MarginalProblem` provides support alignment helpers and high-level input views:

- `shared_support(mode="same"|"union"|"intersection"|"first")`
- `weights_on_shared_support(...)`
- `solver_inputs(include_cost=True)`
- `point_cloud_inputs(...)`
- `grid_inputs(...)`

`"same"` is the default and should be the common path for notebook and benchmark
use: it validates that marginals already share a support and then stacks weights
without constructing a union. Use `"union"` or `"intersection"` explicitly when
supports genuinely differ. For shared-grid problems, prefer `grid_inputs()`
because it bypasses point-cloud alignment entirely.

For notebook convenience helpers, see `uot/utils/notebook_helpers.py` and the
examples in `docs/generators.md`.

## Dataset inspection

File: `uot/problems/inspect_store.py`

You can render quick visualizations of stored datasets (Plotly HTML + PNG):

```
python -m uot.problems.inspect_store --dataset datasets/synthetic --outdir plots
```

Use `--store` to target a single store path or a `.h5` file.
