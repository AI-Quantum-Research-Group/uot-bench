# Measures and cost functions

## Measures

A **measure** represents a probability distribution as a discrete set of support
points with associated weights. All measures inherit from `uot.data.BaseMeasure`.

### `PointCloudMeasure`

Scattered support points with arbitrary positions.

```python
from uot import PointCloudMeasure
import numpy as np

points = np.random.standard_normal((100, 2))   # (n, d)
weights = np.ones(100) / 100                   # (n,)  must sum to 1

mu = PointCloudMeasure(points, weights, name="mu")
mu = PointCloudMeasure(points, weights, normalize=True)  # auto-normalize
```

Key methods:

| Method | Returns | Description |
|---|---|---|
| `as_point_cloud(include_zeros=True)` | `(points, weights)` | The raw arrays. |
| `support(include_zeros=True)` | `ArrayLike` | Support points only. |
| `get_jax()` | `PointCloudMeasure` | Convert internal arrays to JAX. |

### `GridMeasure`

A measure defined on a regular Cartesian grid. Preferred for algorithms that
exploit the grid structure (e.g. back-and-forth solvers).

```python
from uot import GridMeasure
import numpy as np

axes = [np.linspace(-3, 3, 64)]          # list of 1-D axis arrays, one per dimension
weights_nd = np.ones((64,)) / 64          # n-D weight tensor
mu = GridMeasure(axes, weights_nd, normalize=True)
```

Key methods:

| Method | Returns | Description |
|---|---|---|
| `as_grid(backend, dtype, device)` | `(axes, weights_nd)` | Axes list + N-D weights. |
| `for_grid_solver(...)` | `(axes, weights_nd)` | Same, but normalized and suitable for grid solvers. |
| `as_point_cloud()` | `(points, weights)` | Flattened to point cloud (loses grid structure). |

### `BaseMeasure` contract

Any `Problem` or `Generator` can reference measures through the abstract base:

```python
from uot import BaseMeasure

def inspect(mu: BaseMeasure) -> None:
    pts, w = mu.as_point_cloud()
    print(f"n={pts.shape[0]}, d={pts.shape[1]}, sum(w)={w.sum():.4f}")
```

## Cost functions

Cost functions take two point arrays `(X: [n, d], Y: [m, d])` and return an
`[n, m]` cost matrix.

```python
from uot.utils.costs import cost_euclid_squared
C = cost_euclid_squared(X, Y)  # shape (n, m)
```

### Built-in cost functions

| Function | Formula | Note |
|---|---|---|
| `cost_euclid_squared` | `‖X_i − Y_j‖²` | Standard for Sinkhorn. Set `requires_squared_euclidean=True` on solvers that need this. |
| `cost_euclid` | `‖X_i − Y_j‖₂` | Euclidean (not squared). |
| `cost_manhattan` | `∑_k |X_ik − Y_jk|` | ℓ₁ norm. |
| `cost_cosine` | `1 − X_i·Y_j / (‖X_i‖·‖Y_j‖)` | Cosine distance. |

All functions are importable from `uot.utils.costs`.

### Using a custom cost function

Any callable with signature `(X: ArrayLike, Y: ArrayLike) -> ArrayLike` works:

```python
import jax.numpy as jnp
from uot import TwoMarginalProblem
from uot import PointCloudMeasure

def cost_linf(X, Y):
    return jnp.max(jnp.abs(X[:, None, :] - Y[None, :, :]), axis=-1)

problem = TwoMarginalProblem("my-problem", mu, nu, cost_linf)
```

In a YAML config, use the fully qualified name:

```yaml
generators:
  my-dataset:
    generator: uot.problems.generators.GaussianMixtureGenerator
    cost_fn: mypackage.costs.cost_linf
    ...
```
