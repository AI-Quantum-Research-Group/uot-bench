# Problem Generators

This module contains synthetic problem generators that yield
`TwoMarginalProblem` or `BarycenterProblem` instances. Generators implement the
`ProblemGenerator` interface:

```python
from uot.problems.problem_generator import ProblemGenerator

class MyGenerator(ProblemGenerator):
    def generate(self, *args, **kwargs):
        yield problem
```

Most generators discretize a continuous distribution on a Cartesian grid and
then build either a `GridMeasure` or a `DiscreteMeasure` via
`uot.utils.build_measure._build_measure`.

## Common parameters and behavior

Many generators share these parameters:

- `dim`: ambient dimension (often 1, 2, or 3 depending on the generator)
- `n_points`: number of grid points per axis
- `num_datasets`: number of problems to yield
- `borders`: tuple `(low, high)` defining the hyper-rectangle domain
- `cost_fn`: cost function used by `TwoMarginalProblem`
- `measure_mode`: `'grid' | 'discrete' | 'auto'`
  - `'grid'`/`'auto'` produce a `GridMeasure` (weights reshaped to ND)
  - `'discrete'` produces a `DiscreteMeasure` (points, weights)
- `cell_discretization`: `'cell-centered' | 'vertex-centered'`
  - if `cell-centered`, weights are multiplied by the grid cell volume before
    normalization (Riemann-sum style)

## Using generators in Python

```python
from uot.problems.generators import GaussianMixtureGenerator
from uot.utils.costs import cost_euclid_squared

gen = GaussianMixtureGenerator(
    name="gmm",
    dim=1,
    num_components=2,
    n_points=64,
    num_datasets=5,
    borders=(-5.0, 5.0),
    cost_fn=cost_euclid_squared,
    use_jax=True,
    seed=0,
)

problem = next(gen.generate())
mu, nu = problem.get_marginals()
C = problem.get_costs()[0]
```

## Using generators in notebooks

`uot/utils/notebook_helpers.py` provides small utilities:

```python
from uot.utils.notebook_helpers import one_problem, barycenter_inputs
from uot.problems.generators.toy_barycenter_generator import (
    ToyBarycenterGenerator,
    FixedListSelector,
)
from uot.utils.costs import cost_euclid_squared

selector = FixedListSelector(names=("Ring", "Square", "Star"))

gen = ToyBarycenterGenerator(
    selector=selector,
    n_points=64,
    cost_fn=cost_euclid_squared,
    num_datasets=1,
    measure_mode="grid",
)

problem = one_problem(gen, num_marginals=3)
measures, lambdas, cost, weights = barycenter_inputs(problem)
```

`one_problem(...)` pulls a single item from any `ProblemGenerator`.
`barycenter_inputs(...)` extracts the inputs needed by barycenter solvers.
`stack_measure_weights(...)` turns a list of measures into a `(M, N)` weight array.

## Using generators via YAML configs

The serializer resolves `generator` and `cost_fn` paths, and any nested config
keys ending with `_cfg`.

Minimal example:

```yaml
generators:
  1D-gaussians:
    generator: uot.problems.generators.GaussianMixtureGenerator
    dim: 1
    num_components: 2
    n_points: 64
    num_datasets: 10
    borders: (-5, 5)
    cost_fn: uot.utils.costs.cost_euclid_squared
    use_jax: true
    seed: 42
```

Serialize:
```
pixi run serialize --config configs/generators/gaussians.yaml --export-dir datasets/synthetic
```

### PairedGenerator config

`PairedGenerator` uses nested configs under `gen_a_cfg` and `gen_b_cfg`:

```yaml
generators:
  paired:
    generator: uot.problems.generators.PairedGenerator
    num_datasets: 30
    gen_a_cfg:
      class: uot.problems.generators.CauchyGenerator
      params:
        dim: 1
        n_points: 32
        borders: (-1, 1)
        cost_fn: uot.utils.costs.cost_euclid_squared
        use_jax: false
        seed: 1
    gen_b_cfg:
      class: uot.problems.generators.GaussianMixtureGenerator
      params:
        dim: 1
        num_components: 2
        n_points: 32
        borders: (-1, 1)
        cost_fn: uot.utils.costs.cost_euclid_squared
        use_jax: true
        seed: 2
```

## Generator catalog

### GaussianMixtureGenerator
File: `uot/problems/generators/gaussian_mixture_generator.py`

- Two-marginal problems from GMMs evaluated on a grid.
- Supports JAX (`use_jax=True`) and NumPy/SciPy backends.
- Useful for smooth, multi-modal synthetic benchmarks.

### CauchyGenerator
File: `uot/problems/generators/cauchy_generator.py`

- 1D Cauchy distributions sampled on a grid.
- Good for heavy-tailed marginals.

### ExponentialGenerator
File: `uot/problems/generators/exponential_generator.py`

- 1D exponential distributions with random scale.
- Useful for asymmetric, long-tailed weights.

### IndependentCauchyGenerator
File: `uot/problems/generators/independent_cauchy.py`

- Independent Cauchy factors per dimension (product distribution).
- Samples separate location and scale for each marginal and dimension.

### IndependentExponentialGenerator
File: `uot/problems/generators/independent_exponential.py`

- Independent exponential factors per dimension.
- Each marginal has its own loc/scale per dimension.

### GeneralizedHyperbolicMixtureGenerator
File: `uot/problems/generators/generalized_hyperbolic_mixture.py`

- Mixtures of independent generalized hyperbolic components.
- Useful for heavier tails and more complex shapes.

### StudentTGenerator
File: `uot/problems/generators/students_t.py`

- Multivariate Student's t marginals with Wishart-sampled covariance.
- Good for heavier tails and robust distributions.

### PairedGenerator
File: `uot/problems/generators/paired_generator.py`

- Combines two generators: mu from generator A and nu from generator B.
- Useful for controlled distribution mismatch.

### ToyBarycenterGenerator
File: `uot/problems/generators/toy_barycenter_generator.py`

- Builds barycenter problems from simple shape fields on a 2D grid.
- Uses a `ShapeSelector` to choose which shapes become the marginals.

Selectors:
- `FixedListSelector`: use a fixed set of shape names
- `RoundRobinSelector`: cycle across groups of shapes

Example (Python):

```python
from uot.problems.generators.toy_barycenter_generator import (
    ToyBarycenterGenerator,
    FixedListSelector,
)
from uot.utils.costs import cost_euclid_squared

selector = FixedListSelector(names=("Ring", "Square", "Star"))

gen = ToyBarycenterGenerator(
    selector=selector,
    n_points=64,
    cost_fn=cost_euclid_squared,
    num_datasets=3,
    measure_mode="grid",
)

problem = next(gen.generate(num_marginals=3))
```
