# Generating datasets

Pre-generating and persisting problems decouples expensive sampling from solver
benchmarking: generate once, benchmark many times.

## Command

```bash
# Pickle backend (default)
uot-serialize --config configs/generators/example.yaml --export-dir datasets/synthetic

# HDF5 backend (requires uot-bench[storage])
uot-serialize --config configs/generators/example.yaml --export-hdf5 datasets/synthetic.h5

# Equivalents via python -m:
python -m uot.problems.problem_serializer --config ... --export-dir ...
python -m uot.problems.problem_serializer --config ... --export-hdf5 ...
```

**Output:**

* `--export-dir <dir>` — one subfolder per generator under `<dir>`, each containing
  per-problem pickle files and a `meta.yaml` with the config snapshot.
* `--export-hdf5 <file>` — single `.h5` file using `gzip` compression
  (requires `pip install "uot-bench[storage]"`).

## Generator config schema

```yaml
generators:
  <dataset-name>:
    generator: <fully.qualified.GeneratorClass>
    dim: <int>
    n_points: <int>
    num_datasets: <int>
    cost_fn: <fully.qualified.cost_function>
    borders: (<float>, <float>)   # support range
    use_jax: <bool>
    seed: <int>
    # ... any extra kwargs forwarded to the generator's __init__
```

The `generator:` key must be the fully qualified class name of a subclass of
`uot.Generator`. All remaining keys are passed as constructor arguments.

### YAML anchors

Use YAML anchors to avoid repetition when defining multiple generators:

```yaml
defaults: &g
  dim: 1
  n_points: 64
  cost_fn: uot.utils.costs.cost_euclid_squared
  use_jax: true
  seed: 42

generators:
  1D-gaussians:
    <<: *g
    generator: uot.problems.generators.GaussianMixtureGenerator
    num_components: 1
    num_datasets: 30
    borders: (-6, 6)

  1D-cauchy:
    <<: *g
    generator: uot.problems.generators.CauchyGenerator
    num_datasets: 20
    borders: (-10, 10)
```

### Paired generator

To draw `mu` from one distribution and `nu` from another, use `PairedGenerator`:

```yaml
generators:
  1D-cauchy-vs-gmm:
    generator: uot.problems.generators.PairedGenerator
    num_datasets: 10
    gen_a_cfg:
      class: uot.problems.generators.CauchyGenerator
      params:
        dim: 1
        n_points: 64
        borders: [-3, 3]
        cost_fn: uot.utils.costs.cost_euclid_squared
        seed: 42
    gen_b_cfg:
      class: uot.problems.generators.GaussianMixtureGenerator
      params:
        dim: 1
        n_points: 64
        borders: [-3, 3]
        cost_fn: uot.utils.costs.cost_euclid_squared
        num_components: 3
        seed: 24
```

## Built-in generators

| Class | Description |
|---|---|
| `uot.problems.generators.GaussianMixtureGenerator` | Samples Gaussian mixture models on a fixed grid. Parameters: `dim`, `num_components`, `n_points`, optional Wishart hyper-parameters. |
| `uot.problems.generators.CauchyGenerator` | 1-D Cauchy-distributed marginals. Parameters: `dim`, `n_points`, `borders`. |
| `uot.problems.generators.ExponentialGenerator` | 1-D exponential distributions with a random scale parameter. |
| `uot.problems.generators.PairedGenerator` | Composes two generators: `mu` from `gen_a_cfg`, `nu` from `gen_b_cfg`. |

For the full parameter listings see [uot.problems API reference](../api/problems.md).

## Inspecting a dataset

```bash
uot-inspect-store --dataset datasets/synthetic --outdir plots/
# or for HDF5:
uot-inspect-store --store datasets/synthetic.h5 --outdir plots/
```

Saves distribution plots to `plots/`. Useful for sanity-checking before
running expensive benchmarks.

## Custom generators

See [Writing a custom Generator](../guide/custom-generator.md).
Custom generators can be used in YAML configs by referencing their fully
qualified class name, e.g.:

```yaml
generators:
  my-dataset:
    generator: mypackage.generators.MyGenerator
    n_points: 128
    seed: 0
```
