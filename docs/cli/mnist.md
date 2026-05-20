# MNIST classification experiment

The MNIST classification experiment runs in two steps: first compute pairwise OT
distance matrices over the full dataset, then train SVM classifiers on those matrices.

Requires the `mnist` extra: `pip install "uot-bench[mnist]"`.

## Step 1 — Pairwise OT distances

```bash
uot-mnist-distances --config configs/mnist/mnist_dist_example.yaml
# or:
python -m uot.experiments.real_data.mnist_classification.count_pairwise_distances \
    --config configs/mnist/mnist_dist_example.yaml
```

**Config schema:**

```yaml
param-grids:
  epsilons:
    - reg: 1
    - reg: 0.01

solvers:
  sinkhorn:
    solver: uot.solvers.sinkhorn.SinkhornTwoMarginalSolver
    jit: true
    param-grid: epsilons

batch-size: 5000
output-dir: ./outputs/mnist/costs
```

| Key | Description |
|---|---|
| `batch-size` | Number of simultaneous JAX operations (memory vs speed). |
| `output-dir` | Directory to write one CSV distance matrix per solver configuration. |

Each output file is named `<solver-name>_<param1_val1_...>.csv`.

## Step 2 — Classification

```bash
uot-mnist-classification --config configs/mnist/mnist_classification_example.yaml
# or:
python -m uot.experiments.real_data.mnist_classification.mnist_classification \
    --config configs/mnist/mnist_classification_example.yaml
```

**Config schema:**

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

sample-sizes:
  - 100
  - 250

costs-dir: ./outputs/mnist/costs
output-dir: ./outputs/mnist/classification
rng-seed: 42
```

| Key | Description |
|---|---|
| `sample-sizes` | List of training subset sizes to evaluate. |
| `costs-dir` | Directory written by Step 1. |
| `output-dir` | Where to write `mnist_results_<timestamp>.csv`. |
| `rng-seed` | Seed for reproducible train/test splits. |

For every solver configuration and sample size, a scikit-learn SVM is trained with
a kernel matrix built from the precomputed OT distances. The output CSV has columns
`solver`, `sample_size`, `accuracy`, plus any solver parameter columns.
