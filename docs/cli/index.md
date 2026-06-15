# CLI & YAML overview

uot-bench exposes its pipeline through console scripts installed with the
package. Every command is also available as `python -m <module>` — the two
forms are exactly equivalent.

## Command reference

| Console script | `python -m` equivalent | What it does | Details |
|---|---|---|---|
| `uot-serialize` | `python -m uot.problems.problem_serializer` | Generate and persist problems from a YAML generator config | [serialize](serialize.md) |
| `uot-benchmark` | `python -m uot.experiments.synthetic.benchmark` | Run solvers over problems, write results CSV | [benchmark](benchmark.md) |
| `uot-color-transfer` | `python -m uot.experiments.real_data.color_transfer.color_transfer` | Color transfer experiment | [color-transfer](color-transfer.md) |
| `uot-color-transfer-viz` | `python -m uot.experiments.real_data.color_transfer.visualization` | Visual dashboard for color transfer results | |
| `uot-mnist-distances` | `python -m uot.experiments.real_data.mnist_classification.count_pairwise_distances` | Step 1 of MNIST: pairwise OT distances | [mnist](mnist.md) |
| `uot-mnist-classification` | `python -m uot.experiments.real_data.mnist_classification.mnist_classification` | Step 2 of MNIST: KNN classification | [mnist](mnist.md) |
| `uot-inspect-store` | `python -m uot.problems.inspect_store` | Visualize a serialized problem dataset | |

Run any command with `--help` for the full flag listing.

## Typical benchmark workflow

```
configs/generators/example.yaml
         │
         ▼
  uot-serialize --config <generator.yaml> --export-dir datasets/synthetic
         │
         ▼  (or skip serialize and use --generators-config for online generation)
         │
  uot-benchmark --config <runner.yaml> --dataset datasets/synthetic
                --folds 3 --export results/my_run.csv
         │
         ▼
  results/my_run.csv   ← pandas DataFrame, one row per (problem, solver, params, fold)
```

The two config files serve different purposes:

* **Generator config** (`configs/generators/`) — defines *what problems to create*:
  distribution family, dimensions, number of samples.
  See [Generating datasets](serialize.md).

* **Runner config** (`configs/runners/`) — defines *which solvers to run* and
  *how to measure them*: solver class, parameter grids, measurement function.
  See [Running benchmarks](benchmark.md).

They are intentionally separate so you can reuse the same dataset across many
runner configs, and run any runner against any dataset.

## SLURM

See [SLURM](slurm.md) for `sbatch` wrappers around the benchmark and color-transfer commands.
