# SLURM job submission

The scripts in `scripts/` are SBATCH wrappers around the CLI tools.
They assume the project venv is already active — activate it before
submitting (or set `PATH` via `--export`).

Each script sets JAX environment variables for GPU use:

```bash
JAX_ENABLE_X64=True
JAX_PLATFORM_NAME=gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

## Benchmark (pickle store)

```bash
sbatch -J task-name --time 14-00:00:00 \
    --export=DATA_DIR=datasets/synthetic,RESULT_DIR=results/synthetic \
    scripts/run_benchmark.sh \
    configs/generators/example.yaml \
    configs/runners/cot/sinkhorn.yaml
```

The script serializes the dataset (if not already present under `DATA_DIR`) then
runs the benchmark. Results are written to `RESULT_DIR/<runner>.csv`.

## Benchmark (HDF5 store)

```bash
sbatch -J task-name --time 14-00:00:00 \
    --export=DATA_FILE=datasets/synthetic.h5,RESULT_DIR=results/synthetic \
    scripts/run_benchmark_hdf5.sh \
    configs/generators/example.yaml \
    configs/runners/cot/sinkhorn.yaml
```

Requires the `storage` extra: `pip install "uot-bench[storage]"`.

## Benchmark (online generation — no serialization step)

```bash
sbatch -J task-name --time 14-00:00:00 \
    --export=RESULT_DIR=results/synthetic \
    scripts/run_benchmark_online.sh \
    configs/generators/example.yaml \
    configs/runners/cot/sinkhorn.yaml
```

Problems are generated on the fly via `--generators-config`. Useful when
storage space is limited or for quick iteration.

## Color transfer

```bash
sbatch -J color-transfer --time 02-00:00:00 \
    scripts/run_color_transfer.sh \
    configs/color_transfer/example.yaml
```

## Monitoring GPU usage

```bash
srun --jobid 123456 --pty watch -n 30 nvidia-smi
```
