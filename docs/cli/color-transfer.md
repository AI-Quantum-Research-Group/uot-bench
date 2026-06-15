# Color transfer experiment

Run an end-to-end color transfer experiment via optimal transport.

## Command

```bash
uot-color-transfer --config configs/color_transfer/example.yaml
# or equivalently:
python -m uot.experiments.real_data.color_transfer.color_transfer \
    --config configs/color_transfer/example.yaml
```

Results are written to a timestamped subfolder under `experiment.output-dir`
and include `color_transfer_results.csv` plus per-image output files.

## Configuration schema

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

bin-number:
  - 16
  - 32
soft-extension:
  - no
  - yes
displacement-interpolation:
  - 0.0
  - 1.0
color-space: rgb
# active-channels: [r, g]
batch-size: 100000
pair-number: 3
images-dir: ./datasets/images
rng-seed: 42

drop-columns:
  - transport_plan
  - monge_map
  - u_final
  - v_final

experiment:
  name: Time and test
  output-dir: ./outputs/color_transfer
```

### Key parameters

| Key | Type | Description |
|---|---|---|
| `bin-number` | int or list[int] | Color grid resolution per channel. Each value triggers a full benchmark pass. |
| `batch-size` | int | Number of simultaneous JAX operations (memory vs speed). |
| `pair-number` | int | Image pairs sampled per solver configuration (excluding warm-up). |
| `images-dir` | str | Directory of source images. |
| `experiment.output-dir` | str | Parent folder for timestamped output subdirectory. |
| `rng-seed` | int | Seed for reproducible image pairing. |
| `drop-columns` | list[str] | Columns to drop from the result CSV (e.g. large arrays: `transport_plan`, `monge_map`). |
| `soft-extension` | bool/yes/no or list | Whether to apply soft-extension post-processing. Each value runs as a separate mode. |
| `displacement-interpolation` | float or list[float] | Displacement interpolation alpha ∈ [0, 1]. Each value runs as a separate mode. |
| `color-space` | str | `rgb` or `lab`/`cielab`. |
| `active-channels` | list | Subset of channels by name or index (e.g. `[l, a]` or `[0, 1]`). Optional. |

The `solvers:` and `param-grids:` blocks follow the same schema as the
[benchmark runner](benchmark.md).

### Workflow

For each solver, `pair-number` source–target image pairs are sampled. The
OT plan is computed and the source is transported to the target. The result CSV
accumulates metrics for all pairs, solvers, and parameter combinations.

## Programmatic API

!!! warning "`ColorTransferExperiment` is deprecated"
    The dedicated `ColorTransferExperiment` class and
    `run_color_transfer_pipeline` are deprecated (they now emit a
    `DeprecationWarning`). Use the generic `Experiment` + `run_pipeline` with the
    `ColorTransferHook` post-solve hook instead:

    ```python
    from uot.experiments import Experiment, run_pipeline
    from uot.experiments.measurement import measure_time_and_output
    from uot.experiments.real_data.color_transfer.hooks import ColorTransferHook

    hook = ColorTransferHook(
        output_dir="output/color_transfer",
        soft_extension_modes=[False, True],
        displacement_alphas=[1.0, 0.5],
    )
    experiment = Experiment(name="CT", solve_fn=measure_time_and_output, hooks=[hook])
    df = run_pipeline(experiment, solvers, iterators)
    ```

    The hook reconstructs transported images, computes domain metrics, and fans
    out one result row per `(soft_extension, displacement_alpha)` combination.
    See [Post-solve hooks](../guide/hooks.md). The `uot-color-transfer` CLI
    behaviour is unchanged.

## Visualization dashboard

```bash
uot-color-transfer-viz --origin_folder <path_to_input_images> \
                        --results_folder <path_to_resulting_images>
# or:
python -m uot.experiments.real_data.color_transfer.visualization \
    --origin_folder <path_to_input_images> \
    --results_folder <path_to_resulting_images>
```

Launches a Dash web server at `http://localhost:8050` for visual comparison.
Requires the `color-transfer` extra: `pip install "uot-bench[color-transfer]"`.

## Back-and-forth solver notes

- For `BackNForthSqEuclideanSolver`, the returned Monge map is in index coordinates.
- The CIC map construction mirrors the CIC pushforward; pushforward-from-map and
  pushforward-from-potential should be close (up to interpolation error).
- The adaptive map is a representative average of adaptive samples, so a
  map-based pushforward can differ from the adaptive pushforward. This is expected;
  use CIC for tighter agreement.
