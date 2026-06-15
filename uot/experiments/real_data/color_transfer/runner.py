"""Deprecated: use run_pipeline from uot.experiments instead.

The generic :func:`~uot.experiments.runner.run_pipeline` now handles
colour-transfer experiments when combined with
:class:`~uot.experiments.real_data.color_transfer.hooks.ColorTransferHook`.

    from uot.experiments import run_pipeline, Experiment
    from uot.experiments.measurement import measure_time_and_output
    from uot.experiments.real_data.color_transfer.hooks import ColorTransferHook

    hook = ColorTransferHook(output_dir="output/color_transfer")
    experiment = Experiment(name="CT", solve_fn=measure_time_and_output, hooks=[hook])
    df = run_pipeline(experiment, solvers, iterators)
"""

import warnings as _warnings


def __getattr__(name: str):
    if name == "run_color_transfer_pipeline":
        _warnings.warn(
            "run_color_transfer_pipeline is deprecated.  "
            "Use run_pipeline from uot.experiments instead.  "
            "See uot.experiments.real_data.color_transfer.hooks.",
            DeprecationWarning,
            stacklevel=2,
        )
        from uot.experiments.runner import run_pipeline
        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
