"""Deprecated: use the generic Experiment + ColorTransferHook instead.

    from uot.experiments import Experiment, run_pipeline
    from uot.experiments.measurement import measure_time_and_output
    from uot.experiments.real_data.color_transfer.hooks import ColorTransferHook

    hook = ColorTransferHook(output_dir="output/color_transfer")
    experiment = Experiment(name="CT", solve_fn=measure_time_and_output, hooks=[hook])
    df = run_pipeline(experiment, solvers, iterators)
"""

import warnings as _warnings


def __getattr__(name: str):
    if name == "ColorTransferExperiment":
        _warnings.warn(
            "ColorTransferExperiment is deprecated.  Use the generic "
            "Experiment + ColorTransferHook instead.  "
            "See uot.experiments.real_data.color_transfer.hooks.",
            DeprecationWarning,
            stacklevel=2,
        )
        from uot.experiments.experiment import Experiment as _Experiment
        return _Experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
