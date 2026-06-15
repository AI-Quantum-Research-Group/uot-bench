# `uot.solvers`

Solver base class, concrete solvers, and output types.

See [Writing a custom Solver](../guide/custom-solver.md) for a guide on subclassing.

## Base class and output type

::: uot.solvers.base_solver.BaseSolver
    options:
      show_root_heading: true
      show_source: false

::: uot.solvers.base_solver.SolverOutput
    options:
      show_root_heading: true
      show_source: false

::: uot.solvers.solver_config.SolverConfig
    options:
      show_root_heading: true
      show_source: false

## Built-in solvers

::: uot.solvers
    options:
      show_root_heading: false
      show_source: false
      members:
        - SinkhornTwoMarginalSolver
        - SinkhornTwoMarginalLogJaxSolver
        - LBFGSTwoMarginalSolver
        - GradientAscentTwoMarginalSolver
        - LinearProgrammingTwoMarginalSolver
        - BackNForthSqEuclideanSolver

## OTT-JAX solvers

See [uot.interop.ott](interop_ott.md) for the OTT-JAX solver wrappers
(`OTTSinkhornSolver`, `OTTLRSinkhornSolver`, `OTTGromovWassersteinSolver`, etc.).
Requires `pip install "uot-bench[ott]"`.
