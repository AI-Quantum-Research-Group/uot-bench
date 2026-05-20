# uot-bench

**uot-bench** is a Python toolkit for optimal transport solvers and benchmarking.
It provides JAX-first implementations of common OT methods, utilities for generating
problems and measures, and a configurable pipeline for running experiments at scale.

---

## Where to start

| Goal | Where to go |
|---|---|
| Install the package | [Install](install.md) |
| Run a solver in 5 lines of Python | [Quickstart](quickstart.md) |
| Understand the core concepts (Problem, Generator, Solver, Experiment) | [Concepts](guide/concepts.md) |
| Write a custom Problem, Generator, or Solver | [Library guide](guide/custom-problem.md) |
| Run experiments via YAML config files | [CLI & YAML](cli/index.md) |
| API reference | [Reference](api/uot.md) |

---

## Key abstractions

```python
from uot import (
    Problem, Generator,           # base classes — subclass these
    TwoMarginalProblem,           # concrete problem type
    BaseSolver, SolverConfig,     # solver infrastructure
    Experiment, run_pipeline,     # experiment runner
)
```

See [guide/concepts.md](guide/concepts.md) for a diagram of how these pieces fit together.
