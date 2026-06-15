"""OTT-JAX interoperability for uot-bench.

Provides :class:`~uot.solvers.base_solver.BaseSolver` wrappers around the
headline OTT-JAX solvers so they can be used in the same benchmark harness
(YAML configs, SLURM scripts, Dash dashboard) as uot-bench's native solvers.

Requires ``pip install uot-bench[ott]`` (adds ``ott-jax>=0.4.7``).

Attempting to import this module without ``ott-jax`` installed raises an
:class:`ImportError` with a clear install instruction.

Example::

    from uot.interop.ott import OTTSinkhornSolver

    # OTT linear/quadratic wrappers consume a pre-built OTT problem object
    # (``input_kind`` is not ``"marginals_costs"``).
    linear_problem = problem.to_ott_linear_problem(epsilon=1e-2)
    solver = OTTSinkhornSolver(max_iterations=4000, threshold=1e-6)
    out = solver.solve(linear_problem, epsilon=1e-2)
"""

from uot.interop.ott.solvers import (  # noqa: F401
    OTTDiscreteBarycenterSolver,
    OTTGromovWassersteinSolver,
    OTTGWBarycenterSolver,
    OTTLRGromovWassersteinSolver,
    OTTLRSinkhornSolver,
    OTTSinkhornDivergence,
    OTTSinkhornSolver,
    OTTUnivariateSolver,
)

from uot.interop.ott._representations import _register_ott_representations  # noqa: F401

_register_ott_representations()

__all__ = [
    "OTTSinkhornSolver",
    "OTTLRSinkhornSolver",
    "OTTGromovWassersteinSolver",
    "OTTLRGromovWassersteinSolver",
    "OTTSinkhornDivergence",
    "OTTDiscreteBarycenterSolver",
    "OTTGWBarycenterSolver",
    "OTTUnivariateSolver",
]
