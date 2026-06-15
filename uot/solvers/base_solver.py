from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import NotRequired, TypedDict

import jax
import numpy as np

from uot.data.measure import BaseMeasure
from uot.utils.types import ArrayLike


class SolverOutput(TypedDict):
    """Return type for :meth:`BaseSolver.solve`.

    All fields except ``cost`` are optional — solvers may omit fields they
    do not compute.
    """

    cost: jax.Array | float
    transport_plan: NotRequired[jax.Array]
    coupling: NotRequired[jax.Array]          # back-compat alias for transport_plan
    iterations: NotRequired[int]
    converged: NotRequired[bool]
    error: NotRequired[float | jax.Array]
    # Potentials (dual variables)
    u_final: NotRequired[jax.Array]
    v_final: NotRequired[jax.Array]
    potentials: NotRequired[tuple[jax.Array, ...]]
    # Monge map / push-forward
    monge_map: NotRequired[jax.Array]
    # Barycenter / multi-marginal extras
    residual_l2: NotRequired[float | jax.Array]
    transport_plans: NotRequired[jax.Array]
    barycenter: NotRequired[jax.Array]
    us_final: NotRequired[jax.Array]
    vs_final: NotRequired[jax.Array]
    # Low-rank transport plan factors (Q, R, g) — used by low-rank solvers
    # to avoid materialising the full (n×m) coupling matrix.
    # The dense plan is Q @ diag(1/g) @ R.T.
    low_rank_plan: NotRequired[tuple[jax.Array, jax.Array, jax.Array]]
    # Extra solver-specific fields — solvers may add more keys
    time: NotRequired[float]


class BaseSolver(ABC):
    """Base class for all OT solvers.

    Subclass and implement :meth:`solve`. The method must return a
    :class:`SolverOutput` dict with at least the ``cost`` key set to a
    ``jax.Array``.

    Example::

        class MySolver(BaseSolver):
            def solve(self, marginals, costs, reg=0.1, **kwargs):
                ...
                return {"cost": jnp.array(transport_cost)}

    **Representation negotiation**: set the class attribute ``input_kind`` to a
    string registered in :mod:`uot.experiments.representations` to request a
    pre-built representation object.  The runner will pass this object as the
    first positional argument to ``solve`` — see :mod:`uot.experiments.representations`
    for details.  The default ``"marginals_costs"`` preserves the existing
    ``solve(marginals, costs, **kwargs)`` calling convention.
    """

    requires_squared_euclidean: bool = False
    input_kind: str = "marginals_costs"

    @abstractmethod
    def solve(
        self,
        marginals: Sequence[BaseMeasure],
        costs: Sequence[ArrayLike],
        *args,
        **kwargs,
    ) -> SolverOutput:
        """Solve a (multi-)marginal OT problem.

        Returns a :class:`SolverOutput` dict; ``cost`` is always present.
        """
