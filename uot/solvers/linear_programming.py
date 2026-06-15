import ot
import numpy as np
import jax.numpy as jnp
from collections.abc import Sequence

from uot.data.measure import BaseMeasure, PointCloudMeasure
from uot.solvers.base_solver import BaseSolver, SolverOutput
from uot.utils.types import ArrayLike


class LinearProgrammingTwoMarginalSolver(BaseSolver):

    def solve(
        self,
        marginals: Sequence[BaseMeasure],
        costs: Sequence[ArrayLike],
        numItermax: int = 100_000,
    ) -> SolverOutput:
        if len(costs) == 0:
            raise ValueError("Cost tensors not defined.")
        if len(marginals) != 2:
            raise ValueError("This linear programming solver accepts only two marginals.")

        mu_np = np.asarray(marginals[0].as_point_cloud()[1])
        nu_np = np.asarray(marginals[1].as_point_cloud()[1])
        cost_np = np.asarray(costs[0])

        coupling, log = ot.emd(mu_np, nu_np, cost_np, numItermax=numItermax, log=True)

        return {
            "transport_plan": jnp.asarray(coupling),
            "cost": jnp.asarray(log["cost"]),
            "u_final": jnp.asarray(log["u"]),
            "v_final": jnp.asarray(log["v"]),
            "iterations": int(log.get("numIter", 0)),
            "error": float(
                max(
                    np.max(np.abs(coupling.sum(axis=1) - mu_np)),
                    np.max(np.abs(coupling.sum(axis=0) - nu_np)),
                )
            ),
        }
