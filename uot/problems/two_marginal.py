import ot
from collections.abc import Callable
from uot.data.measure import BaseMeasure
from uot.problems.base_problem import Problem as MarginalProblem
from uot.utils.types import ArrayLike

from uot.utils.logging import logger


class TwoMarginalProblem(MarginalProblem):
    def __init__(
        self,
        name: str,
        mu: BaseMeasure,
        nu: BaseMeasure,
        cost_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    ):
        super().__init__(name, [mu, nu], [cost_fn])
        self._mu = mu.get_jax()
        self._nu = nu.get_jax()
        self._cost_fn = cost_fn
        self._C = None

        self._exact_cost = None
        self._exact_coupling = None

    def get_marginals(self) -> list[BaseMeasure]:
        return [self._mu, self._nu]

    def get_costs(self) -> list[ArrayLike]:
        """
        Returns a single‐element list containing the cost matrix between
        self._mu and self._nu, caching it in self._C so that repeated
        calls do not recompute.
        """
        if self._C is None:
            X, _ = self._mu.as_point_cloud()  # X: ArrayLike of shape (n, d)
            Y, _ = self._nu.as_point_cloud()  # Y: ArrayLike of shape (m, d)

            C = self._cost_fn(X, Y)  # should return an (n × m) array

            self._C = [C]

        return self._C

    def get_exact_cost(self) -> float:
        """
        Return exact cost of transportation between measures
        self._mu and self._nu, caching it in the self._exact_cost,
        such that repeated calls do not recompute
        """
        if self._exact_cost is None:
            self._compute_exact_solution()
        assert self._exact_cost is not None
        return self._exact_cost

    def get_exact_coupling(self) -> ArrayLike:
        """
        Return exact transport plan between self._mu and self._nu,
        caching it so repeated calls do not recompute.
        """
        if self._exact_coupling is None:
            self._compute_exact_solution()
        assert self._exact_coupling is not None
        return self._exact_coupling

    def to_dict(self) -> dict:
        mu_size = len(self._mu.as_point_cloud()[0])
        nu_size = len(self._nu.as_point_cloud()[0])
        return {
            "dataset": self.name,
            "type": "two_marginal",
            "n_mu": mu_size,
            "n_nu": nu_size,
            "cost": self.cost_name,
        }

    def _compute_exact_solution(self):
        """
        Compute exact solution of transportation between
        self._mu and self._nu and cache it in self._exact_cost and 
        self._exact_coupling
        """
        a = self._mu.as_point_cloud()[1]
        b = self._nu.as_point_cloud()[1]
        C = self.get_costs()[0]

        T, log = ot.emd(a, b, C, log=True, numItermax=10000000)
        if log['warning'] is not None:
            logger.warning(f"Computing ground truth for {self.to_dict()} didn't converge")

        self._exact_coupling = T
        self._exact_cost = log['cost']

    def to_ott_linear_problem(
        self,
        *,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        scale_cost: float | str = 1.0,
        batch_size: int | None = None,
        epsilon: float = 1e-2,
    ):
        """Return an OTT-JAX LinearProblem from this two-marginal problem.

        Requires ``pip install uot-bench[ott]``.
        """
        from uot.interop.ott._problems import two_marginal_to_linear_problem
        return two_marginal_to_linear_problem(
            self._mu, self._nu,
            cost_name=self.cost_name,
            scale_cost=scale_cost,
            batch_size=batch_size,
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
        )

    def to_ott_quadratic_problem(
        self,
        *,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        fused_penalty: float = 0.0,
        scale_cost: float | str = 1.0,
        epsilon: float = 1e-2,
        gw_unbalanced_correction: bool = True,
        ranks: int | tuple[int, ...] = -1,
        tolerances: float | tuple[float, ...] = 1e-2,
    ):
        """Return an OTT-JAX QuadraticProblem (GW) from this problem.

        Intra-space geometries are built from each measure's self-cost.
        Set ``fused_penalty > 0`` to include the cross-space (fused GW) term.
        Requires ``pip install uot-bench[ott]``.
        """
        from uot.interop.ott._problems import two_marginal_to_quadratic_problem
        return two_marginal_to_quadratic_problem(
            self._mu, self._nu,
            cost_name=self.cost_name,
            scale_cost=scale_cost,
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
            fused_penalty=fused_penalty,
            gw_unbalanced_correction=gw_unbalanced_correction,
            ranks=ranks,
            tolerances=tolerances,
        )

    def free_memory(self) -> None:
        self._C = None
