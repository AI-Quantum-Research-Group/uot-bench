from collections.abc import Callable
from typing import Any, cast

import jax
from uot.data.measure import BaseMeasure
from uot.utils.types import ArrayLike, ShareMode
from uot.problems.base_problem import Problem


class BarycenterProblem(Problem):
    analytic_barycenter: Any
    analytic_barycenter_error: str

    def __init__(self, name: str,
                 measures: list[BaseMeasure],
                 cost_fn: Callable[..., Any],
                 lambdas: ArrayLike):
        super().__init__(name, measures, [cost_fn])
        if not len(self.measures):
            raise ValueError('measures list should not be empty')
        self._lambdas = lambdas
        self._C = None
        self._cost_fn = cost_fn

    def lambdas(self) -> ArrayLike:
        return self._lambdas

    def get_lambdas(self) -> ArrayLike:
        return self._lambdas

    def get_marginals(self) -> list[BaseMeasure]:
        return self.measures

    def get_costs(self) -> list[ArrayLike]:
        if self._C is None:
            mu, *_ = self.measures
            X, _ = mu.as_point_cloud()
            C = self._cost_fn(X, X)
            self._C = [C]
        return self._C

    def shared_support_inputs(
        self,
        *,
        mode: str = "same",
        include_zeros: bool = True,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        share_mode = cast(ShareMode, mode)
        inputs = self.point_cloud_inputs(
            shared_support=share_mode,
            include_cost=True,
            include_zeros=include_zeros,
            atol=atol,
            rtol=rtol,
        )
        assert inputs.cost is not None, "Cost is None after point_cloud_inputs with include_cost=True"
        return inputs.support, inputs.weights, inputs.cost, self._lambdas

    def to_dict(self) -> dict:
        marginals_size = self.measures[0].as_point_cloud()[1].size
        return {
            "dataset": self.name,
            "type": "barycenter",
            "marginals_size": marginals_size,
            "cost": self.cost_name,
        }

    def to_ott_barycenter_problem(
        self,
        *,
        epsilon: float = 1e-2,
        cost_name: str | None = None,
        scale_cost: float | str = 1.0,
        batch_size: int | None = None,
    ):
        """Return an OTT-JAX FreeBarycenterProblem from this problem.

        Requires ``pip install uot-bench[ott]``.
        """
        from ott.problems.linear.barycenter_problem import FreeBarycenterProblem
        from uot.interop.ott._costs import cost_fn_for_name

        import jax.numpy as jnp
        name = cost_name or self.cost_name

        ys = []
        bs = []
        for m in self.measures:
            pts, wts = m.as_point_cloud()
            ys.append(jnp.asarray(pts))
            bs.append(jnp.asarray(wts))

        return FreeBarycenterProblem(
            y=jnp.stack(ys),
            b=jnp.stack(bs),
            epsilon=epsilon,
            cost_fn=cost_fn_for_name(name),
        )

    def free_memory(self) -> None:
        self._C = None
        self._cost_cache = [None] * len(self.cost_fns)
