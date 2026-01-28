from abc import ABC
from collections.abc import Callable
from typing import Any

from jax import Array
from numpy import dtype, ndarray
from uot.data.measure import BaseMeasure
from uot.utils.types import ArrayLike
from uot.problems.base_problem import MarginalProblem


class BarycenterProblem(MarginalProblem):
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

    def get_marginals(self) -> list[BaseMeasure]:
        return self.measures

    def get_costs(self) -> list[ndarray[tuple[Any, ...], dtype[Any]] | Array]:
        if self._C is None:
            mu, *_ = self.measures
            X, _ = mu.to_discrete()
            C = self._cost_fn(X, X)  # should return an (n × m) array
            self._C = [C]
        return self._C

    def shared_support_inputs(
        self,
        *,
        mode: str = "union",
        include_zeros: bool = True,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        support, weights = self.weights_on_shared_support(
            mode=mode,
            include_zeros=include_zeros,
            atol=atol,
            rtol=rtol,
        )
        cost = self._cost_fn(support, support)
        return support, weights, cost, self._lambdas

    def to_dict(self) -> dict:
        marginals_size = self.measures[0].to_discrete()[1].size
        return {
            "dataset": self.name,
            "marginals_size": marginals_size,
            "cost": self._cost_fn.__name__,
        }
