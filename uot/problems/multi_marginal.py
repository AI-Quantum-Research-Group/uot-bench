from collections.abc import Callable

from uot.data.measure import BaseMeasure
from uot.problems.base_problem import Problem
from uot.utils.types import ArrayLike


class MultiMarginalProblem(Problem):
    """Base class for multi-marginal OT problems.

    Subclass this (or :class:`~uot.problems.Problem` directly) and implement
    :meth:`~uot.problems.Problem.get_marginals`,
    :meth:`~uot.problems.Problem.get_costs`,
    :meth:`~uot.problems.Problem.to_dict`, and
    :meth:`~uot.problems.Problem.free_memory`.
    """

    def __init__(
        self,
        name: str,
        measures: list[BaseMeasure],
        cost_fns: list[Callable[[ArrayLike, ArrayLike], ArrayLike]],
    ) -> None:
        super().__init__(name, measures, cost_fns)
