from uot.problems.base_problem import Problem


class MultiMarginalProblem(Problem):
    """Base class for multi-marginal OT problems.

    Subclass this (or :class:`~uot.problems.Problem` directly) and implement
    :meth:`~uot.problems.Problem.get_marginals`,
    :meth:`~uot.problems.Problem.get_costs`,
    :meth:`~uot.problems.Problem.to_dict`, and
    :meth:`~uot.problems.Problem.free_memory`.
    """

    def __init__(self, name, measures, cost_fns):
        super().__init__(name, measures, cost_fns)
