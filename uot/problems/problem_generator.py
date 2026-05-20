from abc import ABC, abstractmethod
from uot.problems.base_problem import GridInputs, Problem, MarginalProblem, PointCloudInputs, SolverInputs
from collections.abc import Iterator


class Generator(ABC):
    """Base class for problem generators.

    Subclass this to create a reusable source of :class:`~uot.problems.Problem`
    instances, then pass it to :func:`~uot.experiments.run_pipeline`.

    **Minimal subclass example**::

        from collections.abc import Iterator
        from uot import Generator, TwoMarginalProblem
        from uot.data import PointCloudMeasure
        import numpy as np

        class MyGenerator(Generator):
            def __init__(self, n: int, seed: int = 0):
                self.n = n
                self.rng = np.random.default_rng(seed)

            def generate(self) -> Iterator[TwoMarginalProblem]:
                from uot.utils.costs import cost_euclid_squared
                for _ in range(10):
                    X = self.rng.standard_normal((self.n, 2))
                    Y = self.rng.standard_normal((self.n, 2))
                    mu = PointCloudMeasure(X, np.ones(self.n) / self.n)
                    nu = PointCloudMeasure(Y, np.ones(self.n) / self.n)
                    yield TwoMarginalProblem("my_problem", mu, nu, cost_euclid_squared)

    The generator stores its hyper-parameters in ``__init__``; :meth:`generate`
    takes no arguments and yields problems one at a time.
    """

    @abstractmethod
    def generate(self) -> Iterator[Problem]:
        """Yield :class:`~uot.problems.Problem` instances."""
        raise NotImplementedError

    def one(self) -> Problem:
        """Return the first problem from this generator."""
        return next(self.generate())

    def solver_inputs(self, *, include_cost: bool = True) -> SolverInputs:
        """Return :class:`~uot.problems.SolverInputs` for the first generated problem."""
        return self.one().solver_inputs(include_cost=include_cost)

    def point_cloud_inputs(
        self,
        *,
        shared_support: str = "same",
        include_cost: bool = True,
        include_zeros: bool = True,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> PointCloudInputs:
        """Return :class:`~uot.problems.PointCloudInputs` for the first generated problem."""
        return self.one().point_cloud_inputs(
            shared_support=shared_support,
            include_cost=include_cost,
            include_zeros=include_zeros,
            atol=atol,
            rtol=rtol,
        )

    def grid_inputs(
        self,
        *,
        include_cost: bool = False,
        backend: str = "auto",
        dtype=None,
        device=None,
    ) -> GridInputs:
        """Return :class:`~uot.problems.GridInputs` for the first generated problem."""
        return self.one().grid_inputs(
            include_cost=include_cost,
            backend=backend,
            dtype=dtype,
            device=device,
        )


import warnings as _warnings


def __getattr__(name: str):
    if name == "ProblemGenerator":
        _warnings.warn(
            "ProblemGenerator is deprecated and will be removed in a future release. "
            "Use Generator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Generator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Keep a direct reference for `from uot.problems.problem_generator import ProblemGenerator`.
ProblemGenerator = Generator
