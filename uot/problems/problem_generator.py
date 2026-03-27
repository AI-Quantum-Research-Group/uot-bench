from abc import ABC, abstractmethod
from uot.problems.base_problem import GridInputs, MarginalProblem, PointCloudInputs, SolverInputs
from collections.abc import Iterator


class ProblemGenerator(ABC):

    @abstractmethod
    def generate(self, *args, **kwargs) -> Iterator[MarginalProblem]:
        "Return a list of MarginalProblem objects."
        raise NotImplementedError

    def one(self, **generate_kwargs) -> MarginalProblem:
        return next(self.generate(**generate_kwargs))

    def solver_inputs(self, *, include_cost: bool = True, **generate_kwargs) -> SolverInputs:
        return self.one(**generate_kwargs).solver_inputs(include_cost=include_cost)

    def point_cloud_inputs(
        self,
        *,
        shared_support: str = "same",
        include_cost: bool = True,
        include_zeros: bool = True,
        atol: float = 0.0,
        rtol: float = 0.0,
        **generate_kwargs,
    ) -> PointCloudInputs:
        return self.one(**generate_kwargs).point_cloud_inputs(
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
        **generate_kwargs,
    ) -> GridInputs:
        return self.one(**generate_kwargs).grid_inputs(
            include_cost=include_cost,
            backend=backend,
            dtype=dtype,
            device=device,
        )
