import pytest

from uot.solvers.base_solver import BaseSolver
from uot.utils.instantiate_solver import instantiate_solver


class _VarKwargsSolver(BaseSolver):
    def solve(self, marginals, costs, **kwargs):
        return {"cost": 0.0}


class _FixedArgSolver(BaseSolver):
    def __init__(self, reg: float = 0.1) -> None:
        self.reg = reg

    def solve(self, marginals, costs, **kwargs):
        return {"cost": 0.0}


class _RequiredArgSolver(BaseSolver):
    def __init__(self, backend: str) -> None:
        self.backend = backend

    def solve(self, marginals, costs, **kwargs):
        return {"cost": 0.0}


def test_instantiate_var_kwargs_solver_succeeds():
    solver = instantiate_solver(_VarKwargsSolver, {"reg": 0.01})
    assert isinstance(solver, _VarKwargsSolver)


def test_instantiate_fixed_arg_solver_uses_matching_kwarg():
    solver = instantiate_solver(_FixedArgSolver, {"reg": 0.5, "extra": "ignored"})
    assert solver.reg == 0.5


def test_instantiate_fixed_arg_solver_uses_default_when_missing():
    solver = instantiate_solver(_FixedArgSolver, {})
    assert solver.reg == 0.1


def test_instantiate_required_arg_raises_when_missing():
    with pytest.raises(TypeError, match="missing required init args"):
        instantiate_solver(_RequiredArgSolver, {"unrelated": 1})


def test_instantiate_required_arg_succeeds_when_provided():
    solver = instantiate_solver(_RequiredArgSolver, {"backend": "jax"})
    assert solver.backend == "jax"
