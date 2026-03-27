from copy import deepcopy

import numpy as np

from uot.data.measure import GridMeasure, PointCloudMeasure
from uot.experiments.experiment import Experiment
from uot.problems.iterator import OnlineProblemIterator
from uot.problems.problem_generator import ProblemGenerator
from uot.problems.two_marginal import TwoMarginalProblem
from uot.solvers.base_solver import BaseSolver
from uot.utils.costs import cost_euclid, cost_euclid_squared


class DummyGenerator(ProblemGenerator):
    def __init__(self, problem):
        self.problem = problem

    def generate(self):
        while True:
            yield self.problem


class CostAwareSolver(BaseSolver):
    def solve(self, marginals, costs, *args, **kwargs) -> dict:
        return {"cost": float(len(costs))}


class GridOnlySqEuclideanSolver(BaseSolver):
    requires_squared_euclidean = True

    def solve(self, marginals, costs, *args, **kwargs) -> dict:
        axes, mu = marginals[0].as_grid()
        _, nu = marginals[1].as_grid()
        return {"cost": float(mu.sum() + nu.sum() + len(axes))}


def _solve_fn(problem, solver, marginals, costs, **kwargs):
    return solver.solve(marginals=marginals, costs=costs, **kwargs)


def test_run_on_problems_respects_use_cost_matrix():
    mu = PointCloudMeasure(np.array([[0.0], [1.0]]), np.array([0.5, 0.5]))
    nu = PointCloudMeasure(np.array([[0.0], [1.0]]), np.array([0.5, 0.5]))
    problem = TwoMarginalProblem("pc", mu, nu, cost_euclid_squared)
    experiment = Experiment(name="dummy", solve_fn=_solve_fn)

    df = experiment.run_on_problems([problem], solver=CostAwareSolver, use_cost_matrix=False)

    assert df.iloc[0]["cost"] == 0.0


def test_run_on_problems_rejects_non_sqeuclidean_for_specialized_solver():
    axes = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
    mu = GridMeasure(axes, np.array([[0.25, 0.25], [0.25, 0.25]]), normalize=False)
    nu = GridMeasure(axes, np.array([[0.25, 0.25], [0.25, 0.25]]), normalize=False)
    problem = TwoMarginalProblem("grid", mu, nu, cost_euclid)
    experiment = Experiment(name="dummy", solve_fn=_solve_fn)

    df = experiment.run_on_problems([problem], solver=GridOnlySqEuclideanSolver, use_cost_matrix=False)

    assert df.iloc[0]["status"] == "failed"
    assert "requires squared Euclidean cost" in df.iloc[0]["exception"]


def test_online_problem_iterator_roundtrip_rebuilds_generator_iterator():
    mu = PointCloudMeasure(np.array([[0.0], [1.0]]), np.array([0.5, 0.5]))
    nu = PointCloudMeasure(np.array([[0.0], [1.0]]), np.array([0.5, 0.5]))
    problem = TwoMarginalProblem("pc", mu, nu, cost_euclid_squared)
    iterator = OnlineProblemIterator(DummyGenerator(problem), num=3)

    first = next(iterator)
    assert first.name == "pc"

    restored = deepcopy(iterator)
    second = next(restored)
    assert second.name == "pc"
