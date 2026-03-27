from .base_problem import GridInputs, PointCloudInputs, SolverInputs
from .barycenter_problem import BarycenterProblem
from .two_marginal import TwoMarginalProblem

__all__ = [
    "SolverInputs",
    "PointCloudInputs",
    "GridInputs",
    "TwoMarginalProblem",
    "BarycenterProblem",
]
