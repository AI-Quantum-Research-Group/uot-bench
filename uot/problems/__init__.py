from .base_problem import (
    GridInputs,
    MarginalProblem,
    PointCloudInputs,
    Problem,
    SolverInputs,
)
from .barycenter_problem import BarycenterProblem
from .multi_marginal import MultiMarginalProblem
from .problem_generator import Generator, ProblemGenerator
from .two_marginal import TwoMarginalProblem

__all__ = [
    # Primary names
    "Problem",
    "Generator",
    # Input/output dataclasses
    "SolverInputs",
    "PointCloudInputs",
    "GridInputs",
    # Concrete problem types
    "TwoMarginalProblem",
    "BarycenterProblem",
    "MultiMarginalProblem",
    # Deprecated aliases (kept for backward compat)
    "MarginalProblem",
    "ProblemGenerator",
]
