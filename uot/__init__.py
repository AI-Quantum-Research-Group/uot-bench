from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("uot_bench")
except PackageNotFoundError:
    __version__ = "unknown"

# Core abstractions — subclass these to define your own problems and generators
from uot.problems.base_problem import Problem, MarginalProblem
from uot.problems.problem_generator import Generator, ProblemGenerator

# Concrete problem types
from uot.problems.two_marginal import TwoMarginalProblem
from uot.problems.barycenter_problem import BarycenterProblem
from uot.problems.multi_marginal import MultiMarginalProblem

# Input/output dataclasses
from uot.problems.base_problem import SolverInputs, PointCloudInputs, GridInputs

# Solver abstractions
from uot.solvers.base_solver import BaseSolver
from uot.solvers.solver_config import SolverConfig

# Experiment runner
from uot.experiments.experiment import Experiment
from uot.experiments.runner import run_pipeline

# Measures
from uot.data.measure import BaseMeasure, PointCloudMeasure, GridMeasure

__all__ = [
    "__version__",
    # Core abstractions
    "Problem",
    "Generator",
    # Concrete problem types
    "TwoMarginalProblem",
    "BarycenterProblem",
    "MultiMarginalProblem",
    # Input/output
    "SolverInputs",
    "PointCloudInputs",
    "GridInputs",
    # Solvers
    "BaseSolver",
    "SolverConfig",
    # Experiments
    "Experiment",
    "run_pipeline",
    # Measures
    "BaseMeasure",
    "PointCloudMeasure",
    "GridMeasure",
    # Deprecated aliases
    "MarginalProblem",
    "ProblemGenerator",
]
