from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

import jax.numpy as jnp
import pandas as pd

from uot.data.measure import BaseMeasure
from uot.problems.base_problem import Problem
from uot.solvers.base_solver import BaseSolver, SolverOutput
from uot.utils.logging import logger
from uot.utils.instantiate_solver import instantiate_solver
from uot.utils.types import ArrayLike
from uot.solvers.gradient_ascent._smith_best_lr import best_lr as _best_lr


class SolveFn(Protocol):
    """Signature expected by :class:`Experiment` for its ``solve_fn`` argument."""

    def __call__(
        self,
        prob: Problem,
        instance: BaseSolver,
        marginals: list[BaseMeasure],
        costs: list[ArrayLike],
        **kwargs: Any,
    ) -> dict[str, Any]: ...


class Experiment:
    def __init__(
        self,
        name: str,
        solve_fn: SolveFn,
    ):
        """
        Parameters
        ----------
        name:
            Human-readable experiment name (stored in result DataFrames).
        solve_fn:
            A callable matching :class:`SolveFn` — typically one of the
            ``measure_*`` functions from :mod:`uot.experiments.measurement`.
        """
        self.name = name
        self.solve_fn = solve_fn

    def _run_lr_finder(
        self,
        solver: BaseSolver,
        marginals: list[BaseMeasure],
        costs: list[ArrayLike],
        **solver_kwargs: Any,
    ) -> float:
        if not hasattr(solver, "find_lr"):
            raise RuntimeError(f"Solver {solver.__class__.__name__} has no `find_lr` method")
        lrs, losses = solver.find_lr(marginals=marginals, costs=costs, **solver_kwargs)
        return float(lrs[jnp.argmax(jnp.array(losses))])

    def run_on_problems(
        self,
        problems: Iterable[Problem],
        solver: BaseSolver,
        progress_callback: Callable[[int], None] | None = None,
        use_cost_matrix: bool = True,
        **solver_kwargs: Any,
    ) -> pd.DataFrame:
        results = []
        for i, problem in enumerate(problems):
            solver_inputs = problem.solver_inputs(include_cost=use_cost_matrix)
            marginals = solver_inputs.marginals
            costs: list[ArrayLike] = solver_inputs.costs
            try:
                solver_init_kwargs = solver_kwargs or {}
                if getattr(solver, "requires_squared_euclidean", False) and not solver_inputs.is_squared_euclidean:
                    raise ValueError(
                        f"{solver.__name__} requires squared Euclidean cost, got {solver_inputs.cost_name}"
                    )
                if "learning_rate" in solver_init_kwargs and solver_init_kwargs["learning_rate"] == "auto":
                    logger.info(f"Finding learning rate for {solver.__name__} on {problem}")
                    lr = self._run_lr_finder(
                        solver=solver,
                        marginals=marginals,
                        costs=costs,
                        **solver_init_kwargs,
                    )
                    solver_init_kwargs["learning_rate"] = lr
                    logger.info(f"Found learning rate: {lr:.3e}")

                instance = instantiate_solver(solver_cls=solver, init_kwargs=solver_init_kwargs)

                logger.info(f"Starting {solver.__name__} with {solver_kwargs} on {problem}")
                metrics = self.solve_fn(
                    problem,
                    instance,
                    marginals,
                    costs,
                    **solver_kwargs,
                )
                metrics["status"] = "success"
                logger.info(f"Successfully finished {solver.__name__} with {solver_kwargs}")
            except Exception as e:
                logger.error(f"{solver.__qualname__} failed with error {e}")
                metrics = {
                    "status": "failed",
                    "exception": str(e),
                }
            metrics["problem_index"] = i
            for solver_key, solver_value in solver_init_kwargs.items():
                metrics[solver_key] = solver_value
            df_row = problem.to_dict()
            df_row.update(metrics)
            results.append(df_row)
            if progress_callback:
                progress_callback(1)
        return pd.DataFrame(results)

    def run_single(
        self,
        problem: Problem,
        solver: BaseSolver,
        **solver_kwargs: Any,
    ) -> dict[str, Any]:
        solver_inputs = problem.solver_inputs(include_cost=True)
        marginals = solver_inputs.marginals
        costs = solver_inputs.costs
        return self.solve_fn(
            problem,
            solver,
            marginals,
            costs,
            **solver_kwargs,
        )
