from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

import jax.numpy as jnp
import pandas as pd

from uot.data.measure import BaseMeasure
from uot.experiments.hooks import PostSolveHook, apply_hooks
from uot.experiments.representations import build_representation
from uot.problems.base_problem import Problem
from uot.solvers.base_solver import BaseSolver
from uot.utils.logging import logger
from uot.utils.instantiate_solver import instantiate_solver
from uot.utils.types import ArrayLike


class SolveFn(Protocol):
    """Signature expected by :class:`Experiment` for its ``solve_fn`` argument.

    The *view* argument is the representation built by the runner outside the
    timed region.  For the default ``"marginals_costs"`` kind this is a
    :class:`~uot.problems.base_problem.SolverInputs` dataclass; other kinds
    pass a backend-specific pre-built problem object.
    """

    def __call__(
        self,
        prob: Problem,
        instance: BaseSolver,
        view: Any,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


class Experiment:
    def __init__(
        self,
        name: str,
        solve_fn: SolveFn,
        hooks: list[PostSolveHook] | None = None,
    ):
        """
        Parameters
        ----------
        name:
            Human-readable experiment name (stored in result DataFrames).
        solve_fn:
            A callable matching :class:`SolveFn` — typically one of the
            ``measure_*`` functions from :mod:`uot.experiments.measurement`.
        hooks:
            Optional list of :class:`~uot.experiments.hooks.PostSolveHook`
            callables to run after every ``solve_fn`` call.  Problem-level
            hooks (from :meth:`~uot.problems.base_problem.Problem.post_solve_hooks`)
            are prepended automatically.
        """
        self.name = name
        self.solve_fn = solve_fn
        self.hooks: list[PostSolveHook] = list(hooks or [])

    def _run_lr_finder(
        self,
        solver: BaseSolver,
        marginals: list[BaseMeasure],
        costs: list[ArrayLike],
        **solver_kwargs: Any,
    ) -> float:
        if not hasattr(solver, "find_lr"):
            raise RuntimeError(f"Solver {solver.__class__.__name__} has no `find_lr` method")
        _solver_any: Any = solver
        lrs, losses = _solver_any.find_lr(marginals=marginals, costs=costs, **solver_kwargs)
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
            solver_init_kwargs: dict[str, Any] = {}
            row_metrics_list: list[dict[str, Any]] = []

            try:
                solver_init_kwargs = solver_kwargs or {}

                # Build representation OUTSIDE the timed solve region.
                kind = getattr(solver, "input_kind", "marginals_costs")
                view = build_representation(
                    problem, kind,
                    include_cost=use_cost_matrix,
                    **solver_init_kwargs,
                )

                # Squared-euclidean check (only relevant for views that carry cost metadata).
                if getattr(solver, "requires_squared_euclidean", False) and hasattr(view, "is_squared_euclidean"):
                    if not view.is_squared_euclidean:
                        raise ValueError(
                            f"{solver.__name__} requires squared Euclidean cost, "
                            f"got {view.cost_name}"
                        )

                # Auto learning-rate finder (native solvers only).
                if "learning_rate" in solver_init_kwargs and solver_init_kwargs["learning_rate"] == "auto":
                    logger.info(f"Finding learning rate for {solver.__name__} on {problem}")
                    _si = view if hasattr(view, "marginals") else problem.solver_inputs(include_cost=use_cost_matrix)
                    lr = self._run_lr_finder(
                        solver=solver,
                        marginals=_si.marginals,
                        costs=_si.costs,
                        **solver_init_kwargs,
                    )
                    solver_init_kwargs["learning_rate"] = lr
                    logger.info(f"Found learning rate: {lr:.3e}")
                    # Rebuild view with updated kwargs.
                    view = build_representation(
                        problem, kind,
                        include_cost=use_cost_matrix,
                        **solver_init_kwargs,
                    )

                instance = instantiate_solver(solver_cls=solver, init_kwargs=solver_init_kwargs)  # type: ignore[arg-type]

                logger.info(f"Starting {solver.__name__} with {solver_kwargs} on {problem}")
                base_metrics = self.solve_fn(
                    problem,
                    instance,
                    view,
                    **solver_kwargs,
                )
                base_metrics["status"] = "success"
                logger.info(f"Successfully finished {solver.__name__} with {solver_kwargs}")

                all_hooks = list(problem.post_solve_hooks()) + self.hooks
                if all_hooks:
                    context: dict[str, Any] = {
                        "problem_index": i,
                        "solver_name": getattr(solver, "__name__", str(solver)),
                        "solver_kwargs": dict(solver_kwargs),
                    }
                    row_metrics_list = apply_hooks(problem, view, base_metrics, all_hooks, context)
                else:
                    row_metrics_list = [base_metrics]

            except Exception as e:
                logger.error(f"{solver.__qualname__} failed with error {e}")
                row_metrics_list = [{"status": "failed", "exception": str(e)}]

            for row_metrics in row_metrics_list:
                row_metrics["problem_index"] = i
                for solver_key, solver_value in solver_init_kwargs.items():
                    row_metrics[solver_key] = solver_value
                df_row = problem.to_dict()
                df_row.update(row_metrics)
                results.append(df_row)

            problem.free_memory()

            if progress_callback:
                progress_callback(1)

        return pd.DataFrame(results)

    def run_single(
        self,
        problem: Problem,
        solver: BaseSolver,
        **solver_kwargs: Any,
    ) -> dict[str, Any]:
        kind = getattr(solver, "input_kind", "marginals_costs")
        view = build_representation(problem, kind, include_cost=True, **solver_kwargs)
        return self.solve_fn(
            problem,
            solver,
            view,
            **solver_kwargs,
        )
