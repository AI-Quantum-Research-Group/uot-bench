"""Post-solve hook protocol for the benchmarking pipeline.

Hooks run after each :meth:`~uot.experiments.Experiment.solve_fn` call and can
add domain-specific metrics to the result row or fan it out into multiple rows
(e.g. colour-transfer soft-extension × displacement-alpha grid).

A hook callable must match :class:`PostSolveHook`.  Register problem-level hooks
via :meth:`~uot.problems.base_problem.Problem.post_solve_hooks`; register
experiment-level hooks via ``Experiment(…, hooks=[…])``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from uot.problems.base_problem import Problem


@runtime_checkable
class PostSolveHook(Protocol):
    """Protocol for a post-solve hook.

    Return values
    -------------
    ``None``
        No change; keep the single base-metrics row as-is.
    ``dict``
        Merged into the base-metrics row (keys override existing ones).
    ``list[dict]``
        **Replaces** the base-metrics row with one row per list element.
        Each dict should be a self-contained metrics row.  Use this for
        fan-out (e.g. one row per post-processing mode).
    """

    def __call__(
        self,
        problem: Problem,
        view: Any,
        metrics: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any] | list[dict[str, Any]] | None: ...


def apply_hooks(
    problem: Problem,
    view: Any,
    base_metrics: dict[str, Any],
    hooks: list[PostSolveHook],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply *hooks* in order, starting from *base_metrics*.

    A hook returning a ``list`` replaces the current row set (fan-out).
    A hook returning a ``dict`` merges into every current row.
    A hook returning ``None`` is a no-op.
    """
    rows: list[dict[str, Any]] = [base_metrics]
    for hook in hooks:
        new_rows: list[dict[str, Any]] = []
        for row in rows:
            result = hook(problem, view, row, context)
            if result is None:
                new_rows.append(row)
            elif isinstance(result, list):
                new_rows.extend(result)
            else:
                new_rows.append({**row, **result})
        rows = new_rows
    return rows
