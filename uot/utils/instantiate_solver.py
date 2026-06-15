from __future__ import annotations

import inspect
from typing import Any, TypeVar

from uot.solvers.base_solver import BaseSolver

S = TypeVar("S", bound=BaseSolver)


def instantiate_solver(solver_cls: type[S], init_kwargs: dict[str, Any]) -> S:
    """Instantiate *solver_cls* by matching *init_kwargs* to its ``__init__`` signature.

    Only parameters present in the signature are forwarded; extra keys are ignored.
    Raises :class:`TypeError` if any required parameter is missing.
    """
    sig = inspect.signature(solver_cls.__init__)
    params = {name: p for name, p in sig.parameters.items() if name != "self"}

    passed = {k: v for k, v in init_kwargs.items() if k in params}
    missing = [
        name
        for name, p in params.items()
        if p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and name not in passed
    ]
    if missing:
        raise TypeError(f"{solver_cls.__name__} missing required init args: {missing}")

    return solver_cls(**passed)
