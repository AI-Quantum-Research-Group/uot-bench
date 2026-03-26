from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, Sequence, Callable

import numpy as np
import jax
import jax.numpy as jnp

from uot.problems.barycenter_problem import BarycenterProblem
from uot.problems.problem_generator import ProblemGenerator
from uot.utils.generator_helpers.get_axes import get_axes, CellDiscretization
from uot.utils.generator_helpers import shapes
from uot.utils.generate_nd_grid import generate_nd_grid
from uot.utils.types import (
    MeasureMode,
    ArrayLike,
)


class ShapeSelector(Protocol):
    def select(
        self,
        *,
        available: Sequence[str],
        k: int,
        key: jax.Array,
    ) -> list[str]:
        ...


class ToyBarycenterGenerator(ProblemGenerator):
    def __init__(self, selector: ShapeSelector,
                 n_points: int,
                 cost_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
                 num_datasets: int = 1,
                 cell_discretization: CellDiscretization = 'cell-centered',
                 measure_mode: MeasureMode = "grid",
                 use_jax: bool = False,
                 seed: int = 42,
                 ) -> None:
        super().__init__()
        self.selector = selector
        self.dim = 2
        self.borders = [0., 1.]
        self.n_points = n_points
        self.num_datasets = num_datasets
        self.cost_fn = cost_fn
        self.cell_discretization = cell_discretization
        self.use_jax = use_jax
        self.measure_mode = measure_mode
        if self.use_jax:
            self._key = jax.random.PRNGKey(seed)
        else:
            self._rng = np.random.default_rng(seed)

    def generate(self, *args,
                 num_marginals: int, **kwargs) -> Iterator[BarycenterProblem]:
        axes = get_axes(dim=self.dim,
                        borders=self.borders,
                        n_points=self.n_points,
                        cell_discretization=self.cell_discretization,
                        use_jax=self.use_jax)
        X, Y = shapes.get_xy_grid(axes)
        points = generate_nd_grid(axes, self.use_jax)
        toy_source_factories = shapes.get_toy_source_factories(
            X,
            Y,
            n_points=self.n_points,
            use_jax=self.use_jax,
        )
        available = tuple(toy_source_factories.keys())
        selector_key = self._key if self.use_jax else self._rng

        for _ in range(self.num_datasets):
            chosen_names = self.selector.select(
                available=available,
                k=num_marginals,
                key=selector_key,
            )
            chosen_fields = [toy_source_factories[name]() for name in chosen_names]

            measures = shapes.build_measures_from_fields(
                chosen_fields,
                points=points,
                axes=axes,
                measure_mode=self.measure_mode,
                use_jax=self.use_jax,
            )
            lambdas = jnp.ones((len(measures),))
            lambdas /= lambdas.sum()
            yield BarycenterProblem(
                name='ToyExamples',
                measures=measures,
                lambdas=lambdas,
                cost_fn=self.cost_fn,
            )


@dataclass(frozen=True)
class FixedListSelector(ShapeSelector):
    names: tuple[str, ...]
    strict_k: bool = True   # if we pass from generator k -> enforce match

    def select(self, available: Sequence[str],
               k: int | None, key=None) -> list[str]:
        missing = [n for n in self.names if n not in available]
        if missing:
            raise ValueError(
                f"Unknown shape(s) requested {missing}. Available {list(available)}")
        if k is not None and self.strict_k and k != len(self.names):
            raise ValueError(f"FixedListSelector has {len(self.names)} but {k=}")

        return list(self.names if k is None else self.names[:k])


@dataclass
class RoundRobinSelector:
    """Cycle across groups; within each group cycle through names."""
    groups: tuple[tuple[str, ...], ...]
    start_group: int = 0

    _group_cursor: int = field(init=False, default=0)
    _item_cursors: list[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._group_cursor = self.start_group % max(1, len(self.groups))
        self._item_cursors = [0 for _ in self.groups]

    def select(self, *,
               available: Sequence[str], k: int, key=None) -> list[str]:
        # Filter groups to only shapes that exist in `available`
        avail = set(available)
        filtered = [tuple(n for n in grp if n in avail) for grp in self.groups]
        filtered = [grp for grp in filtered if len(grp) > 0]

        if not filtered:
            raise ValueError(
                "RoundRobinSelector: no shapes available after filtering.")

        # If filtering changed number of groups, reset cursors safely
        if len(filtered) != len(self._item_cursors):
            self._item_cursors = [0 for _ in filtered]
            self._group_cursor %= len(filtered)

        out: list[str] = []
        G = len(filtered)

        for _ in range(k):
            g = self._group_cursor
            grp = filtered[g]
            i = self._item_cursors[g] % len(grp)

            out.append(grp[i])

            self._item_cursors[g] += 1
            self._group_cursor = (self._group_cursor + 1) % G

        return out
