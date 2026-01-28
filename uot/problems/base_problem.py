from abc import ABC
import hashlib
import pickle
from collections.abc import Callable

import numpy as np
import jax
import jax.numpy as jnp

from uot.data.measure import BaseMeasure, _snap_points, _row_view
from uot.utils.types import ArrayLike


class MarginalProblem(ABC):
    def __init__(
        self, name: str, measures: list[BaseMeasure], cost_fns: list[Callable]
    ):
        super().__init__()
        if len(measures) < 2:
            raise ValueError("Need at least two marginals")
        self.name = name
        self.measures = measures
        self.cost_fns = cost_fns
        # guys, I followed your approach to cache the cost function
        # BUT I think that there might be better ones: just store all cost function or use other caching procedures. so I ask you:
        # TODO: for now just compute WHOLE cost matrix and store it as it is.
        self._cost_cache = [None] * len(cost_fns)
        self.__hash = None

    def __repr__(self):
        space_size = 'x'.join(
            str(marginal.to_discrete()[0].size)
            for marginal in self.get_marginals()
        )
        return f"<{self.__class__.__name__}[{self.name}] {space_size}\
        with ({map(lambda fn: fn.__name__, self.cost_fns)})>"

    def key(self) -> str:
        if self.__hash is None:
            blob = pickle.dumps(self, protocol=4)
            self.__hash = hashlib.sha1(blob).hexdigest()
        return self.__hash

    def __hash__(self) -> int:
        """
        Return an integer hash, derived from the SHA1 key.
        """
        # Convert the first 16 hex digits of the key into a Python int
        hex_key = self.key()[:16]
        return int(hex_key, 16)

    def get_marginals(self) -> list[BaseMeasure]:
        raise NotImplementedError()

    def get_costs(self) -> list[ArrayLike]:
        raise NotImplementedError()

    def shared_support(
        self,
        *,
        mode: str = "union",
        include_zeros: bool = True,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> ArrayLike:
        supports = [m.support(include_zeros=include_zeros) for m in self.get_marginals()]
        if not supports:
            return np.zeros((0, 0))

        want_jax = any(isinstance(s, jax.Array) for s in supports)

        if mode == "first":
            support = supports[0]
            return jnp.asarray(support) if want_jax else np.asarray(support)

        supports_np = [_snap_points(s, atol=atol, rtol=rtol) for s in supports]

        if mode == "union":
            stacked = np.concatenate(supports_np, axis=0)
            view = _row_view(stacked)
            _, idx = np.unique(view, return_index=True)
            support = stacked[np.sort(idx)]
        elif mode == "intersection":
            support = supports_np[0]
            for other in supports_np[1:]:
                mask = np.in1d(_row_view(support), _row_view(other))
                support = support[mask]
        else:
            raise ValueError("mode must be 'union', 'intersection', or 'first'")

        return jnp.asarray(support) if want_jax else support

    def weights_on_shared_support(
        self,
        *,
        mode: str = "union",
        include_zeros: bool = True,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> tuple[ArrayLike, ArrayLike]:
        support = self.shared_support(
            mode=mode,
            include_zeros=include_zeros,
            atol=atol,
            rtol=rtol,
        )
        weights = [
            m.weights_on(support, include_zeros=include_zeros, atol=atol, rtol=rtol)
            for m in self.get_marginals()
        ]
        want_jax = isinstance(support, jax.Array) or any(isinstance(w, jax.Array) for w in weights)
        xp = jnp if want_jax else np
        support = xp.asarray(support)
        weights = xp.stack([xp.asarray(w) for w in weights], axis=0)
        return support, weights

    def to_dict(self) -> dict:
        raise NotImplementedError()

    def free_memory(self):
        # TODO: do we actually need this one?
        raise NotImplementedError()
